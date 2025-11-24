#!/usr/bin/env python3
import rclpy
import torch
import math
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistWithCovarianceStamped
from sensor_msgs.msg import Imu
from devbot_interfaces.msg import WheelVels
from dkf_localization.dkf_model_structure import DeepKalmanFilter
from dkf_localization.dkf_model_utils import minmax_norm, minmax_unnorm, train_dkf_model
from dkf_localization.dkf_model_utils import runge_kutta_4, load_samples_from_csvs, DKFLoss
from collections import deque
import matplotlib.pyplot as plt


class DKFOdometryEstimation(Node):
    def __init__(self):
        super().__init__('dkf_odometry_estimation_node')
        self.internal_counter = 0
        self.training_steps = 0
        self.extra_counter = 0
        self.twist_msg = None
        self.left_wheel_velocity = None
        self.right_wheel_velocity = None
        self.imu_msg = None
        self.ground_truth_msg = None
        self.prev_odom_low = {}
        self.prev_odom_high = {}
        self.rmse_array = []
        self.nll_array = []

        # Load the model
        self.declare_parameter('model_path', '')
        self.declare_parameter('learning_rate', 1e-3)
        self.declare_parameter('weight_decay', 5e-7)
        self.declare_parameter('dkf_odom_rate', 50)  # Frequency of the dkf odometry node
        self.declare_parameter('load_samples_from_csvs', False)
        self.declare_parameter('data_dir', '/home/devin1126/dev_ws/src/dynamic_data_collector/combined_data')

        model_path = self.get_parameter('model_path').value
        learning_rate = self.get_parameter('learning_rate').value
        dkf_odom_rate = self.get_parameter('dkf_odom_rate').value
        weight_decay = self.get_parameter('weight_decay').value
        load_samples = self.get_parameter('load_samples_from_csvs').value
        data_dir = self.get_parameter('data_dir').value

        # Subscribe to low/high fidelity odometry sources
        self.twist_sub = self.create_subscription(
            TwistWithCovarianceStamped,
            'wheel_twist',
            self.twist_listener_callback,
            10
        )

        self.wheel_sub = self.create_subscription(
            WheelVels,
            'wheel_vels',
            self.wheel_listener_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            'imu_data',
            self.imu_listener_callback,
            10
        )

        self.ukf_odom_sub = self.create_subscription(
            Odometry,
            'odometry/filtered',
            self.ukf_listener_callback,
            10
        )

        # Publisher for NP-predicted odometry data
        self.np_odom_pub = self.create_publisher(
            Odometry,
            'np_odometry',
            20
        )

        # Timer to control the rate of the node
        self.create_timer(1.0 / dkf_odom_rate, self.odometry_publisher_callback)

        # Initialize the DKF model
        self.model = DeepKalmanFilter(
        hidden_size=128,
        observed_size=5,
        latent_size=5,
        control_size=2
        )

        # Load pretrained model weights
        if len(model_path) > 0:
            loaded_dict = torch.load(model_path)
            self.model.load_state_dict(loaded_dict['model_state_dict'])
            print(f'Loaded DKF model weights from {model_path}')

        # Initializing training/optimization parameters
        self.declare_parameter('batch_size', 50)
        self.declare_parameter('iters_before_optimize', 50)
        self.batch_size = self.get_parameter('batch_size').value
        self.iters_before_optimize = self.get_parameter('iters_before_optimize').value
        self.loss_function = DKFLoss()

        # DKF optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=learning_rate, weight_decay=weight_decay, amsgrad=True)

        # Use deque for the replay buffer to store past odometry data
        self.declare_parameter('max_buffer_size', int(5e5))  # Max size of the buffer
        self.max_buffer_size = self.get_parameter('max_buffer_size').value
        self.replay_buffer = deque(maxlen=self.max_buffer_size)
        self.purge_threshold = 100000  # Number of oldest tensors to purge

        # Load samples from CSV files into the replay buffer
        if load_samples:
            loaded_sample_low, loaded_sample_high = load_samples_from_csvs(
                data_dir=data_dir,
                time_step=1.0 / dkf_odom_rate,
                max_files=None
            )
            for i in range(loaded_sample_low.shape[0]):
                sample_low = loaded_sample_low[i].unsqueeze(0)   # (1,2,8)
                sample_high = loaded_sample_high[i].unsqueeze(0) # (1,2,8)
                self.replay_buffer.append((sample_low, sample_high))

            print(f'sample_low shape: {loaded_sample_low.shape}, sample_high shape: {loaded_sample_high.shape}')

        # Initialize GPS data parameters
        self.curr_pose = None
        self.pose_update_freq = 10  # Update pose from GPS every N iterations
        self.time_step = 1.0 / dkf_odom_rate  # Time step for integration (in seconds)

    def twist_listener_callback(self, msg):
        if self.twist_msg is None:
            self.prev_odom_low['twist_x'] = [msg.twist.twist.linear.x]
            self.prev_odom_low['twist_y'] = [msg.twist.twist.linear.y]

        self.twist_msg = msg

    def wheel_listener_callback(self, msg):
        if self.left_wheel_velocity is None or self.right_wheel_velocity is None:
            self.prev_odom_low['left_wheel_velocity'] = [msg.wheel_velocities[0]]
            self.prev_odom_low['right_wheel_velocity'] = [msg.wheel_velocities[1]]
        
        self.left_wheel_velocity = msg.wheel_velocities[0]
        self.right_wheel_velocity = msg.wheel_velocities[1]
    
    def imu_listener_callback(self, msg):
        if (math.isnan(msg.orientation.z) or math.isnan(msg.orientation.w) or 
            math.isnan(msg.angular_velocity.z)):
            self.get_logger().warning('NaN detected in IMU data, skipping update...')
            return

        if self.imu_msg is None:
            self.prev_odom_low['orientation'] = [msg.orientation.z, msg.orientation.w]
            self.prev_odom_low['thetadot'] = [msg.angular_velocity.z]

        self.imu_msg = msg

    def ukf_listener_callback(self, msg):
        if self.ground_truth_msg is None:
            self.prev_odom_high['twist_x'] = [msg.twist.twist.linear.x]
            self.prev_odom_high['twist_y'] = [msg.twist.twist.linear.y]
            self.prev_odom_high['orientation'] = [msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
            self.prev_odom_high['thetadot'] = [msg.twist.twist.angular.z]

        self.ground_truth_msg = msg

    # Callback function to publish np odometry data at a fixed rate
    def odometry_publisher_callback(self):

        # Initial check for odometry data
        if self.twist_msg is None or self.imu_msg is None or self.ground_truth_msg is None:
            self.get_logger().info('Waiting for all odometry data to be available...')
            return

        # Update current pose based on GPS data
        if self.internal_counter % self.pose_update_freq == 0 or self.curr_pose is None:
            self.curr_pose = torch.tensor([
                self.ground_truth_msg.pose.pose.position.x,
                self.ground_truth_msg.pose.pose.position.y
            ])

        # Converting odometry data into tensor and storing in replay buffer
        odom_low = {
            'twist_x': [self.twist_msg.twist.twist.linear.x],
            'twist_y': [self.twist_msg.twist.twist.linear.y],
            'orientation': [self.imu_msg.orientation.z, self.imu_msg.orientation.w],
            'thetadot': [self.imu_msg.angular_velocity.z],
            'left_wheel_velocity': [self.left_wheel_velocity],
            'right_wheel_velocity': [self.right_wheel_velocity]
        }
        odom_high = {
            'twist_x': [self.ground_truth_msg.twist.twist.linear.x],
            'twist_y': [self.ground_truth_msg.twist.twist.linear.y],
            'orientation': [self.ground_truth_msg.pose.pose.orientation.z, self.ground_truth_msg.pose.pose.orientation.w],
            'thetadot': [self.ground_truth_msg.twist.twist.angular.z],
        }

        prev_state_vec_low = torch.tensor(self.prev_odom_low['twist_x'] +  self.prev_odom_low['twist_y'] + self.prev_odom_low['orientation'] + self.prev_odom_low['thetadot'], dtype=torch.float32)
        next_state_vec_low = torch.tensor(odom_low['twist_x'] + odom_low['twist_y'] + odom_low['orientation'] + odom_low['thetadot'], dtype=torch.float32)
        prev_state_vec_high = torch.tensor(self.prev_odom_high['twist_x'] +  self.prev_odom_high['twist_y'] + self.prev_odom_high['orientation'] + self.prev_odom_high['thetadot'], dtype=torch.float32)
        next_state_vec_high = torch.tensor(odom_high['twist_x'] + odom_high['twist_y'] + odom_high['orientation'] + odom_high['thetadot'], dtype=torch.float32)
        additional_control = torch.tensor([self.time_step] + self.prev_odom_low['left_wheel_velocity'] + self.prev_odom_low['right_wheel_velocity'], dtype=torch.float32)
        
        sample_low = torch.cat((prev_state_vec_low, additional_control,  next_state_vec_low), dim=0).unsqueeze(0)  # Shape (1, 13)
        sample_high = torch.cat((prev_state_vec_high, additional_control, next_state_vec_high), dim=0).unsqueeze(0)  # Shape (1, 13)      
        self.replay_buffer.append((sample_low, sample_high))

        # Check if buffer size exceeds the purge threshold
        if len(self.replay_buffer) == self.max_buffer_size:
            self.get_logger().info(f"Purging {self.purge_threshold} oldest tensors from replay buffer...")
            for _ in range(self.purge_threshold):
                self.replay_buffer.popleft()

        # Normalizing the odometry tensor for DKF model
        batch_observed_transitions = minmax_norm(sample_low.clone())
        batch_latent_transitions = minmax_norm(sample_high.clone())

        # Getting model predictions
        z_pred, P_z, x_pred, R_x = self.model(batch_latent_transitions, batch_observed_transitions, is_testing=True)

        #print(f'pred_states: {pred_states}')
        #print(f'mu_low: {mu_low},\n mu_res: {mu_res}')
        #print(f'fused_mu: {fused_mu},\nsample_high: {norm_sample_high}\n')

        # Updating current pose based on predicted velocities using RK4
        pred_velocity = torch.tensor([z_pred[0, 0].item(), z_pred[0, 1].item()])
        pred_quat = torch.tensor([0, 0, z_pred[0, 2].item(), z_pred[0, 3].item()])
        pred_quat = pred_quat / torch.norm(pred_quat)
        self.curr_pose = runge_kutta_4(self.curr_pose, pred_velocity, pred_quat, self.time_step)

        # Training the model (only done after every 'self.batch_size' iterations to lower computation costs)
        if self.internal_counter % self.iters_before_optimize == 0 and self.internal_counter > 0:
            self.training_steps += 1
            self.extra_counter += 1
            self.get_logger().info('Training model...')
            self.model, loss, RMSE, RMSE_std, NLL = train_dkf_model(self.model, 
                                                                    self.replay_buffer, 
                                                                    self.optimizer, 
                                                                    self.loss_function, 
                                                                    self.batch_size)

            self.get_logger().info(f"Weight Update Complete! Step: {self.extra_counter}, Current loss: {loss.item()}, Current RMSE: {RMSE.item()}, Current RMSE Std: {RMSE_std.item()}")
            self.get_logger().info(f'Current NLL: {NLL.item()}\n')
            self.rmse_array.append(RMSE.item())
            self.nll_array.append(NLL.item())

            # Set the model back to testing mode
            self.model.eval()

        # Extracting predicted odometry state into ROS2 message to be published
        timestamp = self.ground_truth_msg.header.stamp.sec + self.ground_truth_msg.header.stamp.nanosec * 1e-9
        z_pred, P_z = minmax_unnorm(z_pred, P_z)
        pred_odom = Odometry()
        next_timestamp = timestamp + self.time_step
        pred_odom.header.stamp.sec = int(next_timestamp)
        pred_odom.header.stamp.nanosec = int((next_timestamp - int(next_timestamp)) * 1e9)
        pred_odom.header.frame_id = 'odom'
        pred_odom.child_frame_id = 'base_link'
        pred_odom.pose.pose.position.x = self.curr_pose[0].item()
        pred_odom.pose.pose.position.y = self.curr_pose[1].item()
        pred_odom.pose.pose.position.z = 0.0
        pred_odom.pose.pose.orientation.x = pred_quat[0].item()
        pred_odom.pose.pose.orientation.y = pred_quat[1].item()
        pred_odom.pose.pose.orientation.z = pred_quat[2].item()
        pred_odom.pose.pose.orientation.w = pred_quat[3].item()
        pred_odom.twist.twist.linear.x = z_pred[0, 0].item()
        pred_odom.twist.twist.linear.y = 0.0 #z_pred[0, 1].item()
        pred_odom.twist.twist.angular.z = z_pred[0, 4].item()
        self.np_odom_pub.publish(pred_odom)

        # Update previous LF/HF odometry states
        self.prev_odom_low['twist_x'] = odom_low['twist_x']
        self.prev_odom_low['twist_y'] = odom_low['twist_y']
        self.prev_odom_low['orientation'] = odom_low['orientation']
        self.prev_odom_low['thetadot'] = odom_low['thetadot']
        self.prev_odom_low['left_wheel_velocity'] = [self.left_wheel_velocity]
        self.prev_odom_low['right_wheel_velocity'] = [self.right_wheel_velocity]

        self.prev_odom_high['twist_x'] = odom_high['twist_x']
        self.prev_odom_high['twist_y'] = odom_high['twist_y']
        self.prev_odom_high['orientation'] = odom_high['orientation']
        self.prev_odom_high['thetadot'] = odom_high['thetadot']
        self.prev_odom_high['left_wheel_velocity'] = [self.left_wheel_velocity]
        self.prev_odom_high['right_wheel_velocity'] = [self.right_wheel_velocity]
        
        self.internal_counter += 1


def main(args=None):
    try:
        rclpy.init(args=args)
        # Now pass the model to the ROS2 node
        node = DKFOdometryEstimation()
        rclpy.spin(node)
        rclpy.shutdown()

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down node.")
        # Plot RMSE and NLL over time
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(node.rmse_array, '-ro')
        plt.title('RMSE Convergence Dynamics')
        plt.xlabel('Training Step')
        plt.ylabel('RMSE')

        plt.subplot(1, 2, 2)
        plt.plot(node.nll_array, '-bo')
        plt.title('NLL Convergence Dynamics')
        plt.xlabel('Training Step')
        plt.ylabel('NLL')
        plt.legend()

        plt.tight_layout()
        plt.savefig('/home/devin1126/dev_ws/src/np_localization/np_localization/rmse_nll_plot.png')
        print("Plot saved to rmse_nll_plot.png")
        np.save('rmse_array.npy', np.array(node.rmse_array))
        np.save('nll_array.npy', np.array(node.nll_array))
        print("RMSE and NLL arrays saved as .npy files.")



if __name__ == '__main__':
    main()
