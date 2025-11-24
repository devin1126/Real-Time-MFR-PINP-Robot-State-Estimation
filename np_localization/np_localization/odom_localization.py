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
from np_localization.mfr_pinp_model_structure import LowFidelityNP, ResidualHighFidelityNP
from np_localization.np_model_utils import minmax_norm, minmax_unnorm, compute_state_priors, train_np_model
from np_localization.np_model_utils import runge_kutta_4, load_samples_from_csvs, ELBO
from collections import deque
import matplotlib.pyplot as plt


class NPOdometryEstimation(Node):
    def __init__(self):
        super().__init__('np_odometry_estimation_node')
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
        self.nll_marginal_array = []

        # Load the model
        self.declare_parameter('model_path_low', '/home/devin1126/dev_ws/src/np_localization/models/gazebo-trained-model.pt')
        self.declare_parameter('model_path_res', '/home/devin1126/dev_ws/src/np_localization/models/gazebo-trained-model-res.pt')
        self.declare_parameter('enable_ode_prior', True)   # Activate PI version of algorithm
        self.declare_parameter('learning_rate', 1e-3)
        self.declare_parameter('weight_decay', 5e-7)
        self.declare_parameter('np_odom_rate', 50)  # Frequency of the np odometry node
        self.declare_parameter('load_samples_from_csvs', True)
        self.declare_parameter('data_dir', '/home/devin1126/dev_ws/src/dynamic_data_collector/combined_data')

        self.enable_ode_prior = self.get_parameter('enable_ode_prior').value
        model_path_low = self.get_parameter('model_path_low').value
        model_path_res = self.get_parameter('model_path_res').value
        learning_rate = self.get_parameter('learning_rate').value
        np_odom_rate = self.get_parameter('np_odom_rate').value
        weight_decay = self.get_parameter('weight_decay').value
        load_samples = self.get_parameter('load_samples_from_csvs').value
        data_dir = self.get_parameter('data_dir').value
        self.q_marginal = None

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
        self.create_timer(1.0 / np_odom_rate, self.odometry_publisher_callback)

        # Initialize the NP models
        self.model_low = LowFidelityNP(state_dim=5, control_dim=2,device=torch.device('cpu'), enable_ode_prior=self.enable_ode_prior)
        self.model_res = ResidualHighFidelityNP(state_dim=5, control_dim=2, device=torch.device('cpu'), enable_ode_prior=False)
       # self.model_low.load_state_dict(torch.load(model_path_low))
       # self.model_res.load_state_dict(torch.load(model_path_res))
        self.model_low.eval()  # Set the model to testing mode
        self.model_res.eval()  # Set the model to testing mode

        # Initialize a frozen copy of the low-fidelity model for stability
        self.FREEZE_UPDATE_STEPS = 5  # Number of training steps between target model updates
        self.model_low_target = LowFidelityNP(state_dim=5, control_dim=2, device=torch.device('cpu'), enable_ode_prior=self.enable_ode_prior)
        self.model_low_target.load_state_dict(self.model_low.state_dict())
        self.model_low_target.eval()
        for param in self.model_low_target.parameters():
            param.requires_grad = False

        # Initializing training/optimization parameters
        self.declare_parameter('batch_size', 50)
        self.declare_parameter('iters_before_optimize', 50)
        self.batch_size = self.get_parameter('batch_size').value
        self.iters_before_optimize = self.get_parameter('iters_before_optimize').value
        self.loss_function = ELBO()

        # Joint optimizer
        self.joint_optimizer = torch.optim.Adam(
            list(self.model_low.parameters()) + list(self.model_res.parameters()),
            lr=learning_rate, weight_decay=weight_decay, amsgrad=True
        )

        # Use deque for the replay buffer to store past odometry data
        self.declare_parameter('max_buffer_size', int(5e5))  # Max size of the buffer
        self.max_buffer_size = self.get_parameter('max_buffer_size').value
        self.replay_buffer = deque(maxlen=self.max_buffer_size)
        self.purge_threshold = 100000  # Number of oldest tensors to purge

        # Load samples from CSV files into the replay buffer
        if load_samples:
            loaded_sample_low, loaded_sample_high = load_samples_from_csvs(
                data_dir=data_dir,
                time_step=1.0 / np_odom_rate,
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
        self.time_step = 1.0 / np_odom_rate  # Time step for integration (in seconds)

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

        #self.get_logger().info(f"left_wheel_velocity: {self.left_wheel_velocity}, right_wheel_velocity: {self.right_wheel_velocity}")
        #self.get_logger().info(f'twist_x: {self.prev_odom_low['twist_x']}, twist_y: {self.prev_odom_low['twist_y']}, orientation: {self.prev_odom_low['orientation']}, thetadot: {self.prev_odom_low['thetadot']}')
        #print(f'prev_odom_low: {self.prev_odom_low}')
        tensor1 = torch.tensor([0.0] + self.prev_odom_low['left_wheel_velocity'] + self.prev_odom_low['right_wheel_velocity'] + self.prev_odom_low['twist_x'] + 
                               self.prev_odom_low['twist_y'] + self.prev_odom_low['orientation'] + self.prev_odom_low['thetadot'], dtype=torch.float32)
        tensor2 = torch.tensor([self.time_step] + [0]*2 + odom_low['twist_x'] + odom_low['twist_y'] + odom_low['orientation'] + odom_low['thetadot'], dtype=torch.float32)

        tensor3 = torch.tensor([0.0] + self.prev_odom_low['left_wheel_velocity'] + self.prev_odom_low['right_wheel_velocity'] + self.prev_odom_high['twist_x'] +
                                self.prev_odom_high['twist_y'] + self.prev_odom_high['orientation'] + self.prev_odom_high['thetadot'], dtype=torch.float32)
        tensor4  = torch.tensor([self.time_step] + [0]*2 + odom_high['twist_x'] + odom_high['twist_y'] + odom_high['orientation'] + odom_high['thetadot'], dtype=torch.float32)
        #print(f'tensor1: {tensor1},\n tensor2: {tensor2}')
        sample_low = torch.cat((tensor1.unsqueeze(0), tensor2.unsqueeze(0)), dim=0).unsqueeze(0)
        sample_high = torch.cat((tensor3.unsqueeze(0), tensor4.unsqueeze(0)), dim=0).unsqueeze(0)        
        self.replay_buffer.append((sample_low, sample_high))

        # Check if buffer size exceeds the purge threshold
        if len(self.replay_buffer) == self.max_buffer_size:
            self.get_logger().info(f"Purging {self.purge_threshold} oldest tensors from replay buffer...")
            for _ in range(self.purge_threshold):
                self.replay_buffer.popleft()

        # Normalizing the odometry tensor for NP model
        norm_sample_low = minmax_norm(sample_low.clone())
        norm_sample_high = minmax_norm(sample_high.clone())

        # Defining context/target sets for NP model
        context_x = norm_sample_low[:,:1,:3].clone()
        context_y_low = norm_sample_low[:,:1,3:].clone()
        target_x = norm_sample_low[:,:,:3].clone()

        # Initializing query object for NP model
        query_low = ((context_x, context_y_low), target_x)

        # Predicting the next odometry state and capturing loss info
        pred_states = compute_state_priors(sample_low, dt=self.time_step)
        #print(f'NEXT: sample_low: {sample_low}, sample_high: {sample_high}, pred_states: {pred_states}\n')
        mu_low, sigma_low, z_low = self.model_low(query_low, pred_states=pred_states, is_testing=True)
  
        # Initializing query object for model
        query_res = ((context_x, mu_low), target_x)

        # Predicting the residual dynamcis and capturing loss info
        query_res = ((context_x, mu_low.detach()), target_x)
        residual_y = norm_sample_high[:,:,3:] - mu_low.detach()
        mu_res, sigma_res  = self.model_res(query_res, z_low=z_low.detach(), residual_y=residual_y, is_testing=True)

        # Fusing the low fidelity and residual model predictions
        fused_mu = mu_low + mu_res
        fused_var = sigma_low**2 + sigma_res**2

        #print(f'pred_states: {pred_states}')
        #print(f'mu_low: {mu_low},\n mu_res: {mu_res}')
        #print(f'fused_mu: {fused_mu},\nsample_high: {norm_sample_high}\n')

        # Updating current pose based on predicted velocities using RK4
        pred_velocity = torch.tensor([fused_mu[0, -1, 0].item(), fused_mu[0, -1, 1].item()])
        pred_quat = torch.tensor([0, 0, fused_mu[0, -1, 2].item(), fused_mu[0, -1, 3].item()])
        pred_quat = pred_quat / torch.norm(pred_quat)
        self.curr_pose = runge_kutta_4(self.curr_pose, pred_velocity, pred_quat, self.time_step)

        # Training the model (only done after every 'self.batch_size' iterations to lower computation costs)
        if self.internal_counter % self.iters_before_optimize == 0 and self.internal_counter > 0:
            self.training_steps += 1
            self.extra_counter += 1
            self.get_logger().info('Training model...')
            self.model_low, self.model_res, loss, RMSE, RMSE_std, NLL, NLL_marginal, q_marginal = train_np_model(self.model_low,
                                                                                self.model_res, 
                                                                                self.replay_buffer, 
                                                                                self.joint_optimizer, 
                                                                                self.loss_function, 
                                                                                self.batch_size, 
                                                                                self.model_low_target,
                                                                                self.enable_ode_prior)

            self.get_logger().info(f"Weight Update Complete! Step: {self.extra_counter}, Current loss: {loss.item()}, Current RMSE: {RMSE.item()}, Current RMSE Std: {RMSE_std.item()}")
            self.get_logger().info(f'Current NLL (uncalibrated): {NLL.item()}, Current NLL (marginal): {NLL_marginal.item()}\n')
            self.rmse_array.append(RMSE.item())
            self.nll_array.append(NLL.item())
            self.nll_marginal_array.append(NLL_marginal.item())
            self.q_marginal = q_marginal

            # Set the models back to testing mode
            self.model_low.eval() 
            self.model_res.eval()  

        # Extracting predicted odometry state into ROS2 message to be published
        timestamp = self.ground_truth_msg.header.stamp.sec + self.ground_truth_msg.header.stamp.nanosec * 1e-9
        fused_mu, fused_var = minmax_unnorm(fused_mu, fused_var)
        if self.q_marginal is not None: fused_var[:,-1,:] *= self.q_marginal.unsqueeze(0)  # Adjust variance with conformal quantiles
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
        pred_odom.twist.twist.linear.x = fused_mu[0,-1,0].item()
        pred_odom.twist.twist.linear.y = 0.0 #fused_mu[0,-1,1].item()
        pred_odom.twist.twist.angular.z = fused_mu[0,-1,4].item()
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

        # --- Update frozen LF target every N training steps ---
        if self.training_steps % self.FREEZE_UPDATE_STEPS == 0 and self.training_steps > 0:
            print("Updating frozen low-fidelity model!")
            self.model_low_target.load_state_dict(self.model_low.state_dict())
            self.training_steps = 0  # Reset training steps
        
        self.internal_counter += 1


def main(args=None):
    try:
        rclpy.init(args=args)

        # Now pass the model to the ROS2 node
        node = NPOdometryEstimation()
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
        plt.plot(node.nll_array, '-bo',label='Uncalibrated')
        plt.plot(node.nll_marginal_array, '-go', label='Split Conformal Calibrated')
        plt.title('NLL Convergence Dynamics')
        plt.xlabel('Training Step')
        plt.ylabel('NLL')
        plt.legend()

        plt.tight_layout()
        plt.savefig('/home/devin1126/dev_ws/src/np_localization/np_localization/rmse_nll_plot.png')
        print("Plot saved to rmse_nll_plot.png")
        np.save('rmse_array.npy', np.array(node.rmse_array))
        np.save('nll_array.npy', np.array(node.nll_array))
        np.save('nll_marginal_array.npy', np.array(node.nll_marginal_array))
        print("RMSE and NLL arrays saved as .npy files.")



if __name__ == '__main__':
    main()
