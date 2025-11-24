#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from geometry_msgs.msg import TwistWithCovarianceStamped
from sensor_msgs.msg import Imu
from devbot_interfaces.msg import WheelVels
import serial
import time


class TeleopSubTwistIMUPub(Node):
    def __init__(self):
        super().__init__('teleop_twist_imu_node')

        # ===== Parameters =====
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 57600)
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('imu_frame', 'imu_link')
        self.declare_parameter('twist_publish_rate', 50.0)  # Hz
        self.declare_parameter('imu_publish_rate', 50.0)   # Hz
        self.declare_parameter('twist_linear_velocity_covariance', 6e-4) 
        self.declare_parameter('twist_angular_velocity_covariance', 0.2)  
        self.declare_parameter('imu_orientation_covariance', 2e-4)
        self.declare_parameter('imu_angvel_covariance', 0.003)
        self.declare_parameter('imu_linacc_covariance', 0.6)

        self.serial_port = self.get_parameter('serial_port').value
        self.baud_rate = self.get_parameter('baud_rate').value
        self.frame_id = self.get_parameter('frame_id').value
        self.imu_frame = self.get_parameter('imu_frame').value
        self.imu_pub_rate = self.get_parameter('imu_publish_rate').value
        self.twist_linear_velocity_covariance = self.get_parameter('twist_linear_velocity_covariance').value
        self.twist_angular_velocity_covariance = self.get_parameter('twist_angular_velocity_covariance').value
        self.imu_orientation_covariance = self.get_parameter('imu_orientation_covariance').value
        self.imu_angvel_covariance = self.get_parameter('imu_angvel_covariance').value
        self.imu_linacc_covariance = self.get_parameter('imu_linacc_covariance').value

        # ===== Serial Setup =====
        self.get_logger().info(f"Connecting to Arduino on {self.serial_port}...")
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=0.1)
            time.sleep(2.0)  # Allow Arduino reset time
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception as e:
            self.get_logger().error(f"Serial open failed: {e}")
            raise

        # ===== Send Handshake =====
        self.get_logger().info("Sending rosConnected handshake...")
        self.ser.write(b"rosConnected\n")

        # Wait for ACK
        ack_received = False
        start_time = time.time()
        while time.time() - start_time < 5.0:  # 5-second timeout
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if "ROS_ACK" in line:
                    ack_received = True
                    self.get_logger().info("Arduino handshake ACK received.")
                    break
            time.sleep(0.1)
        if not ack_received:
            self.get_logger().warn("No ROS_ACK received from Arduino — continuing anyway.")

        # ===== ROS Pub/Sub =====
        self.create_subscription(String, 'teleop_cmd', self.teleop_cmd_callback, 10)
        self.twist_pub = self.create_publisher(TwistWithCovarianceStamped, 'wheel_twist', 10)
        self.wheel_pub = self.create_publisher(WheelVels, 'wheel_vels', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu_data', 20)


        # ===== Timers =====
        self.create_timer(0.01, self.serial_read_loop)
        self.create_timer(1.0 / self.imu_pub_rate, self.publish_imu_from_buffer)

        self.last_imu_data = None
        self.last_mag_data = None

        self.get_logger().info("Handshake complete. ROS 2 <-> Arduino link established.")

    # -----------------------------------------------------------
    def teleop_cmd_callback(self, msg):
        try:
            self.ser.write((msg.data.strip() + '\n').encode('utf-8'))
        except Exception as e:
            self.get_logger().error(f"Serial write failed: {e}")

    # -----------------------------------------------------------
    def serial_read_loop(self):
        try:
            while self.ser.in_waiting > 0:
                raw = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if raw.startswith("VEL,"):
                    self.handle_wheel_data(raw[4:])
                elif raw.startswith("IMU,"):
                    self.handle_imu_data(raw[4:])
                elif raw not in ("ROS_ACK", "WAITING_FOR_ROS", ""):
                    self.get_logger().warn(f"Unrecognized serial data: {raw}")
        except Exception as e:
            self.get_logger().error(f"Serial read error: {e}")

    # -----------------------------------------------------------
    def handle_wheel_data(self, data):
        try:
            vals = [float(v) for v in data.split(',')]
            if len(vals) != 4:
                return
            
            # Publish twist messages
            twist_msg = TwistWithCovarianceStamped()
            twist_msg.header = Header()
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.header.frame_id = self.frame_id
            twist_msg.twist.twist.linear.x = vals[0]
            twist_msg.twist.twist.angular.z = vals[1]
            twist_msg.twist.covariance[0] = self.twist_linear_velocity_covariance
            twist_msg.twist.covariance[7] = self.twist_linear_velocity_covariance
            twist_msg.twist.covariance[14] = self.twist_linear_velocity_covariance
            twist_msg.twist.covariance[21] = self.twist_angular_velocity_covariance
            twist_msg.twist.covariance[28] = self.twist_angular_velocity_covariance   
            twist_msg.twist.covariance[35] = self.twist_angular_velocity_covariance
            self.twist_pub.publish(twist_msg)

            # Publish wheel velocity messages
            wheel_msg = WheelVels()
            wheel_msg.wheel_velocities = vals[2:]
            self.wheel_pub.publish(wheel_msg)

        except Exception as e:
            self.get_logger().warn(f"Wheel parse error: {e}")

    def handle_imu_data(self, data):
        try:
            vals = [float(v) for v in data.split(',')]
            if len(vals) != 6:
                return
            self.last_imu_data = vals
        except Exception as e:
            self.get_logger().warn(f"IMU parse error: {e}")
            pass


    def publish_imu_from_buffer(self):
        if self.last_imu_data is None:
            return
        qx, qy, qz, qw, gz, ax = self.last_imu_data
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.imu_frame
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw
        msg.orientation_covariance = [
                self.imu_orientation_covariance, 0.0, 0.0,
                0.0, self.imu_orientation_covariance, 0.0,
                0.0, 0.0, self.imu_orientation_covariance,
            ]
        
        msg.angular_velocity.z = gz
        msg.angular_velocity_covariance = [
                self.imu_angvel_covariance, 0.0, 0.0,
                0.0, self.imu_angvel_covariance, 0.0,
                0.0, 0.0, self.imu_angvel_covariance,
            ]
        
        msg.linear_acceleration.x = ax
        msg.linear_acceleration_covariance = [
                self.imu_linacc_covariance, 0.0, 0.0,
                0.0, self.imu_linacc_covariance, 0.0,
                0.0, 0.0, self.imu_linacc_covariance,
            ]
        self.imu_pub.publish(msg)

    def destroy_node(self):
        try:
            if self.ser.is_open:
                self.ser.write(b"rosDisconnected\n")
                self.ser.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TeleopSubTwistIMUPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
