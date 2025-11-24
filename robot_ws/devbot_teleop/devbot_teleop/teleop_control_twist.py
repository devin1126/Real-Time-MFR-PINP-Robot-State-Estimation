#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import TwistWithCovarianceStamped
from std_msgs.msg import Header
import serial
import time

class TeleopControlAndTwistPublisher(Node):
    def __init__(self):
        super().__init__('teleop_control_and_twist_publisher')

        # Declare parameters
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 57600)
        self.declare_parameter('linear_velocity_covariance', 6e-4)  # variance, not sqrt
        self.declare_parameter('angular_velocity_covariance', 0.2)  # variance, not sqrt
        self.declare_parameter('frame_id', 'base_link')

        # Get parameters
        self.serial_port = self.get_parameter('serial_port').value
        self.baud_rate = self.get_parameter('baud_rate').value
        self.linear_velocity_covariance = self.get_parameter('linear_velocity_covariance').value
        self.angular_velocity_covariance = self.get_parameter('angular_velocity_covariance').value
        self.frame_id = self.get_parameter('frame_id').value

        # Serial setup
        self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=0.1)
        self.ser.dtr = False
        time.sleep(0.1)
        self.ser.dtr = True
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        # Subscribers and Publishers
        self.teleop_sub = self.create_subscription(String, 'teleop_cmd', self.teleop_cmd_callback, 10)
        self.velocity_pub = self.create_publisher(TwistWithCovarianceStamped, 'wheel_twist', 10)

        # Preallocate Twist message
        self.twist_msg = TwistWithCovarianceStamped()
        self.twist_msg.header.frame_id = self.frame_id

        # Timer
        self.create_timer(0.05, self.read_and_publish_velocity)

        # Log initialization
        self.get_logger().info(f"Teleop Control Node started on {self.serial_port} at {self.baud_rate} baud")

    def teleop_cmd_callback(self, msg):
        self.send_to_arduino(msg.data)

    def send_to_arduino(self, direction):
        try:
            self.ser.write((direction + '\n').encode("utf-8"))
        except Exception as e:
            self.get_logger().error(f"Error sending to Arduino: {e}")

    def read_and_publish_velocity(self):
        try:
            if self.ser.in_waiting > 0:
                raw_data = self.ser.readline().decode('utf-8', errors='ignore').strip()
                vel_data = [float(v) for v in raw_data.split(',')]

                if len(vel_data) != 2:
                    self.get_logger().warn(f"Malformed velocity data: {raw_data}", throttle_duration_sec=5.0)
                    return

                twist_msg = self.twist_msg
                twist_msg.header.stamp = self.get_clock().now().to_msg()
                twist_msg.twist.twist.linear.x = vel_data[0]
                twist_msg.twist.twist.angular.z = vel_data[1]
                twist_msg.twist.covariance[0] = self.linear_velocity_covariance
                twist_msg.twist.covariance[35] = self.angular_velocity_covariance
                self.velocity_pub.publish(twist_msg)
            else:
                self.ser.reset_input_buffer()

        except Exception as e:
            self.get_logger().error(f"Velocity read error: {e}")

    def cleanup(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

def main(args=None):
    rclpy.init(args=args)
    node = TeleopControlAndTwistPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

