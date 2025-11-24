#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
import sys
import select
import tty
import termios


class TeleopControl(Node):

    def __init__(self):
        super().__init__('teleop_control')
        self.pub = self.create_publisher(String, 'teleop_cmd', 10)
        self.teleop_cmd = String()
        self.teleop_cmd.data = ''
        self.running = True

        # Timer to publish command at a fixed frequency (e.g., 100 Hz)
        self.timer = self.create_timer(0.01, self.publish_cmd)  # 100 Hz interval
        self.get_logger().info("Teleop Control Started!")
        self.get_logger().info("Use Arrow keys to control:")
        self.get_logger().info("  ↑ - Forward")
        self.get_logger().info("  ↓ - Backward") 
        self.get_logger().info("  ← - Left")
        self.get_logger().info("  → - Right")
        self.get_logger().info("  Q - Quit")

    def get_key(self):
        """Get a single keypress from stdin, handling escape sequences for arrow keys"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            
            # Handle escape sequences (arrow keys)
            if ord(ch) == 27:  # ESC sequence
                ch += sys.stdin.read(2)  # Read the next 2 characters
                
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def keyboard_thread(self):
        """Thread function to handle keyboard input"""
        while self.running:
            try:
                key = self.get_key()
                
                # Handle arrow keys (escape sequences)
                if key == '\x1b[A':  # Up arrow
                    self.teleop_cmd.data = 'forward'
                    self.get_logger().info("Forward!")
                elif key == '\x1b[B':  # Down arrow
                    self.teleop_cmd.data = 'backward'
                    self.get_logger().info("Backward!")
                elif key == '\x1b[D':  # Left arrow
                    self.teleop_cmd.data = 'left'
                    self.get_logger().info("Left!")
                elif key == '\x1b[C':  # Right arrow
                    self.teleop_cmd.data = 'right'
                    self.get_logger().info("Right!")
                elif key.lower() == 'q':
                    self.get_logger().info("Teleop node terminated!")
                    self.running = False
                    break
                else:
                    self.teleop_cmd.data = ''
                    
            except Exception as e:
                self.get_logger().error(f"Keyboard input error: {e}")
                break

    def publish_cmd(self):
        """Publish the teleoperation command periodically."""
        if self.running:
            self.pub.publish(self.teleop_cmd)


def main(args=None):
    teleop_control = None
    try:
        rclpy.init(args=args)
        teleop_control = TeleopControl()

        # Start the keyboard input thread
        keyboard_thread = threading.Thread(target=teleop_control.keyboard_thread)
        keyboard_thread.daemon = True
        keyboard_thread.start()

        # Spin the ROS node
        while teleop_control.running and rclpy.ok():
            rclpy.spin_once(teleop_control, timeout_sec=0.1)

    except KeyboardInterrupt:
        # Handle the case where the user interrupts the program
        if teleop_control:
            teleop_control.get_logger().info("Keyboard Interrupt, shutting down...")
            teleop_control.running = False

    except Exception as e:
        # Handle other exceptions
        if teleop_control:
            teleop_control.get_logger().error(f"An error occurred: {e}")
            teleop_control.running = False

    finally:
        # Cleanup
        if teleop_control:
            teleop_control.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()   