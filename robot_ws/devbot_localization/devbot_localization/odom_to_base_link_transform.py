import rclpy
from rclpy.node import Node
import tf2_ros
import geometry_msgs.msg

class TransformListenerNode(Node):
    def __init__(self):
        super().__init__('dynamic_transform_listener')
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.timer = self.create_timer(0.1, self.timer_callback)  # check every 0.1 seconds
        
    def timer_callback(self):
        try:
            # Look for transform between base_link and odom
            transform = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            self.get_logger().info(f"Transform: {transform}")
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"Could not get transform: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = TransformListenerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
