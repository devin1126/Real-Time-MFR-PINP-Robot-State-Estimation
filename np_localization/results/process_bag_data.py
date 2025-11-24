import rosbag2_py
import numpy as np
import os
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def process_bag_file(bag_file_path, output_dir):
    """
    Reads odometry data from a ROS2 bag file and saves each dynamic state as a separate numpy array.
    :param bag_file_path: Path to the ROS2 bag file.
    :param output_dir: Directory to save the numpy arrays.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize storage and reader
    storage_options = rosbag2_py.StorageOptions(uri=bag_file_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Initialize dictionaries to store time series data
    data = {
        '/odom': {'position_x': [], 'position_y': [], 'linear_vel_x': [], 'linear_vel_y': [], 'q_z': [], 'q_w': [], 'yaw_vel': []},
        '/np_odometry': {'position_x': [], 'position_y': [], 'linear_vel_x': [], 'linear_vel_y': [], 'q_z': [], 'q_w': [], 'yaw_vel': []},
        '/odometry/filtered': {'position_x': [], 'position_y': [], 'linear_vel_x': [], 'linear_vel_y': [], 'q_z': [], 'q_w': [], 'yaw_vel': []}
    }

    # Read messages from the bag file
    while reader.has_next():
        topic, msg_bytes, _ = reader.read_next()
        if topic in data:
            msg = deserialize_message(msg_bytes, get_message("nav_msgs/msg/Odometry"))  # Corrected usage
            data[topic]['position_x'].append(msg.pose.pose.position.x)
            data[topic]['position_y'].append(msg.pose.pose.position.y)
            data[topic]['linear_vel_x'].append(msg.twist.twist.linear.x)
            data[topic]['linear_vel_y'].append(msg.twist.twist.linear.y)
            data[topic]['q_z'].append(msg.pose.pose.orientation.z)  # Quaternion z
            data[topic]['q_w'].append(msg.pose.pose.orientation.w)  # Quaternion w
            data[topic]['yaw_vel'].append(msg.twist.twist.angular.z)

    # Save each time series as a numpy array
    for topic, states in data.items():
        for state, values in states.items():
            array_name = f"{topic.strip('/').replace('/', '_')}_{state}"
            np.save(os.path.join(output_dir, f"{array_name}.npy"), np.array(values))
            print(f"Saved {array_name}.npy")

# Example usage
if __name__ == "__main__":
    bag_file_path = "/home/devin1126/dev_ws/src/np_localization/results/results_0.db3"
    output_dir = "/home/devin1126/dev_ws/src/np_localization/results/numpy_arrays"
    process_bag_file(bag_file_path, output_dir)
