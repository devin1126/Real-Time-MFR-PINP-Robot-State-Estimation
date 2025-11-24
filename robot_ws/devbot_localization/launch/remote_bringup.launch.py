#!/usr/bin/env python3
import launch
from launch import LaunchDescription
from launch_ros.actions import Node  # Updated import statement
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('devbot_localization')
    return LaunchDescription([
        # Static transform publisher between base_link and imu_link
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_imu_transform",
            arguments=["0.0", "0.0", "0.1339", "0.0", "0.0", "0.0", "base_link", "imu_link"]
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_lidar_transform",
            arguments=["0.1910", "0.0", "0.2156", "0.0", "0.0", "-1.5708", "base_link", "laser_link"]
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_camera_transform",
            arguments=["0.2445", "0.0", "0.1214", "-1.5708", "0.0", "-1.5708", "base_link", "camera_link"]
        ),

        Node(
            package='robot_localization',
            executable='ukf_node',
            name='ukf_localization_node',
            output='screen',
            parameters=[pkg_share + '/config/ukf.yaml'],
        ),

        Node(
            package='devbot_teleop',
            executable='teleop_control.py',
            name='teleop_transmitter',
            output='screen'
        )
    ])
