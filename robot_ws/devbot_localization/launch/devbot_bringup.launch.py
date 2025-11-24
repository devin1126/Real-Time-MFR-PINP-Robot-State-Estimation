#!/usr/bin/env python3
import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('devbot_localization')
    return LaunchDescription([
        # Teleop control + Twist/IMU publisher node
        Node(
            package='devbot_teleop',
            executable='teleop_control_twist_IMU',
            name='teleop_receiver',
            output='screen',
            parameters=[{
                'serial_port': '/dev/ttyACM0',
                'baud_rate': 57600,
                'linear_velocity_covariance': 6e-4,
                'angular_velocity_covariance': 0.2,
                'frame_id': 'base_link'
            }]
        ),

        # Static transform publisher between base_link and imu_link
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_imu_transform",
            arguments=["0.0", "0.0", "0.1339", "0.0", "0.0", "0.0", "base_link", "imu_link"]
        ),

        # RPLidar A2M8 driver node
        Node(
            package='rplidar_ros',
            executable='rplidar_node',
            name='rplidar',
            output='screen',
            parameters=[{
                'serial_port': '/dev/ttyUSB0',
                'serial_baudrate': 115200,
                'frame_id': 'laser_link'
            }]
        ),

        # Static transform publisher between base_link and laser_link
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_lidar_transform",
            arguments=["--x", "0.1810",
                    "--y", "0.0",
                    "--z", "0.2000",
                    "--roll", "0.0",
                    "--pitch", "0.0",
                    "--yaw", "-1.5708",
                    "--frame-id", "base_link",
                    "--child-frame-id", "laser_link"],
        ),


        # ICP Odometry node from rtabmap_odom
        Node(
            package='rtabmap_odom',
            executable='icp_odometry',
            name='icp_odometry',
            output='screen',
            parameters=[{
                'frame_id': 'base_link',
                'odom_frame_id': 'odom',
                'subscribe_scan': True,
                'subscribe_scan_cloud': False,
                'subscribe_rgbd': False,
                'subscribe_stereo': False,
                'approx_sync': True,
                'use_sim_time': False,
                'icp/PointToPlane': True,
                'icp/MaxCorrespondenceDistance': '0.5',
                'Icp/RangeMin': '0.1',
                'Icp/RangeMax': '8.0',
                'Icp/PointToPlaneK': '10',
                'Icp/VoxelSize': '0.05',
                'Icp/MaxTranslation': '1.0',
                'Icp/MaxRotation': '0.5',
                'wait_for_transform_duration': '0.2',
                'publish_null_when_lost': False,
                'publish_tf': True
            }],
            remappings=[
                ('scan', '/scan'),
                ('odom', '/odom_lidar')
            ]
        ),

        # Robot Localization UKF node
        Node(
            package='robot_localization',
            executable='ukf_node',
            name='ukf_localization_node',
            output='screen',
            parameters=[pkg_share + '/config/ukf.yaml'],
        )
    ])


'''
 # RealSense camera node (Wi-Fi optimized)
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            output='screen',
            parameters=[{
                'enable_infra1': True,
                'enable_infra2': True,
                'enable_depth': True,
                'enable_color': False,
                'enable_gyro': True,
                'enable_accel': True,

                'unite_imu_method': 1,
                'enable_sync': True,
                'publish_tf': True,

                'qos_image': 'BEST_EFFORT',
                'qos_imu': 'BEST_EFFORT',
            }],
        ),


# Image transport republishers for Wi-Fi (compressed topics)
        # Infrared 1 republisher
        Node(
            package='image_transport',
            executable='republish',
            name='infra1_republish_compressed',
            arguments=['raw', 'compressed'],
            remappings=[
                ('in', '/camera/camera/infra1/image_rect_raw'),
                ('out', '/camera/camera/infra1/image_rect_compressed')
            ]
        ),

        # Optional: Infrared 2 republisher (uncomment if IR2 is enabled above)
        Node(
            package='image_transport',
            executable='republish',
            name='infra2_republish_compressed',
            arguments=['raw', 'compressed'],
            remappings=[
                ('in', '/camera/camera/infra2/image_rect_raw'),
                ('out', '/camera/camera/infra2/image_rect_compressed')
            ]
        ),

    # IMU Madgwick filter node
        Node(
            package='imu_filter_madgwick',
            executable='imu_filter_madgwick_node',
            name='imu_filter',
            output='screen',
            parameters=[{
                'use_mag': False,
                'world_frame': 'odom',
                'base_imu_frame': 'imu_link',
                'output_frame': 'base_link',
                'publish_tf': True,
                'frequency': 30.0,
                'beta': 0.1,
                'z_up': False
            }],
            remappings=[
                ('/imu/data_raw', '/imu_data')  # <— remap your actual IMU topic
            ]
        ),


    # IMU Madgwick filter node
        Node(
            package='imu_filter_madgwick',
            executable='imu_filter_madgwick_node',
            name='imu_filter',
            output='screen',
            parameters=[{
                'use_mag': True,
                'world_frame': 'odom',
                'base_imu_frame': 'imu_link',
                'output_frame': 'base_link',
                'publish_tf': True,
                'frequency': 30.0,
                'beta': 0.1,
                'z_up': False
            }],
            remappings=[
                ('/imu/data_raw', '/imu_data')  # <— remap your actual IMU topic
            ]
        ),

'''