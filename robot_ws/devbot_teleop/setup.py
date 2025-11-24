from setuptools import find_packages, setup

package_name = 'devbot_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='devin1126',
    maintainer_email='devin1126@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'teleop_control_twist_IMU = devbot_teleop.teleop_receiver_twist_IMU_publish:main',
        'teleop_control_twist = devbot_teleop.teleop_control_twist:main',
        'teleop_control = devbot_teleop.teleop_control:main'
        ],
    },
)
