from setuptools import find_packages, setup

package_name = 'np_localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),  # Automatically finds all submodules
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='devin1126',
    maintainer_email='devin.hunter2017@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'odom_localization = np_localization.odom_localization:main',
        ]
    },
)