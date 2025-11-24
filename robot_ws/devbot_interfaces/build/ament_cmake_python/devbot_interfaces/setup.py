from setuptools import find_packages
from setuptools import setup

setup(
    name='devbot_interfaces',
    version='0.0.0',
    packages=find_packages(
        include=('devbot_interfaces', 'devbot_interfaces.*')),
)
