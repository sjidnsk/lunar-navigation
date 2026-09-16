from glob import glob
from setuptools import setup

name = 'lunar_obj_tcp_sim'
setup(
    name=name, version='0.1.0', packages=[name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + name]),
        ('share/' + name, ['package.xml', 'README.md']),
        ('share/' + name + '/launch', glob('launch/*.launch.py')),
        ('share/' + name + '/config', glob('config/*.yaml') + glob('config/*.json')),
        ('share/' + name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='Kai', maintainer_email='kai@example.com', license='Apache-2.0',
    description='OBJ virtual observation and C++ VehicleID compatible TCP vehicle integration',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'prepare_obj_map = lunar_obj_tcp_sim.prepare:main',
        'tcp_vehicle_bridge = lunar_obj_tcp_sim.bridge_node:main',
        'obj_virtual_sensor = lunar_obj_tcp_sim.sensor_node:main',
        'obj_scan_controller = lunar_obj_tcp_sim.scan_controller:main',
    ]},
)
