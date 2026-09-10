from glob import glob
from setuptools import setup
name = 'lunar_incremental_controller_demo'
setup(name=name, version='0.1.0', packages=[name],
      data_files=[('share/ament_index/resource_index/packages', ['resource/'+name]),
                  ('share/'+name, ['package.xml', 'README.md']),
                  ('share/'+name+'/launch', glob('launch/*.launch.py')),
                  ('share/'+name+'/rviz', glob('rviz/*.rviz'))],
      install_requires=['setuptools'], zip_safe=True,
      maintainer='Kai', maintainer_email='kai@example.com',
      description='Command-driven incremental navigation demonstration', license='Apache-2.0',
      tests_require=['pytest'], entry_points={'console_scripts': [
          'vehicle_sim = '+name+'.vehicle_sim:main', 'run_case = '+name+'.run_case:main']})
