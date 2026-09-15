from setuptools import find_packages, setup


setup(
    name="lunar_drl_exploration",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/lunar_drl_exploration"]),
        ("share/lunar_drl_exploration", ["package.xml"]),
    ],
    install_requires=["numpy", "PyYAML"],
)
