from setuptools import find_packages, setup


PACKAGE_NAME = "lunar_exploration_policy"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (f"share/{PACKAGE_NAME}/launch", ["launch/interface_v1.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Lunar Navigation Maintainers",
    maintainer_email="maintainers@example.invalid",
    description="Frozen fed9 exploration policy to PlanMotion ROS 2 closed loop.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "interface_v1_policy = lunar_exploration_policy.node:main",
        ],
    },
)
