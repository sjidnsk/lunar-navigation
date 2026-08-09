from setuptools import find_packages, setup


PACKAGE_NAME = "lunar_external_adapter"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{PACKAGE_NAME}"],
        ),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="Lunar Navigation Maintainers",
    maintainer_email="maintainers@example.invalid",
    description=(
        "Fixed-registry adapter from replaceable external ROS interfaces "
        "to lunar navigation inputs."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "lunar_external_adapter = lunar_external_adapter.node:main",
        ],
    },
)
