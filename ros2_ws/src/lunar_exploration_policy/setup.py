from glob import glob
from pathlib import Path

from setuptools import find_namespace_packages, setup


PACKAGE_NAME = "lunar_exploration_policy"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TRAINING_ROOT = REPOSITORY_ROOT / "training/lunar_policy_training"
TRAINING_PACKAGES = find_namespace_packages(
    where=str(TRAINING_ROOT),
    include=("lunar_policy_training", "lunar_policy_training.*"),
)
# ament_python 要求 data_files 的源路径相对 setup.py 所在包目录。
CAPABILITY_CONFIG = Path("../lunar_navigation_config/config")
CAPABILITY_SHARE = (
    f"share/{PACKAGE_NAME}/runtime_repository/ros2_ws/src/"
    "lunar_navigation_config/config"
)


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    # 从 fed9 冻结源直接安装同一份观测/候选实现，避免部署时复制第二套公式。
    packages=["lunar_exploration_policy", *TRAINING_PACKAGES],
    package_dir={
        "lunar_exploration_policy": "lunar_exploration_policy",
        "lunar_policy_training": str(TRAINING_ROOT / "lunar_policy_training"),
    },
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
        (
            CAPABILITY_SHARE,
            [
                str(CAPABILITY_CONFIG / "platform_capability_schema_v2.yaml"),
                str(CAPABILITY_CONFIG / "three_platform_capability_freeze_v1.yaml"),
            ],
        ),
    ],
    install_requires=["setuptools", "numpy", "PyYAML", "rasterio", "shapely"],
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
