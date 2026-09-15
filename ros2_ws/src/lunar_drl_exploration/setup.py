from pathlib import Path
import os
from setuptools import Extension, find_packages, setup

package_root = Path(__file__).resolve().parent
core_include = package_root.parent / "lunar_incremental_navigation_core" / "include"
prefixes = [
    Path(p) for p in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if p
]
core_libs = [
    str(p / "lib")
    for p in prefixes
    if (p / "lib/liblunar_incremental_navigation_core.so").exists()
]


setup(
    name="lunar_drl_exploration",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/lunar_drl_exploration"],
        ),
        ("share/lunar_drl_exploration", ["package.xml"]),
    ],
    ext_modules=[
        Extension(
            "lunar_drl_terrain_native",
            ["native/terrain.cpp", "native/visibility.cpp"],
            include_dirs=[str(core_include)] + [str(p / "include") for p in prefixes],
            depends=["native/grid_buffer.hpp"],
            library_dirs=core_libs,
            libraries=["lunar_incremental_navigation_core"],
            runtime_library_dirs=core_libs,
            language="c++",
            extra_compile_args=["-std=c++20", "-O3", "-Wall", "-Wextra"],
        )
    ],
    entry_points={"console_scripts": ["lunar-drl = lunar_drl_exploration.cli:main"]},
    tests_require=["pytest"],
    install_requires=["numpy", "PyYAML", "scipy"],
)
