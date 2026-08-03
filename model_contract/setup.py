from setuptools import find_packages, setup


setup(
    name="lunar_model_contract",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/lunar_model_contract"]),
        ("share/lunar_model_contract", ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
)
