from setuptools import setup, find_packages

setup(
    name="labrador",
    version="0.2",
    packages=find_packages(where="labrador"),
    package_dir={"": "labrador"},
)
