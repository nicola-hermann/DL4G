from setuptools import setup, find_packages


setup(
    name="dl4g",
    version="0.1.0",
    description="Jassbot for the module DL4G HS24 at HSLU",
    author="Nicola Hermann",
    author_email="nicola.hermann@stud.hslu.ch",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)
