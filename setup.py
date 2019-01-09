import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyFEM",
    version="1.0.0",
    author="Cristian Danilo Ramirez Vargas  ",
    author_email="rvcristiand@unal.edu.co",
    description="Finite element method with python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rvcristiand/pyFEM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)