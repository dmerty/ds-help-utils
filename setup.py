from setuptools import find_packages, setup

setup(
    name="ds-help-utils",
    version="0.0.2",
    description="A package to help do DS tasks more easily",
    author="Dmitriy Emelianov",
    author_email="asdqwer92@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy==1.23.1",
        "scikit-learn==1.1.1"
    ]
)