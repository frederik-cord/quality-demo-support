from setuptools import setup, find_packages

setup(
    name="quality-demo-support",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25",
        "urllib3>=1.24",
        "PILLOW>=9.0.0",
        "numpy>=1.21.0",
        "encord-dataset @ git+https://github.com/encord-team/encord-dataset.git",
        "tqdm",
        "matplotlib>=3.5.1",
        "pandas>=1.1",
        "torchvision>=0.11",
    ],
    python_requires=">=3.7",
)
