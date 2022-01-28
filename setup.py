from setuptools import setup, find_packages

setup(
    name="quality-demo-support",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25",
        "urllib3>=1.24",
        "PILLOW>=9.0.0",
        "numpy>=1.21.0",
        "cord-client-python @ git+https://github.com/cord-team/cord-client-python.git@rp/floaty-tqdm",
        "cord-pytorch-dataset @ git+https://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@github.com/cord-team/cord-pytorch-dataset.git",
        "tqdm",
        "matplotlib>=3.5.1",
        "pandas>=1.1",
        "torchvision>=0.11",
    ],
    python_requires=">=3.7",
)
