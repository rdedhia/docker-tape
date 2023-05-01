from setuptools import find_packages
from setuptools import setup


setup(
    # Package information
    name="docker-tape",
    version="0.0.1",
    maintainer="Rahil Dedhia, Arjun Dharma, Thomas Waldschmidt",
    url="github.com/rdedhia/docker-tape",
    # Package data
    packages=find_packages(),
    include_package_data=True,
    # Insert dependencies list here
    install_requires=[
        "Flask==2.3.2",
        "torch==1.5.0",
        "torchvision==0.6.0",
        "tape_proteins==0.4",
        "plotly==4.5.1",
        "scikit-learn==0.21.2",
        "requests==2.23.0",
        "pandas==0.25.1",
    ],
)
