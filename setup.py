from setuptools import find_packages
from setuptools import setup


setup(
    # Package information
    name='docker-tape',
    version='0.0.1',
    maintainer='Rahil Dedhia, Arjun Dharma, Thomas Waldschmidt',
    url='github.com/rdedhia/docker-tape',

    # Package data
    packages=find_packages(),
    include_package_data=True,

    # Insert dependencies list here
    install_requires=[
        'Flask==1.0.2',
        'torch',
        'torchvision',
        'tape_proteins'
    ],
)
