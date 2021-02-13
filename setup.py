from setuptools import setup, find_packages

setup(
    name='SoCG21',
    version='0.0.1',
    packages=find_packages(),
    python_requires='>=3.6, <3.9',
    install_requires=[
        'numpy',
        'torch',
        'gym',
        'cgshop2021_pyutils',
        'pyyaml',
    ]
)
