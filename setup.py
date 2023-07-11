from setuptools import setup, find_packages
from os import path

# Get the long description from the README file
with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='loss_landscapes',
    version='0.0.1',
    packages=find_packages(exclude='tests'),
    url='https://github.com/geshijoker/loss-landscapes-segmentation',
    license='MIT',
    author='Ge Shi',
    author_email='geshi@lbl.gov',
    description='A library for approximating loss landscapes in low-dimensional parameter subspaces',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
    install_requires=['numpy'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
