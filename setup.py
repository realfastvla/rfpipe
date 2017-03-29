from setuptools import setup, find_packages
import numpy, glob

setup(
    name = 'rfpipe',
    description = 'realfast pipeline',
    author = 'Casey Law',
    author_email = 'caseyjlaw@gmail.com',
    version = '0.15',
    url = 'http://github.com/realfastvla/rfpipe',
    packages = find_packages(),        # get all python scripts in realtime
    install_requires=['numpy', 'scipy', 'pwkit', 'sdmpy>=1.35', 'pyfftw', 'click', 'distributed>=1.13', 'attrs', 'future'],
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 2.7'
        ]
)
