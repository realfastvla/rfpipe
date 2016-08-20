from setuptools import setup, find_packages
import numpy, glob

setup(
    name = 'rfpipe',
    description = 'realfast pipeline',
    author = 'Casey Law',
    author_email = 'caseyjlaw@gmail.com',
    version = '0.1',
    url = 'http://github.com/realfastvla/rfpipe',
    packages = find_packages(),        # get all python scripts in realtime
    install_requires=['numpy', 'scipy', 'pwkit', 'sdmpy>=1.35', 'pyfftw', 'click'],
    zip_safe=False
)
