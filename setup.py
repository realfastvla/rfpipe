from setuptools import setup, find_packages
import numpy, glob

exec(open('rfpipe/version.py').read())  # set __version__

setup(
    name = 'rfpipe',
    description = 'realfast pipeline',
    author = 'Casey Law',
    author_email = 'caseyjlaw@gmail.com',
    version = __version__,
    url = 'http://github.com/realfastvla/rfpipe',
    packages = find_packages(),        # get all python scripts in realtime
    install_requires=['numpy', 'scipy', 'pwkit', 'sdmpy>=1.35', 'pyfftw', 'click', 'dask',
                      'distributed>=1.13', 'attrs', 'future', 'astropy', 'pyyaml', 'lxml'],
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 2.7'
        ]
)
