from setuptools import setup, find_packages, Extension
import numpy, glob
from Cython.Build import cythonize

extensions = [
    Extension("*", ["cython/*.pyx"],
        include_dirs = ["/home/cbe-master/realfast/anaconda/include", "/home/cbe-master/realfast/anaconda/include/python2.7"],
        libraries = ["vysmaw", "vys", "python2.7"],
        library_dirs = ["/home/cbe-master/realfast/anaconda/lib/python2.7/site-packages"],)
]

setup(
    name = 'rfpipe',
    description = 'realfast pipeline',
    author = 'Casey Law',
    author_email = 'caseyjlaw@gmail.com',
    version = '0.15',
    url = 'http://github.com/realfastvla/rfpipe',
    packages = find_packages(),        # get all python scripts in realtime
    install_requires=['numpy', 'scipy', 'pwkit', 'sdmpy>=1.35', 'pyfftw', 'click', 'distributed>=1.13', 'attrs', 'future'],
    ext_modules = cythonize(extensions),
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 2.7'
        ]
)
