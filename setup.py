#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'rfpipe'
DESCRIPTION = 'Radio interferometric transient search pipeline for realfast'
URL = 'http://github.com/realfastvla/rfpipe'
EMAIL = 'claw@astro.berkeley.edu'
AUTHOR = 'Casey Law'

# What packages are required for this module to be executed?
REQUIRED = ['numpy', 'scipy', 'pwkit', 'sdmpy', 'pyfftw', 'bokeh',
            'attrs', 'future', 'astropy<3.0', 'pyyaml', 'numba'],
            # optional 'rfgpu', 'distributed'

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.rst' is present in your MANIFEST.in file!
#with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
#    long_description = '\n' + f.read()
long_description = """The realfast project will perform real-time commensal
fast radio transient searches at the Very Large Array. rfpipe is a Python
implementation of the transient search pipeline that operations on the 
realfast cluster. More project info at http://realfast.io.
"""

# Load the package's __version__.py module as a dictionary.
with open(os.path.join(here, NAME, 'version.py')) as f:
    exec(f.read())


class PublishCommand(Command):
    """Support setup.py publish."""

    description = 'Build and publish the package.'
    user_options = []
    
    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous buildsâ¦')
            rmtree(os.path.join(here, 'dist'))
        except FileNotFoundError:
            pass

        self.status('Building Source and Wheel (universal) distributionâ¦')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPi via Twineâ¦')
        os.system('twine upload dist/*')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=__version__,
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    package_data={NAME: ["tests/data/*xml", "tests/data/realfast.yml"]},
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    include_package_data=True,
    license='BSD',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
    # $ setup.py publish support. 
    cmdclass={
        'publish': PublishCommand,
    },
)
