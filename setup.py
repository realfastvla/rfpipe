from setuptools import setup, find_packages

exec(open('rfpipe/version.py').read())  # set __version__

setup(
    name='rfpipe',
    description='realfast pipeline',
    author='Casey Law',
    author_email='caseyjlaw@gmail.com',
    version=__version__,
    url='http://github.com/realfastvla/rfpipe',
    packages=find_packages(),        # get all python scripts in realtime
    install_requires=['numpy', 'scipy', 'pwkit', 'sdmpy', 'pyfftw',
                      'click', 'dask', 'distributed', 'attrs', 'future',
                      'astropy', 'pyyaml', 'lxml', 'numba', 'rtpipe'],
    package_data={"rfpipe": ["tests/data/*xml", "tests/data/realfast.yml"]},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 2.7'
        ]
)
