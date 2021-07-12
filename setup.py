from setuptools import setup, Extension
import setuptools
import os, sys

# get __version__, __author__, and __email__
exec(open("./smallworld/metadata.py").read())

setup(
    name = 'smallworld',
    version = __version__,
    author = __author__,
    author_email = __email__,
    url = 'https://github.com/benmaier/pysmallworld',
    license = __license__,
    description = "Generate modified small-world networks and compare with theoretical predictions.",
    long_description = '',
    packages = setuptools.find_packages(),
    setup_requires = [
            ],
    install_requires = [
                'networkx>=2.4',
                'numpy>=1.14',
                'scipy>=1.1',
            ],
    include_package_data = True,
    zip_safe = False,
)
