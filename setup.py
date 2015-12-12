from setuptools import setup, find_packages
import sys

try:
    from Cython.Build import cythonize
except ImportError as e:
    print("No version of Cython installed: please install Cython before continuing with the installation.")
    sys.exit(1)

setup(
    name='pydsm',
    version='0.1',
    packages=find_packages(),
    package_dir={'pydsm': 'pydsm'},
    package_data={'pydsm.resources': ['*.pickle']},
    url='http://github.com/jimmycallin/pydsm',
    license='MIT',
    author='Jimmy Callin',
    author_email='jimmy.callin@gmail.com',
    description='A framework for building and exploring distributional semantic models.',
    test_suite='pydsm.tests',
    install_requires=['tabulate'],
    ext_modules=cythonize(["pydsm/cmodel.pyx", "pydsm/cindexmatrix.pyx"])
)
