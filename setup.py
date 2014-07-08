from setuptools import setup, find_packages

setup(
    name='pydsm',
    version='0.1',
    packages=find_packages(),
    package_dir={'pydsm': 'pydsm'},
    url='http://github.com/jimmycallin/pydsm',
    license='GPLv2',
    author='Jimmy Callin',
    author_email='jimmy.callin@gmail.com',
    description='A framework for building and exploring distributional semantic models.',
    test_suite='pydsm.tests'
)
