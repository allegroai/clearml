"""
TRAINS - Artificial Intelligence Version Control
https://github.com/allegroai/trains
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from six import exec_
from pathlib2 import Path


here = Path(__file__).resolve().parent

# Get the long description from the README file
long_description = (here / 'README.md').read_text()


def read_version_string():
    result = {}
    exec_((here / 'trains/version.py').read_text(), result)
    return result['__version__']


version = read_version_string()

requirements = (here / 'requirements.txt').read_text().splitlines()

setup(
    name='trains',
    version=version,
    description='TRAINS - Auto-Magical Experiment Manager & Version Control for AI',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # The project's main homepage.
    url='https://github.com/allegroai/trains',
    author='Allegroai',
    author_email='trains@allegro.ai',
    license='Apache License 2.0',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Version Control',
        'Topic :: System :: Logging',
        'Topic :: System :: Monitoring',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: Apache Software License',
    ],
    keywords='trains development machine deep learning version control machine-learning machinelearning '
             'deeplearning deep-learning experiment-manager experimentmanager',
    packages=find_packages(exclude=['contrib', 'docs', 'data', 'examples', 'tests']),
    install_requires=requirements,
    package_data={
        'trains': ['config/default/*.conf', 'backend_api/config/default/*.conf']
    },
    include_package_data=True,
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
          'trains-init = trains.config.default.__main__:main',
        ],
    },
)
