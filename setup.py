#!/usrbin/env python
from setuptools import setup

description = "BB MASTER - The SO BB MASTER pipeline"

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md') as f:
    long_description = f.read()

setup(name="soopercool",
      version="0.0.0",
      description=description,
      long_description=long_description,
      long_description_content_type='text/markdown',
      url="https://github.com/simonsobs/SOOPERCOOL",
      author="David Alonso",
      author_email="david.alonso@physics.ox.ac.uk",
      install_requires=requirements,
      packages=['soopercool'],
)
