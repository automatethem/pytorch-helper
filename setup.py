from setuptools import *

LONG_DESC = """
This is pytorch utilities
"""

setup(name='pytorch_utils',
	  version='0.0.1',
	  description='Pytorch utilities',
	  long_description=LONG_DESC,
	  author='Sang Ki Kwon',
	  url='https://github.com/automatethem/pytorch_utils',
	  install_requires=['torchsummaryX'],
	  author_email='automatethem@gmail.com',
	  license='MIT',
	  packages=find_packages(),
	  zip_safe=False)
