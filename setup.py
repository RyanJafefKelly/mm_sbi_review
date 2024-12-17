from io import open

from setuptools import setup  # type: ignore

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# Note: removed all identity information from this file

setup(name='mm_sbi_review',
      version='0.0.1',
      description='Experiments for MM-SBI review',
      license='GPL',
      packages=['mm_sbi_review'],
      zip_safe=False,
      python_requires='>=3.7',
      install_requires=requirements
      )
