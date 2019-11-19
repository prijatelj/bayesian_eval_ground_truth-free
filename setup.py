from setuptools import setup, find_packages
from setuptools.command.install import install

long_description = ''
with open('README.md', 'r') as f:
    long_description = f.read()

install_requires = ''
with open('requirements.txt', 'r') as f:
    install_requires = f.read()

setup(
    name='psych_metric',
    version='0.1.0',
    author='Derek Prijatelj',
    author_email='dprijate@nd.edu',
    packages=[
        'psych_metric',
        'psych_metric.datasets',
        'psych_metric.datasets.crowd_layer',
        'psych_metric.datasets.facial_beauty',
    ],
    #scripts
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    install_requires=install_requires,
    python_requires='>=3.6',
)
