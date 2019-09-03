from distutils.core import setup

from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='sqladservice',
    version='1.0.0a0',
    description='SQL-Server Anomaly Detection Service',
    author='Data Analytics at Texas A&M (DATA) Lab, Yuening Li',
    author_email='yuehningli@gmail.com',
    download_url='https://github.com/yli96/PyOutlierDetectionSys/archive/master.zip',
    keywords=['AnomalyDetection', 'SQLServer'],
    install_requires=[
        'tensorflow>=2.0.0b1',
        'taos',
        'scikit-learn',
        'numpy',
        'luminol',
        'seaborn',
        'torch',
        'tqdm',
        'pandas',
        'matplotlib',
        'cipy',
    ],
    packages=find_packages(),
    python_requires='>=3.6',
)
