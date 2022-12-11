import os

from setuptools import find_packages, setup

__version__ = os.getenv('tinynn', '0.1.0')


def setup_tinynn():
    requires = [
        'numpy'
    ]
    setup(
        name='tinynn',
        version=__version__,
        description='my deep learning study',
        python_requires='>=3.8',
        install_requires=requires,
        packages=find_packages(
            include=['tinynn', 'tinynn.*']),
    )


if __name__ == '__main__':
    setup_tinynn()
