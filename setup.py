# -*- coding: utf-8 -*-
import os
import re
from setuptools import setup, find_packages


def get_version(*file_paths):
    """Retrieves the version from prompt_responses/__init__.py"""
    filename = os.path.join(os.path.dirname(__file__), *file_paths)
    version_file = open(filename).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')


version = get_version("cges", "__init__.py")
readme = open('README.md').read()

setup(
    name='CGES',
    version=version,
    description="""Install Version of 'Combined Group and Exclusive Sparsity for Deep Networks'""",
    long_description=readme,
    author='Jaehong Yoon (Original Author)',
    url='https://github.com/dongkwan-kim/cges',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
    ],
    zip_safe=False,
    keywords='ICML',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)