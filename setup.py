from distutils.core import setup

from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension('adr_2d.solver',
              sources=['adr_2d/solver.pyx'],
              language='c')
]

setup(
    name='AC274',
    version='0.01',
    packages=['adr_1d', 'adr_2d'],
    url='',
    license='',
    author='',
    author_email='bweinstein@seas.harvard.edu',
    description='',
    ext_modules = cythonize(extensions, annotate=True, reload_support=True)
)
