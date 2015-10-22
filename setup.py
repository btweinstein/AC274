from distutils.core import setup

from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("adr_1d_solver.solver",
              sources=["adr_1d_solver/solver.pyx"],
              language="c")
]


setup(
    name='AC274',
    version='0.01',
    packages=['adr_1d_solver'],
    url='',
    license='',
    author='',
    author_email='bweinstein@seas.harvard.edu',
    description='',
    ext_modules=cythonize(extensions, annotate=True, reload_support=True)
)
