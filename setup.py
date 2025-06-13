from setuptools import setup, Extension
import pybind11

cpp_args = ['-std=c++11']

sfc_module = Extension(
    'pathery_pathfinding',
    ['pathery_pathfinding.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=cpp_args,
    )

setup(
    name='pathery_pathfinding',
    version='1.0',
    description='A C++ extension for Pathery pathfinding',
    ext_modules=[sfc_module],
)
