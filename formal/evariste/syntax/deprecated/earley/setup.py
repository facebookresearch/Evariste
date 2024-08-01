# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from distutils.core import setup, Extension
from Cython.Build import cythonize

parser_module = Extension(
    "earley_parser",
    sources=["earley_parser.pyx"],
    extra_compile_args=["-std=c++14", "-O3"],
    language="c++",
)

setup(
    name="Earley parser", ext_modules=cythonize(parser_module), zip_safe=False,
)
