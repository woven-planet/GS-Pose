# MIT License
# Copyright (c) 2017 Fei Xia
# Permission is granted to use, copy, modify, merge, publish, and distribute this software.
# The software is provided "as is", without warranty of any kind.
# For more details, see the full license https://opensource.org/license/MIT.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CppExtension,CUDAExtension

setup(name='my_lib_cuda',
      ext_modules=[CUDAExtension('my_lib_cuda',['src/my_lib_cuda.cpp', 'src/nnd_cuda.cu']
              )],
      cmdclass={'build_ext': BuildExtension}
      )

