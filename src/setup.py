#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "tools",
    ext_modules = cythonize("/content/drive/My Drive/Colab Notebooks/HGDTI/src/tools.py")
)
