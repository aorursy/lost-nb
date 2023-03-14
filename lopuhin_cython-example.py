#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('load_ext', 'Cython')




get_ipython().run_cell_magic('cython', '', 'cpdef int add(int x, int y):\n    return x + y ')




add(1, 2)

