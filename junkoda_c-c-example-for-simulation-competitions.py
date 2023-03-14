#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('writefile', 'setup.py', "\nfrom distutils.core import setup, Extension\n\nsetup(name='myspam',\n      version='0.0.1',\n      description='Trivial python module written in C++',\n      ext_modules=[\n          Extension('myspam', ['spam.cpp'],\n                    extra_compile_args = ['-std=c++11'])\n      ]\n)")


# In[2]:


get_ipython().run_cell_magic('writefile', 'spam.cpp', '\n#include <cstdlib>\n#include "Python.h"\n\n// C++ funcutions here\nPyObject* myrand(PyObject *self, PyObject *args)\n{\n  return Py_BuildValue("i", rand() % 4);\n}\n\n// List the functions accessible from Python\nstatic PyMethodDef methods[] = {\n  {"myrand", myrand, METH_VARARGS, "return random integer 0-3"},\n  {NULL, NULL, 0, NULL}\n};\n\nstatic struct PyModuleDef module = {\n  PyModuleDef_HEAD_INIT,\n  "myspam",               // name of this module\n  "C++ module example",   // Doc String\n  -1,\n  methods\n};\n\nPyMODINIT_FUNC\nPyInit_myspam(void) {  \n  return PyModule_Create(&module);\n}')


# In[3]:


get_ipython().system('python3 setup.py build_ext --inplace')


# In[4]:


get_ipython().system(' ls')


# In[5]:


# You can use the module in the notebook like this,
# but submission.py probably cannot access this file
import myspam
myspam.myrand()


# In[6]:


import base64

with open('myspam.cpython-37m-x86_64-linux-gnu.so', 'rb') as f:
    encoded_string = base64.b64encode(f.read())

with open('submission.py', 'w') as f:
    f.write(f'module_str={encoded_string}')


# In[7]:


get_ipython().run_cell_magic('writefile', '-a submission.py', "\nimport base64\nimport kaggle_environments.envs.halite.helpers as hh\nwith open('myspam.cpython-37m-x86_64-linux-gnu.so', 'wb') as f:\n    f.write(base64.b64decode(module_str))\nimport myspam\n\nactions = [hh.ShipAction.NORTH, hh.ShipAction.EAST,\n           hh.ShipAction.SOUTH, hh.ShipAction.WEST]\n\n# Trivial Halite agent randomly walking\ndef agent(obs, config):\n    board = hh.Board(obs, config)\n    me = board.current_player\n\n    for ship in me.ships:\n        i = myspam.myrand()  # Use the C++ code\n        ship.next_action = actions[i]\n    \n    return me.next_actions")

