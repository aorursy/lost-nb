#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'Cython')


# In[2]:


get_ipython().run_cell_magic('cython', '', 'cpdef int add(int x, int y):\n    return x + y ')


# In[3]:


add(1, 2)

