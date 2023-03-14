#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
cities = pd.read_csv("../input/cities.csv", nrows=5000, index_col=['CityId'])
coords = cities.values
# dummy tour: 0, 1, 2, 3...
tour = np.array([i for i in range(5000)])
print("There are", len(coords), "cities in coords")


# In[ ]:


def two_opt_python():
    min_change = 0
    num_cities = len(tour)
    # Find the best move
    for i in range(num_cities - 2):
        for j in range(i + 2, num_cities - 1):
            change = dist(i, j) + dist(i+1, j+1) - dist(i, i+1) - dist(j, j+1)
            if change < min_change:
                min_change = change
                min_i, min_j = i, j
    # Update tour with best move
    if min_change < 0:
        tour[min_i+1:min_j+1] = tour[min_i+1:min_j+1][::-1]        

def dist(a, b):
    """Return the euclidean distance between cities tour[a] and tour[b]."""
    return np.hypot(coords[tour[a], 0] - coords[tour[b], 0],
                    coords[tour[a], 1] - coords[tour[b], 1])


# In[ ]:


get_ipython().run_line_magic('time', 'two_opt_python()')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'Cython')


# In[ ]:


get_ipython().run_cell_magic('cython', '', 'import numpy as np\ncimport numpy as np\ncimport cython\nfrom libc.math cimport sqrt\n\ncpdef two_opt_cython(double[:,:] coords, int[:] tour_):\n    cdef float min_change, change\n    cdef int i, j, min_i, min_j, num_cities\n    num_cities = len(tour_)\n    min_change = 0\n    # Find the best move\n    for i in range(num_cities - 2):\n        for j in range(i + 2, num_cities - 1):\n            change = dist(i, j, tour_, coords) + dist(i+1, j+1, tour_, coords)\n            change = - dist(i, i+1, tour_, coords) - dist(j, j+1, tour_, coords)\n            if change < min_change:\n                min_change = change\n                min_i, min_j = i, j\n    # Update tour with best move\n    if min_change < 0:\n        tour_[min_i+1:min_j+1] = tour_[min_i+1:min_j+1][::-1]\n    return np.asarray(tour_)  # memoryview to numpy array\n\ncdef float dist(int a, int b, int[:] tour_view, double[:,:] coords_view):\n    """Return the euclidean distance between cities tour[a] and tour[b]."""\n    return sqrt((coords_view[tour_view[a], 0] - coords_view[tour_view[b], 0])**2 +\n                (coords_view[tour_view[a], 1] - coords_view[tour_view[b], 1])**2)')


# In[ ]:


get_ipython().run_line_magic('time', "two_opt_cython(coords, tour.astype('int32'))")

