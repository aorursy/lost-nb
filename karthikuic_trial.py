# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bokeh.plotting import figure, show, output_notebook
import matplotlib.mlab as mlab

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_dir = "../input"
train_file = "train.csv"
fbcheckin_train_tbl = pd.read_csv(os.path.join(train_dir, train_file))
fb_train_subset = fbcheckin_train_tbl[:1000]
#colors = np.random.rand(len(fb_train_subset))
#area = np.pi * (fb_train_subset['accuracy'])**2

#plt.scatter(fb_train_subset['x'], fb_train_subset['y'], s=area, c=colors, alpha=0.5)
#plt.show


print(fb_train_subset['time
'])


