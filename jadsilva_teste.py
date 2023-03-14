# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy.stats as stats
import hashlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

logs = pd.read_csv('../input/train.csv')
logs.head()

print logs['DEPTH'].head()
print logs['DEPTH'].tail()

min = np.amin(logs['DEPTH'])
max = np.amax(logs['DEPTH'])

print 'min:' min
print 'max:' max

logs.describe()

logs.VP.hist(bins=50, color='black', by=logs.L, figsize=(15,2), layout=(1,3), lw=0)
