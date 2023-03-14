# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print("train min %d" % train.time.min())
print("train max %d" % train.time.max())
print("train %4.4f years" % ((train.time.max() - train.time.min()) / (60 * 24 * 365)))

print("test min %d" % test.time.min())
print("test max %d" % test.time.max())
print("test %4.4f years" % ((test.time.max() - test.time.min()) / (60 * 24 * 365)))

print("train x min %d" % train.x.min())
print("train x max %d" % train.x.max())
print("train y min %d" % train.y.min())
print("train y max %d" % train.y.max())

print("test x min %d" % test.x.min())
print("test x max %d" % test.x.max())
print("test y min %d" % test.y.min())
print("test y max %d" % test.y.max())

import math
def int_xy(x, range_x):
    ix = math.floor(range_x * x / 10)
    if ix < 0:
        ix = 0
    if ix >= range_x:
        ix = range_x - 1
    return ix

x_range = 1000
y_range = 2000
train["ix"] = train["x"].apply(lambda x: int_xy(x, x_range))
train["iy"] = train["y"].apply(lambda x: int_xy(x, y_range))
uni_mat = np.empty((x_range, y_range))
train.groupby(["ix", "iy"])["place_id"].nunique()

mat = train.groupby(["ix", "iy"])["place_id"].nunique()

np.reshape(mat.values, (x_range, y_range))

mat.values.shape

place_mat = [[mat["ix] for j in range(y_range)] for i in range(x_range)]


