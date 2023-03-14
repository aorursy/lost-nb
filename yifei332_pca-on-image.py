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

import numpy as np
import pandas as pd
from time import time
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sklearn
import glob, os
%matplotlib inline
import skimage
from skimage.feature import greycomatrix, greycoprops,corner_harris
from skimage.filters import sobel,gaussian
from skimage.color import rgb2gray
from skimage.transform import resize 
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit



drivers = pd.read_csv('../input/driver_imgs_list.csv')
train_files = [f for f in glob.glob("../input/train/*/*.jpg")]
test_files = ["../input/test/" + f for f in os.listdir("../input/test/")]
print(train_files[:10])
print(test_files[:10])

# read training images
img_all = []
img_all_avg = []
img_all_gs = []
data_all = []

img_all_rz = []
img_all_avg_rz = []
img_all_gs_rz = []
data_all_rz = []
target_all = []
subject_all = []
for i in range(0,len(train_files)):
    if (i%5000==0):
        print(str(i) + ' images read')
    path = train_files[i]
    #print path
    im_read = plt.imread(path)
    im_read_avg = im_read[:,:,0]+im_read[:,:,1]+im_read[:,:,2]
    img_gray = rgb2gray(im_read)
    dims = np.shape(img_gray)
    img_data= np.reshape(img_gray, (dims[0] * dims[1], 1))
    
    img_gray_rz = gaussian(resize(img_gray,(84,112)),sigma=1)
    im_read_rz = resize(im_read,(84,112))
    im_read_avg_rz = im_read_rz[:,:,0]+im_read_rz[:,:,1]+im_read_rz[:,:,2]
    dims = np.shape(img_gray_rz)
    img_data_rz= np.reshape(img_gray_rz, (dims[0] * dims[1], 1))
    data_all_rz.append(img_data_rz)
    target_all.append(drivers.loc[i]['classname'])
    subject_all.append(drivers.loc[i]['subject'])


## Converting data to NP-array
data_all_model = np.asarray(data_all_rz)
target_all = np.asarray(target_all)
subject_all =np.asarray(subject_all)
data_all_model = data_all_model[:,:,0]


n_components = 200
print("Extracting the top %d PCs from %d images"
      % (n_components, data_all_model.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(data_all_model)
print("done in %0.3fs" % (time() - t0))

path = train_files[0]
print path
img = plt.imread(path)
img_rz = resize(img,(84,112))
img_rz_gs = rgb2gray(img_rz)
plt.figure(figsize=(15,20))
plt.subplot(2,2,1)
plt.imshow(img)
plt.subplot(2,2,2)
plt.imshow(img_rz_gs,cmap='gray')

print(np.shape(img_rz_gs))
img_rz_gs_v = np.reshape(img_rz_gs,(1,9408))
img_PC = pca.transform(img_rz_gs_v)

img_PC2 = pca.inverse_transform(img_PC)

plt.figure(figsize=(15,20))
plt.subplot(2,2,1)
plt.imshow(resize(img_PC2,(84,112)),cmap='gray')
plt.subplot(2,2,2)
plt.imshow(img_PC_rz,cmap='gray')

np.shape(img_PC_rz)
print img_PC2
print img_rz_gs_v


