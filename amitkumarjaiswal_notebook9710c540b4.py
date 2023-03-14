#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

#Read the Image from the train folder
data = "../Data/"

testfile=open(data+"test.txt")

for x in testfile.readlines():
	img = cv2.imread(data+x[:-1],0)

	#Fixed Thresholding with 'threshold value' as 170 (Ideal range is from 165-175)
	ret,thresh1 = cv2.threshold(img,170,255,cv2.THRESH_BINARY)

	cleaned_file = "thresh/"+str(x.split('.')[0].split('/')[1])+"_cleaned.jpg"
	print cleaned_file
	#print cv2.imwrite(cleaned_file,thresh1);


# In[3]:


import numpy as np
import cv2
from matplotlib import pyplot as plt

data = "../Data/"
#data1 = "../Data/train_cleaned/"
testfile=open(data+"test.txt")
for x in testfile.readlines():
	img = cv2.imread(data+x[:-1],0)
	#img1 = cv2.imread(data1+str(x)+".png",0)
	#if(img==None or img1==None):
	#	continue	
	print data+x
	img_c=img.copy()
	hist = cv2.calcHist([img],[0],None,[256],[0,256])
	#hist1 = cv2.calcHist([img1],[0],None,[256],[0,256])
	#L=[]	
	#for i in xrange(255):
	#	L.append([int(hist1[i]),i])
	#L.sort()
	#for x in L[::-1]:
	#	print x
	
	for i in xrange(img_c.shape[0]):
		for j in xrange(img_c.shape[1]):
			if(img_c[i][j]>=50 and img_c[i][j]<=150):
				continue
			else:
				img_c[i][j]=255
	cleaned_file = "prelim1/"+str(x.split('.')[0].split('/')[1])+"_cleaned.jpg"
	print cleaned_file
	#hist2 = cv2.calcHist([img_c],[0],None,[256],[0,256])
	print cv2.imwrite(cleaned_file,img_c);
	'''	
	plt.subplot(321),plt.imshow(img,'gray'),plt.xticks([]),plt.yticks([])
	plt.subplot(322),plt.plot(hist),plt.xlim([-10,256]),plt.yticks([])
	plt.subplot(323),plt.imshow(img1,'gray'),plt.xticks([]),plt.yticks([])
	plt.subplot(324),plt.plot(hist1),plt.xlim([-10,256]),plt.yticks([])
	plt.subplot(325),plt.imshow(img_c,'gray'),plt.xticks([]),plt.yticks([])
	plt.subplot(326),plt.plot(hist2),plt.xlim([-10,256]),plt.yticks([])
	plt.tight_layout()	
	plt.savefig(str(x)+"_cleaned.jpg")
	plt.close()
	'''


# In[4]:




