#!/usr/bin/env python
# coding: utf-8



print("出力したい内容")


print(2**3)
print(10**9+7)

print(5%3)
print(7%3)

print(7//2)

x=3
print(x)
print(x+10,x*20,x**30)

x=x+1
print(x)
x+=1

print(1+2)#3
print("1"+"2")#12

print('あいう'+'えお')

print("あいうえお")

print("Hello,World!")

print(6/2)

print(6/2)

print(int(6/2))

x=10000

if x>1000:
    print('でかい')

elif x>=500:
    print('そこそこ')

else:
    print('ちいさい')




y=10

if x==y:
    print('same')
else:
    print('no')

print(x!=y)

n=12345

lstr=['a','b','c']
lint=[1,1,2,3,5,8]

print(len(lint))

print(lstr[1])

if lint[3]+lint[4]==lint[5]:
    print('fivo')


lint[2]+=100
print(lint)

lstr.append('d')
print(lstr)

st='あいうえお'
lstr2=list(st)
print(lstr2)

a = "abcd"
print(type(a))
b = 1234
print(type(b))

print(lstr2[1:4])

#list[::-1]
print(lstr2[::-1])

lnum=list(range(1,15,2))
print(lnum)

lnum2=list(range(5))
print(lnum2)

for i in range(1,10):
    if i%2==0:
    print(i)

su=0
fact=1
for j in range(1,20):
    su=su+j
    fact=fact*j
print(su,fact)


list=['x','y','z']
s=''
for i in list:
    s+=i
print(s)

for i in range(10):
    print('WOW!')

n=100
while n>0:
    n-=1
    print(n)

def po(a,b):
    return a*b
c=po(a,b)
print(c*3)




def po(a,b):
    print(a**b)
po(a,b)


import os

print(os.getcwd())

x = os.listdir("/kaggle/input")
print(x)
print("データ型",type(x))

os.chdir("/kaggle/input/house-prices-dataset")
print(os.getcwd())


os.mkdir("/kaggle/working/new_dir")
print(os.listdir("/kaggle/working"))

os.chdir("/kaggle/input/house-prices-dataset")
get_ipython().system('ls')

import pandas as pd

house=pd.read_csv("/kaggle/input/house-prices-dataset/train.csv")
print(type(house))#DataFrame
print(len(house))

house.shape

house.head(10)

df_1 = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})

df_1.shape

df_1.head()#DataFrame

s_1 = pd.Series([1, 2, 3, 4, 5])

house.LotArea

house["LotArea"]

house["LotArea"][0]

house.iloc[0]

house.iloc[:, 0]

a = [1,4,6]
house.iloc[a, 0]

house.loc[:, ['MSSubClass', 'Utilities', 'LotShape']]

house.iloc[:, 0:10]

house.loc[:, 0:10]

house.loc[house["LotShape"] == 'Reg']

reviews.loc[(house["LotShape"] == 'Reg') & (house["MiscVal"] >= 10)]

house.YrSold#dtype: int64

house.loc[house.YrSold.isin([2006, 2007])]

house["new_column_1"]="WOW"

house.describe()

house.mean()

house["SaleCondition"].unique()

house["SaleCondition"]..value_counts()

house_SalePrice_mean = house.SalePrice.mean()
house["SalePrice"].map(lambda p: p - house_SalePrice_mean)

def remean_points(row):
    row.SalePrice = row.SalePrice - house_SalePrice_mean
    return row

house.apply(remean_points, axis="SalePrice")

house.head()

house.groupby("SaleCondition").SaleCondition.count()

house.groupby("SaleCondition").count()

df_ = house.groupby("SaleCondition")
df_.head()

df_.apply(lambda df: df.MSZoning.iloc[0])

house=pd.read_csv("/kaggle/input/house-prices-dataset/train.csv")

house.groupby(['SalePrice', 'MoSold']).max()

house.groupby(['SalePrice']).price.agg([len, min, max])

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplitlib', 'inline#notebookで画像などを表示させるための宣言')




import pandas as pd
train = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
train["image_id"] = train["image_id"]+".jpg"
train.head()




file_paths = train["image_id"].values[:]




import glob
x = glob.glob("/kaggle/input/plant-pathology-2020-fgvc7/images/*")


import cv2
import os
import matplotlib.pyplot as plt
path = os.path.join("/kaggle/input/plant-pathology-2020-fgvc7/images",file_paths[0])
img = cv2.imread(path)
print(type(img))
print(img.shape)
plt.imshow(img)
plt.show()

flip0=cv2.flip(img,0)
plt.imshow(flip0)
plt.show()

flip1=cv2.flip(img,1)
plt.imshow(flip1)
plt.show()

flip2=cv2.flip(img,-1)
plt.imshow(flip2)
plt.show()

#resize
print(img.shape)
resize =cv2.resize(img,(224,224))
print(resize.shape)
plt.imshow(resize)
plt.show()

import cv2
height=img.shape[0]
width=img.shape[1]
center=(width//2,height//2)

affin_trans=cv2.getRotationMatrix2D(center,33.0,1.0)
rotate=cv2.warpAffine(img,affin_trans,(width,height))
plt.imshow(rotate)
plt.show()



clip=img[40:height-20,40:width-20]
print(clip.shape)
plt.imshow(clip)
plt.show()

import numpy as np
import pandas as pd
import cv2
import sys
import os
import matplotlib.pyplot as plt
from itertools import product
import time
from tqdm.notebook import tqdm
import glob

get_ipython().run_line_magic('matplotlib', 'inline')

def crop(image):
    # 
    img = cv2.imread(image)
    
    # 
    h, w = img.shape[:2]
    h1, h2 = int(h * 0.05), int(h * 0.95)
    w1, w2 = int(w * 0.05), int(w * 0.95)
    img = img[h1: h2, w1: w2]
    bgr = img.copy()
    
    # 
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
​
    # 
    img2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('img2', img2)
​
    # 
    contours = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    # 
    x1 = [] 
    y1 = [] 
    x2 = []
    y2 = [] 
    for i in range(1, len(contours)):# i = 1 
        ret = cv2.boundingRect(contours[i])
        x1.append(ret[0])
        y1.append(ret[1])
        x2.append(ret[0] + ret[2])
        y2.append(ret[1] + ret[3])


    x1_min = min(x1)
    y1_min = min(y1)
    x2_max = max(x2)
    y2_max = max(y2)
    croparea = cv2.rectangle(bgr, (x1_min, y1_min), (x2_max, y2_max), (255, 0, 0), 3)

train = train.sample(n=100, random_state=2020)




train.head()


labels = train.iloc[:,1:]
labels.head()

from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from keras import backend as K
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import  ResNet50
from keras.applications.densenet import  DenseNet121
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPool2D,Multiply,GlobalAveragePooling2D, Input,Activation, Flatten, BatchNormalization,Dropout,Concatenate,GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from keras.optimizers import Adam,SGD,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np
num_folds=3
kf = KFold(n_splits=num_folds, shuffle=True)
import sklearn.metrics as metric
from keras.utils import np_utils
import cv2


vgg16_model=VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None)




get_ipython().system('ls /kaggle/input')




get_ipython().system('pwd')




from tqdm import tqdm
images_paths = train["image_id"].values[:]
def get_images_from_df(df):
    images = np.empty((len(df),224,224,3))
    images_paths = df["image_id"].values[:]
    #print(images_paths[0])
    for i in tqdm(range(len(images_paths))):
        img=cv2.imread("/kaggle/input/plant-pathology-2020-fgvc7/images/{}".format(images_paths[i]))
        #print(img.shape)
        img = cv2.resize(img,(224,224))
        images[i, :, :, :] = img
    return images
def get_one_hot_form_df(df):
    df = df[["healthy","multiple_diseases","rust","scab"]]
    label_onehot = df.values
    return label_onehot 




train.head()




y = get_one_hot_form_df(train)


def get_model_finetune(base_model,input_shape=[None,None,3], num_classes=4):
    base = base_model
    for layer in base_model.layers:
        layer.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x) 
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    prediction = Dense(num_classes, activation='sigmoid')(x)

    model = Model(input=base_model.input, output=prediction)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=1e-4),
        metrics=['accuracy']
    )
    return model




test = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
test["image_id"] = test["image_id"]+".jpg"
test.head()
test_x = get_images_from_df(test)

#start_weight=model_test.get_weights()
cnt=0
preds =[]
for train_index, eval_index in kf.split(train):
    tra_df, val_df = train.iloc[train_index], train.iloc[eval_index]
    tra_x = get_images_from_df(tra_df)
    tra_y = get_one_hot_form_df(tra_df)
    val_x = get_images_from_df(val_df)
    val_y = get_one_hot_form_df(val_df)
    cnt+=1
    print("---------epoch{}--------".format(cnt))
    model = get_model_finetune(vgg16_model)
    model.fit(tra_x,tra_y)
    val_pred = model.predict(val_x)
    
    pred =model.predict(test_x)
    
    preds.append(pred)

sub = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")
sub.head()




df1 =pd.DataFrame(np.array(pred))
sub[["healthy","multiple_diseases","rust","scab"]]=df1




sub.head()




df1 =pd.DataFrame(np.array(pred))
df1.head()




import os
os.listdir('/kaggle/input/pytorch-efnet-ns')
import geffnet




import torch
import sys
sys.path.insert(0, '/kaggle/input/pytorch-efnet-ns/')
import geffnet
model = geffnet.tf_efficientnet_b2_ns(pretrained=False)
PATH = "/kaggle/input/pytorch-efnet-ns-weights-abebe/tf_efficientnet_b2_aa-60c94f97.pth"
model.load_state_dict(torch.load(PATH))
print(model)

