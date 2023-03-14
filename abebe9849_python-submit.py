#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#足し算
3+2


# In[ ]:





# In[ ]:


#引き算
3-2


# In[ ]:


#掛け算
3*2


# In[ ]:


#割り算
3/2


# In[ ]:


print("出力したい内容")#とします


# In[ ]:


#指数
print(2**3)
print(10**9+7)

#nで割ったあまり
print(5%3)
print(7%3)

#切り捨て割り算
print(7//2)


# In[ ]:


#変数を使うと便利

#xに3を代入
x=3
print(x)
print(x+10,x*20,x**30)

#xの値を更新（xにx+1を再代入）
x=x+1
print(x)
#これは省略して以下のようにかける
x+=1


# In[ ]:


'''
重要！！！！！！！！！！！
"1"　と　1　は異なる

上は数字の１と２であるが、
下は「１」「２」という文字である
'''
print(1+2)#3
print("1"+"2")#12
#このちがいがわかりましたでしょうか？


# In[ ]:


#↑のようにpythonではstring同士の足し算が可能
print('あいう'+'えお')

#しかし文字を''で囲まずにprintするとエラー
print(あいうえお)


# In[ ]:


#「Hello,World!」という文字列を出力してみよう
print("Hello,World!")


# In[ ]:


#tips:割り算に注意

#割り算の答えは勝手にfloatになる
print(6/2)

#防止するには
print(6//2)

#あるいは
print(int(6/2))


# In[ ]:


x=10000

if x>1000:
    print('でかい')

elif x>=500:
    print('そこそこ')

else:
    print('ちいさい')


# In[ ]:


y=10

if x==y:
    print('same')
else:
    print('no')

print(x!=y)

#「同じかどうか」の判定には==を使う、否定の場合は!=


# In[ ]:


#nが偶数なら２で割って、奇数ならそのまま出力するコードを書こう
n=12345


# In[ ]:


lstr=['a','b','c']
lint=[1,1,2,3,5,8]


# In[ ]:


#長さの取得...なんこのデータが入っているのか
print(len(lint))

#要素（リストに入っているデータ）へのアクセス
#pythonでは、前から0,1,2番目..と数える
print(lstr[1])

if lint[3]+lint[4]==lint[5]:
    print('fivo')


# In[ ]:


#要素の書き換え
lint[2]+=100
print(lint)

#要素の追加
lstr.append('d')
print(lstr)

#strのlist化
st='あいうえお'
lstr2=list(st)
print(lstr2)


# In[ ]:


#データがかの確認
a = "abcd"
print(type(a))
b = 1234
print(type(b))


# In[ ]:


#スライス
#[左端:右端+1:何個飛ばし]でリストの一部を切り出せる
print(lstr2[1:4])


# In[ ]:


#tips:リストの反転
#list[::-1]
print(lstr2[::-1])


# In[ ]:


#tips:整数のリストはrangeで作れる
#range(スタート,ゴール+1,何個飛ばし)
lnum=list(range(1,15,2))
print(lnum)

#スタート、何個飛ばしは省略可
lnum2=list(range(5))
print(lnum2)


# In[ ]:


for i in range(1,10):
    if i%2==0:
    print(i)

#変数iにリストの中身を一つずつ代入してその都度下のコードを実行している


# In[ ]:


su=0
fact=1
for j in range(1,20):
    su=su+j
    fact=fact*j
print(su,fact)


# In[ ]:


list=['x','y','z']
s=''
for i in list:
    s+=i
print(s)


# In[ ]:


#ある動作をn回やりたいときに重宝

for i in range(10):
    print('WOW!')


# In[ ]:


#条件を満たしている間繰り返し続ける while もある(遅い)
n=100
while n>0:
    n-=1
    print(n)


# In[ ]:


#3


# In[ ]:


#4


# In[ ]:


def po(a,b):
    return a*b
c=po(a,b)
print(c*3)


# In[ ]:


def po(a,b):
    print(a**b)#aのb乗
po(a,b)


# In[ ]:


import os
#今いるディレクトリの表示
print(os.getcwd())


#パス（住所のようなもの）で指定したディレクトリの配下にあるものをリストで表示
x = os.listdir("/kaggle/input")
print(x)
print("データ型",type(x))


# In[ ]:


#パス（住所のようなもの）で指定したディレクトリに移動
os.chdir("/kaggle/input/house-prices-dataset")
print(os.getcwd())#移動したかわかる


# In[ ]:


#パス（住所のようなもの）で指定したディレクトリを新規に作成
os.mkdir("/kaggle/working/new_dir")
print(os.listdir("/kaggle/working"))#できた


# In[ ]:


#ここで


# In[ ]:





# In[ ]:





# In[ ]:


os.chdir("/kaggle/input/house-prices-dataset")
get_ipython().system('ls')


# In[ ]:


#まずは読み込み
import pandas as pd

house=pd.read_csv("/kaggle/input/house-prices-dataset/train.csv")
print(type(house))#DataFrameというデータ型です。
print(len(house))


# In[ ]:


house.shape


# In[ ]:


house.head(10)#上からK行のデータを参照できます。


# In[ ]:


df_1 = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})


# In[ ]:


#DataFrame.shapeで大きさを得ます。
df_1.shape


# In[ ]:


df_1.head()#DataFrameは自作できます。{}はdictというデータ型です。


# In[ ]:


s_1 = pd.Series([1, 2, 3, 4, 5])
#これは自分でdfを作成するときにちょっと使ったりします。


# In[ ]:


house.LotArea


# In[ ]:


house["LotArea"]


# In[ ]:


house["LotArea"][0]#0行目


# In[ ]:


house.iloc[0]


# In[ ]:


house.iloc[:, 0]#スライスにおいて『 : 』は”任意の”を表します。この場合は全ての行とそれに対応する0列目を取ってきている。という意味です。


# In[ ]:


#ここに


# In[ ]:


a = [1,4,6]
house.iloc[a, 0]#indexにはlistを入れても良いです。


# In[ ]:


house.loc[:, ['MSSubClass', 'Utilities', 'LotShape']]


# In[ ]:


house.iloc[:, 0:10]


# In[ ]:


house.loc[:, 0:10]


# In[ ]:


house.loc[house["LotShape"] == 'Reg']


# In[ ]:


reviews.loc[(house["LotShape"] == 'Reg') & (house["MiscVal"] >= 10)]


# In[ ]:


house.YrSold#dtype: int64なので格納されているのは整数です。


# In[ ]:


house.loc[house.YrSold.isin([2006, 2007])]


# In[ ]:


house["new_column_1"]="WOW"


# In[ ]:


#ここに


# In[ ]:


house.describe()


# In[ ]:


house.mean()#統計値の一つを見ることもできます。


# In[ ]:


house["SaleCondition"].unique()#


# In[ ]:


house["SaleCondition"]..value_counts()


# In[ ]:


house_SalePrice_mean = house.SalePrice.mean()
house["SalePrice"].map(lambda p: p - house_SalePrice_mean)#順に適応していく


# In[ ]:


#各行に対して自作の関数を適応させる（細かいことをさせるために）ときはapply
def remean_points(row):
    row.SalePrice = row.SalePrice - house_SalePrice_mean
    return row

house.apply(remean_points, axis="SalePrice")


# In[ ]:


house.head()


# In[ ]:


#ここに


# In[ ]:


house.groupby("SaleCondition").SaleCondition.count()


# In[ ]:


house.groupby("SaleCondition").count()


# In[ ]:


df_ = house.groupby("SaleCondition")
df_.head()#df.groupby(列)はdfを列で集計した新しい表これに集計関数を.func()でつけて使う


# In[ ]:


df_.apply(lambda df: df.MSZoning.iloc[0])?


# In[ ]:


house=pd.read_csv("/kaggle/input/house-prices-dataset/train.csv")
#複数の列について集計...列名をlistで囲う
house.groupby(['SalePrice', 'MoSold']).max()


# In[ ]:


#複数の関数を適応
house.groupby(['SalePrice']).price.agg([len, min, max])


# In[ ]:


#並べ替え...df.sort_values(by="列の名前")大きい順にするのは調べてください『python df　sort』


# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplitlib', 'inline#notebookで画像などを表示させるための宣言')


# In[1]:


import pandas as pd
train = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
train["image_id"] = train["image_id"]+".jpg"
train.head()


# In[6]:


file_paths = train["image_id"].values[:]


# In[4]:


import glob
x = glob.glob("/kaggle/input/plant-pathology-2020-fgvc7/images/*")


# In[ ]:


path


# In[8]:


import cv2
import os
import matplotlib.pyplot as plt
path = os.path.join("/kaggle/input/plant-pathology-2020-fgvc7/images",file_paths[0])
img = cv2.imread(path)
print(type(img))
print(img.shape)
plt.imshow(img)
plt.show()


# In[9]:


#画像の反転
flip0=cv2.flip(img,0)
plt.imshow(flip0)
plt.show()

flip1=cv2.flip(img,1)
plt.imshow(flip1)
plt.show()

flip2=cv2.flip(img,-1)
plt.imshow(flip2)
plt.show()


# In[12]:


#resize
print(img.shape)
resize =cv2.resize(img,(224,224))
print(resize.shape)
plt.imshow(resize)
plt.show()


# In[13]:


#回転
import cv2
height=img.shape[0]
width=img.shape[1]
center=(width//2,height//2)

affin_trans=cv2.getRotationMatrix2D(center,33.0,1.0)
rotate=cv2.warpAffine(img,affin_trans,(width,height))
plt.imshow(rotate)
plt.show()


# In[14]:


#トリミング
clip=img[40:height-20,40:width-20]
print(clip.shape)
plt.imshow(clip)
plt.show()


# In[ ]:


#ここに白黒化


# In[ ]:





# In[3]:


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

def crop(image): #引数は画像の相対パス
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
    x1 = [] #x座標の最小値
    y1 = [] #y座標の最小値
    x2 = [] #x座標の最大値
    y2 = [] #y座標の最大値
    for i in range(1, len(contours)):# i = 1 は画像全体の外枠になるのでカウントに入れない
        ret = cv2.boundingRect(contours[i])
        x1.append(ret[0])
        y1.append(ret[1])
        x2.append(ret[0] + ret[2])
        y2.append(ret[1] + ret[3])

    # 輪郭の一番外枠を切り抜き
    x1_min = min(x1)
    y1_min = min(y1)
    x2_max = max(x2)
    y2_max = max(y2)
    croparea = cv2.rectangle(bgr, (x1_min, y1_min), (x2_max, y2_max), (255, 0, 0), 3)


# In[ ]:


#csvの読み込みをして、train_csvについてhead()を表示してください


# In[ ]:


#モデルがしっかり動くかどうかの確認のために何件かとってくる
train = train.sample(n=100, random_state=2020)


# In[ ]:


train.head()


# In[ ]:





# In[16]:


labels = train.iloc[:,1:]
labels.head()


# In[4]:


#使いそうなライブラリ
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


# In[5]:


#使うモデル
vgg16_model=VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None)


# In[49]:


get_ipython().system('ls /kaggle/input')


# In[48]:


get_ipython().system('pwd')


# In[6]:


from tqdm import tqdm
images_paths = train["image_id"].values[:]
def get_images_from_df(df):
    images = np.empty((len(df),224,224,3))
    images_paths = df["image_id"].values[:]
    #print(images_paths[0])
    for i in tqdm(range(len(images_paths))):
        img=cv2.imread("/kaggle/input/plant-pathology-2020-fgvc7/images/{}".format(images_paths[i]))
        #画像サイズを224*224*3にしてください
        #print(img.shape)
        img = cv2.resize(img,(224,224))
        images[i, :, :, :] = img
    return images
def get_one_hot_form_df(df):
    df = df[["healthy","multiple_diseases","rust","scab"]]
    label_onehot = df.values
    return label_onehot 


# In[42]:


train.head()


# In[28]:


y = get_one_hot_form_df(train)


# In[29]:





# In[7]:


def get_model_finetune(base_model,input_shape=[None,None,3], num_classes=4):#num_classesは今回何ですか？
    base = base_model
    for layer in base_model.layers:
        layer.trainable = True#これはなんでしょうか？
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # GlobalAveragePooling2DをGlobalMaxPooling2Dに変更...https://qiita.com/mine820/items/1e49bca6d215ce88594a 全結合層の代わりに働き軽い。
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    prediction = Dense(num_classes, activation='sigmoid')(x)#最終層の活性化関数を適切に変更してください

    model = Model(input=base_model.input, output=prediction)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=1e-4),
        metrics=['accuracy']
    )#『keras loss』で調べてbinary_crossentropyを直してください
    return model


# In[13]:


test = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
test["image_id"] = test["image_id"]+".jpg"
test.head()
test_x = get_images_from_df(test)


# In[ ]:


def auc(array):
    
    


# In[20]:


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





    
    
    


# In[31]:


sub = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")
sub.head()


# In[36]:


df1 =pd.DataFrame(np.array(pred))
sub[["healthy","multiple_diseases","rust","scab"]]=df1


# In[37]:


sub.head()


# In[34]:


df1 =pd.DataFrame(np.array(pred))
df1.head()


# In[ ]:


import os
os.listdir('/kaggle/input/pytorch-efnet-ns')
import geffnet


# In[ ]:


import torch
import sys
sys.path.insert(0, '/kaggle/input/pytorch-efnet-ns/')
import geffnet
model = geffnet.tf_efficientnet_b2_ns(pretrained=False)
PATH = "/kaggle/input/pytorch-efnet-ns-weights-abebe/tf_efficientnet_b2_aa-60c94f97.pth"
model.load_state_dict(torch.load(PATH))
print(model)

