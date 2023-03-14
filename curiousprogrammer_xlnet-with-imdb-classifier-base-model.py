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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


#Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go


# In[2]:


df_train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
df_test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
df_sample = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')


# In[3]:


df_train.head(20)


# In[4]:


df_test.head(20)


# In[5]:


df_train['comment_text'][0]


# In[6]:


df_train.shape, df_test.shape


# In[7]:


lengths = df_train.comment_text.str.len()
lengths.mean(), lengths.std(), lengths.min(), lengths.max()


# In[8]:


lengths = df_test.comment_text.str.len()
lengths.mean(), lengths.std(), lengths.min(), lengths.max()


# In[9]:


def preprocess_reviews(text):
    text = re.sub(r'<[^>]*>', ' ', text, re.UNICODE)
    text = re.sub(r'[^\w\s]', '', text, re.UNICODE)
    text = re.sub(r'[^0-9a-zA-Z]+',' ',text, re.UNICODE)
    text = text.lower()
    return text

df_train['processed_comment_text'] = df_train.comment_text.apply(lambda x: preprocess_reviews(x))
df_test['processed_comment_text'] = df_test.comment_text.apply(lambda x: preprocess_reviews(x))

df_train = df_train.sample(frac=0.4)


# In[10]:


# import module we'll need to import our custom module
from shutil import copyfile

# copy our file into the working directory (make sure it has .py suffix)
for f in os.listdir('../input/xlnetcode/'):
    try:
        if f.split('.')[1] in ['py', 'json']:
            copyfile(src = "../input/xlnetcode/"+f, dst = "../working/"+f)
    except:
        continue
print(os.listdir('../working'))


# In[11]:


get_ipython().system("mkdir '../input/train'")
get_ipython().system("mkdir '../input/test'")


# In[12]:


get_ipython().system("mkdir '../input/train/pos'")
get_ipython().system("mkdir '../input/train/neg'")


# In[13]:


train = df_train[['target','processed_comment_text']]
train['target'] = np.where(train['target']>=0.5,1,0)
train.head(10)


# In[14]:


for index, data in train.iterrows():
    if data['target'] == 0:
        f = open('../input/train/neg/' + str(index) + '.txt', "w")
        f.write(data['processed_comment_text'])
        f.close()
    else:
        f = open('../input/train/pos/' + str(index) + '.txt', "w")
        f.write(data['processed_comment_text'])
        f.close()


# In[15]:


test = df_test[['processed_comment_text']]
test.head(10)


# In[16]:


overwrite_data = True
output_dir = '../working'
data_dir = '../working'
iterations_per_loop = 1000
num_hosts = 1
num_core_per_host = 8
max_save = 0
save_steps = None
strategy = None
max_seq_length = 128
num_passes = 1


# In[17]:


SCRIPTS_DIR = '../working' #@param {type:"string"}
DATA_DIR = '../input/' #@param {type:"string"}
OUTPUT_DIR = '../' #@param {type:"string"}
PRETRAINED_MODEL_DIR = '../input/xlnetcode' #@param {type:"string"}
CHECKPOINT_DIR = '../' #@param {type:"string"}


# In[18]:


train_command = "python run_classifier.py   --do_train=True   --do_eval=False   --eval_all_ckpt=True   --task_name=imdb   --data_dir="+DATA_DIR+"   --output_dir="+OUTPUT_DIR+"   --model_dir="+CHECKPOINT_DIR+"   --uncased=False   --spiece_model_file="+PRETRAINED_MODEL_DIR+"/spiece.model   --model_config_path="+PRETRAINED_MODEL_DIR+"/xlnet_config.json   --init_checkpoint="+PRETRAINED_MODEL_DIR+"/xlnet_model.ckpt   --max_seq_length=128   --train_batch_size=8   --eval_batch_size=8   --num_hosts=1   --num_core_per_host=1   --learning_rate=2e-5   --train_steps=4000   --warmup_steps=500   --save_steps=500   --iterations=2"

get_ipython().system(' {train_command}')


# In[ ]:




