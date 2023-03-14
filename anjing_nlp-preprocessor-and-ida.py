#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
print(os.listdir("./"))
print(os.listdir("../input/googlenewsvectorsnegative300"))
print(os.listdir("../input/jigsaw-unintended-bias-in-toxicity-classification"))

# Any results you write to the current directory are saved as output.


# In[2]:


import numpy as np
import pandas as pd
pd.set_option('precision', 4)  #设置显示精度
pd.set_option('display.float_format',lambda x: '%.4f'%x)  # float 不用科学计数法
pd.set_option('display.expand_frame_repr',False)   # 不允许换行显示
pd.set_option('max_colwidth',300) #显示长度默认为50，这里设置300

from tqdm import tqdm   ## 进度条
tqdm.pandas()
import gc   ## 回收内存


# In[3]:


train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')


# In[4]:


coll = ['black','white','homosexual_gay_or_lesbian','muslim']

identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']


# In[5]:


np.set_printoptions(threshold=1000)


# In[6]:


weights = np.ones((len(train),)) / 4
weights


# In[7]:


# Subgroup  identity_columns  
weights += (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4
np.unique(weights)


# In[8]:


# Background Positive, Subgroup Negative  -----target>0.5, identity_columns都<0.5
weights += (( (train['target'].values>=0.5).astype(bool).astype(np.int) +(train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
np.unique(weights)  


# In[9]:


# Background Negative, Subgroup Positive  -----target<0.5, identity_columns都>0.5
weights += (( (train['target'].values<0.5).astype(bool).astype(np.int) +
   (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
np.unique(weights) 
len(train)


# In[10]:


len(weights)


# In[11]:


train.head()


# In[12]:


train.isnull().sum()/len(train)


# In[13]:


train['target'].describe()


# In[14]:


train.loc[train['target']>=0.5,'target'] = 1
train.loc[train['target']<0.5,'target'] = 0


# In[15]:


train['target'].value_counts()


# In[16]:


list_all = list(train['comment_text'])+list(test['comment_text'])
from tqdm import tqdm
length_list = []
word_all = []
for i in tqdm(list_all):
    length_list.append(len(i))
    for j in i.split():
        word_all.append(j)
set_all = set(word_all)


# In[17]:


我们的预处理流程很大程度上取决于我们将用于分类任务的word2vec嵌入。 **原则上，我们的预处理应该与训练单词嵌入之前使用的预处理相匹配**。

按照由易到难到顺序，依次进行：
1. 全部转为小写
2. 去除数字
3. 替换常见简写
4. 去除标点符号，以及各种特殊字符
5. 检查拼写错误

除了上述技术之外，还有其他文本预处理技术，如词干化，词形还原和删除词。由于这些技术不与Deep Learning NLP模型一起使用，因此我们不会谈论它们


# In[18]:


# lower
train['comment_text'] = train['comment_text'].str.lower()
test['comment_text'] = test['comment_text'].str.lower()


# In[19]:


def clean_numbers_1(x):
    return re.sub('\d+', ' ', x)

def clean_numbers_2(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x


# In[20]:


import re
train['comment_text'] = train['comment_text'].apply(clean_numbers_1)
test['comment_text'] = test['comment_text'].apply(clean_numbers_1)


# In[21]:


contractions_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}


# In[22]:


def _get_contractions(contractions_dict):
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
    return contractions_dict, contractions_re

def replace_typical_contractions(text):
    contractions, contractions_re = _get_contractions(contractions_dict)

    def replace(match):
        return contractions[match.group(0)]

    return contractions_re.sub(replace, text)


# In[23]:


# clean misspellings
train['comment_text'] = train['comment_text'].apply(replace_typical_contractions)
test['comment_text'] = test['comment_text'].apply(replace_typical_contractions)


# In[24]:


list_all = list(train['comment_text'])+list(test['comment_text'])
from tqdm import tqdm
word_all = []
for i in tqdm(list_all):
    for j in i.split():
        word_all.append(j)
set_all = set(word_all)

list_1 = []
temp = ''
for k in set_all:
    common = 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM'
    for l in common:
        k = k.replace(l,'')
    list_1.append(k)
# set(list_1)

str_1 = ''
for i in list_1:
    str_1+=i
set_1 = set(str_1)

punct_1 = ''
for i in set_1:
    punct_1+=i
punct_1


# In[25]:


# 删除中间变量，节省内存
del list_all,word_all,set_all,list_1,temp,str_1 
gc.collect()


# In[26]:


def preprocess_special_chars(data):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    def clean_special_chars(text, punct_1):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    return data


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




