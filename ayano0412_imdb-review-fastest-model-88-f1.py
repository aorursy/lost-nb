#!/usr/bin/env python
# coding: utf-8

# In[1]:


from importlib import reload
import sys
from imp import reload
import warnings
warnings.filterwarnings('ignore')
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")


# In[2]:


import zipfile
with zipfile.ZipFile('../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip') as existing_zip:
    existing_zip.extractall()


# In[3]:


import pandas as pd

df1 = pd.read_csv('labeledTrainData.tsv', delimiter="\t")
df1 = df1.drop(['id'], axis=1)
df1.head()


# In[7]:


df = df1
df.head()


# In[8]:


import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

df['Processed_Reviews'] = df.review.apply(lambda x: clean_text(x))


# In[9]:


df.head()


# In[10]:


df.Processed_Reviews.apply(lambda x: len(x.split(" "))).mean()


# In[11]:


with zipfile.ZipFile('../input/word2vec-nlp-tutorial/testData.tsv.zip') as existing_zip:
    existing_zip.extractall()


# In[12]:


df_test=pd.read_csv("testData.tsv",header=0, delimiter="\t", quoting=3)
df_test.head()
df_test["review"]=df_test.review.apply(lambda x: clean_text(x))
df_test["sentiment"] = df_test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)
y_test = df_test["sentiment"]


# In[13]:


get_ipython().system('git clone https://github.com/facebookresearch/fastText.git')


# In[26]:


cd fastText/


# In[27]:


get_ipython().system('pip install .')


# In[28]:


cd ..


# In[29]:


import fasttext
def train_ft(train_filename, test_filename, autotune_is=False):
    if autotune_is:
        ft = fasttext.train_supervised(
            input=train_filename,
            autotuneValidationFile=test_filename,
        )
    else:
        ft = fasttext.train_supervised(
            train_filename,
            lr=0.489508510723173,
            lrUpdateRate=100,
            dim=100,
            minn=2,
            maxn=5,
            ws=5,
            epoch=100,
            neg=5,
        )
    return ft


# In[30]:


def print_fasttext_params(model):
    print("lr: {}".format(model.lr))
    print("lrUpdateRate: {}".format(model.lrUpdateRate))
    print("dim: {}".format(model.dim))
    print("minCount: {}".format(model.minCount))
    print("minCountLabel: {}".format(model.minCountLabel))
    print("minn: {}".format(model.minn))
    print("maxn: {}".format(model.maxn))
    print("wordNgrams: {}".format(model.wordNgrams))
    print("ws: {}".format(model.ws))
    print("epoch: {}".format(model.epoch))
    print("neg: {}".format(model.neg))
    print("loss: {}".format(model.loss))


# In[31]:


train_filename = "train.txt"
test_filename = "test.txt"
autotune_is = True
with open(train_filename, 'w') as f_train:
    for row_data in df.itertuples():
        f_train.write('__label__{} '.format(row_data.sentiment) + row_data.Processed_Reviews + '\n')
with open(test_filename, 'w') as f_test:
    for row_data in df_test.itertuples():
        f_test.write('__label__{} '.format(row_data.sentiment) + row_data.review + '\n')

            
            
ft = train_ft(train_filename, test_filename, autotune_is)


# In[32]:


print_fasttext_params(ft)


# In[33]:


pred_labels, pred_probs = ft.predict(df_test["review"].tolist(), k=-1)
y_pred = [int(x[0][-1]) for x in pred_labels]


from sklearn.metrics import precision_score,recall_score,f1_score, confusion_matrix
print('precision-score: {0}'.format(precision_score(y_pred, y_test)))
print('recall-score: {0}'.format(recall_score(y_pred, y_test)))
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)


# In[ ]:





# In[34]:


# ouput submission file 
import copy
df_sub = copy.deepcopy(df_test)
df_sub["sentiment"] = y_pred
df_sub = df_sub[['id','sentiment']]
df_sub.to_csv("submission.csv",index=False)


# In[ ]:




