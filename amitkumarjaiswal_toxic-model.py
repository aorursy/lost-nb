#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd

from model_tool import ToxModel
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords
import re


# In[2]:


training_3rd_vis = pd.read_csv('clean_train_aug_third.csv')

display(training_3rd_vis["comment_text"].head(n=20))
n_records_features_vis_3rd = len(training_3rd_vis)
print(" Number of features {}".format(n_records_features_vis_3rd))
train = 'clean_train_aug_third.csv'


# In[3]:


model_list = []


# In[ ]:


MODEL_NAME = 'aug_gruconv_twitter'
debias_random_model = ToxModel()
debias_random_model.train(2,train, text_column = 'comment_text', toxic = 'toxic', severe_toxic = 'severe_toxic', obscene = 'obscene', threat = 'threat', insult = 'insult', identity_hate = 'identity_hate', model_name = MODEL_NAME, model_list = model_list)


# In[ ]:


random_test = pd.read_csv('cleaned_test_clean.csv')


# In[14]:


random_test.head()


# In[4]:


MODEL_NAME = 'concat_bigru'
debias_random_model = ToxModel()
debias_random_model.predict_test(2,train, text_column = 'comment_text', toxic = 'toxic', severe_toxic = 'severe_toxic', obscene = 'obscene', threat = 'threat', insult = 'insult', identity_hate = 'identity_hate', model_name = MODEL_NAME, model_list = model_list)


# In[ ]:


from keras.models import load_model
import os
model_list = []
for fold_id in range(0, 10):
    model_path = 'augmentori_gru_lstm' + str(fold_id)
    model = load_model(
        os.path.join('models', '%s_model.h5' % model_path))
    model_list.append(model)
    


# In[8]:


from keras.models import load_model
import numpy as np
import os
model_list = []
for fold_id in range(0, 10):
    model_path = 'augmentori_gru_lstm' + str(fold_id)
    model = load_model(
        os.path.join('models', '%s_model.h5' % model_path))
    model_path = os.path.join('models', "model{0}_weights.npy".format(fold_id))
    weights = np.load(model_path)
    model.set_weights(weights)
    model_list.append(model)


# In[11]:


from keras.preprocessing.sequence import pad_sequences
import cPickle
import os
def prep_text(texts):
    """Turns text into into padded sequences.

    The tokenizer must be initialized before calling this method.

    Args:
      texts: Sequence of text strings.

    Returns:
      A tokenized and padded text sequence as a model input.
    """
    model_name = 'augmentori_gru_lstm'
    tokenizer = cPickle.load(
        open(
            os.path.join('models', '%s_tokenizer.pkl' % model_name),
            'rb'))
    text_sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(
        text_sequences, maxlen=250)


# In[12]:



total_meta = []
meta_train = pd.read_csv('final_train.csv')
X_test = meta_train['comment_text']
X_test = prep_text(X_test)
X= X_test
fold_size = len(X) // 10
for fold_id in range(0, 10):
    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size
            
    if fold_id == 10 - 1:
        fold_end = len(X)

    train_x = np.concatenate([X[:fold_start], X[fold_end:]])

    val_x = X[fold_start:fold_end]
          
    meta = model_list[fold_id].predict(val_x, batch_size=128)
    if (fold_id == 0):
        total_meta = meta
    else:
        total_meta = np.concatenate((total_meta, meta), axis=0)


# In[13]:


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
subm = pd.read_csv('sample_submission.csv')
submid = pd.DataFrame({'id': subm["id"]})
total_meta_data = pd.concat([submid, pd.DataFrame(total_meta, columns = label_cols)], axis=1)


# In[14]:


display(total_meta_data.head(n=20))


# In[15]:


total_meta_data.to_csv('augmentori_meta_grulstmCV_nopretrain.csv', index=False)


# In[8]:


test_predicts = pd.read_csv('gru_cv_output.csv')
display(test_predicts.head(n=20))
test_predicts.shape


# In[3]:


MODEL_NAME = 'multi-labelNLP_charrnn'
debias_random_model = ToxModel()
debias_random_model.train(1,train, text_column = 'comment_text', toxic = 'toxic', severe_toxic = 'severe_toxic', obscene = 'obscene', threat = 'threat', insult = 'insult', identity_hate = 'identity_hate', model_name = MODEL_NAME)


# In[8]:


MODEL_NAME = 'multi-labelNLP-second'
second_model = ToxModel()
second_model.train(0,train, text_column = 'comment_text', toxic = 'toxic', severe_toxic = 'severe_toxic', obscene = 'obscene', threat = 'threat', insult = 'insult', identity_hate = 'identity_hate', model_name = MODEL_NAME)


# In[7]:


debias_random_model = ToxModel(model_name="multi-labelNLP-gru-cv0") 


# In[4]:


second_model = ToxModel(model_name="multi-labelNLP-second") 


# In[8]:


import numpy as np
random_test = pd.read_csv('test.csv')
np.where(pd.isnull(random_test)) #check null rows


# In[7]:


print(random_test.iloc[52300]) #print value of null row


# In[9]:


random_test = pd.read_csv('test.csv')
prediction = debias_random_model.predict(random_test['comment_text'])


# In[10]:


prediction.shape


# In[9]:


random_test = pd.read_csv('test.csv')
random_test = random_test.dropna()
print(random_test.iloc[52300])


# In[10]:


random_test.shape


# In[8]:


for id, p in enumerate(prediction):
    if(id <20):
        print(p)


# In[9]:


#second model
random_test = pd.read_csv('test.csv')
random_test = random_test.dropna()
prediction_second = second_model.predict(random_test['comment_text'])
prediction_second.shape


# In[10]:


for id, p in enumerate(prediction_second):
    if(id <20):
        print(p)


# In[11]:


random_test = pd.read_csv('test.csv')
test_id = random_test['id'].astype(str)


# In[12]:


test_id.shape


# In[13]:


header = ["id"]
df = pd.DataFrame(test_id, columns=header)

df.id = df.id.astype("str")
print(df.dtypes)
display(df.head(n=20))


# In[14]:


#IF NO SPLIT
headers = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
test_df = pd.DataFrame(prediction, columns=headers, dtype=float)
display(test_df.head(n=20))
print(np.where(pd.isnull(test_df)))
print(test_df.shape)
test_df.reset_index(drop=True, inplace=True)


# In[14]:


#IF SPLIT
headers = ["toxic","severe_toxic"]
test_df_second = pd.DataFrame(prediction_second, columns=headers, dtype=float)
display(test_df_second.head(n=20))
print(np.where(pd.isnull(test_df_second)))
print(test_df_second.shape)
test_df_second.reset_index(drop=True, inplace=True)


# In[15]:


headers = ["obscene","threat","insult","identity_hate"]
test_df = pd.DataFrame(prediction, columns=headers, dtype=float)
display(test_df.head(n=20))
print(np.where(pd.isnull(test_df)))
print(test_df.shape)
test_df.reset_index(drop=True, inplace=True)


# In[15]:


#IF NO SPLIT
df_new = pd.concat([df,test_df], axis=1)
#df_new = df.merge(test_df, how='outer')
#df_new.id = df_new.id.astype("int")
display(df_new.head(n=20))
print(df_new.dtypes)

np.where(pd.isnull(df_new))


# In[18]:


#IF SPLIT
df_new = pd.concat([df,test_df_second,test_df], axis=1)
display(df_new.head(n=20))
print(df_new.dtypes)

np.where(pd.isnull(df_new))


# In[13]:


df_new.shape


# In[16]:


head = ["id","toxic","severe_toxic","obscene","threat","insult","identity_hate"]
df_new.to_csv('cv_gru_output.csv', columns = head, index=False)


# In[ ]:


for id, p in enumerate(prediction):
    


# In[7]:


MODEL_NAME = 'cnn_wiki_tox_v3'
wiki_model = ToxModel()
wiki_model.train(wiki['train'], wiki['dev'], text_column = 'comment', label_column = 'is_toxic', model_name = MODEL_NAME)


# In[8]:


wiki_test = pd.read_csv(wiki['test'])
wiki_model.score_auc(wiki_test['comment'], wiki_test['is_toxic'])


# In[9]:


MODEL_NAME = 'cnn_debias_tox_v3'
debias_model = ToxModel()
debias_model.train(debias['train'], debias['dev'], text_column = 'comment', label_column = 'is_toxic', model_name = MODEL_NAME)


# In[11]:


debias_test = pd.read_csv(debias['test'])
debias_model.prep_data_and_score(debias_test['comment'], debias_test['is_toxic'])


# In[ ]:




