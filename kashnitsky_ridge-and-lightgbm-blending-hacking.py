#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import json
from tqdm import tqdm_notebook
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
import lightgbm as lgb


# In[2]:


from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


# In[3]:


PATH_TO_DATA = '../input/'


# In[4]:


def read_json_line(line=None):
    result = None
    try:        
        result = json.loads(line)
    except Exception as e:      
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')',''))      
        # Remove the offending character:
        new_line = list(line)
        new_line[idx_to_replace] = ' '
        new_line = ''.join(new_line)     
        return read_json_line(line=new_line)
    return result


# In[5]:


def extract_titles_from_json(path_to_inp_json_file, path_to_out_txt_file, total_length):
    '''
    :param path_to_inp_json_file: path to a JSON file with train/test data
    :param path_to_out_txt_file: path to extracted features (here titles), one per line
    :param total_length: we'll pass the hardcoded file length to make tqdm even more convenient
    '''
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file,          open(path_to_out_txt_file, 'w', encoding='utf-8') as out_file:
        for line in tqdm_notebook(inp_file, total=total_length):
            json_data = read_json_line(line)
            content = json_data['title'].replace('\n', ' ').replace('\r', ' ')
            content_no_html_tags = strip_tags(content)
            out_file.write(content_no_html_tags + '\n')


# In[6]:


get_ipython().run_cell_magic('time', '', "extract_titles_from_json(path_to_inp_json_file=os.path.join(PATH_TO_DATA, 'train.json'),\n           path_to_out_txt_file='train_titles.txt', total_length=62313)")


# In[7]:


get_ipython().run_cell_magic('time', '', "extract_titles_from_json(path_to_inp_json_file=os.path.join(PATH_TO_DATA, 'test.json'),\n           path_to_out_txt_file='test_titles.txt', total_length=34645)")


# In[8]:


tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 3))


# In[9]:


get_ipython().run_cell_magic('time', '', "with open('train_titles.txt', encoding='utf-8') as input_train_file:\n    X_train = tfidf.fit_transform(input_train_file)")


# In[10]:


get_ipython().run_cell_magic('time', '', "with open('test_titles.txt', encoding='utf-8') as input_test_file:\n    X_test = tfidf.transform(input_test_file)")


# In[11]:


X_train.shape, X_test.shape


# In[12]:


train_target = pd.read_csv(os.path.join(PATH_TO_DATA, 
                                        'train_log1p_recommends.csv'), 
                           index_col='id')


# In[13]:


y_train = train_target['log_recommends'].values


# In[14]:


plt.hist(y_train, bins=30, alpha=.5, color='red', 
         label='original', range=(0,10));
plt.hist(np.log1p(y_train), bins=30, alpha=.5, color='green', 
         label='log1p', range=(0,10));
plt.legend();


# In[15]:


train_part_size = int(0.7 * train_target.shape[0])
X_train_part = X_train[:train_part_size, :]
y_train_part = y_train[:train_part_size]
X_valid =  X_train[train_part_size:, :]
y_valid = y_train[train_part_size:]


# In[16]:


ridge = Ridge(random_state=17)


# In[17]:


get_ipython().run_cell_magic('time', '', 'ridge.fit(X_train_part, np.log1p(y_train_part));')


# In[18]:


ridge_pred = np.expm1(ridge.predict(X_valid))


# In[19]:


lgb_x_train_part = lgb.Dataset(X_train_part.astype(np.float32), 
                           label=np.log1p(y_train_part))


# In[20]:


lgb_x_valid = lgb.Dataset(X_valid.astype(np.float32), 
                      label=np.log1p(y_valid))


# In[21]:


param = {'num_leaves': 255, 
         'objective': 'mean_absolute_error',
         'metric': 'mae'}


# In[22]:


num_round = 200
bst_lgb = lgb.train(param, lgb_x_train_part, num_round, 
                    valid_sets=[lgb_x_valid], early_stopping_rounds=20)


# In[23]:


lgb_pred = np.expm1(bst_lgb.predict(X_valid.astype(np.float32), 
                                    num_iteration=bst_lgb.best_iteration))


# In[24]:


plt.hist(y_valid, bins=30, alpha=.5, color='red', label='true', range=(0,10));
plt.hist(ridge_pred, bins=30, alpha=.5, color='green', label='Ridge', range=(0,10));
plt.hist(lgb_pred, bins=30, alpha=.5, color='blue', label='Lgbm', range=(0,10));
plt.legend();


# In[25]:


ridge_valid_mae = mean_absolute_error(y_valid, ridge_pred)
ridge_valid_mae


# In[26]:


lgb_valid_mae = mean_absolute_error(y_valid, lgb_pred)
lgb_valid_mae


# In[27]:


mean_absolute_error(y_valid, .4 * lgb_pred + .6 * ridge_pred)


# In[28]:


get_ipython().run_cell_magic('time', '', 'ridge.fit(X_train, np.log1p(y_train));')


# In[29]:


get_ipython().run_cell_magic('time', '', 'ridge_test_pred = np.expm1(ridge.predict(X_test))')


# In[30]:


lgb_x_train = lgb.Dataset(X_train.astype(np.float32),
                          label=np.log1p(y_train))


# In[31]:


num_round = 60
bst_lgb = lgb.train(param, lgb_x_train, num_round)


# In[32]:


lgb_test_pred = np.expm1(bst_lgb.predict(X_test.astype(np.float32)))


# In[33]:


mix_pred = .4 * lgb_test_pred + .6 * ridge_test_pred


# In[34]:


mean_test_target = 4.33328 


# In[35]:


mix_test_pred_modif = mix_pred + mean_test_target - y_train.mean()


# In[36]:


def write_submission_file(prediction, filename,
                          path_to_sample=os.path.join(PATH_TO_DATA, 
                                                      'sample_submission.csv')):
    submission = pd.read_csv(path_to_sample, index_col='id')
    
    submission['log_recommends'] = prediction
    submission.to_csv(filename)


# In[37]:


#write_submission_file(ridge_test_pred + mean_test_target - y_train.mean(), 'ridge_submission.csv')


# In[38]:


#write_submission_file(lgb_test_pred + mean_test_target - y_train.mean(), 'lgb_submission.csv')


# In[39]:


write_submission_file(mix_test_pred_modif, 'submission.csv')

