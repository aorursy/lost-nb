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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text  import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.validation import column_or_1d
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from bokeh.plotting import figure, show
from sklearn.metrics import log_loss
from bokeh.io import output_notebook
from bokeh.charts import Bar, output_file, show
from bokeh.sampledata.autompg import autompg as df
from bokeh.plotting import figure
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
import numpy as np
from bokeh.charts import Histogram
print(check_output(["ls", "../input"]).decode("utf8"))
PIPELINE_PICKLE = "pipeline.pkl"

# Any results you write to the current directory are saved as output.


# In[2]:


n_rows = 5000


# In[3]:


train = pd.read_csv("../input/training_text", sep = '\|\|'
                    , skiprows= 1
                    ,names = ['Text']
                    ,engine = 'python', nrows = n_rows)
test = pd.read_csv("../input/training_variants"
                  ,  usecols = ['Class'], dtype= 'category', nrows = n_rows)
hold_out = pd.read_csv("../input/test_text", sep = '\|\|', skiprows = 1,
                      engine = 'python')


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:





# In[6]:


X_train,X_test,Y_train, Y_test = train_test_split(train['Text'], pd.get_dummies(test,prefix_sep="")
                                                  ,random_state = 11
                                                  ,test_size = .3)


# In[7]:


aa = Y_train.agg


# In[8]:


factors = list(X_test)
factors


# In[9]:


factors


# In[10]:


output_notebook()


# In[11]:


Y_train.shape


# In[12]:


Y_test.head()


# In[13]:


from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

vect = CountVectorizer(tokenizer=LemmaTokenizer())  


# In[14]:


pre_process_pl = Pipeline([
    ('vect', CountVectorizer(stop_words= 'english', tokenizer=LemmaTokenizer())),
    ('tfidf', TfidfTransformer())
])


# In[15]:


joblib.load(pre_process_pl,PIPELINE_PICKLE)


# In[16]:


param_grid = {'clf__estimator__C':[ 1,2 , 4 ,7, 10,12,20]}


# In[17]:


X_train.shape


# In[18]:


gv_search = GridSearchCV(pre_process_pl, param_grid = param_grid, n_jobs=-1, scoring="neg_log_loss")


# In[19]:


get_ipython().run_cell_magic('time', '', 'pre_process_pl.fit(X_train, Y_train)')


# In[20]:


pl_clf_train = Pipeline([('pre',pre_process_pl)
                         , ('clf',OneVsRestClassifier(LogisticRegression()))])


# In[21]:


get_ipython().run_cell_magic('time', '', 'pl_clf_train.fit(X_train,Y_train)')


# In[22]:


gv_search.cv_results_


# In[23]:


gv_search.score(X_test, Y_test)


# In[24]:


gv_search.predict_proba(X_test)[1]


# In[25]:


hold_out_data = gv_search.predict_proba(hold_out)


# In[26]:


estimator_dataframe = pd.DataFrame(gv_search.cv_results_)


# In[27]:


row_column = np.arange(0,hold_out.shape[1])


# In[28]:


p = figure(x_axis_label = 'C', y_axis_label = 'Log Loss',plot_width = 300, plot_height = 300)


p.line(estimator_dataframe.param_clf__estimator__C, estimator_dataframe.mean_test_score, line_width=4)
p.circle(estimator_dataframe.param_clf__estimator__C, estimator_dataframe.mean_test_score, fill_color = 'white',
         size = 10
        )
show(p)


# In[29]:


joblib.dump(pre_process_pl,PIPELINE_PICKLE)


# In[30]:





# In[30]:





# In[30]:




