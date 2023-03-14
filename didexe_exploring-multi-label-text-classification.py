#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix, multilabel_confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import warnings
warnings.filterwarnings('ignore')


# In[3]:


train_set = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test_set = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
test_labels = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test_labels.csv')


# In[4]:


print('train set', train_set.head(10))
print('test set', test_set.head(10))
print('test labels', test_labels.head(10))


# In[5]:


print(train_set.isna().sum())
print(test_set.isna().sum())
print(test_labels.isna().sum())


# In[6]:


test_set = test_set[test_labels['toxic'] != -1]
test_labels = test_labels[test_labels['toxic'] != -1]
test_features = test_set.comment_text
train_features = train_set.comment_text
print('test labels', test_labels.head(10))
train_labels = train_set.drop(['id', 'comment_text'], axis = 1)
test_labels = test_labels.drop(['id'], axis = 1)
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# In[7]:


def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

test_features_cleaned = test_features.map(lambda com : clean_text(com))
train_features_cleaned = train_features.map(lambda com : clean_text(com))


# In[8]:


print('Percentage of comments without labels: ')
print(len((train_set[(train_set.toxic == 0) & (train_set.severe_toxic == 0) & (train_set.obscene == 0) & (train_set.insult == 0) & (train_set.insult == 0) & (train_set.identity_hate == 0)])) / len(train_set)*100)
print('Percentage of comments with one or more labels: ')
print(len(train_set[(train_set.toxic == 1) | (train_set.severe_toxic == 1) | (train_set.obscene == 1) | (train_set.insult == 1) | (train_set.insult == 1) | (train_set.identity_hate == 1)]) / len(train_set)*100)


# In[9]:


test_labels.describe()


# In[10]:


fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([0,0,1,1])
total_count = []
for label in labels:
    total_count.append(len(train_labels[train_labels[label] == 1]))
ax.bar(labels,total_count, color=['red', 'green', 'blue', 'purple', 'orange', 'yellow'])
for i,data in enumerate(total_count):
    plt.text(i-.25, 
              data/total_count[i]+100, 
              total_count[i], 
              fontsize=12)
plt.title('Number of comments per label')
plt.xlabel('Labels')
plt.ylabel('Number of comments')

plt.show()


# In[11]:


NB_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                       ('nb_model', OneVsRestClassifier(MultinomialNB(), n_jobs=-1))])

LR_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                       ('lr_model', OneVsRestClassifier(LogisticRegression(), n_jobs=-1))])

# SVM_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
#                        ('svm_model', OneVsRestClassifier(SVC(kernel='linear',probability=True)))])

SVM_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                       ('svm_model', OneVsRestClassifier(LinearSVC(), n_jobs=-1))])

def plot_roc_curve(test_features, predict_prob):
    fpr, tpr, thresholds = roc_curve(test_features, predict_prob)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for toxic comments')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.legend(labels)

def run_pipeline(pipeline, train_feats, train_lbls, test_feats, test_lbls):
    pipeline.fit(train_feats, train_labels)
    predictions = pipeline.predict(test_feats)
    pred_proba = pipeline.predict_proba(test_feats)
    print('roc_auc: ', roc_auc_score(test_lbls, pred_proba))
    print('accuracy: ', accuracy_score(test_lbls, predictions))
    print('confusion matrices: ')
    print(multilabel_confusion_matrix(test_lbls, predictions))
    print('classification_report: ')
    print(classification_report(test_lbls, predictions, target_names=labels))
    
def run_SVM_pipeline(pipeline, train_feats, train_lbls, test_feats, test_lbls):
    pipeline.fit(train_feats, train_labels)
    predictions = pipeline.predict(test_feats)
    print('accuracy: ', accuracy_score(test_lbls, predictions))
    print('confusion matrices: ')
    print(multilabel_confusion_matrix(test_lbls, predictions))
    print('classification_report: ')
    print(classification_report(test_lbls, predictions, target_names=labels))
    
def plot_pipeline_roc_curve(pipeline, train_feats, train_lbls, test_feats, test_lbls):
    for label in labels:
        pipeline.fit(train_feats, train_set[label])
        pred_proba = pipeline.predict_proba(test_feats)[:,1]
        plot_roc_curve(test_lbls[label], pred_proba)


# In[12]:


run_pipeline(NB_pipeline, train_features, train_labels, test_features, test_labels)


# In[13]:


run_pipeline(LR_pipeline, train_features, train_labels, test_features, test_labels)


# In[14]:


# run_pipeline(SVM_pipeline, train_features, train_labels, test_features, test_labels)
run_SVM_pipeline(SVM_pipeline, train_features, train_labels, test_features, test_labels)


# In[15]:


plot_pipeline_roc_curve(LR_pipeline, train_features, train_labels, test_features, test_labels)


# In[16]:


plot_pipeline_roc_curve(NB_pipeline, train_features, train_labels, test_features, test_labels)


# In[17]:


# TAKES ABOUT 40min to run
# alpha = [0.1,1,10]
# penalty=['l1','l2']
# n_gram=[(1,1),(1,2)]
# param_grid = {
#     'tfidf__ngram_range': n_gram,
#     'lr_model__estimator__C': alpha
# }
# gsearch_cv = GridSearchCV(LR_pipeline, param_grid=param_grid, cv=5)
# gsearch_cv.fit(train_features, train_labels)


# In[18]:


gsearch_cv.best_score_
# 0.919784923325667


# In[19]:


gsearch_cv.best_params_
# {'lr_model__estimator__C': 10, 'tfidf__ngram_range': (1, 2)}


# In[20]:


LR_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
                       ('lr_model', OneVsRestClassifier(LogisticRegression(C=10), n_jobs=-1))])


# In[21]:


run_pipeline(LR_pipeline, train_features, train_labels, test_features, test_labels)


# In[22]:


# change parameters manually to get the results above
LR_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
                       ('lr_model', OneVsRestClassifier(LogisticRegression(C=10), n_jobs=-1))])


# In[23]:


run_pipeline(LR_pipeline, train_features_cleaned, train_labels, test_features_cleaned, test_labels)

