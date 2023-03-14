#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import os
import re
import pickle
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_20newsgroups, load_files
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[2]:


PATH_TO_ALL_DATA = '../input/spooky-vw-tutorial/'
data_demo = pd.read_csv(os.path.join(PATH_TO_ALL_DATA, 'weights_heights.csv'))


# In[3]:


plt.scatter(data_demo['Weight'], data_demo['Height']);
plt.xlabel('Weight in lb')
plt.ylabel('Height in inches');


# In[4]:


df = pd.read_csv(os.path.join(PATH_TO_ALL_DATA, 'bank_train.csv'))
labels = pd.read_csv(os.path.join(PATH_TO_ALL_DATA,
                                  'bank_train_target.csv'), header=None)

df.head()


# In[5]:


df['education'].value_counts().plot.barh();


# In[6]:


label_encoder = LabelEncoder()


# In[7]:


mapped_education = pd.Series(label_encoder.fit_transform(df['education']))
mapped_education.value_counts().plot.barh()
print(dict(enumerate(label_encoder.classes_)))


# In[8]:


df['education'] = mapped_education
df.head()


# In[9]:


categorical_columns = df.columns[df.dtypes == 'object'].union(['education'])
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])
df.head()


# In[10]:


df.loc[1].job - df.loc[2].job


# In[11]:


def logistic_regression_accuracy_on(dataframe, labels):
    features = dataframe.as_matrix()
    train_features, test_features, train_labels, test_labels =         train_test_split(features, labels)

    logit = LogisticRegression()
    logit.fit(train_features, train_labels)
    return classification_report(test_labels, logit.predict(test_features))

print(logistic_regression_accuracy_on(df[categorical_columns], labels))


# In[12]:


one_hot_example = pd.DataFrame([{i: 0 for i in range(10)}])
one_hot_example.loc[0, 6] = 1
one_hot_example


# In[13]:


onehot_encoder = OneHotEncoder(sparse=False)


# In[14]:


encoded_categorical_columns = pd.DataFrame(onehot_encoder.fit_transform(df[categorical_columns]))
encoded_categorical_columns.head()


# In[15]:


print(logistic_regression_accuracy_on(encoded_categorical_columns, labels))


# In[16]:


for s in ('university.degree', 'high.school', 'illiterate'):
    print(s, '->', hash(s))


# In[17]:


hash_space = 25
for s in ('university.degree', 'high.school', 'illiterate'):
    print(s, '->', hash(s) % hash_space)


# In[18]:


hashing_example = pd.DataFrame([{i: 0.0 for i in range(hash_space)}])
for s in ('job=student', 'marital=single', 'day_of_week=mon'):
    print(s, '->', hash(s) % hash_space)
    hashing_example.loc[0, hash(s) % hash_space] = 1
hashing_example


# In[19]:


assert hash('no') == hash('no')
assert hash('housing=no') != hash('loan=no')


# In[20]:


get_ipython().run_cell_magic('capture', '', '!git clone --recursive https://github.com/VowpalWabbit/vowpal_wabbit.git \n!cd vowpal_wabbit/; make \n!cd vowpal_wabbit/; make install ')


# In[21]:


get_ipython().system('vw --help')


# In[22]:


get_ipython().system(" echo '1 1.0 |Subject WHAT car is this |Organization University of Maryland:0.5 College Park' | vw")


# In[23]:


# load data with sklearn's fubction 
newsgroups = fetch_20newsgroups(PATH_TO_ALL_DATA)


# In[24]:


newsgroups['target_names']


# In[25]:


text = newsgroups['data'][0]
target = newsgroups['target_names'][newsgroups['target'][0]]

print('-----')
print(target)
print('-----')
print(text.strip())
print('----')


# In[26]:


def to_vw_format(document, label=None):
    return str(label or '') + ' |text ' + ' '.join(re.findall('\w{3,}', document.lower())) + '\n'

to_vw_format(text, 1 if target == 'rec.autos' else -1)


# In[27]:


all_documents = newsgroups['data']
all_targets = [1 if newsgroups['target_names'][target] == 'rec.autos' 
               else -1 for target in newsgroups['target']]


# In[28]:


train_documents, test_documents, train_labels, test_labels =     train_test_split(all_documents, all_targets, random_state=7)
    
with open(os.path.join('20news_train.vw'), 'w') as vw_train_data:
    for text, target in zip(train_documents, train_labels):
        vw_train_data.write(to_vw_format(text, target))
with open(os.path.join('20news_test.vw'), 'w') as vw_test_data:
    for text in test_documents:
        vw_test_data.write(to_vw_format(text))


# In[29]:


get_ipython().system('vw -d 20news_train.vw --loss_function hinge -f 20news_model.vw')


# In[30]:


get_ipython().system('vw -i 20news_model.vw -t -d 20news_test.vw -p 20news_test_predictions.txt')


# In[31]:


with open(os.path.join('20news_test_predictions.txt')) as pred_file:
    test_prediction = [float(label) 
                             for label in pred_file.readlines()]

auc = roc_auc_score(test_labels, test_prediction)
roc_curve = roc_curve(test_labels, test_prediction)

with plt.xkcd():
    plt.plot(roc_curve[0], roc_curve[1]);
    plt.plot([0,1], [0,1])
    plt.xlabel('FPR'); 
    plt.ylabel('TPR'); 
    plt.title('test AUC = %f' % (auc)); 
    plt.axis([-0.05,1.05,-0.05,1.05]);


# In[32]:


all_documents = newsgroups['data']
topic_encoder = LabelEncoder()
all_targets_mult = topic_encoder.fit_transform(newsgroups['target']) + 1


# In[33]:


train_documents, test_documents, train_labels_mult, test_labels_mult =     train_test_split(all_documents, all_targets_mult, random_state=7)
    
with open(os.path.join('20news_train_mult.vw'), 'w') as vw_train_data:
    for text, target in zip(train_documents, train_labels_mult):
        vw_train_data.write(to_vw_format(text, target))
with open(os.path.join('20news_test_mult.vw'), 'w') as vw_test_data:
    for text in test_documents:
        vw_test_data.write(to_vw_format(text))


# In[34]:


get_ipython().run_cell_magic('time', '', '!vw --oaa 20 20news_train_mult.vw -f 20news_model_mult.vw --loss_function=hinge')


# In[35]:


get_ipython().system('vw -i 20news_model_mult.vw -t -d 20news_test_mult.vw -p 20news_test_predictions_mult.txt')


# In[36]:


with open('20news_test_predictions_mult.txt') as pred_file:
    test_prediction_mult = [float(label) for label in pred_file.readlines()]


# In[37]:


accuracy_score(test_labels_mult, test_prediction_mult)


# In[38]:


M = confusion_matrix(test_labels_mult, test_prediction_mult)
for i in np.where(M[0,:] > 0)[0][1:]:
    print(newsgroups['target_names'][i], M[0,i])


# In[39]:


# path_to_movies = os.path.expanduser('/Users/y.kashnitsky/Documnents/imdb_reviews')
# reviews_train = load_files(os.path.join(path_to_movies, 'train'))
with open(os.path.join(PATH_TO_ALL_DATA, 'reviews_train.pkl'), 'rb') as reviews_train_pkl:
    reviews_train = pickle.load(reviews_train_pkl)
text_train, y_train = reviews_train.data, reviews_train.target


# In[40]:


print("Number of documents in training data: %d" % len(text_train))
print(np.bincount(y_train))


# In[41]:


# reviews_test = load_files(os.path.join(path_to_movies, 'test'))
with open(os.path.join(PATH_TO_ALL_DATA, 'reviews_test.pkl'), 'rb') as reviews_test_pkl:
    reviews_test = pickle.load(reviews_test_pkl)
text_test, y_test = reviews_test.data, reviews_train.target
print("Number of documents in test data: %d" % len(text_test))
print(np.bincount(y_test))


# In[42]:


text_train[0]


# In[43]:


y_train[0] # good review


# In[44]:


text_train[1]


# In[45]:


y_train[1] # bad review


# In[46]:


to_vw_format(str(text_train[1]), 1 if y_train[0] == 1 else -1)


# In[47]:


train_share = int(0.7 * len(text_train))
train, valid = text_train[:train_share], text_train[train_share:]
train_labels, valid_labels = y_train[:train_share], y_train[train_share:]


# In[48]:


len(train_labels), len(valid_labels)


# In[49]:


with open('movie_reviews_train.vw', 'w') as vw_train_data:
    for text, target in zip(train, train_labels):
        vw_train_data.write(to_vw_format(str(text), 1 if target == 1 else -1))
with open('movie_reviews_valid.vw', 'w') as vw_train_data:
    for text, target in zip(valid, valid_labels):
        vw_train_data.write(to_vw_format(str(text), 1 if target == 1 else -1))
with open('movie_reviews_test.vw', 'w') as vw_test_data:
    for text in text_test:
        vw_test_data.write(to_vw_format(str(text)))


# In[50]:


get_ipython().system('head -2 movie_reviews_train.vw')


# In[51]:


get_ipython().system('head -2 movie_reviews_valid.vw')


# In[52]:


get_ipython().system('head -2 movie_reviews_test.vw')


# In[53]:


get_ipython().system('vw -d movie_reviews_train.vw --loss_function hinge -f movie_reviews_model.vw --quiet')


# In[54]:


get_ipython().system('vw -i movie_reviews_model.vw -t -d movie_reviews_valid.vw -p movie_valid_pred.txt --quiet')


# In[55]:


with open('movie_valid_pred.txt') as pred_file:
    valid_prediction = [float(label) 
                             for label in pred_file.readlines()]
print("Accuracy: {}".format(round(accuracy_score(valid_labels, 
               [int(pred_prob > 0) for pred_prob in valid_prediction]), 3)))
print("AUC: {}".format(round(roc_auc_score(valid_labels, valid_prediction), 3)))


# In[56]:


get_ipython().system('vw -i movie_reviews_model.vw -t -d movie_reviews_test.vw -p movie_test_pred.txt --quiet')


# In[57]:


with open('movie_test_pred.txt') as pred_file:
    test_prediction = [float(label) 
                             for label in pred_file.readlines()]
print("Accuracy: {}".format(round(accuracy_score(y_test, 
               [int(pred_prob > 0) for pred_prob in test_prediction]), 3)))
print("AUC: {}".format(round(roc_auc_score(y_test, test_prediction), 3)))


# In[58]:


get_ipython().system('vw -d movie_reviews_train.vw --loss_function hinge --ngram 2 -f movie_reviews_model2.vw --quiet')


# In[59]:


get_ipython().system('vw -i movie_reviews_model2.vw -t -d movie_reviews_valid.vw -p movie_valid_pred2.txt --quiet')


# In[60]:


with open('movie_valid_pred2.txt') as pred_file:
    valid_prediction = [float(label) 
                             for label in pred_file.readlines()]
print("Accuracy: {}".format(round(accuracy_score(valid_labels, 
               [int(pred_prob > 0) for pred_prob in valid_prediction]), 3)))
print("AUC: {}".format(round(roc_auc_score(valid_labels, valid_prediction), 3)))


# In[61]:


get_ipython().system('vw -i movie_reviews_model2.vw -t -d movie_reviews_test.vw -p movie_test_pred2.txt --quiet')


# In[62]:


with open('movie_test_pred2.txt') as pred_file:
    test_prediction2 = [float(label) 
                             for label in pred_file.readlines()]
print("Accuracy: {}".format(round(accuracy_score(y_test, 
               [int(pred_prob > 0) for pred_prob in test_prediction2]), 3)))
print("AUC: {}".format(round(roc_auc_score(y_test, test_prediction2), 3)))


# In[63]:


get_ipython().system('head -3 $PATH_TO_ALL_DATA/stackoverflow_sample.vw')


# In[64]:


# !du -hs $PATH_TO_STACKOVERFLOW_DATA/stackoverflow_*.vw
# 4,7G stackoverflow_10mln.vw
# 1,6G stackoverflow_test.vw
# 3,1G stackoverflow_train.vw


# In[65]:


# %%time
# !vw --oaa 10 -d $PATH_TO_STACKOVERFLOW_DATA/stackoverflow_train.vw \
# -f vw_model1_10mln.vw -b 28 --random_seed 17 --quiet
# CPU times: user 567 ms, sys: 212 ms, total: 778 ms
# Wall time: 36.5 s


# In[66]:


# %%time
# !vw -t -i vw_model1_10mln.vw -d $PATH_TO_STACKOVERFLOW_DATA/stackoverflow_test.vw \
# -p vw_test_pred.csv --random_seed 17 --quiet
# CPU times: user 222 ms, sys: 86.4 ms, total: 308 ms
# Wall time: 14.4 s


# In[67]:


vw_pred = np.loadtxt(os.path.join(PATH_TO_ALL_DATA, 'vw_test_pred.csv'))
test_labels = np.loadtxt(os.path.join(PATH_TO_ALL_DATA, 'stackoverflow_test_labels.txt'))
accuracy_score(test_labels, vw_pred)


# In[68]:


train_texts = pd.read_csv('../input/spooky-author-identification/train.zip', index_col='id')
test_texts = pd.read_csv('../input/spooky-author-identification/test.zip', index_col='id')
sample_sub = pd.read_csv('../input/spooky-author-identification/sample_submission.zip', 
                         index_col='id')


# In[69]:


author_code = {"EAP": 1, "MWS": 2,"HPL": 3}


# In[70]:


train_texts["author_code"] = train_texts["author"].map(author_code)


# In[71]:


train_texts_part, valid_texts = train_test_split(train_texts, test_size=0.3, random_state=17, 
                                                 stratify=train_texts["author_code"], shuffle=True)


# In[72]:


train_texts_part.shape[0], valid_texts.shape[0]


# In[73]:


def to_vw_only_text(out_vw, df, is_train=True):
    with open(out_vw, "w") as out:
        for i in range(df.shape[0]):
            
            if is_train:
                target = df["author_code"].iloc[i]
            else:
                # for the test set we can pick any target label – we don't need it actually
                target = 1 
                       
            # remove special VW symbols
            text = df["text"].iloc[i].strip().replace('|', '').replace(':', '').lower() 
            # leave only words of 3 and more chars
            words = re.findall("\w{3,}", text) 
            new_text = " ".join(words) 

            s = "{} |text {}\n".format(target, new_text)

            out.write(s)    


# In[74]:


to_vw_only_text("train_part_only_text.vw", train_texts_part)


# In[75]:


get_ipython().system('head -2 train_part_only_text.vw')


# In[76]:


to_vw_only_text("valid_only_text.vw", valid_texts)


# In[77]:


get_ipython().system('head -2 valid_only_text.vw')


# In[78]:


to_vw_only_text("train_only_text.vw", train_texts)


# In[79]:


get_ipython().system('head -2 train_only_text.vw')


# In[80]:


to_vw_only_text("test_only_text.vw", test_texts, is_train=False)


# In[81]:


get_ipython().system('head -2 test_only_text.vw')


# In[82]:


get_ipython().system('vw --oaa 3 train_part_only_text.vw -f model_only_text_part.vw -b 28 --random_seed 17 --loss_function logistic --ngram 2 --passes 10 -k -c')


# In[83]:


get_ipython().system('vw -i model_only_text_part.vw -t -d valid_only_text.vw -p valid_pred1.txt --random_seed 17 -r valid_prob1.txt')


# In[84]:


def evaluate_vw_prediction(path_to_vw_pred_probs, is_test=False, target=None, write_submission=False,
                          submission_file=None, test_index=test_texts.index, columns=['EAP', 'MWS', 'HPL']):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z)) 
    
    with open(path_to_vw_pred_probs) as pred_file:
        pred_probs =  np.array([[float(pair.split(':')[1]) for pair in line.strip().split()] 
                         for line in pred_file.readlines()])
        pred_probs  = sigmoid(pred_probs)
        
        if target is not None and not is_test:
            print(log_loss(target, pred_probs))
        
        if write_submission and submission_file is not None:
            subm_df = pd.DataFrame(pred_probs, columns=columns)
            subm_df.index = test_index
            subm_df.to_csv(submission_file)


# In[85]:


evaluate_vw_prediction('valid_prob1.txt', 
                       target=valid_texts['author_code'])


# In[86]:


get_ipython().run_cell_magic('time', '', '!vw --oaa 3 train_only_text.vw -f model_only_text.vw -b 28 --random_seed 17 \\\n--loss_function logistic --ngram 2 --passes 10 -k -c --quiet')


# In[87]:


get_ipython().system('vw -i model_only_text.vw -t -d test_only_text.vw -p test_pred1.txt --random_seed 17 -r test_prob1.txt --quiet')


# In[88]:


evaluate_vw_prediction('test_prob1.txt', 
                       is_test=True, write_submission=True,
                       submission_file='submission1_only_text.csv')


# In[89]:


get_ipython().system('head -3 submission1_only_text.csv')


# In[90]:


max_words_in_text = train_texts['text'].apply(lambda text: len(re.findall("\w{3,}", text.strip()))).max()
max_unique_words_in_text = train_texts['text'].apply(lambda text: len(set(re.findall("\w{3,}", text.strip())))).max()
max_aver_word_len_in_text = train_texts['text'].apply(lambda text: 
                                                      sum([len(w) for w in re.findall("\w{3,}", text.strip())]) / 
                                                      len(re.findall("\w{3,}", text.strip()))).max()


# In[91]:


max_words_in_text, max_unique_words_in_text, max_aver_word_len_in_text


# In[92]:


def to_vw_text_and_some_features(out_vw, df, is_train=True):
    with open(out_vw, "w") as out:
        for i in range(df.shape[0]):
            
            if is_train:
                target = df["author_code"].iloc[i]
            else:
                # for the test set we can pick any target label – we don't need it actually
                target = 1 
                       
            # remove special VW symbols
            text = df["text"].iloc[i].strip().replace('|', '').replace(':', '').lower() 
            # leave only words of 3 and more chars
            words = re.findall("\w{3,}", text) 
            new_text = " ".join(words)    
            
            num_words = round(len(words) / max_words_in_text, 4)
            num_uniq_words = round(len(set(words)) / max_unique_words_in_text, 4)
            aver_word_len = round(sum([len(w) for w in words]) / len(words) / max_aver_word_len_in_text, 4)

            features = [num_words, num_uniq_words, aver_word_len] 
            features_vw = ' '.join(['{}:{}'.format(i[0], i[1]) for i in zip(range(len(features)), features)])
            s = "{} |text {} |num {}\n".format(target, new_text, features_vw)

            out.write(s)   
 


# In[93]:


to_vw_text_and_some_features("train_part_text_feat.vw", train_texts_part)


# In[94]:


get_ipython().system('head -2 train_part_text_feat.vw')


# In[95]:


to_vw_text_and_some_features("valid_text_feat.vw", valid_texts)


# In[96]:


to_vw_text_and_some_features("train_text_feat.vw", train_texts)


# In[97]:


to_vw_text_and_some_features("test_text_feat.vw", test_texts, is_train=False)


# In[98]:


get_ipython().run_cell_magic('time', '', '!vw --oaa 3 train_part_text_feat.vw -f model_text_feat_part.vw -b 28 --random_seed 17 \\\n--loss_function logistic --ngram 2 --passes 10 -k -c --quiet')


# In[99]:


get_ipython().system('vw -i model_text_feat_part.vw -t -d valid_text_feat.vw -p valid_pred2.txt --random_seed 17 -r valid_prob2.txt --quiet')


# In[100]:


evaluate_vw_prediction('valid_prob2.txt', 
                       target=valid_texts['author_code'])


# In[101]:


get_ipython().run_cell_magic('time', '', '!vw --oaa 3 train_text_feat.vw -f model_text_feat.vw -b 28 --random_seed 17 \\\n--loss_function logistic --ngram 2 --passes 10 -k -c --quiet')


# In[102]:


get_ipython().system('vw -i model_text_feat.vw -t -d test_text_feat.vw -p test_pred2.txt --random_seed 17 -r test_prob2.txt --quiet')


# In[103]:


evaluate_vw_prediction('test_prob2.txt', 
                       is_test=True, write_submission=True,
                       submission_file='submission2_text_feat.csv')


# In[104]:


def validate_submission_local_and_lb_mix(local_score, public_lb_score, local_size=5874, public_lb_size=2517):
    return 1. / (local_size + public_lb_size) * (local_size * local_score +
                                                public_lb_size * public_lb_score)


# In[105]:


# first submission
validate_submission_local_and_lb_mix(local_score=.47951, public_lb_score=.43187)


# In[106]:


# second submission
validate_submission_local_and_lb_mix(local_score=.469, 
                                      public_lb_score=.43267)

