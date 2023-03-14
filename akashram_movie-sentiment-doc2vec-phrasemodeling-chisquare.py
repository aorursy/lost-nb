#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.tsv', sep = '\t')
test = pd.read_csv('../input/test.tsv', sep = '\t')
sub = pd.read_csv('../input/sampleSubmission.csv')


# In[ ]:


train.head()


# In[ ]:


train['Phrase'] = train['Phrase'].str.replace(r'\'s', '')
train['Phrase'] = train['Phrase'].str.replace(r'.', '')
train['Phrase'] = train['Phrase'].str.replace(r',', '')
train['Phrase'] = train['Phrase'].str.replace(r'does n\'t', 'does not')
train['Phrase'] = train['Phrase'].str.replace(r'is n\'t', 'is not')
train['Phrase'] = train['Phrase'].str.replace(r'were n\'t', 'were not')
train['Phrase'] = train['Phrase'].str.replace(r'are n\'t', 'are not')
train['Phrase'] = train['Phrase'].str.replace(r'had n\'t', 'had not')
train['Phrase'] = train['Phrase'].str.replace(r'have n\'t', 'have not')
train['Phrase'] = train['Phrase'].str.replace(r'would n\'t', 'would not')
train['Phrase'] = train['Phrase'].str.replace(r'ca n\'t', 'can not')
train['Phrase'] = train['Phrase'].str.replace(r'could n\'t', 'could not')
train['Phrase'] = train['Phrase'].str.replace(r'must n\'t', 'must not')
train['Phrase'] = train['Phrase'].str.replace(r'should n\'t', 'should not')
train['Phrase'] = train['Phrase'].str.replace(r'wo n\'t', 'will not')
train['Phrase'] = train['Phrase'].str.replace(r'n\'t', 'not')


# In[ ]:


test['Phrase'] = test['Phrase'].str.replace(r'\'s', '')
test['Phrase'] = test['Phrase'].str.replace(r'.', '')
test['Phrase'] = test['Phrase'].str.replace(r',', '')
test['Phrase'] = test['Phrase'].str.replace(r'does n\'t', 'does not')
test['Phrase'] = test['Phrase'].str.replace(r'is n\'t', 'is not')
test['Phrase'] = test['Phrase'].str.replace(r'were n\'t', 'were not')
test['Phrase'] = test['Phrase'].str.replace(r'are n\'t', 'are not')
test['Phrase'] = test['Phrase'].str.replace(r'had n\'t', 'had not')
test['Phrase'] = test['Phrase'].str.replace(r'have n\'t', 'have not')
test['Phrase'] = test['Phrase'].str.replace(r'would n\'t', 'would not')
test['Phrase'] = test['Phrase'].str.replace(r'ca n\'t', 'can not')
test['Phrase'] = test['Phrase'].str.replace(r'could n\'t', 'could not')
test['Phrase'] = test['Phrase'].str.replace(r'must n\'t', 'must not')
test['Phrase'] = test['Phrase'].str.replace(r'should n\'t', 'should not')
test['Phrase'] = test['Phrase'].str.replace(r'wo n\'t', 'will not')
test['Phrase'] = test['Phrase'].str.replace(r'n\'t', 'not')


# In[ ]:


x_train = train['Phrase']

y_train = train['Sentiment']


# In[ ]:


from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import multiprocessing
from sklearn import utils

def labelize_phrases_ug(Phrase, label):
    result = []
    prefix = label
    for i, t in zip(Phrase.index, Phrase):
        result.append(LabeledSentence(t.split(), [prefix + '_%s' % i]))
    return result
  
all_x = pd.concat([x_train])
all_x_w2v = labelize_phrases_ug(all_x, 'all')


# In[ ]:


all_x_w2v


# In[ ]:


x_test = test['Phrase']


# In[ ]:


from sklearn.linear_model import LogisticRegression

cores = multiprocessing.cpu_count()
model_ug_dbow = Doc2Vec(dm=0, size=100, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_dbow.build_vocab([x for x in tqdm(all_x_w2v)])

for epoch in range(30):
    model_ug_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_dbow.alpha -= 0.002
    model_ug_dbow.min_alpha = model_ug_dbow.alpha
    
def get_vectors(model, corpus, size):
    vecs = np.zeros((len(corpus), size))
    n = 0
    for i in corpus.index:
        prefix = 'all_' + str(i)
        vecs[n] = model.docvecs[prefix]
        n += 1
    return vecs
  
train_vecs_dbow = get_vectors(model_ug_dbow, x_train, 100)

clf = LogisticRegression()
clf.fit(train_vecs_dbow, y_train)


# In[ ]:


test_vecs_dbow = get_vectors(model_ug_dbow, x_test, 100)


# In[ ]:


dbow_prediction = clf.predict(test_vecs_dbow)


# In[ ]:


lr_dbow_sub = pd.DataFrame({"PhraseId": test['PhraseId'], "Sentiment" : dbow_prediction})

lr_dbow_sub.to_csv("LR_DBOW_submission.csv",index=False)


# In[ ]:


cores = multiprocessing.cpu_count()
model_ug_dmc = Doc2Vec(dm=1, dm_concat=1, size=100, window=2, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_dmc.build_vocab([x for x in tqdm(all_x_w2v)])

for epoch in range(30):
    model_ug_dmc.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_dmc.alpha -= 0.002
    model_ug_dmc.min_alpha = model_ug_dmc.alpha
   
train_vecs_dmc = get_vectors(model_ug_dmc, x_train, 100)

clf = LogisticRegression()
clf.fit(train_vecs_dmc, y_train)


# In[ ]:


test_vecs_dm = get_vectors(model_ug_dbow, x_test, 100)

dm_prediction = clf.predict(test_vecs_dm)


# In[ ]:


lr_dm_sub = pd.DataFrame({"PhraseId": test['PhraseId'], "Sentiment" : dm_prediction})

lr_dm_sub.to_csv("LR_DM_submission.csv",index=False)


# In[ ]:


model_ug_dmc.most_similar('darkly')


# In[ ]:


cores = multiprocessing.cpu_count()
model_ug_dmm = Doc2Vec(dm=1, dm_mean=1, size=100, window=4, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_dmm.build_vocab([x for x in tqdm(all_x_w2v)])

for epoch in range(30):
    model_ug_dmm.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_dmm.alpha -= 0.002
    model_ug_dmm.min_alpha = model_ug_dmm.alpha
    
train_vecs_dmm = get_vectors(model_ug_dmm, x_train, 100)

clf = LogisticRegression()
clf.fit(train_vecs_dmm, y_train)


# In[ ]:


test_vecs_dmm = get_vectors(model_ug_dmm, x_test, 100)

dmm_prediction = clf.predict(test_vecs_dmm)


# In[ ]:


lr_dmm_sub = pd.DataFrame({"PhraseId": test['PhraseId'], "Sentiment" : dmm_prediction})

lr_dmm_sub.to_csv("LR_DMM_submission.csv",index=False)


# In[ ]:


def get_concat_vectors(model1,model2, corpus, size):
    vecs = np.zeros((len(corpus), size))
    n = 0
    for i in corpus.index:
        prefix = 'all_' + str(i)
        vecs[n] = np.append(model1.docvecs[prefix],model2.docvecs[prefix])
        n += 1
    return vecs

train_vecs_dbow_dmc = get_concat_vectors(model_ug_dbow, model_ug_dmc, x_train, 200)
clf = LogisticRegression()
clf.fit(train_vecs_dbow_dmc, y_train)


# In[ ]:


test_vecs_dbow_dmc = get_concat_vectors(model_ug_dbow, model_ug_dmc, x_test, 200)

dbow_dmc_prediction = clf.predict(test_vecs_dbow_dmc)


# In[ ]:


lr_dbow_dmc_sub = pd.DataFrame({"PhraseId": test['PhraseId'], "Sentiment" : dbow_dmc_prediction})

lr_dbow_dmc_sub.to_csv("lr_dbow_dmc_sub.csv",index=False)


# In[ ]:


train_vecs_dbow_dmm = get_concat_vectors(model_ug_dbow,model_ug_dmm, x_train, 200)

clf = LogisticRegression()
clf.fit(train_vecs_dbow_dmm, y_train)


# In[ ]:


test_vecs_dbow_dmm = get_concat_vectors(model_ug_dbow, model_ug_dmm, x_test, 200)

dbow_dmm_prediction = clf.predict(test_vecs_dbow_dmm)


# In[ ]:


lr_dbow_dmm_sub = pd.DataFrame({"PhraseId": test['PhraseId'], "Sentiment" : dbow_dmm_prediction})

lr_dbow_dmm_sub.to_csv("lr_dbow_dmm_sub.csv",index=False)


# In[ ]:


from gensim.models.phrases import Phrases, Phraser

tokenized_train = [t.split() for t in x_train]
phrases = Phrases(tokenized_train)
bigram = Phraser(phrases)


# In[ ]:


bigram[x_train[15].split()]


# In[ ]:


def labelize_Phrase_bg(Phrase,label):
    result = []
    prefix = label
    for i, t in zip(Phrase.index, Phrase):
        result.append(LabeledSentence(bigram[t.split()], [prefix + '_%s' % i]))
    return result
  
all_x = pd.concat([x_train])
all_x_w2v_bg = labelize_Phrase_bg(all_x, 'all')


# In[ ]:


cores = multiprocessing.cpu_count()
model_bg_dbow = Doc2Vec(dm=0, size=100, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_bg_dbow.build_vocab([x for x in tqdm(all_x_w2v_bg)])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for epoch in range(30):\n    model_bg_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v_bg)]), total_examples=len(all_x_w2v_bg), epochs=1)\n    model_bg_dbow.alpha -= 0.002\n    model_bg_dbow.min_alpha = model_bg_dbow.alpha')


# In[ ]:


train_vecs_dbow_bg = get_vectors(model_bg_dbow, x_train, 100)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf = LogisticRegression()\nclf.fit(train_vecs_dbow_bg, y_train)')


# In[ ]:


test_vecs_dbow_bg = get_vectors(model_bg_dbow, x_test, 100)


# In[ ]:


dbow_bigram_prediction = clf.predict(test_vecs_dbow_bg)


# In[ ]:


lr_dbow_bigram_sub = pd.DataFrame({"PhraseId": test['PhraseId'], "Sentiment" : dbow_bigram_prediction})

lr_dbow_bigram_sub.to_csv("lr_dbow_bigram_sub.csv",index=False)


# In[ ]:


cores = multiprocessing.cpu_count()
model_bg_dmc = Doc2Vec(dm=1, dm_concat=1, size=100, window=2, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_bg_dmc.build_vocab([x for x in tqdm(all_x_w2v_bg)])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for epoch in range(30):\n    model_bg_dmc.train(utils.shuffle([x for x in tqdm(all_x_w2v_bg)]), total_examples=len(all_x_w2v_bg), epochs=1)\n    model_bg_dmc.alpha -= 0.002\n    model_bg_dmc.min_alpha = model_bg_dmc.alpha')


# In[ ]:


model_bg_dmc.most_similar('movie')


# In[ ]:


train_vecs_dmc_bg = get_vectors(model_bg_dmc, x_train, 100)
test_vecs_dmc_bg = get_vectors(model_bg_dmc, x_test, 100)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf = LogisticRegression()\nclf.fit(train_vecs_dmc_bg, y_train)')


# In[ ]:


test_vecs_dmc_bg = get_vectors(model_bg_dmc, x_test, 100)

dmc_bigram_prediction = clf.predict(test_vecs_dmc_bg)

lr_dmc_bigram_sub = pd.DataFrame({"PhraseId": test['PhraseId'], "Sentiment" : dmc_bigram_prediction})

lr_dmc_bigram_sub.to_csv("lr_dmc_bigram_sub.csv",index=False)


# In[ ]:


cores = multiprocessing.cpu_count()
model_bg_dmm = Doc2Vec(dm=1, dm_mean=1, size=100, window=4, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_bg_dmm.build_vocab([x for x in tqdm(all_x_w2v_bg)])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for epoch in range(30):\n    model_bg_dmm.train(utils.shuffle([x for x in tqdm(all_x_w2v_bg)]), total_examples=len(all_x_w2v_bg), epochs=1)\n    model_bg_dmm.alpha -= 0.002\n    model_bg_dmm.min_alpha = model_bg_dmm.alpha')


# In[ ]:


train_vecs_dmm_bg = get_vectors(model_bg_dmm, x_train, 100)
test_vecs_dmm_bg = get_vectors(model_bg_dmm, x_test, 100)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf = LogisticRegression()\nclf.fit(train_vecs_dmm_bg, y_train)')


# In[ ]:


test_vecs_dmm_bg = get_vectors(model_bg_dmm, x_test, 100)

dmm_bigram_prediction = clf.predict(test_vecs_dmm_bg)

lr_dmm_bigram_sub = pd.DataFrame({"PhraseId": test['PhraseId'], "Sentiment" : dmm_bigram_prediction})

lr_dmm_bigram_sub.to_csv("lr_dmm_bigram_sub.csv",index=False)


# In[ ]:


train_vecs_dbow_dmc_bg = get_concat_vectors(model_bg_dbow, model_bg_dmc, x_train, 200)
test_vecs_dbow_dmc_bg = get_concat_vectors(model_bg_dbow, model_bg_dmc, x_test, 200)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf = LogisticRegression()\nclf.fit(train_vecs_dbow_dmc_bg, y_train)')


# In[ ]:


dbow_dmc_bigram_prediction = clf.predict(test_vecs_dbow_dmc_bg)

lr_dbow_dmc_bigram_sub = pd.DataFrame({"PhraseId": test['PhraseId'], "Sentiment" : dbow_dmc_bigram_prediction})

lr_dbow_dmc_bigram_sub.to_csv("lr_dbow_dmc_bigram_sub.csv",index=False)


# In[ ]:


train_vecs_dbow_dmm_bg = get_concat_vectors(model_bg_dbow,model_bg_dmm, x_train, 200)
test_vecs_dbow_dmm_bg = get_concat_vectors(model_bg_dbow,model_bg_dmm, x_test, 200)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf = LogisticRegression()\nclf.fit(train_vecs_dbow_dmm_bg, y_train)')


# In[ ]:


dbow_dmm_bigram_prediction = clf.predict(test_vecs_dbow_dmm_bg)

lr_dbow_dmm_bigram_sub = pd.DataFrame({"PhraseId": test['PhraseId'], "Sentiment" : dbow_dmm_bigram_prediction})

lr_dbow_dmm_bigram_sub.to_csv("lr_dbow_dmm_bigram_sub.csv",index=False)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tg_phrases = Phrases(bigram[tokenized_train])\ntrigram = Phraser(tg_phrases)')


# In[ ]:


trigram[bigram[x_train[15].split()]]


# In[ ]:


def labelize_Phrase_tg(Phrase,label):
    result = []
    prefix = label
    for i, t in zip(Phrase.index, Phrase):
        result.append(LabeledSentence(trigram[bigram[t.split()]], [prefix + '_%s' % i]))
    return result

all_x = pd.concat([x_train])
all_x_w2v_tg = labelize_Phrase_tg(all_x, 'all')


# In[ ]:


model_tg_dbow = Doc2Vec(dm=0, size=100, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_tg_dbow.build_vocab([x for x in tqdm(all_x_w2v_tg)])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for epoch in range(30):\n    model_tg_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v_tg)]), total_examples=len(all_x_w2v_tg), epochs=1)\n    model_tg_dbow.alpha -= 0.002\n    model_tg_dbow.min_alpha = model_tg_dbow.alpha')


# In[ ]:


train_vecs_dbow_tg = get_vectors(model_tg_dbow, x_train, 100)
test_vecs_dbow_tg = get_vectors(model_tg_dbow, x_test, 100)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf = LogisticRegression()\nclf.fit(train_vecs_dbow_tg, y_train)')


# In[ ]:


test_vecs_dbow_tg = get_vectors(model_tg_dbow, x_test, 100)

dbow_trigram_prediction = clf.predict(test_vecs_dbow_tg)

lr_dbow_trigram_sub = pd.DataFrame({"PhraseId": test['PhraseId'], "Sentiment" : dbow_trigram_prediction})

lr_dbow_trigram_sub.to_csv("lr_dbow_trigram_sub.csv",index=False)


# In[ ]:


cores = multiprocessing.cpu_count()
model_tg_dmc = Doc2Vec(dm=1, dm_concat=1, size=100, window=2, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_tg_dmc.build_vocab([x for x in tqdm(all_x_w2v_tg)])


# In[ ]:


get_ipython().run_cell_magic('time', '', ' for epoch  in  range(30):\n    model_tg_dmc.train(utils.shuffle([x for x in tqdm(all_x_w2v_tg)]), total_examples=len(all_x_w2v_tg), epochs=1)\n    model_tg_dmc.alpha -= 0.002\n    model_tg_dmc.min_alpha = model_tg_dmc.alpha')


# In[ ]:


train_vecs_dmc_tg = get_vectors(model_tg_dmc, x_train, 100)
test_vecs_dmc_tg = get_vectors(model_tg_dmc, x_test, 100)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf = LogisticRegression()\nclf.fit(train_vecs_dmc_tg, y_train)')


# In[ ]:


test_vecs_dmc_tg = get_vectors(model_tg_dmc, x_test, 100)

dmc_trigram_prediction = clf.predict(test_vecs_dmc_tg)

lr_dmc_trigram_sub = pd.DataFrame({"PhraseId": test['PhraseId'], "Sentiment" : dmc_trigram_prediction})

lr_dmc_trigram_sub.to_csv("lr_dmc_trigram_sub.csv",index=False)


# In[ ]:


cores = multiprocessing.cpu_count()
model_tg_dmm = Doc2Vec(dm=1, dm_mean=1, size=100, window=4, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_tg_dmm.build_vocab([x for x in tqdm(all_x_w2v_tg)])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for epoch in range(30):\n    model_tg_dmm.train(utils.shuffle([x for x in tqdm(all_x_w2v_tg)]), total_examples=len(all_x_w2v_tg), epochs=1)\n    model_tg_dmm.alpha -= 0.002\n    model_tg_dmm.min_alpha = model_tg_dmm.alpha')


# In[ ]:


train_vecs_dmm_tg = get_vectors(model_tg_dmm, x_train, 100)
test_vecs_dmm_tg = get_vectors(model_tg_dmm, x_test, 100)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf = LogisticRegression()\nclf.fit(train_vecs_dmm_tg, y_train)')


# In[ ]:


dmm_trigram_prediction = clf.predict(test_vecs_dmm_tg)

lr_dmm_trigram_sub = pd.DataFrame({"PhraseId": test['PhraseId'], "Sentiment" : dmm_trigram_prediction})

lr_dmm_trigram_sub.to_csv("lr_dmm_trigram_sub.csv",index=False)


# In[ ]:


train_vecs_dbow_dmc_tg = get_concat_vectors(model_tg_dbow, model_tg_dmc, x_train, 200)
test_vecs_dbow_dmc_tg = get_concat_vectors(model_tg_dbow, model_tg_dmc, x_test, 200)   


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf = LogisticRegression()\nclf.fit(train_vecs_dbow_dmc_tg, y_train)')


# In[ ]:


dbow_dmc_trigram_prediction = clf.predict(test_vecs_dbow_dmc_tg)

lr_dbow_dmc_trigram_sub = pd.DataFrame({"PhraseId": test['PhraseId'], "Sentiment" : dbow_dmc_trigram_prediction})

lr_dbow_dmc_trigram_sub.to_csv("lr_dbow_dmc_trigram_sub.csv",index=False)


# In[ ]:


train_vecs_dbow_dmm_tg = get_concat_vectors(model_tg_dbow, model_tg_dmm, x_train, 200)
test_vecs_dbow_dmm_tg = get_concat_vectors(model_tg_dbow, model_tg_dmm, x_test, 200)   


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf = LogisticRegression()\nclf.fit(train_vecs_dbow_dmm_tg, y_train)')


# In[ ]:


dbow_dmm_trigram_prediction = clf.predict(test_vecs_dbow_dmm_tg)

lr_dbow_dmm_trigram_sub = pd.DataFrame({"PhraseId": test['PhraseId'], "Sentiment" : dbow_dmm_trigram_prediction})

lr_dbow_dmm_trigram_sub.to_csv("lr_dbow_dmm_trigram_sub.csv",index=False)


# In[ ]:


from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer

tvec = TfidfVectorizer(max_features=100000,ngram_range=(1, 3))
x_train_tfidf = tvec.fit_transform(x_train)
x_test_tfidf = tvec.transform(x_test)
chi2score = chi2(x_train_tfidf, y_train)[0]

plt.figure(figsize=(15,10))
wscores = zip(tvec.get_feature_names(), chi2score)
wchi2 = sorted(wscores, key=lambda x:x[1])
topchi2 = list(zip(*wchi2[-20:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
plt.barh(x,topchi2[1], align='center', alpha=0.2)
plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8)
plt.yticks(x, labels)
plt.xlabel('$\chi^2$')

