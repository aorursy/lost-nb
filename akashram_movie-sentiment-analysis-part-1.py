#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("../input/train.tsv", sep = '\t')
test = pd.read_csv("../input/test.tsv", sep = '\t')
sub = pd.read_csv("../input/sampleSubmission.csv")




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




y_train = train['Sentiment']




#train = train.drop('Sentiment', axis=1)
#train




# Vectorization

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
cv.fit(train.Phrase)
X = cv.transform(train.Phrase)
X_test = cv.transform(test.Phrase)




pos = [3,4]
neg = [0,1]
neutral = 2

train_pos = train[train.Sentiment.isin(pos)]
train_pos = train_pos['Phrase']
train_neg = train[train.Sentiment.isin(neg)]
train_neg = train_neg['Phrase']




from wordcloud import WordCloud,STOPWORDS

def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if not word.startswith(',')
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print("Positive words")
wordcloud_draw(train_pos,'white')
print("Negative words")
wordcloud_draw(train_neg)




# Building Classifier




from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

X_train, X_val, y_train, y_val = train_test_split(
    X, y_train, train_size = 0.75
)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, lr.predict(X_val))))




y_train.shape




final_model = LogisticRegression(C=1)

final_model.fit(X_train, y_train)




len(cv.get_feature_names())




from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cv.fit(train.Phrase)




neg_doc_mat = train.Phrase[train['Sentiment'] == 0]
neg_document_matrix = cv.transform(train.Phrase[train['Sentiment'] == 0])




pos_doc_mat = train.Phrase[train['Sentiment'] == 4]
pos_document_matrix = cv.transform(pos_doc_mat)




neu_doc_mat = train.Phrase[train['Sentiment'] == 2]
neu_document_matrix = cv.transform(neu_doc_mat)




sneg_doc_mat = train.Phrase[train['Sentiment'] == 1]
sneg_document_matrix = cv.transform(sneg_doc_mat)




spos_doc_mat = train.Phrase[train['Sentiment'] == 3]
spos_document_matrix = cv.transform(spos_doc_mat)




get_ipython().run_cell_magic('time', '', 'neg_batches = np.linspace(0,156061,100).astype(int)\ni=0\nneg_tf = []\nwhile i < len(neg_batches)-1:\n    batch_result = np.sum(neg_document_matrix[neg_batches[i]:neg_batches[i+1]].toarray(),axis=0)\n    neg_tf.append(batch_result)\n    if (i % 10 == 0) | (i == len(neg_batches)-2):\n        print(neg_batches[i+1],"entries\' term frequency calculated")\n    i += 1')




get_ipython().run_cell_magic('time', '', 'pos_batches = np.linspace(0,156061,100).astype(int)\ni=0\npos_tf = []\nwhile i < len(pos_batches)-1:\n    batch_result = np.sum(pos_document_matrix[pos_batches[i]:pos_batches[i+1]].toarray(),axis=0)\n    pos_tf.append(batch_result)\n    if (i % 10 == 0) | (i == len(pos_batches)-2):\n        print(pos_batches[i+1],"entries\' term frequency calculated")\n    i += 1')




get_ipython().run_cell_magic('time', '', 'neu_batches = np.linspace(0,156061,100).astype(int)\ni=0\nneu_tf = []\nwhile i < len(neu_batches)-1:\n    batch_result = np.sum(neu_document_matrix[neu_batches[i]:neu_batches[i+1]].toarray(),axis=0)\n    neu_tf.append(batch_result)\n    if (i % 10 == 0) | (i == len(neu_batches)-2):\n        print(neu_batches[i+1],"entries\' term frequency calculated")\n    i += 1')




get_ipython().run_cell_magic('time', '', 'sneg_batches = np.linspace(0,156061,100).astype(int)\ni=0\nsneg_tf = []\nwhile i < len(sneg_batches)-1:\n    batch_result = np.sum(sneg_document_matrix[sneg_batches[i]:sneg_batches[i+1]].toarray(),axis=0)\n    sneg_tf.append(batch_result)\n    if (i % 10 == 0) | (i == len(sneg_batches)-2):\n        print(sneg_batches[i+1],"entries\' term frequency calculated")\n    i += 1')




get_ipython().run_cell_magic('time', '', 'spos_batches = np.linspace(0,156061,100).astype(int)\ni=0\nspos_tf = []\nwhile i < len(spos_batches)-1:\n    batch_result = np.sum(spos_document_matrix[spos_batches[i]:spos_batches[i+1]].toarray(),axis=0)\n    spos_tf.append(batch_result)\n    if (i % 10 == 0) | (i == len(spos_batches)-2):\n        print(spos_batches[i+1],"entries\' term frequency calculated")\n    i += 1')




neg = np.sum(neg_tf,axis=0)
pos = np.sum(pos_tf,axis=0)
neu = np.sum(neu_tf,axis=0)
sneg = np.sum(sneg_tf,axis=0)
spos = np.sum(spos_tf,axis=0)
term_freq_df = pd.DataFrame([neg,pos, neu, sneg, spos],columns=cv.get_feature_names()).transpose()
term_freq_df.head()




term_freq_df.columns = ['negative', 'positive', 'neutral', 'somewhat negative', 'somewhat positive']
term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive'] + term_freq_df['neutral'] + term_freq_df['somewhat negative'] + term_freq_df['somewhat positive']
term_freq_df.sort_values(by='total', ascending=False).iloc[:10]




type(term_freq_df)




neg_phrases = train[train.Sentiment == 0]
neg_string = []
for t in neg_phrases.Phrase:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')
from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()




pos_phrases = train[train.Sentiment == 4]
pos_string = []
for t in pos_phrases.Phrase:
    pos_string.append(t)
pos_string = pd.Series(pos_string).str.cat(sep=' ')
from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(pos_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()




y_pos = np.arange(500)
plt.figure(figsize=(10,8))
s = 1
expected_zipf = [term_freq_df.sort_values(by='total', ascending=False)['total'][0]/(i+1)**s for i in y_pos]
plt.bar(y_pos, term_freq_df.sort_values(by='total', ascending=False)['total'][:500], align='center', alpha=0.5)
plt.plot(y_pos, expected_zipf, color='r', linestyle='--',linewidth=2,alpha=0.5)
plt.ylabel('Frequency')
plt.title('Top 500 tokens in the Phrases')




from sklearn.feature_extraction.text import CountVectorizer
cvec = CountVectorizer(stop_words='english',max_features=10000)
cvec.fit(train.Phrase)




neg_document_matrix_nostop = cvec.transform(train.Phrase[train['Sentiment'] == 0])




get_ipython().run_cell_magic('time', '', 'neg_batches = np.linspace(0,156061,100).astype(int)\ni=0\nneg_tf = []\nwhile i < len(neg_batches)-1:\n    batch_result = np.sum(neg_document_matrix_nostop[neg_batches[i]:neg_batches[i+1]].toarray(),axis=0)\n    neg_tf.append(batch_result)\n    print(neg_batches[i+1],"entries\' term frequency calculated")\n    i += 1')




pos_document_matrix_nostop = cvec.transform(train.Phrase[train['Sentiment'] == 4])




get_ipython().run_cell_magic('time', '', 'pos_batches = np.linspace(0,156061,100).astype(int)\ni=0\npos_tf = []\nwhile i < len(pos_batches)-1:\n    batch_result = np.sum(pos_document_matrix_nostop[pos_batches[i]:pos_batches[i+1]].toarray(),axis=0)\n    pos_tf.append(batch_result)\n    print(pos_batches[i+1],"entries\' term frequency calculated")\n    i += 1')




neg = np.sum(neg_tf,axis=0)
pos = np.sum(pos_tf,axis=0)
term_freq_df2 = pd.DataFrame([neg,pos],columns=cvec.get_feature_names()).transpose()
term_freq_df2.columns = ['negative', 'positive']
term_freq_df2['total'] = term_freq_df2['negative'] + term_freq_df2['positive']
term_freq_df2.sort_values(by='total', ascending=False).iloc[:10]




y_pos = np.arange(50)
plt.figure(figsize=(12,10))
plt.bar(y_pos, term_freq_df2.sort_values(by='negative', ascending=False)['negative'][:50], align='center', alpha=0.5)
plt.xticks(y_pos, term_freq_df2.sort_values(by='negative', ascending=False)['negative'][:50].index,rotation='vertical')
plt.ylabel('Frequency')
plt.xlabel('Top 50 negative words')
plt.title('Top 50 tokens in negative sentiments')




y_pos = np.arange(50)
plt.figure(figsize=(12,10))
plt.bar(y_pos, term_freq_df2.sort_values(by='positive', ascending=False)['positive'][:50], align='center', alpha=0.5)
plt.xticks(y_pos, term_freq_df2.sort_values(by='positive', ascending=False)['positive'][:50].index,rotation='vertical')
plt.ylabel('Frequency')
plt.xlabel('Top 50 positive tokens')
plt.title('Top 50 tokens in positive sentiments')




import seaborn as sns
plt.figure(figsize=(8,6))
ax = sns.regplot(x="negative", y="positive",fit_reg=False, scatter_kws={'alpha':0.5},data=term_freq_df2)
plt.ylabel('Positive Frequency')
plt.xlabel('Negative Frequency')
plt.title('Negative Frequency vs Positive Frequency')




term_freq_df2['pos_rate'] = term_freq_df2['positive'] * 1./term_freq_df2['total']
term_freq_df2.sort_values(by='pos_rate', ascending=False).iloc[:10]




term_freq_df2['pos_freq_pct'] = term_freq_df2['positive'] * 1./term_freq_df2['positive'].sum()
term_freq_df2.sort_values(by='pos_freq_pct', ascending=False).iloc[:10]




from scipy.stats import hmean

term_freq_df2['pos_hmean'] = term_freq_df2.apply(lambda x: (hmean([x['pos_rate'], x['pos_freq_pct']])
                                                                   if x['pos_rate'] > 0 and x['pos_freq_pct'] > 0 
                                                                   else 0), axis=1)                                                        
term_freq_df2.sort_values(by='pos_hmean', ascending=False).iloc[:10]




import warnings
warnings.filterwarnings('ignore')

term_freq_df2['neg_rate'] = term_freq_df2['negative'] * 1./term_freq_df2['total']
term_freq_df2['neg_freq_pct'] = term_freq_df2['negative'] * 1./term_freq_df2['negative'].sum()
term_freq_df2['neg_hmean'] = term_freq_df2.apply(lambda x: (hmean([x['neg_rate'], x['neg_freq_pct']])
                                                                   if x['neg_rate'] > 0 and x['neg_freq_pct'] > 0 
                                                                   else 0), axis=1)                                                        
#term_freq_df2['neg_rate_normcdf'] = normcdf(term_freq_df2['neg_rate'])
#term_freq_df2['neg_freq_pct_normcdf'] = normcdf(term_freq_df2['neg_freq_pct'])
#term_freq_df2['neg_normcdf_hmean'] = hmean([term_freq_df2['neg_rate_normcdf'], term_freq_df2['neg_freq_pct_normcdf']])
#term_freq_df2.sort_values(by='neg_normcdf_hmean', ascending=False).iloc[:10]




from scipy.stats import norm
def normcdf(x):
    return norm.cdf(x, x.mean(), x.std())

term_freq_df2['pos_rate_normcdf'] = normcdf(term_freq_df2['pos_rate'])
term_freq_df2['pos_freq_pct_normcdf'] = normcdf(term_freq_df2['pos_freq_pct'])
#term_freq_df2['pos_normcdf_hmean'] = hmean([term_freq_df2['pos_rate_normcdf'], term_freq_df2['pos_freq_pct_normcdf']])
#term_freq_df2.sort_values(by='pos_normcdf_hmean', ascending=False).iloc[:10]




plt.figure(figsize=(8,6))
ax = sns.regplot(x="neg_hmean", y="pos_hmean",fit_reg=False, scatter_kws={'alpha':0.5},data=term_freq_df2)
plt.ylabel('Positive Rate and Frequency Harmonic Mean')
plt.xlabel('Negative Rate and Frequency Harmonic Mean')
plt.title('neg hmean vs pos hmean')




x = train.Phrase
y = train.Sentiment
from sklearn.cross_validation import train_test_split

SEED = 2000

x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)




from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from time import time
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score




def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
    t0 = time()
    sentiment_fit = pipeline.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(x_test)
    train_test_time = time() - t0
    accuracy = accuracy_score(y_test, y_pred)
    print("null accuracy: {0:.2f}%".format(null_accuracy*100))
    print("accuracy score: {0:.2f}%".format(accuracy*100))
    if accuracy > null_accuracy:
        print("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100))
    elif accuracy == null_accuracy:
        print("model has the same accuracy with the null accuracy")
    else:
        print("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100))
    print("train and test time: {0:.2f}s".format(train_test_time))
    print("-"*80)
    return accuracy, train_test_time
cvec = CountVectorizer()
lr = LogisticRegression()
n_features = np.arange(10000,100001,10000)

def nfeature_accuracy_checker(vectorizer=cvec, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=lr):
    result = []
    print (classifier)
    print("\n")
    for n in n_features:
        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print("Validation result for {} features".format(n))
        nfeature_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
        result.append((n,nfeature_accuracy,tt_time))
    return result




# Just to check if these top 10 words in term frequency data frame are actually included in Sklearn’s stop words list




from sklearn.feature_extraction import text

a = frozenset(list(term_freq_df.sort_values(by='total', ascending=False).iloc[:10].index))
b = text.ENGLISH_STOP_WORDS
set(a).issubset(set(b))




my_stop_words = frozenset(list(term_freq_df.sort_values(by='total', ascending=False).iloc[:10].index))




print("RESULT FOR UNIGRAM WITHOUT STOP WORDS\n")
feature_result_wosw = nfeature_accuracy_checker(stop_words='english')

print("RESULT FOR UNIGRAM WITH STOP WORDS\n")
feature_result_ug = nfeature_accuracy_checker()

print("RESULT FOR UNIGRAM WITHOUT CUSTOM STOP WORDS (Top 10 frequent words)\n")
feature_result_wocsw = nfeature_accuracy_checker(stop_words=my_stop_words)




nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ug_wocsw = pd.DataFrame(feature_result_wocsw,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ug_wosw = pd.DataFrame(feature_result_wosw,columns=['nfeatures','validation_accuracy','train_test_time'])

plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='with stop words')
plt.plot(nfeatures_plot_ug_wocsw.nfeatures, nfeatures_plot_ug_wocsw.validation_accuracy,label='without custom stop words')
plt.plot(nfeatures_plot_ug_wosw.nfeatures, nfeatures_plot_ug_wosw.validation_accuracy,label='without stop words')
plt.title("Without stop words VS With stop words (Unigram): Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()




print("RESULT FOR BIGRAM WITH STOP WORDS\n")
feature_result_bg = nfeature_accuracy_checker(ngram_range=(1, 2))

print("RESULT FOR TRIGRAM WITH STOP WORDS\n")
feature_result_tg = nfeature_accuracy_checker(ngram_range=(1, 3))




nfeatures_plot_tg = pd.DataFrame(feature_result_tg,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_bg = pd.DataFrame(feature_result_bg,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy','train_test_time'])

plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram')
plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram')
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram')
plt.title("N-gram(1~3) test result : Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()




print("RESULT FOR BIGRAM WITHOUT STOP WORDS\n")
feature_result_bg_nostop = nfeature_accuracy_checker(ngram_range=(1, 2), stop_words='english')

print("RESULT FOR TRIGRAM WITHOUT STOP WORDS\n")
feature_result_tg_nostop = nfeature_accuracy_checker(ngram_range=(1, 3), stop_words='english')




nfeatures_plot_bg_nostop = pd.DataFrame(feature_result_bg_nostop,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_tg_nostop = pd.DataFrame(feature_result_tg_nostop,columns=['nfeatures','validation_accuracy','train_test_time'])

plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_tg_nostop.nfeatures, nfeatures_plot_tg_nostop.validation_accuracy,label='trigram')
plt.plot(nfeatures_plot_bg_nostop.nfeatures, nfeatures_plot_bg_nostop.validation_accuracy,label='bigram')
plt.title("N-gram(2~3) test result : Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()




plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy, label='Bigram')
plt.plot(nfeatures_plot_bg_nostop.nfeatures, nfeatures_plot_bg_nostop.validation_accuracy, label='Bigram-Nostop')
plt.title("N-gram(2~3) test result : Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()




def train_test_and_evaluate(pipeline, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
    sentiment_fit = pipeline.fit(x_train, y_train)
    
    y_pred = sentiment_fit.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4]))
    confusion = pd.DataFrame(conmat, index=['negative','somewhat negative', 'neutral','somewhat positive', 'positive'],
                         columns=['predicted negative','predicted somewhat negative', 'predicted neutral','predicted somewhat positive', 'predicted positive'])
    
    print("null accuracy: {0:.2f}%".format(null_accuracy*100))
    print("accuracy score: {0:.2f}%".format(accuracy*100))
    if accuracy > null_accuracy:
        print("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100))
    elif accuracy == null_accuracy:
        print("model has the same accuracy with the null accuracy")
    else:
        print("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100))
    print("-"*80)
    print("Confusion Matrix\n")
    print(confusion)
    print("-"*80)
    print("Classification Report\n")
    print(classification_report(y_test, y_pred, target_names=['negative','somewhat negative', 'neutral','somewhat positive', 'positive']))




get_ipython().run_cell_magic('time', '', "tg_cvec = CountVectorizer(max_features=80000,ngram_range=(1, 3))\ntg_pipeline = Pipeline([\n        ('vectorizer', tg_cvec),\n        ('classifier', lr)\n    ])\ntrain_test_and_evaluate(tg_pipeline, x_train, y_train, x_validation, y_validation)")




get_ipython().run_cell_magic('time', '', "bg_cvec = CountVectorizer(max_features=80000,ngram_range=(1, 2))\nbg_pipeline = Pipeline([\n        ('vectorizer', bg_cvec),\n        ('classifier', lr)\n    ])\ntrain_test_and_evaluate(bg_pipeline, x_train, y_train, x_validation, y_validation)")




cv_train = bg_cvec.fit_transform(train.Phrase)
print(cv_train.shape)

cv_test = bg_cvec.transform(test.Phrase)
print(cv_test.shape)




from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel

lr = LogisticRegression()
mn = MultinomialNB()
bnb = BernoulliNB()
rc = RidgeClassifier()
abc = AdaBoostClassifier()
lsvc = LinearSVC()

lr.fit(cv_train,y)
mn.fit(cv_train,y)
bnb.fit(cv_train,y)
rc.fit(cv_train,y)
abc.fit(cv_train,y)
lsvc.fit(cv_train,y)




lr_y_pred = lr.predict(cv_test)

lr_sub = pd.DataFrame({"PhraseId": sub['PhraseId'], "Sentiment" : lr_y_pred})

lr_sub.to_csv("LR_submission.csv",index=False)




mn_y_pred = mn.predict(cv_test)

mn_sub = pd.DataFrame({"PhraseId": sub['PhraseId'], "Sentiment" : mn_y_pred})

mn_sub.to_csv("MN_submission.csv",index=False)




rc_y_pred = rc.predict(cv_test)

rc_sub = pd.DataFrame({"PhraseId": sub['PhraseId'], "Sentiment" : rc_y_pred})

rc_sub.to_csv("RC_submission.csv",index=False)




abc_y_pred = abc.predict(cv_test)

abc_sub = pd.DataFrame({"PhraseId": sub['PhraseId'], "Sentiment" : abc_y_pred})

abc_sub.to_csv("abc_submission.csv",index=False)




bnb_y_pred = bnb.predict(cv_test)

bnb_sub = pd.DataFrame({"PhraseId": sub['PhraseId'], "Sentiment" : bnb_y_pred})

bnb_sub.to_csv("bnb_submission.csv",index=False)




from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS
tvec = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')




txt_fitted = tvec.fit(train.Phrase)
txt_transformed = txt_fitted.transform(train.Phrase)




idf = tvec.idf_
print(dict(zip(txt_fitted.get_feature_names(), idf)))




# Instantiate the vectorizer

word_vectorizer = TfidfVectorizer(
    stop_words='english',
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
   # token_pattern=r'\w{2,}',  #vectorize 2-character words or more
    ngram_range=(1, 2),
    max_features=80000)

# fit and transform on it the training features
word_vectorizer.fit(train.Phrase)
X_train_word_features = word_vectorizer.transform(train.Phrase)

#transform the test features to sparse matrix
test_features = word_vectorizer.transform(test.Phrase)




td_train = word_vectorizer.fit_transform(train.Phrase)
print(td_train.shape)

td_test = word_vectorizer.transform(test.Phrase)
print(td_test.shape)




from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel

lr = LogisticRegression()
mn = MultinomialNB()
bnb = BernoulliNB()
rc = RidgeClassifier()
abc = AdaBoostClassifier()
lsvc = LinearSVC()




lr.fit(td_train,y)
mn.fit(td_train,y)
bnb.fit(td_train,y)
rc.fit(td_train,y)
abc.fit(td_train,y)
lsvc.fit(td_train,y)




lr_y_pred_td = lr.predict(td_test)

lr_sub_td = pd.DataFrame({"PhraseId": sub['PhraseId'], "Sentiment" : lr_y_pred_td})

lr_sub_td.to_csv("LR_td_submission.csv",index=False)




mn_y_pred_td = mn.predict(td_test)

mn_sub_td = pd.DataFrame({"PhraseId": sub['PhraseId'], "Sentiment" : mn_y_pred_td})

mn_sub_td.to_csv("MN_td_submission.csv",index=False)




bnb_y_pred_td = bnb.predict(td_test)

bnb_sub_td = pd.DataFrame({"PhraseId": sub['PhraseId'], "Sentiment" : bnb_y_pred_td})

bnb_sub_td.to_csv("BN_td_submission.csv",index=False)




rc_y_pred_td = rc.predict(td_test)

rc_sub_td = pd.DataFrame({"PhraseId": sub['PhraseId'], "Sentiment" : rc_y_pred_td})

rc_sub_td.to_csv("RC_td_submission.csv",index=False)




abc_y_pred_td = rc.predict(td_test)

abc_sub_td = pd.DataFrame({"PhraseId": sub['PhraseId'], "Sentiment" : abc_y_pred_td})

abc_sub_td.to_csv("ABC_td_submission.csv",index=False)




lsvc_y_pred_td = lsvc.predict(td_test)

lsvc_sub_td = pd.DataFrame({"PhraseId": sub['PhraseId'], "Sentiment" : lsvc_y_pred_td})

lsvc_sub_td.to_csv("LSVC_td_submission.csv",index=False)




get_ipython().run_cell_magic('time', '', 'print("RESULT FOR UNIGRAM WITH STOP WORDS (Tfidf)\\n")\nfeature_result_ugt = nfeature_accuracy_checker(vectorizer=tvec)\n\nprint("RESULT FOR UNIGRAM WITHOUT STOP WORDS (Tfidf)\\n")\nfeature_result_ugt_nostop = nfeature_accuracy_checker(vectorizer=tvec, stop_words = \'english\')')




max(feature_result_ugt)




get_ipython().run_cell_magic('time', '', 'print("RESULT FOR BIGRAM WITH STOP WORDS (Tfidf)\\n")\nfeature_result_bgt = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1, 2))')




get_ipython().run_cell_magic('time', '', 'print("RESULT FOR TRIGRAM WITH STOP WORDS (Tfidf)\\n")\nfeature_result_tgt = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1, 3))')




nfeatures_plot_tgt = pd.DataFrame(feature_result_tgt,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_bgt = pd.DataFrame(feature_result_bgt,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ugt = pd.DataFrame(feature_result_ugt,columns=['nfeatures','validation_accuracy','train_test_time'])

plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_tgt.nfeatures, nfeatures_plot_tgt.validation_accuracy, label='trigram tfidf vectorizer',color='royalblue')
plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy, label='trigram count vectorizer',linestyle=':', color='royalblue')
plt.plot(nfeatures_plot_bgt.nfeatures, nfeatures_plot_bgt.validation_accuracy, label='bigram tfidf vectorizer',color='orangered')
plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy, label='bigram count vectorizer',linestyle=':',color='orangered')
plt.plot(nfeatures_plot_ugt.nfeatures, nfeatures_plot_ugt.validation_accuracy, label='unigram tfidf vectorizer',color='gold')
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram count vectorizer',linestyle=':',color='gold')
plt.title("N-gram(1~3) test result : Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()




get_ipython().run_cell_magic('time', '', 'print("RESULT FOR UNIGRAM WITHOUT STOP WORDS (Tfidf)\\n")\nfeature_result_ugt_nostop = nfeature_accuracy_checker(vectorizer=tvec, stop_words=STOPWORDS)')




get_ipython().run_cell_magic('time', '', 'print("RESULT FOR BIGRAM WITHOUT STOP WORDS (Tfidf)\\n")\nfeature_result_bgt_nostop = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1, 2),stop_words=STOPWORDS)')




get_ipython().run_cell_magic('time', '', 'print("RESULT FOR TRIGRAM WITH STOP WORDS (Tfidf)\\n")\nfeature_result_tgt_nostop = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1, 3),stop_words=STOPWORDS)')




nfeatures_plot_tgt_nostop = pd.DataFrame(feature_result_tgt_nostop,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_bgt_nostop = pd.DataFrame(feature_result_bgt_nostop,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ugt_nostop = pd.DataFrame(feature_result_ugt_nostop,columns=['nfeatures','validation_accuracy','train_test_time'])

plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_tgt_nostop.nfeatures, nfeatures_plot_tgt.validation_accuracy,label='trigram tfidf vectorizer',color='royalblue')
plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram count vectorizer',linestyle=':', color='royalblue')
plt.plot(nfeatures_plot_bgt_nostop.nfeatures, nfeatures_plot_bgt.validation_accuracy,label='bigram tfidf vectorizer',color='orangered')
plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram count vectorizer',linestyle=':',color='orangered')
plt.plot(nfeatures_plot_ugt_nostop.nfeatures, nfeatures_plot_ugt.validation_accuracy, label='unigram tfidf vectorizer',color='gold')
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram count vectorizer',linestyle=':',color='gold')
plt.title("N-gram(1~3) test result : Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()




print("RESULT FOR BIGRAM WITHOUT STOP WORDS\n")
feature_result_bg_nostop = nfeature_accuracy_checker(vectorizer=cvec, ngram_range=(1, 2), stop_words=STOPWORDS)

print("RESULT FOR TRIGRAM WITHOUT STOP WORDS\n")
feature_result_tg_nostop = nfeature_accuracy_checker(vectorizer=cvec, ngram_range=(1, 3), stop_words=STOPWORDS)

print("RESULT FOR UNIGRAM WITHOUT STOP WORDS\n")
feature_result_ug_nostop = nfeature_accuracy_checker(vectorizer=cvec, ngram_range=(1, 1), stop_words=STOPWORDS)




nfeatures_plot_tg_nostop = pd.DataFrame(feature_result_tg_nostop, columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_bg_nostop = pd.DataFrame(feature_result_bg_nostop, columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ug_nostop = pd.DataFrame(feature_result_ug_nostop, columns=['nfeatures','validation_accuracy','train_test_time'])




nfeatures_plot_tgt_nostop = pd.DataFrame(feature_result_tgt_nostop,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_bgt_nostop = pd.DataFrame(feature_result_bgt_nostop,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ugt_nostop = pd.DataFrame(feature_result_ugt_nostop,columns=['nfeatures','validation_accuracy','train_test_time'])

plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_tgt_nostop.nfeatures, nfeatures_plot_tgt_nostop.validation_accuracy,label='trigram tfidf vectorizer',color='royalblue')
plt.plot(nfeatures_plot_tg_nostop.nfeatures, nfeatures_plot_tg_nostop.validation_accuracy,label='trigram count vectorizer',linestyle=':', color='royalblue')
plt.plot(nfeatures_plot_bgt_nostop.nfeatures, nfeatures_plot_bgt_nostop.validation_accuracy,label='bigram tfidf vectorizer',color='orangered')
plt.plot(nfeatures_plot_bg_nostop.nfeatures, nfeatures_plot_bg_nostop.validation_accuracy,label='bigram count vectorizer',linestyle=':',color='orangered')
plt.plot(nfeatures_plot_ugt_nostop.nfeatures, nfeatures_plot_ugt_nostop.validation_accuracy, label='unigram tfidf vectorizer',color='gold')
plt.plot(nfeatures_plot_ug_nostop.nfeatures, nfeatures_plot_ug_nostop.validation_accuracy, label='unigram count vectorizer',linestyle=':',color='gold')
plt.title("N-gram(1~3) test result : Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()






