#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


os.getcwd()


# In[3]:


train_variants_df = pd.read_csv("../input/training_variants")
test_variants_df = pd.read_csv("../input/test_variants")
train_text_df = pd.read_csv("../input/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
test_text_df = pd.read_csv("../input/test_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])


# In[4]:


print("Train Variant".ljust(15), train_variants_df.shape)
print("Train Text".ljust(15), train_text_df.shape)
print("Test Variant".ljust(15), test_variants_df.shape)
print("Test Text".ljust(15), test_text_df.shape)


# In[5]:


train_variants_df.head()


# In[6]:


train_variants_df.info()


# In[7]:


train_text_df.head()


# In[8]:


train_text_df.info()


# In[9]:


train_variants_df.columns


# In[10]:


train_variants_df.describe()


# In[11]:


#train_variants_df.Gene = train_variants_df.Gene.astype('category')


# In[12]:


#len(train_variants_df.Gene.cat.categories)


# In[13]:


train_variants_df.Class = train_variants_df.Class.astype('category')


# In[14]:


len(train_variants_df.Class.cat.categories)


# In[15]:


train_variants_df.info()


# In[16]:


print("For training data, there are a total of",
len(train_variants_df.ID.unique()), "IDs,", end='')
print(len(train_variants_df.Gene.unique()), "unique genes,", end='')
print(len(train_variants_df.Variation.unique()), "unique variations and ", end='')
print(len(train_variants_df.Class.unique()),  "classes")


# In[17]:


plt.figure(figsize=(12,8))
sns.countplot(x='Class', data=train_variants_df, palette="Blues_d")
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Class', fontsize=14)
plt.title('Distribution of genetic mutation classes', fontsize=18)
plt.show()


# In[18]:


gene_group = train_variants_df.groupby("Gene")['Gene'].count()
minimal_occ_genes = gene_group.sort_values(ascending=True)[:10]
print("Genes with maximal occurances\n",
     gene_group.sort_values(ascending=False)[:10])
print("\nGenes with minimal occurances\n", minimal_occ_genes)


# In[19]:


fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(15,15))

for i in range(3):
    for j in range(3):
        gene_count_grp = train_variants_df[train_variants_df["Class"]==                                          ((i*3+j)+1)].groupby('Gene')['ID']        .count().reset_index()
        sorted_gene_group = gene_count_grp.sort_values('ID', ascending=False)
        sorted_gene_group_top_7 = sorted_gene_group[:7]
        sns.barplot(x="Gene", y="ID", data=sorted_gene_group_top_7,                    ax=axs[i][j])


# In[20]:


train_text_df.head()


# In[21]:


train_text_df.loc[:, 'Text Count'] = train_text_df["Text"].apply(lambda x: len(x.split()))
train_text_df.head()


# In[22]:


train_full = train_variants_df.merge(train_text_df, how='inner',                                     left_on='ID', right_on='ID')
train_full[train_full['Class']==1].head()


# In[23]:


for i in list(range(1,10)):
    print(train_full[train_full['Class']==i].head())


# In[24]:


count_grp = train_full.groupby('Class')['Text Count']
count_grp.describe()


# In[25]:


#Some entries have text count 1.


# In[26]:


train_full[train_full["Text Count"]==1.0]


# In[27]:


train_full[train_full["Text Count"]<500.0]


# In[28]:


plt.figure(figsize=(12,8))
gene_count_grp = train_full.groupby('Gene')['Text Count'].sum().reset_index()
sns.violinplot(x='Class', y='Text Count', data=train_full, inner=None)
sns.swarmplot(x="Class", y="Text Count", data=train_full, color="w", alpha=.5);
plt.ylabel('Text Count', fontsize=14)
plt.xlabel('Class', fontsize=14)
plt.title("Text length distribution", fontsize=18)
plt.show()


# In[29]:


fog, axs = plt.subplots(ncols=3, nrows=3, figsize=(15,15))

for i in range(3):
    for j in range(3):
        gene_count_grp = train_full[train_full["Class"]==((i*3+j)+1)]        .groupby('Gene')["Text Count"].mean().reset_index()
        sorted_gene_group = gene_count_grp.sort_values('Text Count',                                                       ascending=False)
        sorted_gene_group_top_7 = sorted_gene_group[:7]
        sns.barplot(x="Gene", y="Text Count", data=sorted_gene_group_top_7,                    ax=axs[i][j])


# In[30]:


def top_tfidf_feats(row, features, top_n=10):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=10):
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=10):
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=10):
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs):
    fig = plt.figure(figsize=(12, 100), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        #z = int(str(int(i/3)+1) + str((i%3)+1))
        ax = fig.add_subplot(9, 1, i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=16)
        ax.set_ylabel("Gene", labelpad=16, fontsize=16)
        ax.set_title("Class = " + str(df.label), fontsize=18)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()


# In[31]:


tfidf = TfidfVectorizer(min_df=5, max_features=16000, strip_accents='unicode',lowercase =True,
analyzer='word', token_pattern=r'\w+', use_idf=True, 
smooth_idf=True, sublinear_tf=True, stop_words = 'english').fit(train_full["Text"])

Xtr = tfidf.fit_transform(train_full["Text"])
y = train_full["Class"]
features = tfidf.get_feature_names()
top_dfs = top_feats_by_class(Xtr, y, features)


# In[32]:


plot_tfidf_classfeats_h(top_dfs)


# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


# In[34]:


STOPWORDS.add('et')
STOPWORDS.add('wild type')
STOPWORDS.add('amino acid')
STOPWORDS.add('al')
STOPWORDS.add('â')
STOPWORDS.add('Figure')
STOPWORDS.add('figure')
STOPWORDS.add('fig')
STOPWORDS.add('Supplementary')
STOPWORDS.add('cell')
STOPWORDS.add('cancer')
STOPWORDS.add('mutation')
STOPWORDS.add('variant')
STOPWORDS.add('patient')
STOPWORDS.add('tumor')
STOPWORDS.add('table')
STOPWORDS.add('data')
STOPWORDS.add('analysis')
STOPWORDS.add('study')
STOPWORDS.add('method')
STOPWORDS.add('result')
STOPWORDS.add('author')
STOPWORDS.add('conclusion')
STOPWORDS.add('find')
STOPWORDS.add('found')
STOPWORDS.add('show')
STOPWORDS.add('perform')
STOPWORDS.add('demonstrate')
STOPWORDS.add('evaluate')
STOPWORDS.add('discuss')
STOPWORDS.add('mutations')
STOPWORDS.add('variants')
STOPWORDS.add('cells')
STOPWORDS.add('patients')
STOPWORDS.add('protein')
STOPWORDS.add('gene')
STOPWORDS.add('mutant')


# In[35]:


class1DF = train_full[train_full.Class == 1]
class2DF = train_full[train_full.Class == 2]
class3DF = train_full[train_full.Class == 3]
class4DF = train_full[train_full.Class == 4]
class5DF = train_full[train_full.Class == 5]
class6DF = train_full[train_full.Class == 6]
class7DF = train_full[train_full.Class == 7]
class8DF = train_full[train_full.Class == 8]
class9DF = train_full[train_full.Class == 9]


# In[36]:


class1 = class1DF['Text'].tolist()
string1 = ''
for i in range(len(class1)):
    string1 += class1[i]

class2 = class2DF['Text'].tolist()
string2 = ''
for i in range(len(class2)):
    string2 += class2[i]

class3 = class3DF['Text'].tolist()
string3 = ''
for i in range(len(class3)):
    string3 += class3[i]

class4 = class4DF['Text'].tolist()
string4 = ''
for i in range(len(class4)):
    string4 += class4[i]

class5 = class5DF['Text'].tolist()
string5 = ''
for i in range(len(class5)):
    string5 += class5[i]

class6 = class6DF['Text'].tolist()
string6 = ''
for i in range(len(class6)):
    string6 += class6[i]

class7 = class7DF['Text'].tolist()
string7 = ''
for i in range(len(class7)):
    string7 += class7[i]

class8 = class8DF['Text'].tolist()
string8 = ''
for i in range(len(class8)):
    string8 += class8[i]

class9 = class9DF['Text'].tolist()
string9 = ''
for i in range(len(class9)):
    string9 += class9[i]


# In[37]:


wordcloud1 = WordCloud(   stopwords=STOPWORDS,
                          background_color='white',

                       max_words=25
                         ).generate(string1)

wordcloud2 = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='white',

                        max_words=25
                         ).generate(string2)

wordcloud3 = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='white',

                        max_words=25
                         ).generate(string3)

wordcloud4 = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='white',

                        max_words=25
                         ).generate(string4)

wordcloud5 = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='white',

                        max_words=25
                         ).generate(string5)

wordcloud6 = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='white',

                        max_words=25
                         ).generate(string6)

wordcloud7 = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='white',

                        max_words=25
                         ).generate(string7)

wordcloud8 = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='white',

                        max_words=25
                         ).generate(string8)

wordcloud9 = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='white',

                        max_words=25
                         ).generate(string9)


# In[38]:


print("Class 1")
plt.imshow(wordcloud1)
plt.axis('off')
plt.show()

print("Class 2")
plt.imshow(wordcloud2)
plt.axis('off')
plt.show()

print("Class 3")
plt.imshow(wordcloud3)
plt.axis('off')
plt.show()

print("Class 4")
plt.imshow(wordcloud4)
plt.axis('off')
plt.show()

print("Class 5")
plt.imshow(wordcloud5)
plt.axis('off')
plt.show()

print("Class 6")
plt.imshow(wordcloud6)
plt.axis('off')
plt.show()

print("Class 7")
plt.imshow(wordcloud7)
plt.axis('off')
plt.show()

print("Class 8")
plt.imshow(wordcloud8)
plt.axis('off')
plt.show()

print("Class 9")
plt.imshow(wordcloud9)
plt.axis('off')
plt.show()


# In[39]:


import numpy as np
import pandas as pd

from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


from nltk.corpus import stopwords
import re
import gc

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD


# In[40]:


train_variants_df = pd.read_csv("../input/training_variants")
test_variants_df = pd.read_csv("../input/test_variants")
train_text_df = pd.read_csv("../input/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
test_text_df = pd.read_csv("../input/test_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])


# In[41]:


train_variants_df.head()


# In[42]:


train_text_df.head()


# In[43]:


print(train_text_df['Text'][0][:100], '  ', len(train_text_df['Text'][0]))


# In[44]:


varsGeneCount = Counter(train_variants_df.Gene)
print(len(varsGeneCount))


# In[45]:


plt.figure(figsize=(12,8))
ax = sns.countplot(x='Class', data=train_variants_df)
plt.ylabel('Frequency')
plt.xlabel('Class')
plt.title('Freq. of classes in training variants')
plt.show()


# In[46]:


varsVariationCount = Counter(train_variants_df.Variation)
print('Number of unique variaitons in training data\n', len(varsVariationCount))


# In[47]:


fig, ax = plt.subplots(1,1,figsize=(12,8))
ax = sns.distplot(pd.factorize(train_variants_df['Variation'])[0]/                  len(train_variants_df), bins=150, color='r')


# In[48]:


def textClean(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stopw = {'so', 'his', 't', 'y', 'ours', 'herself', 
             'your', 'all', 'some', 'they', 'i', 'of', 'didn', 
             'them', 'when', 'will', 'that', 'its', 'because', 
             'while', 'those', 'my', 'don', 'again', 'her', 'if',
             'further', 'now', 'does', 'against', 'won', 'same', 
             'a', 'during', 'who', 'here', 'have', 'in', 'being', 
             'it', 'other', 'once', 'itself', 'hers', 'after', 're',
             'just', 'their', 'himself', 'theirs', 'whom', 'then', 'd', 
             'out', 'm', 'mustn', 'where', 'below', 'about', 'isn',
             'shouldn', 'wouldn', 'these', 'me', 'to', 'doesn', 'into',
             'the', 'until', 'she', 'am', 'under', 'how', 'yourself',
             'couldn', 'ma', 'up', 'than', 'from', 'themselves', 'yourselves',
             'off', 'above', 'yours', 'having', 'mightn', 'needn', 'on', 
             'too', 'there', 'an', 'and', 'down', 'ourselves', 'each',
             'hadn', 'ain', 'such', 've', 'did', 'be', 'or', 'aren', 'he', 
             'should', 'for', 'both', 'doing', 'this', 'through', 'do', 'had',
             'own', 'but', 'were', 'over', 'not', 'are', 'few', 'by', 
             'been', 'most', 'no', 'as', 'was', 'what', 's', 'is', 'you', 
             'shan', 'between', 'wasn', 'has', 'more', 'him', 'nor',
             'can', 'why', 'any', 'at', 'myself', 'very', 'with', 'we', 
             'which', 'hasn', 'weren', 'haven', 'our', 'll', 'only',
             'o', 'before',"fig", "figure", "et", "al", "table", "data", 
             "analysis", "analyze", "study","method", "result", "conclusion",
             "author", "find", "found", "show", "perform", "demonstrate", "evaluate", 
             "discuss", 'et',"al","â","Figure","figure","fig","Supplementary","cell",
             "cancer","mutation","variant","patient","tumor","table","data","analysis",
             "study","method","result","author","conclusion","find","found","show","perform",
             "demonstrate","evaluate","discuss","mutations","variants","cells","patients",
             "protein","gene","mutant"}
    text = [w for w in text if not w in stopw]    
    text = " ".join(text)
    text = text.replace("."," ").replace(","," ")
    return(text)


# In[49]:


trainText = []
for it in train_text_df['Text']:
    newText = textClean(it)
    trainText.append(newText)
testText = []
for it in test_text_df['Text']:
    newText = textClean(it)
    testText.append(newText)


# In[50]:


trainText[0][:100]


# In[51]:


for i in range(10):
    print('\n Doc', str(i))
    stopCheck = Counter(trainText[i].split())
    print(stopCheck.most_common()[:10])


# In[52]:


tops = Counter(str(trainText).split()).most_common()[:20]
labs, vals = zip(*tops)
idx = np.arange(len(labs))
wid=0.6

fig, ax = plt.subplots(1,1,figsize=(14,8))
ax = plt.bar(idx, vals, wid, color='b')
ax = plt.xticks(idx - wid/8, labs, rotation=25, size=14)
plt.title('Top twenty counts of most-common words among text')
plt.show()


# In[53]:


gc.collect()


# In[54]:


topInc = Counter(str(trainText).split()).most_common()[:30]
labsInc, valsInc = zip(*topInc)


# In[55]:


def stopCheck(text, stops):
    text = text.split()
#     stops = {'mutations', 'cancer'}
    text = [w for w in text if not w in stops]    
    text = " ".join(text)
    return text


# In[56]:


trainText2 = []
for it in trainText:
    newText = stopCheck(it, labsInc)
    trainText2.append(newText)
testText2 = []
for it in testText:
    newText = stopCheck(it, labsInc)
    testText2.append(newText)


# In[57]:


gc.collect()


# In[58]:


trainText2[2][:100]


# In[59]:


tops = Counter(str(trainText2).split()).most_common()[:20]
labs, vals = zip(*tops)
idx = np.arange(len(labs))
wid = 0.6

fig, ax = plt.subplots(1,1,figsize=(14,8))
ax = plt.bar(idx, vals, wid, color='b')
ax = plt.xticks(idx - wid/8, labs, rotation=25, size=14)
plt.title('Top Twenty Counts of Most-Common Words Among Text')
plt.show()


# In[60]:


gc.collect()


# In[61]:


maxFeats=500


# In[62]:


tfidf = TfidfVectorizer(min_df=5, max_features=maxFeats, ngram_range=(1,3),                        strip_accents='unicode', lowercase = True,                        analyzer='word', token_pattern=r'\w+',                        use_idf=True, smooth_idf=True, sublinear_tf=True,
                        stop_words='english')


# In[63]:


tfidf.fit(trainText2)


# In[64]:


countVec = CountVectorizer(min_df=5, ngram_range=(1,3), max_features=maxFeats, 
                           strip_accents='unicode',lowercase =True, 
                           analyzer='word', token_pattern=r'\w+',
                           stop_words = 'english')


# In[65]:


countVec.fit(trainText2)


# In[66]:


len(trainText2)


# In[67]:


svd = TruncatedSVD(n_components=390)
svdFit = svd.fit_transform(tfidf.transform(trainText2))


# In[68]:


def buildFeats(texts, variations):
    temp = variations.copy()
    print('Encoding...')
    temp['Gene'] = pd.factorize(variations['Gene'])[0]
    temp['Variation'] = pd.factorize(variations['Variation'])[0]
    temp['Gene_to_Variation_Ratio'] = temp['Gene']/temp['Variation']
    
    print('Lengths...')
    temp['doc_len'] = [len(x) for x in texts]
    temp['unique_words'] = [len(set(x))  for x in texts]
    
    print('TFIDF...')
    temp_tfidf = tfidf.transform(texts)
    temp['tfidf_sum'] = temp_tfidf.sum(axis=1)
    temp['tfidf_mean'] = temp_tfidf.mean(axis=1)
    temp['tfidf_len'] =  (temp_tfidf != 0).sum(axis = 1)
    
    print('Count Vecs...')
    temp_cvec = countVec.transform(texts)
    temp['cvec_sum'] = temp_cvec.sum(axis=1)
    temp['cvec_mean'] = temp_cvec.mean(axis=1)
    temp['cvec_len'] =  (temp_cvec != 0).sum(axis = 1)
    
    print('Latent Semantic Analysis Cols...')
    tempc = list(temp.columns)
    temp_lsa = svd.transform(temp_tfidf)
    
    for i in range(np.shape(temp_lsa)[1]):
        tempc.append('lsa'+str(i+1))
    temp = pd.concat([temp, pd.DataFrame(temp_lsa, index=temp.index)], axis=1)
    
    return temp, tempc


# In[69]:


#temp = train_variants_df.copy()


# In[70]:


#temp['Gene']=pd.factorize(train_variants_df['Gene'])[0]
#temp['Variation']=pd.factorize(train_variants_df['Variation'])[0]
#temp['Gene_to_Variation_Ratio']=temp['Gene']/temp['Variation']


# In[71]:


#temp['doc_len'] = [len(x) for x in trainText2]
#temp['unique_words'] = [len(set(x))  for x in trainText2]


# In[72]:


#temp_tfidf = tfidf.transform(trainText2)


# In[73]:


#temp['tfidf_sum'] = temp_tfidf.sum(axis=1)
#temp_tfidf.sum(axis=1)


# In[74]:


#temp['tfidf_mean'] = temp_tfidf.mean(axis=1)


# In[75]:


#temp['tfidf_len'] =  (temp_tfidf != 0).sum(axis = 1)


# In[76]:


#temp_cvec = countVec.transform(trainText2)


# In[77]:


#temp['cvec_sum'] = temp_cvec.sum(axis=1)
#temp['cvec_mean'] = temp_cvec.mean(axis=1)
#temp['cvec_len'] =  (temp_cvec != 0).sum(axis = 1)


# In[78]:


#tempc = list(temp.columns)


# In[79]:


#temp_lsa = svd.transform(temp_tfidf)


# In[80]:


#for i in range(np.shape(temp_lsa)[1]):
#    tempc.append('lsa'+str(i+1))
#temp = pd.concat([temp, pd.DataFrame(temp_lsa, index=temp.index)], axis=1)


# In[81]:


#type(temp_lsa)


# In[82]:


#plt.plot(temp_lsa)
#plt.show()


# In[83]:


#temp.head()


# In[84]:


trainDF, trainCol = buildFeats(trainText2, train_variants_df)
testDF, testCol = buildFeats(testText2, test_variants_df)


# In[85]:


testDF.head()


# In[86]:


type(trainCol)


# In[87]:


trainDF.columns = trainCol
testDF.columns = testCol


# In[88]:


trainDF.head()


# In[89]:


classes = train_variants_df.Class - 1 #python indexing starts from 0
print('Original:', Counter(train_variants_df.Class), '\n ReHashed: ', Counter(classes))


# In[90]:


X_train, X_test, y_train, y_test = train_test_split(trainDF.drop(['ID','Class'],                                                                 axis=1),
                                    classes,
                                    test_size = 0.1,
                                    random_state=31415)
print(np.shape(X_train))


# In[91]:


print('Format a Train and Test Set for LGB')
d_train = lgb.Dataset(X_train, label=y_train)
d_val = lgb.Dataset(X_test, label=y_test)
               
gc.collect()


# In[92]:


parms = {'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 9,
    'metric': {'multi_logloss'},
    'learning_rate': 0.05, 
    'max_depth': 5,
    'num_iterations': 400, 
    'num_leaves': 95, 
    'min_data_in_leaf': 60, 
    'lambda_l1': 1.0,
    'feature_fraction': 0.8, 
    'bagging_fraction': 0.8, 
    'bagging_freq': 5}

rnds = 260
mod = lgb.train(parms, train_set=d_train, num_boost_round=rnds,
               valid_sets=[d_val], valid_names=['dval'], verbose_eval=20,
               early_stopping_rounds=20)


# In[93]:


lgb.plot_importance(mod, max_num_features=30, figsize=(14,10))


# In[94]:


pred = mod.predict(testDF.drop(['ID'],axis=1))


# In[95]:


mod.best_score


# In[96]:


mod.best_iteration


# In[97]:


sub = pd.DataFrame(pred, index=testDF.index)
sub.head()


# In[98]:


sub.shape


# In[99]:


#sub.to_csv('submission.csv', index=False)


# In[100]:


sub['ID'] = sub.index


# In[101]:


cols = sub.columns.tolist()


# In[102]:


cols = cols[-1:] + cols[:-1]
cols
sub = sub[cols]


# In[103]:


sub.columns = ['ID', 'class1', 'class2', 'class3', 'class4','class5','class6','class7','class8','class9']


# In[104]:


sub.head()


# In[105]:


sub.to_csv('submission.csv', index=False)


# In[106]:




