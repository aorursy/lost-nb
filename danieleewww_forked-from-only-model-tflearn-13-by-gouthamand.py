#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# This notebook was copied from https://www.kaggle.com/gouthamand/only-model-tflearn-13


# In[ ]:


import pandas as pd 
import numpy as np
import re
    
import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk 
from nltk.corpus import stopwords
from gensim.models import word2vec
import collections 
import logging

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.embedding_ops import embedding 
from tflearn.layers.merge_ops import merge
from tflearn.data_utils import pad_sequences
from tflearn.models.dnn import DNN
from tflearn.layers.estimator import regression
import tflearn.summaries as summary

import gc
import csv
import string 
import time
import pickle

print ( 'Finished importing....')
modeling_start_time = time.time()
print ('******Modeling start time******:' , 
       (time.strftime("%Hhrs:%Mm:%Ss", time.gmtime((modeling_start_time)))) )
print ( 'Loading data....')
print ( 'Setting opition pd.set_option display.max_colwidth to avoid truncation of columns....')
pd.set_option('display.max_colwidth', -1)
# Read the file into dataframe
#origtrain = pd.read_csv('../input/train.tsv', sep='\t', encoding='utf-8' )
#origtest  = pd.read_csv('../input/test.tsv',  sep='\t', encoding='utf-8' )
origtrain = pd.read_table( '../input/train.tsv',
                        converters = {'brand_name': np.str , 
                               'category_name':  np.str,
                               'item_condition_id': np.uint8,
                               'item_description':  np.str,
                               'name': np.str,
                               'price': np.float64,
                               'shipping': np.uint8,
                               'train_id' :  np.uint64
                               })
origtest = pd.read_table('../input/test.tsv',
                        converters = {'brand_name': np.str , 
                               'category_name':  np.str,
                               'item_condition_id': np.uint8,
                               'item_description':  np.str,
                               'name': np.str,
                               'price': np.float64,
                               'shipping': np.uint8,
                               'train_id' :  np.uint64
                               })
                                           
                        
# The rows in train and test data 
print ("   The number of rows in train", origtrain.shape)
print ("   The number of rows in test", origtest.shape)

# See if there are any price columns with zero value
print ("   Number of rows with price zero : " ,origtrain.price[origtrain.price == 0.0].count())
print ("   Number of rows :  ", origtrain.shape[0])
#df= df[df["score"] > 50]
origtrain = origtrain[origtrain.price > 0.0]
nrow_train = origtrain.shape[0]
print ("   Number of rows aferdrop :  ", nrow_train)
test_id = origtest.test_id.values 


# Move the data into working dataframe  
combdata = pd.concat([origtrain , origtest],0 )           
combdata = combdata.reindex()
#prepare for ntlk
#porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
print ( 'Finished loading data....')

print
del origtrain
del origtest
gc.collect()
#review the memory used by the dataset 
#print('Memory review :',combdata.info(memory_usage= 'deep' )) 
#!free -h


#review the memory used by the dataset 
#print('Memory review :',combdata.info(memory_usage= 'deep' )) 
#!free -h


# In[ ]:


def removeSpecial (text):
    text = re.sub('\W+',' ', text.lower())
    return(text)


# In[ ]:


# Preprocssing of the of text columns 
# replace null with a '0000 aaaa'
# remove special character from the columns
# create  for new columns for each of the text columns 
# they are newName/description/newBrand/newCategory 

start_time = time.time()
print ('Starting pre-processing of text columns at start time is:' , 
       (time.strftime("%Hhrs:%Mm:%Ss", time.gmtime((start_time)))) )

#brand name
print ('   pre-processing brand name...')
combdata.brand_name = combdata.brand_name.apply(removeSpecial) 
combdata.loc[combdata['brand_name'].values == '', 'brand_name'] = '0000aaaa'
combdata['newBrand'] = combdata['brand_name'].str.strip()

#Price
print ('   pre-processing price...')
combdata.price = combdata.price.fillna(0) 
combdata.price = combdata.price.astype('float16') 

#shipping
print ('   pre-processing shipping...')
combdata.shipping =  combdata.shipping.astype('uint8')

#item condition
print ('   pre-processing item condition...')
combdata.item_condition_id =  combdata.item_condition_id.astype('uint8')

#category name
print ('   pre-processing category name...')
combdata.loc[combdata['category_name'].values == '', 'category_name'] = '0000aaaa'
combdata['newCategory'] =  combdata.category_name.apply(removeSpecial)

#test_id
print ('   pre-processing test_id...')
combdata.test_id = combdata.test_id.fillna(9999999) 
combdata.test_id = combdata.test_id.astype('uint32') 

#train_id
print ('   pre-processing train id...')
combdata.train_id = combdata.train_id.fillna(9999999) 
combdata.train_id = combdata.train_id.astype('uint32') 

#name 
print ( '   filling the null with 0000aaaa...')
combdata.loc[combdata['name'].values == '', 'name'] = '0000aaaa'
combdata.loc[combdata['item_description'].values == '', 'item_description'] = '0000aaaa'

print ( '   creating new columns newName and description... ')
combdata['newName'] =  combdata.name.values
combdata['description'] =  combdata.item_description.values

print ( '   removing special character from newName... ')
combdata['newName'] =  combdata.newName.apply(removeSpecial)

print ( '   removing special character from description... ')
combdata['description'] =  combdata.description.apply(removeSpecial)

#combdata.reindex()

# Arrange the data sorted by brand name
#print ( '   Rearraging the dataframe by brand name...')
#combdata = combdata.sort_values(by = ['brand_name'])     
#combdata = combdata.sort_values(by = ['newBrand'])     
#combdata.reindex()
#print ( '   Rearraged the dataframe by brand name...')

end_time = end_time = time.time()
print ('Completed pre-processing of text columns and time taken:', 
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((end_time - start_time))))))
gc.collect()
#review the memory used by the dataset 
#print('Memory review :',combdata.info(memory_usage= 'deep' )) 
#!free -h 


# In[ ]:


# Creating a raw text for the vocab using newNew and description columns
# this would enable to build a vocab
# sentences = [['first', 'sentence'], ['second', 'sentence']]
start_time = time.time()
print ('Building sentneces start time is:' , 
       (time.strftime("%Hhrs:%Mm:%Ss", time.gmtime((start_time)))) )
sentences = []
mask =    (combdata['description'].values != 'no description yet') & (combdata['description'].values != '0000 aaaa')
cmask = (combdata['newCategory'].values != '0000aaaa')
bmask =    (combdata['newBrand'].values != '0000aaaa')
vtext = [ [x] for x in combdata.description[mask].values]
vtext.extend([ [x] for x in combdata.newCategory[cmask].values])
vtext.extend([ [x] for x in combdata.newBrand[bmask].values ])
vtext.extend([ [x] for x in combdata.newName.values ])
for sent in vtext:
    for words in sent :
        w  = [ word  for word in words.split(' ') if ((len(word) > 1)  & ( word not in stop_words))  ]
        sentences.append(w)

#raw_text = [[ word for word in words.split(' ')] for words in vtext] 
print (' The raw_text type : %s and no of words : %d \n sample : %s'
             %(type(sentences), len(sentences) , sentences[0:1]) )

end_time = time.time()
print ('Completed building sentneces and time taken::', 
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((end_time - start_time))))))
del vtext
gc.collect()
#!free -h


# In[ ]:


#word2vec.Word2Vec
#Three such matrices are held in RAM (work is underway to reduce that number to two, or even one). 
#So if your input contains 100,000 unique words, and you asked for layer size=200, 
#the model will require approx. 100,000*200*4*3 bytes = ~229MB.

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

noOfOccurance = 10 
wordVecSize = 200
windowSize = 5 
noOfWorkers = 8 
vocab_size = len(sentences)
                 
start_time = time.time()
print ('Building the Word2Vec model at start time is:' , 
       (time.strftime("%Hhrs:%Mm:%Ss", time.gmtime((start_time)))) )

word2vecModel = word2vec.Word2Vec(sentences, size=wordVecSize, window=windowSize,
                                  min_count=noOfOccurance, workers=noOfWorkers)
# trim unneeded model memory = use (much) less RAM
word2vecModel.init_sims(replace=True)

#print( '  Saving the model')
#word2vecModel.save('data/word2vecModel')

gc.collect()
end_time = time.time()
print ('Completed building word2vec word2vecModel and time taken::', 
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((end_time - start_time))))))


# In[ ]:


# We will create a tfid scores matrix for every word in vocab
# this will a score for individual words
# we will us this score along with word2vec matrix to get a average score for each word 
# in the description and name columns

start_time = time.time()
print ('Building tfidf score matrix for the vocab and and start time is:' , 
       (time.strftime("%Hhrs:%Mm:%Ss", time.gmtime((start_time)))) )

noOfocurance = 10 
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=noOfocurance)
matrix = vectorizer.fit_transform([x  for x in sentences])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print ('   vocab size(%d) ...' %( len(tfidf)))


# Store data (serialize)
#with open('data/sentences.pickle', 'wb') as fp:
#    pickle.dump(sentences, fp, protocol=pickle.HIGHEST_PROTOCOL)

# Load data (deserialize)
#with open('data/sentences.pickle', 'rb') as fp:
#    unserialized_data = pickle.load(fp)

    
end_time = time.time()
print ('Completed tfidf score matrix for the vocab and time taken::', 
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((end_time - start_time))))))


# In[ ]:


# We build a two dictionary 
# a > brandCounts : unique brand names and there count
# b > brandNames : brand name as key and value use that to search the name column
# c > brandMultiNames : multi names  (multi-gram) brand names as key and value use that to search the name column
# d > strbrandNames : brand names as string use that to search the name column
# e > strbrandMultiNames : multi names  (multi-gram) brand names as string use use that to search the name column
#
start_time = time.time()
print ('Building dictionary to replace null brand and start time is:' , 
       (time.strftime("%Hhrs:%Mm:%Ss", time.gmtime((start_time)))) )

noOfocurance = 50 
brandCounts = dict()
brandNames = dict()
brandMultiNames = dict()
brandCount = collections.Counter(combdata["newBrand"].values).most_common()

for  brand,  count in brandCount:
    if ((count >= noOfocurance) &  (len(brand) > 1)):
        brandCounts[brand]=  count
    
#brandNames = dict(zip(brandCounts.keys(),brandCounts.keys()))
brandNames['ipad'] = 'apple'
brandNames ['iphone'] ='apple'
brandNames['galaxy'] = 'samsung'

for value in  brandCounts.keys():
    if (len(value.split(' ')) > 1):
        brandMultiNames[value] = value
    else:
        brandNames[value] = value
brandNames.pop('0000aaaa')
       
strbrandNames = ' '.join(brandNames.keys())
strbrandMultiNames = ' '.join(brandMultiNames.keys())

end_time = time.time()
print ('Completed  dictionary to replace null brand and time taken:', 
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((end_time - start_time))))))

#print('UNI: =============================================================')
#for value in  brandNames.values(): 
#    print('UNI:(%d) %s %d' % (len(brandNames),value, len(value.split(' '))))
#print('MULTI: =============================================================')
#for value in  brandMultiNames.values(): 
#    print('MULTI:(%d) %s %d' % (len(brandMultiNames),value, len(value.split(' '))))
#print('UNI: The type (%s) and value:\n %s' %(type(strbrandMultiNames), strbrandMultiNames))


# In[ ]:


# searching for the brand in name 
# Use the brand dic's to find the brand name in name, return appropriate brand name 
# if brand name (partial word)exists in name, then return apporiate brand name from the dict
# if the brand name happens to be multigram, then return appropriate multi gram brand name from the dict

# Please note that we are using two string nameString & descString. These are newName and description
# column's text. These correspondse to a row where brand is '0000aaaa'

def getBrand(text):
    global globvar
    global nameString
    global nameDesc 
    quit = False

    if (nameDesc):
        nameText = nameString[globvar]
    else:
        nameText = descString[globvar]
        
    ### since we are using apply function on Brand text passed is always'0000aaaa'. We use
    #### *****globvar*** as variable that iterates through nameString & descString for corresponding 
    ###  to Brand column value in search of brand value.
    ### This variable keeps brand in-line with corresponding newName and description column values
    
    globvar += 1  
        
    words = nameText.split()
    #print('getBrand : ' ,text, words)
    rtext ='0000aaaa'
    for word in words:
        brandFound =  brandNames.get(word, 'NO')
        if brandFound != 'NO':
            rtext = brandFound
            quit = True
            break
            
    if not quit:
        for brand in brandMultiNames.keys():
            result = ''
            result = re.search('\\b'+brand+'\\b', nameText)
            if (result):
                rtext = brand
                quit = True
                break

    #if ((globvar%100000) == 0 ):
       # print ('   Are we using nameString : ', nameDesc)
       # print ('   Iteration : ', globvar)
        #print ('type (%s) and name string (%s) : '%(type(nameText),nameText))
        #print()        
        #print('The brand returned:' , rtext)        
        #print()        
        
    return (rtext)


# In[ ]:


# Search name and description fileds for brand name 
# first we will search the name column and then we wiil go for the 
# description. 
start_time = time.time()
print ('Starting replacing null brands and start time is:' ,  
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((start_time))))))
globvar = 0
nameString = []
descString = []
nameDesc = True

mask =    (combdata['newBrand'].values == '0000aaaa') 
noOfNull =  combdata.newBrand[mask].values
nameString =  combdata.newName[mask].values

print ('   The count of brands with 0000aaaa are %d Before searching through newName'  %(len(noOfNull))  )

combdata['Brand'] = combdata['newBrand'].apply(lambda x: x if x != '0000aaaa' else getBrand(x) )
mask = combdata['Brand'].values == '0000aaaa'
noOfNull = combdata.newBrand[mask]
print ('   The count of brands with 0000aaaa are %d After searching through newName'  %(len(noOfNull))  )

# Now serach the scription column
globvar = 0
descString = []
nameDesc = False

mask =  ((combdata['Brand'].values == '0000aaaa') )
btext =  combdata.Brand[mask].values
descString =  combdata.description[mask].values
print ('   The count of brands with 0000aaaa are %d Before searching through description'  %(len(btext))  )
combdata['Brand'] = combdata['Brand'].apply(lambda x: x if x != '0000aaaa' else getBrand(x) )
mask = combdata['Brand'].values == '0000aaaa'
x = combdata.newBrand[mask]
print ('   The count of brands with 0000aaaa are %d After searching through description'  %(len(x))  )

del nameString
del descString 
del btext
del mask
gc.collect()
end_time = time.time()
print ('Completed replacing null brands and time taken to process:', 
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((end_time - start_time))))))


# In[ ]:


# creating two new features lenght for newName and descritpion columns
# a > len_description: length of description  
# b > len_name: length of newName column

start_time = time.time()
print ('Creating new feature len_description and len_name and start time is:' ,  
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((start_time))))))
print('   Creating new feature len_description for column description...')
try:
    combdata['len_description'] = combdata['description'].map(len)
except TypeError  as e:
    print(e)
x = np.max(combdata['len_description'])

print('   Creating new feature len_name for column newName...')
try:
    combdata['len_name'] = combdata['newName'].map(len)
except TypeError  as e:
    print(e)
y = np.max(combdata['len_name'])
print('   max length of column newName is (%d) and column description is (%d)e...' %(y,x))

end_time = end_time = time.time()
print ('Completed creating new feature len_description and len_name and time taken::', 
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((end_time - start_time))))))


# In[ ]:


# Creating a new feature using one hot encoding of shipping and item_condition columns
# hot_shp_item_cnd : one hot encoding of shipping and item_condition columns
start_time = time.time()
print ('Encoding of item_condition_id and shipping and start time is:' ,  
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((start_time))))))

combdata['hot_shp_item_cnd'] = csr_matrix(pd.get_dummies(combdata[["item_condition_id", "shipping" ]], sparse=True).values)
print ('   Shape of X_shp_item_cnd after encoding %s ...'%(str(combdata['hot_shp_item_cnd'].shape)))
end_time = end_time = time.time()
print ('Completed encoding of item_condition_id and shipping and time taken::', 
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((end_time - start_time))))))


# In[ ]:


# LabelEncoder on brand and newCategory
# creating two new features lenght for newName and descritpion columns
# a > lbl_category: LabelEncoder of newCategory column
# b > lbl_brand_name: LabelEncoder of Brand column

start_time = time.time()
print ('Starting encoding Brand and newCategory and start time is:' ,  
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((start_time))))))

le = LabelEncoder()

le.fit(np.hstack([combdata.newCategory]))
combdata['lbl_category'] = le.transform(combdata.newCategory)

le.fit(np.hstack([combdata.Brand]))
combdata['lbl_brand_name'] = le.transform(combdata.Brand)

del le
gc.collect()

end_time = time.time()
print ('Completed encoding Brand and newCategory and time taken to process:', 
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((end_time - start_time))))))


# In[ ]:


# we will creating a average score for each token 
# using tfid score and word2vec matrix to calculate 
# token = vector multiplicaiton of word2vec matrix (200) and tfidf score (1)
# this would then be divided by the no of words
 
def buildWordVector(tokens, size, vec):
    #size = 200
    #print ('The tokens:',type(tokens), tokens, vec)
    #vec = np.zeros(size).reshape((1, size))
    count = 0.
    for words in tokens.split(' '):
        #print( 'The type (words) : ', (words))
        try:
            #print('The word :', word)
            vec = vec + word2vecModel.wv.get_vector(words).reshape((1, size)) * tfidf[words]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
                #print ('Unknow word:',word,KeyError)
                continue
                
    if count != 0:
        vec /= count
    #finalVec = vec
    #print(vec)
    return vec


# In[ ]:


# Let us clean up our data and delete some of the columns that we do not require
# this free up some memry 
start_time = time.time()
print ('Deleting unwanted data and freeing memory and start time is:' ,  
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((start_time))))))
cols = combdata.columns
print ('   We have following columns....\n   ', cols)

rcols = ('brand_name', 'category_name', 'item_condition_id', 'item_description',
       'name', 'shipping', 'Brand','newBrand', 'newCategory') 

print()
print ('   We will remove following columns...\n   ', rcols)
print()
combdata = combdata.filter(['train_id','test_id','price', 'newName', 'description', 'shipping',
                        'hot_shp_item_cnd','lbl_category','item_condition_id',
                        'lbl_brand_name', 'len_description', 'len_name' ], axis= 1) 
cols = combdata.columns
print()
print ('   We have following columns after removing unwanted columns....\n   ', cols)

# The processed data to file 
#print ('   Saving the dataframe to disk (processdata.pkl) for future use...')
#combdata.to_pickle('data/processdata.pkl')    #to save the dataframe, df to 123.pkl
#combdata = pd.read_pickle('123.pkl')   #to load 123.pkl back to the dataframe df

#del combdata
gc.collect()

end_time = time.time()
print ('Completed deleting unwanted data and freeing memory and time taken to process:', 
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((end_time - start_time))))))


# In[ ]:


# We will be splitting the trainning and test data
# we would split the trainning data in to train and validation data 

start_time = time.time()
print ('Spliting train and test data and start time is:' , 
       (time.strftime("%Hhrs:%Mm:%Ss", time.gmtime((start_time)))) )

print('The original train no of rows(%d)'  %(nrow_train))
#Separate the train and test data 
org_train = combdata[:nrow_train]
X_test = combdata[nrow_train:]
print('   org_train shape ', org_train.shape)
print('   X_test shape ',  X_test.shape)

print ('   Spliting training data into train and validation set...')
X_splt_train, X_splt_val  = train_test_split(org_train, random_state=42, test_size=0.10)
y_train = X_splt_train['price'].values 
y_train = y_train.reshape(len(y_train),1)
y_val = X_splt_val['price'].values 
y_val = y_val.reshape(len(y_val),1)

X_TRAIN_SHAPE = X_splt_train.shape[1]
Y_TRAIN_SHAPE = y_train.shape[1]
X_VAL_SHAPE = X_splt_val.shape[1]
Y_VAL_SHAPE = y_val.shape[1]

print('   X_splt_train shape ', X_splt_train.shape)
print('   X_splt_val shape ', X_splt_val.shape)
print('   y_train shape ', y_train.shape)
print('   y_val shape ', y_val.shape)
del combdata
gc.collect()

end_time = time.time()
print ('Completed spliting train and test data and time taken::', 
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((end_time - start_time))))))


# In[ ]:


#Tflearn DATA DEFINITION
# Creating a two dictionary 
# x with all the data 
# shape ditionary with each of the columns shape[1] to be used as input for algo

def get_tflearn_data(dataset, scaler):

    print ('     Creating data dictionary...')
    
    wordVecSize = 200
    #The global variable
    zeroVec = np.zeros(wordVecSize, dtype=np.float32).reshape((1, wordVecSize))
    print ('      Creating name feature...')
    name = np.concatenate([buildWordVector(z, wordVecSize, zeroVec)  for z in dataset.newName], axis=0)
    print ('      The shape of namefeature... ', name.shape)

    print ('      Creating description feature...')
    description = np.concatenate([buildWordVector(z, wordVecSize, zeroVec)  for z in dataset.description], axis=0)
    print ('      The shape of description feature...', description.shape)
    
    
    brand_name =  np.array(dataset.lbl_brand_name.values)
    brand_name = brand_name.reshape(len(brand_name),1)
    
    category =  np.array(dataset.lbl_category.values)
    category = category.reshape(len(category),1)
    
    hot_shp_item_cnd = np.array(dataset.item_condition_id.values)
    #hot_shp_item_cnd = hot_shp_item_cnd.reshape(len(hot_shp_item_cnd), 1)
    item_condition = np.array(dataset.item_condition_id.values)
    item_condition = item_condition.reshape(len(item_condition), 1)
    
    shipping = np.array(dataset.shipping.values)
    shipping = shipping.reshape(len(shipping), 1)
    
    print ('      Scaling inputs features...')
    name = scaler.fit_transform(name)
    description =  scaler.fit_transform(description)
    brand_name =  scaler.fit_transform(brand_name)
    category = scaler.fit_transform(category) 
    item_condition = scaler.fit_transform(item_condition)
    shipping = scaler.fit_transform(shipping)
    print ('      Finished scaling inputs features...')

    X =  {
        'name': name,
        'item_desc' : description,
        'brand_name' : brand_name,
        'category' : category, #'hot_shp_item_cnd':hot_shp_item_cnd
        'item_condition' : item_condition, 
        'shipping' : shipping  
        }

    print ('     Finished data dictionary...')
       
    return (X)


# In[ ]:


# Build the model using our train dictionary data set

def getModel(dataset):
    print('Preparing the model...')
    tf.reset_default_graph()
    
    wordVecSize = 200

    name = input_data (shape= [None, wordVecSize], name="name" )
    item_desc = input_data(shape= [None, wordVecSize] , name="item_desc" ) 
    brand_name = input_data(shape= [None, 1], name="brand_name" ) 
    category = input_data(shape= [None,1],  name="category" ) 
    item_condition = input_data(shape= [None, 1], name="item_condition" ) 
    shipping = input_data(shape= [None, 1], name="shipping" ) 

    
    flat_name = flatten(name)
    flat_item_desc =  flatten(item_desc)
    
    net = merge ([ flat_name, 
                  flat_item_desc,
                  brand_name,
                  category,
                  item_condition,
                  shipping],  
                 mode = 'concat', axis = 1)
    
    X_TRAIN_SHAPE = net.shape[1]
    print('   Buildig the model...')
    print('   Shape of the final merge input dataset:',X_TRAIN_SHAPE )
    
    # create model
    #print('      The graph :' , tf.Graph.version)
    print('      Creating input layer...')
    net1 = fully_connected(net, X_TRAIN_SHAPE, activation='ReLU')
    net2 = fully_connected(net1,  X_TRAIN_SHAPE, activation='ReLU')
    net3 = fully_connected(net2, 256, activation='ReLU')
    net4 = fully_connected(net3, 1, activation='linear')
    net5 = regression(net4, optimizer='Adam', loss='mean_square',  metric='R2')

    model = DNN(net5)
    print('  Finished building model...')

    print('Finished preparing the model...')

    return(model)


# In[ ]:


# prepare the data defination for the model
# for train/validation set and test set

#tf.reset_default_graph()

start_time = time.time()
print ('Preparing data definations and start time is:' , 
       (time.strftime("%Hhrs:%Mm:%Ss", time.gmtime((start_time)))) )


scaler = sklearn.preprocessing.RobustScaler()

print ('   Preparing Train data set...')
X_train_dict = get_tflearn_data(X_splt_train,scaler )

print ('   Preparing Validation data set...')
X_val_dict = get_tflearn_data(X_splt_val,scaler)

print ('   Preparing Test data set...')
X_test_dict = get_tflearn_data(X_test, scaler)

# convert the price into log1 value for scaling 
print ('   Scaling price feature through log...')
y_train = np.log1p(y_train)
y_val = np.log1p(y_val)

# drop the word2vec and tfid model to recover memory
#print ('   Droping word2vec and tfid...')
#with open('vectorizer.pk', 'wb') as tfid:
#    pickle.dump(vectorizer, tfid)
del word2vecModel
del tfidf 
del sentences 
del X_splt_train
del X_splt_val
del X_test
gc.collect()

#print ('   Preparing train array set...')
#X_train = get_dict_array(X_train_dict)

#print ('   Preparing validation array set...')
#X_val = get_dict_array(X_val_dict)

#print ('   Preparing test array set...')
#X_test = get_dict_array(X_test_dict)

#print('   X_train shape ', X_train.shape)
#print('   X_test shape ', X_test.shape)
#print('   X_val shape ', X_val.shape)

print ('Finished data definations...')
end_time = time.time()
print ('Completed preparing data definations and time taken::', 
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((end_time - start_time))))))
#print (end running)


# In[ ]:


#Compute the Root Mean Squared Log Error for hypthesis h and targets y
#
#Args:
#y_pred - numpy array containing predictions with shape (n_samples, n_targets)
#y_actual - numpy array containing targets with shape (n_samples, n_targets)
#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_pred, y_actual): 
    return np.sqrt(np.square(np.log(y_pred + 1) - np.log(y_actual + 1)).mean())


# In[ ]:


# We would be building the model 
# training the model 
start_time = time.time()
print ('Building and training  the modeland start time is:' , 
       (time.strftime("%Hhrs:%Mm:%Ss", time.gmtime((start_time)))) )


modelR = getModel(X_train_dict)
print('Training the model...')

modelR.fit( X_train_dict,  y_train, n_epoch=5, 
           batch_size=256, snapshot_step= 1000, # validation_set=[ X_val_dict, y_val],           
           show_metric=True,run_id='tflMP')

# Target label used for training
#labels = np.array(data[label], dtype=np.float32)
# Reshape target label from (6605,) to (6605, 1)
#labels =np.reshape(y_train,(-1,1)) #makesure the labels has the shape of (?,1)

end_time = time.time()
print ('Completed building and training  the model and time taken::', 
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((end_time - start_time))))))


# In[ ]:


#let us evluate the model
start_time = time.time()
print ('Evaluating the model  the modeland start time is:' , 
       (time.strftime("%Hhrs:%Mm:%Ss", time.gmtime((start_time)))) )

score = modelR.evaluate(X_val_dict, y_val, batch_size=256 )
print('   The validation data set accuracy is:', score)

# calculate the rmsle for val  data set     
val_rmsle = 0
y_val_pred = modelR.predict(X_val_dict)
val_rmsle = rmsle(y_val_pred, y_val)
print ('   The validation data set rmsle is (%f) '%(val_rmsle))

end_time = time.time()
print ('Completed evaluation of the model and time taken::', 
       (time.strftime("%Hhrs:%Mm:%Ss",( time.gmtime((end_time - start_time))))))


# In[ ]:


# We are ready to predict price for test features

start_time = time.time()
print ('Predicting and submitting results and start time is:' , 
       (time.strftime("%Hhrs:%Mm:%Ss", time.gmtime((start_time)))) )


y_pred = np.zeros(X_test_dict['name'].shape[0])

y_pred = modelR.predict(X_test_dict)
y_pred = np.expm1(y_pred)

print('   preparing for submission...')

pricePred = pd.DataFrame()
pricePred['test_id'] = test_id
pricePred['price'] = np.round(y_pred,2)
pricePred.to_csv('Sample_sub.csv',index=False)


print('   submitted...')

end_time = time.time()
print ('******Price recommended in ******:' , 
       (time.strftime("%Hhrs:%Mm:%Ss", time.gmtime((end_time - modeling_start_time)))) )

