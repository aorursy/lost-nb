#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re
from wordcloud import WordCloud, STOPWORDS 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import vstack, hstack, csr_matrix
from scipy import sparse




data_train = pd.read_csv('/kaggle/input/mercari-price-suggestion-challenge/train.tsv', delimiter='\t')




data_test =  pd.read_csv('/kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv', delimiter='\t')




data_train.head(5)




data_test.head(5)




#we will use isnull() function to get missing values
print("Train DataFrame")
print(pd.isnull(data_train).sum())
print("="*50)
print("Test DataFrame")
print(pd.isnull(data_test).sum())




print((pd.isnull(data_train['brand_name']).sum())*100/data_train.shape[0])




print("{0:.2f}% brand_name has missing value in train data".format((pd.isnull(data_train['brand_name']).sum())*100/data_train.shape[0]))

print("{0:.2f}% category_name has missing value in train data".format((pd.isnull(data_train['category_name']).sum())*100/data_train.shape[0]))




print("{0:.2f}% brand_name has missing value in test data".format((pd.isnull(data_test['brand_name']).sum())*100/data_test.shape[0]))

print("{0:.2f}% category_name has missing value in test data".format((pd.isnull(data_test['category_name']).sum())*100/data_test.shape[0]))




# filling missing values with brand_name as Unknown
data_train['brand_name'] = data_train['brand_name'].fillna('Unknown')
data_test['brand_name'] = data_test['brand_name'].fillna('Unknown')




# filling missing values with category_name as NO/NO/NO or we cn simply remove these rows
data_train['category_name'] = data_train['category_name'].fillna('No/No/No')
data_test['category_name'] = data_test['category_name'].fillna('No/No/No')




#checking the levels of sub categories
levels =[]

for values in data_train['category_name']:
    levels.append(values.count("/"))


print("MIN no of levels:        {0:.0f} ".format(np.min(levels)+1))
print("MEDIAN of levels:        {0:.0f} ".format(np.percentile(levels, 50)+1))
print("90 percentile  of levels:{0:.0f} ".format(np.percentile(levels, 90)+1))
print("MAX no of levels:        {0:.0f}".format(np.max(levels)+1))




##### seperate sub categories from main
def find_sub_cat(X):
    try:
        return(X.split("/"))
        return("jubu")
    except:
        return("None","None","None")
    




data_train['main_category']=''
data_train['sub_category_1']=''
data_train['sub_category_2']=''

data_train['main_category'],data_train['sub_category_1'],data_train['sub_category_2']                                = zip(*data_train['category_name'].apply(lambda x: find_sub_cat(x)))

##Because we have list of list we have to use zip(*) to get final 3 list
#https://www.youtube.com/watch?v=Rlak6CTcUDI




data_test['main_category']=''
data_test['sub_category_1']=''
data_test['sub_category_2']=''

data_test['main_category'],data_test['sub_category_1'],data_test['sub_category_2']                                = zip(*data_test['category_name'].apply(lambda x: find_sub_cat(x)))

##Because we have list of list we have to use zip(*) to get final 3 list
#https://www.youtube.com/watch?v=Rlak6CTcUDI




#data_train['main_category']




#data_train = data_train.drop('category_name',axis=1)
data_train.head(5)




#to get better display format follow below link
#https://stackoverflow.com/questions/55394854/how-to-change-the-format-of-describe-output
pd.set_option('display.float_format', lambda x: '%.5f' % x)

data_train.describe()




plt.figure(figsize=(10,5))
plt.hist(data_train['price'],bins=50, edgecolor='white',range=[0,300])
plt.ylabel('Frequency')
plt.xlabel('Price')
plt.show()




plt.figure(figsize=(10,5))
plt.hist(np.log(data_train['price']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.show()




ax = sns.countplot('item_condition_id',data=data_train)
ax.set_title('Count of each Item_condition_id')




#list of top 10 frequent brand name in data

#brands = data_train.groupby(['brand_name']).count().sort_values(ascending=False)[:10]
#print(brands)
brands = data_train['brand_name'].value_counts()
print(brands[:10])




plt.figure(figsize=(20,15))
plt.subplot(331)
plt.hist(np.log(data_train['price'][data_train['brand_name']=='PINK']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('PINK')

plt.subplot(332)
plt.hist(np.log(data_train['price'][data_train['brand_name']=='Nike']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Nike')

plt.subplot(333)
plt.hist(np.log(data_train['price'][data_train['brand_name']=="Victoria's Secret"]+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title("Victoria's Secret")

plt.subplot(334)
plt.hist(np.log(data_train['price'][data_train['brand_name']=='LuLaRoe']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('LuLaRoe')

plt.subplot(335)
plt.hist(np.log(data_train['price'][data_train['brand_name']=='Apple']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Apple')

plt.subplot(336)
plt.hist(np.log(data_train['price'][data_train['brand_name']=='FOREVER 21']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('FOREVER 21')

plt.subplot(337)
plt.hist(np.log(data_train['price'][data_train['brand_name']=='Nintendo']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Nintendo')

plt.subplot(338)
plt.hist(np.log(data_train['price'][data_train['brand_name']=='Lululemon']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Lululemon')

plt.subplot(339)
plt.hist(np.log(data_train['price'][data_train['brand_name']=='Michael Kors']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Michael Kors')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()




#list of less 10 frequent brand
print(brands[200:210])




plt.figure(figsize=(20,15))
plt.subplot(331)
plt.hist(np.log(data_train['price'][data_train['brand_name']=='Spin Master']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Spin Master')

plt.subplot(332)
plt.hist(np.log(data_train['price'][data_train['brand_name']=='Yankee Candle']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Yankee Candle')

plt.subplot(333)
plt.hist(np.log(data_train['price'][data_train['brand_name']=="James Avery"]+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title("James Avery")

plt.subplot(334)
plt.hist(np.log(data_train['price'][data_train['brand_name']=='Dr. Martens']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Dr. Martens')

plt.subplot(335)
plt.hist(np.log(data_train['price'][data_train['brand_name']=='Keurig']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Keurig')

plt.subplot(336)
plt.hist(np.log(data_train['price'][data_train['brand_name']=='WWE']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('WWE')

plt.subplot(337)
plt.hist(np.log(data_train['price'][data_train['brand_name']=='Bullhead']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Bullhead')

plt.subplot(338)
plt.hist(np.log(data_train['price'][data_train['brand_name']=='AmazonBasics']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('AmazonBasics')

plt.subplot(339)
plt.hist(np.log(data_train['price'][data_train['brand_name']=='NBA']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('NBA')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()




(data_train['shipping'].value_counts())*100/data_train.shape[0]




plt.figure(figsize=(10,8))
plt.hist(np.log(data_train['price'][data_train['shipping']==0]+1),bins=30, edgecolor='white',color="blue", label='shipping not paid by customer') #log(0) is undefined
plt.hist(np.log(data_train['price'][data_train['shipping']==1]+1),bins=30, edgecolor='white',color="green",label="shipping paid by customer") #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.legend(loc='upper right')
plt.title('Shiping')




print("There are %d unique main_categories." % data_train['main_category'].nunique())




plt.figure(figsize=(10,8))
ax = sns.countplot('main_category',data=data_train)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_title('Count of each main_category')




plt.figure(figsize=(20,15))
plt.subplot(331)
plt.hist(np.log(data_train['price'][data_train['main_category']=='Men']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Men')

plt.subplot(332)
plt.hist(np.log(data_train['price'][data_train['main_category']=='Electronics']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Electronics')

plt.subplot(333)
plt.hist(np.log(data_train['price'][data_train['main_category']=="Women"]+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title("Women")

plt.subplot(334)
plt.hist(np.log(data_train['price'][data_train['main_category']=='Home']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Home')

plt.subplot(335)
plt.hist(np.log(data_train['price'][data_train['main_category']=='Sports & Outdoors']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Keurig')

plt.subplot(336)
plt.hist(np.log(data_train['price'][data_train['main_category']=='Beauty']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Beauty')

plt.subplot(337)
plt.hist(np.log(data_train['price'][data_train['main_category']=='Kids']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Kids')

plt.subplot(338)
plt.hist(np.log(data_train['price'][data_train['main_category']=='Handmade']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Handmade')

plt.subplot(339)
plt.hist(np.log(data_train['price'][data_train['main_category']=='Other']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Other')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()




data_train.groupby('main_category')['price'].describe()




print("There are %d unique sub_category_1." % data_train['sub_category_1'].nunique())




plt.figure(figsize=(10,8))
ax = sns.countplot('sub_category_1',data=data_train,order=data_train.sub_category_1.value_counts().iloc[:15].index)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_title('Count of each sub_category_1')




plt.figure(figsize=(20,15))
plt.subplot(331)
plt.hist(np.log(data_train['price'][data_train['sub_category_1']=='Athletic Apparel']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Athletic Apparel')

plt.subplot(332)
plt.hist(np.log(data_train['price'][data_train['sub_category_1']=='Makeup']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Makeup')

plt.subplot(333)
plt.hist(np.log(data_train['price'][data_train['sub_category_1']=="Tops & Blouses"]+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title("Tops & Blouses")

plt.subplot(334)
plt.hist(np.log(data_train['price'][data_train['sub_category_1']=='Shoes']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Shoes')

plt.subplot(335)
plt.hist(np.log(data_train['price'][data_train['sub_category_1']=='Jewelry']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Jewelry')

plt.subplot(336)
plt.hist(np.log(data_train['price'][data_train['sub_category_1']=='Toys']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Toys')

plt.subplot(337)
plt.hist(np.log(data_train['price'][data_train['sub_category_1']=='Cell Phones & Accessories']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Cell Phones & Accessories')

plt.subplot(338)
plt.hist(np.log(data_train['price'][data_train['sub_category_1']=="Jeans"]+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title("Jeans")

plt.subplot(339)
plt.hist(np.log(data_train['price'][data_train['sub_category_1']=='Dresses']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Dresses')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()




data_train.groupby('sub_category_1')['price'].describe()[:10]




print("There are %d unique sub_category_2." % data_train['sub_category_2'].nunique())




plt.figure(figsize=(10,8))
ax = sns.countplot('sub_category_2',data=data_train,order=data_train.sub_category_2.value_counts().iloc[:15].index)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_title('Count of each sub_category_2')




plt.figure(figsize=(20,15))
plt.subplot(331)
plt.hist(np.log(data_train['price'][data_train['sub_category_2']=='Boots']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Boots')

plt.subplot(332)
plt.hist(np.log(data_train['price'][data_train['sub_category_2']=='Other']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Other')

plt.subplot(333)
plt.hist(np.log(data_train['price'][data_train['sub_category_2']=="Face"]+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title("Face")

plt.subplot(334)
plt.hist(np.log(data_train['price'][data_train['sub_category_2']=='T-Shirts']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('T-Shirts')

plt.subplot(335)
plt.hist(np.log(data_train['price'][data_train['sub_category_2']=='Shoes']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Shoes')

plt.subplot(336)
plt.hist(np.log(data_train['price'][data_train['sub_category_2']=='Games']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Games')

plt.subplot(337)
plt.hist(np.log(data_train['price'][data_train['sub_category_2']=='Athletic']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Athletic')

plt.subplot(338)
plt.hist(np.log(data_train['price'][data_train['sub_category_2']=="Eyes"]+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title("Eyes")

plt.subplot(339)
plt.hist(np.log(data_train['price'][data_train['sub_category_2']=='Shorts']+1),bins=50, edgecolor='white') #log(0) is undefined
plt.ylabel('Frequency')
plt.xlabel('log_Price')
plt.title('Shorts')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()




data_train.groupby('sub_category_2')['price'].describe()[:10]




# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list: 'no', 'nor', 'not'
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",             'won', "won't", 'wouldn', "wouldn't"]




import re

def decontracted(phrase):
    # specific
    try:
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase
    except:
        return 0




def wordCount_with_cleaning(sentance):
    try:
        sent = decontracted(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\"', ' ')
        sent = sent.replace('\\n', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        # https://gist.github.com/sebleier/554280
        sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
        sent = sent.strip()
        return(len(sent.split()))
    except:
        return 0 




def wordCount_without_cleaning(sentance):
    try:   
        # https://gist.github.com/sebleier/554280
        sent = ' '.join(e for e in sentance.split())
        sent = sent.strip()
        return(len(sent.split()))
    except:
        return 0




def len_str(x):
    return(len(x))




# words in name

x = data_train['name'].apply(lambda x: wordCount_without_cleaning(x))
plt.hist(x,bins = 30,range=[0,10])
plt.show()




#length of name
x = data_train['name'].apply(lambda x: len_str(x))
plt.hist(x,bins = 30,range=[0,50],edgecolor='white')
plt.show()




text = data_train['name'].values
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = STOPWORDS, 
                min_font_size = 10).generate(str(text))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()




# words count without cleaning in description
x = data_train['item_description'].apply(lambda x: wordCount_without_cleaning(x))
plt.figure(figsize=(8,4))
plt.hist(x,bins = 30,range=[0,10])
plt.show()




#length of description
x = data_train['item_description'].apply(lambda x: len_str(str(x)))
plt.figure(figsize=(8,4))
plt.hist(x,bins = 30,range=[0,50],edgecolor='white')
plt.show()




fig = plt.figure(figsize = (40, 30))
text = data_train['item_description'][data_train['main_category']=='Men'].values
#text = data_train['item_description'].values
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = STOPWORDS, 
                min_font_size = 10, max_words=50,random_state=42).generate(str(text))
plt.subplot(131)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=1)
plt.title("Men",fontsize= 50)




text = data_train['item_description'][data_train['main_category']=='Electronics'].values
#text = data_train['item_description'].values
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = STOPWORDS, 
                min_font_size = 10,max_words=50,random_state=42).generate(str(text))
plt.subplot(132)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=1)
plt.title("Electronics",fontsize= 50)




text = data_train['item_description'][data_train['main_category']=='Kids'].values
#text = data_train['item_description'].values
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = STOPWORDS, 
                min_font_size = 10,max_words=50,random_state=42).generate(str(text))
plt.subplot(133)
plt.title("Kids",fontsize= 50)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=1)

plt.show()




fig = plt.figure(figsize = (40, 30))
#    facecolor = 'k',
#    edgecolor = 'k')
text = data_train['item_description'][data_train['sub_category_1']=='Athletic Apparel'].values
#text = data_train['item_description'].values
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = STOPWORDS, 
                min_font_size = 10, max_words=50,random_state=42).generate(str(text))
plt.subplot(131)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=1)
plt.title("Athletic Apparel",fontsize= 50)




text = data_train['item_description'][data_train['sub_category_1']=='Makeup'].values
#text = data_train['item_description'].values
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = STOPWORDS, 
                min_font_size = 10,max_words=50,random_state=42).generate(str(text))
plt.subplot(132)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=1)
plt.title("Makeup",fontsize= 50)




text = data_train['item_description'][data_train['sub_category_1']=="Tops & Blouses"].values
#text = data_train['item_description'].values
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = STOPWORDS, 
                min_font_size = 10,max_words=50,random_state=42).generate(str(text))
plt.subplot(133)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=1)
plt.title("Tops & Blouses",fontsize= 50)

plt.show()




fig = plt.figure(figsize = (40, 30))
text = data_train['item_description'][data_train['sub_category_2']=='Games'].values
#text = data_train['item_description'].values
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = STOPWORDS, 
                min_font_size = 10, max_words=50,random_state=42).generate(str(text))
plt.subplot(131)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=1)
plt.title("Games",fontsize= 50)




text = data_train['item_description'][data_train['sub_category_2']=='Face'].values
#text = data_train['item_description'].values
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = STOPWORDS, 
                min_font_size = 10,max_words=50,random_state=42).generate(str(text))
plt.subplot(132)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=1)
plt.title("Face",fontsize= 50)




text = data_train['item_description'][data_train['sub_category_2']=="T-Shirts"].values
#text = data_train['item_description'].values
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = STOPWORDS, 
                min_font_size = 10,max_words=50,random_state=42).generate(str(text))
plt.subplot(133)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=1)
plt.title("T-Shirts",fontsize= 50)



plt.show()




from tqdm import tqdm
preprocessed_item_description = []
# tqdm is for printing the status bar
for sentance in tqdm(data_train['item_description'].values):
    sent = decontracted(str(sentance))
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
    preprocessed_item_description.append(sent.lower().strip())




from tqdm import tqdm
preprocessed_test_item_description = []
# tqdm is for printing the status bar
for sentance in tqdm(data_test['item_description'].values):
    sent = decontracted(str(sentance))
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
    preprocessed_test_item_description.append(sent.lower().strip())




# after preprocesing
data_train = data_train.drop('item_description',axis=1)
data_train['item_description'] = preprocessed_item_description
data_train.head(2)




# after preprocesing
data_test = data_test.drop('item_description',axis=1)
data_test['item_description'] = preprocessed_test_item_description
data_test.head(2)




#data  = data_train.drop(['main_category','sub_category_1','sub_category_2'],axis=1)




data_train.head(5)




y = np.log10(np.array(data_train['price'])+1)
X = data_train.drop('price',axis=1)




X_train,X_cv,Y_train,Y_cv = train_test_split(X, y, test_size=0.20, random_state=42)
#X_test,X_cv,Y_test,Y_cv = train_test_split(X_test, Y_test, test_size=0.50, random_state=42)









# creating dictionary containing brand_name and category_name
def concat_categories(x):
    return set(x.values)


#function returning brand name using brand and category_name
def brandfinder(name, category):    
    for brand in brands_sorted_by_size:
        if brand in name and category in brand_names_categories[brand]:
            return brand
    return 'Unknown'




brand_names_categories = dict(X_train[X_train['brand_name'] != 'Unknown'][['brand_name','category_name']].astype('str')                              .groupby('brand_name').agg(concat_categories).reset_index().values.tolist())
#Brands sorted by length (decreasinly), so that longer brand names have precedence in the null brand search
brands_sorted_by_size = list(sorted(filter(lambda y: len(y) >= 3,                                            list(brand_names_categories.keys())),                                             key = lambda x: -len(x)))




train_names_unknown_brands_train = X_train[X_train['brand_name'] == 'Unknown'][['name','category_name']].                            astype('str').values

train_names_unknown_brands_cv = X_cv[X_cv['brand_name'] == 'Unknown'][['name','category_name']].                            astype('str').values




train_names_unknown_brands_data_test = data_test[data_test['brand_name'] == 'Unknown'][['name','category_name']].                            astype('str').values




train_estimated_brands_train = []
for name, category in tqdm(train_names_unknown_brands_train):
    brand = brandfinder(name, category) 
    train_estimated_brands_train.append(brand)




train_estimated_brands_cv = []
for name, category in tqdm(train_names_unknown_brands_cv):
    brand = brandfinder(name, category) 
    train_estimated_brands_cv.append(brand)




train_estimated_brands_data_test = []
for name, category in tqdm(train_names_unknown_brands_data_test):
    brand = brandfinder(name, category) 
    train_estimated_brands_data_test.append(brand)




X_train.loc[X_train['brand_name'] == 'Unknown', 'brand_name'] = train_estimated_brands_train

X_cv.loc[X_cv['brand_name'] == 'Unknown', 'brand_name'] = train_estimated_brands_cv

data_test.loc[data_test['brand_name'] == 'Unknown', 'brand_name'] = train_estimated_brands_data_test




max_brand = 2500
max_category_name = 1000
max_category = 1000
name_min_df =10
max_item_description_features = 50000




# cutting brand names
keep_brand = X_train['brand_name'].value_counts().loc[lambda x: x.index != 'Unknown'].index[:max_brand]

X_train.loc[~X_train['brand_name'].isin(keep_brand), 'brand_name'] = 'Unknown'
X_cv.loc[~X_cv['brand_name'].isin(keep_brand), 'brand_name'] = 'Unknown'
data_test.loc[~data_test['brand_name'].isin(keep_brand), 'brand_name'] = 'Unknown'




# cutting category
keep_category1 = X_train['main_category'].value_counts().loc[lambda x: x.index != 'Unknown'].index[:max_category]
keep_category2 = X_train['sub_category_1'].value_counts().loc[lambda x: x.index != 'Unknown'].index[:max_category]
keep_category3 = X_train['sub_category_2'].value_counts().loc[lambda x: x.index != 'Unknown'].index[:max_category]

X_train.loc[~X_train['main_category'].isin(keep_category1), 'main_category'] = 'Unknown'
X_train.loc[~X_train['sub_category_1'].isin(keep_category2), 'sub_category_1'] = 'Unknown'
X_train.loc[~X_train['sub_category_2'].isin(keep_category3), 'sub_category_1'] = 'Unknown'
                                                             
X_cv.loc[~X_cv['main_category'].isin(keep_category1), 'main_category'] = 'Unknown'
X_cv.loc[~X_cv['sub_category_1'].isin(keep_category2), 'sub_category_1'] = 'Unknown'
X_cv.loc[~X_cv['sub_category_2'].isin(keep_category3), 'sub_category_1'] = 'Unknown'

data_test.loc[~data_test['main_category'].isin(keep_category1), 'main_category'] = 'Unknown'
data_test.loc[~data_test['sub_category_1'].isin(keep_category2), 'sub_category_1'] = 'Unknown'
data_test.loc[~data_test['sub_category_2'].isin(keep_category3), 'sub_category_1'] = 'Unknown'




# item_description...TfidfVectorizer
tv = TfidfVectorizer(max_features=max_item_description_features,ngram_range=(1, 3),token_pattern=r'(?u)\b\w+\b',stop_words='english')
#tv = TfidfVectorizer(max_features=50000, ngram_range=(1, 3),token_pattern=r'(?u)\b\w+\b',stop_words='english')
tv.fit(X_train['item_description'])
X_train_item_description_tfidf = tv.transform(X_train['item_description'])
X_cv_item_description_tfidf = tv.transform(X_cv['item_description'])
data_test_item_description_tfidf = tv.transform(data_test['item_description'])




# name.......CountVectorizer
#cv = CountVectorizer(min_df=name_min_df,ngram_range=(1, 2),stop_words='english')
#tv = TfidfVectorizermin_df=name_min_df,ngram_range=(1, 2),stop_words='english')
tv.fit(X_train['name'])
X_train_name_tfidf = tv.transform(X_train['name'])
X_cv_name_tfidf = tv.transform(X_cv['name'])
data_test_name_tfidf = tv.transform(data_test['name'])




# category_name......CountVectorizer
cv = CountVectorizer(min_df=name_min_df)

#category_name(used in lgbm)
cv.fit(X_train['category_name'].astype('category'))
X_train_category =cv.transform(X_train['category_name'].astype('category'))
X_cv_category =cv.transform(X_cv['category_name'].astype('category'))
data_test_category =cv.transform(data_test['category_name'].astype('category'))

##################################################################################################################
#main_category
#cv.fit(X_train['main_category'].astype('category'))
#X_train_main_category =cv.transform(X_train['main_category'].astype('category'))
#X_cv_main_category =cv.transform(X_cv['main_category'].astype('category'))
#data_test_main_category =cv.transform(data_test['main_category'].astype('category'))

#sub_category_1
#cv.fit(X_train['sub_category_1'].astype('category'))
#X_train_sub_category_1 =cv.transform(X_train['sub_category_1'].astype('category'))
#X_cv_sub_category_1 =cv.transform(X_cv['sub_category_1'].astype('category'))
#data_test_sub_category_1 =cv.transform(data_test['sub_category_1'].astype('category'))

#sub_category_2
#cv.fit(X_train['sub_category_2'].astype('category'))
#X_train_sub_category_2 =cv.transform(X_train['sub_category_2'].astype('category'))
#X_cv_sub_category_2 =cv.transform(X_cv['sub_category_2'].astype('category'))
#data_test_sub_category_2 =cv.transform(data_test['sub_category_2'].astype('category'))




# brand_name.....LabelBinarizer
lb = LabelBinarizer(sparse_output=True)
lb.fit(X_train['brand_name'].astype('category'))
X_train_brand = lb.transform(X_train['brand_name'].astype('category'))
X_cv_brand = lb.transform(X_cv['brand_name'])
data_test_brand = lb.transform(data_test['brand_name'])




# shipping...pd.getdummies
# item_condition id...pd.getdummies
#X_train['item_condition_id'] = X_train['item_condition_id'].astype('category')
#X_test['item_condition_id'] = X_test['item_condition_id'].astype('category')

X_train_dummies = csr_matrix(pd.get_dummies(X_train[['item_condition_id', 'shipping']], sparse=True).values)
X_cv_dummies = csr_matrix(pd.get_dummies(X_cv[['item_condition_id', 'shipping']], sparse=True).values)
data_test_dummies = csr_matrix(pd.get_dummies(data_test[['item_condition_id', 'shipping']], sparse=True).values)




# word count for item description
X_train_word_count_item_desc = X_train['item_description'].apply(lambda x: wordCount_with_cleaning(x))
X_cv_word_count_item_desc = X_cv['item_description'].apply(lambda x: wordCount_with_cleaning(x))
data_test_word_count_item_desc = data_test['item_description'].apply(lambda x: wordCount_with_cleaning(x))




# word count for name
X_train_word_count_name = X_train['name'].apply(lambda x: wordCount_with_cleaning(x))
X_cv_word_count_name = X_cv['name'].apply(lambda x: wordCount_with_cleaning(x))
data_test_word_count_name = data_test['name'].apply(lambda x: wordCount_with_cleaning(x))




#check the shapes
print(X_train_item_description_tfidf.shape)
print(X_train_name_tfidf.shape)
print(X_train_category.shape)
#print(X_train_main_category.shape)
#print(X_train_sub_category_1.shape)
#print(X_train_sub_category_2.shape)
print(X_train_brand.shape)
print(X_train_dummies.shape)
print(X_train_word_count_item_desc.shape)
print(X_train_word_count_name.shape)




#check the shapes
print(X_cv_item_description_tfidf.shape)
print(X_cv_name_tfidf.shape)
print(X_cv_category.shape)
#print(X_cv_main_category.shape)
#print(X_cv_sub_category_1.shape)
#print(X_cv_sub_category_2.shape)
print(X_cv_brand.shape)
print(X_cv_dummies.shape)
print(X_cv_word_count_item_desc.shape)
print(X_cv_word_count_name.shape)




#check the shapes
print(data_test_item_description_tfidf.shape)
print(data_test_name_tfidf.shape)
print(data_test_category.shape)
#print(data_test_main_category.shape)
#print(data_test_sub_category_1.shape)
#print(data_test_sub_category_2.shape)
print(data_test_brand.shape)
print(data_test_dummies.shape)
print(data_test_word_count_item_desc.shape)
print(data_test_word_count_name.shape)




# stacking features vectors together
X_train_vectorized = hstack((X_train_item_description_tfidf,                             X_train_name_tfidf,                             X_train_category,                             X_train_brand,                             X_train_dummies,                             X_train_word_count_item_desc.values.reshape(-1,1),                             X_train_word_count_name.values.reshape(-1,1))).tocsr()




X_train_vectorized.shape




# stacking features vectors together
#X_train_vectorized_with_sub_cat = hstack((X_train_item_description_tfidf,\
#                             X_train_name_tfidf,\
#                             X_train_main_category,\
#                             X_train_sub_category_1,\
#                             X_train_sub_category_2,\
#                             X_train_brand,\
#                             X_train_dummies,\
#                             X_train_word_count_item_desc.values.reshape(-1,1),\
#                             X_train_word_count_name.values.reshape(-1,1))).tocsr()




#X_train_vectorized_with_sub_cat.shape




# stacking features vectors together
X_cv_vectorized = hstack((X_cv_item_description_tfidf,                             X_cv_name_tfidf,                             X_cv_category,                             X_cv_brand,                             X_cv_dummies,                             X_cv_word_count_item_desc.values.reshape(-1,1),                             X_cv_word_count_name.values.reshape(-1,1))).tocsr()




X_cv_vectorized.shape




# stacking features vectors together
#X_cv_vectorized_with_sub_cat = hstack((X_cv_item_description_tfidf,\
#                             X_cv_name_tfidf,\
#                             X_cv_main_category,\
#                             X_cv_sub_category_1,\
#                             X_cv_sub_category_2,\
#                             X_cv_brand,\
#                             X_cv_dummies,\
#                             X_cv_word_count_item_desc.values.reshape(-1,1),\
#                             X_cv_word_count_name.values.reshape(-1,1))).tocsr()




#X_cv_vectorized_with_sub_cat.shape




# stacking features vectors together
data_test_vectorized = hstack((data_test_item_description_tfidf,                            data_test_name_tfidf,                            data_test_category,                            data_test_brand,                            data_test_dummies,                            data_test_word_count_item_desc.values.reshape(-1,1),                            data_test_word_count_name.values.reshape(-1,1))).tocsr()




data_test_vectorized.shape




# stacking features vectors together
#data_test_vectorized_with_sub_cat = hstack((data_test_item_description_tfidf,\
#                             data_test_name_tfidf,\
#                             data_test_main_category,\
#                             data_test_sub_category_1,\
#                             data_test_sub_category_2,\
#                             data_test_brand,\
#                             data_test_dummies,\
#                             data_test_word_count_item_desc.values.reshape(-1,1),\
#                             data_test_word_count_name.values.reshape(-1,1))).tocsr()




#data_test_vectorized_with_sub_cat.shape




import gc
del(data_train,preprocessed_item_description,preprocessed_test_item_description,X,y)
del(train_estimated_brands_train,train_estimated_brands_cv,train_estimated_brands_data_test,keep_brand)
del(data_test_item_description_tfidf,
    data_test_name_tfidf,\
    data_test_category,\
    #data_test_main_category,\
    #data_test_sub_category_1,\
    #data_test_sub_category_2,\
    data_test_brand,\
    data_test_dummies,\
    data_test_word_count_item_desc,\
    data_test_word_count_name)
del(X_train_item_description_tfidf,
    X_train_name_tfidf,\
    X_train_category,\
    #X_train_main_category,\
    #X_train_sub_category_1,\
    #X_train_sub_category_2,\
    X_train_brand,\
    X_train_dummies,\
    X_train_word_count_item_desc,\
    X_train_word_count_name)
del(X_cv_item_description_tfidf,
    X_cv_name_tfidf,\
    X_cv_category,\
    #X_cv_main_category,\
    #X_cv_sub_category_1,\
    #X_cv_sub_category_2,\
    X_cv_brand,\
    X_cv_dummies,\
    X_cv_word_count_item_desc,\
    X_cv_word_count_name)
gc.collect()




def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))




Error_dict  = {}




import lightgbm as lgb


params = {
         'colsample_bytree': 0.42799939792816927,
          'max_depth': 10,
          'min_child_samples': 370,
          'min_child_weight': 0.01,
          'num_leaves': 49,
          'reg_lambda': 5,
          'subsample': 0.6739316550896339,
          'learning_rate':0.1,
          'reg_alpha' :0.5,
          'boosting_type': 'gbdt',
          'objective' : 'regression',
          'metric' : 'RMSE',
          'verbosity': -1
         }

d_train = lgb.Dataset(X_train_vectorized, label=Y_train)
d_valid = lgb.Dataset(X_cv_vectorized, label=Y_cv)
watchlist = [d_train, d_valid]

model_1 = lgb.train(params, train_set=d_train,valid_sets=watchlist,num_boost_round=2000,verbose_eval=200,early_stopping_rounds=100) 
#d_train = lgb.Dataset(X_train_vectorized, label=Y_train)
#clf = lgb.train(params, d_train)




lgbm_1_pred_train=model_1.predict(X_train_vectorized)
lgbm_1_pred_cv=model_1.predict(X_cv_vectorized)
lgbm_1_pred_data_test=model_1.predict(data_test_vectorized)




print("Train rmsle: "+str(rmsle(10 ** Y_train-1, 10 ** lgbm_1_pred_train-1)))
print("CV rmsle: "+str(rmsle(10 ** Y_cv-1, 10 ** lgbm_1_pred_cv-1)))




Error_dict['lgbm_1_Train_rmsle'] = rmsle(10 ** Y_train-1, 10 ** lgbm_1_pred_train-1) 
Error_dict['lgbm_1_cv_rmsle'] = rmsle(10 ** Y_cv-1, 10 ** lgbm_1_pred_cv-1) 




# Ridge model
from sklearn.linear_model import Ridge
reg_ridge = Ridge(solver='sag', alpha=5)
reg_ridge.fit(X_train_vectorized, Y_train)




reg_ridge_pred_train=reg_ridge.predict(X_train_vectorized)
reg_ridge_pred_cv=reg_ridge.predict(X_cv_vectorized)
reg_ridge_pred_data_test=reg_ridge.predict(data_test_vectorized)




print("Train rmsle: "+str(rmsle(10 ** Y_train-1, 10 ** reg_ridge_pred_train-1)))
print("CV rmsle: "+str(rmsle(10 ** Y_cv-1, 10 ** reg_ridge_pred_cv-1)))




Error_dict['reg_ridge_Train_rmsle'] = rmsle(10 ** Y_train-1, 10 ** reg_ridge_pred_train-1) 
Error_dict['reg_ridge_Cv_rmsle'] = rmsle(10 ** Y_cv-1, 10 ** reg_ridge_pred_cv-1 ) 




stacked_pred_train = pd.DataFrame(index=np.array(range(X_train.shape[0])))
stacked_pred_cv = pd.DataFrame(index=np.array(range(X_cv.shape[0])))
stacked_pred_data_test = pd.DataFrame(index=np.array(range(data_test.shape[0])))

stacked_pred_train.insert(0,"lgbm_model_pred",lgbm_1_pred_train)
stacked_pred_train.insert(1,"Ridge_model_pred",reg_ridge_pred_train)

stacked_pred_cv.insert(0,"lgbm_model_pred",lgbm_1_pred_cv)
stacked_pred_cv.insert(1,"Ridge_model_pred",reg_ridge_pred_cv)

stacked_pred_data_test.insert(0,"lgbm_model_pred",lgbm_1_pred_data_test)
stacked_pred_data_test.insert(1,"Ridge_model_pred",reg_ridge_pred_data_test)


#stacked_pred_train = (lgbm_1_pred_train + reg_ridge_pred_train)/2
#stacked_pred_cv = (lgbm_1_pred_cv + reg_ridge_pred_cv)/2
#stacked_pred_data_test = (lgbm_1_pred_data_test + reg_ridge_pred_data_test)/2

#final_pred_train = (lgbm_1_pred_train + lgbm_2_pred_train)/2
#final_pred_test = (lgbm_1_pred_test + lgbm_2_pred_test)/2




X_train_stacked = sparse.csr_matrix(hstack([X_train_vectorized, sparse.csr_matrix(stacked_pred_train)]))
X_cv_stacked = sparse.csr_matrix(hstack([X_cv_vectorized, sparse.csr_matrix(stacked_pred_cv)]))
data_test_stacked = sparse.csr_matrix(hstack([data_test_vectorized, sparse.csr_matrix(stacked_pred_data_test)]))




import lightgbm as lgb


params = {
         'colsample_bytree': 0.42799939792816927,
          'max_depth': 10,
          'min_child_samples': 370,
          'min_child_weight': 0.01,
          'num_leaves': 49,
          'reg_lambda': 5,
          'subsample': 0.6739316550896339,
          'learning_rate':0.1,
          'reg_alpha' :0.5,
          'boosting_type': 'gbdt',
          'objective' : 'regression',
          'metric' : 'RMSE',
          'verbosity': -1
         }

d_train = lgb.Dataset(X_train_stacked, label=Y_train)
d_valid = lgb.Dataset(X_cv_stacked, label=Y_cv)
watchlist = [d_train, d_valid]

final_lgbm = lgb.train(params, train_set=d_train,valid_sets=watchlist,num_boost_round=2000,verbose_eval=200,early_stopping_rounds=100) 




final_pred_train=final_lgbm.predict(X_train_stacked)
final_pred_cv=final_lgbm.predict(X_cv_stacked)
final_pred_data_test=final_lgbm.predict(data_test_stacked)




print("Train rmsle: "+str(rmsle(10 ** Y_train-1, 10 ** final_pred_train-1)))
print("CV rmsle: "+str(rmsle(10 ** Y_cv-1, 10 ** final_pred_cv-1)))




submission = pd.DataFrame()
submission['test_id'] = data_test['test_id']

lightgbm_submission = submission.copy()
lightgbm_submission['price'] = pd.DataFrame(10 ** final_pred_data_test - 1)
lightgbm_submission.to_csv('stacked_submission_1.csv', index=False)

