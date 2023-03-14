#!/usr/bin/env python
# coding: utf-8

# In[1]:


from io import BytesIO
import requests
from PIL import Image
response = requests.get("https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2F1.bp.blogspot.com%2F-qKqxEyuXEQo%2FVD71mHi8sDI%2FAAAAAAAAPhg%2FVwDyvAlXDnY%2Fs1600%2Frgicontransparent.png&f=1&nofb=1")
img = Image.open(BytesIO(response.content))
img.resize((300,300), Image.ANTIALIAS)


# In[2]:


# Ignore all your warnings
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

# Loading Libraries
import datetime
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
import pylab 
import scipy.stats as stats
from scipy.stats import boxcox

import re
import pickle
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import os
from wordcloud import WordCloud
from matplotlib_venn import venn2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob


# In[3]:


response = requests.get("https://storage.googleapis.com/kaggle-media/competitions/google-research/human_computable_dimensions_1.png")
Image.open(BytesIO(response.content))


# In[4]:


# This image is created by me for better understanding of problem
response = requests.get("https://i.postimg.cc/NFNFYPPG/regression.png")
Image.open(BytesIO(response.content))


# In[5]:


import os 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[6]:


train_df = pd.read_csv("../input/google-quest-challenge/train.csv")
test_df = pd.read_csv("../input/google-quest-challenge/test.csv")
sample_submission = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")


# In[7]:


# train_df data
print(f"shape of train_df: {train_df.shape} \n{'='*50}")
train_df.head(2)


# In[8]:


# test_df data
print(f"shape of test_df: {test_df.shape} \n{'='*50}")
test_df.head(2)


# In[9]:


target_vars = sample_submission.columns[1:]
for idx,target in enumerate(target_vars):
    print(idx+1,":",target)


# In[10]:


# sample of Target variable
train_df[target_vars].head()


# In[11]:


x_columns = [columns for columns in train_df.columns if columns not in sample_submission.columns[1:]]
for idx,x_var in enumerate(x_columns):
    print(idx+1,":",x_var)


# In[12]:


# sample of dependent variables
train_df[x_columns].head()


# In[13]:


# train_df data
print(f"shape of train_df: {train_df.shape} \n{'='*50}")


# In[14]:


# This includes both independent variables and target variables
train_df.info()


# In[15]:


# sample
sample=train_df.iloc[0]
sample_question = sample[['qa_id', 'question_title', 'question_body']]
sample_answer = sample[['answer']]
sample_question_target_labels = sample[['question_asker_intent_understanding',
       'question_body_critical', 'question_conversational',
       'question_expect_short_answer', 'question_fact_seeking',
       'question_has_commonly_accepted_answer',
       'question_interestingness_others', 'question_interestingness_self',
       'question_multi_intent', 'question_not_really_a_question',
       'question_opinion_seeking', 'question_type_choice',
       'question_type_compare', 'question_type_consequence',
       'question_type_definition', 'question_type_entity',
       'question_type_instructions', 'question_type_procedure',
       'question_type_reason_explanation', 'question_type_spelling',
       'question_well_written']]
sample_answer_target_labels = sample[['answer_helpful',
       'answer_level_of_information', 'answer_plausible', 'answer_relevance',
       'answer_satisfaction', 'answer_type_instructions',
       'answer_type_procedure', 'answer_type_reason_explanation',
       'answer_well_written']]


# In[16]:


for i in sample_question.index:
    print(i,":",sample_question[i],"\n")


# In[17]:


print(sample_answer[0])


# In[18]:


print(sample_question_target_labels)


# In[19]:


print(sample_answer_target_labels)


# In[20]:


target_vars = sample_submission.columns[1:]
for idx,target in enumerate(target_vars):
    print(idx+1,":",target)


# In[21]:


plt.figure(figsize=(28,20))
for idx,target in enumerate(target_vars): 
    sns.distplot(train_df[target],ax=plt.subplot(5,6,idx+1))
    plt.grid()
plt.show()


# In[22]:


corr = train_df[target_vars].corr()

# plot the heatmap
plt.figure(figsize=(16,14))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,      
        vmin=-1, vmax=1, center=0,)
plt.show()


# In[23]:


## Distribution plots for highly positive Correlated labels
high_pos_corr_feat = [['answer_type_procedure','question_fact_seeking'],
["question_type_instructions","answer_type_instructions"],
['question_type_reason_explanation','answer_type_reason_explanation'],
['question_interestingness_self','question_interestingness_others'],
['answer_level_of_information','answer_helpful'],
['answer_plausible','answer_helpful'],
['answer_satisfaction','answer_helpful']]

print("****Distribtion Plots for postitive Correlation Target labels****\n\n")
for idx, feat in enumerate(high_pos_corr_feat):
    print(f"plot: {idx+1} ")
    plt.figure(figsize=(8,4))
    sns.distplot(train_df[f'{feat[0]}'], label=f"{feat[0]}")
    sns.distplot(train_df[f'{feat[1]}'], label=f"{feat[1]}")
    plt.legend()
    plt.xlabel(None)
    plt.title(f"Distribuion of  {feat[0]}  V/S  {feat[1]}\n")
    plt.grid()
    plt.show()


## Distribution plots for highly negative Correlated labels
high_neg_corr_feat = [['question_fact_seeking','question_opinion_seeking']]

print("\n\n****Distribtion Plots for high Negative Correlation Target labels****\n")
for idx, feat in enumerate(high_neg_corr_feat):
    print(f"plot: {idx+1} ")
    plt.figure(figsize=(8,4))
    sns.distplot(train_df[f'{feat[0]}'], label=f"{feat[0]}")
    sns.distplot(train_df[f'{feat[1]}'], label=f"{feat[1]}")
    plt.legend()
    plt.xlabel(None)
    plt.title(f"Distribuion of  {feat[0]}  V/S  {feat[1]}\n")
    plt.grid()
    plt.show()


# In[24]:


# Utility function to plot lineplot and distplot using seaborn
def plot_sns(data,feature,color='lightblue',title=None,subtitle=None):
    
    """   
    Utility function to plot lineplot and distplot using seaborn
    
    plot_sns(data,feature,color='lightblue',title=None,subtitle=None):
    
    data = data 
    feature = coulum name
    color = color of plot
    title = Either 'length' or 'number' based on which to plot. Otherwise by default='None'
    subtitle = Either 'train_df' or 'test_df'. Otherwise by default='None'  
    
    """    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))
    
    # line plot
    sns.lineplot(np.arange(len(data)),data,ax=ax1,color=color)    
    if title=='number':
        ax1.set(xlabel=f"Idx of {feature}", ylabel=f"Number of words in {feature}", title=f'Number of words in {feature} in {subtitle}\n')
    elif title=='length':
        ax1.set(xlabel=f"Idx of {feature}", ylabel=f"Length of {feature}", title=f'Length of {feature} in {subtitle}\n')   
    ax1.grid()

    # distribution plot
    sns.distplot(data,ax=ax2,color=color)
    if title=='number':
        ax2.set(xlabel=f"Idx of {feature}", ylabel=f"Number of words in {feature}", title=f'Number of words in {feature} in {subtitle}\n')
    elif title=='length':
        ax2.set(xlabel=f"Idx of {feature}", ylabel=f"Length of {feature}", title=f'Length of {feature} in {subtitle}\n')   
    ax2.grid()
    plt.show()

#=======================================================================================================================================================================================    
# Utility function to plot bar graph for both train and test using seaborn
def plot_bar(train_data,test_data,feature=None,x_label=None, y_label=None):
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))

    # for train_df
    sns.barplot(train_data,np.arange(len(train_data)),ax=ax1)
    ax1.set(xlabel=f"{x_label}", ylabel=f"{y_label} {feature}", title='train_df\n')
    ax1.grid()
    
    # for test_df
    sns.barplot(test_data,np.arange(len(test_data)),ax=ax2)
    ax2.set(xlabel=f"{x_label}", ylabel=f"{y_label} {feature}", title='test_df\n')
    ax2.grid()
    plt.show()
    
#=======================================================================================================================================================================================  
# Utility function to plot requency of most popular words
def word_frequency_plot(dataframe, title=None):
    list_of_all_words = []
    for sent in dataframe:
        list_of_all_words.extend(sent.split())

    top_50_words = pd.Series(list_of_all_words).value_counts()[:50]
    top_50_words_prob_dist = top_50_words.values/sum(top_50_words.values)

    #  plot of frequency of polpular words in train
    plt.figure(figsize=(16,7))
    sns.barplot(top_50_words.index, top_50_words_prob_dist)
    plt.xlabel("words")
    plt.ylabel("frequency")
    plt.title(f"Frequency of most popular words {title}\n")
    plt.xticks(rotation=70)
    plt.grid()
    plt.show()

#=======================================================================================================================================================================================
# Utility function to check if feature or variable follows Normal distribution using Q-Q Plot   
def q_q_plot(train_data, test_data, feature_name=None):
    """
    # code refer: https://stackoverflow.com/a/13865874
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    
    measurements = train_data
    stats.probplot(measurements, dist="norm", plot=ax1)
    ax1.set(title=f'train : Q-Q Plot for {feature_name} \n')

    measurements = test_data
    stats.probplot(measurements, dist="norm", plot=ax2)
    ax2.set(title=f'test : Q-Q Plot for {feature_name} \n')
    plt.show()
        
#=======================================================================================================================================================================================    
# Utility function for box plot
def box_plot(train_data, test_data, feature_name=None):
    
    # for train data
    plt.figure(figsize=(26,4))
    sns.violinplot(train_data,color='darkred')
    plt.title(f'Train : violinplot Plot for {feature_name} \n')
    plt.xlabel(f"{feature_name}")
    plt.ylabel(f"Distribution")   
    plt.grid()
    plt.show()
    
    # for test data
    plt.figure(figsize=(26,4))
    sns.violinplot(test_data,color='orangered')
    plt.title(f'Test : violinplot Plot for {feature_name} \n')
    plt.xlabel(f"{feature_name}")
    plt.ylabel(f"Distribution")   
    plt.grid()
    plt.show()


# In[25]:


#Counts of repeated question_title in train
pd.DataFrame(train_df['question_title'].value_counts())


# In[26]:


#Counts of repeated question_title in test
pd.DataFrame(test_df['question_title'].value_counts())


# In[27]:


# Number of repeated question_title in train
n_repeated_question_title_train = train_df['question_title'].value_counts().values

# Number of repeated question_title in test
n_repeated_question_title_test = test_df['question_title'].value_counts().values

# plot for Number of repeated question_title in train and test
plot_bar(n_repeated_question_title_train, n_repeated_question_title_test,feature='question_title',x_label="Number of times repeated same question ",y_label="Counts of repeated")


# In[28]:


# Length of question_title in train
len_question_title_train = sorted(train_df['question_title'].apply(lambda x: len(x)),reverse=True)

# Length of question_title in test
len_question_title_test = sorted(test_df['question_title'].apply(lambda x: len(x)),reverse=True)

# plot for train_df
plot_sns(len_question_title_train,"question_title",color='darkblue',title='length',subtitle='train_df')

# plot for test_df
plot_sns(len_question_title_test,"question_title",color='lightblue',title='length',subtitle='test_df')


# In[29]:


# Box plot of Length of question_title in train and test
box_plot(len_question_title_train, len_question_title_test, "question_title")


# In[30]:


# Checking weather len_question_title follows normal distribution using Q-Q plot
q_q_plot(len_question_title_train, len_question_title_test, "len_question_title")


# In[31]:


# number of words in question_title in train
n_words_in_question_title_train = sorted(train_df['question_title'].apply(lambda x: len(x.split(" "))),reverse=True)

# number of words in question_title in test
n_words_in_question_title_test = sorted(test_df['question_title'].apply(lambda x: len(x.split(" "))),reverse=True)


# In[32]:


# plot for train_df
plot_sns(n_words_in_question_title_train,"question_title",color='darkred',title='number',subtitle='train_df')

# plot for test_df
plot_sns(n_words_in_question_title_train,"question_title",color='orangered',title='number',subtitle='test_df')


# In[33]:


# Box plot of Length of question_title in train and test
box_plot(n_words_in_question_title_train, n_words_in_question_title_test, "question_title")


# In[34]:


# Checking weather  n_words_in_question_title follows normal distribution using Q-Q plot
q_q_plot(n_words_in_question_title_train, n_words_in_question_title_test, "n_words_in_question_title")


# In[35]:


# refer: https://www.datacamp.com/community/tutorials/wordcloud-python

# For train_df
text_train = " ".join(word for word in train_df['question_title'])

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text_train)

# Display the generated image:
plt.figure(figsize=(9,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("WordCloud of question_title in train\n")
plt.axis("off")
plt.show()

#===================================================================

# For test_df
text_test = " ".join(word for word in test_df['question_title'])

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text_test)

# Display the generated image:
plt.figure(figsize=(9,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("WordCloud of question_title in test")
plt.axis("off")
plt.show()


# In[36]:


# Frequency of most popular 50 words in train_df
word_frequency_plot(train_df['question_title'], title='train')

# Frequency of most popular words in test_df
word_frequency_plot(test_df['question_title'], title='test')


# In[37]:


plt.figure(figsize=(9,5))
venn2([set(train_df['question_title'].unique()), set(test_df['question_title'].unique())], set_labels = ('Train set', 'Test set') )
plt.title("Common question_title in training and test data", fontsize=15)
plt.show()


# In[38]:


#Counts of repeated questions in train
train_df['question_body'].value_counts().values


# In[39]:


#Counts of repeated questions in test
test_df['question_body'].value_counts().values


# In[40]:


# Number of repeated question_body in train
n_repeated_question_body_train = train_df['question_body'].value_counts().values

# Number of repeated question_body in test
n_repeated_question_body_test = test_df['question_body'].value_counts().values

# plot for Number of repeated question_body in train and test
plot_bar(n_repeated_question_body_train, n_repeated_question_body_test,feature='question_title',x_label="Number of times repeated same question ",y_label="Counts of repeated")


# In[41]:


# Length of question_title in train
len_question_body_train = sorted(train_df['question_body'].apply(lambda x: len(x)),reverse=True)

# Length of question_title in test
len_question_body_test = sorted(test_df['question_body'].apply(lambda x: len(x)),reverse=True)

# plot for train_df
plot_sns(len_question_body_train,"question_body",color='darkblue',title='length',subtitle='train_df')

# plot for test_df
plot_sns(len_question_body_test,"question_body",color='lightblue',title='length',subtitle='test_df')


# In[42]:


# Box plot of length of question_body
box_plot(len_question_body_train, len_question_body_test, "len_question_body" )


# In[43]:


box_cox_len_question_body_train = boxcox(len_question_body_train)[0]
box_cox_len_question_body_test =  boxcox(len_question_body_test)[0]

train_df['box_cox_len_question_body'] = box_cox_len_question_body_train
test_df['box_cox_len_question_body'] = box_cox_len_question_body_test

# Checking weather box cox transformed len_question_body_box_cox follows normal distribution or not using Q-Q plot
q_q_plot(box_cox_len_question_body_train, box_cox_len_question_body_test, "box-cox transformed length of question_body ")


# In[44]:


# number of words in question_title in train
n_words_in_question_body_train = sorted(train_df['question_body'].apply(lambda x: len(x.split(" "))),reverse=True)

# number of words in question_title in test
n_words_in_question_body_test = sorted(test_df['question_body'].apply(lambda x: len(x.split(" "))),reverse=True)

# plot for train_df
plot_sns(n_words_in_question_body_train,"question_body",color='darkred',title='number',subtitle='train_df')

# plot for test_df
plot_sns(n_words_in_question_body_test,"question_body",color='orangered',title='number',subtitle='test_df')


# In[45]:


# Box plot of length of question_body
box_plot(n_words_in_question_body_train, n_words_in_question_body_test, "n_words_in_question_body" )


# In[46]:


box_cox_n_words_in_question_body_train = boxcox(n_words_in_question_body_train)[0]
box_cox_n_words_in_question_body_test =  boxcox(n_words_in_question_body_test)[0]

# Saving box_cox_n_words_in_question_body as feature
train_df['box_cox_n_words_in_question_body'] = box_cox_n_words_in_question_body_train 
test_df['box_cox_n_words_in_question_body'] = box_cox_n_words_in_question_body_test

# Checking weather box cox transformed len_question_body_box_cox follows normal distribution or not using Q-Q plot
q_q_plot(box_cox_n_words_in_question_body_train, box_cox_n_words_in_question_body_test, "box-cox transformed number of words in question_body ")


# In[47]:


# refer: https://www.datacamp.com/community/tutorials/wordcloud-python

# For train_df
text_train = " ".join(word for word in train_df['question_body'])

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text_train)

# Display the generated image:
plt.figure(figsize=(9,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("WordCloud of question_title in train\n")
plt.axis("off")
plt.show()

#===================================================================

# For test_df
text_test = " ".join(word for word in test_df['question_body'])

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text_test)

# Display the generated image:
plt.figure(figsize=(9,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("WordCloud of question_title in test")
plt.axis("off")
plt.show()


# In[48]:


# Frequency of most popular worlds in question_body of train
word_frequency_plot(train_df['question_body'], title='train')

# Frequency of most popular worlds in question_body of test_df
word_frequency_plot(test_df['question_body'], title='test')


# In[49]:


# for train_df
n_count_question_user_name_train=train_df['question_user_name'].value_counts().values

# for test_df
n_count_question_user_name_test=test_df['question_user_name'].value_counts().values

# plot for Distribution of counts question_user_name in train and test
plot_bar(n_count_question_user_name_train, n_count_question_user_name_test,feature='question_user_name',x_label="Number of questions  ",y_label="Counts of ")


# In[50]:


# for train_df
n_user_unique_question_train = train_df.drop_duplicates(subset=['question_title'])['question_user_name'].value_counts()

# for test_df
n_user_unique_question_test = test_df.drop_duplicates(subset=['question_title'])['question_user_name'].value_counts()

# plot for Which user has most number of unique question based on question_title?
plot_bar(n_user_unique_question_train.values, n_user_unique_question_test.values,feature='question_user_name',x_label="Number of questions  ",y_label="Counts of ")


# In[51]:


# Top 10 user who has asked most number of unique question
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))

sns.barplot(n_user_unique_question_train[:10].values,n_user_unique_question_train[:10].index,ax=ax1)
ax1.set(xlabel = "number of unique question", ylabel=f"top 10  question_user_name", title='train_df\n')
ax1.grid()


sns.barplot(n_user_unique_question_test[:10].values,n_user_unique_question_test[:10].index , ax=ax2)
ax2.set(xlabel = "number of unique question", ylabel=f"top 10  question_user_name", title='test_df\n')
ax2.grid()
plt.show()


# In[52]:


n_user_unique_question_train.describe()


# In[53]:


n_user_unique_question_test.describe()


# In[54]:


print(f"Unique number of question_user_name in Train : {len(train_df['question_user_name'].unique())}")
print(f"Unique number of question_user_name in Test : {len(test_df['question_user_name'].unique())}")


# In[55]:


# Venn plot of  Common unique question_user_name in train and test
# refer: https://www.kaggle.com/codename007/start-from-here-quest-complete-eda-fe

plt.figure(figsize=(9,5))

venn2([set(train_df.question_user_name.unique()), set(test_df.question_user_name.unique())], set_labels = ('Train set', 'Test set') )
plt.title("Common question_user_name in training and test data", fontsize=15)
plt.show()


# In[56]:


# Finding the unique question users asked unique questions based on question_title
unique_question_user_with_unique_questions = pd.DataFrame(train_df.groupby(['question_user_name',"question_title"])["question_title"].unique())
unique_question_user_with_unique_questions.reset_index(level=1,drop=True,inplace=True)
unique_question_user_with_unique_questions.reset_index(inplace=True)

# Find the number of words in each question_title asked by unique question_user
number_of_words = unique_question_user_with_unique_questions['question_title'].apply(lambda x : len(x[0].split()))
unique_question_user_with_unique_questions["number_of_words"] = number_of_words

# Apply aggrigation function( ) to find "sum","min","max","mean","median" of question_title asked by each unique question_user
unique_question_user_with_unique_questions_title_agg_train = unique_question_user_with_unique_questions.groupby('question_user_name')['number_of_words'].agg(["sum","min","max","mean","median"])
unique_question_user_with_unique_questions_title_agg_train = unique_question_user_with_unique_questions_title_agg_train.reset_index(level=0)

# Renaming column names
unique_question_user_with_unique_questions_title_agg_train.rename({'sum': 'sum_title_len', 'min': 'min_title_len', 'max': 'max_title_len', 'mean': 'mean_title_len', 'median': "median_title_len"}, axis=1, inplace=True)

# merging to train dataframe
train_df = pd.merge(left=train_df ,right=unique_question_user_with_unique_questions_title_agg_train ,how='inner',on="question_user_name")


# In[57]:


# Finding the unique question users asked unique questions based on question_title
unique_question_user_with_unique_questions = pd.DataFrame(test_df.groupby(['question_user_name',"question_title"])["question_title"].unique())
unique_question_user_with_unique_questions.reset_index(level=1,drop=True,inplace=True)
unique_question_user_with_unique_questions.reset_index(inplace=True)

# Find the number of words in each question_title asked by unique question_user
number_of_words = unique_question_user_with_unique_questions['question_title'].apply(lambda x : len(x[0].split()))
unique_question_user_with_unique_questions["number_of_words"] = number_of_words

# Apply aggrigation function( ) to find "sum","min","max","mean","median" of question_title asked by each unique question_user
unique_question_user_with_unique_questions_title_agg_test = unique_question_user_with_unique_questions.groupby('question_user_name')['number_of_words'].agg(["sum","min","max","mean","median"])
unique_question_user_with_unique_questions_title_agg_test = unique_question_user_with_unique_questions_title_agg_test.reset_index(level=0)

# Renaming column names
unique_question_user_with_unique_questions_title_agg_test.rename({'sum': 'sum_title_len', 'min': 'min_title_len', 'max': 'max_title_len', 'mean': 'mean_title_len', 'median': "median_title_len"}, axis=1, inplace=True)

# merging to test dataframe
test_df = pd.merge(left=test_df ,right=unique_question_user_with_unique_questions_title_agg_test ,how='inner',on="question_user_name")


# In[58]:


# Ploting user behaviour
user_behaviour_column_on_queston_titile = ['sum_title_len', 'min_title_len', 'max_title_len', 'mean_title_len','median_title_len']

for idx,column in enumerate(user_behaviour_column_on_queston_titile):
    
    train_set = unique_question_user_with_unique_questions_title_agg_train.sort_values(by= column ,ascending=False)[column]
    test_set = unique_question_user_with_unique_questions_title_agg_test.sort_values(by= column ,ascending=False)[column]
    
    print(f"\n{idx+1}: Plot for {column}")
    
    # Ploting lineplot for trainset
    f, (ax1, ax2 , ax3 , ax4 ) = plt.subplots(1, 4, figsize=(24,5))
    
    sns.lineplot(np.arange(len(train_set)), train_set,ax=ax1, color = "darkred")
    ax1.set(xlabel = "Count of unique_users", ylabel=f"{column}", title='train_df\n')
    ax1.grid()

    # Ploting lineplot for testset
    sns.lineplot(np.arange(len(test_set)), test_set,ax=ax2 ,color = "orangered")
    ax2.set(xlabel = "Count of unique_users", ylabel=f"{column}", title='test_df\n')
    ax2.grid()
    
    # Ploting distplot for trainset
    sns.distplot(train_set,ax=ax3, color = "darkred")
    ax3.set(xlabel = "Count of unique_users", ylabel=f"{column}", title='train_df\n')
    ax3.grid()

    # Ploting distplot for testset
    sns.distplot( test_set, ax=ax4 ,color = "orangered")
    ax4.set(xlabel = "Count of unique_users", ylabel=f"{column}", title='test_df\n')
    ax4.grid()
    plt.show()


# In[59]:


# Q-Q Plot of Box Cox transformed user_behaviour_column_on_queston_titile features
for idx,column_name in enumerate(user_behaviour_column_on_queston_titile):
    
    # Box Cox Transform
    boxcox_transformed_feature_train = boxcox(train_df[column_name])[0]
    boxcox_transformed_feature_test = boxcox(test_df[column_name])[0]
    
    # Saving the transformed user_behaviour_column_on_queston_titile column in dataframe
    train_df[f"boxcox_transformed_{column_name}"] = boxcox_transformed_feature_train
    test_df[f"boxcox_transformed_{column_name}"] = boxcox_transformed_feature_test
    
    print(f"\n{idx+1}: Q-Q Plot for box cox transformed feature of {column_name}\n")
    
    # Q-Q Plot of transformed Feature
    q_q_plot(boxcox_transformed_feature_train, boxcox_transformed_feature_test, f"box cox transformed {column_name}")
    


# In[60]:


# Finding the unique question users asked unique questions based on question_title
unique_question_user_with_unique_questions = pd.DataFrame(train_df.groupby(['question_user_name',"question_title"])["question_body"].unique())
unique_question_user_with_unique_questions.reset_index(level=1,drop=True,inplace=True)
unique_question_user_with_unique_questions.reset_index(inplace=True)

# Find the number of words in each question_title asked by unique question_user
number_of_words = unique_question_user_with_unique_questions['question_body'].apply(lambda x : len(x[0].split()))
unique_question_user_with_unique_questions["number_of_words"] = number_of_words

# Apply aggrigation function( ) to find "sum","min","max","mean","median" of question_title asked by each unique question_user
unique_question_user_with_unique_questions_body_agg_train = unique_question_user_with_unique_questions.groupby('question_user_name')['number_of_words'].agg(["sum","min","max","mean","median"])
unique_question_user_with_unique_questions_body_agg_train = unique_question_user_with_unique_questions_body_agg_train.reset_index(level=0)

# Renaming column names
unique_question_user_with_unique_questions_body_agg_train.rename({'sum': 'sum_body_len', 'min': 'min_body_len', 'max': 'max_body_len', 'mean': 'mean_body_len', 'median': "median_body_len"}, axis=1, inplace=True)

# merging to train dataframe
train_df = pd.merge(left=train_df ,right=unique_question_user_with_unique_questions_body_agg_train ,how='inner',on="question_user_name")


# In[61]:


# Finding the unique question users asked unique questions based on question_title
unique_question_user_with_unique_questions = pd.DataFrame(test_df.groupby(['question_user_name',"question_title"])["question_body"].unique())
unique_question_user_with_unique_questions.reset_index(level=1,drop=True,inplace=True)
unique_question_user_with_unique_questions.reset_index(inplace=True)

# Find the number of words in each question_title asked by unique question_user
number_of_words = unique_question_user_with_unique_questions['question_body'].apply(lambda x : len(x[0].split()))
unique_question_user_with_unique_questions["number_of_words"] = number_of_words

# Apply aggrigation function( ) to find "sum","min","max","mean","median" of question_title asked by each unique question_user
unique_question_user_with_unique_questions_body_agg_test = unique_question_user_with_unique_questions.groupby('question_user_name')['number_of_words'].agg(["sum","min","max","mean","median"])
unique_question_user_with_unique_questions_body_agg_test = unique_question_user_with_unique_questions_body_agg_test.reset_index(level=0)

# Renaming column names
unique_question_user_with_unique_questions_body_agg_test.rename({'sum': 'sum_body_len', 'min': 'min_body_len', 'max': 'max_body_len', 'mean': 'mean_body_len', 'median': "median_body_len"}, axis=1, inplace=True)

# merging to test dataframe
test_df = pd.merge(left=test_df ,right=unique_question_user_with_unique_questions_body_agg_test ,how='inner',on="question_user_name")


# In[62]:


unique_question_user_with_unique_questions_body_agg_test


# In[63]:


# Ploting user behaviour
user_behaviour_column_on_queston_body = ['sum_body_len', 'min_body_len', 'max_body_len', 'mean_body_len','median_body_len']

for idx,column in enumerate(user_behaviour_column_on_queston_body):
    
    train_set = unique_question_user_with_unique_questions_body_agg_train.sort_values(by= column ,ascending=False)[column]
    test_set = unique_question_user_with_unique_questions_body_agg_test.sort_values(by= column ,ascending=False)[column]
    
    print(f"\n{idx+1}: Plot for {column}")
    
    # Ploting lineplot for trainset
    f, (ax1, ax2 , ax3 , ax4 ) = plt.subplots(1, 4, figsize=(24,5))
    
    sns.lineplot(np.arange(len(train_set)), train_set,ax=ax1, color = "darkred")
    ax1.set(xlabel = "Count of unique_users", ylabel=f"{column}", title='train_df\n')
    ax1.grid()

    # Ploting lineplot for testset
    sns.lineplot(np.arange(len(test_set)), test_set,ax=ax2 ,color = "orangered")
    ax2.set(xlabel = "Count of unique_users", ylabel=f"{column}", title='test_df\n')
    ax2.grid()
    
    # Ploting distplot for trainset
    sns.distplot(train_set,ax=ax3, color = "darkred")
    ax3.set(xlabel = "Count of unique_users", ylabel=f"{column}", title='train_df\n')
    ax3.grid()

    # Ploting distplot for testset
    sns.distplot( test_set, ax=ax4 ,color = "orangered")
    ax4.set(xlabel = "Count of unique_users", ylabel=f"{column}", title='test_df\n')
    ax4.grid()
    plt.show()


# In[64]:


# Q-Q Plot of Box Cox transformed user_behaviour_column_on_queston_body features
for idx,column_name in enumerate(user_behaviour_column_on_queston_body):
    
    print(f"Feature name: {column_name}")
    
    # Continues the loop if box cox fails to transform
    if sum(train_df[f'{column_name}']<1)>0:
        print(f"{idx+1}: Box Cox Transformation can not be applied on feature '{column_name}' because all the values of data must be positive for transformation\n\n")
        continue

    # Box Cox Transform
    boxcox_transformed_feature_train = boxcox(train_df[column_name])[0]
    boxcox_transformed_feature_test = boxcox(test_df[column_name])[0]
    
             
    # Saving the transformed user_behaviour_column_on_queston_body column in dataframe
    train_df[f"boxcox_transformed_{column_name}"] = boxcox_transformed_feature_train
    test_df[f"boxcox_transformed_{column_name}"] = boxcox_transformed_feature_test
    
    print(f"{idx+1}: Q-Q Plot for box cox transformed feature of {column_name}\n")
    
    # Q-Q Plot of transformed Feature
    q_q_plot(boxcox_transformed_feature_train, boxcox_transformed_feature_test, f"box cox transformed {column_name}")
    


# In[65]:


n_repeated_answer_train = sum(train_df['answer'].value_counts().values>1)
n_repeated_answer_test = sum(test_df['answer'].value_counts().values>1)

print(f"Number of repeated answer in train: {n_repeated_answer_train}")
print(f"Number of repeated answer in test: {n_repeated_answer_test}")


# In[66]:


# Length of answer in train
len_answer_train = sorted(train_df['answer'].apply(lambda x: len(x)),reverse=True)

# Length of answer in test
len_answer_test = sorted(test_df['answer'].apply(lambda x: len(x)),reverse=True)

# plot for train_df
plot_sns(len_answer_train,"answer",color='darkblue',title='length',subtitle='train_df')

# plot for test_df
plot_sns(len_answer_test,"answer",color='lightblue',title='length',subtitle='test_df')


# In[67]:


# Box plot of length of question_body
box_plot(len_answer_train, len_answer_test, "length of answer " )


# In[68]:


box_cox_len_answer_train = boxcox(len_answer_train)[0]
box_cox_len_answer_test =  boxcox(len_answer_test)[0]

train_df['box_cox_len_answer'] = box_cox_len_answer_train
test_df['box_cox_len_answer'] = box_cox_len_answer_test

# Checking weather box cox transformed len_question_body_box_cox follows normal distribution or not using Q-Q plot
q_q_plot(box_cox_len_question_body_train, box_cox_len_question_body_test, "box-cox transformed length of answer  ")


# In[69]:


# number of words in answer in train
n_words_in_answer_train = sorted(train_df['answer'].apply(lambda x: len(x.split(" "))),reverse=True)

# number of words in answer in test
n_words_in_answer_test = sorted(test_df['answer'].apply(lambda x: len(x.split(" "))),reverse=True)

# plot for train_df
plot_sns(n_words_in_answer_train,"answer",color='darkred',title='number',subtitle='train_df')

# plot for test_df
plot_sns(n_words_in_answer_test,"answer",color='orangered',title='number',subtitle='test_df')


# In[70]:


# Box plot of length of question_body
box_plot(n_words_in_answer_train, n_words_in_answer_test, "number of words in answer" )


# In[71]:


box_cox_n_words_in_answer_train = boxcox(n_words_in_answer_train)[0]
box_cox_n_words_in_answer_test =  boxcox(n_words_in_answer_test)[0]

train_df['box_cox_n_words_in_answer'] = box_cox_n_words_in_answer_train
test_df['box_cox_n_words_in_answer'] = box_cox_n_words_in_answer_test

# Checking weather box cox transformed len_question_body_box_cox follows normal distribution or not using Q-Q plot
q_q_plot(box_cox_n_words_in_answer_train, box_cox_n_words_in_answer_test, "box-cox transformed of n_words_in_answer ")


# In[72]:


# refer: https://www.datacamp.com/community/tutorials/wordcloud-python

# For train_df
text_train = " ".join(word for word in train_df['answer'])

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text_train)

# Display the generated image:
plt.figure(figsize=(9,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("WordCloud of answer in train\n")
plt.axis("off")
plt.show()

#===================================================================

# For test_df
text_test = " ".join(word for word in test_df['answer'])

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text_test)

# Display the generated image:
plt.figure(figsize=(9,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("WordCloud of answer in test")
plt.axis("off")
plt.show()


# In[73]:


# Frequency of most popular worlds in answer of train
word_frequency_plot(train_df['answer'], title='train')

# Frequency of most popular worlds in answer of test_df
word_frequency_plot(test_df['answer'], title='test')


# In[74]:


# Number of unique answer_user in train and test
print(f'Number of unique answer_user in train: {len(train_df["answer_user_name"].unique())}')
print(f'Number of unique answer_user in test: {len(test_df["answer_user_name"].unique())}')


# In[75]:


plt.figure(figsize=(9,5))
venn2([set(train_df.answer_user_name.unique()), set(test_df.answer_user_name.unique())], set_labels = ('Train set', 'Test set') )
plt.title("Common  answer_user in both train and test", fontsize=15)
plt.show()


# In[76]:


# for train_df
n_count_answer_user_name_train=train_df['answer_user_name'].value_counts().values

# for test_df
n_count_answer_user_name_test=test_df['answer_user_name'].value_counts().values

# plot for Distribution of counts answer_user_name in train and test
plot_bar(n_count_answer_user_name_train, n_count_answer_user_name_test,feature='answer_user_name',x_label="Number of answer  ",y_label="Counts of ")


# In[77]:


# for train_df
n_user_unique_answer_train = train_df.drop_duplicates(subset=['question_title'])['answer_user_name'].value_counts()

# for test_df
n_user_unique_answer_test = test_df.drop_duplicates(subset=['question_title'])['answer_user_name'].value_counts()

# plot of Which user has answered most number of unique question based on question_title?
plot_bar(n_user_unique_answer_train.values, n_user_unique_answer_test.values,feature='answer_user_name',x_label="Number of questions  ",y_label="Counts of ")


# In[78]:


# Users who has answered the same question more than once but differently
user_answered_same_ques_twice = pd.DataFrame(train_df.groupby(['question_title','answer_user_name'])['answer_user_name'].agg(['count'])).sort_values(by='count',ascending=False)
user_answered_same_ques_twice = user_answered_same_ques_twice.reset_index(level=1)
user_answered_same_ques_twice.head(20)


# In[79]:


print(f' Number of unique users who has answerd the same question more than once: {sum(user_answered_same_ques_twice["count"]>1)} out of {len(train_df["answer_user_name"].unique())} ({round(sum(user_answered_same_ques_twice["count"]>1) / len(train_df["answer_user_name"].unique()),4)})%')


# In[80]:


# Top 10 user who has asked most number of unique question
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))

sns.barplot(n_user_unique_answer_train[:10].values,n_user_unique_answer_train[:10].index,ax=ax1)
ax1.set(xlabel = "number of unique question", ylabel=f"top 10 answer_user_name", title='train_df\n')
ax1.grid()


sns.barplot(n_user_unique_answer_test[:10].values,n_user_unique_answer_test[:10].index , ax=ax2)
ax2.set(xlabel = "number of unique question", ylabel=f"top 10  answer_user_name", title='test_df\n')
ax2.grid()
plt.show()


# In[81]:


n_user_unique_answer_train.describe()


# In[82]:


n_user_unique_answer_test.describe()


# In[83]:


# Find the number of words in each answer 
temp = train_df[['answer_user_name','answer']]
number_of_words = temp['answer'].apply(lambda x : len(x[0].split()))
temp["number_of_words"] = number_of_words

# Apply aggrigation function( ) to find "sum","min","max","mean","median" of answer answered by each unique answer_user
answer_user_agg_behaviour_train = temp.groupby('answer_user_name')['number_of_words'].agg(["sum","min","max","mean","median"])
answer_user_agg_behaviour_train = answer_user_agg_behaviour_train.reset_index(level=0)

# Renaming column names
answer_user_agg_behaviour_train.rename({'sum': 'sum_answer_len', 'min': 'min_answer_len', 'max': 'max_answer_len', 'mean': 'mean_answer_len', 'median': "median_answer_len"}, axis=1, inplace=True)

# merging to train dataframe
train_df = pd.merge(left=train_df ,right=answer_user_agg_behaviour_train ,how='inner',on="answer_user_name")


# In[84]:


# Find the number of words in each answer 
temp = test_df[['answer_user_name','answer']]
number_of_words = temp['answer'].apply(lambda x : len(x[0].split()))
temp["number_of_words"] = number_of_words

# Apply aggrigation function( ) to find "sum","min","max","mean","median" of answer answered by each unique answer_user
answer_user_agg_behaviour_test = temp.groupby('answer_user_name')['number_of_words'].agg(["sum","min","max","mean","median"])
answer_user_agg_behaviour_test = answer_user_agg_behaviour_test.reset_index(level=0)

# Renaming column names
answer_user_agg_behaviour_test.rename({'sum': 'sum_answer_len', 'min': 'min_answer_len', 'max': 'max_answer_len', 'mean': 'mean_answer_len', 'median': "median_answer_len"}, axis=1, inplace=True)

# merging to train dataframe
test_df = pd.merge(left=test_df ,right=answer_user_agg_behaviour_test ,how='inner',on="answer_user_name")


# In[85]:


# Ploting user behaviour
user_behaviour_column_on_answer = ['sum_answer_len', 'sum_answer_len', 'min_answer_len', 'mean_answer_len','median_answer_len']

for idx,column in enumerate(user_behaviour_column_on_answer):
    
    train_set = answer_user_agg_behaviour_train.sort_values(by= column ,ascending=False)[column]
    test_set = answer_user_agg_behaviour_test.sort_values(by= column ,ascending=False)[column]
    
    print(f"\n{idx+1}: Plot for {column}")
    
    # Ploting lineplot for trainset
    f, (ax1, ax2 , ax3 , ax4 ) = plt.subplots(1, 4, figsize=(24,5))
    
    sns.lineplot(np.arange(len(train_set)), train_set,ax=ax1, color = "darkred")
    ax1.set(xlabel = "Count of unique_users", ylabel=f"{column}", title='train_df\n')
    ax1.grid()

    # Ploting lineplot for testset
    sns.lineplot(np.arange(len(test_set)), test_set,ax=ax2 ,color = "orangered")
    ax2.set(xlabel = "Count of unique_users", ylabel=f"{column}", title='test_df\n')
    ax2.grid()
    
    # Ploting distplot for trainset
    sns.distplot(train_set,ax=ax3, color = "darkred")
    ax3.set(xlabel = "Count of unique_users", ylabel=f"{column}", title='train_df\n')
    ax3.grid()

    # Ploting distplot for testset
    sns.distplot( test_set, ax=ax4 ,color = "orangered")
    ax4.set(xlabel = "Count of unique_users", ylabel=f"{column}", title='test_df\n')
    ax4.grid()
    plt.show()


# In[86]:


# Q-Q Plot of Box Cox transformed user_behaviour_column_on_answer features
for idx,column_name in enumerate(user_behaviour_column_on_answer[:2]):
    
    print(f"Feature name: {column_name}")
    
    # Continues the loop if box cox fails to transform
    if sum(train_df[f'{column_name}']<1)>0:
        print(f"{idx+1}: Box Cox Transformation can not be applied on feature '{column_name}' because all the values of data must be positive for transformation\n\n")
        continue

    # Box Cox Transform
    boxcox_transformed_feature_train = boxcox(train_df[column_name])[0]
    boxcox_transformed_feature_test = boxcox(test_df[column_name])[0]
    
             
    # Saving the transformed user_behaviour_column_on_answer column in dataframe
    train_df[f"boxcox_transformed_{column_name}"] = boxcox_transformed_feature_train
    test_df[f"boxcox_transformed_{column_name}"] = boxcox_transformed_feature_test
    
    print(f"{idx+1}: Q-Q Plot for box cox transformed feature of {column_name}\n")
    
    # Q-Q Plot of transformed Feature
    q_q_plot(boxcox_transformed_feature_train, boxcox_transformed_feature_test, f"box cox transformed {column_name}")
    


# In[87]:


# Venn diagram for train_df
plt.figure(figsize=(16,8))
plt.subplot(211)
venn2([set(train_df.question_user_name.unique()), set(train_df.answer_user_name.unique())], set_labels = ('question_user set', 'answer_user set') )
plt.title("Common users who has asked the question and answered by himself in train data", fontsize=15)

# Venn diagram for test_df
plt.subplot(212)
venn2([set(test_df.question_user_name.unique()), set(test_df.answer_user_name.unique())], set_labels = ('question_user set', 'answer_user set') )
plt.title("Common users who has asked the question and answered by himself in test data", fontsize=15)
plt.show()


# In[88]:


# unique categories
print(train_df['category'].unique())


# In[89]:


# For train_df
print(f"Unique number of category: {len(train_df['category'].unique())}\n")

category_dist_df_train = pd.DataFrame(train_df['category'].unique(),columns=['category'])
category_dist_df_train["values_count"] = train_df['category'].value_counts().values
category_dist_df_train["distribution"] = train_df['category'].value_counts().values/sum(train_df['category'].value_counts().values)
category_dist_df_train


# In[90]:


# For test_df
print(f"Unique number of category: {len(test_df['category'].unique())}\n")

category_dist_df_test = pd.DataFrame(test_df['category'].unique(),columns=['category'])
category_dist_df_test["values_count"] = test_df['category'].value_counts().values
category_dist_df_test["distribution"] = test_df['category'].value_counts().values/sum(test_df['category'].value_counts().values)
category_dist_df_test


# In[91]:


f, (ax1, ax2  ) = plt.subplots(1, 2, figsize=(24,7))

# Categories dist for train_df
ax1.pie(category_dist_df_train.values_count, labels=category_dist_df_train.category, shadow=True, autopct='%.1f%%')
ax1.set( title='Categories distribution: train\n')


# Categories dist for test_df
ax2.pie(category_dist_df_test.values_count, labels=category_dist_df_test.category, shadow=True, autopct='%.1f%%')
ax2.set(title='Categories distribution: test\n')
plt.show()


# In[92]:



# Venn diagram for train_df
plt.figure(figsize=(16,8))
plt.subplot(111)
venn2([set(train_df.host.unique()), set(test_df.host.unique())], set_labels = ('Train set', 'Test set') )
plt.title("Common number of Host in train and test", fontsize=15)
plt.show()


# In[93]:


# For train_df
host_dist_df_train = pd.DataFrame(train_df['host'].value_counts().index,columns=['host'])
host_dist_df_train["values_count"] = train_df['host'].value_counts().values
host_dist_df_train["distribution"] = train_df['host'].value_counts().values/sum(train_df['category'].value_counts().values)

# For test_df
host_dist_df_test = pd.DataFrame(test_df['host'].value_counts().index,columns=['host'])
host_dist_df_test["values_count"] = test_df['host'].value_counts().values
host_dist_df_test["distribution"] = test_df['host'].value_counts().values/sum(train_df['category'].value_counts().values)


# In[94]:


print(f"Unique number of host in train: {len(train_df['host'].unique())}\n")

# plot for distribution of host in train
plt.figure(figsize=(28,8))
sns.barplot(x=host_dist_df_train['host'], y=host_dist_df_train['distribution'])
plt.title("Host distribution: Train\n")
plt.xticks(rotation=85)
plt.grid()
plt.show()


# In[95]:


print(f"Unique number of host in train: {len(test_df['host'].unique())}\n")

# plot for distribution of host in train
plt.figure(figsize=(28,8))
sns.barplot(x=host_dist_df_test['host'], y=host_dist_df_test['distribution'])
plt.title("Host distribution: Test\n")
plt.xticks(rotation=85)
plt.grid()
plt.show()


# In[96]:


n_self_question_answer_users_df_train = pd.DataFrame(columns=['n_of_self_question_answer_users'])

for idx, host_name in enumerate(train_df['host'].unique()):
    ques_user =set(train_df[train_df['host']==host_name]['question_user_name'])
    ans_user =set(train_df[train_df['host']==host_name]['answer_user_name'])
    n_self_question_answer_users = len(ques_user.intersection(ans_user))
    n_self_question_answer_users_df_train.loc[host_name] = n_self_question_answer_users

n_self_question_answer_users_df_train.sort_values(by ='n_of_self_question_answer_users' ,ascending=False,inplace=True)


# plot for distribution of top 10 host where number of users  has asked the question and answered by himself
plt.figure(figsize=(16,7))
sns.barplot(x = n_self_question_answer_users_df_train.head(10).index, y = n_self_question_answer_users_df_train['n_of_self_question_answer_users'].head(10))
plt.title("Host distribution: Train\n")
plt.xticks(rotation=85)
plt.grid()
plt.show()


# In[97]:


n_self_question_answer_users_df_test = pd.DataFrame(columns=['n_of_self_question_answer_users'])

for idx, host_name in enumerate(test_df['host'].unique()):
    ques_user =set(test_df[test_df['host']==host_name]['question_user_name'])
    ans_user =set(test_df[test_df['host']==host_name]['answer_user_name'])
    n_self_question_answer_users = len(ques_user.intersection(ans_user))
    n_self_question_answer_users_df_test.loc[host_name] = n_self_question_answer_users

n_self_question_answer_users_df_test.sort_values(by ='n_of_self_question_answer_users' ,ascending=False,inplace=True)


# plot for distribution of top 10 host where number of users  has asked the question and answered by himself
plt.figure(figsize=(16,7))
sns.barplot(x = n_self_question_answer_users_df_test.head(10).index, y = n_self_question_answer_users_df_test['n_of_self_question_answer_users'].head(10))
plt.title("Host distribution: Test\n")
plt.xticks(rotation=85)
plt.grid()
plt.show()


# In[98]:


# refer: https://www.kaggle.com/kabure/qa-eda-and-nlp-modelling-insights-and-data-vis

host = train_df.groupby(['host'])['url'].nunique().sort_values(ascending=False)
category = train_df.groupby(['category'])['url'].nunique().sort_values(ascending=False)

plt.figure(figsize=(14,10))
plt.suptitle('Unique URL by Host and Categories', size=22)

plt.subplot(211)
g0 = sns.barplot(x=category.index, y=category.values, color='blue')
g0.set_title("Unique Questions by category", fontsize=22)
g0.set_xlabel("Category Name", fontsize=19)
g0.set_ylabel("Total Count", fontsize=19)
#g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
for p in g0.patches:
    height = p.get_height()
    g0.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.1f}%'.format(height/category.sum()*100),
            ha="center",fontsize=11) 

plt.subplot(212)
g1 = sns.barplot(x=host[:20].index, y=host[:20].values, color='blue')
g1.set_title("TOP 20 HOSTS with more UNIQUE questions", fontsize=22)
g1.set_xlabel("Host Name", fontsize=19)
g1.set_ylabel("Total Count", fontsize=19)
g1.set_xticklabels(g1.get_xticklabels(),rotation=85)
for p in g1.patches:
    height = p.get_height()
    g1.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.1f}%'.format(height/host.sum()*100),
            ha="center",fontsize=11) 
    
plt.subplots_adjust(hspace = 0.3, top = 0.90)

plt.show()


# In[99]:


# refer: https://www.kaggle.com/kabure/qa-eda-and-nlp-modelling-insights-and-data-vis

import matplotlib.gridspec as gridspec # to do the grid of plots

grid = gridspec.GridSpec(3, 3)
plt.figure(figsize=(16,3*4))

plt.suptitle('Intersection QA USERS \nQuestions and Answers by different CATEGORIES', size=20)

for n, col in enumerate(train_df['category'].value_counts().index):
    ax = plt.subplot(grid[n])
    venn2([set(train_df[train_df.category == col]['question_user_name'].value_counts(dropna=False).index), 
           set(train_df[train_df.category == col]['answer_user_name'].value_counts(dropna=False).index)],
      set_labels=('Question Users', 'Answer Users'), )
    ax.set_title(str(col), fontsize=15)
    ax.set_xlabel('')
    #plt.subplots_adjust(top = 0.98, wspace=.9, hspace=.9)
    
plt.subplots_adjust(top = 0.9, hspace=.1)

plt.show()


# In[100]:


# refer: https://www.kaggle.com/kabure/qa-eda-and-nlp-modelling-insights-and-data-vis

grid = gridspec.GridSpec(5, 3)
plt.figure(figsize=(16,4.5*4))

plt.suptitle('Intersection QA USERS - TOP 15 \nQuestions and Answers by different HOSTS', size=20)
top_host = train_df['host'].value_counts()[:15].index
for n, col in enumerate(top_host):
    ax = plt.subplot(grid[n])
    venn2([set(train_df[train_df.host == col]['question_user_name'].value_counts(dropna=False).index), 
           set(train_df[train_df.host == col]['answer_user_name'].value_counts(dropna=False).index)],
      set_labels=('Question Users', 'Answer Users'), )
    ax.set_title(str(col), fontsize=15)
    ax.set_xlabel('')
    #plt.subplots_adjust(top = 0.98, wspace=.9, hspace=.9)
    
plt.subplots_adjust(top = 0.9, hspace=.1)

plt.show()


# In[101]:


# Tokenize each item in the review column
from nltk import word_tokenize
word_tokens = [word_tokenize(question) for question in train_df.question_body]

# Create an empty list to store the length of the reviews
len_tokens = []

# Iterate over the word_tokens list and determine the length of each item
for i in range(len(word_tokens)):
     len_tokens.append(len(word_tokens[i]))

# Create a new feature for the lengh of each review
train_df['question_n_words'] = len_tokens


# In[102]:


grid = gridspec.GridSpec(5, 3)
plt.figure(figsize=(16,6*4))

plt.suptitle('Title and Question Lenghts by Different Categories \nThe Mean in RED - Also 5% and 95% lines', size=20)
count=0
top_cats = train_df['category'].value_counts().index
for n, col in enumerate(top_cats):
    for i, q_t in enumerate(['question_title', 'question_body', 'question_n_words']):
        ax = plt.subplot(grid[count])
        if q_t == 'question_n_words':
            sns.distplot(train_df[train_df['category'] == col][q_t], bins = 50, 
                         color='g', label="RED - 50%") 
            ax.set_title(f"Distribution of {str(col)} \nQuestion #Total Words Distribution", fontsize=15)
            ax.axvline(train_df[train_df['category'] == col][q_t].quantile(.95))
            ax.axvline(train_df[train_df['category'] == col][q_t].quantile(.05))
            mean_val = train_df[train_df['category'] == col][q_t].mean()
            ax.axvline(mean_val, color='red' )
            ax.set_xlabel('')            
        else:
            sns.distplot(train_df[train_df['category'] == col][q_t].str.len(), bins = 50, 
                         color='g', label="RED - 50%") 
            ax.set_title(f"Distribution of {str(col)} \n{str(q_t)}", fontsize=15)
            ax.axvline(train_df[train_df['category'] == col][q_t].str.len().quantile(.95))
            ax.axvline(train_df[train_df['category'] == col][q_t].str.len().quantile(.05))
            mean_val = train_df[train_df['category'] == col][q_t].str.len().mean()
            ax.axvline(mean_val, color='red' )
            #ax.text(x=mean_val*1.1, y=.02, s='Holiday in US', alpha=0.7, color='#334f8d')
            ax.set_xlabel('')
        count+=1
        
plt.subplots_adjust(top = 0.90, hspace=.4, wspace=.15)
plt.show()


# In[103]:


# Scaling targets or labels with mean=0, and variance=1 (getting targets ready for PCA)
sc=StandardScaler(with_mean=True)
scalar_targets = sc.fit_transform(train_df[target_vars])

# Pca fitting and transform
pca = PCA()
pca_component = pca.fit_transform(scalar_targets)

# variance explained by top 2 eigen vector values
print(f"variance explained by top 2 eigen vector values: {round(sum(pca.explained_variance_ratio_[:2]),2)} %")


# In[104]:


pca__target_component_1 = pca_component[:,0]
pca__target_component_2 = pca_component[:,1]

# Ploting
plt.figure(figsize=(12,7))
sns.scatterplot(pca__target_component_1, pca__target_component_2, hue=train_df['category'])
plt.title("Visualisation of pca__target_component_1 V/S pca__target_component_2 ")
plt.xlabel("pca__target_component_1")
plt.ylabel("pca__target_component_2")
plt.show()


# In[105]:


from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)

for idx, cat in enumerate(train_df['category'].value_counts().index):
    
    f, (ax1, ax2 ) = plt.subplots(1, 2, figsize=(16,5))
    
    wordcloud_question = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=250,
        max_font_size=100, 
        width=400, height=280,
        random_state=42,
    ).generate(" ".join(train_df[train_df['category'] == cat]['question_body'].astype(str)))
    
    print(f" {idx+1}: category {cat}")
    ax1.imshow(wordcloud_question)
    ax1.set(title=f'Category: {cat}\n Question_body')
    ax1.axis('off')
    
    wordcloud_anser = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=250,
        max_font_size=100, 
        width=400, height=280,
        random_state=42,
    ).generate(" ".join(train_df[train_df['category'] == cat]['answer'].astype(str)))
    
    ax2.imshow(wordcloud_anser)
    ax2.set(title=f'Category: {cat}\n Answer')
    ax2.axis('off')


# In[106]:


pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

train_df['q_title_polarity'] = train_df['question_title'].apply(pol)
train_df['q_title_subjectivity'] = train_df['question_title'].apply(sub)

train_df[['question_title', 'category', 'q_title_polarity', 'q_title_subjectivity']].head()


# In[107]:


#Polarity and subjectivity plot
plt.figure(figsize=(12,5))
g = sns.scatterplot(x='q_title_polarity', y='q_title_subjectivity', 
                    data=train_df, hue='category')
g.set_title("Sentiment Analyzis (Polarity x Subjectivity) of question_title by 'Category' Feature", fontsize=16)
g.set_xlabel("Polarity distribution",fontsize=18)
g.set_ylabel("Subjective ",fontsize=18)
plt.show()


# In[108]:


pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

train_df['q_body_polarity'] = train_df['question_body'].apply(pol)
train_df['q_body_subjectivity'] = train_df['question_body'].apply(sub)

train_df[['question_body', 'category', 'q_body_polarity', 'q_body_subjectivity']].head()


# In[109]:


#Polarity and subjectivity plot
plt.figure(figsize=(12,5))
g = sns.scatterplot(x='q_body_polarity', y='q_body_subjectivity', 
                    data=train_df, hue='category')
g.set_title("Sentiment Analyzis (Polarity x Subjectivity) of question_body by 'Category' Feature", fontsize=16)
g.set_xlabel("Polarity distribution",fontsize=18)
g.set_ylabel("Subjective ",fontsize=18)
plt.show()


# In[110]:


pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

train_df['answer_polarity'] = train_df['answer'].apply(pol)
train_df['answer_subjectivity'] = train_df['answer'].apply(sub)

train_df[['answer', 'category', 'answer_polarity', 'answer_subjectivity']].head()


# In[111]:


#Polarity and subjectivity plot
plt.figure(figsize=(12,5))
g = sns.scatterplot(x='answer_polarity', y='answer_subjectivity', 
                    data=train_df, hue='category')
g.set_title("Sentiment Analyzis (Polarity x Subjectivity) of answer by 'Category' Feature", fontsize=16)
g.set_xlabel("Polarity distribution",fontsize=18)
g.set_ylabel("Subjective ",fontsize=18)
plt.show()


# In[112]:


# Saving all the transformed features and behaviour features into pickle file
"""train_df.to_pickle("train_df.pkl")
test_df.to_pickle("test_df.pkl")"""


# In[113]:


all_feat = [col for col in train_df.columns if col not in target_vars]
print(" All the features after all the box cox transformation and user behaviour analyses:\n")
for idx,f in enumerate(all_feat):
    print(f"{idx+1}: {f}")

