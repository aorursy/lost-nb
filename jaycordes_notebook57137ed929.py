#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
print(check_output(["ls", "../working"]).decode("utf8"))




clicks_train = pd.read_csv("../input/clicks_train.csv")
mini_clicks_train = clicks_train.sample(40000, random_state = 0)
mini_clicks_train.to_csv("mini_clicks_train.csv")

#get an error on this, "../input/page_views.csv" does not exist idk
#page_views = pd.read_csv("../input/page_views.csv")
#mini_page_views = page_views[page_views["document_id"].isin(mini_promoted["document_id"])]
#mini_page_views.to_csv("mini_page_views.csv")




promoted_content = pd.read_csv("../input/promoted_content.csv")
mini_promoted = promoted_content[promoted_content["ad_id"].isin(mini_clicks_train["ad_id"])]
mini_promoted.to_csv("mini_promoted.csv")




doc_cats = pd.read_csv("../input/documents_categories.csv")
mini_doc_cats = doc_cats[doc_cats["document_id"].isin(mini_promoted["document_id"])]
mini_doc_cats.to_csv("mini_doc_cats.csv")




doc_ents = pd.read_csv("../input/documents_entities.csv")
mini_doc_ents = doc_ents[doc_ents["document_id"].isin(mini_promoted["document_id"])]
mini_doc_ents.to_csv("mini_doc_ents.csv")




doc_meta = pd.read_csv("../input/documents_meta.csv")
mini_doc_meta = doc_meta[doc_meta["document_id"].isin(mini_promoted["document_id"])]
mini_doc_meta.to_csv("mini_doc_meta.csv")




doc_topics = pd.read_csv("../input/documents_topics.csv")
mini_doc_topics = doc_topics[doc_topics["document_id"].isin(mini_promoted["document_id"])]
mini_doc_topics.to_csv("mini_doc_topics.csv")




events = pd.read_csv("../input/events.csv")
mini_events = events[events["display_id"].isin(mini_clicks_train["display_id"])]
mini_events.to_csv("mini_events.csv")




# Code for full datasets

#clicks_train = pd.read_csv("clicks_train.csv")
#events = pd.read_csv("events.csv") 
#promoted = pd.read_csv("promoted_content.csv")

#doc_cats = pd.read_csv("documents_categories.csv")
#doc_ents = pd.read_csv("documents_entities.csv")
#doc_meta = pd.read_csv("documents_meta.csv")
#doc_topics = pd.read_csv("documents_topics.csv")




Note: This is a Python3 script because that is what Kaggle uses. 




clicks_train = pd.read_csv("mini_clicks_train.csv")#got
doc_cats = pd.read_csv("mini_doc_cats.csv")
doc_ents = pd.read_csv("mini_doc_ents.csv")
doc_meta = pd.read_csv("mini_doc_meta.csv")
doc_topics = pd.read_csv("mini_doc_topics.csv")
events = pd.read_csv("mini_events.csv") #got
#page_views = pd.read_csv("mini_page_views.csv") Once I get this imported
promoted = pd.read_csv("mini_promoted.csv")#got




#clicks_train and events have a 1:1 relationship
print(len(events["display_id"].unique()))
print(len(clicks_train["display_id"].unique()))




#the first column seems to be the old index, we don't need this
clicks_train = clicks_train.set_index('display_id')
del clicks_train["Unnamed: 0"]
clicks_train.head()




del events["Unnamed: 0"]
events = events.set_index("display_id")
events.head()




data = clicks_train.join(events)
data.head()




len(promoted)




#there is not a one-to-one relationship between document_id in promoted and the master data
#This is because the same ad is being shown in different documents I think
print(len(promoted["document_id"].unique()))
print(len(data["document_id"].unique()))




promoted.head()
del promoted["Unnamed: 0"]
del promoted['document_id'] #I think all we want from here is the link between ad_id and campaign id
promoted.head()




#there is a one-to-one relationship between ad_id in promoted and the master data
print(len(promoted["ad_id"].unique())) #each add can appear more than once
print(len(data["ad_id"].unique()))




data.head()




print(len(data))
print(len(data["ad_id"].unique())) #adds appear on average slightly more than twice in our minidata set




#make dictionaries to look up advertizer id and campaign id for each ad_id
advertiser_dict = dict(zip(promoted.ad_id, promoted.advertiser_id))
campaign_dict = dict(zip(promoted.ad_id, promoted.campaign_id))




data["campaign_id"] = data["ad_id"].map(campaign_dict)
data["advertiser_id"] = data["ad_id"].map(advertiser_dict)
data.head()




print(len(data))
print(len(data["ad_id"].unique())) #adds appear on average slightly more than twice in our minidata set




#Why aren't there the same number of unique documents in each of these
print(len(data["document_id"].unique()))
print(len(doc_cats["document_id"].unique()))
print(len(doc_ents["document_id"].unique()))
print(len(doc_meta["document_id"].unique()))
print(len(doc_topics["document_id"].unique()))




#each document has multiple possible entities, categories, topics with different confidence level. 
#maybe we should just for now keep the most likely entity, topic and category? 
doc_ents.head()




doc_cats.head()




# print (clicks_train.head())
print (clicks_train[0:])





reg = 10 # trying anokas idea of regularization
eval = False # True = split off 10% of training data for validation and test performance

train = clicks_train

if eval:
    ids = train.index.values
    ids = np.random.choice(ids, size=len(ids)//10, replace=False)
    notids = set(train.index.values) - set(ids)
    
    #print(ids)
    #print(len(ids))
    valid = train.loc[ids] # random 10% for validation data 
    #print("length of train:" + str(len(train)))
    #print("length of valid: " + str(len(valid)))
    #print(valid.head())
    
    train = train.loc[notids] # remaining 90% as training data
    #print(train.head())
    print (valid.shape, train.shape)

cnt = train[train.clicked==1].ad_id.value_counts() # group # of clicks by ad 
cntall = train.ad_id.value_counts() # group # of displays by ad
del train

def get_prob(k):
    if k not in cnt:
        return 0
    return cnt[k]/(float(cntall[k]) + reg)  # return the proportion of ad clicks / displays

def srt(x):
    ad_ids = map(int, x.split()) # take in list of ads shown to each user
    ad_ids = sorted(ad_ids, key=get_prob, reverse=True) # re-sort the ads by training clicks / displays
    return " ".join(map(str,ad_ids)) # return the list with the ads sorted for submission
   
if eval:
    from ml_metrics import mapk

    y = valid[valid.clicked==1].ad_id.values # create list of ad click counts in validation set
    y = [[_] for _ in y]
    p = valid.groupby('display_id').ad_id.apply(list) #TODO: Blows up because display_id is an index field
    p = [sorted(x, key=get_prob, reverse=True) for x in p] # create list in order expected

    print (mapk(y, p, k=12)) # compare predicted order vs. actual order in validation set

else:
    subm = pd.read_csv("../input/sample_submission.csv") # load the sample submission file
    subm['ad_id'] = subm.ad_id.apply(lambda x: srt(x)) # re-sort the ads by overall training clicks / display
    subm.to_csv("subm_reg_2.csv", index=False) 




print(check_output(["ls", "../working"]).decode("utf8"))

