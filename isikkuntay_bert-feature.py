#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import zipfile
import sys
import time

# Any results you write to the current directory are saved as output.


# In[2]:


from keras import backend as K
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.engine.topology import Layer


# In[3]:


#downloading weights and cofiguration file for the model
get_ipython().system('wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip')
with zipfile.ZipFile("uncased_L-12_H-768_A-12.zip","r") as zip_ref:
    zip_ref.extractall()
get_ipython().system("ls 'uncased_L-12_H-768_A-12'")


# In[4]:


#!wget https://raw.githubusercontent.com/google-research/bert/master/modeling.py 
get_ipython().system('wget https://raw.githubusercontent.com/isikkuntay/AIND/master/modeling.py')
get_ipython().system('wget https://raw.githubusercontent.com/isikkuntay/AIND/master/run_classifier.py')
get_ipython().system('wget https://raw.githubusercontent.com/google-research/bert/master/extract_features.py ')
get_ipython().system('wget https://raw.githubusercontent.com/google-research/bert/master/tokenization.py')


# In[5]:


import modeling
import extract_features
import tokenization
import tensorflow as tf


# In[6]:


get_ipython().system('wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv')
get_ipython().system('wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv')
get_ipython().system('wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv')
get_ipython().system('ls')


# In[7]:


def compute_offset_no_spaces(text, offset):
	count = 0
	for pos in range(offset):
		if text[pos] != " ": count +=1
	return count

def count_chars_no_special(text):
	count = 0
	special_char_list = ["#"]
	for pos in range(len(text)):
		if text[pos] not in special_char_list: count +=1
	return count

def count_length_no_special(text):
	count = 0
	special_char_list = ["#", " "]
	for pos in range(len(text)):
		if text[pos] not in special_char_list: count +=1
	return count


# In[8]:


#Copying Simple NLP notebook over here:

import os
import csv
import json
import string
import keras
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from math import floor
import spacy

get_ipython().run_line_magic('matplotlib', 'inline')

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

import time
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
import regex as re

import nltk 
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize 

nlp = spacy.load('en_core_web_sm')

def word_locate(sentence, location): 
    count_words = 0
    count_chars = 2 #2 is to count for the two spaces in the beginning
    for word in sentence.split():
        count_words += 1
        if location == count_chars:
            return word, count_words
        count_chars += len(word)
        count_chars += 1 #for space
        
def name_btwn_paran(sentence):
    capture = ""
    trigger_on = 0
    for char in sentence:
        if char == ")":
            trigger_on = 0
        if trigger_on == 1:
            capture += char
        if char == "(":
            trigger_on = 1
    return capture

def which_name_first(sentence, name1, name2): #If name1 is first, return True
    name1_check = 0
    for word_punct in sentence.split():
        for word_comma in word_punct.split(";"):
            for word in word_comma.split(","):
                if word == name2 and name1_check == 0:
                    return False
                if word == name1:
                    name1_check = 1
    return True

def curr_prev_sentence(sentence, loc):
    current_sentence = ""
    prev_sentence = ""
    trunc_curr_sentence = ""
    remainder_curr = ""
    detect = 0
    count = 0
    for char in sentence:
        count += 1
        current_sentence += char
        remainder_curr += char
        if ((char == "." or char == ";") and detect == 0 and sentence[count] != ","): #the last arguement to prevent ., as in sent #4
            prev_sentence = current_sentence 
            current_sentence = ""
        if char == "." and detect == 1:
            return current_sentence, prev_sentence, trunc_curr_sentence, remainder_curr
        if count == loc:
            detect = 1
            trunc_curr_sentence = current_sentence
            remainder_curr = ""
    return current_sentence, prev_sentence, trunc_curr_sentence, remainder_curr

def remove_last_word(sentence):
    new_sent = sentence.split()
    new_sent = new_sent[:-1]
    return " ".join(new_sent)

def check_if_capital(word):
    if word[0] in ["A","B","C","D","E","F","G","H","I","J","K","L","M","O","P","Q","R","S","T","U","V","W","X","Y","Z"]:
        return True
    else:
        return False
    
def list_of_name_words(tokenized):
    names_list = []
    for word_tuple in nltk.pos_tag(tokenized):
        if word_tuple[1] == "NNP":
            names_list.append(word_tuple[0])
    return names_list

def check_if_name(tokenized,word):
    text = tokenized
    for word_tuple in nltk.pos_tag(text):
        if word_tuple[0] == word:
            if word_tuple[1] == "NNP":
                return True
            else:
                return False
            
def find_name_words(sentence):
    name = "none"
    for word in sentence.split():
        if check_if_capital(word):
            return word
    return name

def remove_first_word(sentence):
    new_sent = sentence.split()
    new_sent = new_sent[1:]
    return " ".join(new_sent)

def find_nth_subj(doc, n): #finds subject number n
    subject = "none"
    count = 0
    for token in doc:
        if (token.dep_ == "nsubj" or token.dep_ == "nsubjpass"):
            count += 1
            if count == n:
                subject = token.text
    return subject

def find_nth_dobj(doc, n): #finds direct object number n
    dobj = "none"
    count = 0
    for token in doc:
        if (token.dep_ == "dobj"):
            count += 1
            if count == n:
                dobj = token.text
    return dobj

def find_nth_poss(doc, n): #finds possessing noun number n
    poss = "none"
    count = 0
    for token in doc:
        if (token.dep_ == "poss"):
            count += 1
            if count == n:
                poss = token.text
    return poss

def find_nth_appos(doc, n): #finds appos number n; sometimes Spacy mislabels nsubj as appos
    appos = "none"
    count = 0
    for token in doc:
        if (token.dep_ == "appos"):
            count += 1
            if count == n:
                appos = token.text
    return appos

def check_if_poss_her(doc, pronoun): #tells whether it is her as in his or her as in him
    #assumes only one her in the whole sentence (inaccurate?)
    for token in doc:
        if token.text == pronoun:
            if token.dep_ == "poss":
                return True
            else:
                return False
            
if 1 == 1:    
    def get_feature_vector(pronoun, text, A, B, proffset, inquiry_part = "A"):
        row = {}
        row['A'] = A
        row['B'] = B
        curr, prev, trunc_curr, remainder = curr_prev_sentence(text, proffset)
        curr_doc = nlp(curr)
        prev_doc = nlp(prev) 
        curr_tok = word_tokenize(curr)
        prev_tok = word_tokenize(prev)
        trunc_curr_tok = word_tokenize(trunc_curr)
        train_vector = []
        #get first subj in prev @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_p_f_s = "none"
        for n in [1,2,3,4,5]: #number of n is from common sense
            dummy_p_f_s = find_nth_subj(prev_doc,n)
            if check_if_name(prev_tok,dummy_p_f_s) and get_p_f_s == "none":
                get_p_f_s = dummy_p_f_s
        
        ####For sentence no. 5, spacy and nltk both failed to identify Collins as a propn.
        ### therefore, we will add a new line here making sure we have a name.
        
        if get_p_f_s == "none":
            if check_if_capital(find_nth_subj(prev_doc,1)):
                get_p_f_s = find_nth_subj(prev_doc,1)
        
        ### We are changing the feature conditions in this kernel (1st of fork of simple nlp):
        if get_p_f_s in row[inquiry_part] or row[inquiry_part] in get_p_f_s:
            train_vector.append(1)
        else:
            train_vector.append(0)
        
        #get last  subj in prev @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_p_l_s = "none"
        for n in [1,2,3,4,5]:
            dummy_p_l_s = find_nth_subj(prev_doc,n)
            if check_if_name(prev_tok,dummy_p_l_s):
                get_p_l_s = dummy_p_l_s
               
        ### pls Random forest classifier label special line:
        if get_p_l_s in row[inquiry_part] or row[inquiry_part] in get_p_l_s:
            train_vector.append(1)
        else:
            train_vector.append(0)
        
        #get first  obj in prev @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_p_f_o = "none"
        for n in [1,2,3,4,5]: 
            dummy_p_f_o = find_nth_dobj(prev_doc,n)
            if check_if_name(prev_tok,dummy_p_f_o) and get_p_f_o == "none":
                get_p_f_o = dummy_p_f_o
          
        ### pfo Random forest classifier label special line:
        if get_p_f_o in row[inquiry_part] or row[inquiry_part] in get_p_f_o:
            train_vector.append(1)
        else:
            train_vector.append(0)
            
        #get last  dobj in prev @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_p_l_o = "none"
        for n in [1,2,3,4,5]: 
            dummy_p_l_o = find_nth_dobj(prev_doc,n)
            if check_if_name(prev_tok,dummy_p_l_o):
                get_p_l_o = dummy_p_l_o
          
        ### plo Random forest classifier label special line:
        if get_p_l_o in row[inquiry_part] or row[inquiry_part] in get_p_l_o: 
            train_vector.append(1)
        else:
            train_vector.append(0)
        
        #get last  subj in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_l_s = "none"
        for n in [1,2,3,4]:
            dummy_tc_l_s = find_nth_subj(curr_doc,n)
            if check_if_name(curr_tok,dummy_tc_l_s)                    and (dummy_tc_l_s in trunc_curr): #this is slightly inaccurate but oh well
                get_tc_l_s = dummy_tc_l_s 
        
        ### tcls Random forest classifier label special line:
        if get_tc_l_s in row[inquiry_part] or row[inquiry_part] in get_tc_l_s: 
            train_vector.append(1)
        else:
            train_vector.append(0)
            
        #get last  dobj in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_l_o = "none"
        for n in [1,2,3,4]:
            dummy_tc_l_o = find_nth_dobj(curr_doc,n)
            if (dummy_tc_l_o in trunc_curr)                                        and check_if_name(curr_tok,dummy_tc_l_o): 
                get_tc_l_o = dummy_tc_l_o 
         
        ### tclo Random forest classifier label special line:
        if get_tc_l_o in row[inquiry_part] or row[inquiry_part] in get_tc_l_o: 
            train_vector.append(1)
        else:
            train_vector.append(0)
        
        #get last  poss in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_l_p = "none"
        for n in [1,2,3,4]:
            dummy_tc_l_p = find_nth_poss(curr_doc,n)
            if (dummy_tc_l_p in trunc_curr)                                        and check_if_name(curr_tok,dummy_tc_l_p): 
                get_tc_l_p = dummy_tc_l_p 
        
        ### tclp Random forest classifier label special line:
        if get_tc_l_p in row[inquiry_part] or row[inquiry_part] in get_tc_l_p:
            train_vector.append(1)
        else:
            train_vector.append(0)
        
        #get first subj in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_f_s = "none"
        for n in [1,2,3,4]:
            dummy_tc_f_s = find_nth_subj(curr_doc,n)
            if check_if_name(curr_tok,dummy_tc_f_s) and get_tc_f_s == "none":
                get_tc_f_s = dummy_tc_f_s 
           
        ### tcfs Random forest classifier label special line:
        if get_tc_f_s in row[inquiry_part] or row[inquiry_part] in get_tc_f_s:
            train_vector.append(1)
        else:
            train_vector.append(0)
            
        #get first dobj in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_f_o = "none"
        for n in [1,2,3,4]:
            dummy_tc_f_o = find_nth_dobj(curr_doc,n)
            if check_if_name(curr_tok,dummy_tc_f_o) and get_tc_f_o == "none": 
                get_tc_f_o = dummy_tc_f_o 
          
        ### tcfo Random forest classifier label special line:
        if get_tc_f_o in row[inquiry_part] or row[inquiry_part] in get_tc_f_o:  
            train_vector.append(1)
        else:
            train_vector.append(0)
    
        #get last  non-subj name word  in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_l_nw = "none"
        candidate = "none"
        tc_name_words = list_of_name_words(trunc_curr_tok) 
        if len(tc_name_words) > 0:
            candidate = tc_name_words[-1]
        if candidate in get_tc_f_s or candidate in get_tc_l_s:
            if len(tc_name_words) > 1:
                candidate = tc_name_words[-1]
        if check_if_name(curr_tok,candidate):
            get_tc_l_nw = candidate
        
        ### tclnw Random forest classifier label special line:
        if get_tc_l_nw in row[inquiry_part] or row[inquiry_part] in get_tc_l_nw: 
            train_vector.append(1)
        else:
            train_vector.append(0)
        
        #get first aposs in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_f_a = "none"
        for n in [1,2,3,4]:
            dummy_tc_f_a = find_nth_appos(curr_doc,n)
            if check_if_name(curr_tok,dummy_tc_f_a) and get_tc_f_a == "none": 
                get_tc_f_a = dummy_tc_f_a 
         
        ### tcfa Random forest classifier label special line:
        if get_tc_f_a in row[inquiry_part] or row[inquiry_part] in get_tc_f_a: 
            train_vector.append(1)
        else:
            train_vector.append(0)
    
        #get word btwn paranthesis in prev @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_p_f_wp = find_name_words(name_btwn_paran(prev))
          
        ### pfwp Random forest classifier label special line:
        if get_p_f_wp in row[inquiry_part] or row[inquiry_part] in get_p_f_wp:
            train_vector.append(1)
        else:
            train_vector.append(0)
            
        #get word btwn paranthesis in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_l_wp = find_name_words(name_btwn_paran(curr))  
              
        ### tclwp Random forest classifier label special line:
        if get_tc_l_wp in row[inquiry_part] or row[inquiry_part] in get_tc_l_wp: 
            train_vector.append(1)
        else:
            train_vector.append(0)
            
        #get last subj in remainder @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_r_f_s = "none"
        for n in [1,2,3,4,5,6,7,8]: #in the final version, each of the name subjects will be accunted for
            dummy_r_f_s = find_nth_subj(curr_doc,n)
            if dummy_r_f_s in remainder and check_if_name(curr_tok,dummy_r_f_s):
                get_r_f_s = dummy_r_f_s 
           
        ### rfs Random forest classifier label special line:
        if get_r_f_s in row[inquiry_part] or row[inquiry_part] in get_r_f_s:
            train_vector.append(1)
        else:
            train_vector.append(0)
            
        #get last dobj in remainder @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_r_f_o = "none"
        for n in [1,2,3,4,5,6,7,8]: #in the final version, each of the name objects will be accunted for
            dummy_r_f_o = find_nth_dobj(curr_doc,n)
            if dummy_r_f_o in remainder and check_if_name(curr_tok,dummy_r_f_o):
                get_r_f_o = dummy_r_f_o 
              
        ### rfo Random forest classifier label special line:
        if get_r_f_o in row[inquiry_part] or row[inquiry_part] in get_r_f_o:
            train_vector.append(1)
        else:
            train_vector.append(0)
            
        #get last appos in remainder @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_r_f_a = "none"
        for n in [1,2,3,4]:
            dummy_r_f_a = find_nth_appos(curr_doc,n)
            if dummy_r_f_a in remainder and check_if_name(curr_tok,dummy_r_f_a): 
                get_r_f_a = dummy_r_f_a 
          
        ### rfa Random forest classifier label special line:
        if get_r_f_a in row[inquiry_part] or row[inquiry_part] in get_r_f_a:
            train_vector.append(1)
        else:
            train_vector.append(0)
        
        #get first appos in current @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_c_f_a = "none"
        for n in [1,2,3,4]:
            dummy_c_f_a = find_nth_appos(curr_doc,n)
            if check_if_name(curr_tok,dummy_c_f_a) and get_c_f_a == "none": 
                get_c_f_a = dummy_c_f_a 
               
        ### cfa Random forest classifier label special line:
        if get_c_f_a in row[inquiry_part] or row[inquiry_part] in get_c_f_a: 
            train_vector.append(1)
        else:
            train_vector.append(0)
        
        #get first appos in prev @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_p_f_a = "none"
        for n in [1,2,3,4]:
            dummy_p_f_a = find_nth_appos(prev_doc,n)
            if check_if_name(prev_tok,dummy_p_f_a) and get_p_f_a == "none": 
                get_p_f_a = dummy_p_f_a 
              
        ### pfa Random forest classifier label special line:
        if get_p_f_a in row[inquiry_part] or row[inquiry_part] in get_p_f_a:
            train_vector.append(1)
        else:
            train_vector.append(0)
    
        #check_if_poss_her
        get_poss_her = check_if_poss_her(curr_doc, pronoun)
        
        #rand_forest classifier for pronoun type:
        if pronoun == "he" or pronoun == "she": 
            train_vector.append(1)
        elif pronoun == "He" or pronoun == "She": 
            train_vector.append(2)
        elif pronoun == "his" or (pronoun == "her" and get_poss_her): 
            train_vector.append(3)
        elif pronoun == "him" or (pronoun == "her" and not get_poss_her): 
            train_vector.append(4)
        elif pronoun == "His" or (pronoun == "Her" and get_poss_her): 
            train_vector.append(5)
        else:
            train_vector.append(6)
    
        return train_vector


# In[9]:





# In[9]:


def run_bert(data):
	'''
	Runs a forward propagation of BERT on input text, extracting contextual word embeddings
	Input: data, a pandas DataFrame containing the information in one of the GAP files

	Output: emb, a pandas DataFrame containing contextual embeddings for the words A, B and Pronoun. Each embedding is a numpy array of shape (768)
	columns: "emb_A": the embedding for word A
	         "emb_B": the embedding for word B
	         "emb_P": the embedding for the pronoun
	         "label": the answer to the coreference problem: "A", "B" or "NEITHER"
	'''
    # From the current file, take the text only, and write it in a file which will be passed to BERT
	text = data["Text"]
	text.to_csv("input.txt", index = False, header = False)

	task_name = "kepler"

#	processors = {"kepler": run_classifier.KeplerProcessor}
#	processors = {"kepler": run_classifier.MrpcProcessor}
#processor = processors["kepler"]

    # The script extract_features.py runs forward propagation through BERT, and writes the output in the file output.jsonl
    # I'm lazy, so I'm only saving the output of the last layer. Feel free to change --layers = -1 to save the output of other layers.
	os.system("python3 extract_features.py 	  --input_file=input.txt 	  --output_file=output.jsonl 	  --vocab_file=uncased_L-12_H-768_A-12/vocab.txt 	  --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json 	  --init_checkpoint=uncased_L-12_H-768_A-12/bert_model.ckpt 	  --layers=-2 	  --max_seq_length=256 	  --batch_size=8")

	bert_output = pd.read_json("output.jsonl", lines = True)

	os.system("rm output.jsonl")
	os.system("rm input.txt")

	index = data.index
	columns = ["emb_A", "emb_B", "emb_P", "feat_A", "feat_B", "label"]
	emb = pd.DataFrame(index = index, columns = columns)
	emb.index.name = "ID"

	for i in range(len(data)): # For each line in the data file
		# get the words A, B, Pronoun. Convert them to lower case, since we're using the uncased version of BERT
		P = data.loc[i,"Pronoun"]
		A = data.loc[i,"A"]
		B = data.loc[i,"B"]

		# For each word, find the offset not counting spaces. This is necessary for comparison with the output of BERT
		P_offset = compute_offset_no_spaces(data.loc[i,"Text"], data.loc[i,"Pronoun-offset"])
		A_offset = compute_offset_no_spaces(data.loc[i,"Text"], data.loc[i,"A-offset"])
		B_offset = compute_offset_no_spaces(data.loc[i,"Text"], data.loc[i,"B-offset"])
		# Figure out the length of A, B, not counting spaces or special characters
		A_length = count_length_no_special(A)
		B_length = count_length_no_special(B)

		# Initialize embeddings with zeros
		emb_A = np.zeros(768)
		emb_B = np.zeros(768)
		emb_P = np.zeros(768)

		# Initialize counts
		count_chars = 0
		cnt_A, cnt_B, cnt_P = 0, 0, 0

		features = pd.DataFrame(bert_output.loc[i,"features"]) # Get the BERT embeddings for the current line in the data file
		for j in range(2,len(features)):  # Iterate over the BERT tokens for the current line; we skip over the first 2 tokens, which don't correspond to words
			token = features.loc[j,"token"]

			# See if the character count until the current token matches the offset of any of the 3 target words
			if count_chars  == P_offset: 
				# print(token)
				emb_P += np.array(features.loc[j,"layers"][0]['values'])
				cnt_P += 1
			if count_chars in range(A_offset, A_offset + A_length): 
				# print(token)
				emb_A += np.array(features.loc[j,"layers"][0]['values'])
				cnt_A += 1
			if count_chars in range(B_offset, B_offset + B_length): 
				# print(token)
				emb_B += np.array(features.loc[j,"layers"][0]['values'])
				cnt_B += 1
# Update the character count
			count_chars += count_length_no_special(token)
		# Taking the average between tokens in the span of A or B, so divide the current value by the count	
		emb_A /= cnt_A
		emb_B /= cnt_B

		# Work out the label of the current piece of text
		label = "Neither"
		if (data.loc[i,"A-coref"] == True):
			label = "A"
		if (data.loc[i,"B-coref"] == True):
			label = "B"

		pro_offset = data.loc[i,"Pronoun-offset"]
		this_text = data.loc[i,"Text"]

		feat_A = get_feature_vector(P, this_text, A, B, pro_offset, inquiry_part = "A")
		feat_B = get_feature_vector(P, this_text, A, B, pro_offset, inquiry_part = "B")
            
		# Put everything together in emb
		emb.iloc[i] = [emb_A, emb_B, emb_P, np.asarray(feat_A), np.asarray(feat_B), label]

	return emb


# In[10]:


print("Started at ", time.ctime())
test_data = pd.read_csv("gap-test.tsv", sep = '\t')
test_emb = run_bert(test_data)
test_emb.to_json("contextual_embeddings_gap_test.json", orient = 'columns')

validation_data = pd.read_csv("gap-validation.tsv", sep = '\t')
validation_emb = run_bert(validation_data)
validation_emb.to_json("contextual_embeddings_gap_validation.json", orient = 'columns')

with open('../input/gendered-pronoun-resolution/test_stage_2.tsv') as tsvfile:
    development_data = pd.read_csv(tsvfile, sep = '\t')
development_emb = run_bert(development_data)
development_emb.to_json("contextual_embeddings_gap_development.json", orient = 'columns')
print("Finished at ", time.ctime())


# In[11]:


from keras import backend, models, layers, initializers, regularizers, constraints, optimizers
from keras import callbacks as kc
from keras import optimizers as ko

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import log_loss
import time


dense_layer_sizes = [37]
dropout_rate = 0.6
learning_rate = 0.001
n_fold = 5
batch_size = 32
epochs = 1000
patience = 100
# n_test = 100
lambd = 0.1 # L2 regularization


# In[12]:


class IsLayer(Layer):
    #Layer to be used after a dense one. It will multiply all the elements with each other.
    #In a sense, it allows the neurons to have a say on  each others' outputs. This layer, hopefully,
    #compares the relative importance of neurons.The compound prob is regulated with weights.
    #The idea follows from attention layer, but is more basic than that. As it is multiplicative, it is 
    #an alternative to the vanilla additive layer where outputs are added at the next layer.
    
    def __init__(self, **kwargs):
        super(IsLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        #Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='W', 
                                 shape=(input_shape[1], 1), 
                                 initializer='uniform',
                                 trainable=True)
        super(IsLayer, self).build(input_shape)
        
    def call(self, x):
        x_W = K.dot(x, self.W)
        x_new = x*x_W 
        return x_new
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# In[13]:


def build_mlp_model(input_shape, num_output):
	X_input = layers.Input(input_shape)

	# First dense layer
	X = layers.Dense(dense_layer_sizes[0], name = 'dense0')(X_input)
	X = layers.BatchNormalization(name = 'bn0')(X)
	X = layers.Activation('relu')(X)
	X = layers.Dropout(dropout_rate, seed = 7)(X)
#	X = IsLayer()(X)

	# Second dense layer
# 	X = layers.Dense(dense_layer_sizes[0], name = 'dense1')(X)
# 	X = layers.BatchNormalization(name = 'bn1')(X)
# 	X = layers.Activation('relu')(X)
# 	X = layers.Dropout(dropout_rate, seed = 9)(X)

	# Output layer
	X = layers.Dense(num_output, name = 'output', kernel_regularizer = regularizers.l2(lambd))(X)
	X = layers.Activation('sigmoid')(X)

	# Create model
	model = models.Model(input = X_input, output = X, name = "classif_model")
	return model


# In[14]:


def parse_json(embeddings):
	'''
	Parses the embeddigns given by BERT, and suitably formats them to be passed to the MLP model

	Input: embeddings, a DataFrame containing contextual embeddings from BERT, as well as the labels for the classification problem
	columns: "emb_A": contextual embedding for the word A
	         "emb_B": contextual embedding for the word B
	         "emb_P": contextual embedding for the pronoun
	         "label": the answer to the coreference problem: "A", "B" or "NEITHER"

	Output: X, a numpy array containing, for each line in the GAP file, the concatenation of the embeddings of the target words
	        Y, a numpy array containing, for each line in the GAP file, the one-hot encoded answer to the coreference problem
	'''
	embeddings.sort_index(inplace = True) # Sorting the DataFrame, because reading from the json file messed with the order
	X = np.zeros((len(embeddings)*2,2*768+19)) #19 is the length of special feature vector
	Y = np.zeros((len(embeddings)*2, 1))

	# Concatenate features (A first batch)
	for i in range(len(embeddings)):
		A = np.array(embeddings.loc[i,"emb_A"])
		P = np.array(embeddings.loc[i,"emb_P"])
		F = np.array(embeddings.loc[i,"feat_A"])        
		X[i] = np.concatenate((A, P, F))

	# One-hot encoding for labels
	for i in range(len(embeddings)):
		label = embeddings.loc[i,"label"]
		if label == "A":
			Y[i] = 1
		else:
			Y[i] = 0

	# Concatenate features (B second batch)
	for i in range(len(embeddings)):
		B = np.array(embeddings.loc[i,"emb_B"])
		P = np.array(embeddings.loc[i,"emb_P"])
		F = np.array(embeddings.loc[i,"feat_B"])                
		X[i+len(embeddings)] = np.concatenate((B, P, F)) 

	# One-hot encoding for labels ; A's and B's concatenated like same since they are symmetrical
	for i in range(len(embeddings)):
		label = embeddings.loc[i,"label"]
		if label == "B":
			Y[i+len(embeddings)] = 1
		else:
			Y[i+len(embeddings)] = 0
        
	return X, Y


# In[15]:


# Read development embeddigns from json file - this is the output of Bert
development = pd.read_json("contextual_embeddings_gap_development.json")
X_development, Y_development = parse_json(development)

validation = pd.read_json("contextual_embeddings_gap_validation.json")
X_validation, Y_validation = parse_json(validation)

test = pd.read_json("contextual_embeddings_gap_test.json")
X_test, Y_test = parse_json(test)


# In[16]:


# There may be a few NaN values, where the offset of a target word is greater than the max_seq_length of BERT.
# They are very few, so I'm just dropping the rows.
remove_test = [row for row in range(len(X_test)) if np.sum(np.isnan(X_test[row]))]
X_train = np.delete(X_test, remove_test, 0)
Y_train = np.delete(Y_test, remove_test, 0)

# We want predictions for all validation rows. So instead of removing rows, make them 0
remove_validation = [row for row in range(len(X_validation)) if np.sum(np.isnan(X_validation[row]))]
X_validation[remove_validation] = np.zeros(2*768+19)

# We want predictions for all development rows. So instead of removing rows, make them 0
remove_development = [row for row in range(len(X_development)) if np.sum(np.isnan(X_development[row]))]
X_development[remove_development] = np.zeros(2*768+19)


# In[17]:


# Will train on data from the gap-test and gap-validation files, in total 2454 rows
#X_train = np.concatenate((X_test, X_validation), axis = 0)
#Y_train = np.concatenate((Y_test, Y_validation), axis = 0)

# Will predict probabilities for data from the gap-development file; initializing the predictions
#prediction = np.zeros((len(X_development),1)) # testing predictions

val_prediction = np.zeros((len(X_validation),1)) # valid predictions


# In[18]:


# Training and cross-validation
folds = KFold(n_splits=n_fold, shuffle=True, random_state=3)
scores = []
for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):
	# split training and validation data
	print('Fold', fold_n, 'started at', time.ctime())
	X_tr, X_val = X_train[train_index], X_train[valid_index]
	Y_tr, Y_val = Y_train[train_index], Y_train[valid_index]

	# Define the model, re-initializing for each fold
	classif_model = build_mlp_model([X_train.shape[1]],1)
	classif_model.compile(optimizer = optimizers.Adam(lr = learning_rate), loss = "binary_crossentropy")
	callbacks = [kc.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights = True)]

	# train the model
	classif_model.fit(x = X_tr, y = Y_tr, epochs = epochs, batch_size = batch_size, callbacks = callbacks, validation_data = (X_val, Y_val), verbose = 0)

	# make predictions on validation and test data
	pred_valid = classif_model.predict(x = X_val, verbose = 0)

	# oof[valid_index] = pred_valid.reshape(-1,)
	scores.append(log_loss(Y_val, pred_valid))
    
val_prediction = classif_model.predict(x = X_validation, verbose = 0)

# Print CV scores, as well as score on the test data
print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
print(scores)
print("Test score:", log_loss(Y_validation,val_prediction))


# In[19]:


def build_neither_mlp(input_shape, num_output):
	X_input = layers.Input(input_shape)

	# First dense layer
	X = layers.Dense(dense_layer_sizes[0], name = 'dense0')(X_input)
	X = layers.BatchNormalization(name = 'bn0')(X)
	X = layers.Activation('relu')(X)
	X = layers.Dropout(dropout_rate, seed = 7)(X)

    # Output layer
	X = layers.Dense(num_output, name = 'output', kernel_regularizer = regularizers.l2(lambd))(X)
	X = layers.Activation('sigmoid')(X)

	# Create model
	model = models.Model(input = X_input, output = X, name = "neither_model")
	return model


# In[20]:


X_val_A = val_prediction[: int(len(val_prediction)/2)]
X_val_B = val_prediction[int(len(val_prediction)/2) :]
X_train_neither = np.concatenate((X_val_A, X_val_B), axis=1)

Y_val_A = Y_validation[: int(len(Y_validation)/2)]
Y_val_B = Y_validation[int(len(Y_validation)/2) :]
Y_train_neither = 1 - Y_val_A - Y_val_B


# In[21]:


print(X_val_A.shape)
print(X_val_B.shape)
print(X_train_neither.shape)
print(Y_val_A.shape)
print(Y_val_B.shape)
print(Y_train_neither.shape)


# In[22]:


neither_model = build_neither_mlp([X_train_neither.shape[1]],1)
neither_model.compile(optimizer = optimizers.Adam(lr = learning_rate), loss = "binary_crossentropy")
neither_model.fit(X_train_neither, y = Y_train_neither, epochs = epochs, batch_size = batch_size, validation_data = (X_train_neither, Y_train_neither), verbose = 0)

dev_prediction = classif_model.predict(x = X_development, verbose = 0)


# In[23]:


X_dev_A = dev_prediction[: int(len(dev_prediction)/2)]
X_dev_B = dev_prediction[int(len(dev_prediction)/2) :]
X_dev_neither = np.concatenate((X_dev_A, X_dev_B), axis=1)


# In[24]:


dev_neither = neither_model.predict(x = X_dev_neither, verbose = 0)


# In[25]:


# Write the prediction to file for submission
submission = pd.read_csv("../input/sample_submission_stage_1.csv", index_col = "ID")

submission["A"] = X_dev_A
submission["B"] = X_dev_B
submission["NEITHER"] = dev_neither
submission.to_csv("submission_bert.csv")

