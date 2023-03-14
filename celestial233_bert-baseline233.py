#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict, Set
from collections import Counter
from time import time
import random
import json
import math
import nltk
from matplotlib import pyplot as plt

import torchtext
from torchtext import data, datasets
from torch.utils.data.dataset import random_split

import os
"""
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
# Any results you write to the current directory are saved as output.


# In[2]:


batch_size = 8
n_epochs = 2
max_context_len = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device: ", device)


# In[3]:


def divide_dataset(__train, factor, fields):
    train_size = int(len(__train) * factor)
    valid_size = len(__train) - train_size
    valid_indices = list(range(train_size+valid_size))
    random.shuffle(valid_indices)
    train_indices = valid_indices[:train_size]
    valid_indices = valid_indices[train_size:]
    train = [__train[idx] for idx in train_indices]
    valid = [__train[idx] for idx in valid_indices]
    train = data.Dataset(train, fields)
    valid = data.Dataset(valid, fields)
    return train, valid
    
def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n"     or ord(c) == 0x202F:
        return True
    return False

def character_filter(c):
    if c == "\t": return ""
    if ord(c)<128: return c
    if c in "≠•∞™ˈʃʊʁʁiʁɑ̃ʃɔ.̃ºª¶§¡£¢ç": return "z"
    if c in "àáâãäåæ": return "a"
    if c in "èéêë": return "e"
    if c in "ìíîï": return "i"
    if c in "òóôõöōŏő": return "o"
    if c in "ùúûü": return "u"

    if c in "ÀÁÂÃÄÅ": return "A"
    if c in "ÈÉÊË": return "E"
    if c in "ÌÍÎÏ": return "I" 
    if c in "ÒÓÔÕÖŌŎŐ": return "O"
    if c in "ÙÚÛÜ": return "U"

    return "z"
def postproc(text, x):
    ret = []
    for sent in text:
        ret.append(tokenizer.convert_tokens_to_ids(sent))
    return ret


# In[4]:


def read_SQuAD(file, fields, train):
    input_data = pd.read_csv(file)
    examples = []
    word_to_char_list = []
    context_raw_test = []
    
    for idx, row in input_data.iterrows():
       # print(row['text'], )
        context_raw = row['text']
        question_raw = row['sentiment']
        if type(context_raw) != str:
            print("data", idx, "is illegal")
            continue
        context_raw = context_raw.strip()
        if not train:
            context_raw_test.append(context_raw)
        context_raw = "".join(map(character_filter, context_raw))

        pt = 0
        char_to_word = [-1]*len(context_raw)
        word_to_char = []
        context  = tokenizer.tokenize(context_raw)
        question = tokenizer.tokenize(question_raw)
        
        word_to_char = [-1]*(len(question)+2)
        if (len(context)>max_context_len): 
            print("data", idx, ":too long")
            continue
        if unk_token in context:
            print("data", idx, "tokenize with [UNK]", ":discard")
            continue
        for idx, word in enumerate(context):
            if model_name == "bert":
                if word[0:2]=="##":
                    word = word[2:]
            if model_name == "roberta":
                if word[0]=="Ġ":
                    word = word[1:]
            if model_name == "xlnet":
                if word[0]=="▁":
                    word = word[1:]
            if pt>=len(context_raw):
                print(context_raw)
                print(context)
                print(idx, word)
            while is_whitespace(context_raw[pt]): 
                pt+=1
                if pt>=len(context_raw):
                    print(context_raw)
                    print(context)
                    print(idx, word)
                    

            if word == '"':
                if context_raw[pt:pt+2] != "``" and context_raw[pt:pt+2] != "''":
                    print(context_raw)
                    print(pt)
                    print(context_raw[pt:pt+len(word)], word)
                char_to_word[pt:pt+2]= [idx]*len(word)
                word_to_char.append(pt)
                pt = pt + 2
            else:
                if context_raw[pt:pt+len(word)].lower() != word.lower():
                    print(context_raw)
                    print(pt)
                    print(context_raw[pt:pt+len(word)], word)
                char_to_word[pt:pt+len(word)]= [idx]*len(word)
                word_to_char.append(pt)
                pt = pt + len(word)
        word_to_char.append(pt)
        
             
        if train:    
            answer_text = row['selected_text']
            answer_text = "".join(map(character_filter, answer_text))
            answer = tokenizer.tokenize(answer_text)
            
            raw_start = context_raw.find(answer_text)
            assert(raw_start>=0)
            answer_start= char_to_word[raw_start]
            answer_end  = char_to_word[raw_start + len(answer_text) - 1]
  
            """
            for i in range(len(context)):
                flag = True
                for j, w in enumerate(answer):
                    if context[i+j] != w:
                        flag = False
                        break
                if flag:
                    answer_start = i
                    answer_end   = i + len(answer) - 1
                    break
            """
                    
            assert(answer_end>=0 and answer_start>=0)
            answer_start += len(question)+2
            answer_end   += len(question)+2
        else:
            answer_start = -1
            answer_end   = -1
            assert(len(word_to_char) == len(question) + len(context) + 3)
            word_to_char_list.append(word_to_char)
            

        example = data.Example.fromlist([
            [cls_token]+question+[sep_token]+ context+[sep_token], 
            answer_start, 
            answer_end],
            fields
        )
        examples.append(example)

    if train:
        return data.Dataset(examples, fields)
    else:
        return data.Dataset(examples, fields), context_raw_test, word_to_char_list


# In[5]:


from transformers import BertTokenizer   , BertForQuestionAnswering   , BertConfig
from transformers import XLNetTokenizer  , XLNetForQuestionAnsweringSimple  , XLNetConfig
from transformers import RobertaTokenizer, RobertaModel               , RobertaConfig

model_name = "roberta"
PATH = "/kaggle/input/huggingfacetransformermodels/model_classes/"

if model_name == "xlnet":
    tmp = PATH + model_name + "/" + model_name + "-large-cased-"
    Tokenizer =  XLNetTokenizer
    TPATH = tmp + "tokenizer/"
    model = XLNetForQuestionAnsweringSimple.from_pretrained(tmp+"pytorch-model/").to(device)

class QAModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.conv = nn.Conv1d(1024, 128, 3, padding=1)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(128, 2)
    def forward(self, x):
        out = self.model(x)[0].transpose(1, 2)
        out = self.dropout(out) # (32, 1024, L)
        out = F.relu(self.conv(out)).transpose(1, 2)
        out = self.fc(out)
        return out[:,:,0], out[:,:,1]
    
if model_name == "roberta":
    QA_PATH = '/kaggle/input/roberta-transformers-pytorch/roberta-large/'
    Tokenizer = RobertaTokenizer
    TPATH = QA_PATH
    
    model = RobertaModel.from_pretrained(QA_PATH).to(device)
    model = QAModel(model).to(device)
    
if model_name == "bert":
    QA_PATH = '/kaggle/input/bertlargewholewordmaskingfinetunedsquad/'
    prefix = "bert-large-uncased-whole-word-masking-"
    Tokenizer = BertTokenizer
    TPATH = QA_PATH + prefix + 'vocab.txt'
    
    config = BertConfig.from_pretrained(
        QA_PATH + prefix + 'finetuned-squad-config.json')
    model = BertForQuestionAnswering.from_pretrained(
        QA_PATH + prefix + 'finetuned-squad-pytorch_model.bin', config=config).to(device)

tokenizer = Tokenizer.from_pretrained(TPATH)
cls_token = tokenizer.cls_token
sep_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token


# In[6]:


CONTEXT     = data.Field(batch_first=True, use_vocab=False, postprocessing=postproc, pad_token=pad_token)#, preprocessing=preproc)
LABEL       = data.Field(batch_first=True, use_vocab=False, sequential=False)
# make splits for data
fields = [
    ("text", CONTEXT),
    ("answer_start" , LABEL),
    ("answer_end"   , LABEL),
]

print("read train data...")
__train = read_SQuAD("/kaggle/input/tweet-sentiment-extraction/train.csv", fields, True)
train, valid = divide_dataset(__train, 0.9, fields)

print("read test data...")
test, context_raw_test, word_to_char_list =     read_SQuAD("/kaggle/input/tweet-sentiment-extraction/test.csv", fields, False)

train_size = len(train)
valid_size = len(valid)
test_size  = len(test)
print("read finish")

print("train size :", train_size)
print("valid size :", valid_size)
print("test size :" , test_size)

# make iterator for splits
train_iter = data.BucketIterator(train, batch_size=batch_size, train=True, sort_key=lambda x:len(x.text), device=device)
valid_iter = data.BucketIterator(valid, batch_size=batch_size, train=False,sort_key=lambda x:len(x.text), device=device)
test_iter  = data.Iterator(test , batch_size=1, train=False, sort=False, device=device)


# In[7]:


def data_check():
    for i in range(5):
        print(train[i].text)
        print(train[i].text[train[i].answer_start: train[i].answer_end+1])

    train_length_freqs = Counter([len(example.text) for example in train])
    print("train: min len :",min(train_length_freqs),"max len :",max(train_length_freqs))
    
    test_length_freqs = Counter([len(example.text) for example in test])
    print("test: min len :",min(test_length_freqs),"max len :",max(test_length_freqs))
    
    """
  #  plt.hist([len(example.text) for example in train], log=True, bins=20, range=(0, 120))
  #  plt.savefig("/kaggle/working/train_len.png")
    plt.hist([len(example.text) for example in test] , log=True, bins=20, range=(0, 120))
   # plt.savefig("/kaggle/working/full_len.png")
    
  #  plt.hist([example.answer_end-example.answer_start+1 for example in train],
   #          log=True, bins=20, range=(0, 120))
  #  plt.savefig("/kaggle/working/train_answer_len.png")
    plt.hist([example.answer_end-example.answer_start+1 for example in test],
             log=True, bins=20, range=(0, 120))
  #  plt.savefig("/kaggle/working/full_len.png")
    """
data_check()


# In[8]:


class MRCMetrics(object):
    def __init__(self, criterion):
        self.loss = 0
        self.size = 0
        self.F1 = 0
        self.EM = 0
        self.ja = 0
        self.criterion = criterion
    def update(self, st_prob, ed_prob, st, ed):
        st_pred = st_prob.argmax(-1)
        ed_pred = ed_prob.argmax(-1)
        size = st_pred.shape[0]

        zeros = torch.zeros_like(st_pred, dtype=torch.float)
  
        F1 = torch.max(zeros  , zeros+torch.min(ed_pred, ed) - torch.max(st_pred, st)+1)*2             /torch.max(zeros+2, zeros+ed_pred+ed-st_pred-st+2)
        ja = torch.max(zeros  , zeros+torch.min(ed_pred, ed) - torch.max(st_pred, st)+1)             /torch.max(zeros  , zeros+torch.max(ed_pred, ed) - torch.min(st_pred, st)+1)

        loss = self.criterion(st_prob, st) + criterion(ed_prob, ed)*size
      #      print("grad:", loss_t.requires_grad)
    #        print(F1.shape, F1_t.shape)
     #       print(F1.dtype, F1_t.dtype)
        EM = ((st_pred == st)*(ed_pred == ed)).to(torch.float)
        self.F1 += F1.sum().item()
        self.EM += EM.sum().item()
        self.ja += ja.sum().item()
        self.loss += loss.sum().item()
        self.size += size
        return loss/size
    def __getitem__(self, key):
        return getattr(self, key)/self.size
    def __str__(self):
        return ("loss: %.4f, EM: %.4f, F1: %.4f, ja: %.4f" %(self["loss"], self["EM"], self["F1"], self["ja"]))


# In[9]:


total_num = sum(p.numel() for p in model.parameters())
print('Total:', total_num)

criterion = torch.nn.CrossEntropyLoss().to(device)
if model_name == "bert":
    lr = 1e-5
elif model_name == "roberta":
    lr = 1e-5
elif model_name == "xlnet":
    lr = 6e-6
clip_grad_value = 5
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-3)

#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3)


# In[10]:


def model_check():
    batch = next(iter(valid_iter))
    print(batch.text[0])
    
    st_prob, ed_prob = model(batch.text)
    #out = model(batch.text)
    #print(out)
    #print(out.shape)
    print(st_prob.shape)
    """
    print(st_prob[0])
    print(ed_prob[0])
    print(batch.answer_start[0])
    print(batch.answer_end[0])
    """
    metrics = MRCMetrics(criterion)
    loss = metrics.update(st_prob, ed_prob, batch.answer_start, batch.answer_end)
    print(loss)

    optimizer.zero_grad()
    #loss.backward()
    print(metrics)
#model_check()


# In[11]:


def run_epoch(data_iter, train):
    # Train the model
    metrics = MRCMetrics(criterion)
    num_iter = len(data_iter)
    if train:
        model.train()
    else:
        model.eval()
    for i, batch in enumerate(data_iter):
        text = batch.text.to(device)
        answer_start = batch.answer_start.to(device)
        answer_end   = batch.answer_end  .to(device)
   #     print(context.shape, question.shape)
   #     print(answer_start.shape)
        
        if train:
            st_prob, ed_prob = model(text)
            loss = metrics.update(st_prob, ed_prob, answer_start, answer_end)
            loss.backward()
        #    nn.utils.clip_grad_norm_(model.parameters(), clip_grad_value)
            optimizer.step()
            optimizer.zero_grad()
            

            if ((i+1)%400==0): 
                scheduler.step(metrics["ja"])
                print(i, ": train",  metrics, "lr:", optimizer.param_groups[0]["lr"])
            """
            if loss.item()<loss_min:
                loss_min = loss.item()
                loss_count = 0
            else:
                loss_count += 1
            if loss_count >= 300:
                scheduler.step()
                loss_count = 0
                print("scheduler")
            """
        else:
            with torch.no_grad():
                st_prob, ed_prob = model(text)
                loss = metrics.update(st_prob, ed_prob, answer_start, answer_end)

    return metrics


# In[12]:


for epoch in range(n_epochs):
    start_time = time()
    train_metrics = run_epoch(train_iter, True)
    valid_metrics = run_epoch(valid_iter, False)

    secs = int(time() - start_time)
    print("epoch", epoch,"finished in "+str(secs)+"s")
    print("train:", train_metrics)
    print("valid:", valid_metrics)
    # train loss: 0.83, EM: 0.53, F1: 0.74, ja: 0.69


# In[13]:


def decode(st_prob, ed_prob, word_to_char):
    global cnt
    if st_prob.argmax().item() > ed_prob.argmax().item():
        cnt = cnt + 1
        print("empty out ", cnt)
    n = st_prob.shape[1]-1
    mx = -12345
    start, end = -1, -1
    for i in range(n):
        for j in range(i, n):
            if word_to_char[i] < 0: continue
            if word_to_char[j] < 0: continue
            logit = st_prob[0,i].item() + ed_prob[0,j].item()
            if logit > mx:
                mx = logit
                start, end = i, j
    return start, end
# 80 zero


# In[14]:


model.eval()
out = []
cnt = 0
for batch, context_raw, word_to_char in zip(test_iter, context_raw_test, word_to_char_list):
    text = batch.text.to(device)
    with torch.no_grad():
        st_prob, ed_prob = model(text)
        assert(st_prob.shape[0]==1)
        
        start = st_prob.argmax().item()
        end   = ed_prob.argmax().item()
      #  print(start, end)
        start, end = decode(st_prob, ed_prob, word_to_char)
        answer= context_raw[word_to_char[start]: word_to_char[end+1]]
        out.append(answer)
       # print(answer)
        """
        if i%100==0: 
            print(i)
            print(phrase)
            print(tokens)
            print(string)
        """
sample = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
sample.loc[:, 'selected_text'] = out
sample.to_csv("submission.csv", index=False)


# In[15]:


sample.head()

