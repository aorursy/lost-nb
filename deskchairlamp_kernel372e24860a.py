#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
get_ipython().system('mkdir data')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# For kaggle kernel submission


import json

train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
train_df = train_df.dropna()
test_df = test_df.dropna()

def convert_train_to_json(data):
    output = {}
    output['data'] = []
    for idx, row in data.iterrows():
        title = F'Tweet {idx}'
        qid = row['textID']
        context = row['text']
        question = row['sentiment']
        answer = row['selected_text']
        answer_start = context.find(answer)
        answers = [{'answer_start': answer_start, 'text': answer}]
        qas = [{'answers': answers, 'question': question, 'id': qid}]
        paragraphs = [{'context': context, 'qas': qas}]
        output['data'].append({'title': title, 'paragraphs': paragraphs})
        
    with open('data/train.json', 'w') as outfile:
        json.dump(output, outfile)
        
        
def convert_test_to_json(data):
    output = {}
    output['data'] = []
    for idx, row in data.iterrows():
        title = F'Tweet {idx}'
        qid = row['textID']
        context = row['text']
        question = row['sentiment']
        answers = [{'answer_start': None, 'text': '__None__'}]
        qas = [{'answers': answers, 'question': question, 'id': qid}]
        paragraphs = [{'context': context, 'qas': qas}]
        output['data'].append({'title': title, 'paragraphs': paragraphs})
        
    with open('data/test.json', 'w') as outfile:
        json.dump(output, outfile)

convert_train_to_json(train_df)
convert_test_to_json(test_df)


# In[ ]:





# In[2]:


get_ipython().system('python /kaggle/input/transformers/transformers/examples/question-answering/run_squad.py --model_type roberta --model_name_or_path roberta-large --do_lower_case --do_train --do_eval --data_dir ./data --cache_dir /kaggle/input/robertacache/cache --train_file train.json --predict_file test.json --learning_rate 5e-5 --num_train_epochs 4 --max_seq_length 192 --doc_stride 64 --output_dir /kaggle/working --per_gpu_eval_batch_size=8 --per_gpu_train_batch_size=16 --save_steps=100000 --overwrite_output_dir')


# In[3]:


predictions = json.load(open('/kaggle/working/predictions_.json', 'r'))
submission = pd.read_csv(open('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv', 'r'))
for i in range(len(submission)):
    id_ = submission['textID'][i]
    if test_df['sentiment'][i] == 'neutral': # neutral postprocessing
        submission.loc[i, 'selected_text'] = test_df['text'][i]
    else:
        submission.loc[i, 'selected_text'] = predictions[id_]
        
submission.to_csv('submission.csv', index=False)


# In[4]:


# import os
# os.chdir(r'/kaggle/working')
# from IPython.display import FileLink
# FileLink(r'*')

