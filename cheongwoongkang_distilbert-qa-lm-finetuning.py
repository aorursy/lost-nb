#!/usr/bin/env python
# coding: utf-8



# Hyperparameters
batch_size = 16 # batch size
lr = 5e-5 # learning rate
epochs = 2 # number of epochs




import numpy as np
import pandas as pd
import json
import os




pd_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
pd_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
#pd_external = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv', sep=',', header=None, encoding='latin')




pd_external.head()




np_train = np.array(pd_train)
np_test = np.array(pd_test)
#np_external = np.array(pd_external)




lm_train = open('tweet.train.raw', 'w')
for line in np_train:
    context = line[1]
    if type(context) != str:
        continue
    context = context.lower()
    lm_train.write(context.strip() + '\n')

#for line in np_external:
#    context = line[-1].lower()
#    if type(context) != str:
#        continue
#    context = context.lower()
#    lm_train.write(context.strip() + '\n')
    
lm_test = open('tweet.test.raw', 'w')
for line in np_test:
    context = line[1].lower()
    if type(context) != str:
        continue
    context = context.lower()
    lm_test.write(context.strip() + '\n')
    
    lm_train.write(context.strip() + '\n')

lm_train.close()
lm_test.close()




get_ipython().system('cd /kaggle/input/pytorchtransformers/transformers-2.5.1; pip install .')




def run_script(train_file, predict_file, batch_size=16, lr=5e-5, epochs=2):
    get_ipython().system('python /kaggle/input/pytorchtransformers/transformers-2.5.1/examples/run_language_modeling.py     --output_dir=results     --model_type=distilbert     --model_name_or_path=distilbert-base-uncased     --cache_dir /kaggle/input/cached-distilbert-base-uncased/cache     --do_train     --train_data_file=$train_file     --do_eval     --eval_data_file=$predict_file     --learning_rate=$lr     --mlm     --line_by_line     --num_train_epochs=$epochs     --per_gpu_eval_batch_size=$batch_size     --per_gpu_train_batch_size=$batch_size     --save_steps=1000000')




get_ipython().system('mkdir results')




train_file = "tweet.train.raw"
predict_file = "tweet.test.raw"
run_script(train_file, predict_file, batch_size, lr, epochs)

