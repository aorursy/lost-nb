#!/usr/bin/env python
# coding: utf-8



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from pandas import DataFrame
import numpy as np
import json




get_ipython().system('mkdir data')




get_ipython().system('mkdir bertoutput')




get_ipython().system('cp -r ../input/bertmodel/cased_l-12_h-768_a-12/* .')




get_ipython().system('cp -r cased_L-12_H-768_A-12/* .')




get_ipython().system('ls')




def preprocess(data):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, '')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    return data




train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

train.comment_text = preprocess(train['comment_text'])
test.comment_text = preprocess(test['comment_text'])


df_bert = pd.DataFrame({'id':train['id'],
            'target':np.where(train.target >= .8,1,0),
            'alpha':['a']*train.shape[0],
            'comment_text':train["comment_text"].replace(r'\n',' ',regex=True)})

df_0 = df_bert[df_bert['target'] == 0].sample(1000, random_state = 101) #max 144334
df_1 = df_bert[df_bert['target'] == 1].sample(1000, random_state = 101)


df_calibrated = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)

df_calibrated = shuffle(df_calibrated)

#df_bert_train, df_bert_dev = train_test_split(df_calibrated, test_size=0.01, random_state = 101)

df_bert_test = pd.DataFrame({'User_ID':test['id'],
                 'text':test['comment_text'].replace(r'\n',' ',regex=True)})
del(train)

df_calibrated.to_csv('data/train.tsv', sep='\t', index=False, header=False)
#df_bert_dev.to_csv('data/dev.tsv', sep='\t', index=False, header=False)
df_bert_test.to_csv('data/test.tsv', sep='\t', index=False, header=True)




get_ipython().system("python ../input/bertclassifier/repository/google-research-bert-0fce551/run_classifier.py --task_name=cola --do_train=True --do_predict=True --data_dir='data' --vocab_file='vocab.txt' --bert_config_file='bert_config.json' --init_checkpoint='bert_model.ckpt' --max_seq_length=220 --train_batch_size=8 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir='bertoutput' --do_lower_case=False")




result = pd.read_csv("bertoutput/test_results.tsv",sep="\t",header=None)

#sub = submission.to_numpy
preds_id = test['id'].values
preds = result.iloc[:,1].values

submission_file = pd.DataFrame({'id':preds_id,
                               'prediction':preds})
#pred_id = test['id']
submission_file.to_csv('submission.csv',sep=",",index=None)




sub  = pd.read_csv('submission.csv')




len(sub)

