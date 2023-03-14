#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




TRAIN_PATH = "data/train.json"
TEST_PATH = "data/test.json"
RESULT_DIR = "results"
INPUT_DIR = "../input/tweet-sentiment-extraction/"

MODEL_TYPE = "roberta"
MODEL_NAME_OR_PATH = "roberta-large"

EPOCHS = 3
LEARNING_RATE = 4e-5
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 192
DOC_STRIDE = 64




data_train, data_test = pd.read_csv(f'{INPUT_DIR}train.csv'), pd.read_csv(f'{INPUT_DIR}test.csv')
data_train.head(5)




import json

def find_all(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1

def convert2squad(data, output_path):
  output = {}
  output['version'] = 'v1.0'
  output['data'] = []

  if 'selected_text' in data.columns:
    clear_data = zip(data['textID'], data['sentiment'].apply(str).values.tolist(),                    data['text'].apply(str).values.tolist(), data['selected_text'].apply(str).values.tolist())
  else:
    clear_data = zip(data['textID'], data['sentiment'].apply(str).values.tolist(),                    data['text'].apply(str).values.tolist(), [None] * data.shape[0])

  for qid, question, context, answer in clear_data:
    answers = []
    
    if answer:
      answer_starts = find_all(context, answer)
      for answer_start in answer_starts:
        answers.append({'answer_start': answer_start, 'text': answer})
    else:
      answers.append({'answer_start': 1000000, 'text': '__None__'})

    qas = [{'question': question, 'id': qid, 'is_impossible': False, 'answers': answers}]

    paragraphs = [{'context': context, 'qas': qas}]
    output['data'].append({'title': 'None', 'paragraphs': paragraphs})
    
  with open(output_path, 'w') as output_file:
    json.dump(output, output_file)




get_ipython().system('mkdir -p data')
convert2squad(data_train, TRAIN_PATH)
convert2squad(data_test, TEST_PATH)




get_ipython().system('mkdir -p $RESULT_DIR')




get_ipython().system('git clone https://github.com/huggingface/transformers; cd transformers; pip install .')




get_ipython().system('python transformers/examples/run_squad.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --do_lower_case --do_train --do_eval --data_dir . --cache_dir ./cache --train_file $TRAIN_PATH --predict_file $TEST_PATH --learning_rate $LEARNING_RATE --num_train_epochs $EPOCHS --max_seq_length $MAX_SEQ_LENGTH --doc_stride $DOC_STRIDE --output_dir $RESULT_DIR --per_gpu_eval_batch_size=$BATCH_SIZE --per_gpu_train_batch_size=$BATCH_SIZE --save_steps=100000')




predictions = json.load(open(f'{RESULT_DIR}/predictions_.json', 'r'))

submission = pd.read_csv(open(f'{INPUT_DIR}sample_submission.csv', 'r'))

for i in range(len(submission)):
    id_ = submission['textID'][i]
    if data_test['sentiment'][i] == 'neutral': # neutral postprocessing
        submission.loc[i, 'selected_text'] = data_test['text'][i]
    else:
        submission.loc[i, 'selected_text'] = predictions[id_]

submission.head(5)




submission.to_csv('submission.csv', index=False)

