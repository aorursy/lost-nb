#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import sys
import zipfile
import datetime
print(os.listdir("../input"))
print(os.listdir("../input/huggingfacepytorchpretrainedbertchinese"))
print(os.listdir("../input/huggingfacepytorchpretrainedbertchinese/bert-base-chinese"))
# for d in os.listdir("../input"):
#     print(os.listdir(f"../input/{d}"))
# print(os.listdir("../input/googles-bert-model/chinese_l-12_h-768_a-12/chinese_L-12_H-768_A-12"))
# Any results you write to the current directory are saved as output.




VOCAB = '../input/huggingfacepytorchpretrainedbertchinese/bert-base-chinese-vocab.txt'
MODEL = '../input/huggingfacepytorchpretrainedbertchinese/bert-base-chinese'




get_ipython().system('pip install pytorch_pretrained_bert ')




import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification




# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')




# Tokenized input
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)
tokenized_text




tokenizer.tokenize('su')




if os.path.isdir("../input/fake-news-pair-classification-challenge"):
    TRAIN_CSV_PATH = '../input/fake-news-pair-classification-challenge/train.csv'
    TEST_CSV_PATH = '../input/fake-news-pair-classification-challenge/test.csv'
    TOKENIZED_TRAIN_CSV_PATH = "../input/siamese-network-lstm/tokenized_train.csv"
else:
    TRAIN_CSV_PATH = '../input/train.csv'
    TEST_CSV_PATH = '../input/test.csv'
    TOKENIZED_TRAIN_CSV_PATH = ""
    
train = pd.read_csv(TRAIN_CSV_PATH, index_col='id')
test = pd.read_csv(TEST_CSV_PATH, index_col='id')
cols = ['title1_zh', 
        'title2_zh', 
        'label']
train = train.loc[:, cols]
test = test.loc[:, cols]
train.fillna('UNKNOWN', inplace=True)
test.fillna('UNKNOWN', inplace=True)
train.head(3)




from collections import Counter
Counter(train.label)




get_ipython().system('wget https://raw.githubusercontent.com/huggingface/pytorch-pretrained-BERT/master/examples/run_classifier.py')
    
    




from sklearn.model_selection     import train_test_split

VALIDATION_RATIO = 0.1

RANDOM_STATE = 9527

train, val=     train_test_split(
        train, 
        test_size=VALIDATION_RATIO, 
        random_state=RANDOM_STATE
)




label_list = ['unrelated', 'agreed', 'disagreed']




from run_classifier import *

train_examples = [InputExample('train', row.title1_zh, row.title2_zh, row.label) for row in train.itertuples()]
val_examples = [InputExample('val', row.title1_zh, row.title2_zh, row.label) for row in val.itertuples()]
test_examples = [InputExample('test', row.title1_zh, row.title2_zh, 'unrelated') for row in test.itertuples()]

len(train_examples)




orginal_total = len(train_examples)
train_examples = train_examples[:int(orginal_total*0.2)]




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
gradient_accumulation_steps = 1
train_batch_size = 32
eval_batch_size = 128
train_batch_size = train_batch_size // gradient_accumulation_steps
output_dir = 'output'
bert_model = 'bert-base-chinese'
num_train_epochs = 3
num_train_optimization_steps = int(
            len(train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
cache_dir = "model"
learning_rate = 5e-5
warmup_proportion = 0.1
max_seq_length = 128
label_list = ['unrelated', 'agreed', 'disagreed']




tokenizer = BertTokenizer.from_pretrained(VOCAB)
model = BertForSequenceClassification.from_pretrained(MODEL,
              cache_dir=cache_dir,
              num_labels = 3)
model.to(device)
if n_gpu > 1:
    model = torch.nn.DataParallel(model)
model, tokenizer




# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)





global_step = 0
nb_tr_steps = 0
tr_loss = 0

train_features = convert_examples_to_features(
    train_examples, label_list, max_seq_length, tokenizer)
logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info("  Batch size = %d", train_batch_size)
logger.info("  Num steps = %d", num_train_optimization_steps)
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

model.train()
for _ in trange(int(num_train_epochs), desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    total_step = len(train_data) // train_batch_size
    ten_percent_step = total_step // 10
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        loss = model(input_ids, segment_ids, input_mask, label_ids)
        if n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu.
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            
        if step % ten_percent_step == 0:
            print("Fininshed: {:.2f}% ({}/{})".format(step/total_step*100, step, total_step))




if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save a trained model and the associated configuration
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
torch.save(model_to_save.state_dict(), output_model_file)
output_config_file = os.path.join(output_dir, CONFIG_NAME)
with open(output_config_file, 'w') as f:
    f.write(model_to_save.config.to_json_string())




# Load a trained model and config that you have fine-tuned
config = BertConfig(output_config_file)
model = BertForSequenceClassification(config, num_labels=len(label_list))
model.load_state_dict(torch.load(output_model_file))
model.to(device)  # important to specific device
if n_gpu > 1:
    model = torch.nn.DataParallel(model)




config




# val
eval_examples = val_examples
eval_features = convert_examples_to_features(
    eval_examples, label_list, max_seq_length, tokenizer)
logger.info("***** Running evaluation *****")
logger.info("  Num examples = %d", len(eval_examples))
logger.info("  Batch size = %d", eval_batch_size)
all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
# Run prediction for full data
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

model.eval()
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)

    with torch.no_grad():
        tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
        logits = model(input_ids, segment_ids, input_mask)

    logits = logits.detach().cpu().numpy()
    label_ids = label_ids.to('cpu').numpy()
    tmp_eval_accuracy = accuracy(logits, label_ids)

    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy

    nb_eval_examples += input_ids.size(0)
    nb_eval_steps += 1

eval_loss = eval_loss / nb_eval_steps
eval_accuracy = eval_accuracy / nb_eval_examples
loss = tr_loss/nb_tr_steps
result = {'eval_loss': eval_loss,
          'eval_accuracy': eval_accuracy,
          'global_step': global_step,
          'loss': loss}

output_eval_file = os.path.join(output_dir, "eval_results.txt")
with open(output_eval_file, "w") as writer:
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))




device




get_ipython().system('ls output')
get_ipython().system(' cat output/eval_results.txt')




model




def predict(model, tokenizer, examples, label_list, eval_batch_size=128):
    model.to(device)
    eval_examples = examples
    eval_features = convert_examples_to_features(
        eval_examples, label_list, max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    
    res = []
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
#         label_ids = label_ids.to(device)

        with torch.no_grad():
#             tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
#         print(logits)
        res.extend(logits.argmax(-1))
#         label_ids = label_ids.to('cpu').numpy()
#         tmp_eval_accuracy = accuracy(logits, label_ids)

#         eval_loss += tmp_eval_loss.mean().item()
#         eval_accuracy += tmp_eval_accuracy

#         nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

#     eval_loss = eval_loss / nb_eval_steps
#     eval_accuracy = eval_accuracy / nb_eval_examples
#     loss = tr_loss/nb_tr_steps 
#     result = {'eval_loss': eval_loss,
#               'eval_accuracy': eval_accuracy,
#               'global_step': global_step,
#               'loss': loss}

#     output_eval_file = os.path.join(output_dir, "eval_results.txt")
#     with open(output_eval_file, "w") as writer:
#         logger.info("***** Eval results *****")
#         for key in sorted(result.keys()):
#             logger.info("  %s = %s", key, str(result[key]))
#             writer.write("%s = %s\n" % (key, str(result[key])))
    return res
    




res = predict(model, tokenizer, test_examples, label_list)




label_list




set(res)




predict(model, tokenizer, test_examples[:10], label_list)




cat_map = {idx:lab for idx, lab in enumerate(label_list)}
res = [cat_map[c] for c  in res]




#　For Submission

test['Category'] = res


submission = test     .loc[:, ['Category']]     .reset_index()

submission.columns = ['Id', 'Category']
submission.to_csv('submission.csv', index=False)
submission.head()

