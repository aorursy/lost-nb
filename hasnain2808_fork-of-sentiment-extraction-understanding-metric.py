#!/usr/bin/env python
# coding: utf-8



get_ipython().run_cell_magic('writefile', 'utils.py', 'import numpy as np\nimport torch\n\n\nclass AverageMeter:\n    """\n    Computes and stores the average and current value\n    """\n    def __init__(self):\n        self.reset()\n\n    def reset(self):\n        self.val = 0\n        self.avg = 0\n        self.sum = 0\n        self.count = 0\n\n    def update(self, val, n=1):\n        self.val = val\n        self.sum += val * n\n        self.count += n\n        self.avg = self.sum / self.count\n\n\nclass EarlyStopping:\n    def __init__(self, patience=7, mode="max", delta=0.001):\n        self.patience = patience\n        self.counter = 0\n        self.mode = mode\n        self.best_score = None\n        self.early_stop = False\n        self.delta = delta\n        if self.mode == "min":\n            self.val_score = np.Inf\n        else:\n            self.val_score = -np.Inf\n\n    def __call__(self, epoch_score, model, model_path):\n\n        if self.mode == "min":\n            score = -1.0 * epoch_score\n        else:\n            score = np.copy(epoch_score)\n\n        if self.best_score is None:\n            self.best_score = score\n            self.save_checkpoint(epoch_score, model, model_path)\n        elif score < self.best_score + self.delta:\n            self.counter += 1\n            print(\'EarlyStopping counter: {} out of {}\'.format(self.counter, self.patience))\n            if self.counter >= self.patience:\n                self.early_stop = True\n        else:\n            self.best_score = score\n            self.save_checkpoint(epoch_score, model, model_path)\n            self.counter = 0\n\n    def save_checkpoint(self, epoch_score, model, model_path):\n        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:\n            print(\'Validation score improved ({} --> {}). Saving model!\'.format(self.val_score, epoch_score))\n            torch.save(model.state_dict(), model_path)\n        self.val_score = epoch_score\n\n\ndef jaccard(str1, str2): \n    a = set(str1.lower().split()) \n    b = set(str2.lower().split())\n    c = a.intersection(b)\n    return float(len(c)) / (len(a) + len(b) - len(c))')




import utils




# es = utils.EarlyStopping(patience=2, mode="max")




import math




# math.floor(2.2)




# import nltk 
# w1 = set('AI is our friend and it has been friendly'.lower().split())
# w2 = set('AI and humans have always been friendly'.lower().split())
 
# print ("Jaccard similarity of above two sentences is",1-nltk.jaccard_distance(w1, w2))




# w1 = set('Kaggle is awesome'.lower().split())
# w2 = set('kaggle is great way of learning DS'.lower().split())
# print("The Jaccard similarity is:",1-nltk.jaccard_distance(w1, w2))




get_ipython().system('pip install pyspark')




get_ipython().system('pip install "/kaggle/input/chart-studio/chart_studio-1.0.0-py3-none-any.whl"')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from wordcloud import WordCloud, STOPWORDS
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import re
import string

import matplotlib.pyplot as plt
from plotly import tools
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

import os
import tokenizers
import torch
import transformers
import torch.nn as nn
from tqdm import tqdm

# import pyspark
# from pyspark.sql import SparkSession
# import pandas as pd

# spark = SparkSession.builder.appName('pandasToSparkDF').getOrCreate()

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# word level jaccard score: https://www.kaggle.com/c/tweet-sentiment-extraction/overview/evaluation
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))




import os
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import lr_scheduler

from sklearn import model_selection
from sklearn import metrics
import transformers
import tokenizers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm
import utils




# import findspark
# findspark.init()
import pyspark as ps
import warnings
from pyspark.sql import SQLContext

import string

try:
    # create SparkContext on all CPUs available: in my case I have 4 CPUs on my laptop
    sc = ps.SparkContext()
    sqlContext = SQLContext(sc)
    print("Just created a SparkContext")
except ValueError:
    warnings.warn("SparkContext already exists in this scope")




from pyspark.sql import SparkSession
MAX_MEMORY = "85g"
spark = SparkSession(sc).builder     .master('local[*]')     .config("spark.driver.memory", "85g")     .config("spark.executor.memory", "85g")     .appName('my-cool-app')     .getOrCreate()






# train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
# test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
# sub = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')




# sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('/kaggle/input/tweet-sentiment-extraction/test.csv')




sp_train = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('/kaggle/input/tweet-sentiment-extraction/train.csv')
sp_test = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('/kaggle/input/tweet-sentiment-extraction/test.csv')
sp_sub = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')




# sp_train = spark.read.format("csv").option("header", "true").load("/kaggle/input/tweet-sentiment-extraction/train.csv") 
# sp_test = spark.read.format("csv").option("header", "true").load("/kaggle/input/tweet-sentiment-extraction/test.csv") 
# sp_sub = spark.read.format("csv").option("header", "true").load("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv") 




def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text




from pyspark.sql.functions import regexp_replace, col
import re






punc=re.escape('!"#$%&\*+,-./:;<=>?@[\\]^_`{|}~')
# print(punc)
sp_train = sp_train.withColumn('text', regexp_replace('text', '[' + punc +']', ''))
sp_test = sp_test.withColumn('text', regexp_replace('text', '[' + punc +']', ''))

# sp_train.take(5)
# sp_test.take(5)




sp_train = sp_train.dropna()
sp_test = sp_test.dropna()




sp_train.take(5)




sp_test = sp_test.withColumn('selected_text', sp_test['text'])




sp_test.take(5)




print('Sentiment of text : {} \nOur training text :\n{}\nSelected text which we need to predict:\n{}'.format(sp_train.take(1)[0][3],sp_train.take(1)[0][2],sp_train.take(1)[0][1]))




import itertools
sp_train = sp_train.rdd.map(lambda x: [item for item in (x.text, x.selected_text, x.sentiment)])




sp_test = sp_test.rdd.map(lambda x: [item for item in (x.text, x.selected_text, x.sentiment)])




type(sp_train)




sp_train.take(5)




class config:
    MAX_LEN = 128
    BERT_PATH = "../input/roberta-base/"
    MODEL_PATH = "model.bin"
    TRAINING_FILE = "../input/train.csv"
    TOKENIZER = tokenizers.ByteLevelBPETokenizer(
        vocab_file=f"{BERT_PATH}/vocab.json", 
        merges_file=f"{BERT_PATH}/merges.txt", 
        lowercase=True,
        add_prefix_space=True)
    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 16
    EPOCHS = 5




MAX_LEN = 128
VALID_BATCH_SIZE = 8
BERT_PATH = "../input/roberta-base/"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../input/train.csv"
TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f"{BERT_PATH}/vocab.json", 
    merges_file=f"{BERT_PATH}/merges.txt", 
    lowercase=True,
    add_prefix_space=True
)




class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()
        self.bert = transformers.RobertaModel.from_pretrained(BERT_PATH)
        self.l0 = nn.Linear(768, 2)
    
    def forward(self, ids, mask, token_type_ids):
        sequence_output, pooled_output = self.bert(
            ids, 
            attention_mask=mask
        )
        logits = self.l0(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits




def loss_fn(start_logits, end_logits, start_positions, end_positions):
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss
def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for bi, d in enumerate(tk0):

        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        offsets = d["offsets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        model.zero_grad()
        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        )
        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        jaccard_scores = []
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            jaccard_score, _ = calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                idx_start=np.argmax(outputs_start[px, :]),
                idx_end=np.argmax(outputs_end[px, :]),
                offsets=offsets[px]
            )
            jaccard_scores.append(jaccard_score)

        jaccards.update(np.mean(jaccard_scores), ids.size(0))
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)




def calculate_jaccard_score(
    original_tweet, 
    target_string, 
    sentiment_val, 
    idx_start, 
    idx_end, 
    offsets,
    verbose=False):
    
    if idx_end < idx_start:
        idx_end = idx_start
    
    filtered_output  = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            filtered_output += " "

    if sentiment_val == "neutral" or len(original_tweet.split()) < 2:
        filtered_output = original_tweet

    jac = utils.jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output


def eval_fn(data_loader, model, device):
    model.eval()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            offsets = d["offsets"].numpy()

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            outputs_start, outputs_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            jaccard_scores = []
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                jaccard_score, _ = calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px]
                )
                jaccard_scores.append(jaccard_score)

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
    
    print(f"Jaccard = {jaccards.avg}")
    return jaccards.avg




device = torch.device("cuda")
model = TweetModel()
model.to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load("../input/roberta-weights/roberta_model_1.bin"))
model.eval()

model1 = TweetModel()
model1.to(device)
model1 = nn.DataParallel(model1)
model1.load_state_dict(torch.load("../input/roberta-weights/roberta_model_2.bin"))
model1.eval()




class TweetDataset:
    def __init__(self, df):
        self.sp_df = df
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN
    
    def __len__(self):
        return len(self.sp_df)
    
    def __getitem__(self, item):
    
        return {
            'ids': torch.tensor(self.sp_df[item]['ids'], dtype=torch.long),
            'mask': torch.tensor(self.sp_df[item]['mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(self.sp_df[item]['token_type_ids'], dtype=torch.long),
            'targets_start': torch.tensor(self.sp_df[item]['targets_start'], dtype=torch.float),
            'targets_end': torch.tensor(self.sp_df[item]['targets_end'], dtype=torch.float),
            'padding_len': torch.tensor(self.sp_df[item]['padding_len'], dtype=torch.long),
            'orig_tweet': self.sp_df[item]['orig_tweet'],
            'orig_selected': self.sp_df[item]['orig_selected'],
            'sentiment': self.sp_df[item]['sentiment']
        }









# sp_train = sp_train.rdd.map(lambda x: [item for item in (x.text, x.selected_text, x.sentiment)])




# sp_test = sp_test.rdd.map(lambda x: [item for item in (x.text, x.selected_text, x.sentiment)])










def process(item):
 MAX_LEN = 128
 VALID_BATCH_SIZE = 8
 BERT_PATH = "../input/roberta-base/"
 MODEL_PATH = "model.bin"
 TRAINING_FILE = "../input/train.csv"
 TOKENIZER = tokenizers.ByteLevelBPETokenizer(
     vocab_file=f"{BERT_PATH}/vocab.json", 
     merges_file=f"{BERT_PATH}/merges.txt", 
     lowercase=True,
     add_prefix_space=True
 )
 tokenizer = TOKENIZER
 max_len = MAX_LEN
 tweet = " " + " ".join(str(item[0]).split())
 selected_text = " " + " ".join(str(item[1]).split())

 len_st = len(selected_text)
 idx0 = -1
 idx1 = -1
 for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
     if tweet[ind: ind+len_st] == selected_text:
         idx0 = ind
         idx1 = ind + len_st
         break

 char_targets = [0] * len(tweet)
 if idx0 != -1 and idx1 != -1:
     for ct in range(idx0, idx1):
         # if tweet[ct] != " ":
         char_targets[ct] = 1

#     print(f"char_targets: {char_targets}")

 tok_tweet = tokenizer.encode(tweet)
 tok_tweet_tokens = tok_tweet.tokens
 tok_tweet_ids = tok_tweet.ids
 tok_tweet_offsets = tok_tweet.offsets

 targets = [0] * len(tok_tweet_ids)
 target_idx = []
 for j, (offset1, offset2) in enumerate(tok_tweet_offsets):
     if sum(char_targets[offset1: offset2]) > 0:
         targets[j] = 1
         target_idx.append(j)


 targets_start = [0] * len(targets)
 targets_end = [0] * len(targets)

 non_zero = np.nonzero(targets)[0]
 if len(non_zero) > 0:
     targets_start[non_zero[0]] = 1
     targets_end[non_zero[-1]] = 1

 # check padding:
 # <s> pos/neg/neu </s> </s> tweet </s>
 if len(tok_tweet_tokens) > max_len - 5:
     tok_tweet_tokens = tok_tweet_tokens[:max_len - 5]
     tok_tweet_ids = tok_tweet_ids[:max_len - 5]
     targets_start = targets_start[:max_len - 5]
     targets_end = targets_end[:max_len - 5]


 sentiment_id = {
     'positive': 1313,
     'negative': 2430,
     'neutral': 7974
 }

 tok_tweet_ids = [0] + [sentiment_id[item[2]]] + [2] + [2] + tok_tweet_ids + [2]
 targets_start = [0] + [0] + [0] + [0] + targets_start + [0]
 targets_end = [0] + [0] + [0] + [0] + targets_end + [0]
 token_type_ids = [0, 0, 0, 0] + [0] * (len(tok_tweet_ids) - 5) + [0]
 mask = [1] * len(token_type_ids)

 padding_length = max_len - len(tok_tweet_ids)

 tok_tweet_ids = tok_tweet_ids + ([1] * padding_length)
 mask = mask + ([0] * padding_length)
 token_type_ids = token_type_ids + ([0] * padding_length)
 targets_start = targets_start + ([0] * padding_length)
 targets_end = targets_end + ([0] * padding_length)

 return {
     'ids': torch.tensor(tok_tweet_ids, dtype=torch.long),
     'mask': torch.tensor(mask, dtype=torch.long),
     'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
     'targets_start': torch.tensor(targets_start, dtype=torch.float),
     'targets_end': torch.tensor(targets_end, dtype=torch.float),
     'padding_len': torch.tensor(padding_length, dtype=torch.long),
     'orig_tweet': item[0],
     'orig_selected': item[1],
     'sentiment': item[2]
 }




sp_train_processed = sp_train.map(process)









# sp_train_processed.take(5)




sp_test_processed = sp_test.map(process)

# sp_test_processed.take(5)




# sp_test_processed.cache()




# sp_train_processed.cache()




# sp_test_processed.take(1)




# sp_train_processed.take(1)




# sp_test_processed.count()




# sp_train_processed.count()




# test_processed_list  = sp_test_processed.collect()




#sp_test_processed.unpersist(True)




# train_processed_list  = sp_train_processed.collect()




#sp_train_processed.unpersist()




import pickle




# spark.stop()




# dbfile = open('test_processed_list', 'ab') 

# # source, destination 
# pickle.dump(test_processed_list, dbfile)                      
# dbfile.close() 




# dbfile = open('train_processed_list', 'ab') 

# # source, destination 
# pickle.dump(train_processed_list, dbfile)                      
# dbfile.close() 




def loadData(name): 
    # for reading also binary mode is important 
    dbfile = open(name, 'rb')      
    db = pickle.load(dbfile) 
    dbfile.close() 
    return db
    




get_ipython().system('ls /kaggle/input/sentiment-extraction-understanding-metric-eda/test_processed_list')




train_processed_list = loadData('/kaggle/input/sentiment-extraction-understanding-metric-eda/train_processed_list')
test_processed_list = loadData('/kaggle/input/sentiment-extraction-understanding-metric-eda/test_processed_list')




len(train_processed_list)




len(test_processed_list)




test_dataset = TweetDataset(
#         tweet=sp_test.select('text')#.values,
#         sentiment=sp_test.select('sentiment')#.values,
#         selected_text=sp_test.select('selected_text')#.values
    df = test_processed_list)

data_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=VALID_BATCH_SIZE,
    num_workers=1
)




def run(fold):
#     dfx = pd.read_csv(config.TRAINING_FILE)

    df_train = train_processed_list[:(len(train_processed_list)-math.floor(len(train_processed_list)*0.2))]
    df_valid = train_processed_list[math.floor(len(train_processed_list)*0.2):]
    
    train_dataset = TweetDataset(
    df = df_train
    )


    train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=False,
    batch_size=config.TRAIN_BATCH_SIZE,
    num_workers=1
    )

    valid_dataset = TweetDataset(
        df = df_valid
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
    shuffle=False,
    batch_size=config.VALID_BATCH_SIZE,
    num_workers=1
    )

    device = torch.device("cuda")
#     model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
#     model_config.output_hidden_states = True
#     model = TweetModel(conf=model_config)
    model = TweetModel()
    model.to(device)

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )

    es = utils.EarlyStopping(patience=2, mode="max")
    print(f"Training is Starting for fold={fold}")
    
    # I'm training only for 3 epochs even though I specified 5!!!
    for epoch in range(3):
        train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler)
        jaccard = eval_fn(valid_data_loader, model, device)
        print(f"Jaccard Score = {jaccard}")
        es(jaccard, model, model_path=f"model_{fold}.bin")
        if es.early_stop:
            print("Early stopping")
            break




run(fold=245)




device = torch.device("cuda")
model = TweetModel()
model.to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load("../input/roberta-weights/roberta_model_245.bin"))
model.eval()

model1 = TweetModel()
model1.to(device)
model1 = nn.DataParallel(model1)
model1.load_state_dict(torch.load("../input/roberta-weights/roberta_model_245.bin"))
model1.eval()




# for d in data_loader:
#     print(d["ids"])
#     print(d["token_type_ids"])
#     print(d["mask"])
#     print(d["padding_len"])
#     print(d["sentiment"])
#     print(d["orig_selected"])
#     print(d["orig_tweet"])
#     print(d["targets_start"])
#     print(d["targets_end"])

#     print(i['orignal_iext'])
#     pass




all_outputs = []
fin_outputs_start = []
fin_outputs_end = []
fin_outputs_start1 = []
fin_outputs_end1 = []
fin_padding_lens = []
fin_orig_selected = []
fin_orig_sentiment = []
fin_orig_tweet = []
fin_tweet_token_ids = []

with torch.no_grad():
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        padding_len = d["padding_len"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)

        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start1, outputs_end1 = model1(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        

        fin_outputs_start.append(torch.sigmoid(outputs_start).cpu().detach().numpy())
        fin_outputs_end.append(torch.sigmoid(outputs_end).cpu().detach().numpy())
        
        fin_outputs_start1.append(torch.sigmoid(outputs_start1).cpu().detach().numpy())
        fin_outputs_end1.append(torch.sigmoid(outputs_end1).cpu().detach().numpy())
        
        fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())
        fin_tweet_token_ids.append(ids.cpu().detach().numpy().tolist())

        fin_orig_sentiment.extend(sentiment)
        fin_orig_selected.extend(orig_selected)
        fin_orig_tweet.extend(orig_tweet)

fin_outputs_start = np.vstack(fin_outputs_start)
fin_outputs_end = np.vstack(fin_outputs_end)

fin_outputs_start1 = np.vstack(fin_outputs_start1)
fin_outputs_end1 = np.vstack(fin_outputs_end1)

fin_outputs_start = (fin_outputs_start + fin_outputs_start1) / 2
fin_outputs_end = (fin_outputs_end + fin_outputs_end1) / 2


fin_tweet_token_ids = np.vstack(fin_tweet_token_ids)
jaccards = []
threshold = 0.2
for j in range(fin_outputs_start.shape[0]):
    target_string = fin_orig_selected[j]
    padding_len = fin_padding_lens[j]
    sentiment_val = fin_orig_sentiment[j]
    original_tweet = fin_orig_tweet[j]

    if padding_len > 0:
        mask_start = fin_outputs_start[j, 4:-1][:-padding_len] >= threshold
        mask_end = fin_outputs_end[j, 4:-1][:-padding_len] >= threshold
        tweet_token_ids = fin_tweet_token_ids[j, 4:-1][:-padding_len]
    else:
        mask_start = fin_outputs_start[j, 4:-1] >= threshold
        mask_end = fin_outputs_end[j, 4:-1] >= threshold
        tweet_token_ids = fin_tweet_token_ids[j, 4:-1][:-padding_len]

    mask = [0] * len(mask_start)
    idx_start = np.nonzero(mask_start)[0]
    idx_end = np.nonzero(mask_end)[0]
    if len(idx_start) > 0:
        idx_start = idx_start[0]
        if len(idx_end) > 0:
            idx_end = idx_end[0]
        else:
            idx_end = idx_start
    else:
        idx_start = 0
        idx_end = 0

    for mj in range(idx_start, idx_end + 1):
        mask[mj] = 1

    output_tokens = [x for p, x in enumerate(tweet_token_ids) if mask[p] == 1]

    filtered_output = TOKENIZER.decode(output_tokens)
    filtered_output = filtered_output.strip().lower()

    if sentiment_val == "neutral" or len(original_tweet.split()) < 4:
        filtered_output = original_tweet

    all_outputs.append(filtered_output.strip())




sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
sample.loc[:, 'selected_text'] = all_outputs
sample.to_csv("submission.csv", index=False)




# sample




sample.head()






