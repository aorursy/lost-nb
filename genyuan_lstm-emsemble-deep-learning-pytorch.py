#!/usr/bin/env python
# coding: utf-8



import re
import time
import gc
import random
import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
import torch.utils.data

tqdm.pandas()




def seed_torch(seed=43):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True




embed_size = 300      # how big is each word vector
max_features = 100000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70           # max number of words in a question to use

batch_size = 512
train_epochs = 4

SEED = 43




puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

mispell_dict = {"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)


first_word_mispell_dict = {
                'whta': 'what', 'howdo': 'how do', 'Whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 
                'howmany': 'how many', 'whydo': 'why do', 'doi': 'do i', 'howdoes': 'how does', "whst": 'what', 
                'shoupd': 'should', 'whats': 'what is', "im": "i am", "whatis": "what is", "iam": "i am", "wat": "what",
                "wht": "what","whts": "what is", "whtwh": "what", "whtat": "what", "whtlat": "what", "dueto to": "due to",
                "dose": "does", "wha": "what", 'hw': "how", "its": "it is", "whay": "what", "ho": "how", "whart": "what", 
                "woe": "wow", "wt": "what", "ive": "i have","wha": "what", "wich": "which", "whic": "which", "whys": "why", 
                "doe": "does", "wjy": "why", "wgat": "what", "hiw": "how","howto": "how to", "lets": "let us", "haw": "how", 
                "witch": "which", "wy": "why", "girlfriend": "girl friend", "hows": "how is","whyis": "why is", "whois": "who is",
                "dont": "do not", "hat": "what", "whos": "who is", "whydoes": "why does", "whic": "which","hy": "why", "w? hy": "why",
                "ehat": "what", "whate": "what", "whai": "what", "whichis": "which is", "whi": "which", "isit": "is it","ca": "can", 
                "wwhat": "what", "wil": "will", "wath": "what", "plz": "please", "ww": "how", "hou": "how", "whch": "which",
                "ihave": "i have", "cn": "can", "doesnt": "does not", "shoul": "should", "whatdo": "what do", "isnt": "is not", 
                "whare": "what are","whick": "which", "whatdoes": "what does", "hwo": "how", "howdid": "how did", "why dose": "why does"
}
def correct_first_word(x):
    for key in first_word_mispell_dict.keys():
        if x.startswith(key + " "):
            x = x.replace(key + " ", first_word_mispell_dict[key] + " ")
            break
    return x




def load_and_prec():
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)
    
    # lower
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: x.lower())
    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: x.lower())
    
    # clean first word
    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: correct_first_word(x))
    
    # Clean the text
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_text(x))
    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_text(x))
    
    # Clean numbers
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_numbers(x))
    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_numbers(x))
    
    # Clean speelings
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
    
    ## fill up the missing values
    train_X = train_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features, oov_token='OOV', filters='')
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)
    print("tokenization finished, word count: %d" % len(tokenizer.word_index))

    ## Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    ## Get the target values
    train_y = train_df['target'].values
    
    #shuffling the data
    np.random.seed(SEED)
    trn_idx = np.random.permutation(len(train_X))

    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]
    
    return train_X, test_X, train_y, tokenizer.word_index
train_X, test_X, train_y, word_index = load_and_prec()




def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'    
    emb_mean, emb_std = -0.005838499, 0.48782197
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    with open(EMBEDDING_FILE, 'r', encoding="utf8") as f:
        for line in tqdm(f):
            word, vec = line.split(' ', 1)
            if word not in word_index:
                continue
            i = word_index[word]
            if i >= nb_words:
                continue
            embedding_vector = np.asarray(vec.split(' '), dtype='float32')[:300]
            if len(embedding_vector) == 300:
                embedding_matrix[i] = embedding_vector                
            
    return embedding_matrix
    
def load_fasttext(word_index):    
    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    emb_mean, emb_std = -0.0033469985, 0.109855495
    nb_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    with open(EMBEDDING_FILE, 'r') as f:
        for line in tqdm(f):
            if len(line) <= 100:
                continue
            word, vec = line.split(' ', 1)
            if word not in word_index:
                continue
            i = word_index[word]
            if i >= nb_words:
                continue
            embedding_vector = np.asarray(vec.split(' '), dtype='float32')[:300]
            if len(embedding_vector) == 300:
                embedding_matrix[i] = embedding_vector   

    return embedding_matrix

def load_para(word_index):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    emb_mean, emb_std = -0.0053247833, 0.49346462
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    with open(EMBEDDING_FILE, 'r', encoding="utf8", errors='ignore') as f:
        for line in tqdm(f):
            if len(line) <= 100:
                continue
            word, vec = line.split(' ', 1)
            if word not in word_index:
                continue
            i = word_index[word]
            if i >= nb_words:
                continue
            embedding_vector = np.asarray(vec.split(' '), dtype='float32')[:300]
            if len(embedding_vector) == 300:
                embedding_matrix[i] = embedding_vector 
    
    return embedding_matrix




embedding_matrix_1 = load_glove(word_index)
embedding_matrix_2 = load_fasttext(word_index)
embedding_matrix_3 = load_para(word_index)
embedding_matrix123 = np.concatenate([embedding_matrix_1, embedding_matrix_2, embedding_matrix_3], axis = 1)
embedding_matrix12 = np.concatenate([embedding_matrix_1, embedding_matrix_2], axis = 1)
embedding_matrix13 = np.concatenate([embedding_matrix_1, embedding_matrix_3], axis = 1)
embedding_matrix23 = np.concatenate([embedding_matrix_2, embedding_matrix_3], axis = 1)
embedding_matrixs = [embedding_matrix12, embedding_matrix13, embedding_matrix23, embedding_matrix13]




class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)




class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix):
        super(NeuralNet, self).__init__()
        
        hidden_size = 128
        
        self.embedding = nn.Embedding(max_features, embed_size*2)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.embedding_dropout = nn.Dropout2d(0.15)
        self.lstm = nn.LSTM(embed_size*2, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)
        
        self.lstm_attention = Attention(hidden_size*2, maxlen)
        self.gru_attention = Attention(hidden_size*2, maxlen)
        
        self.linear = nn.Linear(1024, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.15)
        self.out = nn.Linear(64, 1)
        
    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))
        
        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)
        
        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)
        
        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)
        
        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        
        return out




class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, factor=0.6, min_lr=1e-4, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range']                 and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration
        
        self.last_loss = np.inf
        self.min_lr = min_lr
        self.factor = factor
        
    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def step(self, loss):
        if loss > self.last_loss:
            self.base_lrs = [max(lr * self.factor, self.min_lr) for lr in self.base_lrs]
            self.max_lrs = [max(lr * self.factor, self.min_lr) for lr in self.max_lrs]
            
    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs




def sigmoid(x):
    return 1 / (1 + np.exp(-x))




def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result




def train_and_predict1():

    seed_torch(SEED)

    x_test_cuda = torch.tensor(test_X, dtype=torch.long).cuda()
    test = torch.utils.data.TensorDataset(x_test_cuda)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    train_x0, valid_x0, train_y0, valid_y0 = train_test_split(train_X, train_y, test_size=0.05, shuffle=True, random_state=43)
    valid_preds = np.zeros((len(valid_y0)))
    test_preds = np.zeros((len(test_X)))

    epochs=3
    for i in range(epochs):
        x_train_fold = torch.tensor(train_x0, dtype=torch.long).cuda()
        y_train_fold = torch.tensor(train_y0.reshape(len(train_y0), 1), dtype=torch.float32).cuda()
        x_val_fold = torch.tensor(valid_x0, dtype=torch.long).cuda()
        y_val_fold = torch.tensor(valid_y0.reshape(len(valid_y0), 1), dtype=torch.float32).cuda()

        model = NeuralNet(embedding_matrixs[i])
        model.cuda()

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
        optimizer = torch.optim.Adam(model.parameters())
        step_size = 300
        scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.002,
                         step_size=step_size, mode='exp_range',
                         gamma=0.99994)

        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        print(f'Fold {i + 1}')
    
        train_epochs = 4
        for epoch in range(train_epochs):
            start_time = time.time()

            model.train()
            avg_loss = 0.
            for x_batch, y_batch in tqdm(train_loader, disable=True):
                y_pred = model(x_batch)
                scheduler.batch_step()
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)

            model.eval()
            valid_preds_fold = np.zeros((x_val_fold.size(0)))
            test_preds_fold = np.zeros(len(test_X))
            avg_val_loss = 0.
            for i, (x_batch, y_batch) in enumerate(valid_loader):
                y_pred = model(x_batch).detach()
                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

            elapsed_time = time.time() - start_time 
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, train_epochs, avg_loss, avg_val_loss, elapsed_time))
            search_result = threshold_search(valid_y0, valid_preds_fold)
            print(search_result)

        for i, (x_batch,) in enumerate(test_loader):
            y_pred = model(x_batch).detach()

            test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        valid_preds += valid_preds_fold / epochs
        test_preds += test_preds_fold / epochs  
        
    return valid_y0, valid_preds, test_preds




def train_and_predict2():

    seed_torch(SEED)

    x_test_cuda = torch.tensor(test_X, dtype=torch.long).cuda()
    test = torch.utils.data.TensorDataset(x_test_cuda)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    test_preds = np.zeros((len(test_X)))

    epochs=3
    avgThreshold = 0
    for i in range(epochs):
        train_x0, valid_x0, train_y0, valid_y0 = train_test_split(train_X, train_y, test_size=0.05, shuffle=True, random_state=43 * i)
        valid_preds = np.zeros((len(valid_y0)))
        
        x_train_fold = torch.tensor(train_x0, dtype=torch.long).cuda()
        y_train_fold = torch.tensor(train_y0.reshape(len(train_y0), 1), dtype=torch.float32).cuda()
        x_val_fold = torch.tensor(valid_x0, dtype=torch.long).cuda()
        y_val_fold = torch.tensor(valid_y0.reshape(len(valid_y0), 1), dtype=torch.float32).cuda()

        model = NeuralNet(embedding_matrixs[i])
        model.cuda()

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
        optimizer = torch.optim.Adam(model.parameters())

        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        print(f'Fold {i + 1}')
    
        train_epochs = 4
        for epoch in range(train_epochs):
            start_time = time.time()

            model.train()
            avg_loss = 0.
            for x_batch, y_batch in tqdm(train_loader, disable=True):
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)

            model.eval()
            valid_preds_fold = np.zeros((x_val_fold.size(0)))
            test_preds_fold = np.zeros(len(test_X))
            avg_val_loss = 0.
            for i, (x_batch, y_batch) in enumerate(valid_loader):
                y_pred = model(x_batch).detach()
                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

            elapsed_time = time.time() - start_time 
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, train_epochs, avg_loss, avg_val_loss, elapsed_time))
            search_result = threshold_search(valid_y0, valid_preds_fold)
            if epoch == train_epochs - 1:
                avgThreshold += search_result['threshold'] / epochs
            print(search_result)

        for i, (x_batch,) in enumerate(test_loader):
            y_pred = model(x_batch).detach()

            test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        test_preds += test_preds_fold / epochs  
        
    return avgThreshold, test_preds




def train_and_predict3():

    seed_torch(SEED)

    x_test_cuda = torch.tensor(test_X, dtype=torch.long).cuda()
    test = torch.utils.data.TensorDataset(x_test_cuda)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    test_preds = np.zeros((len(test_X)))
    
    splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED).split(train_X, train_y))
    train_preds = np.zeros((len(train_X)))

    for i, (train_idx, valid_idx) in enumerate(splits):
        x_train_fold = torch.tensor(train_X[train_idx], dtype=torch.long).cuda()
        y_train_fold = torch.tensor(train_y[train_idx, np.newaxis], dtype=torch.float32).cuda()
        x_val_fold = torch.tensor(train_X[valid_idx], dtype=torch.long).cuda()
        y_val_fold = torch.tensor(train_y[valid_idx, np.newaxis], dtype=torch.float32).cuda()

        model = NeuralNet(embedding_matrixs[i])
        model.cuda()

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
        optimizer = torch.optim.Adam(model.parameters())
        #step_size = 300
        #scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.003, step_size=step_size, mode='exp_range', gamma=0.99994)

        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        print(f'Fold {i + 1}')
    
        train_epochs = 4
        for epoch in range(train_epochs):
            start_time = time.time()

            model.train()
            avg_loss = 0.
            for x_batch, y_batch in tqdm(train_loader, disable=False):
                y_pred = model(x_batch)
                #scheduler.batch_step()
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)

            model.eval()
            valid_preds_fold = np.zeros((x_val_fold.size(0)))
            test_preds_fold = np.zeros(len(test_X))
            avg_val_loss = 0.
            for i, (x_batch, y_batch) in enumerate(valid_loader):
                y_pred = model(x_batch).detach()
                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

            elapsed_time = time.time() - start_time 
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, train_epochs, avg_loss, avg_val_loss, elapsed_time))
            search_result = threshold_search(train_y[valid_idx], valid_preds_fold)
            print(search_result)

        for i, (x_batch,) in enumerate(test_loader):
            y_pred = model(x_batch).detach()

            test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        train_preds[valid_idx] = valid_preds_fold
        test_preds += test_preds_fold / len(splits)   
        
    return train_preds, test_preds




#valid_y0, valid_preds, test_preds = train_and_predict1()
#search_result = threshold_search(valid_y0, valid_preds)
#print(search_result)
#sub = pd.read_csv('../input/sample_submission.csv')
#sub.prediction = (test_preds > search_result['threshold']).astype(int)
#sub.to_csv("submission.csv", index=False)




#avgThreshold, test_preds = train_and_predict2()
#print("avg threshold: %.6f" % avgThreshold)
#sub = pd.read_csv('../input/sample_submission.csv')
#sub.prediction = (test_preds > avgThreshold).astype(int)
#sub.to_csv("submission.csv", index=False)




train_preds, test_preds = train_and_predict3()
search_result = threshold_search(train_y, train_preds)
sub = pd.read_csv('../input/sample_submission.csv')
sub.prediction = (test_preds > search_result['threshold']).astype(int)
sub.to_csv("submission.csv", index=False)




print(search_result)

