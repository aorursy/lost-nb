#!/usr/bin/env python
# coding: utf-8



# basics
import os
import time
import re
import regex
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool

# machine learning
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler

# nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# use only for tokenizer and padding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences




cuda_idx = 0




all_start = time.time()




def timer(fn):
    def func(*args):
        start = time.time()
        res = fn(*args)
        elapsed = time.time() - start
        print('timer: {:.3f} elapsed by {}'.format(elapsed, fn))
        return res
        
    return func




def seed_torch(seed=1019):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 1019
seed_torch(SEED)




# model parameters
class Config:
    num_epochs = 6
    num_embedding_learn_epochs = 1
    embedding_noise_var = 0.01
    batch_size = [256, 512, 512, 1024, 1024, 1536]
    test_batch_size = 512
    vocab_size = 120000
    max_length = 72
    embedding_size = 300
    hidden_size = 96
    num_layers = 1
    num_gru_layers = 1
    embedding_dropout = 0.3
    layer_dropout = 0.1
    dense_size = [hidden_size*2*4+3, int(hidden_size/4)] # depend on concat num
    output_size = 1
    num_cv_splits = 5
    num_routings = 4
    num_capsules = 5
    dim_capsules = 5
    learning_rate = 0.001
    max_learning_rate = 0.003
    clr_step_size = 300
    clr_gamma = 0.9999
    clip_grad = 5.0
    embeddings = ['glove', 'paragram', 'fasttext']  # ['word2vec', 'glove', 'paragram', 'fasttext']
    # datadir = Path('../../../data/quora_insincere')
    datadir = Path('../input') # for kernel

c = Config()




puncts = [
    ',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&',
    '/', '[', ']', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£',
    '·', '_', '{', '}', '©', '^', '®', '`', '→', '°', '€', '™', '›',
    '♥', '←', '×', '§', '″', '′', 'Â', '█', 'à', '…', '“', '★', '”',
    '–', '●', 'â', '►', '−', '¢', '¬', '░', '¶', '↑', '±',  '▾',
    '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '⊕', '▼',
    '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
    'è', '¸', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',
    '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
    '¹', '≤', '‡', '₹', '´'
]




abbreviations = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "this's": "this is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "here's": "here is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "who'd": "who would",
    "who're": "who are",
    "'re": " are",
    "tryin'": "trying",
    "doesn'": "does not",
    'howdo': 'how do',
    'whatare': 'what are',
    'howcan': 'how can',
    'howmuch': 'how much',
    'howmany': 'how many',
    'whydo': 'why do',
    'doI': 'do I',
    'theBest': 'the best',
    'howdoes': 'how does',
}




spells = {
    'colour': 'color',
    'centre': 'center',
    'favourite': 'favorite',
    'travelling': 'traveling',
    'counselling': 'counseling',
    'theatre': 'theater',
    'cancelled': 'canceled',
    'labour': 'labor',
    'organisation': 'organization',
    'wwii': 'world war 2',
    'citicise': 'criticize',
    'youtu.be': 'youtube',
    'youtu ': 'youtube ',
    'qoura': 'quora',
    'sallary': 'salary',
    'Whta': 'what',
    'whta': 'what',
    'narcisist': 'narcissist',
    'mastrubation': 'masturbation',
    'mastrubate': 'masturbate',
    "mastrubating": 'masturbating',
    'pennis': 'penis',
    'Etherium': 'ethereum',
    'etherium': 'ethereum',
    'narcissit': 'narcissist',
    'bigdata': 'big data',
    '2k17': '2017',
    '2k18': '2018',
    'qouta': 'quota',
    'exboyfriend': 'ex boyfriend',
    'exgirlfriend': 'ex girlfriend',
    'airhostess': 'air hostess',
    "whst": 'what',
    'watsapp': 'whatsapp',
    'demonitisation': 'demonetization',
    'demonitization': 'demonetization',
    'demonetisation': 'demonetization',
    'quorans': 'quora user',
    'quoran': 'quora user',
    'pokémon': 'pokemon',
}




@timer
def load_data(datadir):
    train_df = pd.read_csv(datadir / 'train.csv')
    test_df = pd.read_csv(datadir / 'test.csv')
    print("Train shape : ", train_df.shape)
    print("Test shape : ", test_df.shape)
    return train_df, test_df

def clean(df):
    df = clean_lower(df)
    df = clean_unicode(df)
    df = clean_math(df)
    df = clean_abbreviation(df, abbreviations)
    df = clean_spells(df, spells)
    df = clean_language(df)
    df = clean_puncts(df, puncts)
    df = clean_space(df)
    return df

def clean_unicode(df):
    codes = ['\x7f', '\u200b', '\xa0', '\ufeff', '\u200e', '\u202a', '\u202c', '\u2060', '\uf0d8', '\ue019', '\uf02d', '\u200f', '\u2061', '\ue01b']
    df["question_text"] = df["question_text"].apply(lambda x: _clean_unicode(x, codes))
    return df

def _clean_unicode(x, codes):
    for u in codes:
        if u in x:
            x = x.replace(u, '')
    return x

def clean_language(df):
    langs1 = r'[\p{Katakana}\p{Hiragana}\p{Han}]' # regex
    langs2 = r'[ஆய்தஎழுத்துஆயுதஎழுத்துशुषछछशुषدوउसशुष북한내제តើបងប្អូនមានមធ្យបាយអ្វីខ្លះដើម្បីរកឃើញឯកសារអំពីប្រវត្តិស្ត្រនៃប្រាសាទអង្គរវट्टरौरआदસંઘરાજ્યपीतऊनअहএকটিবাড়িএকটিখামারএরঅধীনেপদেরবাছাইপরীক্ষাএরপ্রশ্নওউত্তরসহকোথায়পেতেপারিص、。Емелядуракلكلمقاممقال수능ί서로가를행복하게기乡국고등학교는몇시간업니《》싱관없어나이रचा키کپڤ」मिलगईकलेजेकोठंडकऋॠऌॡर]'
    compiled_langs1 = regex.compile(langs1)
    compiled_langs2 = re.compile(langs2)
    df['question_text'] = df['question_text'].apply(lambda x: _clean_language(x, compiled_langs1))
    df['question_text'] = df['question_text'].apply(lambda x: _clean_language(x, compiled_langs2))
    return df

def _clean_language(x, compiled_re):
    return compiled_re.sub(' <lang> ', x)

def clean_bitcoin(df, bitcoins):
    compiled_bitcoins = re.compile('(%s)' % '|'.join(bitcoins))
    df['question_text'] = df['question_text'].apply(lambda x: _clean_language(x, compiled_bitcoins))
    return df

def _clean_bitcoin(x, compiled_re):
    return compiled_re.sub(' Bitcoin ', x)

def clean_math(df):
    math_puncts = 'θπα÷⁴≠β²¾∫≥⇒¬∠＝∑Φ√½¼'
    math_puncts_long = [r'\\frac', r'\[math\]', r'\[/math\]', r'\\lim']
    compiled_math = re.compile('(%s)' % '|'.join(math_puncts))
    compiled_math_long = re.compile('(%s)' % '|'.join(math_puncts_long))
    df['question_text'] = df['question_text'].apply(lambda x: _clean_math(x, compiled_math_long))
    df['question_text'] = df['question_text'].apply(lambda x: _clean_math(x, compiled_math))
    return df

def _clean_math(x, compiled_re):
    return compiled_re.sub(' <math> ', x)

def clean_lower(df):
    df["question_text"] = df["question_text"].apply(lambda x: x.lower())
    return df

def clean_puncts(df, puncts):
    df['question_text'] = df['question_text'].apply(lambda x: _clean_puncts(x, puncts))
    return df
    
def _clean_puncts(x, puncts):
    x = str(x)
    # added space around puncts after replace
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, f' {punct} ')
    return x

def clean_bad_case_words(df, bad_case_words):
    compiled_bad_case_words = re.compile('(%s)' % '|'.join(bad_case_words.keys()))
    def replace(match):
        return bad_case_words[match.group(0)]
    df['question_text'] = df["question_text"].apply(
        lambda x: _clean_bad_case_words(x, compiled_bad_case_words, replace)
    )
    return df
    
def _clean_bad_case_words(x, compiled_re, replace):
    return compiled_re.sub(replace, x)

def clean_spells(df, spells):
    compiled_spells = re.compile('(%s)' % '|'.join(spells.keys()))
    def replace(match):
        return spells[match.group(0)]
    df['question_text'] = df["question_text"].apply(
        lambda x: _clean_spells(x, compiled_spells, replace)
    )
    return df
    
def _clean_spells(x, compiled_re, replace):
    return compiled_re.sub(replace, x)

def clean_abbreviation(df, abbreviations):
    compiled_abbreviation = re.compile('(%s)' % '|'.join(abbreviations.keys()))
    def replace(match):
        return abbreviations[match.group(0)]
    df['question_text'] = df["question_text"].apply(
        lambda x: _clean_abreviation(x, compiled_abbreviation, replace)
    )
    return df
    
def _clean_abreviation(x, compiled_re, replace):
    return compiled_re.sub(replace, x)

def clean_space(df):
    compiled_re = re.compile(r"\s+")
    df['question_text'] = df["question_text"].apply(lambda x: _clean_space(x, compiled_re))
    return df

def _clean_space(x, compiled_re):
    return compiled_re.sub(" ", x)
        
def prepare_tokenizer(texts, max_words):
    tokenizer = Tokenizer(num_words=max_words, filters='', oov_token='<unk>')
    tokenizer.fit_on_texts(list(texts))
    return tokenizer

def tokenize_and_padding(texts, tokenizer, max_length):
    texts = tokenizer.texts_to_sequences(texts)
    texts = pad_sequences(texts, maxlen=max_length)
    return texts

def get_all_vocabs(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab




def add_features(df):
    df['question_text'] = df['question_text'].apply(lambda x:str(x))
    df['lower_question_text'] = df['question_text'].apply(lambda x: x.lower())
    df['total_length'] = df['question_text'].apply(len)
    df['capitals'] = df['question_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']), axis=1)
    df['num_words'] = df['question_text'].str.count('\S+')
    df['num_unique_words'] = df['question_text'].apply(lambda comment: len(set(comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words'] 
    return df[['caps_vs_length', 'words_vs_unique', 'num_words']]




class Embeddings(nn.Module):
    
    def __init__(self, config: Config, tokenizer, all_vocabs, embedding_weights = None):
        super(Embeddings, self).__init__()
        
        self.embedding_map = {
            'fasttext': self._load_fasttext,
            'glove': self._load_glove,
            'paragram': self._load_paragram
        }
        self.c = config
        self.tokenizer = tokenizer
        self.all_vocabs = all_vocabs
        
        if embedding_weights is None:
            embedding_weights = self._load_embeddings(self.c.embeddings)
            
        self.original_embedding_weights = embedding_weights
        self.embeddings = nn.Embedding(self.c.vocab_size + 1, self.c.embedding_size, padding_idx=0)
        self.embeddings.weight = nn.Parameter(embedding_weights)
        self.embeddings.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(self.c.embedding_dropout)
        
    def forward(self, x):
        embedding = self.embeddings(x)
        if self.training:
            embedding += torch.randn_like(embedding) * self.c.embedding_noise_var
        return self.embedding_dropout(embedding.permute(0, 2, 1)).permute(0, 2, 1)
    
    def reset_weights(self):
        self.embeddings.weight = nn.Parameter(self.original_embedding_weights)
        self.embeddings.weight.requires_grad = False
    
    def _load_embeddings(self, embedding_list: list):
        embedding_weights = np.zeros((self.c.vocab_size, self.c.embedding_size))
        pool = Pool(num_cores)
        embedding_weights = np.mean(pool.map(self._load_an_embedding, embedding_list), 0)
        pool.close()
        pool.join()
        return torch.tensor(embedding_weights, dtype=torch.float32)

    def _load_an_embedding(self, emb):
        return self.embedding_map[emb](self.tokenizer.word_index)
        
    def _get_embeddings_pair(self, word, *arr): 
        return word, np.asarray(arr, dtype='float32')
        
    def _make_embeddings(self, embeddings_index, word_index, emb_mean, emb_std):
        nb_words = min(self.c.vocab_size, len(word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, self.c.embedding_size))
        embedding_matrix[0] = np.zeros(self.c.embedding_size)
        for word, i in word_index.items():
            if i >= self.c.vocab_size:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix
    
    def _load_glove(self, word_index):
        print('loading glove')
        filepath = self.c.datadir / 'embeddings/glove.840B.300d/glove.840B.300d.txt'
        embeddings_index = dict(
            self._get_embeddings_pair(*o.split(" "))
            for o in open(filepath)
            if o.split(" ")[0] in word_index
        )
        emb_mean, emb_std = -0.005838499, 0.48782197
        return self._make_embeddings(embeddings_index, word_index, emb_mean, emb_std)
    
    def _load_fasttext(self, word_index):    
        print('loading fasttext')
        filepath = self.c.datadir / 'embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
        embeddings_index = dict(
            self._get_embeddings_pair(*o.split(" "))
            for o in open(filepath)
            if len(o) > 100 and o.split(" ")[0] in word_index
        )
        emb_mean, emb_std = -0.0033469985, 0.109855495
        return self._make_embeddings(embeddings_index, word_index, emb_mean, emb_std)

    def _load_paragram(self, word_index):
        print('loading paragram')
        filepath = self.c.datadir / 'embeddings/paragram_300_sl999/paragram_300_sl999.txt'
        embeddings_index = dict(
            self._get_embeddings_pair(*o.split(" "))
            for o in open(filepath, encoding="utf8", errors='ignore')
            if len(o) > 100 and o.split(" ")[0] in word_index
        )
        emb_mean, emb_std = -0.0053247833, 0.49346462
        return self._make_embeddings(embeddings_index, word_index, emb_mean, emb_std)




num_cores = 2
@timer
def df_parallelize_run(df, func, num_cores=2):
    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df




train_df, test_df = load_data(c.datadir)
train_features = df_parallelize_run(train_df, add_features)
test_features = df_parallelize_run(test_df, add_features)
train_df = df_parallelize_run(train_df, clean)
test_df = df_parallelize_run(test_df, clean)
train_x, train_y = train_df['question_text'].values, train_df['target'].values
test_x = test_df['question_text'].values
tokenizer = prepare_tokenizer(train_x, c.vocab_size)
train_x = tokenize_and_padding(train_x, tokenizer, c.max_length)
test_x = tokenize_and_padding(test_x, tokenizer, c.max_length)




ss = StandardScaler()
ss.fit(np.vstack((train_features, test_features)))
train_features = ss.transform(train_features)
test_features = ss.transform(test_features)




start = time.time()
all_vocabs = get_all_vocabs(train_df['question_text'])
print('all_vocabs: ', len(all_vocabs))
embeddings = Embeddings(c, tokenizer, all_vocabs)
print(time.time() - start)




class GRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(GRULayer, self).__init__()
        
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bias=False,
                          bidirectional=True,
                          batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.init_weights()
        
    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for k in ih:
            nn.init.xavier_uniform_(k)
        for k in hh:
            nn.init.orthogonal_(k)
        for k in b:
            nn.init.constant_(k, 0)

    def forward(self, x):
        gru_outputs, gru_state = self.gru(x)
        return self.dropout(gru_outputs), gru_state
            




class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(LSTMLayer, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bias=False,
                            bidirectional=True,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.init_weights()
        
    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for k in ih:
            nn.init.xavier_uniform_(k)
        for k in hh:
            nn.init.orthogonal_(k)
        for k in b:
            nn.init.constant_(k, 0)

    def forward(self, x):
        lstm_outputs, (lstm_states, _) = self.lstm(x)
        return self.dropout(lstm_outputs), lstm_states




# https://github.com/anandsaha/pytorch.cyclic.learning.rate/blob/master/cls.py
class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
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

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

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




class SimpleRNN(nn.Module):
    def __init__(self, config: Config, embeddings):
        super(SimpleRNN, self).__init__()
        self.c = config
        
        self.embedding = embeddings
        self.lstm = LSTMLayer(input_size=self.c.embedding_size,
                              hidden_size=self.c.hidden_size,
                              num_layers=self.c.num_layers,
                              dropout_rate=self.c.layer_dropout)
        self.gru = GRULayer(input_size=self.c.hidden_size*2,
                            hidden_size=self.c.hidden_size,
                            num_layers=self.c.num_gru_layers,
                            dropout_rate=self.c.layer_dropout)
        
        self.cell_dropout = nn.Dropout(self.c.layer_dropout)
        self.linear = nn.Linear(self.c.dense_size[0], self.c.dense_size[1])
        self.batch_norm = torch.nn.BatchNorm1d(self.c.dense_size[1])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.c.layer_dropout)
        self.out = nn.Linear(self.c.dense_size[1], self.c.output_size)
        
    def forward(self, x, features):
        h_embedding = self.embedding(x)
        o_lstm, h_lstm = self.lstm(h_embedding)
        o_gru, h_gru = self.gru(o_lstm)
        
        avg_pool = torch.mean(o_gru, 1)
        max_pool, _ = torch.max(o_gru, 1)
        
        h_lstm = self.cell_dropout(torch.cat(h_lstm.split(1, 0), -1).squeeze(0))
        h_gru = self.cell_dropout(torch.cat(h_gru.split(1, 0), -1).squeeze(0))

        concat = torch.cat([h_lstm, h_gru, avg_pool, max_pool, features], 1)
        concat = self.linear(concat)
        concat = self.batch_norm(concat)
        concat = self.relu(concat)
        concat = self.dropout(concat)
        out = self.out(concat)
        
        return out




def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@timer
def threshold_search(y_true, y_proba, plot=False):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.001) 
    F = 2 / (1/precision + 1/recall)
    best_score = np.max(F)
    best_th = thresholds[np.argmax(F)]
    if plot:
        plt.plot(thresholds, F, '-b')
        plt.plot([best_th], [best_score], '*r')
        plt.show()
    search_result = {'threshold': best_th , 'f1': best_score}
    return search_result 




def cut_length(data, mask):
    max_length = data.shape[1]
    transposed = torch.transpose(data, 1, 0)
    res = (transposed == mask).all(1)
    for i, r in enumerate(res):
        if r == 0:
            break
    data = data[:, -(max_length - i):]
    return data




@timer
def training(train_x, train_f, train_y, test_x, test_f, c, embeddings):
    splits = list(StratifiedKFold(n_splits=c.num_cv_splits, shuffle=True, random_state=SEED).split(train_x, train_y))
    x_test_cuda = torch.tensor(test_x, dtype=torch.long).cuda(cuda_idx)
    f_test_cuda = torch.tensor(test_f, dtype=torch.float).cuda(cuda_idx)
    test = torch.utils.data.TensorDataset(x_test_cuda, f_test_cuda)
    test_loader = torch.utils.data.DataLoader(test, batch_size=c.test_batch_size, shuffle=False)
    train_preds = np.zeros((len(train_x)))
    test_preds = np.zeros((len(test_x)))
    
    mask = torch.zeros((c.max_length, 1), dtype=torch.long).cuda(cuda_idx)
    
    for i, (train_idx, valid_idx) in enumerate(splits):
        x_train_fold = torch.tensor(train_x[train_idx], dtype=torch.long).cuda(cuda_idx)
        f_train_fold = torch.tensor(train_f[train_idx], dtype=torch.float).cuda(cuda_idx)
        y_train_fold = torch.tensor(train_y[train_idx, np.newaxis], dtype=torch.float32).cuda(cuda_idx)
        x_val_fold = torch.tensor(train_x[valid_idx], dtype=torch.long).cuda(cuda_idx)
        f_val_fold = torch.tensor(train_f[valid_idx], dtype=torch.float).cuda(cuda_idx)
        y_val_fold = torch.tensor(train_y[valid_idx, np.newaxis], dtype=torch.float32).cuda(cuda_idx)

        embeddings.reset_weights()
        model = SimpleRNN(c, embeddings)
        model.cuda(cuda_idx)
        model.embedding.original_embedding_weights.cuda(cuda_idx)

        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=c.learning_rate)
        scheduler = CyclicLR(optimizer, base_lr=c.learning_rate, max_lr=c.max_learning_rate,
                             step_size=c.clr_step_size, mode='exp_range', gamma=0.9999)

        train = torch.utils.data.TensorDataset(x_train_fold, f_train_fold, y_train_fold)
        valid = torch.utils.data.TensorDataset(x_val_fold, f_val_fold, y_val_fold)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=c.test_batch_size, shuffle=False)

        print(f'Fold {i + 1}')

        for epoch in range(c.num_epochs):
            train_loader = torch.utils.data.DataLoader(train, batch_size=c.batch_size[epoch], shuffle=True)
            start_time = time.time()
            if epoch >= c.num_epochs - c.num_embedding_learn_epochs:
                model.embedding.embeddings.weight.requires_grad = True

            model.train()
            avg_loss = 0.
            for x_batch, f_batch, y_batch in tqdm(train_loader, disable=True):
                x_batch = cut_length(x_batch, mask)
                y_pred = model(x_batch, f_batch)
                scheduler.batch_step()
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), c.clip_grad)
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)

            model.eval()
            valid_preds_fold = np.zeros((x_val_fold.size(0)))
            test_preds_fold = np.zeros(len(test_x))
            avg_val_loss = 0.

            # validation prediction
            for i, (x_batch, f_batch, y_batch) in enumerate(valid_loader):
                x_batch = cut_length(x_batch, mask)
                y_pred = model(x_batch, f_batch).detach()
                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds_fold[i * c.test_batch_size:(i+1) * c.test_batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

            for ps in optimizer.param_groups:
                current_learning_rate = ps['lr']
            elapsed_time = time.time() - start_time 
            print('Epoch {}/{}  loss={:.4f}  val_loss={:.4f}  time={:.2f}s  lr={:.6f}'.format(
                epoch + 1, c.num_epochs, avg_loss, avg_val_loss, elapsed_time, current_learning_rate))

        # test prediction
        for i, (x_batch, f_batch) in enumerate(test_loader):
            x_batch = cut_length(x_batch, mask)
            y_pred = model(x_batch, f_batch).detach()

            test_preds_fold[i * c.test_batch_size:(i+1) * c.test_batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        train_preds[valid_idx] = valid_preds_fold
        test_preds += test_preds_fold / len(splits)
    return train_preds, test_preds




train_preds, test_preds = training(train_x, train_features, train_y, test_x, test_features, c, embeddings)




search_result = threshold_search(train_y, train_preds)
search_result




all_elapsed = time.time() - all_start
all_elapsed




# kernel only
submission = test_df[['qid']].copy()
submission['prediction'] = (test_preds > search_result['threshold']).astype(int)
submission.to_csv("submission.csv", index=False)

