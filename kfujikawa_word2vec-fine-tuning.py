#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import random
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import sklearn
import torch
from gensim.models import Word2Vec, KeyedVectors

get_ipython().run_line_magic('load_ext', 'Cython')


# In[2]:


get_ipython().run_cell_magic('cython', '', "import re\nfrom multiprocessing import Pool\n\nimport numpy as np\ncimport numpy as np\n\n\ncdef class StringReplacer:\n    cpdef public dict rule\n    cpdef list keys\n    cpdef list values\n    cpdef int n_rules\n\n    def __init__(self, dict rule):\n        self.rule = rule\n        self.keys = list(rule.keys())\n        self.values = list(rule.values())\n        self.n_rules = len(rule)\n\n    def __call__(self, str x):\n        cdef int i\n        for i in range(self.n_rules):\n            if self.keys[i] in x:\n                x = x.replace(self.keys[i], self.values[i])\n        return x\n\n    def __getstate__(self):\n        return (self.rule, self.keys, self.values, self.n_rules)\n\n    def __setstate__(self, state):\n        self.rule, self.keys, self.values, self.n_rules = state\n        \n        \ncdef class RegExpReplacer:\n    cdef dict rule\n    cdef list keys\n    cdef list values\n    cdef regexp\n    cdef int n_rules\n\n    def __init__(self, dict rule):\n        self.rule = rule\n        self.keys = list(rule.keys())\n        self.values = list(rule.values())\n        self.regexp = re.compile('(%s)' % '|'.join(self.keys))\n        self.n_rules = len(rule)\n\n    @property\n    def rule(self):\n        return self.rule\n\n    def __call__(self, str x):\n        def replace(match):\n            x = match.group(0)\n            if x in self.rule:\n                return self.rule[x]\n            else:\n                for i in range(self.n_rules):\n                    x = re.sub(self.keys[i], self.values[i], x)\n                return x\n        return self.regexp.sub(replace, x)\n    \n\ncdef class ApplyNdArray:\n    cdef func\n    cdef dtype\n    cdef dims\n    cdef int processes\n\n    def __init__(self, func, processes=1, dtype=object, dims=None):\n        self.func = func\n        self.processes = processes\n        self.dtype = dtype\n        self.dims = dims\n\n    def __call__(self, arr):\n        if self.processes == 1:\n            return self.apply(arr)\n        else:\n            return self.apply_parallel(arr)\n\n    cpdef apply(self, arr):\n        cdef int i\n        cdef int n = len(arr)\n        if self.dims is not None:\n            shape = (n, *self.dims)\n        else:\n            shape = n\n        cdef res = np.empty(shape, dtype=self.dtype)\n        for i in range(n):\n            res[i] = self.func(arr[i])\n        return res\n\n    cpdef apply_parallel(self, arr):\n        cdef list arrs = np.array_split(arr, self.processes)\n        with Pool(processes=self.processes) as pool:\n            outputs = pool.map(self.apply, arrs)\n        return np.concatenate(outputs, axis=0)")


# In[3]:


def load_qiqc(n_rows=None):
    train_df = pd.read_csv(f'{os.environ["DATADIR"]}/train.csv', nrows=n_rows)
    submit_df = pd.read_csv(f'{os.environ["DATADIR"]}/test.csv', nrows=n_rows)
    n_labels = {
        0: (train_df.target == 0).sum(),
        1: (train_df.target == 1).sum(),
    }
    train_df['target'] = train_df.target.astype('f')
    train_df['weights'] = train_df.target.apply(lambda t: 1 / n_labels[t])

    return train_df, submit_df


def build_datasets(train_df, submit_df, holdout, seed):
    submit_dataset = QIQCDataset(submit_df)
    if holdout:
        # Train : Test split for holdout training
        splitter = sklearn.model_selection.StratifiedShuffleSplit(
            n_splits=1, test_size=0.1, random_state=seed)
        train_indices, test_indices = list(splitter.split(
            train_df, train_df.target))[0]
        train_indices.sort(), test_indices.sort()
        train_dataset = QIQCDataset(
            train_df.iloc[train_indices].reset_index(drop=True))
        test_dataset = QIQCDataset(
            train_df.iloc[test_indices].reset_index(drop=True))
    else:
        train_dataset = QIQCDataset(train_df)
        test_dataset = QIQCDataset(train_df.head(0))

    return train_dataset, test_dataset, submit_dataset


class QIQCDataset(object):

    def __init__(self, df):
        self.df = df

    @property
    def tokens(self):
        return self.df.tokens.values

    @tokens.setter
    def tokens(self, tokens):
        self.df['tokens'] = tokens

    @property
    def positives(self):
        return self.df[self.df.target == 1]

    @property
    def negatives(self):
        return self.df[self.df.target == 0]

    def build(self, device):
        self._X = self.tids
        self.X = torch.Tensor(self._X).type(torch.long).to(device)
        if 'target' in self.df:
            self._t = self.df.target[:, None]
            self._W = self.df.weights
            self.t = torch.Tensor(self._t).type(torch.float).to(device)
            self.W = torch.Tensor(self._W).type(torch.float).to(device)
        if hasattr(self, '_X2'):
            self.X2 = torch.Tensor(self._X2).type(torch.float).to(device)
        else:
            self._X2 = np.zeros((self._X.shape[0], 1), 'f')
            self.X2 = torch.Tensor(self._X2).type(torch.float).to(device)

    def build_labeled_dataset(self, indices):
        return torch.utils.data.TensorDataset(
            self.X[indices], self.X2[indices],
            self.t[indices], self.W[indices])
    
## Pretrained vector

def load_pretrained_vectors(names, token2id, test=False):
    assert isinstance(names, list)
    with Pool(processes=len(names)) as pool:
        f = partial(load_pretrained_vector, token2id=token2id, test=test)
        vectors = pool.map(f, names)
    return dict([(n, v) for n, v in zip(names, vectors)])


def load_pretrained_vector(name, token2id, test=False):
    loader = dict(
        gnews=GNewsPretrainedVector,
        wnews=WNewsPretrainedVector,
        paragram=ParagramPretrainedVector,
        glove=GlovePretrainedVector,
    )
    return loader[name].load(token2id, test)


class BasePretrainedVector(object):

    @classmethod
    def load(cls, token2id, test=False, limit=None):
        embed_shape = (len(token2id), 300)
        freqs = np.zeros((len(token2id)), dtype='f')

        if test:
            np.random.seed(0)
            vectors = np.random.normal(0, 1, embed_shape)
            vectors[0] = 0
            vectors[len(token2id) // 2:] = 0
        else:
            vectors = np.zeros(embed_shape, dtype='f')
            path = f'{os.environ["DATADIR"]}/{cls.path}'
            for i, o in enumerate(
                    open(path, encoding="utf8", errors='ignore')):
                token, *vector = o.split(' ')
                token = str.lower(token)
                if token not in token2id or len(o) <= 100:
                    continue
                if limit is not None and i > limit:
                    break
                freqs[token2id[token]] += 1
                vectors[token2id[token]] += np.array(vector, 'f')

        vectors[freqs != 0] /= freqs[freqs != 0][:, None]
        vec = KeyedVectors(300)
        vec.add(list(token2id.keys()), vectors, replace=True)

        return vec


class GNewsPretrainedVector(object):

    name = 'GoogleNews-vectors-negative300'
    path = f'embeddings/{name}/{name}.bin'

    @classmethod
    def load(cls, tokens, limit=None):
        raise NotImplementedError
        path = f'{os.environ["DATADIR"]}/{cls.path}'
        return KeyedVectors.load_word2vec_format(
            path, binary=True, limit=limit)


class WNewsPretrainedVector(BasePretrainedVector):

    name = 'wiki-news-300d-1M'
    path = f'embeddings/{name}/{name}.vec'


class ParagramPretrainedVector(BasePretrainedVector):

    name = 'paragram_300_sl999'
    path = f'embeddings/{name}/{name}.txt'


class GlovePretrainedVector(BasePretrainedVector):

    name = 'glove.840B.300d'
    path = f'embeddings/{name}/{name}.txt'

    
class WordVocab(object):

    def __init__(self, mincount=1):
        self.counter = Counter()
        self.n_documents = 0
        self._counters = {}
        self._n_documents = defaultdict(int)
        self.mincount = mincount

    def __len__(self):
        return len(self.token2id)

    def add_documents(self, documents, name):
        self._counters[name] = Counter()
        for document in documents:
            bow = dict.fromkeys(document, 1)
            self._counters[name].update(bow)
            self.counter.update(bow)
            self.n_documents += 1
            self._n_documents[name] += 1

    def build(self):
        counter = dict(self.counter.most_common())
        self.word_freq = {
            **{'<PAD>': 0},
            **counter,
        }
        self.token2id = {
            **{'<PAD>': 0},
            **{word: i + 1 for i, word in enumerate(counter)}
        }
        self.lfq = np.array(list(self.word_freq.values())) < self.mincount
        self.hfq = ~self.lfq
        
        
class PunctSpacer(StringReplacer):

    def __init__(self, edge_only=False):
        puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', '█', '½', '…', '“', '★', '”', '–', '●', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', '¯', '♦', '¤', '▲', '¸', '¾', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]  # NOQA
        if edge_only:
            rule = {
                **dict([(f' {p}', f' {p} ') for p in puncts]),
                **dict([(f'{p} ', f' {p} ') for p in puncts]),
            }
        else:
            rule = dict([(p, f' {p} ') for p in puncts])
        super().__init__(rule)
        
        
class NumberReplacer(RegExpReplacer):

    def __init__(self, with_underscore=False):
        prefix, suffix = '', ''
        if with_underscore:
            prefix += ' __'
            suffix = '__ '
        rule = {
            '[0-9]{5,}': f'{prefix}#####{suffix}',
            '[0-9]{4}': f'{prefix}####{suffix}',
            '[0-9]{3}': f'{prefix}###{suffix}',
            '[0-9]{2}': f'{prefix}##{suffix}',
        }
        super().__init__(rule)


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Pipeline(object):

    def __init__(self, *modules):
        self.modules = modules

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        return x


# In[4]:


get_ipython().run_cell_magic('time', '', "os.environ['DATADIR'] = '/kaggle/input'\nset_seed(0)\ntrain_df, submit_df = load_qiqc()\ndatasets = build_datasets(train_df, submit_df, holdout=False, seed=0)\ntrain_dataset, test_dataset, submit_dataset = datasets")


# In[5]:


get_ipython().run_cell_magic('time', '', 'tokenize = Pipeline(\n    str.lower,\n    PunctSpacer(),\n    NumberReplacer(with_underscore=True),\n    str.split\n)\napply_tokenize = ApplyNdArray(tokenize, processes=2, dtype=object)\ntrain_dataset.tokens, test_dataset.tokens, submit_dataset.tokens = \\\n    [apply_tokenize(d.df.question_text.values) for d in datasets]\ntokens = np.concatenate([d.tokens for d in datasets])')


# In[6]:


get_ipython().run_cell_magic('time', '', "vocab = WordVocab(mincount=1)\nvocab.add_documents(train_dataset.positives.tokens, 'train-pos')\nvocab.add_documents(train_dataset.negatives.tokens, 'train-neg')\nvocab.add_documents(test_dataset.positives.tokens, 'test-pos')\nvocab.add_documents(test_dataset.negatives.tokens, 'test-neg')\nvocab.add_documents(submit_dataset.df.tokens, 'submit')\nvocab.build()")


# In[7]:


get_ipython().run_cell_magic('time', '', "glove = load_pretrained_vector('glove', vocab.token2id)\nword_vectors = {'glove': glove}\nunk = (glove.vectors == 0).all(axis=1)\nknown = ~unk")


# In[8]:


params = dict(
    min_count=1,
    workers=1,
    iter=5,
    size=300,
)


# In[9]:


get_ipython().run_cell_magic('time', '', "model = Word2Vec(**params)\nmodel.build_vocab_from_freq(vocab.word_freq)\nmodel.train(tokens, total_examples=len(tokens), epochs=model.epochs)\nword_vectors['scratch'] = model.wv")


# In[10]:


get_ipython().run_cell_magic('time', '', "model = Word2Vec(**params)\nmodel.build_vocab_from_freq(vocab.word_freq)\nidxmap = np.array(\n    [vocab.token2id[w] for w in model.wv.index2entity])\nmodel.wv.vectors[:] = glove.vectors[idxmap]\nmodel.trainables.syn1neg[:] = glove.vectors[idxmap]\nmodel.train(tokens, total_examples=len(tokens), epochs=model.epochs)\nword_vectors['finetune'] = model.wv")


# In[11]:


word = 'obama'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[12]:


word = 'lgbt'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[13]:


word = 'cosx'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[14]:


word = 'brexit'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[15]:


word = 'coinbase'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[16]:


word = 'tensorflow'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[17]:


word = 'cos2x'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[18]:


word = 'kubernetes'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[19]:


word = 'gdpr'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[20]:


word = '0bama'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[21]:


word = 'germnay'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[22]:


word = 'gogole'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[23]:


word = 'javadoc'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[24]:


word = 'cython'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[25]:


word = 'compresses'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[26]:


word = 'xgboost'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[27]:


word = '2sinxcosx'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[28]:


word = 'germeny'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[29]:


word = 'bigender'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[30]:


word = 'youcanttellyourstoryfromthe'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[31]:


word = '5gfwdhf4rz'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[32]:


word = 'ॡ'
print(vocab.word_freq[word])
pd.DataFrame({name: kv.most_similar(word) for name, kv in word_vectors.items()})


# In[33]:


pd.DataFrame(np.array(list(vocab.word_freq.items()))).to_csv('all.csv', index=False, sep='\t')
pd.DataFrame(np.array(list(vocab.word_freq.items()))[unk]).to_csv('unk.csv', index=False, sep='\t')
pd.DataFrame(np.array(list(vocab.word_freq.items()))[known]).to_csv('known.csv', index=False, sep='\t')


# In[34]:




