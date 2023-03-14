#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm_notebook
import pickle
import gc
from sklearn.model_selection import KFold

import os
import operator
import random
from multiprocessing import Pool
from gensim.models import KeyedVectors
import re
from tqdm import tqdm
from collections import defaultdict
import json
import dask.dataframe as ddf
import platform
from torch.utils import data
from keras.preprocessing import text, sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import time
from sklearn import metrics
from keras.preprocessing.sequence import pad_sequences




tqdm.pandas()


def set_seed(seed: int = 0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def df_parallelize_run(df: pd.DataFrame(), func, npartitions=os.cpu_count()):
    if platform.system() == 'Windows':
        dask_dataframe = ddf.from_pandas(df, npartitions=os.cpu_count())
        result = dask_dataframe.map_partitions(func, meta=df)
        df = result.compute()
    elif platform.system() == 'Linux':
        df_split = np.array_split(df, npartitions)
        pool = Pool(npartitions)
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()

    else:
        print('No idea what to do with your OS :(')

    return df


def load_embed(filepath: str):
    """
    Load embeddings.

    :param filepath: path to the embeddings
    :return:
    """
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')
    if '.pkl' in filepath:
        with open(filepath,'rb') as f:
            return pickle.load(f)
    if 'news' in filepath:
        embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in open(filepath) if len(o) > 100)
    elif '.bin' in filepath:
        embeddings_index = KeyedVectors.load_word2vec_format(filepath, binary=True)
    else:
        embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in open(filepath, encoding='utf-8'))

    return embeddings_index


def build_vocab(texts: pd.Series()) -> dict:
    """
    Creates a vocabulary of the text, which can be used to check text coverage.

    :param texts: pandas series with text.
    :return: dictionary with words and their counts
    """
    # sentences = texts.progress_apply(lambda x: x.split()).values
    vocab = defaultdict(lambda: 0)
    for sentence in texts.values:
        for word in str(sentence).split():
            vocab[word] += 1

    return vocab


def check_coverage(vocab: dict, embeddings_index) -> list:
    """
    Check word coverage of embedding. Returns words which aren't in embeddings_index

    :param vocab: Dictionary with words and their counts.
    :param embeddings_index: embedding index
    :return: list of tuples with unknown words and their count
    """
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        if word in embeddings_index:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        elif word.capitalize() in embeddings_index:
            known_words[word] = embeddings_index[word.capitalize()]
            nb_known_words += vocab[word]
        elif word.lower() in embeddings_index:
            known_words[word] = embeddings_index[word.lower()]
            nb_known_words += vocab[word]
        elif word.upper() in embeddings_index:
            known_words[word] = embeddings_index[word.upper()]
            nb_known_words += vocab[word]
        else:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]

    vocab_rate = len(known_words) / len(vocab)
    print(f'Found embeddings for {vocab_rate:.2%} of vocab')

    text_rate = nb_known_words / (nb_known_words + nb_unknown_words)
    print(f'Found embeddings for {text_rate:.2%} of all text')
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words


def vocab_check_coverage(df: pd.DataFrame(), *args) -> list:
    """
    Calculate word coverage for the passed dataframe and embeddings.
    Can do it for one or several embeddings.

    :param df: dataframe for which coverage rate will be calculated.
    :param args: one or several embeddings
    :return: list of dicts with out of vocab rate and words
    """

    oovs = []
    vocab = build_vocab(df['comment_text'])

    for emb in args:
        oov = check_coverage(vocab, emb)
        oov = {"oov_rate": len(oov) / len(vocab), 'oov_words': oov}
        oovs.append(oov)

    return oovs


def remove_space(text: str, spaces: list, only_clean: bool = True) -> str:
    """
    Remove extra spaces and ending space if any.

    :param text: text to clean
    :param text: spaces
    :param only_clean: simply clean texts or also replace texts
    :return: cleaned text
    """
    if not only_clean:
        for space in spaces:
            text = text.replace(space, ' ')

    text = text.strip()
    text = re.sub('\s+', ' ', text)

    return text


def replace_words(text: str, mapping: dict) -> str:
    """
    Replaces unusual punctuation with normal.

    :param text: text to clean
    :param mapping: dict with mapping
    :return: cleaned text
    """
    for word in mapping:
        if word in text:
            text = text.replace(word, mapping[word])

    return text


def clean_number(text: str) -> str:
    """
    Cleans numbers.

    :param text: text to clean
    :return: cleaned text
    """
    text = re.sub(r'(\d+)([a-zA-Z])', '\g<1> \g<2>', text)
    text = re.sub(r'(\d+) (th|st|nd|rd) ', '\g<1>\g<2> ', text)
    text = re.sub(r'(\d+),(\d+)', '\g<1>\g<2>', text)
    text = re.sub(r'(\d+),', '\g<1>', text)
    text = re.sub(r'(\d+)(e)(\d+)', '\g<1> \g<3>', text)

    return text


def spacing_punctuation(text: str, punctuation: str) -> str:
    """
    Add space before and after punctuation and symbols.

    :param text: text to clean
    :param punctuation: string with symbols
    :return: cleaned text
    """
    for punc in punctuation:
        if punc in text:
            text = text.replace(punc, f' {punc} ')

    return text


def fixing_with_regex(text) -> str:
    """
    Additional fixing of words.

    :param text: text to clean
    :return: cleaned text
    """

    mis_connect_list = ['\b(W|w)hat\b', '\b(W|w)hy\b', '(H|h)ow\b', '(W|w)hich\b', '(W|w)here\b', '(W|w)ill\b']
    mis_connect_re = re.compile('(%s)' % '|'.join(mis_connect_list))

    text = re.sub(r" (W|w)hat+(s)*[A|a]*(p)+ ", " WhatsApp ", text)
    text = re.sub(r" (W|w)hat\S ", " What ", text)
    text = re.sub(r" \S(W|w)hat ", " What ", text)
    text = re.sub(r" (W|w)hy\S ", " Why ", text)
    text = re.sub(r" \S(W|w)hy ", " Why ", text)
    text = re.sub(r" (H|h)ow\S ", " How ", text)
    text = re.sub(r" \S(H|h)ow ", " How ", text)
    text = re.sub(r" (W|w)hich\S ", " Which ", text)
    text = re.sub(r" \S(W|w)hich ", " Which ", text)
    text = re.sub(r" (W|w)here\S ", " Where ", text)
    text = re.sub(r" \S(W|w)here ", " Where ", text)
    text = mis_connect_re.sub(r" \1 ", text)
    text = text.replace("What sApp", ' WhatsApp ')

    # Clean repeated letters.
    text = re.sub(r"(I|i)(I|i)+ng", "ing", text)
    text = re.sub(r"(-+|\.+)", " ", text)

    text = re.sub(r'[\x00-\x1f\x7f-\x9f\xad]', '', text)
    text = re.sub(r'(\d+)(e)(\d+)', r'\g<1> \g<3>', text)  # is a dup from above cell...
    text = re.sub(r"(-+|\.+)\s?", "  ", text)
    text = re.sub("\s\s+", " ", text)
    text = re.sub(r'ᴵ+', '', text)

    text = re.sub(r"(W|w)on(\'|\’)t ", "will not ", text)
    text = re.sub(r"(C|c)an(\'|\’)t ", "can not ", text)
    text = re.sub(r"(Y|y)(\'|\’)all ", "you all ", text)
    text = re.sub(r"(Y|y)a(\'|\’)ll ", "you all ", text)

    text = re.sub(r"(I|i)(\'|\’)m ", "i am ", text)
    text = re.sub(r"(A|a)in(\'|\’)t ", "is not ", text)
    text = re.sub(r"n(\'|\’)t ", " not ", text)
    text = re.sub(r"(\'|\’)re ", " are ", text)
    text = re.sub(r"(\'|\’)s ", " is ", text)
    text = re.sub(r"(\'|\’)d ", " would ", text)
    text = re.sub(r"(\'|\’)ll ", " will ", text)
    text = re.sub(r"(\'|\’)t ", " not ", text)
    text = re.sub(r"(\'|\’)ve ", " have ", text)

    text = re.sub(
        r'(by|been|and|are|for|it|TV|already|justhow|some|had|is|will|would|should|shall|must|can|his|here|there|them|these|their|has|have|the|be|that|not|was|he|just|they|who)(how)',
        '\g<1> \g<2>', text)

    return text


def load_preprocessing_data() -> dict:
    """
    Loads dict with various mappings and strings for cleaning.

    :return:
    """

    if os.path.exists('../input/jigsaw-public-files/mapping_dict.json'):
        path = '../input/jigsaw-public-files/mapping_dict.json'
    else:
        path = '../input/mapping_dict.json'
        
    with open(path, 'r') as f:
        mapping_dict = json.load(f)

    # combine several dicts into one
    replace_dict = {**mapping_dict['contraction_mapping'],
                    **mapping_dict['mispell_dict'],
                    **mapping_dict['special_punc_mappings'],
                    **mapping_dict['rare_words_mapping'],
                    **mapping_dict['bad_case_words'],
                    **mapping_dict['mis_spell_mapping']}

    mapping_dict = {'spaces': mapping_dict['spaces'],
                    'punctuation': mapping_dict['punctuation'],
                    'words_to_replace': replace_dict}

    return mapping_dict


def preprocess(text: str) -> str:
    """
    Apply all preprocessing.

    :param text: text to clean.
    :return: cleaned text
    """

    text = remove_space(text, mapping_dict['spaces'], only_clean=False)
    text = clean_number(text)
    text = spacing_punctuation(text, mapping_dict['punctuation'])
    text = fixing_with_regex(text)
    text = replace_words(text, mapping_dict['words_to_replace'])

    for punct in "/-'":
        if punct in text:
            text = text.replace(punct, ' ')

    text = clean_number(text)
    text = remove_space(text, mapping_dict['spaces'])

    return text


def text_clean_wrapper(df):
    df["comment_text"] = df["comment_text"].apply(preprocess).astype(str)
    return df

mapping_dict = load_preprocessing_data()


def build_matrix(word_index, path: str, embed_size: int):
    embedding_index = load_embed(path)
    embedding_matrix = np.zeros((len(word_index) + 1, embed_size))
    unknown_words = []

    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)

    return embedding_matrix, unknown_words


# class TextDataset(data.Dataset):
#     def __init__(self, text, lens, y=None):
#         self.text = text
#         self.lens = lens
#         self.y = y

#     def __len__(self):
#         return len(self.lens)

#     def __getitem__(self, idx):
#         if self.y is None:
#             return self.text[idx], self.lens[idx]
#         return self.text[idx], self.lens[idx], self.y[idx]


# class Collator(object):
#     def __init__(self, test: bool = False, max_length: int = 220):
#         self.test = test
#         self.max_length = max_length

#     def __call__(self, batch):

#         if self.test:
#             texts, lens = zip(*batch)
#         else:
#             texts, lens, target = zip(*batch)

#         lens = np.array(lens)
#         max_batch_len = min(max(lens), self.max_length)
#         max_batch_len = self.max_length
#         texts = torch.tensor(sequence.pad_sequences(texts, maxlen=max_batch_len), dtype=torch.long).cuda()

#         if self.test:
#             return texts

#         return texts, torch.tensor(target, dtype=torch.float32).cuda()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_model(model, x_train, y_train, x_val, y_val, test_loader, loss_fn, lr=0.001,
                batch_size=512, n_epochs=4,
                enable_checkpoint_ensemble=False, validate=False):
    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr)

    scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.003,
                         step_size=300, mode='exp_range', gamma=0.99994)

    train = torch.utils.data.TensorDataset(x_train, y_train)
    valid = torch.utils.data.TensorDataset(x_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    # all_test_preds = []
    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]

    for epoch in range(n_epochs):
        start_time = time.time()

        model.train()
        avg_loss = 0.

        for step, (seq_batch, y_batch) in enumerate(train_loader):
            y_pred = model(seq_batch)
            scheduler.batch_step()
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        model.eval()
        test_preds = np.zeros((len(test_loader.dataset)))

        val_loss = 0
        if validate:

            valid_preds = np.zeros((len(valid_loader.dataset)))
            for i, (seq_batch, y_batch) in enumerate(valid_loader):
                y_pred = model(seq_batch).detach()
                val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        for i, seq_batch in enumerate(test_loader):
            y_pred = model(seq_batch.long().cuda()).detach()

            test_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        # all_test_preds.append(test_preds)
        elapsed_time = time.time() - start_time
        print(
            f'Epoch {epoch + 1}/{n_epochs} \t loss={avg_loss:.4f} val_loss={val_loss:.4f} \t time={elapsed_time:.2f}s')

#     if enable_checkpoint_ensemble:
#         prediction = np.average(all_test_preds, weights=checkpoint_weights, axis=0)
#     else:
#         prediction = all_test_preds[-1]

    results_dict = {}
    results_dict['test_preds'] = test_preds
    if validate:
        results_dict['oof'] = valid_preds

    return results_dict


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, embedding_matrix_small, max_features: int = 120000, lstm_units: int = 128,
                 dense_hidden_units: int = 128):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)

        self.embedding1 = nn.Embedding(max_features, 30)
        self.embedding1.weight = nn.Parameter(torch.tensor(embedding_matrix_small, dtype=torch.float32))

        self.lstm1 = nn.LSTM(embed_size, lstm_units, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units, bidirectional=True, batch_first=True)

        self.lstm1s = nn.LSTM(30, int(lstm_units / 8), bidirectional=True, batch_first=True)
        self.lstm2s = nn.LSTM(int(lstm_units / 4), int(lstm_units / 8), bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(576, dense_hidden_units)
        self.linear2 = nn.Linear(576, dense_hidden_units)

        self.linear_out = nn.Linear(832, 1)
        self.linear_aux_out = nn.Linear(832, 6)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        embedding_small = self.embedding1(x)
        embedding_small = self.embedding_dropout(embedding_small)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        h_lstm1s, _ = self.lstm1s(embedding_small)
        h_lstm2s, _ = self.lstm2s(h_lstm1s)

        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)

        # global average pooling
        avg_pools = torch.mean(h_lstm2s, 1)
        # global max pooling
        max_pools, _ = torch.max(h_lstm2s, 1)

        h_conc = torch.cat((max_pool, avg_pool, max_pools, avg_pools), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))
        hidden = torch.cat((h_conc, h_conc_linear1, h_conc_linear2), 1)

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)

        return out


# def make_loaders(data: pd.DataFrame(), data_lens: list, target: pd.DataFrame() = None, test: bool=False):

#     collate = Collator(test)
#     if test:
#         dataset = TextDataset(data, data_lens)
#     else:
#         dataset = TextDataset(data, data_lens, target)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=collate)

#     return loader


def train_on_folds(X_train, x_train_lens, final_y_train, test_loader, x_test_lens, splits, embedding_matrix, embedding_matrix_small, n_epochs=2, validate=False, debug=False):
    if validate:
        scores = []

    test_preds = np.zeros((len(test_loader.dataset), len(splits)))
    train_oof = np.zeros((len(X_train), 1))

    for i, (train_idx, valid_idx) in enumerate(splits):
        # for debugging purposes, to make things faster
        if debug:
            train_idx = train_idx[:1000]

        x_train_fold = torch.tensor(X_train[train_idx], dtype=torch.long).cuda()
        y_train_fold = torch.tensor(final_y_train[train_idx], dtype=torch.float32).cuda()
        x_val_fold = torch.tensor(X_train[valid_idx], dtype=torch.long).cuda()
        y_val_fold = torch.tensor(final_y_train[valid_idx], dtype=torch.float32).cuda()

        print(f'Fold {i + 1}')

        set_seed(42 + i)
        model = NeuralNet(embedding_matrix, embedding_matrix_small)
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        model.cuda()

        results_dict = train_model(model,
                                   x_train_fold, 
                                   y_train_fold, 
                                   x_val_fold, 
                                   y_val_fold, test_loader, loss_fn=loss_fn, n_epochs=n_epochs, validate=True)

        if validate:
            train_oof[valid_idx] = results_dict['oof'].reshape(-1, 1)
            print(metrics.roc_auc_score(final_y_train[valid_idx][:,0], train_oof[valid_idx]))
            
        test_preds[:, i] = results_dict['test_preds']

    return test_preds


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
        return self.gamma ** (x)

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




# setting parameters.
set_seed(42)

crawl_embedding_path = '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'
glove_embedding_path = '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'
num_models = 2
lstm_units = 128
dense_hidden_units = 4 * lstm_units
max_len = 220
embed_size = 300
max_features = 120000




get_ipython().run_cell_magic('time', '', "load = True\nif not load:\n    train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')\n    test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')\n    train = df_parallelize_run(train, text_clean_wrapper)\n    test = df_parallelize_run(test, text_clean_wrapper)\n    train.to_csv('processed_train.csv', index=False)\n    test.to_csv('processed_test.csv', index=False)\nelse:\n    train = pd.read_csv('../input/jigsaw-public-files/train.csv')\n    test = pd.read_csv('../input/jigsaw-public-files/test.csv')\n    # after processing some of the texts are emply\n    train['comment_text'] = train['comment_text'].fillna('')\n    test['comment_text'] = test['comment_text'].fillna('')")




get_ipython().run_cell_magic('time', '', 'glove_embed = load_embed(glove_embedding_path)\noovs = vocab_check_coverage(train, glove_embed)')




del glove_embed
print(oovs[0]['oov_words'][:20])




get_ipython().run_cell_magic('time', '', "if not load:\n    tokenizer = text.Tokenizer(lower=False, num_words=max_features)\n    tokenizer.fit_on_texts(list(train['comment_text']) + list(test['comment_text']))\n    \n    # by default tokenizer keeps all words, I leave only top max_features\n    sorted_by_word_count = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)\n    tokenizer.word_index = {}\n    i = 0\n    for word,count in sorted_by_word_count:\n        if i == max_features:\n            break\n        tokenizer.word_index[word] = i + 1    # <= because tokenizer is 1 indexed\n        i += 1\n    \n    with open(f'tokenizer_{max_features}.pickle', 'wb') as f:\n        pickle.dump(tokenizer, f)\nelse:\n    with open(f'../input/jigsaw-public-files/tokenizer_{max_features}.pickle', 'rb') as f:\n        tokenizer = pickle.load(f)\n    \nX_train = tokenizer.texts_to_sequences(train['comment_text'])\nX_test = tokenizer.texts_to_sequences(test['comment_text'])\nx_train_lens = [len(i) for i in X_train]\nx_test_lens  = [len(i) for i in X_test]")




y_train = np.where(train['target'] >= 0.5, 1, 0)
y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
final_y_train = np.hstack([y_train[:, np.newaxis], y_aux_train])




get_ipython().run_cell_magic('time', '', "crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, crawl_embedding_path, embed_size)\nprint('n unknown words (crawl): ', len(unknown_words_crawl))\n\nglove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, glove_embedding_path, embed_size)\nprint('n unknown words (glove): ', len(unknown_words_glove))\n\nembedding_matrix = crawl_matrix * 0.5 +  glove_matrix * 0.5\n\ndel crawl_matrix\ndel glove_matrix\ngc.collect()")




embedding_matrix_small = np.zeros((embedding_matrix.shape[0], 30))




# splits for training
splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train, final_y_train))




X_train_padded = pad_sequences(X_train, maxlen = max_len)
X_test_padded = pad_sequences(X_test, maxlen = max_len)
batch_size = 512
test_loader = torch.utils.data.DataLoader(X_test_padded, batch_size=batch_size, shuffle=False)




test_preds = train_on_folds(X_train_padded, x_train_lens, final_y_train, test_loader, x_test_lens,
                            splits, embedding_matrix, embedding_matrix_small, n_epochs=1, validate=False, debug=False)




submission = pd.DataFrame.from_dict({
    'id': test['id'],
    'prediction': test_preds.mean(1)
})

submission.to_csv('submission.csv', index=False)

