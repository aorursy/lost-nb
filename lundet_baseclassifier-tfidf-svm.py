#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import regex as re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy
import unicodedata
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn import metrics
from sklearn.svm import LinearSVC
from tqdm import tqdm, tqdm_notebook
import operator
tqdm(tqdm_notebook).pandas()
#tqdm.pandas()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
datapath='../input'
RANDOM_STATE = 2
SHUFFLE = True
TEST_SIZE = 0.8
# Any results you write to the current directory are saved as output.




import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def load_trained_model(model, weights_path):
    model.load_weights(weights_path)
    return model




class DataReader(object):
    def __init__(self,
                 train_file,
                 module,
                 test_file=None):

        if not train_file:
            raise Exception("DataReader requires a train_file!")
        if not module:
            raise Exception("DataReader requires a model that can transform data!")
        self.raw_test = None
        if test_file:
            print("Loading test_data (%s) into dataframe" % test_file)
            self.test_data = pd.read_csv(test_file)
            self.raw_test = self.test_data[['question_text']]
            print("Test data with shape: ", self.test_data.shape)

        print("Loading train_data (%s) into dataframes" % train_file)
        self.train_data = pd.read_csv(train_file)
        self.raw_train = self.train_data[['question_text']]
        print("Train data with shape: ", self.train_data.shape)
        train_test_cut = self.train_data.shape[0]
        if isinstance(self.raw_test, pd.DataFrame):
            df_all = pd.concat([self.raw_train, self.raw_test],
                               axis=0).reset_index(drop=True)
        else:
            df_all = self.raw_train
        self.df_all = df_all


        print("Transforming the data")
        with timer('Transforming data'):
            if module:
                X_features = module.transform(df_all['question_text'])
            else:
                X_features = df_all['question_text']
            # Multiple Inputs
            if isinstance(X_features, list):
                self.X_train = [X[:train_test_cut] for X in X_features]
                self.X_test = [X[train_test_cut:] for X in X_features]
            else:
                self.X_train = X_features[:train_test_cut]
                self.X_test = X_features[train_test_cut:]

    def get_split(self, split=0.8, random_state=2, shuffle_data=True):
        """
        :param split: float - % to be training data
        :param random_state: int - init_state for random to keep random stale
        :param shuffle_data: if to shuffle
        :return: X_t, X_v, y_t, y_v where X = training and Y = validation.
        t = training data & v = class
        """
        print("Creating validation data by splitting (%s)" % split)
        train_data = self.train_data
        X_train = self.X_train

        X_t, X_v, y_t, y_v = train_test_split(
            X_train, train_data.target,
            test_size=(1 - split), random_state=random_state,
            shuffle=shuffle_data, stratify=train_data.target)

        return X_t, X_v, y_t, y_v

    def get_kfold(self, k=5, shuffle_data=True, random_state=2):
        """
        :param k: int - Number of folds.
        :param shuffle_data: boolean - If we should shuffle
        :param random_state: int - init_state for random to keep random stale
        :return: a generator that yields the folds.
        """
        print("Creating validation data by kfold (%s)" % k)
        kfold = StratifiedKFold(n_splits=k, shuffle=shuffle_data, random_state=random_state)
        train_data = self.train_data
        X_train = self.X_train
        folded_data = kfold.split(X_train, train_data.target)

        for i in range(k):
            fold = next(folded_data)
            X_t = X_train.iloc[fold[0]]
            X_v = train_data.iloc[fold[0]]
            y_t = X_train.iloc[fold[1]]
            y_v = train_data.iloc[fold[1]]

            yield X_t, X_v, y_t, y_v

    def get_test(self):
        if isinstance(self.test_data, pd.DataFrame):
            return self.train_data, self.X_train, self.test_data, self.X_test
        raise Exception("No test data provided!")

    def get_all_text(self):
        return self.df_all['question_text']




class PreProcessor(object):
    def __init__(self, text):
        self.text = text
        self.puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$',
                       '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',
                       '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', '<',
                       '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â',
                       '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢',
                       '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
                       '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’',
                       '▀', '¨', '▄', '♫', '☆', 'é', '¯',
                       '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
                       '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³',
                       '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']
        # TODO this varies depending on what task!
        self.mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
                             'counselling': 'counseling',
                             'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                             'organisation': 'organization',
                             'wwii': 'world war 2',
                             'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary',
                             'Whta': 'What',
                             'narcisist': 'narcissist',
                             'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much',
                             'howmany': 'how many', 'whydo': 'why do',
                             'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does',
                             'mastrubation': 'masturbation',
                             'mastrubate': 'masturbate',
                             "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum',
                             'narcissit': 'narcissist',
                             'bigdata': 'big data',
                             '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend',
                             'airhostess': 'air hostess', "whst": 'what',
                             'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                             'demonitization': 'demonetization',
                             'demonetisation': 'demonetization'}
        self.mispellings_re = re.compile('(%s)' % '|'.join(self.mispell_dict.keys()))

    def get_text(self):
        return self.text

    # TODO fix misspellings
    def replace_typical_misspell(self):
        def replace(match):
            return self.mispell_dict[match.group(0)]

        self.text = self.mispellings_re.sub(replace, self.text)

        return self

    def spacy_tokenize_words(self):
        raise NotImplementedError

    def normalize_unicode(self):
        self.text = unicodedata.normalize('NFKD', self.text)
        return self

    def remove_newline(self):
        """
        remove \n and  \t
        """
        self.text = ' '.join(self.text.split())
        return self

    def decontracted(self):
        # specific
        text = re.sub(r"(W|w)on(\'|\’)t", "will not", self.text)
        text = re.sub(r"(C|c)an(\'|\’)t", "can not", text)
        text = re.sub(r"(Y|y)(\'|\’)all", "you all", text)
        text = re.sub(r"(Y|y)a(\'|\’)ll", "you all", text)

        # general
        text = re.sub(r"(I|i)(\'|\’)m", "i am", text)
        text = re.sub(r"(A|a)in(\'|\’)t", "aint", text)
        text = re.sub(r"n(\'|\’)t", " not", text)
        text = re.sub(r"(\'|\’)re", " are", text)
        text = re.sub(r"(\'|\’)s", " is", text)
        text = re.sub(r"(\'|\’)d", " would", text)
        text = re.sub(r"(\'|\’)ll", " will", text)
        text = re.sub(r"(\'|\’)t", " not", text)
        self.text = re.sub(r"(\'|\’)ve", " have", text)

        return self

    def space_punctuation(self):
        for punct in self.puncts:
            if punct in self.text:
                self.text = self.text.replace(punct, f' {punct} ')

                # We could also remove all non p\{L}...

        return self

    def remove_punctuation(self):
        import string
        re_tok = re.compile(f'([{string.punctuation}])')
        self.text = re_tok.sub(' ', self.text)

        return self

    def clean_numbers(self):
        text = self.text
        if bool(re.search(r'\d', text)):
            text = re.sub('[0-9]{5,}', '#####', text)
            text = re.sub('[0-9]{4}', '####', text)
            text = re.sub('[0-9]{3}', '###', text)
            text = re.sub('[0-9]{2}', '##', text)
        self.text = text
        return self

    def clean_and_get_text(self):
        self.clean_numbers()             .space_punctuation()             .decontracted()             .normalize_unicode()             .remove_newline()             .replace_typical_misspell()

        return self.text




class BaseClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C
        self._best_C, self._best_score, self._clf = None, None, None

    def predict(self, X):
        # Verify that model has been fit
        check_is_fitted(self, ['_clf'])
        return self._clf.predict(X)

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)

        self._clf = LinearSVC(C=self.C).fit(X, y)
        return self

    def train(self, X_train, y_train, X_val, y_val, Cs=None):
        """
        trainer to score auc over a grid of Cs
        Parameters
        ----------
        X_train, y_train, X_val, y_val: features and targets
        Cs: list of floats | int
        Return
        ------
        self
        """
        # init grid
        origin_C = self.C
        if Cs is None:
            Cs = [0.01, 0.1, 0.5, 1, 2, 10]
        # score
        scores = {}
        f1 = {}
        for C in Cs:
            # fit
            self.C = C
            model = self.fit(X_train, y_train)
            # predict
            y_pred = model.predict(X_val)
            scores[C] = metrics.roc_auc_score(y_val, y_pred)
            f1[C] = metrics.f1_score(y_val, y_pred)
            print("Val AUC Score: {:.4f}, F1: {:.4f} with C = {}".format(scores[C], f1[C], C))  # noqa

        # get max
        self._best_C, self._best_score = max(f1.items(), key=operator.itemgetter(1))  # noqa
        # reset
        self.C = origin_C
        return self

    @property
    def best_param(self):
        check_is_fitted(self, ['_clf'])
        return self._best_C

    @property
    def best_score(self):
        check_is_fitted(self, ['_clf'])
        return self._best_score

def transform(df_text):
    df_text.progress_apply(clean_text)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                 strip_accents='ascii')
    return vectorizer.fit_transform(list(df_text))


def get_model():
    return BaseClassifier(2)


def clean_text(text):
    return PreProcessor(text).clean_and_get_text()




def train_and_eval(X_train, y_train, X_val, y_val):
    """
    Parameters
    ----------
    X_train, y_train, X_val, y_val: features and targets
    
    Return
    ------
    training logs
    """
    model = get_model()
    print('Training model...')
    model = model.train(X_train, y_train, X_val, y_val)
    best_param = model.best_param
    best_score = model.best_score
    print("Best param: {:.4f} with best score: {}".format(best_param, best_score))
    return pd.DataFrame({'best_param': [best_param], 'best_score': [best_score]})

t0 = time.time()
class fakemodule(object):
    @staticmethod
    def transform(a):
        return transform(a)
dr = DataReader('%s/train.csv' % datapath, fakemodule, os.path.join(datapath, 'test.csv'))

with timer("Load and Preprocess"):
    X_t, X_v, y_t, y_v = dr.get_split(TEST_SIZE)

with timer('Training and Tuning'):
    #df_score = train_and_eval(X_t, y_t, X_v, y_v)
    filepath = os.path.join(datapath, 'trainer_baseline.csv')
    # df_score.to_csv(filepath)
    print('Save CV score file to {}'.format(filepath))

print('Entire program is done and it took {:.2f}s'.format(time.time() - t0))




def create_submission(X_train, y_train, X_test, df_test):
    """
    train model with entire training data, predict test data,
    and create submission file

    Parameters
    ----------
    X_train, y_train, X_test: features and targets
    df_test: dataframe, test data
    module: a python module

    Return
    ------
    df_summission
    """
    model = get_model()
    print('Training model...')
    model = model.fit(X_train, y_train)
    # predict
    print('Predicting test...')
    #y_pred = np.squeeze(model.predict_proba(X_test) > thres).astype('int')
    y_pred = model.predict(X_test)
    # create submission file
    return pd.DataFrame({'qid': df_test.qid, 'prediction': y_pred})

t0 = time.time()

with timer("Load and Preprocess"):
    # Only init if didn't run training.
    # dr = DataReader(os.path.join(datapath, 'quora', 'train.csv'), fakemodule, os.path.join(datapath, 'quora', 'test.csv'))
    df_train, X_train, df_test, X_test = dr.get_test()
# 3. create submission file
with timer('Trainning and Creating Submission'):
    filepath = os.path.join('submission.csv')
    df_submission = create_submission(
        X_train, df_train.target,
        X_test, df_test)
    df_submission.to_csv(filepath, index=False)
    print('Save submission file to {}'.format(filepath))

print('Entire program is done and it took {:.2f}s'.format(time.time() - t0))

