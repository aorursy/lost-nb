#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import regex as re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy
import unicodedata
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import (Input, Embedding, SpatialDropout1D, Bidirectional,
                          LSTM, GRU, GlobalMaxPool1D, Dense)
from keras.models import Model
import tensorflow as tf
from keras import backend as K
from keras import initializers, regularizers, constraints, activations
from keras.layers import Activation, Wrapper, InputSpec
from keras.engine.topology import Layer
from keras.callbacks import (EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau)
from keras.utils.conv_utils import conv_output_length
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
THRESHOLD = 0.35
# Any results you write to the current directory are saved as output.


# In[2]:


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


# In[3]:


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


# In[4]:


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


# In[5]:


class NeuralNetworkClassifier:
    """
    Neural Network classifier - sklearn like
    """

    def __init__(self, model, batch_size=512, epochs=10, val_score='val_loss',
                 reduce_lr=True, balancing_class_weight=False, filepath=None):
        """
        Parameter
        ---------
        model: Keras model

        batch_size: int or None, number of samples per gradient update

        epochs: int, number of epochs to train the model

        val_score: str, score to monitor. ['accuracy', 'precision_score',
            'recall_score', 'f1_score', 'roc_auc_score']

        reduce_lr: bool, if True, add a Keras callback function that
            reduce learning rate when a metric has stopped improving

        balancing_class_weight: bool, if True, uses the values of y to
            automatically adjust weights inversely proportional to
            class frequencies in the input data as
            n_samples / (n_classes * np.bincount(y))

        filepath: str, data directory that stores model pickle
        """
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_score = val_score
        self.reduce_lr = reduce_lr
        self.balancing_class_weight = balancing_class_weight
        self.filepath = filepath
        # compile model
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', precision_score, recall_score,
                     f1_score, roc_auc_score])

    def _get_class_weight(self, y):
        # get class_weight
        if self.balancing_class_weight:
            from sklearn import utils
            return utils.class_weight.compute_class_weight(
                'balanced', np.unique(y), y)
        else:
            return None

    def _get_callbacks(self):
        callbacks = []
        # get callbacks
        callbacks.append(
            EarlyStopping(
                monitor=self.val_score,
                patience=2,
                verbose=1
            )
        )
        if self.filepath:
            callbacks.append(
                ModelCheckpoint(
                    filepath=self.filepath,
                    monitor=self.val_score,
                    save_best_only=True,
                    save_weights_only=True
                )
            )
        if self.reduce_lr:
            callbacks.append(
                ReduceLROnPlateau(
                    monitor=self.val_score,
                    factor=0.6,
                    patience=1,
                    min_lr=0.0001,
                    verbose=2
                )
            )
        return callbacks

    def predict(self, X):
        return (self.predict_proba(X) > THRESHOLD).astype(int)

    def predict_proba(self, X):
        return self.model.predict([X], batch_size=1024, verbose=1)

    def train(self, X_train, y_train, X_val, y_val, verbose=1):
        """
        train neural network and monitor the best iteration with validation

        Parameters
        ----------
        X_train, y_train, X_val, y_val: features and targets

        verbose: int, 0 = silent, 1 = progress bar, 2 = one line per epoch

        Return
        ------
        self
        """
        # get callbacks
        callbacks = self._get_callbacks()
        # get class_weight
        class_weight = self._get_class_weight(y_train)
        # train model
        self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=(X_val, y_val),
            shuffle=True,
            class_weight=class_weight)
        return self

    def fit(self, X, y, best_iteration=6, verbose=0):
        """
        fit lightgbm with best iteration, which is the best model

        Parameters
        ----------
        X, y: features and targets

        best_iteration: int, optional (default=100),
            number of boosting iterations

        verbose: int, 0 = silent, 1 = progress bar, 2 = one line per epoch

        Return
        ------
        self
        """
        # get class_weight
        class_weight = self._get_class_weight(y)
        self.model.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=best_iteration,
            verbose=verbose,
            shuffle=True,
            class_weight=class_weight)
        # save model
        if self.filepath:
            self.model.save_weights(self.filepath)
            print('saved fitted model to {}'.format(self.filepath))
        return self

    @property
    def best_param(self):
        scores = self.model.history.history[self.val_score]
        if 'loss' in self.val_score:
            func = min
        else:
            func = max
        best_iteration, _ = func(enumerate(scores), key=operator.itemgetter(1))
        return best_iteration + 1

    @property
    def best_score(self):
        scores = self.model.history.history[self.val_score]
        if 'loss' in self.val_score:
            func = min
        else:
            func = max
        _, best_val_f1 = func(enumerate(scores), key=operator.itemgetter(1))
        return best_val_f1


"""
Customized metrics during model training
"""


def recall_score(y_true, y_proba, thres=THRESHOLD):
    """
    Recall metric

    Only computes a batch-wise average of recall

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected
    """
    # get prediction
    y_pred = K.cast(K.greater(y_proba, thres), dtype='float32')
    # calc
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_score(y_true, y_proba, thres=THRESHOLD):
    """
    Precision metric

    Only computes a batch-wise average of precision

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant
    """
    # get prediction
    y_pred = K.cast(K.greater(y_proba, thres), dtype='float32')
    # calc
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_proba, thres=THRESHOLD):
    """
    F1 metric: geometric mean of precision and recall
    """
    precision = precision_score(y_true, y_proba, thres)
    recall = recall_score(y_true, y_proba, thres)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def roc_auc_score(y_true, y_proba):
    """
    ROC AUC metric
    """
    roc_auc = tf.metrics.auc(y_true, y_proba)[1]
    K.get_session().run(tf.local_variables_initializer())
    return roc_auc


"""
Customized Keras layers for deep neural networks
"""


class Attention(Layer):
    """
    Keras Layer that implements an Attention mechanism for temporal data.
    Supports Masking.
    Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
    # Input shape
        3D tensor with shape: (samples, steps, features).
    # Output shape
        2D tensor with shape: (samples, features).
    :param kwargs:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True. # noqa
    The dimensions are inferred based on the output shape of the RNN.
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(Attention())
    """

    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


class Capsule(Layer):
    """
    Keras Layer that implements a Capsule for temporal data.
    Literature publication: https://arxiv.org/abs/1710.09829v1
    Youtube video introduction: https://www.youtube.com/watch?v=pPN8d0E3900
    # Input shape
        4D tensor with shape: (samples, steps, features).
    # Output shape
        3D tensor with shape: (samples, num_capsule, dim_capsule).
    :param kwargs:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True. # noqa
    The dimensions are inferred based on the output shape of the RNN.
    Example:
        model.add(
            LSTM(
                64,
                return_sequences=True,
                recurrent_initializer=orthogonal(gain=1.0, seed=10000)
            )
        )
        model.add(
            Capsule(
                num_capsule=10,
                dim_capsule=10,
                routings=4,
                share_weights=True
            )
        )
    """

    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),  # noqa
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),  # noqa
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))  # noqa
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]  # noqa

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]  # noqa
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]    # noqa
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(tf.keras.backend.batch_dot(c, u_hat_vecs, [2, 2]))  # noqa
            if i < self.routings - 1:
                b = tf.keras.backend.batch_dot(outputs, u_hat_vecs, [2, 3])
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


class DropConnect(Wrapper):
    """
    Keras Wrapper that implements a DropConnect Layer.
    When training with Dropout, a randomly selected subset of activations are
    set to zero within each layer. DropConnect instead sets a randomly
    selected subset of weights within the network to zero.
    Each unit thus receives input from a random subset of units in the
    previous layer.

    Reference: https://cs.nyu.edu/~wanli/dropc/
    Implementation: https://github.com/andry9454/KerasDropconnect
    """

    def __init__(self, layer, prob, **kwargs):
        self.prob = prob
        self.layer = layer
        super(DropConnect, self).__init__(layer, **kwargs)
        if 0. < self.prob < 1.:
            self.uses_learning_phase = True

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(DropConnect, self).build()

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def call(self, x):
        if 0. < self.prob < 1.:
            self.layer.kernel = K.in_train_phase(
                K.dropout(self.layer.kernel, self.prob),
                self.layer.kernel)
            self.layer.bias = K.in_train_phase(
                K.dropout(self.layer.bias, self.prob),
                self.layer.bias)
        return self.layer.call(x, )


def _dropout(x, level, noise_shape=None, seed=None):
    x = K.dropout(x, level, noise_shape, seed)
    x *= (1. - level)  # compensate for the scaling by the dropout
    return x


class QRNN(Layer):
    '''Quasi RNN
    # Arguments
        units: dimension of the internal projections and the final output.
    # References
        - [Quasi-recurrent Neural Networks](http://arxiv.org/abs/1611.01576)
    '''

    def __init__(self, units, window_size=2, stride=1,
                 return_sequences=False, go_backwards=False,
                 stateful=False, unroll=False, activation='tanh',
                 kernel_initializer='uniform', bias_initializer='zero',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 dropout=0, use_bias=True, input_dim=None, input_length=None,
                 **kwargs):
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll

        self.units = units
        self.window_size = window_size
        self.strides = (stride, 1)

        self.use_bias = use_bias
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.recurrent_dropout = 0  # not used, added to maintain compatibility with keras.Bidirectional
        self.dropout = dropout
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(QRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec = InputSpec(shape=(batch_size, None, self.input_dim))
        self.state_spec = InputSpec(shape=(batch_size, self.units))

        self.states = [None]
        if self.stateful:
            self.reset_states()

        kernel_shape = (self.window_size, 1, self.input_dim, self.units * 3)
        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.units * 3,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        length = input_shape[1]
        if length:
            length = conv_output_length(length + self.window_size - 1,
                                        self.window_size, 'valid',
                                        self.strides[0])
        if self.return_sequences:
            return (input_shape[0], length, self.units)
        else:
            return (input_shape[0], self.units)

    def compute_mask(self, inputs, mask):
        if self.return_sequences:
            return mask
        else:
            return None

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, units)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.units])  # (samples, units)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        if not self.input_spec:
            raise RuntimeError('Layer has never been called '
                               'and thus has no states.')

        batch_size = self.input_spec.shape[0]
        if not batch_size:
            raise ValueError('If a QRNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.')

        if self.states[0] is None:
            self.states = [K.zeros((batch_size, self.units))
                           for _ in self.states]
        elif states is None:
            for state in self.states:
                K.set_value(state, np.zeros((batch_size, self.units)))
        else:
            if not isinstance(states, (list, tuple)):
                states = [states]
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                                         'but it received ' + str(len(states)) +
                                 'state values. Input received: ' +
                                 str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                if value.shape != (batch_size, self.units):
                    raise ValueError('State ' + str(index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected shape=' +
                                     str((batch_size, self.units)) +
                                     ', found shape=' + str(value.shape))
                K.set_value(state, value)

    def __call__(self, inputs, initial_state=None, **kwargs):
        # If `initial_state` is specified,
        # and if it a Keras tensor,
        # then add it to the inputs and temporarily
        # modify the input spec to include the state.
        if initial_state is not None:
            if hasattr(initial_state, '_keras_history'):
                # Compute the full input spec, including state
                input_spec = self.input_spec
                state_spec = self.state_spec
                if not isinstance(state_spec, list):
                    state_spec = [state_spec]
                self.input_spec = [input_spec] + state_spec

                # Compute the full inputs, including state
                if not isinstance(initial_state, (list, tuple)):
                    initial_state = [initial_state]
                inputs = [inputs] + list(initial_state)

                # Perform the call
                output = super(QRNN, self).__call__(inputs, **kwargs)

                # Restore original input spec
                self.input_spec = input_spec
                return output
            else:
                kwargs['initial_state'] = initial_state
        return super(QRNN, self).__call__(inputs, **kwargs)

    def call(self, inputs, mask=None, initial_state=None, training=None):
        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            initial_states = inputs[1:]
            inputs = inputs[0]
        elif initial_state is not None:
            pass
        elif self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(inputs)

        if len(initial_states) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_states)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs)
        if self.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')
        constants = self.get_constants(inputs, training=None)
        preprocessed_input = self.preprocess_input(inputs, training=None)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        # Properly set learning phase
        if 0 < self.dropout < 1:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def preprocess_input(self, inputs, training=None):
        if self.window_size > 1:
            inputs = K.temporal_padding(inputs, (self.window_size - 1, 0))
        inputs = K.expand_dims(inputs, 2)  # add a dummy dimension

        output = K.conv2d(inputs, self.kernel, strides=self.strides,
                          padding='valid',
                          data_format='channels_last')
        output = K.squeeze(output, 2)  # remove the dummy dimension
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')

        if self.dropout is not None and 0. < self.dropout < 1.:
            z = output[:, :, :self.units]
            f = output[:, :, self.units:2 * self.units]
            o = output[:, :, 2 * self.units:]
            f = K.in_train_phase(1 - _dropout(1 - f, self.dropout), f, training=training)
            return K.concatenate([z, f, o], -1)
        else:
            return output

    def step(self, inputs, states):
        prev_output = states[0]

        z = inputs[:, :self.units]
        f = inputs[:, self.units:2 * self.units]
        o = inputs[:, 2 * self.units:]

        z = self.activation(z)
        f = f if self.dropout is not None and 0. < self.dropout < 1. else K.sigmoid(f)
        o = K.sigmoid(o)

        output = f * prev_output + (1 - f) * z
        output = o * output

        return output, [output]

    def get_constants(self, inputs, training=None):
        return []

    def get_config(self):
        config = {'units': self.units,
                  'window_size': self.window_size,
                  'stride': self.strides[0],
                  'return_sequences': self.return_sequences,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll,
                  'use_bias': self.use_bias,
                  'dropout': self.dropout,
                  'activation': activations.serialize(self.activation),
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(QRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# In[6]:


def load_word_embedding(filepath):
    """
    given a filepath to embeddings file, return a word to vec
    dictionary, in other words, word_embedding

    E.g. {'word': array([0.1, 0.2, ...])}
    """

    def _get_vec(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    print('Load word embeddings...')
    try:
        word_embedding = dict(_get_vec(*w.split(' ')) for w in open(filepath))
    except UnicodeDecodeError:
        word_embedding = dict(_get_vec(*w.split(' ')) for w in open(
            filepath, encoding="utf8", errors='ignore'))
    # sanity check word vector length
    words_to_del = []
    for word, vec in word_embedding.items():
        if len(vec) != 300:
            words_to_del.append(word)
    for word in words_to_del:
        del word_embedding[word]
    return word_embedding


def create_embedding_weights(word_index, word_embedding, max_features):
    """
    create weights for embeddings layer where row is the word index
    and collumns are the embeddings dense vector

    Parameters
    ----------
    word_index: dict, mapping of word to word index. E.g. {'the': 2}
        you can get word_index by keras.tokenizer.word_index

    word_embedding: dict, mapping of word to word embeddings
        E.g. {'the': array([0.1, 0.2, ...])}
        you can get word_index by above function load_word_embedding and
        embeddings filepath

    max_features: int, number of words that we want to keep

    Return
    ------
    embeddings weights: np.array, with shape (number of words, 300)
    """
    print('Create word embeddings weights...')
    # get entire embeddings matrix
    mat_embedding = np.stack(word_embedding.values())
    # get shape
    a, b = min(max_features, len(word_index)), mat_embedding.shape[1]
    print('Embeddings weights matrix with shape: ({}, {})'.format(a, b))
    # init embeddings weight matrix
    embedding_mean, embedding_std = mat_embedding.mean(), mat_embedding.std()
    embedding_weights = np.random.normal(embedding_mean, embedding_std, (a, b))
    # mapping
    for word, idx in word_index.items():
        if idx >= a:
            continue
        word_vec = word_embedding.get(word, None)
        if word_vec is not None:
            embedding_weights[idx] = word_vec
    return embedding_weights


# In[7]:


import gc
MAX_FEATURES = int(2.5e5)  # total word count = 227,538; clean word count = 186,551   # noqa
MAX_LEN = 80    # mean_len = 12; Q99_len = 40; max_len = 189;
RNN_UNITS = 40
DENSE_UNITS_1 = 32
DENSE_UNITS_2 = 16
EMBED_FILEPATH = os.path.join(datapath, 'embeddings/glove.840B.300d/glove.840B.300d.txt')
EMBED_PICKLEPATH = 'glove.pkl'
MODEL_FILEPATH = 'submission.csv'


# In[8]:


def get_network(embed_filepath):
    input_layer = Input(shape=(MAX_LEN, ), name='input')
    # 1. embeddings layer
    # get embeddings weights
    print('load pre-trained embeddings weights...')
    embed_weights = pd.read_pickle(EMBED_PICKLEPATH)
    input_dim = embed_weights.shape[0]
    output_dim = embed_weights.shape[1]
    x = Embedding(
        input_dim=input_dim,
        output_dim=output_dim,
        weights=[embed_weights],
        trainable=False,
        name='embeddings'
    )(input_layer)
    # clean up
    del embed_weights, input_dim, output_dim
    gc.collect()
    # 2. dropout
    x = SpatialDropout1D(rate=0.15)(x)
    # 3. bidirectional lstm & gru
    x = Bidirectional(
        layer=LSTM(RNN_UNITS, return_sequences=True),
        name='bidirectional_lstm'
    )(x)
    #x = Bidirectional(
    #    layer=GRU(RNN_UNITS, return_sequences=True),
    #    name='bidirectional_gru'
    #)(x)
    # 4. global_max_pooling1d
    x = GlobalMaxPool1D(name='global_max_pooling1d')(x)
    # 5. dense
    x = Dense(units=DENSE_UNITS_1, activation='relu', name='dense_1')(x)
    x = Dense(units=DENSE_UNITS_2, activation='relu', name='dense_2')(x)
    # 6. output (sigmoid)
    output_layer = Dense(units=1, activation='sigmoid', name='output')(x)
    return Model(inputs=input_layer, outputs=output_layer)


def get_model():
    print('build network...')
    model = get_network(embed_filepath=EMBED_FILEPATH)
    print(model.summary())
    return NeuralNetworkClassifier(
        model,
        balancing_class_weight=True,
        filepath=MODEL_FILEPATH)

def tokenize(df_text):
    # preprocess
    def _clean(text):
        return PreProcessor(text).clean_and_get_text()
    df_text = df_text.progress_apply(_clean)
    # tokenizer
    tokenizer = Tokenizer(
        num_words=MAX_FEATURES,
        filters='',
        lower=False,
        split=' ')
    # fit to data
    tokenizer.fit_on_texts(list(df_text))
    # tokenize the texts into sequences
    sequences = tokenizer.texts_to_sequences(df_text)
    return sequences, tokenizer


def transform(df_text):
    seqs, _ = tokenize(df_text)
    # pad the sentences
    X = pad_sequences(seqs, maxlen=MAX_LEN, padding='pre', truncating='post')
    return X


# In[9]:


class fakemodule(object):
    @staticmethod
    def transform(a):
        return transform(a)
dr = DataReader('%s/train.csv' % datapath, fakemodule, os.path.join(datapath, 'test.csv'))


# In[10]:


t0 = time.time()
with timer("Extract Word Index From Train and Test Data"):
    print("Loading data...")
    all_text = dr.get_all_text()
    # get word index
    print('Tokenizing text...')
    _, tokenizer = tokenize(all_text)
    word_index = tokenizer.word_index
# 3. create embeddings weights matrix
with timer("Create Embedding Weights Matrix"):
    # load word embeddings
    print('Loading embeddings file')
    word_embed = load_word_embedding(EMBED_FILEPATH)
    # create embeddings weights matrix
    print('Create embeddings weights...')
    embed_weights = create_embedding_weights(
        word_index,
        word_embed,
        MAX_FEATURES)
    # pickle numpy file
    filepath_to_save = '{}.pkl'.format('glove')
    pd.to_pickle(embed_weights, filepath_to_save)
    print('Save embeddings weights to {}'.format(filepath_to_save))
# record time spent
print('Entire program is done and it took {:.2f}s'.format(time.time() - t0))


# In[11]:


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

with timer("Load and Preprocess"):
    X_t, X_v, y_t, y_v = dr.get_split(TEST_SIZE)

with timer('Training and Tuning'):
    #df_score = train_and_eval(X_t, y_t, X_v, y_v)
    filepath = os.path.join(datapath, 'trainer_baseline.csv')
    # df_score.to_csv(filepath)
    print('Save CV score file to {}'.format(filepath))

print('Entire program is done and it took {:.2f}s'.format(time.time() - t0))


# In[12]:


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
    y_pred = np.squeeze(model.predict_proba(X_test) > 0.35).astype('int')
    #y_pred = model.predict(X_test)
    # create submission file
    return pd.DataFrame({'qid': df_test.qid, 'prediction': y_pred})

t0 = time.time()

with timer("Load and Preprocess"):
    # Only init if didn't run training.
    # dr = DataReader(os.path.join(datapath, 'quora', 'train.csv'), fakemodule, os.path.join(datapath, 'quora', 'test.csv'))
    df_train, X_train, df_test, X_test = dr.get_test()
# 3. create submission file
with timer('Trainning and Creating Submission'):
    filepath = 'submission.csv'
    df_submission = create_submission(
        X_train, df_train.target,
        X_test, df_test)
    df_submission.to_csv(filepath, index=False)
    print('Save submission file to {}'.format(filepath))

print('Entire program is done and it took {:.2f}s'.format(time.time() - t0))

