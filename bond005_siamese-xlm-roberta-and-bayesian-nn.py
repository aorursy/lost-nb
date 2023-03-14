#!/usr/bin/env python
# coding: utf-8



import codecs
import copy
import csv
import gc
import os
import pickle
import random
import tempfile
import time
from typing import Dict, List, Sequence, Set, Tuple, Union




import matplotlib.pyplot as plt
from matplotlib import figure
import numpy as np
from scipy.stats import hmean
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import ops, tensor_util
from tensorflow.python.keras.utils import losses_utils, tf_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import util as tf_losses_util
import tensorflow_addons as tfa
from transformers import AutoTokenizer, XLMRobertaTokenizer
from transformers import TFXLMRobertaModel, XLMRobertaConfig




class LossFunctionWrapper(tf.keras.losses.Loss):
    def __init__(self,
                 fn,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name=None,
                 **kwargs):
        super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true):
            y_pred, y_true = tf_losses_util.squeeze_or_expand_dimensions(y_pred, y_true)
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = {}
        for k, v in six.iteritems(self._fn_kwargs):
            config[k] = tf.keras.backend.eval(v) if tf_utils.is_tensor_or_variable(v)                 else v
        base_config = super(LossFunctionWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




def distance_based_log_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    margin = 1.0
    p = (1.0 + tf.math.exp(-margin)) / (1.0 + tf.math.exp(y_pred - margin))
    return tf.keras.backend.binary_crossentropy(target=y_true, output=p)




class DBLLogLoss(LossFunctionWrapper):
    def __init__(self, reduction=losses_utils.ReductionV2.AUTO,
                 name='distance_based_log_loss'):
        super(DBLLogLoss, self).__init__(distance_based_log_loss, name=name,
                                         reduction=reduction)




class MaskCalculator(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MaskCalculator, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaskCalculator, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.keras.backend.permute_dimensions(
            x=tf.keras.backend.repeat(
                x=tf.keras.backend.cast(
                    x=tf.keras.backend.greater(
                        x=inputs,
                        y=0
                    ),
                    dtype='float32'
                ),
                n=self.output_dim
            ),
            pattern=(0, 2, 1)
        )

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 1
        shape = list(input_shape)
        shape.append(self.output_dim)
        return tuple(shape)




class WeightPosteriorCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_every: int):
        self.bayesian_layers = []
        self.layer_names = []
        self.logged_epochs = dict()
        self.log_every = log_every
        super(WeightPosteriorCallback, self).__init__()
    
    def on_train_begin(self, logs=None):
        self.bayesian_layers = list(filter(
            lambda layer: (layer.name.lower().startswith('bayesianhiddenlayer') or \
                           layer.name.lower().startswith('bayesianoutputlayer')),
            self.model.layers
        ))
        self.layer_names = []
        max_layer_idx = 0
        for layer in self.bayesian_layers:
            if layer.name.lower().startswith('bayesianhiddenlayer'):
                new_layer_name = 'Layer'
                start_pos = len('bayesianhiddenlayer')
                find_idx = layer.name[start_pos:].find('_')
                assert find_idx > 0
                layer_idx = int(layer.name[start_pos:(start_pos + find_idx)])
                new_layer_name += '{0}'.format(layer_idx)
                if layer_idx > max_layer_idx:
                    max_layer_idx = layer_idx
            else:
                new_layer_name = 'Layer{0}'.format(max_layer_idx + 1)
            self.layer_names.append(new_layer_name)
        self.logged_epochs = dict()
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch == 0) or (((epoch + 1) % self.log_every) == 0):
            qm_vals = [layer.kernel_posterior.mean() for layer in self.bayesian_layers]
            qs_vals = [layer.kernel_posterior.stddev() for layer in self.bayesian_layers]
            self.logged_epochs[epoch + 1] = (qm_vals, qs_vals)




def plot_weight_posteriors(layer_names: List[str],
                           logged_epochs: Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]]):
    epoch_indices = sorted(list(logged_epochs.keys()))
    assert len(epoch_indices) > 0
    n_layers = len(layer_names)
    assert n_layers > 0
    for epoch_idx in epoch_indices:
        assert len(logged_epochs[epoch_idx][0]) == len(logged_epochs[epoch_idx][1])
        assert len(logged_epochs[epoch_idx][0]) == n_layers
    fig = plt.figure(figsize=(6, 3 * len(epoch_indices)))
    counter = 1
    for epoch_idx in epoch_indices:
        qm_vals = logged_epochs[epoch_idx][0]
        qs_vals = logged_epochs[epoch_idx][1]

        ax = fig.add_subplot(len(epoch_indices), 2, counter)
        for n, qm in zip(layer_names, qm_vals):
            sns.distplot(tf.reshape(qm, shape=[-1]), ax=ax, label=n)
        ax.set_title('Epoch {0}, weight means'.format(epoch_idx))
        ax.set_xlim([-1.5, 1.5])
        ax.legend(loc='best')
        counter += 1

        ax = fig.add_subplot(len(epoch_indices), 2, counter)
        for n, qs in zip(layer_names, qs_vals):
            sns.distplot(tf.reshape(qs, shape=[-1]), ax=ax)
        ax.set_title('Epoch {0}, weight stddevs'.format(epoch_idx))
        ax.set_xlim([0, 1.])
        counter += 1

    fig.tight_layout()
    plt.show()




def regular_encode(texts: List[str], tokenizer: XLMRobertaTokenizer,
                   maxlen: int) -> Tuple[np.ndarray, np.ndarray]:
    enc_di = tokenizer.batch_encode_plus(
        texts,
        return_attention_masks=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    return np.array(enc_di['input_ids']), np.array(enc_di['attention_mask'])




def load_train_set(file_name: str, text_field: str, sentiment_fields: List[str],
                   lang_field: str) -> Dict[str, List[Tuple[str, int]]]:
    assert len(sentiment_fields) > 0, 'List of sentiment fields is empty!'
    header = []
    line_idx = 1
    data_by_lang = dict()
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        data_reader = csv.reader(fp, quotechar='"', delimiter=',')
        for row in data_reader:
            if len(row) > 0:
                err_msg = 'File "{0}": line {1} is wrong!'.format(file_name, line_idx)
                if len(header) == 0:
                    header = copy.copy(row)
                    err_msg2 = err_msg + ' Field "{0}" is not found!'.format(text_field)
                    assert text_field in header, err_msg2
                    for cur_field in sentiment_fields:
                        err_msg2 = err_msg + ' Field "{0}" is not found!'.format(
                            cur_field)
                        assert cur_field in header, err_msg2
                    text_field_index = header.index(text_field)
                    try:
                        lang_field_index = header.index(lang_field)
                    except:
                        lang_field_index = -1
                    indices_of_sentiment_fields = []
                    for cur_field in sentiment_fields:
                        indices_of_sentiment_fields.append(header.index(cur_field))
                else:
                    if len(row) == len(header):
                        text = row[text_field_index].strip()
                        assert len(text) > 0, err_msg + ' Text is empty!'
                        if lang_field_index >= 0:
                            cur_lang = row[lang_field_index].strip()
                            assert len(cur_lang) > 0, err_msg + ' Language is empty!'
                        else:
                            cur_lang = 'en'
                        max_proba = 0.0
                        for cur_field_idx in indices_of_sentiment_fields:
                            try:
                                cur_proba = float(row[cur_field_idx])
                            except:
                                cur_proba = -1.0
                            err_msg2 = err_msg + ' Value {0} is wrong!'.format(
                                row[cur_field_idx]
                            )
                            assert (cur_proba >= 0.0) and (cur_proba <= 1.0), err_msg2
                            if cur_proba > max_proba:
                                max_proba = cur_proba
                        new_label = 1 if max_proba >= 0.5 else 0
                        if cur_lang not in data_by_lang:
                            data_by_lang[cur_lang] = []
                        data_by_lang[cur_lang].append((text, new_label))
            if line_idx % 10000 == 0:
                print('{0} lines of the "{1}" have been processed...'.format(line_idx,
                                                                             file_name))
            line_idx += 1
    if line_idx > 0:
        if (line_idx - 1) % 10000 != 0:
            print('{0} lines of the "{1}" have been processed...'.format(line_idx - 1,
                                                                         file_name))
    return data_by_lang




def load_test_set(file_name: str, id_field: str, text_field: str,
                  lang_field: str) -> Dict[str, List[Tuple[str, int]]]:
    header = []
    line_idx = 1
    data_by_lang = dict()
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        data_reader = csv.reader(fp, quotechar='"', delimiter=',')
        for row in data_reader:
            if len(row) > 0:
                err_msg = 'File "{0}": line {1} is wrong!'.format(file_name, line_idx)
                if len(header) == 0:
                    header = copy.copy(row)
                    err_msg2 = err_msg + ' Field "{0}" is not found!'.format(text_field)
                    assert text_field in header, err_msg2
                    err_msg2 = err_msg + ' Field "{0}" is not found!'.format(id_field)
                    assert id_field in header, err_msg2
                    err_msg2 = err_msg + ' Field "{0}" is not found!'.format(lang_field)
                    assert lang_field in header, err_msg2
                    id_field_index = header.index(id_field)
                    text_field_index = header.index(text_field)
                    lang_field_index = header.index(lang_field)
                else:
                    if len(row) == len(header):
                        try:
                            id_value = int(row[id_field_index])
                        except:
                            id_value = -1
                        err_msg2 = err_msg + ' {0} is wrong ID!'.format(
                            row[id_field_index])
                        assert id_value >= 0, err_msg2
                        text = row[text_field_index].strip()
                        assert len(text) > 0, err_msg + ' Text is empty!'
                        if lang_field_index >= 0:
                            cur_lang = row[lang_field_index].strip()
                            assert len(cur_lang) > 0, err_msg + ' Language is empty!'
                        else:
                            cur_lang = 'en'
                        if cur_lang not in data_by_lang:
                            data_by_lang[cur_lang] = []
                        data_by_lang[cur_lang].append((text, id_value))
            if line_idx % 10000 == 0:
                print('{0} lines of the "{1}" have been processed...'.format(line_idx,
                                                                             file_name))
            line_idx += 1
    if line_idx > 0:
        if (line_idx - 1) % 10000 != 0:
            print('{0} lines of the "{1}" have been processed...'.format(line_idx - 1,
                                                                         file_name))
    return data_by_lang




def build_siamese_dataset(texts: Dict[str, List[Tuple[str, int]]],
                          dataset_size: int, tokenizer: XLMRobertaTokenizer,
                          maxlen: int, batch_size: int,
                          shuffle: bool) -> Tuple[tf.data.Dataset, int]:
    language_pairs = set()
    for language in texts.keys():
        for other_language in texts:
            if other_language == language:
                language_pairs.add((language, other_language))
            else:
                new_pair = (language, other_language)
                new_pair_2 = (other_language, language)
                if (new_pair not in language_pairs) and (new_pair_2 not in language_pairs):
                    language_pairs.add(new_pair)
    language_pairs = sorted(list(language_pairs))
    print('Possible language pairs are: {0}.'.format(language_pairs))
    err_msg = '{0} is too small size of the data set!'.format(dataset_size)
    assert dataset_size >= (len(language_pairs) * 10), err_msg
    n_samples_for_lang_pair = int(np.ceil(dataset_size / float(len(language_pairs))))
    text_pairs_and_labels = []
    for left_lang, right_lang in language_pairs:
        print('{0}-{1}:'.format(left_lang, right_lang))
        left_positive_indices = list(filter(
            lambda idx: texts[left_lang][idx][1] > 0, range(len(texts[left_lang]))
        ))
        left_negative_indices = list(filter(
            lambda idx: texts[left_lang][idx][1] == 0, range(len(texts[left_lang]))
        ))
        right_positive_indices = list(filter(
            lambda idx: texts[right_lang][idx][1] > 0, range(len(texts[right_lang]))
        ))
        right_negative_indices = list(filter(
            lambda idx: texts[right_lang][idx][1] == 0, range(len(texts[right_lang]))
        ))
        used_pairs = set()
        number_of_samples = 0
        for _ in range(n_samples_for_lang_pair // 4):
            left_idx = random.choice(left_positive_indices)
            right_idx = random.choice(right_positive_indices)
            counter = 0
            while ((right_idx == left_idx) or ((left_idx, right_idx) in used_pairs) or
                   ((right_idx, left_idx) in used_pairs)) and (counter < 100):
                right_idx = random.choice(right_positive_indices)
                counter += 1
            if counter < 100:
                used_pairs.add((left_idx, right_idx))
                used_pairs.add((right_idx, left_idx))
                text_pairs_and_labels.append(
                    (
                        texts[left_lang][left_idx][0],
                        texts[right_lang][right_idx][0],
                        1
                    )
                )
                number_of_samples += 1
        print('  number of "1-1" pairs is {0};'.format(number_of_samples))
        number_of_samples = 0
        for _ in range(n_samples_for_lang_pair // 4, (2 * n_samples_for_lang_pair) // 4):
            left_idx = random.choice(left_negative_indices)
            right_idx = random.choice(right_negative_indices)
            counter = 0
            while ((right_idx == left_idx) or ((left_idx, right_idx) in used_pairs) or
                   ((right_idx, left_idx) in used_pairs)) and (counter < 100):
                right_idx = random.choice(right_negative_indices)
                counter += 1
            if counter < 100:
                used_pairs.add((left_idx, right_idx))
                used_pairs.add((right_idx, left_idx))
                text_pairs_and_labels.append(
                    (
                        texts[left_lang][left_idx][0],
                        texts[right_lang][right_idx][0],
                        1
                    )
                )
                number_of_samples += 1
        print('  number of "0-0" pairs is {0};'.format(number_of_samples))
        number_of_samples = 0
        for _ in range((2 * n_samples_for_lang_pair) // 4, n_samples_for_lang_pair):
            left_idx = random.choice(left_negative_indices)
            right_idx = random.choice(right_positive_indices)
            counter = 0
            while ((right_idx == left_idx) or ((left_idx, right_idx) in used_pairs) or
                   ((right_idx, left_idx) in used_pairs)) and (counter < 100):
                right_idx = random.choice(right_positive_indices)
                counter += 1
            if counter < 100:
                used_pairs.add((left_idx, right_idx))
                used_pairs.add((right_idx, left_idx))
                if random.random() >= 0.5:
                    text_pairs_and_labels.append(
                        (
                            texts[left_lang][left_idx][0],
                            texts[right_lang][right_idx][0],
                            0
                        )
                    )
                else:
                    text_pairs_and_labels.append(
                        (
                            texts[right_lang][right_idx][0],
                            texts[left_lang][left_idx][0],
                            0
                        )
                    )
                number_of_samples += 1
        print('  number of "0-1" or "1-0" pairs is {0}.'.format(number_of_samples))
    random.shuffle(text_pairs_and_labels)
    n_steps = len(text_pairs_and_labels) // batch_size
    print('Samples number of the data set is {0}.'.format(len(text_pairs_and_labels)))
    print('Samples number per each language pair is {0}.'.format(n_samples_for_lang_pair))
    tokens_of_left_texts, mask_of_left_texts = regular_encode(
        texts=[cur[0] for cur in text_pairs_and_labels],
        tokenizer=tokenizer, maxlen=maxlen
    )
    tokens_of_right_texts, mask_of_right_texts = regular_encode(
        texts=[cur[1] for cur in text_pairs_and_labels],
        tokenizer=tokenizer, maxlen=maxlen
    )
    siamese_labels = np.array([cur[2] for cur in text_pairs_and_labels], dtype=np.int32)
    print('Number of positive siamese samples is {0} from {1}.'.format(
        int(sum(siamese_labels)), siamese_labels.shape[0]))
    if shuffle:
        err_msg = '{0} is too small number of samples for the data set!'.format(
            len(text_pairs_and_labels))
        assert n_steps >= 50, err_msg
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    tokens_of_left_texts, mask_of_left_texts,
                    tokens_of_right_texts, mask_of_right_texts
                ),
                siamese_labels
            )
        ).repeat().batch(batch_size)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    tokens_of_left_texts, mask_of_left_texts,
                    tokens_of_right_texts, mask_of_right_texts
                ),
                siamese_labels
            )
        ).batch(batch_size)
    del text_pairs_and_labels
    return dataset, n_steps




def build_feature_extractor(transformer_name: str, hidden_state_size: int,
                            max_len: int) -> tf.keras.Model:
    word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32,
                                     name="base_word_ids")
    attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32,
                                           name="base_attention_mask")
    transformer_layer = TFXLMRobertaModel.from_pretrained(
        pretrained_model_name_or_path=transformer_name,
        name='Transformer'
    )
    sequence_output = transformer_layer([word_ids, attention_mask])[0]
    output_mask = MaskCalculator(
        output_dim=hidden_state_size, trainable=False,
        name='OutMaskCalculator'
    )(attention_mask)
    masked_sequence_output = tf.keras.layers.Multiply(
        name='OutMaskMultiplicator'
    )([output_mask, sequence_output])
    masked_sequence_output = tf.keras.layers.Masking(
        name='OutMasking'
    )(masked_sequence_output)
    pooled_output = tf.keras.layers.GlobalAvgPool1D(name='AvePool')(masked_sequence_output)
    text_embedding = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1),
        name='Emdedding'
    )(pooled_output)
    fe_model = tf.keras.Model(
        inputs=[word_ids, attention_mask],
        outputs=text_embedding,
        name='FeatureExtractor'
    )
    fe_model.build(input_shape=[(None, max_len), (None, max_len)])
    return fe_model




def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.keras.backend.sum(tf.keras.backend.square(x - y),
                                      axis=1, keepdims=True)
    return tf.keras.backend.sqrt(
        tf.keras.backend.maximum(sum_square, tf.keras.backend.epsilon())
    )




def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)




def build_siamese_nn(transformer_name: str, hidden_state_size: int, max_len: int,
                     lr: float) -> Tuple[tf.keras.Model, tf.keras.Model]:
    left_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32,
                                          name="left_word_ids")
    left_attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32,
                                                name="left_attention_mask")
    right_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32,
                                           name="right_word_ids")
    right_attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32,
                                                 name="right_attention_mask")
    fe_ = build_feature_extractor(transformer_name, hidden_state_size, max_len)
    left_text_embedding = fe_([left_word_ids, left_attention_mask])
    right_text_embedding = fe_([right_word_ids, right_attention_mask])
    distance_layer = tf.keras.layers.Lambda(
        function=euclidean_distance,
        output_shape=eucl_dist_output_shape,
        name='L2DistLayer'
    )([left_text_embedding, right_text_embedding])
    nn = tf.keras.Model(
        inputs=[left_word_ids, left_attention_mask, right_word_ids, right_attention_mask],
        outputs=distance_layer,
        name='SiameseXLMR'
    )
    nn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=DBLLogLoss()
    )
    fe_.summary()
    nn.summary()
    return nn, fe_




def build_classifier(language: str, feature_vector_size: int, n_train_samples: int,
                     hidden_layer_size: int, n_hidden_layers: int,
                     lr: float, C: float, verbose: bool) -> tf.keras.Model:
    random_seed = 42
    sentence_features = tf.keras.layers.Input(
        shape=(feature_vector_size,), dtype=tf.float32,
        name="SentenceFeatures_{0}".format(language)
    )
    n_samples = tf.cast(float(n_train_samples), dtype=tf.float32)
    kl_weight = tf.cast(1.0 / C, dtype=tf.float32)
    kl_divergence_function = (
        lambda q, p, _: (tfp.distributions.kl_divergence(q, p) * kl_weight / n_samples)
    )
    hidden_layer = tfp.layers.DenseFlipout(
        units=hidden_layer_size,
        kernel_divergence_fn=kl_divergence_function,
        bias_divergence_fn=kl_divergence_function,
        activation=None, seed=random_seed,
        name='BayesianHiddenLayer1_{0}'.format(language)
    )(sentence_features)
    hidden_layer = tf.keras.layers.BatchNormalization(
        name='BatchNorm1_{0}'.format(language)
    )(hidden_layer)
    hidden_layer = tf.keras.layers.ELU(
        name='Activation1_{0}'.format(language)
    )(hidden_layer)
    for layer_idx in range(1, n_hidden_layers):
        hidden_layer = tfp.layers.DenseFlipout(
            units=hidden_layer_size,
            kernel_divergence_fn=kl_divergence_function,
            bias_divergence_fn=kl_divergence_function,
            activation=None, seed=random_seed,
            name='BayesianHiddenLayer{0}_{1}'.format(layer_idx + 1, language)
        )(hidden_layer)
        hidden_layer = tf.keras.layers.BatchNormalization(
            name='BatchNorm{0}_{1}'.format(layer_idx + 1, language)
        )(hidden_layer)
        hidden_layer = tf.keras.layers.ELU(
            name='Activation{0}_{1}'.format(layer_idx + 1, language)
        )(hidden_layer)
    cls_layer = tfp.layers.DenseFlipout(
        units=1,
        kernel_divergence_fn=kl_divergence_function,
        bias_divergence_fn=kl_divergence_function,
        activation='sigmoid',
        name='BayesianOutputLayer_{0}'.format(language),
        seed=random_seed
    )(hidden_layer)
    cls_model = tf.keras.Model(
        inputs=sentence_features,
        outputs=cls_layer,
        name='BayesianNN_{0}'.format(language)
    )
    radam = tfa.optimizers.RectifiedAdam(learning_rate=lr)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    cls_model.compile(
        optimizer=ranger, loss='binary_crossentropy',
        experimental_run_tf_function=False
    )
    if verbose:
        cls_model.summary()
    return cls_model




def build_datasets_for_classifier(
        data_for_training: Tuple[np.ndarray, np.ndarray],
        data_split: Dict[str, Tuple[np.ndarray, np.ndarray]],
        language_for_testing: str
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    indices_for_training, indices_for_testing = data_split[language_for_testing]
    X_train = data_for_training[0][indices_for_training]
    y_train = data_for_training[1][indices_for_training]
    X_test = data_for_training[0][indices_for_testing]
    y_test = data_for_training[1][indices_for_testing]
    del indices_for_training, indices_for_testing
    data_for_training = (X_train, y_train)
    data_for_testing = (X_test, y_test)
    return data_for_training, data_for_testing




def train_classifier(trainset: tf.data.Dataset, n_steps: int, bayesian_classifier: tf.keras.Model,
                     max_epochs: int, verbose: bool, tmp_file_name: str):
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=3, verbose=False,
            mode='min', min_delta=0.05, cooldown=5, min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            monitor='loss', mode="min", save_weights_only=True, save_best_only=True,
            filepath=tmp_file_name
        )
    ]
    if verbose:
        weight_posterior_callback = WeightPosteriorCallback(
            log_every = max(2, max_epochs // 10)
        )
        callbacks.append(weight_posterior_callback)
    else:
        weight_posterior_callback = None
    if verbose:
        print('n_epochs = {0}, steps_per_epoch = {1}'.format(max_epochs, n_steps))
    history = bayesian_classifier.fit(
        trainset,
        steps_per_epoch=n_steps,
        epochs=max_epochs, callbacks=callbacks,
        verbose=verbose
    )
    if verbose:
        show_training_process(history, 'loss', figure_id=1)
        plot_weight_posteriors(layer_names=weight_posterior_callback.layer_names,
                               logged_epochs=weight_posterior_callback.logged_epochs)
    del history, callbacks
    if os.path.exists(tmp_file_name):
        bayesian_classifier.load_weights(tmp_file_name)




def evaluate_classifier(X_test: np.ndarray, y_test: np.ndarray, language_for_testing: str,
                        bayesian_nn: tf.keras.Model,
                        n_monte_carlo: int, batch_size: int, verbose: bool) -> float:
    probabilities = predict_with_model(
        classifier=bayesian_nn,
        input_data=X_test,
        batch_size=batch_size,
        n_monte_carlo=n_monte_carlo
    )
    err_msg = '{0} != {1}'.format(y_test.shape, probabilities.shape)
    assert y_test.shape == probabilities.shape, err_msg
    if verbose:
        show_roc_auc(y_true=y_val, probabilities=probabilities,
                     label='the testing data (language "{0}")'.format(language_for_testing),
                     figure_id=3)
    quality = roc_auc_score(y_true=y_test, y_score=probabilities)
    del probabilities
    return quality




def show_training_process(history: tf.keras.callbacks.History, metric_name: str,
                          figure_id: int=1):
    val_metric_name = 'val_' + metric_name
    err_msg = 'The metric "{0}" is not found! Available metrics are: {1}'.format(
        metric_name, list(history.history.keys()))
    assert metric_name in history.history, err_msg
    plt.figure(figure_id)
    plt.plot(list(range(len(history.history[metric_name]))),
             history.history[metric_name], label='Training {0}'.format(metric_name))
    if val_metric_name in history.history:
        assert len(history.history[metric_name]) == len(history.history['val_' + metric_name])
        plt.plot(list(range(len(history.history['val_' + metric_name]))),
                 history.history['val_' + metric_name], label='Validation {0}'.format(metric_name))
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title('Training process')
    plt.legend(loc='best')
    plt.show()




def train_siamese_nn(nn: tf.keras.Model, trainset: tf.data.Dataset, steps_per_trainset: int,
                     steps_per_epoch: int, validset: tf.data.Dataset, max_duration: int,
                     model_weights_path: str):
    assert steps_per_trainset >= steps_per_epoch
    n_epochs = int(round(10.0 * steps_per_trainset / float(steps_per_epoch)))
    print('Maximal duration of the Siamese NN training is {0} seconds.'.format(max_duration))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss', mode='min',
                                         restore_best_weights=False, verbose=True),
        tfa.callbacks.TimeStopping(seconds=max_duration, verbose=True),
        tf.keras.callbacks.ModelCheckpoint(model_weights_path, monitor='val_loss',
                                           mode='min', save_best_only=True,
                                           save_weights_only=True, verbose=True)
    ]
    history = nn.fit(
        trainset,
        steps_per_epoch=steps_per_epoch,
        validation_data=validset,
        epochs=n_epochs, callbacks=callbacks
    )
    show_training_process(history, 'loss')




def show_roc_auc(y_true: np.ndarray, probabilities: np.ndarray, label: str,
                 figure_id: int=1):
    plt.figure(figure_id)
    plt.plot([0, 1], [0, 1], 'k--')
    print('ROC-AUC score for {0} is {1:.9f}'.format(
        label, roc_auc_score(y_true=y_true, y_score=probabilities)
    ))
    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=probabilities)
    plt.plot(fpr, tpr, label=label.title())
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()




def calculate_features_of_texts(texts: Dict[str, List[Tuple[str, int]]],
                                tokenizer: XLMRobertaTokenizer, maxlen: int,
                                fe: tf.keras.Model, batch_size: int,
                                max_dataset_size: int = 0) -> \
        Dict[str, Tuple[np.ndarray, np.ndarray]]:
    languages = sorted(list(texts.keys()))
    datasets_by_languages = dict()
    if max_dataset_size > 0:
        max_size_per_lang = max_dataset_size // len(languages)
        err_msg = '{0} is too small number of dataset samples!'.format(max_dataset_size)
        assert max_size_per_lang > 0, err_msg
    else:
        max_size_per_lang = 0
    for cur_lang in languages:
        selected_indices = list(range(len(texts[cur_lang])))
        if max_size_per_lang > 0:
            if len(selected_indices) > max_size_per_lang:
                selected_indices = random.sample(
                    population=selected_indices,
                    k=max_size_per_lang
                )
        tokens_of_texts, mask_of_texts = regular_encode(
            texts=[texts[cur_lang][idx][0] for idx in selected_indices],
            tokenizer=tokenizer, maxlen=maxlen
        )
        X = []
        n_batches = int(np.ceil(len(selected_indices) / float(batch_size)))
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(len(selected_indices), batch_start + batch_size)
            res = fe.predict_on_batch(
                [
                    tokens_of_texts[batch_start:batch_end],
                    mask_of_texts[batch_start:batch_end]
                ]
            )
            if not isinstance(res, np.ndarray):
                res = res.numpy()
            X.append(res)
            del res
        X = np.vstack(X)
        y = np.array([texts[cur_lang][idx][1] for idx in selected_indices], dtype=np.int32)
        datasets_by_languages[cur_lang] = (X, y)
        del X, y, selected_indices
    return datasets_by_languages




def predict_with_model(classifier: tf.keras.Model, input_data: np.ndarray,
                       batch_size: int, n_monte_carlo: int) -> np.ndarray:
    assert n_monte_carlo > 1
    predicted = classifier.predict(input_data, batch_size=batch_size).flatten()
    for _ in range(n_monte_carlo - 1):
        predicted += classifier.predict(input_data, batch_size=batch_size).flatten()
    return predicted / float(n_monte_carlo)




def select_best_model(data_for_training: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
                      languages_for_training: List[str],
                      current_strategy: tf.distribute.Strategy,
                      tpu_system: Union[None, tf.distribute.cluster_resolver.TPUClusterResolver],
                      feature_vector_size: int, neural_depth: int, hidden_layer_size: int,
                      n_monte_carlo: int, max_iters: int, batch_size: int,
                      tmp_file_name: str) -> Dict[str, float]:
    assert len(data_for_training) == len(languages_for_training)
    space  = [Real(1e-3, 1e+3, "log-uniform", name='C'),
              Real(5e-4, 5e-1, "log-uniform", name='lr')]
    n_calls = 18
    n_random_starts = 6
    n_restarts_optimizer = 3
    restart_counter = 1
    datasets_for_training = []
    for lang_idx in range(len(languages_for_training)):
        n_train_samples = data_for_training[lang_idx]['train'][1].shape[0]
        steps_per_epoch = int(np.ceil(n_train_samples / float(batch_size)))
        datasets_for_training.append(
            {
                'train': tf.data.Dataset.from_tensor_slices(
                    data_for_training[lang_idx]['train']
                ).repeat().shuffle(n_train_samples).batch(batch_size),
                'n_epochs': int(np.ceil(max_iters / float(n_train_samples))),
                'steps_per_epoch': steps_per_epoch
            }
        )
    
    @use_named_args(space)
    def objective_f(C: float, lr: float) -> float:
        nonlocal restart_counter
        nonlocal current_strategy
        nonlocal tpu_system
        rocauc_scores = []
        n_splits = len(languages_for_training)
        for lang_idx in range(len(languages_for_training)):
            validation_lang = languages_for_training[lang_idx]
            with current_strategy.scope():
                bnn = build_classifier(
                    feature_vector_size=feature_vector_size, n_train_samples=max_iters,
                    hidden_layer_size=hidden_layer_size, n_hidden_layers=neural_depth,
                    lr=float(lr), C=float(C),
                    language='{0}{1}'.format(validation_lang.strip(), restart_counter),
                    verbose=False
                )
            train_classifier(
                trainset=datasets_for_training[lang_idx]['train'],
                n_steps=datasets_for_training[lang_idx]['steps_per_epoch'],
                bayesian_classifier=bnn, verbose=False,
                max_epochs=datasets_for_training[lang_idx]['n_epochs'],
                tmp_file_name=tmp_file_name
            )
            instant_quality = evaluate_classifier(
                X_test=data_for_training[lang_idx]['test'][0],
                y_test=data_for_training[lang_idx]['test'][1],
                language_for_testing=languages_for_training[lang_idx],
                bayesian_nn=bnn,
                n_monte_carlo=n_monte_carlo, batch_size=batch_size, verbose=False
            )
            rocauc_scores.append(instant_quality)
            del bnn
            gc.collect()
            tf.keras.backend.clear_session()
            if os.path.isfile(tmp_file_name):
                os.remove(tmp_file_name)
        rocauc_score = hmean(rocauc_scores)
        del rocauc_scores
        print('  C={0:.9f}, lr={1:.9f}'.format(C, lr))
        print('  ROC-AUC score = {0:.9f}'.format(rocauc_score))
        restart_counter += 1
        if tpu_system:
            tf.tpu.experimental.shutdown_tpu_system(tpu_system)
            del current_strategy
            tf.tpu.experimental.initialize_tpu_system(tpu_system)
            current_strategy = tf.distribute.experimental.TPUStrategy(tpu_system)
        return -rocauc_score
    
    start_time = time.time()
    res_gp = gp_minimize(
        objective_f, space,
        n_calls=n_calls, n_random_starts=n_random_starts,
        n_restarts_optimizer=n_restarts_optimizer, random_state=42,
        verbose=True, n_jobs=1
    )
    best_parameters = {
        'C': float(res_gp.x[0]),
        'lr': float(res_gp.x[1]),
    }
    automl_duration = int(round(time.time() - start_time))
    print('')
    print('Total duration of the AutoML is {0} seconds.'.format(automl_duration))
    print('')
    print('Best parameters are:')
    print('C={0:.9f}, lr={1:.9f}'.format(best_parameters['C'], best_parameters['lr']))
    print('')
    del datasets_for_training
    plot_convergence(res_gp)
    plot_evaluations(res_gp, bins=10)
    return best_parameters




experiment_start_time = time.time()




try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    model_name = 'jplu/tf-xlm-roberta-large'
    max_seq_len = 512
    batch_size_for_siamese = 4 * strategy.num_replicas_in_sync
else:
    strategy = tf.distribute.get_strategy()
    physical_devices = tf.config.list_physical_devices('GPU')
    for device_idx in range(strategy.num_replicas_in_sync):
        tf.config.experimental.set_memory_growth(physical_devices[device_idx], True)
    max_seq_len = 512
    model_name = 'jplu/tf-xlm-roberta-base'
    batch_size_for_siamese = 2 * strategy.num_replicas_in_sync
batch_size_for_cls = max(8, 64 // strategy.num_replicas_in_sync) * strategy.num_replicas_in_sync
print("REPLICAS: ", strategy.num_replicas_in_sync)
print('Model name: {0}'.format(model_name))
print('Maximal length of sequence is {0}'.format(max_seq_len))
print('Batch size for the Siamese XLM-RoBERTa is {0}'.format(
    batch_size_for_siamese))
print('Batch size for the Bayesian NN is {0}'.format(
    batch_size_for_cls))




random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)




siamese_learning_rate = 1e-6
automl_num_monte_carlo = 15
final_num_monte_carlo = 20
max_iters_of_cls = 60000
depth_of_cls = 4
hidden_layer_of_cls = 1000
dataset_dir = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification'
tmp_roberta_name = '/kaggle/working/siamese_xlmr.h5'
tmp_cls_name = '/kaggle/working/bayesian_cls.h5'




xlmroberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
xlmroberta_config = XLMRobertaConfig.from_pretrained(model_name)
print(xlmroberta_config)




sentence_embedding_size = xlmroberta_config.hidden_size
print('Sentence embedding size is {0}'.format(sentence_embedding_size))
assert max_seq_len <= xlmroberta_config.max_position_embeddings




corpus_for_training = load_train_set(
    os.path.join(dataset_dir, "jigsaw-toxic-comment-train.csv"),
    text_field="comment_text", lang_field="lang",
    sentiment_fields=["toxic", "severe_toxic", "obscene", "threat", "insult",
                      "identity_hate"]
)
assert 'en' in corpus_for_training




random.shuffle(corpus_for_training['en'])
n_validation = int(round(0.15 * len(corpus_for_training['en'])))
corpus_for_validation = {'en': corpus_for_training['en'][:n_validation]}
corpus_for_training = {'en': corpus_for_training['en'][n_validation:]}




multilingual_corpus = load_train_set(
    os.path.join(dataset_dir, "validation.csv"),
    text_field="comment_text", lang_field="lang", sentiment_fields=["toxic", ]
)
assert 'en' not in multilingual_corpus
max_size = 0
print('Multilingual data:')
for language in sorted(list(multilingual_corpus.keys())):
    print('  {0}\t\t{1} samples'.format(language, len(multilingual_corpus[language])))
    assert set(map(lambda cur: cur[1], multilingual_corpus[language])) == {0, 1}
    if len(multilingual_corpus[language]) > max_size:
        max_size = len(multilingual_corpus[language])




texts_for_submission = load_test_set(
    os.path.join(dataset_dir, "test.csv"),
    text_field="content", lang_field="lang", id_field="id"
)
print('Data for submission:')
for language in sorted(list(texts_for_submission.keys())):
    print('  {0}\t\t{1} samples'.format(language, len(texts_for_submission[language])))




dataset_for_training, n_batches_per_data = build_siamese_dataset(
    texts=corpus_for_training, dataset_size=150000,
    tokenizer=xlmroberta_tokenizer, maxlen=max_seq_len,
    batch_size=batch_size_for_siamese, shuffle=True
)




dataset_for_validation, n_batches_per_epoch = build_siamese_dataset(
    texts=corpus_for_validation, dataset_size=1000,
    tokenizer=xlmroberta_tokenizer, maxlen=max_seq_len,
    batch_size=batch_size_for_siamese, shuffle=False
)




del corpus_for_training, corpus_for_validation
gc.collect()




preparing_duration = int(round(time.time() - experiment_start_time))
print("Duration of data loading and preparing to the Siamese NN training is "
      "{0} seconds.".format(preparing_duration))




with strategy.scope():
    siamese_network, feature_extractor = build_siamese_nn(
        transformer_name=model_name,
        hidden_state_size=sentence_embedding_size,
        max_len=max_seq_len,
        lr=siamese_learning_rate
    )




train_siamese_nn(nn=siamese_network, trainset=dataset_for_training,
                 steps_per_trainset=n_batches_per_data,
                 steps_per_epoch=min(5 * n_batches_per_epoch, n_batches_per_data),
                 validset=dataset_for_validation,
                 max_duration=int(round(3600 * 0.33 - preparing_duration)),
                 model_weights_path=tmp_roberta_name)




del dataset_for_training
del dataset_for_validation
gc.collect()




siamese_network.load_weights(tmp_roberta_name)




del siamese_network
gc.collect()




dataset_for_training = calculate_features_of_texts(
    texts=multilingual_corpus,
    tokenizer=xlmroberta_tokenizer, maxlen=max_seq_len,
    fe=feature_extractor,
    batch_size=batch_size_for_siamese
)
assert len(dataset_for_training) == 3




dataset_for_submission = calculate_features_of_texts(
    texts=texts_for_submission,
    tokenizer=xlmroberta_tokenizer, maxlen=max_seq_len,
    fe=feature_extractor,
    batch_size=batch_size_for_siamese
)




X_embedded = []
y_embedded = []
split_by_languages = dict()
start_pos = 0
for cur_lang in dataset_for_training:
    X_embedded.append(dataset_for_training[cur_lang][0])
    y_embedded.append(dataset_for_training[cur_lang][1])
    split_by_languages[cur_lang] = (
        set(),
        set(range(start_pos, start_pos + dataset_for_training[cur_lang][1].shape[0]))
    )
    start_pos = start_pos + dataset_for_training[cur_lang][1].shape[0]
featured_data_for_training = (
    np.vstack(X_embedded),
    np.concatenate(y_embedded)
)
for cur_lang in dataset_for_training:
    indices_for_testing = split_by_languages[cur_lang][1]
    indices_for_training = set(range(featured_data_for_training[0].shape[0])) - indices_for_testing
    split_by_languages[cur_lang] = (
        np.array(sorted(list(indices_for_training)), dtype=np.int32),
        np.array(sorted(list(indices_for_testing)), dtype=np.int32)
    )
    del indices_for_training, indices_for_testing
featured_data_for_submission = []
identifies_for_submission = []
for cur_lang in dataset_for_submission:
    X_embedded.append(dataset_for_submission[cur_lang][0])
    featured_data_for_submission.append(dataset_for_submission[cur_lang][0])
    identifies_for_submission.append(dataset_for_submission[cur_lang][1])
    y_embedded.append(
        np.array(
            [-1 for _ in range(dataset_for_submission[cur_lang][0].shape[0])],
            dtype=np.int32
        )
    )
featured_data_for_submission = np.vstack(featured_data_for_submission)
identifies_for_submission = np.concatenate(identifies_for_submission)
X_embedded = np.vstack(X_embedded)
y_embedded = np.concatenate(y_embedded)




del dataset_for_training, dataset_for_submission
del feature_extractor, xlmroberta_tokenizer




all_languages = sorted(list(split_by_languages.keys()))
prev_lang = all_languages[0]
assert len(set(split_by_languages[prev_lang][1].tolist()) &            set(split_by_languages[prev_lang][0].tolist())) == 0
for cur_lang in all_languages[1:]:
    assert len(set(split_by_languages[cur_lang][1].tolist()) &                set(split_by_languages[cur_lang][0].tolist())) == 0
    assert len(set(split_by_languages[cur_lang][1].tolist()) &                set(split_by_languages[prev_lang][1].tolist())) == 0
    prev_lang = cur_lang




indices_of_samples = random.sample(
    list(range(featured_data_for_training[1].shape[0])),
    k=1000
)
indices_of_samples += random.sample(
    list(range(featured_data_for_training[1].shape[0], y_embedded.shape[0])),
    k=1000
)
X_embedded = X_embedded[indices_of_samples]
y_embedded = y_embedded[indices_of_samples]




X_embedded = TSNE(n_components=2, n_jobs=-1).fit_transform(X_embedded)




indices_of_unknown_classes = list(filter(
    lambda sample_idx: y_embedded[sample_idx] < 0,
    range(len(y_embedded))
))
xy = X_embedded[indices_of_unknown_classes]
plt.plot(xy[:, 0], xy[:, 1], 'o', color='b', markersize=2,
         label='Unlabeled data')
indices_of_negative_classes = list(filter(
    lambda sample_idx: y_embedded[sample_idx] == 0,
    range(len(y_embedded))
))
xy = X_embedded[indices_of_negative_classes]
plt.plot(xy[:, 0], xy[:, 1], 'o', color='g', markersize=4,
         label='Normal texts')
indices_of_positive_classes = list(filter(
    lambda sample_idx: y_embedded[sample_idx] > 0,
    range(len(y_embedded))
))
xy = X_embedded[indices_of_positive_classes]
plt.plot(xy[:, 0], xy[:, 1], 'o', color='r', markersize=6,
         label='Toxic texts')
plt.title('Toxic and normal texts')
plt.legend(loc='best')
plt.show()




del indices_of_negative_classes
del indices_of_positive_classes
del indices_of_unknown_classes
del indices_of_samples
del X_embedded, y_embedded




gc.collect()
tf.keras.backend.clear_session()
if tpu:
    tf.tpu.experimental.shutdown_tpu_system(tpu)
    del strategy
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)




splitted_data_for_training = []
for cur_lang in all_languages:
    datasets = build_datasets_for_classifier(
        data_for_training=featured_data_for_training,
        data_split=split_by_languages,
        language_for_testing=cur_lang
    )
    splitted_data_for_training.append(
        {
            'train': datasets[0],
            'test': datasets[1]
        }
    )
    del datasets




experiment_duration = int(round(time.time() - experiment_start_time))
print('Duration of siamese XLM-RoBERTa preparing is {0} seconds.'.format(
    experiment_duration))




bnn_params = select_best_model(
    data_for_training=splitted_data_for_training,
    languages_for_training=all_languages,
    current_strategy=strategy, tpu_system=tpu,
    feature_vector_size=featured_data_for_training[0].shape[1],
    neural_depth=depth_of_cls, hidden_layer_size=hidden_layer_of_cls,
    batch_size=batch_size_for_cls, max_iters=max_iters_of_cls,
    n_monte_carlo=automl_num_monte_carlo,
    tmp_file_name=tmp_cls_name
)




del splitted_data_for_training




if os.path.isfile(tmp_cls_name):
    os.remove(tmp_cls_name)




n_total_train_samples = featured_data_for_training[1].shape[0]
n_steps_per_epoch = int(np.ceil(n_total_train_samples / float(batch_size_for_cls)))




with strategy.scope():
    final_bayesian_classifier = build_classifier(
        feature_vector_size=featured_data_for_training[0].shape[1],
        n_train_samples=max_iters_of_cls,
        hidden_layer_size=hidden_layer_of_cls,
        n_hidden_layers=depth_of_cls,
        lr=bnn_params['lr'], C=bnn_params['C'],
        language='multilang', verbose=True
    )




train_classifier(
    trainset=tf.data.Dataset.from_tensor_slices(
        featured_data_for_training
    ).repeat().shuffle(n_total_train_samples).batch(batch_size_for_cls),
    n_steps=n_steps_per_epoch,
    bayesian_classifier=final_bayesian_classifier, verbose=True,
    max_epochs=int(np.ceil(max_iters_of_cls / float(n_total_train_samples))),\
    tmp_file_name=tmp_cls_name
)




result_of_submission = predict_with_model(
    classifier=final_bayesian_classifier,
    input_data=featured_data_for_submission,
    batch_size=batch_size_for_cls,
    n_monte_carlo=final_num_monte_carlo
)




assert identifies_for_submission.shape == result_of_submission.shape




with codecs.open('submission.csv', mode='w', encoding='utf-8', errors='ignore') as fp:
    fp.write('id,toxic\n')
    for sample_idx in range(identifies_for_submission.shape[0]):
        id_val = identifies_for_submission[sample_idx]
        proba_val = result_of_submission[sample_idx]
        fp.write('{0},{1:.9f}\n'.format(id_val, proba_val))




print('Experiment duration is {0:.3f}.'.format(time.time() - experiment_start_time))

