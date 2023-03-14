#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install efficientnet -U')


# In[2]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics, model_selection
import glob
import os
from PIL import Image
import tqdm.notebook as tqdm
import matplotlib.pyplot as plt
import math
import collections

from efficientnet.tfkeras import (
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
)
from tensorflow.keras.applications import ResNet50, InceptionV3

from albumentations import *

mixed_precision = False
gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(num_gpus, "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
        
    mixed_precision = True
    # turn on mixed precision
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

if num_gpus == 0:
    strategy = tf.distribute.OneDeviceStrategy(device='CPU')
    print("Setting strategy to OneDeviceStrategy(device='CPU')")
elif num_gpus == 1:
    strategy = tf.distribute.OneDeviceStrategy(device='GPU')
    print("Setting strategy to OneDeviceStrategy(device='GPU')")
else:
    strategy = tf.distribute.MirroredStrategy()
    print("Setting strategy to MirroredStrategy()")


# In[3]:


path = '../input/siim-isic-melanoma-classification/'

input_path = '../input/'
train_data = pd.read_csv(path + 'train.csv')
test_data = pd.read_csv(path + 'test.csv')
    
submission_data = pd.read_csv(path + 'sample_submission.csv')
test_data['target'] = 0
print("test shape =", test_data.shape)
print(test_data.head(3))
print("\ntrain shape =", train_data.shape)
print(train_data.head(3))


# In[4]:


augmentor = (
    Compose([
        ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=180,
            p=0.5),
        RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.5),
        HorizontalFlip(
            p=0.5),
        Transpose(
            p=0.5),
    ])
)



def _parse_record(serialized, features_to_parse):
    
    features = {}
    for key in features_to_parse:
        if key == 'image_name' or key == 'image':
            features[key] = tf.io.FixedLenFeature(
                [], tf.string, default_value='')
        else:
            features[key] = tf.io.FixedLenFeature(
                [], tf.int64, default_value=0)   
    example = tf.io.parse_single_example(
        serialized=serialized, features=features)
    
    extracted = {}
    for key in features_to_parse:
        if key == 'image':
            extracted[key] = tf.io.decode_jpeg(example[key], channels=3)
        else:
            extracted[key] = example[key]
    return extracted
    
def _transform_image(*features):
    '''Using albumentations augmentations, which will be 
    wrapped in tf.py_function, is very convenient. However, 
    for better performance, consider using TF operations 
    instead to augment the image data
    '''
    features = list(features)
    features[0] = augmentor(image=features[0].numpy())['image']
    return features

def _preprocess_features(features):
    for key in features.keys():
        if key == 'image':
            features[key] = tf.cast(features[key], dtype=tf.float32) / 255.
        elif key == 'anatom_site_general_challenge':
            features[key] = tf.cast(tf.one_hot(features[key], 7), tf.float32)
        elif key == 'diagnosis':
            features[key] = tf.cast(tf.one_hot(features[key], 10), tf.float32)
        elif key == 'image_name':
            features[key] = tf.expand_dims(features[key], -1)
        else:
            features[key] = tf.expand_dims(tf.cast(features[key], dtype=tf.float32), -1)
    return features

def get_dataset(tfrec_paths,
                batch_size=16,
                augment=False,
                shuffle=False,
                cache=False):
    
    FEATURES_TO_PARSE = [
        'image', 'image_name', 'patient_id', 
        'target', 'anatom_site_general_challenge', 
        'sex', 'age_approx', 'diagnosis'
    ]
    
    def deconstruct(features):
        '''dict(features) --> list(features)'''
        return list(features.values())
    
    def construct(*features):
        '''list(features) --> dict(features)'''
        return dict(zip(FEATURES_TO_PARSE, features))
    
    if cache:
        if not(os.path.isdir('tmp/')):
            os.mkdir('tmp/')
        else:
            files = glob.glob('tmp/*')
            for file in files:
                os.remove(file)

        if isinstance(cache, str):
            cache_path = 'tmp/' + cache
        else:
            cache_path = ''
    
    dataset = tf.data.TFRecordDataset(
        filenames=tfrec_paths,
        num_parallel_reads=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(
        lambda x: _parse_record(
            x, features_to_parse=FEATURES_TO_PARSE),
        tf.data.experimental.AUTOTUNE)

    if cache: 
        dataset = dataset.cache(cache_path)

    if shuffle: 
        dataset = dataset.shuffle(1024)

    if augment:
        dataset = dataset.map(deconstruct, tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda *args: tf.py_function(
                func=_transform_image,
                inp=args,
                Tout=[a.dtype for a in args]),
            tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(construct, tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size)

    dataset = dataset.map(_preprocess_features, tf.data.experimental.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
    
    
# Test
ds = get_dataset(
    tfrec_paths=glob.glob(input_path+'melanoma-384x384/train[0-9]*'),
    batch_size=16,
    augment='heavy',
    shuffle=False,
    cache=None
)

for inp in ds.take(1): pass
del ds

fig, axes = plt.subplots(4, 4, figsize=(20, 30))

for i, ax in enumerate(axes.reshape(-1)):
    ax.imshow(inp['image'].numpy()[i]);
    ax.set_title(
        ' image_name: ' + str(inp['image_name'].numpy()[i]) + '\n' + 
        ' patient_id: ' + str(inp['patient_id'].numpy()[i]) + '\n' + 
        ' target: '     + str(inp['target'].numpy()[i]) + '\n' + 
        ' diagnosis: '  + str(inp['target'].numpy()[i]) + '\n' +
        ' site: '       + str(inp['anatom_site_general_challenge'].numpy()[i]) + '\n' + 
        ' sex: '        + str(inp['sex'].numpy()[i]) + '\n' + 
        ' age: '        + str(inp['age_approx'].numpy()[i]) + '\n'
    )
    ax.axis('off')
    
plt.subplots_adjust(hspace=0.1, wspace=0.025)


# In[5]:


def sigmoid_focal_cross_entropy_with_logits(
    labels, logits, alpha=0.25, gamma=2.0):
    
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero")

    logits = tf.convert_to_tensor(logits)
    labels = tf.convert_to_tensor(labels, dtype=logits.dtype)

    # Get the cross_entropy for each entry
    ce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits) 

    # If logits are provided then convert the predictions into probabilities
    pred_prob = tf.math.sigmoid(logits)
    
    p_t = (labels * pred_prob) + ((1 - labels) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)
        alpha_factor = labels * alpha + (1 - labels) * (1 - alpha)

    if gamma:
        gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    return tf.math.reduce_sum(
        alpha_factor * modulating_factor * ce, axis=-1)


class NeuralNet(tf.keras.Model):
    
    def __init__(self, engine, input_shape, pretrained_weights):

        super(NeuralNet, self).__init__()

        self.engine = engine(
            include_top=False,
            input_shape=input_shape,
            weights=pretrained_weights)

        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.concat = tf.keras.layers.Concatenate()
        
        self.sequential_meta = tf.keras.Sequential([
            tf.keras.layers.Dense(64),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])
        
        self.sequential_merged = tf.keras.Sequential([
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, dtype='float32')
        ])
        
        
    def call(self, inputs, **kwargs):
        
        if isinstance(inputs, dict):
            images = inputs['image'] 
            site = inputs['anatom_site_general_challenge']
            sex = inputs['sex']
            age = inputs['age_approx']
        else:
            # when model.build(input_shape) is called
            images = inputs[0]
            site = inputs[1]
            sex = inputs[2]
            age = inputs[3]
        
        x1 = self.engine(images)
        x1 = self.pool(x1)
        x2 = tf.concat([site, sex, age], axis=-1)
        x2 = self.sequential_meta(x2)
        x3 = self.concat([x1, x2])
        x3 = self.sequential_merged(x3)
        return x3
    
    
class DistributedModel:
    
    def __init__(self, 
                 engine,
                 input_shape=(384, 384, 3),
                 pretrained_weights=None,
                 finetuned_weights=None,
                 batch_size=8,
                 optimizer=None, 
                 strategy=None,
                 mixed_precision=False, 
                 label_smoothing=0.0,
                 tta=1,
                 focal_loss=True,
                 save_best=None):
        
        self.keras_model = NeuralNet(
            engine=engine,
            input_shape=input_shape,
            pretrained_weights=pretrained_weights)
        self.keras_model.build(
            [[None, *input_shape], [None, 7], [None, 1], [None, 1]])
        if finetuned_weights:
            self.keras_model.load_weights(finetuned_weights)
        self._initial_weights = self.keras_model.get_weights()
        self.global_batch_size = batch_size
        self.mixed_precision = mixed_precision
        self.optimizer = optimizer
        self.strategy = strategy
        self.label_smoothing = label_smoothing
        self.tta = tta
        self.focal_loss = focal_loss
        self.save_best = save_best
        
        self.auc_metric = tf.keras.metrics.AUC()
        self.loss_metric = tf.keras.metrics.Mean()
        
        if self.optimizer and self.mixed_precision:
            self.optimizer =                 tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                    optimizer, loss_scale='dynamic')
                
        if self.strategy:
            self.global_batch_size *= self.strategy.num_replicas_in_sync
            
        if save_best and not(os.path.isdir(save_best)):
            os.makedirs(save_best) 
    
    def reset_weights(self):
        self.keras_model.set_weights(self._initial_weights)
    
    def _compute_loss(self, labels, logits):
        if self.focal_loss:
            per_example_loss = sigmoid_focal_cross_entropy_with_logits(
                labels=labels, logits=logits, alpha=0.8, gamma=2.0)
        else:
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits) 
        return tf.nn.compute_average_loss(
            per_example_loss, global_batch_size=self.global_batch_size)
        
    @tf.function
    def _distributed_train_step(self, dist_inputs):
        
        def train_step(inputs):

            if self.label_smoothing:
                target = (
                    inputs['target'] * (1 - self.label_smoothing)
                    + 0.5 * self.label_smoothing
                )
            else:
                target = inputs['target']

            with tf.GradientTape() as tape:
                
                logits = self.keras_model(inputs, training=True)
                loss = self._compute_loss(target, logits)
                self.loss_metric.update_state(loss)
                self.auc_metric.update_state(
                    tf.math.round(target), tf.math.sigmoid(logits))
                if self.mixed_precision:
                    scaled_loss = self.optimizer.get_scaled_loss(loss)

            if self.mixed_precision:
                scaled_gradients = tape.gradient(
                    scaled_loss, self.keras_model.trainable_variables)
                gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
            else:
                gradients = tape.gradient(loss, self.keras_model.trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients, self.keras_model.trainable_variables))

            return loss
        
        per_replica_loss = self.strategy.run(train_step, args=(dist_inputs,))
        return self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
        
    @tf.function
    def _distributed_predict_step(self, dist_inputs):
        
        def predict_step(inputs):
            logits = self.keras_model(inputs, training=False)
            return tf.math.sigmoid(logits), inputs['image_name'], inputs['target']
    
        preds, image_names, trues = self.strategy.run(predict_step, args=(dist_inputs,))
        if tf.is_tensor(preds):
            return [preds], [image_names], [trues]
        else:
            return preds.values, image_names.values, trues.values
    
    def fit(self, ds):

        ds = self.strategy.experimental_distribute_dataset(ds)
        ds = tqdm.tqdm(ds)
        
        for i, inputs in enumerate(ds):
            loss = self._distributed_train_step(inputs)
            ds.set_description(
                "valid AUC {:.4f} : Loss/AUC [{:.4f}, {:.4f}]".format(
                    self.auc_score, 
                    self.loss_metric.result().numpy(), 
                    self.auc_metric.result().numpy()
                )
            )
            
        self.loss_metric.reset_states()
        self.auc_metric.reset_states()

    def predict(self, ds):

        ds = self.strategy.experimental_distribute_dataset(ds.repeat(self.tta))
        ds = tqdm.tqdm(ds)
        
        preds_accum = np.zeros([0, 1], dtype=np.float32)
        names_accum = np.zeros([0, 1], dtype=str)
        trues_accum = np.zeros([0, 1], dtype=np.float32)
        
        for inputs in ds:
            preds, names, trues = self._distributed_predict_step(inputs)
        
            for pred, name, true in zip(preds, names, trues):
                preds_accum = np.concatenate([preds_accum, pred.numpy()], axis=0)
                names_accum = np.concatenate([names_accum, name.numpy()], axis=0)
                trues_accum = np.concatenate([trues_accum, true.numpy()], axis=0)
        
        preds_accum = preds_accum.reshape((self.tta, -1)).mean(axis=0)
        names_accum = names_accum.reshape((self.tta, -1))[0]
        trues_accum = trues_accum.reshape((self.tta, -1)).mean(axis=0).round()
        
        return preds_accum, names_accum, trues_accum
    
    def fit_and_predict(self, fold, epochs, train_ds, valid_ds, test_ds):
        
        self.auc_score = 0.
        self.best_score = 0.
        for epoch in range(epochs):
            
            # fit for an epoch
            self.fit(train_ds)
            
            if epoch >= 9: # if statement temporary added to save time
                # predict on validation set
                valid_preds, valid_names, valid_trues = self.predict(valid_ds)

                # compute auc score and save model if best_score
                self.auc_score = metrics.roc_auc_score(valid_trues, valid_preds)

                if self.auc_score > self.best_score:
                    self.best_score = self.auc_score
                    best_valid_preds = valid_preds.copy()
                    if self.save_best:
                        self.keras_model.save_weights(
                            self.save_best+f'{self.keras_model.layers[0].name}-{fold}-{epoch}.h5')                
                    # predict on test set
                    test_preds, test_names, _ = self.predict(test_ds)
                
        return best_valid_preds, valid_names, test_preds, test_names 


# In[6]:


def get_optimizer(steps_per_epoch, lr_max, lr_min,
                  decay_epochs, warmup_epochs, power=1):

    if decay_epochs > 0:
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=lr_max,
            decay_steps=steps_per_epoch*decay_epochs,
            end_learning_rate=lr_min,
            power=power,
        )
    else:
        learning_rate_fn = lr_max

    if warmup_epochs > 0:
        learning_rate_fn = WarmUp(
            lr_start = lr_min,
            lr_end = lr_max,
            lr_fn = learning_rate_fn,
            warmup_steps=steps_per_epoch*warmup_epochs,
            power=power,
        )

    return tf.keras.optimizers.Adam(learning_rate_fn)


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, lr_start, lr_end, lr_fn, warmup_steps, power=1):
        super().__init__()
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.lr_fn = lr_fn
        self.warmup_steps = warmup_steps
        self.power = power

    def __call__(self, step):
        global_step_float = tf.cast(step, tf.float32)
        warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
        warmup_percent_done = global_step_float / warmup_steps_float
        warmup_learning_rate = tf.add(tf.multiply(
            self.lr_start-self.lr_end,
            tf.math.pow(1-warmup_percent_done, self.power)), self.lr_end)
        return tf.cond(
            global_step_float < warmup_steps_float,
            lambda: warmup_learning_rate,
            lambda: self.lr_fn(step),
        )

    def get_config(self):
        return {
            "lr_start": self.lr_start,
            "lr_end": self.lr_end,
            "lr_fn": self.lr_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
        }


# In[7]:


config = {
    'lr_max': 3e-4,
    'lr_min': 3e-5,
    'lr_decay_epochs': 14,
    'lr_warmup_epochs': 1,
    'lr_decay_power': 1,
    'n_epochs': 12,
    'label_smoothing': 0.05,
    'focal_loss': False,
    'tta': 5, # if tta (> 1), shuffle has to be False
    'save_best': None, # if not None, should be a path
    'pretrained_weights': 'imagenet',
    'finetuned_weights': None,
}

fold_config = {
    0: {
        'engine': EfficientNetB0,
        'input_path': 'melanoma-512x512/',
        'input_shape': (512, 512, 3),
        'batch_size': 32,
    },
    1: {
        'engine': EfficientNetB0,
        'input_path': 'melanoma-384x384/',
        'input_shape': (384, 384, 3),
        'batch_size': 32,
    },
    2: {
        'engine': EfficientNetB0,
        'input_path': 'melanoma-256x256/',
        'input_shape': (256, 256, 3),
        'batch_size': 32,
    },
    3: {
        'engine': EfficientNetB1,
        'input_path': 'melanoma-384x384/',
        'input_shape': (384, 384, 3),
        'batch_size': 32,
    },
    4: {
        'engine': EfficientNetB2,
        'input_path': 'melanoma-384x384/',
        'input_shape': (384, 384, 3),
        'batch_size': 24,
    },
}

splits = model_selection.KFold(
    len(fold_config), shuffle=True, random_state=42).split(X=range(15))

with strategy.scope():
    
    valid_preds_accum, test_preds_accum, test_names_accum = list(), list(), list()
    
    for fold, (train_idx, valid_idx) in enumerate(splits):
        
        optimizer = get_optimizer(
            steps_per_epoch=33126//fold_config[fold]['batch_size'], # rough estimation
            lr_max=config['lr_max'],
            lr_min=config['lr_min'],
            decay_epochs=config['lr_decay_epochs'],
            warmup_epochs=config['lr_warmup_epochs'],
            power=config['lr_decay_power']
        )
        
        dist_model = DistributedModel(
            engine=fold_config[fold]['engine'],
            input_shape=fold_config[fold]['input_shape'],
            pretrained_weights=config['pretrained_weights'],
            finetuned_weights=config['finetuned_weights'],
            batch_size=fold_config[fold]['batch_size'],
            optimizer=optimizer, 
            strategy=strategy,
            mixed_precision=mixed_precision, 
            label_smoothing=config['label_smoothing'],
            tta=config['tta'],
            focal_loss=config['focal_loss'],
            save_best=config['save_best'])
        
        
        tfrec_paths = np.asarray(
            glob.glob(input_path+fold_config[fold]['input_path']+'train[0-9]*'))
        test_paths = glob.glob(
            input_path+fold_config[fold]['input_path']+'test[0-9]*')
        train_paths = tfrec_paths[train_idx]
        valid_paths = tfrec_paths[valid_idx]

        train_ds = get_dataset(
            train_paths, fold_config[fold]['batch_size'], augment=True, shuffle=True)
        valid_ds = get_dataset(
            valid_paths, fold_config[fold]['batch_size'], augment=True)
        test_ds = get_dataset(
            test_paths, fold_config[fold]['batch_size'], augment=True)

        valid_preds, _, test_preds, test_names = dist_model.fit_and_predict(
            fold=fold, 
            epochs=config['n_epochs'], 
            train_ds=train_ds, 
            valid_ds=valid_ds, 
            test_ds=test_ds, 
        )

        valid_preds_accum.append(valid_preds)
        test_preds_accum.append(test_preds)
        test_names_accum.append(test_names)

        # dist_model.reset_weights()


# In[8]:


final_preds = np.average(test_preds_accum, axis=0, weights=[1,1,1,1,1])
final_preds_map = dict(zip(test_names_accum[0].astype('U13'), final_preds))
submission_data['target'] = submission_data.image_name.map(final_preds_map)
submission_data.to_csv('submission.csv', index=False)


# In[ ]:




