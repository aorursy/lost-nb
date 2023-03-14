#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import os
import json
import numpy as np
from pathlib import Path

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation, rc
from IPython.display import HTML

rc('animation', html='jshtml')


data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge')
train_path = data_path / 'training'
valid_path = data_path / 'evaluation'
test_path = data_path / 'test'

train_tasks = {task.stem: json.load(task.open())
               for task in train_path.iterdir()}
valid_tasks = {task.stem: json.load(task.open())
               for task in valid_path.iterdir()}


# In[9]:


cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)


def plot_pictures(pictures, labels = None):
    if labels is None:
        labels = range(len(pictures))
    fig, axs = plt.subplots(1, len(pictures), figsize=(2 * len(pictures), 32))
    for i, (pict, label) in enumerate(zip(pictures, labels)):
        axs[i].imshow(np.array(pict), cmap=cmap, norm=norm)
        axs[i].set_title(label)
    plt.show()


def plot_sample(sample, predict=None):
    if predict is None:
        plot_pictures([sample['input'], sample['output']], ['Input', 'Output'])
    else:
        plot_pictures([sample['input'], sample['output'], predict], [
                      'Input', 'Output', 'Predict'])



def input_output_shape_is_same(task):
    return all([np.array(el['input']).shape == np.array(el['output']).shape for el in task['train']])


def calk_score(task_test, predict):
    return [int(np.equal(sample['output'], pred).all()) for sample, pred in zip(task_test, predict)]

def to_hot(x):
    return tf.one_hot(x,depth=10)
def from_hot(x):
    return tf.argmax(x,axis=-1)

def loss_f(y_pred,y_truth):
    return tf.reduce_mean(tf.keras.backend.categorical_crossentropy(y_pred, y_truth))



task = train_tasks["db3e9e38"]["train"]
test = train_tasks["db3e9e38"]["test"]
for sample in task:
    plot_sample(sample)


# In[10]:


def Model():
    return tf.keras.Sequential([
        Conv2D(128, 3,
        kernel_initializer=tf.random_normal_initializer,
        activation=tf.nn.relu,padding="same"),
        Conv2D(10, 1, activation="softmax",
               kernel_initializer=tf.random_normal_initializer
               ),
    ])


# In[12]:


def solve(task):
    ca = Model()
    num_epochs = 100
    trainer = tf.keras.optimizers.Adam(lr=0.1)
    for e in range(num_epochs):
        loss = 0
        with tf.GradientTape() as g:
            # for sample in task:
            iter_n = tf.random.uniform([], minval=2, maxval=15, dtype=tf.int32)
            for sample in task:
                x = to_hot(sample["input"])[None]
                y = to_hot(sample["output"])[None]
                for i in tf.range(iter_n):
                    x = ca(x)
                factor = tf.cast(iter_n,tf.float32)
                loss += loss_f(x,y)*factor*factor + 1e3*loss_f(ca(y),y)
                # loss += loss_f(x,y)
        grads = g.gradient(loss, ca.weights)
        grads = [g / (tf.norm(g) + 1e-8) for g in grads]
        trainer.apply_gradients(zip(grads, ca.weights))
    return ca



ca = get_ipython().run_line_magic('time', 'solve(task)')


# In[ ]:


def predict(ca,x,num=2):
    for i in range(num):
        r = ca(x)
        # r = to_hot(from_hot(r))
        x = r
    return x

def evaluate(task):
    if input_output_shape_is_same(task):
        ca = solve(task["train"])
        for test in task["test"]:
            to_pred = to_hot(test["input"])[None]
            test["prediction"] = from_hot(predict(ca,to_pred, 13))[0].numpy()
        # return pred
    return None

for idx, task in tqdm(train_tasks.items()):
    evaluate(task)


# In[ ]:


for idx, task in tqdm(train_tasks.items()):
        for test in task["test"]:
            if "prediction" in test and test["prediction"] is not None:
                plot_pictures([test["input"],test["output"],test["prediction"]],["inp","out","pred"])
            else:
                print(None)

