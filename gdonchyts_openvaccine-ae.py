#!/usr/bin/env python
# coding: utf-8













# from google.colab import drive
# drive.mount('/content/drive')




# !mkdir /kaggle
# !ln -s /content/drive/My\ Drive/Kaggle /kaggle/input




# D_WORK='/kaggle/input/stanford-covid-vaccine/work/kg-openvaccine-ae-v2/'
D_WORK='./'




# !mkdir {D_WORK}




# !ls {D_WORK}




get_ipython().system('pip install -q keras-adamw')




pretrain_dir = None#"/kaggle/input/covid-v9-no-consis/"

one_fold = False
# one_fold = True#False
# with_ae = False#True
run_test = False
# run_test = True
denoise = True

# ae_epochs = 20
# ae_epochs_each = 5
# ae_batch_size = 32

ae_epochs = 50
ae_epochs_each = 15
ae_batch_size = 16

# epochs_list = [30, 10, 3, 3, 5, 5]
epochs_list = [60, 20, 5, 3, 3, 5, 5]
batch_size_list = [4, 8, 16, 32, 64, 128, 256]

## copy pretrain model to working dir
import shutil
import glob
if pretrain_dir is not None:
    for d in glob.glob(pretrain_dir + "*"):
        shutil.copy(d, ".")
    
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




import json
import glob
from tqdm.notebook import tqdm

train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json",lines=True)
if denoise:
    print('Number of rows', len(train))
    train = train[train.signal_to_noise > 1].reset_index(drop = True)
    print('Number of rows after SNR filtering', len(train))
test  = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json",lines=True)
test_pub = test[test["seq_length"] == 107]
test_pri = test[test["seq_length"] == 130]
sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")

if run_test:
    train = train[:30]
    test_pub = test_pub[:30]
    test_pri = test_pri[:30]




get_ipython().system('unzip -qq /kaggle/input/stanford-covid-vaccine/bpps.zip -d {D_WORK}')




As = []
for id in tqdm(train["id"]):
    # a = np.load(f"/kaggle/input/stanford-covid-vaccine/bpps/{id}.npy")
    a = np.load(f"{D_WORK}/bpps/{id}.npy")
    As.append(a)
As = np.array(As)
As_pub = []
for id in tqdm(test_pub["id"]):
    # a = np.load(f"/kaggle/input/stanford-covid-vaccine/bpps/{id}.npy")
    a = np.load(f"{D_WORK}/bpps/{id}.npy")
    As_pub.append(a)
As_pub = np.array(As_pub)
As_pri = []
for id in tqdm(test_pri["id"]):
    # a = np.load(f"/kaggle/input/stanford-covid-vaccine/bpps/{id}.npy")
    a = np.load(f"{D_WORK}/bpps/{id}.npy")
    As_pri.append(a)
As_pri = np.array(As_pri)




print(train.shape)
train.head()




print(test.shape)
test.head()




print(sub.shape)
sub.head()




targets = list(sub.columns[1:])
print(targets)

y_train = []
seq_len = train["seq_length"].iloc[0]
seq_len_target = train["seq_scored"].iloc[0]
ignore = -10000
ignore_length = seq_len - seq_len_target
for target in targets:
    y = np.vstack(train[target])
    dummy = np.zeros([y.shape[0], ignore_length]) + ignore
    y = np.hstack([y, dummy])
    y_train.append(y)
y = np.stack(y_train, axis = 2)
y.shape




def get_structure_adj(train):
    Ss = []
    for i in tqdm(range(len(train))):
        seq_length = train["seq_length"].iloc[i]
        structure = train["structure"].iloc[i]
        sequence = train["sequence"].iloc[i]

        cue = []
        a_structures = {
            ("A", "U") : np.zeros([seq_length, seq_length]),
            ("C", "G") : np.zeros([seq_length, seq_length]),
            ("U", "G") : np.zeros([seq_length, seq_length]),
            ("U", "A") : np.zeros([seq_length, seq_length]),
            ("G", "C") : np.zeros([seq_length, seq_length]),
            ("G", "U") : np.zeros([seq_length, seq_length]),
        }
        a_structure = np.zeros([seq_length, seq_length])
        for i in range(seq_length):
            if structure[i] == "(":
                cue.append(i)
            elif structure[i] == ")":
                start = cue.pop()
#                 a_structure[start, i] = 1
#                 a_structure[i, start] = 1
                a_structures[(sequence[start], sequence[i])][start, i] = 1
                a_structures[(sequence[i], sequence[start])][i, start] = 1
        
        a_strc = np.stack([a for a in a_structures.values()], axis = 2)
        a_strc = np.sum(a_strc, axis = 2, keepdims = True)
        Ss.append(a_strc)
    
    Ss = np.array(Ss)
    print(Ss.shape)
    return Ss
Ss = get_structure_adj(train)
Ss_pub = get_structure_adj(test_pub)
Ss_pri = get_structure_adj(test_pri)




def get_distance_matrix(As):
    idx = np.arange(As.shape[1])
    Ds = []
    for i in range(len(idx)):
        d = np.abs(idx[i] - idx)
        Ds.append(d)

    Ds = np.array(Ds) + 1
    Ds = 1/Ds
    Ds = Ds[None, :,:]
    Ds = np.repeat(Ds, len(As), axis = 0)
    
    Dss = []
    for i in [1, 2, 4]:
        Dss.append(Ds ** i)
    Ds = np.stack(Dss, axis = 3)
    print(Ds.shape)
    return Ds

Ds = get_distance_matrix(As)
Ds_pub = get_distance_matrix(As_pub)
Ds_pri = get_distance_matrix(As_pri)




## concat adjecent
As = np.concatenate([As[:,:,:,None], Ss, Ds], axis = 3).astype(np.float32)
As_pub = np.concatenate([As_pub[:,:,:,None], Ss_pub, Ds_pub], axis = 3).astype(np.float32)
As_pri = np.concatenate([As_pri[:,:,:,None], Ss_pri, Ds_pri], axis = 3).astype(np.float32)
del Ss, Ds, Ss_pub, Ds_pub, Ss_pri, Ds_pri
As.shape, As_pub.shape, As_pri.shape




## sequence
def return_ohe(n, i):
    tmp = [0] * n
    tmp[i] = 1
    return tmp

def get_input(train):
    mapping = {}
    vocab = ["A", "G", "C", "U"]
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_node = np.stack(train["sequence"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))

    mapping = {}
    vocab = ["S", "M", "I", "B", "H", "E", "X"]
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_loop = np.stack(train["predicted_loop_type"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))
    
    mapping = {}
    vocab = [".", "(", ")"]
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_structure = np.stack(train["structure"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))
    
    
    X_node = np.concatenate([X_node, X_loop], axis = 2)
    
    ## interaction
    a = np.sum(X_node * (2 ** np.arange(X_node.shape[2])[None, None, :]), axis = 2)
    vocab = sorted(set(a.flatten()))
    print(vocab)
    ohes = []
    for v in vocab:
        ohes.append(a == v)
    ohes = np.stack(ohes, axis = 2)
    X_node = np.concatenate([X_node, ohes], axis = 2).astype(np.float32)
    
    
    print(X_node.shape)
    return X_node

X_node = get_input(train)
X_node_pub = get_input(test_pub)
X_node_pri = get_input(test_pri)




import tensorflow as tf
from tensorflow.keras import layers as L
import tensorflow_addons as tfa
from tensorflow.keras import backend as K




# # Detect hardware, return appropriate distribution strategy
# try:
#     # TPU detection. No parameters necessary if TPU_NAME environment variable is
#     # set: this is always the case on Kaggle.
#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#     print('Running on TPU ', tpu.master())
# except ValueError:
#     tpu = None

# if tpu:
#     tf.config.experimental_connect_to_cluster(tpu)
#     tf.tpu.experimental.initialize_tpu_system(tpu)
#     strategy = tf.distribute.experimental.TPUStrategy(tpu)
# else:
#     # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
#     strategy = tf.distribute.get_strategy()

# print("REPLICAS: ", strategy.num_replicas_in_sync)




import os
os.environ["TF_KERAS"]="1"
from keras_adamw import AdamW




def mcrmse(t, p, seq_len_target = seq_len_target):
    score = np.mean(np.sqrt(np.mean((p - y_va) ** 2, axis = 2))[:, :seq_len_target])
    return score

def mcrmse_loss(t, y, seq_len_target = seq_len_target):
    t = t[:, :seq_len_target]
    y = y[:, :seq_len_target]
    
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_mean((t - y) ** 2, axis = 1)))

    return loss

def attention(x_inner, x_outer, n_factor, dropout):
    x_Q =  L.Conv1D(n_factor, 1, activation='linear', 
                  kernel_initializer='glorot_uniform',
                  bias_initializer='glorot_uniform',
                 )(x_inner)
    x_K =  L.Conv1D(n_factor, 1, activation='linear', 
                  kernel_initializer='glorot_uniform',
                  bias_initializer='glorot_uniform',
                 )(x_outer)
    x_V =  L.Conv1D(n_factor, 1, activation='linear', 
                  kernel_initializer='glorot_uniform',
                  bias_initializer='glorot_uniform',
                 )(x_outer)
    x_KT = L.Permute((2, 1))(x_K)
    res = L.Lambda(lambda c: K.batch_dot(c[0], c[1]) / np.sqrt(n_factor))([x_Q, x_KT])
#     res = tf.expand_dims(res, axis = 3)
#     res = L.Conv2D(16, 3, 1, padding = "same", activation = "relu")(res)
#     res = L.Conv2D(1, 3, 1, padding = "same", activation = "relu")(res)
#     res = tf.squeeze(res, axis = 3)
    att = L.Lambda(lambda c: K.softmax(c, axis=-1))(res)
    att = L.Lambda(lambda c: K.batch_dot(c[0], c[1]))([att, x_V])
    return att

def multi_head_attention(x, y, n_factor, n_head, dropout):
    if n_head == 1:
        att = attention(x, y, n_factor, dropout)
    else:
        n_factor_head = n_factor // n_head
        heads = [attention(x, y, n_factor_head, dropout) for i in range(n_head)]
        att = L.Concatenate()(heads)
        att = L.Dense(n_factor, 
                      kernel_initializer='glorot_uniform',
                      bias_initializer='glorot_uniform',
                      
                     )(att)
    x = L.Add()([x, att])
    x = L.LayerNormalization()(x)
    # x = L.BatchNormalization()(x)
    if dropout > 0:
        x = L.Dropout(dropout)(x)
    return x

def res(x, unit, kernel = 3, rate = 0.1):
    h = L.Conv1D(unit, kernel, 1, padding = "same", activation = None)(x)
    h = L.LayerNormalization()(h)
    # x = L.BatchNormalization()(x)
    h = L.LeakyReLU()(h)
    h = L.Dropout(rate)(h)
    return L.Add()([x, h])

def forward(x, unit, kernel = 3, rate = 0.1):
#     h = L.Dense(unit, None)(x)
    h = L.Conv1D(unit, kernel, 1, padding = "same", activation = None)(x)
    h = L.LayerNormalization()(h)
    # x = L.BatchNormalization()(x)
    h = L.Dropout(rate)(h)
    # h = tf.keras.activations.swish(h)
    h = L.LeakyReLU()(h)
    h = res(h, unit, kernel, rate)
    return h

def adj_attn(x, adj, unit, n = 2, rate = 0.1):
    x_a = x
    x_as = []
    for i in range(n):
        x_a = forward(x_a, unit)
        x_a = tf.matmul(adj, x_a)
        x_as.append(x_a)
    if n == 1:
        x_a = x_as[0]
    else:
        x_a = L.Concatenate()(x_as)
    x_a = forward(x_a, unit)
    return x_a


def get_base(config):
    node = tf.keras.Input(shape = (None, X_node.shape[2]), name = "node")
    adj = tf.keras.Input(shape = (None, None, As.shape[3]), name = "adj")
    
    adj_learned = L.Dense(1, "relu")(adj)
    adj_all = L.Concatenate(axis = 3)([adj, adj_learned])
        
    xs = []
    xs.append(node)
    x1 = forward(node, 128, kernel = 3, rate = 0.0)
    x2 = forward(x1, 64, kernel = 6, rate = 0.0)
    x3 = forward(x2, 32, kernel = 15, rate = 0.0)
    x4 = forward(x3, 16, kernel = 30, rate = 0.0)
    x = L.Concatenate()([x1, x2, x3, x4])
    
    for unit in [64, 32]:
        x_as = []
        for i in range(adj_all.shape[3]):
            x_a = adj_attn(x, adj_all[:, :, :, i], unit, rate = 0.0)
            x_as.append(x_a)
        x_c = forward(x, unit, kernel = 30)
        
        x = L.Concatenate()(x_as + [x_c])
        x = forward(x, unit)
        x = multi_head_attention(x, x, unit, 4, 0.0)
        xs.append(x)
        
    x = L.Concatenate()(xs)

    model = tf.keras.Model(inputs = [node, adj], outputs = [x])
    return model


def get_ae_model(base, config):
    node = tf.keras.Input(shape = (None, X_node.shape[2]), name = "node")
    adj = tf.keras.Input(shape = (None, None, As.shape[3]), name = "adj")

    x = base([L.SpatialDropout1D(0.3)(node), adj])
    x = forward(x, 64, rate = 0.3)
    p = L.Dense(X_node.shape[2], "sigmoid")(x)
    
    loss = - tf.reduce_mean(20 * node * tf.math.log(p + 1e-4) + (1 - node) * tf.math.log(1 - p + 1e-4))
    model = tf.keras.Model(inputs = [node, adj], outputs = [loss])
    
    opt = get_optimizer()
    model.compile(optimizer = opt, loss = lambda t, y : y)
    return model


def get_model(base, config):
    node = tf.keras.Input(shape = (None, X_node.shape[2]), name = "node")
    adj = tf.keras.Input(shape = (None, None, As.shape[3]), name = "adj")
    
    x = base([node, adj])
    x = forward(x, 128, rate = 0.4)
    x = L.Dense(5, None)(x)

    model = tf.keras.Model(inputs = [node, adj], outputs = [x])
    
    opt = get_optimizer()
    model.compile(optimizer = opt, loss = mcrmse_loss)
    return model

def get_optimizer():
    # sgd = tf.keras.optimizers.SGD(0.05, momentum = 0.9, nesterov=True)
    # adam = tf.optimizers.Adam()
    # radam = tfa.optimizers.RectifiedAdam()
    # lookahead = tfa.optimizers.Lookahead(adam, sync_period=6)
    # swa = tfa.optimizers.SWA(adam)
    adam = AdamW(learning_rate=0.0005)

    return adam




import matplotlib.pyplot as plt
from IPython.display import clear_output        
import seaborn as sns
import pathlib

class TrainingPlot(tf.keras.callbacks.Callback):
    def __init__(self, nrows=1, ncols=2, figsize=(10, 5), title=None, save_file=None, old_logs_path=None):
        self.nrows=nrows
        self.ncols=ncols
        self.figsize=figsize
        self.title=title
        self.old_logs_path=old_logs_path
        self.old_logs = []
        self.old_log_file_names = []

        if self.old_logs_path is not None:
            p = pathlib.Path(self.old_logs_path)
            self.old_logs_files = p.parent.glob(p.name)

            for f in self.old_logs_files:
                try:
                    self.old_logs.append(pd.read_csv(f))
                    self.old_log_file_names.append(pathlib.Path(f).stem)
                except:
                    continue

        if save_file:
            self.save_file = save_file
        
        self.metrics = []
        self.logs = []
        
    def add(self, row, col, name, color=None, vmin=None, vmax=None, show_min=False, show_max=False):
        self.metrics.append({'row': row, 'col': col, 'name': name, 'color': color, 'vmin': vmin, 'vmax': vmax, 'show_min': show_min, 'show_max': show_max})
    
    def on_train_begin(self, logs={}):
        self.logs = []
        
        for m in self.metrics:
            m['values'] = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        
        clear_output(wait=True)
        plt.style.use("seaborn")
        # sns.set_style("whitegrid")
        fig, ax = plt.subplots(self.nrows, self.ncols, figsize=self.figsize)

        if len(ax.shape) == 1:
            ax = np.expand_dims(ax, axis=0)
        
        if self.title:
            fig.suptitle(self.title)
        
        for m in self.metrics:
            if logs.get(m['name']) is None:
                v = m['values']
                v.append(np.nan)
                continue
            
            a = ax[m['row'], m['col']]
            
            if m['name'] == 'off':
                a.axis('off')
                continue

            v = m['values']
            v.append(logs.get(m['name']))
                                
            # old logs
            for i, old_log in enumerate(self.old_logs):
                if m['name'] not in old_log:
                    continue
                old_values = old_log[m['name']].values
                a.plot(np.arange(len(old_values)), old_values, 
                     '-', 
                     color=m['color'], 
                    #  label=self.old_log_file_names[i], # gets crowded
                     alpha=0.2, lw=1)

            # new log
            a.plot(np.arange(len(v)), v, '-o', color=m['color'], label=m['name'], lw=1, markersize=3)
            a.set_xlabel('Epoch #', size=14)
            
            yname = m['name']
            if yname.startswith('val_'):
                yname = m['name'][4:]
            a.set_ylabel(yname, size=14)

            xdist = a.get_xlim()[1] - a.get_xlim()[0]
            ydist = a.get_ylim()[1] - a.get_ylim()[0]
            
            if ydist is not None and xdist is not None:
                if m['show_max']:
                    x = np.argmax(v)
                    y = np.max(v)
                    a.scatter(x, y, s=200, color=m['color'], alpha=0.5)
                    a.text(x-0.03*xdist, y-0.13*ydist, f'{round(y, 4)}', size=14)
                if m['show_min']:
                    x = np.argmin(v)
                    y = np.min(v)
                    a.scatter(x, y, s=200, color=m['color'], alpha=0.5)
                    a.text(x-0.03*xdist, y+0.05*ydist, f'{round(y, 4)}', size=14)

            if m['vmin'] is not None:
                a.set_ylim(m['vmin'], m['vmax'])

            a.legend()

        plt.show()

        if self.save_file:
            fig.savefig(self.save_file)

def create_plot(label):
    plot = TrainingPlot(nrows=1, ncols=2, figsize=(20, 5), title=label, 
                      save_file=D_WORK + f'{label}.png', 
                      old_logs_path=D_WORK + f'*.csv'
            )

    plot.add(0, 0, 'loss', 'green')
    plot.add(0, 0, 'val_loss', 'red', show_min=True)
    plot.add(0, 1, 'lr', 'black', vmin=0)
    
    return plot




def cvs_callback(filename):
    return tf.keras.callbacks.CSVLogger(filename)




config = {}

if ae_epochs > 0:
  base = get_base(config)
  ae_model = get_ae_model(base, config)
      
  ## TODO : simultaneous train
  for i in range(ae_epochs//ae_epochs_each):
      print(f"------ {i} ------")
      print("--- train ---")
      ae_model.fit([X_node, As], [X_node[:,0]],
                epochs = ae_epochs_each,
                batch_size = ae_batch_size * 8,
                callbacks=[create_plot(f'train-{i}'), cvs_callback(D_WORK + f'pretrain-train-{i}.csv')])
      print("--- public ---")
      ae_model.fit([X_node_pub, As_pub], [X_node_pub[:,0]],
                epochs = ae_epochs_each,
                batch_size = ae_batch_size * 8,
                callbacks=[create_plot(f'public-{i}'), cvs_callback(D_WORK + f'pretrain-public-{i}.csv')])
      print("--- private ---")
      ae_model.fit([X_node_pri, As_pri], [X_node_pri[:,0]],
                epochs = ae_epochs_each,
                batch_size = ae_batch_size * 8,
                callbacks=[create_plot(f'private-{i}'), cvs_callback(D_WORK + f'pretrain-private-{i}.csv')])
      gc.collect()
  print("****** save ae model ******")
  base.save_weights(D_WORK + "./base_ae.h5")




X_node.shape




n = X_node.shape[2]
fig, ax = plt.subplots(n, 3, figsize=(30, n * 3))

# check if input features in train and test look the same

for i in range(n):
    sns.distplot(X_node[:, :, i],color="Blue", ax=ax[i, 0])
    sns.distplot(X_node_pub[:, :, i],color="Green", ax=ax[i, 1])
    sns.distplot(X_node_pri[:, :, i],color="Red", ax=ax[i, 2])




from sklearn.model_selection import KFold
kfold = KFold(5, shuffle = True, random_state = 42)

scores = []
preds = np.zeros([len(X_node), X_node.shape[1], 5])


for i, (tr_idx, va_idx) in enumerate(kfold.split(X_node, As)):
    print(f"------ fold {i} start -----")
    print(f"------ fold {i} start -----")
    print(f"------ fold {i} start -----")
    X_node_tr = X_node[tr_idx]
    X_node_va = X_node[va_idx]
    As_tr = As[tr_idx]
    As_va = As[va_idx]
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    base = get_base(config)
    if ae_epochs > 0:
        print("****** load ae model ******")
        base.load_weights(D_WORK + "./base_ae.h5")
    model = get_model(base, config)
    if pretrain_dir is not None:
        d = D_WORK + f"./model{i}.h5"
        print(f"--- load from {d} ---")
        model.load_weights(d)
    for epochs, batch_size in zip(epochs_list, batch_size_list):
        print(f"epochs : {epochs}, batch_size : {batch_size}")
        model.fit([X_node_tr, As_tr], [y_tr],
                  validation_data=([X_node_va, As_va], [y_va]),
                  epochs = epochs,
                  batch_size = batch_size, 
                  callbacks=[
                    create_plot(f'train-F{i}-E{epochs}-B{batch_size}'), 
                    cvs_callback(D_WORK + f'train-F{i}-E{epochs}-B{batch_size}.csv'),
                    tf.keras.callbacks.ReduceLROnPlateau(factor=0.75, patience=3, min_lr=0.000001)
                  ],
                  # validation_freq = 3
                  )

    model.save_weights(D_WORK + f"./model{i}.h5")
    p = model.predict([X_node_va, As_va])
    scores.append(mcrmse(y_va, p))
    print(f"fold {i}: mcrmse {scores[-1]}")
    preds[va_idx] = p
    if one_fold:
        break

pd.to_pickle(preds, D_WORK + "oof.pkl")




print(scores)




p_pub = 0
p_pri = 0
for i in range(5):
    model.load_weights(D_WORK + f"./model{i}.h5")
    p_pub += model.predict([X_node_pub, As_pub]) / 5
    p_pri += model.predict([X_node_pri, As_pri]) / 5
    if one_fold:
        p_pub *= 5
        p_pri *= 5
        break

for i, target in enumerate(targets):
    test_pub[target] = [list(p_pub[k, :, i]) for k in range(p_pub.shape[0])]
    test_pri[target] = [list(p_pri[k, :, i]) for k in range(p_pri.shape[0])]




preds_ls = []
for df, preds in [(test_pub, p_pub), (test_pri, p_pri)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=targets)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_ls.append(single_df)

preds_df = pd.concat(preds_ls)
preds_df.to_csv(D_WORK + "submission.csv", index = False)
preds_df.head()




print(scores)
print(np.mean(scores))

