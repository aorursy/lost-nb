#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import tensorflow.keras.layers as L
import tensorflow as tf
import plotly.express as px
from sklearn.preprocessing import quantile_transform,StandardScaler,MinMaxScaler
from transformers import BertConfig,TFBertModel,BertModel


# In[2]:


AUTO = tf.data.experimental.AUTOTUNE
# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[3]:


# This will tell us the columns we are predicting
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


# In[4]:


config = BertConfig() 


# In[5]:


config.num_attention_heads


# In[6]:


def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)

def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True))

def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):
    ids = L.Input(shape=(seq_len,3), dtype=tf.int32)
    flat = L.Flatten()(ids)
    config = BertConfig() 
    config.vocab_size = 7
    config.num_hidden_layers = 3
    config.num_attention_heads = 1
    config.attention_probs_dropout_prob  = 0.5
    config.hidden_size = 120
    config.hidden_act= tf.sinh #tf.tanh
    bert_model = TFBertModel(config=config)

    bert_embeddings = bert_model(flat)[0]

    hidden = L.AveragePooling1D(pool_size=2)(bert_embeddings)
    # Since we are only making predictions on the first part of each sequence, we have
    # to truncate it
    truncated = hidden[:,:pred_len, :]
    out = L.Dense(5, activation='linear')(truncated)

    model = tf.keras.Model(inputs=ids, outputs=out)

    model.compile(tf.keras.optimizers.Adam(), loss=MCRMSE)
    
    return model


# In[7]:


vocab = {
    'sequence': {x:i for i, x in enumerate("A C G U".split())},
    'structure': {x:i for i, x in enumerate("( . )".split())},
    'predicted_loop_type': {x:i for i, x in enumerate("B E H I M S X".split())},
}
def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    
    def f(x):
        return [vocab['sequence'][x] for x in x[0]],                [vocab['structure'][x] for x in x[1]],                [vocab['predicted_loop_type'][x] for x in x[2]],

    return np.array(
            df[cols]
            .apply(f, axis=1)
            .values
            .tolist()
        )


# In[8]:


train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)
sample_df = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')


# In[9]:


print(pd.Series(list(train['structure'][0])).value_counts())
print(pd.Series(list(train['sequence'][0])).value_counts())
print(pd.Series(list(train['predicted_loop_type'][0])).value_counts())


# In[10]:


train.columns


# In[11]:


sorted(train['signal_to_noise'].apply(np.round).astype(int).unique())


# In[12]:


np.bincount(train['signal_to_noise'].apply(np.round).astype(int))


# In[13]:


sorted(train['SN_filter'].apply(np.round).astype(int).unique()) 


# In[14]:


np.bincount(train['SN_filter'].apply(np.round).astype(int))


# In[15]:


train = train.query("signal_to_noise >= 4")
train_inputs = preprocess_inputs(train)
train_labels = np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1))


# In[16]:


train_inputs.shape


# In[17]:


for df in [train,test]:
    df['Paired']=[sum([i=='(' or i==')' for i in j]) for j in df['structure']]
    df['Unpaired']=[sum([i=='.' for i in j]) for j in df['structure']]
    for col in ['E','S','H','I','G','A','U']:
        if col in ['E','S','H','I']:
            df[col]=[sum([i==col for i in j])/len(j) for j in df['predicted_loop_type']]
        else:
            df[col]=[sum([i==col for i in j])/len(j) for j in df['sequence']]
for a in [ 'G', 'A', 'C', 'U']:
    train[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['sequence']]
    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['sequence']]
for a in [ 'E', 'S', 'H',]:
    train[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['predicted_loop_type']]
    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['predicted_loop_type']]
for a in [ 'E', 'S', 'H',]:
    train[a+'']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['predicted_loop_type']]
    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['predicted_loop_type']]


# In[18]:


target_columns = ['reactivity', 'deg_Mg_pH10','deg_pH10', 'deg_Mg_50C', 'deg_50C']
target_columns.extend(['SN_filter', 'signal_to_noise'])
target_columns.extend(['deg_error_pH10', 'deg_error_Mg_50C', 'deg_error_50C', 'reactivity_error', 'deg_error_Mg_pH10'] )
train.drop(target_columns,axis=1,inplace=True)


# In[19]:


SC = MinMaxScaler(feature_range=(-1, 1))
train_measurements = SC.fit_transform(pd.concat((train.select_dtypes('float64'),train.select_dtypes('int64')),axis=1))


# In[20]:


train_measurements.shape


# In[21]:


np.min(train_measurements),np.max(train_measurements)
# np.min(test_measurements),np.max(test_measurements)


# In[22]:


pd.DataFrame(train_measurements).describe().T


# In[23]:


model = build_model()
model.summary()


# In[24]:


train_inputs.shape,train_labels.shape


# In[25]:


public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()
public_test_measurements  = SC.fit_transform(pd.concat((public_df.select_dtypes('float64'),public_df.select_dtypes('int64')),axis=1))
private_test_measurements  = SC.fit_transform(pd.concat((private_df.select_dtypes('float64'),private_df.select_dtypes('int64')),axis=1))
public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)


# In[26]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=5,shuffle=True,random_state=42)


# In[27]:


# with tf.device('/gpu'):
with strategy.scope():
    model = build_model()
    for fold,(idxT,idxV) in enumerate(kf.split(train_inputs)):
        history = model.fit(
            train_inputs[idxT,:,:], train_labels[idxT,:,:], 
            batch_size=64,
            epochs=100,
            validation_split=0.05,
                callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(),
            tf.keras.callbacks.ModelCheckpoint('model'+str(fold)+'.h5',save_weights_only=True,save_best_only=True)
        ]
        )
        # Caveat: The prediction format requires the output to be the same length as the input,
        # although it's not the case for the training data.
        model_short = build_model(seq_len=107, pred_len=107)
        model_long = build_model(seq_len=130, pred_len=130)

        model_short.load_weights('model'+str(fold)+'.h5')
        model_long.load_weights('model'+str(fold)+'.h5')
        
        if fold == 0:
            public_preds = model_short.predict([public_inputs])/5
            private_preds = model_long.predict([private_inputs])/5
        else:
            public_preds += model_short.predict([public_inputs])/5
            private_preds += model_long.predict([private_inputs])/5
            
        fig = px.line(
        history.history, y=['loss', 'val_loss'], 
        labels={'index': 'epoch', 'value': 'Mean Squared Error'}, 
        title='Training History')
        fig.show()


# In[28]:


#Bert's a bitch with weights
# model.save_weights('model.h5')


# In[29]:


print(public_preds.shape, private_preds.shape)


# In[30]:


preds_ls = []

for df, preds in [(public_df, public_preds), (private_df, private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=pred_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_ls.append(single_df)

preds_df = pd.concat(preds_ls)


# In[31]:


submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])
submission.to_csv('submission.csv', index=False)

