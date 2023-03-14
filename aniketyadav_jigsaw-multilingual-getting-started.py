#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, time
import pandas
import tensorflow as tf
import tensorflow_hub as hub
from kaggle_datasets import KaggleDatasets
print(tf.version.VERSION)


# In[2]:


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[3]:


SEQUENCE_LENGTH = 128

# Note that private datasets cannot be copied - you'll have to share any pretrained models 
# you want to use with other competitors!
GCS_PATH = KaggleDatasets().get_gcs_path('jigsaw-multilingual-toxic-comment-classification')
BERT_GCS_PATH = KaggleDatasets().get_gcs_path('bert-multi')
BERT_GCS_PATH_SAVEDMODEL = BERT_GCS_PATH + "/bert_multi_from_tfhub"


# In[4]:


def multilingual_bert_model(max_seq_length=SEQUENCE_LENGTH, trainable_bert=True):
    """Build and return a multilingual BERT model and tokenizer."""
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="all_segment_id")
    
    # Load a SavedModel on TPU from GCS. This model is available online at 
    # https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1. You can use your own 
    # pretrained models, but will need to add them as a Kaggle dataset.
    bert_layer = tf.saved_model.load(BERT_GCS_PATH_SAVEDMODEL)
    # Cast the loaded model to a TFHub KerasLayer.
    bert_layer = hub.KerasLayer(bert_layer, trainable=trainable_bert)

    pooled_output, _ = bert_layer([input_word_ids, input_mask, segment_ids])
    output = tf.keras.layers.Dense(32, activation='relu')(pooled_output)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='labels')(output)

    return tf.keras.Model(inputs={'input_word_ids': input_word_ids,
                                  'input_mask': input_mask,
                                  'all_segment_id': segment_ids},
                          outputs=output)


# In[5]:


def parse_string_list_into_ints(strlist):
    s = tf.strings.strip(strlist)
    s = tf.strings.substr(
        strlist, 1, tf.strings.length(s) - 2)  # Remove parentheses around list
    s = tf.strings.split(s, ',', maxsplit=SEQUENCE_LENGTH)
    s = tf.strings.to_number(s, tf.int32)
    s = tf.reshape(s, [SEQUENCE_LENGTH])  # Force shape here needed for XLA compilation (TPU)
    return s

def format_sentences(data, label='toxic', remove_language=False):
    labels = {'labels': data.pop(label)}
    if remove_language:
        languages = {'language': data.pop('lang')}
    # The remaining three items in the dict parsed from the CSV are lists of integers
    for k,v in data.items():  # "input_word_ids", "input_mask", "all_segment_id"
        data[k] = parse_string_list_into_ints(v)
    return data, labels

def make_sentence_dataset_from_csv(filename, label='toxic', language_to_filter=None):
    # This assumes the column order label, input_word_ids, input_mask, segment_ids
    SELECTED_COLUMNS = [label, "input_word_ids", "input_mask", "all_segment_id"]
    label_default = tf.int32 if label == 'id' else tf.float32
    COLUMN_DEFAULTS  = [label_default, tf.string, tf.string, tf.string]

    if language_to_filter:
        insert_pos = 0 if label != 'id' else 1
        SELECTED_COLUMNS.insert(insert_pos, 'lang')
        COLUMN_DEFAULTS.insert(insert_pos, tf.string)

    preprocessed_sentences_dataset = tf.data.experimental.make_csv_dataset(
        filename, column_defaults=COLUMN_DEFAULTS, select_columns=SELECTED_COLUMNS,
        batch_size=1, num_epochs=1, shuffle=False)  # We'll do repeating and shuffling ourselves
    # make_csv_dataset required a batch size, but we want to batch later
    preprocessed_sentences_dataset = preprocessed_sentences_dataset.unbatch()
    
    if language_to_filter:
        preprocessed_sentences_dataset = preprocessed_sentences_dataset.filter(
            lambda data: tf.math.equal(data['lang'], tf.constant(language_to_filter)))
        #preprocessed_sentences.pop('lang')
    preprocessed_sentences_dataset = preprocessed_sentences_dataset.map(
        lambda data: format_sentences(data, label=label,
                                      remove_language=language_to_filter))

    return preprocessed_sentences_dataset


# In[6]:


def make_dataset_pipeline(dataset, repeat_and_shuffle=True):
    """Set up the pipeline for the given dataset.
    
    Caches, repeats, shuffles, and sets the pipeline up to prefetch batches."""
    cached_dataset = dataset.cache()
    if repeat_and_shuffle:
        cached_dataset = cached_dataset.repeat().shuffle(2048)
    cached_dataset = cached_dataset.batch(32 * strategy.num_replicas_in_sync)
    cached_dataset = cached_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return cached_dataset

# Load the preprocessed English dataframe.
preprocessed_en_filename = (
    GCS_PATH + "/jigsaw-toxic-comment-train-processed-seqlen{}.csv".format(
        SEQUENCE_LENGTH))

# Set up the dataset and pipeline.
english_train_dataset = make_dataset_pipeline(
    make_sentence_dataset_from_csv(preprocessed_en_filename))

# Process the new datasets by language.
preprocessed_val_filename = (
    GCS_PATH + "/validation-processed-seqlen{}.csv".format(SEQUENCE_LENGTH))

nonenglish_val_datasets = {}
for language_name, language_label in [('Spanish', 'es'), ('Italian', 'it'),
                                      ('Turkish', 'tr')]:
    nonenglish_val_datasets[language_name] = make_sentence_dataset_from_csv(
        preprocessed_val_filename, language_to_filter=language_label)
    nonenglish_val_datasets[language_name] = make_dataset_pipeline(
        nonenglish_val_datasets[language_name])

nonenglish_val_datasets['Combined'] = tf.data.experimental.sample_from_datasets(
        (nonenglish_val_datasets['Spanish'], nonenglish_val_datasets['Italian'],
         nonenglish_val_datasets['Turkish']))


# In[7]:


with strategy.scope():
    multilingual_bert = multilingual_bert_model()

    # Compile the model. Optimize using stochastic gradient descent.
    multilingual_bert.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=[tf.keras.metrics.AUC()])

multilingual_bert.summary()


# In[8]:


# Test the model's performance on non-English comments before training.
for language in nonenglish_val_datasets:
    results = multilingual_bert.evaluate(nonenglish_val_datasets[language],
                                         steps=100, verbose=0)
    print('{} loss, AUC before training:'.format(language), results)

results = multilingual_bert.evaluate(english_train_dataset,
                                     steps=100, verbose=0)
print('\nEnglish loss, AUC before training:', results)

print()
# Train on English Wikipedia comment data.
history = multilingual_bert.fit(
    # Set steps such that the number of examples per epoch is fixed.
    # This makes training on different accelerators more comparable.
    english_train_dataset, steps_per_epoch=4000/strategy.num_replicas_in_sync,
    epochs=5, verbose=1, validation_data=nonenglish_val_datasets['Combined'],
    validation_steps=100)
print()

# Re-evaluate the model's performance on non-English comments after training.
for language in nonenglish_val_datasets:
    results = multilingual_bert.evaluate(nonenglish_val_datasets[language],
                                         steps=100, verbose=0)
    print('{} loss, AUC after training:'.format(language), results)

results = multilingual_bert.evaluate(english_train_dataset,
                                     steps=100, verbose=0)
print('\nEnglish loss, AUC after training:', results)


# In[9]:


import numpy as np

TEST_DATASET_SIZE = 63812

print('Making dataset...')
preprocessed_test_filename = (
    GCS_PATH + "/test-processed-seqlen{}.csv".format(SEQUENCE_LENGTH))
test_dataset = make_sentence_dataset_from_csv(preprocessed_test_filename, label='id')
test_dataset = make_dataset_pipeline(test_dataset, repeat_and_shuffle=False)

print('Computing predictions...')
test_sentences_dataset = test_dataset.map(lambda sentence, idnum: sentence)
probabilities = np.squeeze(multilingual_bert.predict(test_sentences_dataset))
print(probabilities)

print('Generating submission file...')
test_ids_dataset = test_dataset.map(lambda sentence, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_dataset.batch(TEST_DATASET_SIZE)))[
    'labels'].numpy().astype('U')  # All in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, probabilities]),
           fmt=['%s', '%f'], delimiter=',', header='id,toxic', comments='')
get_ipython().system('head submission.csv')


# In[10]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.datasets 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


# In[11]:


df=pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train-processed-seqlen128.csv',sep='\t',names=['toxic','comment_text'])


# In[12]:


df.head()


# In[13]:


len(df)


# In[14]:


len(df[df.toxic=='1'])


# In[15]:


df_x=df['comment_text']
df_y=df['toxic']


# In[16]:


df_x


# In[17]:


cv=CountVectorizer()


# In[18]:


x_train, x_test, y_train, y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=101)


# In[19]:


x_traincv=cv.fit_transform(['Hi how are you, how are you doing?',"Hey what's up"])


# In[20]:


x_traincv.toarray()


# In[21]:


cv.get_feature_names()


# In[22]:


cv1=CountVectorizer()


# In[23]:


x_traincv=cv1.fit_transform(x_train.values.astype('U'))


# In[24]:


a=x_traincv.toarray()


# In[25]:


cv=TfidfVectorizer(min_df=1,stop_words='english')


# In[26]:


cv1=TfidfVectorizer(min_df=1,stop_words='english')


# In[27]:


x_traincv=cv1.fit_transform(x_train.values.astype('U'))


# In[28]:


a=x_traincv.toarray()


# In[29]:


cv1.inverse_transform(a[0])


# In[30]:


a


# In[31]:


a[0]


# In[32]:


len(a[0])


# In[33]:


mnb=MultinomialNB()


# In[34]:


y_train=y_train.astype('int')


# In[35]:


import pandas as pd
messages=pd.read_csv("../input/new-jigsaw/trainnew.csv")
messages2=pd.read_csv("../input/new-jigsaw/validationnew.csv")


# In[36]:


print(len(messages))


# In[37]:


messages.head()


# In[38]:


messages_x=messages['comment_text']
messages_y=messages['toxic']


# In[39]:


messages_x


# In[40]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.datasets 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
cv=CountVectorizer()


# In[41]:


x_train, x_test, y_train, y_test=train_test_split(messages_x,messages_y,test_size=0.01,random_state=101)


# In[42]:


x_traincv=cv.fit_transform(x_train)
x_traincv


# In[43]:


# a=x_traincv.toarray()
a=x_traincv.toarray()

a


# In[44]:


cv=TfidfVectorizer(min_df=1,stop_words='english')


# In[45]:


x_traincv=cv.fit_transform(x_train)


# In[46]:


a=x_traincv.toarray()


# In[47]:


cv.inverse_transform(a[0])


# In[48]:


a


# In[49]:


mnb=MultinomialNB()


# In[50]:


y_train=messages2.astype('int')


# In[51]:


mnb.fit(x_traincv,messages2)


# In[52]:


mnb=MultinomialNB()


# In[53]:


y_train=y_train.astype('int')


# In[54]:


mnb.fit(x_traincv,y_train)


# In[55]:


messages2.shape


# In[56]:


messages.shape


# In[57]:


messages3=messages[['id','comment_text','toxic']]


# In[58]:


messages4=messages2[['id','comment_text','toxic']]


# In[59]:


messages3.head()


# In[60]:


messages4.head()


# In[61]:


messages3_x=messages['comment_text']
messages3_y=messages['toxic']


# In[62]:


x_train, x_test, y_train, y_test=train_test_split(messages3_x,messages3_y,test_size=.01,random_state=101)


# In[63]:


cv=TfidfVectorizer(min_df=1,stop_words='english')


# In[64]:


k=cv.fit_transform(x_train)


# In[65]:


a=k.toarray()


# In[66]:


cv.inverse_transform(a[0])


# In[67]:


a


# In[68]:


b=cv.fit_transform(messages4)


# In[69]:


mnb=MultinomialNB()


# In[ ]:





# In[70]:


one_hot_encoded_training_predictors = pd.get_dummies(messages3)
one_hot_encoded_test_predictors = pd.get_dummies(messages4)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)


# In[71]:


# X, y = check_X_y(
#             X, y, accept_sparse="csc", dtype=np.float32, multi_output=True)
mnb.fit()


# In[72]:


pip install -r requirements.txt


# In[73]:


messages3.shape


# In[74]:


messages4.shape


# In[ ]:




