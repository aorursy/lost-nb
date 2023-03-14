#!/usr/bin/env python
# coding: utf-8



# importing essentials
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import re
import warnings
warnings.filterwarnings('ignore')




# reading dataset
df_train = pd.read_csv('../input/google-quest-challenge/train.csv')
df_test = pd.read_csv('../input/google-quest-challenge/test.csv')
df_train.shape, df_test.shape




# download pretrained glove vectors : https://nlp.stanford.edu/projects/glove/
get_ipython().system('wget --header="Host: downloads.cs.stanford.edu" --header="User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9" "http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip" -c -O \'glove.6B.zip\'')




get_ipython().system("mkdir './glove/'")
get_ipython().system("mv './glove.6B.zip' './glove/'")
get_ipython().system("unzip './glove/glove.6B.zip' -d './glove/'")




# defining a function to remove stop_words
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.remove('no'); stop_words.remove('not'); stop_words.remove('nor')

def stopwrd_removal(sent):
  lst = []
  for wrd in sent.split():
    if wrd not in stop_words:
      lst.append(wrd)
  return " ".join(lst)




def text_preprocessor(column, remove_stopwords = False, remove_specialchar = False):
  """pass any column with Text in it from df_train | Note: returns nothing makes inplace changes in df_train"""
  # 1. remove html tags, html urls, replace html comparison operators
  # text = df_train[column].values
  df_train[column] = [re.sub('<.*?>', ' ', i) for i in df_train[column].values]
  df_train[column] = df_train[column].str.replace('&lt;', '<')                                          .str.replace('&gt;', '>')                                          .str.replace('&le;', '<=' )                                          .str.replace('&ge;', '>=')

  # 2. remove latex i,e., if there is any formulas or latex we have to remove it
  df_train[column] = [re.sub('\$.*?\$', ' ', i) for i in df_train[column].values]

  # 3. all lowercase 
  df_train[column] = df_train[column].str.lower()

  # 4. decontractions
  df_train[column] = df_train[column].str.replace("won't", "will not").str.replace("can\'t", "can not").str.replace("n\'t", " not").str.replace("\'re", " are").str.                                                replace("\'s", " is").str.replace("\'d", " would").str.replace("\'ll", " will").str.                                                replace("\'t", " not").str.replace("\'ve", " have").str.replace("\'m", " am")
  
  # 5. removing non-english or hebrew characters
  df_train[column] = [i.encode("ascii", "ignore").decode() for i in df_train[column].values]

  # 6. remove all special-characters other than alpha-numericals
  if remove_specialchar == True:
    df_train[column] = [re.sub('[^A-Za-z0-9]+', ' ', i) for i in df_train[column].values]

  # 7. separating special chars from alphanumerics
  all_sc = [re.findall('[^ A-Za-z0-9]', i) for i in df_train[column].values]
  special_char = np.unique([j for i in all_sc for j in i])
  replace_char = [' '+i+' ' for i in special_char]
  for i,j in zip(special_char, replace_char):
   df_train[column] = df_train[column].str.replace(i, j)

  # 8. Stop_word removal
  if remove_stopwords == True:
    df_train[column] = [stopwrd_removal(i) for i in df_train[column].values]

  # 9. remove all white-space i.e., \n, \t, and extra_spaces
  df_train[column] = df_train[column].str.replace("\n", " ").str.replace("\t", " ").str.rstrip()
  df_train[column] = [re.sub('  +', ' ', i) for i in df_train[column].values]




df_train['clean_title'] = df_train['question_title']
df_train['clean_body'] = df_train['question_body']
df_train['clean_answer'] = df_train['answer']
text_preprocessor('clean_title',  remove_stopwords = False, remove_specialchar = False)
text_preprocessor('clean_body',  remove_stopwords = False, remove_specialchar = False)
text_preprocessor('clean_answer',  remove_stopwords = False, remove_specialchar = False)




# 1. setting up target features
question_tar = ['question_asker_intent_understanding',
       'question_body_critical', 'question_conversational',
       'question_expect_short_answer', 'question_fact_seeking',
       'question_has_commonly_accepted_answer',
       'question_interestingness_others', 'question_interestingness_self',
       'question_multi_intent', 'question_not_really_a_question',
       'question_opinion_seeking', 'question_type_choice',
       'question_type_compare', 'question_type_consequence',
       'question_type_definition', 'question_type_entity',
       'question_type_instructions', 'question_type_procedure',
       'question_type_reason_explanation', 'question_type_spelling',
       'question_well_written']
       
answer_tar = ['answer_helpful',
       'answer_level_of_information', 'answer_plausible', 'answer_relevance',
       'answer_satisfaction', 'answer_type_instructions',
       'answer_type_procedure', 'answer_type_reason_explanation',
       'answer_well_written']

tar_features = question_tar + answer_tar
len(tar_features)




# 2. splitting dataset train_test_split
from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(df_train[['clean_title', 'clean_body', 'clean_answer']], df_train[tar_features], test_size = 0.12, random_state = 42)
X_train.shape, X_cv.shape, y_train.shape, y_cv.shape




# 3. creating training features : title + body = title_body | answer_train | title + body + answer = title_body_answer
title_train = X_train['clean_title'].values
body_train = X_train['clean_body'].values
answer_train = X_train['clean_answer'].values

title_cv = X_cv['clean_title'].values
body_cv = X_cv['clean_body'].values
answer_cv = X_cv['clean_answer'].values

# train data
title_body_train = [i+' '+j for i,j in zip(title_train, body_train)]

# cv data
title_body_cv = [i+' '+j for i,j in zip(title_cv, body_cv)]

len(title_body_train), len(answer_train), len(title_body_cv), len(answer_cv)




import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.keras.layers import Input, Softmax, GRU, LSTM, Conv1D, Embedding, Dense, RepeatVector, TimeDistributed, Bidirectional, Dropout, Concatenate
from tensorflow.keras.models import Model




# 2. tokenizing
from tensorflow.keras.preprocessing.text import Tokenizer
max_words = 20000

# M1 : title_body
title_body_token = Tokenizer(num_words = max_words, filters = '')
title_body_token.fit_on_texts(title_body_train)
title_body_seq_train = title_body_token.texts_to_sequences(title_body_train)
title_body_seq_cv = title_body_token.texts_to_sequences(title_body_cv)
# 3. bulding vocab
title_body_vocab = title_body_token.word_index

# M2 : answer
answer_token = Tokenizer(num_words = max_words, filters = '')
answer_token.fit_on_texts(answer_train)
answer_seq_train = answer_token.texts_to_sequences(answer_train)
answer_seq_cv = answer_token.texts_to_sequences(answer_cv)
# 3. bulding vocab
answer_vocab = answer_token.word_index




# 3. building vocab
print('Total no.of words in title_body vocab =', len(title_body_vocab))
print('Total no.of words in answer vocab =', len(answer_vocab))




# 4.1. padding : max lengths
from tensorflow.keras.preprocessing.sequence import pad_sequences
title_body_max_len = max([len(i) for  i in title_body_seq_train])
answer_max_len = max([len(i) for  i in answer_seq_train])
print('MAX seq_len in title_body sentences = {}\nMAX seq_len in answer sequences = {}'.format(title_body_max_len, answer_max_len))




# 4.2. padding : setting up sequence max_len threshold using percentile method (for padding the seq)
len_lst_1 = [len(i) for  i in title_body_seq_train]
for i in np.arange(80, 101, 1):
  print('percentile = {} | seq_len = {} | no.of datapts NOT covered = {}'.format(round(i, 2), round(np.percentile(len_lst_1, i)), round(len(len_lst_1) - (i*len(len_lst_1)*0.01))))

print('\n')
len_lst_2 = [len(i) for  i in answer_seq_train]
for i in np.arange(80, 101, 1):
  print('percentile = {} | seq_len = {} | no.of datapts NOT covered = {}'.format(round(i, 2), round(np.percentile(len_lst_2, i)), round(len(len_lst_2) - (i*len(len_lst_2)*0.01))))




# 4.3. padding : setting up sequence max_len threshold using elbow_method
lst_1 = []
lst_2 = []
lst_i = []
for i in np.arange(60, 100, 1):
  lst_i.append(round(i, 2))
  lst_1.append(round(np.percentile(len_lst_1, i)))
  lst_2.append(round(np.percentile(len_lst_2, i)))

plt.figure(figsize= (24, 5.5))
plt.subplot(1,2,1)
plt.plot(lst_1)
plt.grid()
plt.title('title_body len vs percentiles')
plt.xlabel('percentiles')
plt.ylabel('title_body seq length')
plt.xticks(ticks = range(0, 40), labels = lst_i)

plt.subplot(1,2,2)
plt.plot(lst_2)
plt.grid()
plt.title('answer len vs percentiles')
plt.xlabel('percentiles')
plt.ylabel('answer sequence length')
plt.xticks(ticks = range(0, 40), labels = lst_i)
plt.show()




# 4.4. padding : padding the train and test sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
title_body_seq_train = pad_sequences(title_body_seq_train, maxlen = 300, padding = 'post', truncating='post')
title_body_seq_cv = pad_sequences(title_body_seq_cv, maxlen = 300, padding = 'post', truncating='post')

answer_seq_train = pad_sequences(answer_seq_train, maxlen = 300, padding = 'post', truncating='post')
answer_seq_cv = pad_sequences(answer_seq_cv, maxlen = 300, padding = 'post', truncating='post')

title_body_seq_train.shape, title_body_seq_cv.shape, answer_seq_train.shape, answer_seq_cv.shape




# 5. initializing Embedding Matrix using glove_dict
# 5.1 glove_dict
file = open('./glove/glove.6B.100d.txt', encoding='utf8')
lst = []
for i in file:
    word, vec = i.split(maxsplit=1)
    vec = np.fromstring(vec, 'f', sep = ' ')
    lst.append(tuple([word, vec]))
glove_dict = dict(lst)
file.close()
print('Total no. of words :', len(glove_dict))




# 5.2 Embedding Matrix : question
max_words = 20000
embedding_matrix_text_body = np.random.normal(loc = 0, scale = 0.3, size = (max_words+1, 100))
for word, i in title_body_vocab.items():
    vector = glove_dict.get(word)
    if vector is not None and i <= max_words:
        embedding_matrix_text_body[i] = glove_dict[word]
print(embedding_matrix_text_body.shape)

# 5. Embedding Matrix : answer
embedding_matrix_answer = np.random.normal(loc = 0, scale = 0.3, size = (max_words+1, 100))
for word, i in answer_vocab.items():
    vector = glove_dict.get(word)
    if vector is not None and i <= max_words:
        embedding_matrix_answer[i] = glove_dict[word]
print(embedding_matrix_answer.shape)




# 6. Constructing a model
tf.keras.backend.clear_session()
def get_model(embedding_matrix_text_body, embedding_matrix_answer):
  # Creating 'Embedding layer'
  seed = 42
  title_body_embedding_layer = Embedding(input_dim = max_words+1, output_dim= 100, weights = [embedding_matrix_text_body],
                                        mask_zero = True, trainable = True, name = 'title_body_embed')

  answer_embedding_layer = Embedding(input_dim = max_words+1, output_dim= 100, weights = [embedding_matrix_answer],
                                    mask_zero = True, trainable = True, name = 'answer_embed') 
  # Title_body model
  tb_inputs = Input(name = 'title_body_seq', shape = (300,))
  tb_embed = title_body_embedding_layer(tb_inputs)
  tb_lstm, tb_hidden, tb_cell, tb_hidden_back, tb_cell_back = Bidirectional(LSTM(name = 'TB_ENCODER', units = 128, dropout = 0.25, return_sequences = True, return_state=True))(tb_embed)

  tb_dense_1 = Dense(units = 128, activation = 'relu', kernel_initializer = tf.keras.initializers.he_normal(seed = seed))(tf.concat([tb_hidden, tb_hidden_back], axis = -1))
  tb_dropout_1 = Dropout(rate = 0.2, seed = seed)(tb_dense_1)
  tb_out = Dense(units = 21, activation = 'sigmoid', kernel_initializer = tf.keras.initializers.GlorotNormal(seed = seed))(tb_dropout_1)

  # answer model
  ans_inputs = Input(name = 'ans_seq', shape = (300,))
  ans_embed = answer_embedding_layer(ans_inputs)
  ans_lstm, ans_hidden, ans_cell, ans_hidden_back, ans_cell_back = Bidirectional(LSTM(name = 'TB_ENCODER', units = 128, dropout = 0.2, return_sequences = True, return_state=True))(ans_embed)
  ans_dense_1 = Dense(units = 64, activation = 'relu', kernel_initializer = tf.keras.initializers.he_normal(seed = seed))(tf.concat([ans_hidden, ans_hidden_back], axis = -1))
  ans_dropout_1 = Dropout(rate = 0.2, seed = seed)(ans_dense_1)
  ans_out = Dense(units = 9, activation = 'sigmoid', kernel_initializer = tf.keras.initializers.GlorotNormal(seed = seed))(ans_dropout_1)

  concat_1 = Concatenate(axis = -1)([tb_out, ans_out])

  model = Model(inputs = [tb_inputs, ans_inputs] , outputs = concat_1)
  return model
model = get_model(embedding_matrix_text_body, embedding_matrix_answer)
model.summary()




# post processing : binning
def return_bins(arr):
  val = np.unique(arr)
  bins = []
  for i in range(len(val)):
    if i > 0:
      bins.append((val[i-1] + val[i])/2)
  return bins
  
unique_val_30 = [np.unique(df_train[tar_features].values[:, i]) for i in range(30)]
bins_30 = [return_bins(df_train[tar_features].values[:, i]) for i in range(30)]

def binned_out(y_pred):
  col = y_pred.shape[1]
  final_pred = np.zeros(y_pred.shape)
  for i in range(col):
    idx = np.digitize(y_pred[:, i], bins_30[i])
    final_pred[:, i] = unique_val_30[i][idx]
  return final_pred




# Defining callbacks
# !rm -r './saved model/'
# !rm -r './logs/'
get_ipython().system("mkdir './saved model/'")
get_ipython().system("mkdir './logs/'")

# tensorboard callback
import datetime
log_dir="./logs/" + datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq=1, write_graph=True, write_grads=True)

# spearman function
from scipy.stats import pearsonr, spearmanr
def compute_spearman(y_true, y_pred, final_pred):
  col = y_true.shape[1]
  lst = []
  for i in range(col):
    # p = round(spearmanr(y_true[:, i], y_pred[:, i])[0], 5)
    p = round(spearmanr(y_true[:, i], final_pred[:, i])[0], 5)
    p = round(p, 5)
    if np.isnan(p):
      p = round(spearmanr(y_true[:, i], y_pred[:, i])[0], 5)
    lst.append(p)
  return np.array(lst), round(sum(lst)/len(lst), 5)

# Custom spearman metric
class print_spearman(tf.keras.callbacks.Callback):
    def __init__(self, train_data, validation_data):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.x, self.y = train_data
        self.val_x, self.val_y = validation_data
    
    def on_train_begin(self, logs={}):
        self.all_feat_spearman = []
        self.spearman_dict = {'train_spearman' :[], 'val_spearman' :[]}

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch
        # 1. Test_set evaluation
        print('\nspearman :')
        y_pred = self.model.predict(x = self.x)
        y_pred_val = self.model.predict(x = self.val_x)

        final_pred = binned_out(y_pred)
        final_pred_val = binned_out(y_pred_val)

        train_spear_lst, train_spearman = compute_spearman(self.y, y_pred, final_pred)
        val_spear_lst, val_spearman = compute_spearman(self.val_y, y_pred_val, final_pred_val)

        self.all_feat_spearman.append({'train_spearman' : train_spear_lst, 'val_spearman' : val_spear_lst})

        self.spearman_dict['train_spearman'].append(train_spearman)
        self.spearman_dict['val_spearman'].append(val_spearman)
        prev_epoch_lr  = tf.keras.backend.eval(self.model.optimizer.lr)
        print("train_spearman : {} | val_spearman : {} | Learning_Rate : {}".format(train_spearman, val_spearman, round(prev_epoch_lr, 6)))
        print('train_spear_lst : ', train_spear_lst, '\n' 'val_spear_lst :', val_spear_lst)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor= 'val_loss', factor=np.sqrt(0.1), patience=4, verbose=1)

checkpt = tf.keras.callbacks.ModelCheckpoint('./saved model/{epoch:1d}', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True)

print_spearman_fn = print_spearman(train_data = ([title_body_seq_train, answer_seq_train], y_train.values),
                                 validation_data = ([title_body_seq_cv, answer_seq_cv], y_cv.values))
callbacks = [print_spearman_fn, reduce_lr, checkpt, tensorboard_callback]




# LSTM : training a model
tf.keras.backend.clear_session()
opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
rmse = tf.keras.metrics.RootMeanSquaredError()

model.compile(loss = 'binary_crossentropy', optimizer = opt,  metrics = [rmse])
history = model.fit(x = [title_body_seq_train, answer_seq_train], y =  y_train.values,
                    validation_data = ([title_body_seq_cv, answer_seq_cv], y_cv.values),
                    batch_size = 32, epochs = 20, callbacks = callbacks)




# plotting model graphs
plt.figure(figsize= (24, 5.5))
plt.subplot(1,3,1)
plt.plot(history.history['loss'], label = 'training loss')
plt.plot(history.history['val_loss'], label = 'validation loss')
plt.title('epochs vs loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.xticks(range(0, 20), range(1, 21))
plt.legend()
plt.grid()

plt.subplot(1,3,2)
plt.plot(history.history['root_mean_squared_error'], label = 'training rmse')
plt.plot(history.history['val_root_mean_squared_error'], label = 'validation rmse')
plt.title('epochs vs rmse')
plt.xlabel('epochs')
plt.ylabel('rmse')
plt.xticks(range(0, 20), range(1, 21))
plt.legend()
plt.grid()

plt.subplot(1,3,3)
plt.plot(print_spearman_fn.spearman_dict['train_spearman'], label = 'training spearman')
plt.plot(print_spearman_fn.spearman_dict['val_spearman'], label = 'validation spearman')
plt.title('epochs vs spearman')
plt.xlabel('epochs')
plt.ylabel('spearman')
plt.xticks(range(0, 20), range(1, 21))
plt.legend()
plt.grid()
plt.show()




def text_preprocessor(column, df, remove_stopwords = False, remove_specialchar = False):
  """pass any column with Text in it from df | Note: returns nothing makes inplace changes in df"""
  # 1. remove html tags, html urls, replace html comparison operators
  # text = df[column].values
  df[column] = [re.sub('<.*?>', ' ', i) for i in df[column].values]
  df[column] = df[column].str.replace('&lt;', '<')                                          .str.replace('&gt;', '>')                                          .str.replace('&le;', '<=' )                                          .str.replace('&ge;', '>=')

  # 2. remove latex i,e., if there is any formulas or latex we have to remove it
  df[column] = [re.sub('\$.*?\$', ' ', i) for i in df[column].values]

  # 3. all lowercase 
  df[column] = df[column].str.lower()

  # 4. decontractions
  df[column] = df[column].str.replace("won't", "will not").str.replace("can\'t", "can not").str.replace("n\'t", " not").str.replace("\'re", " are").str.                                                replace("\'s", " is").str.replace("\'d", " would").str.replace("\'ll", " will").str.                                                replace("\'t", " not").str.replace("\'ve", " have").str.replace("\'m", " am")
  
  # 5. removing non-english or hebrew characters
  df[column] = [i.encode("ascii", "ignore").decode() for i in df[column].values]

  # 6. remove all special-characters other than alpha-numericals
  if remove_specialchar == True:
    df[column] = [re.sub('[^A-Za-z0-9]+', ' ', i) for i in df[column].values]

  # 7. separating special chars from alphanumerics
  all_sc = [re.findall('[^ A-Za-z0-9]', i) for i in df[column].values]
  special_char = np.unique([j for i in all_sc for j in i])
  replace_char = [' '+i+' ' for i in special_char]
  for i,j in zip(special_char, replace_char):
   df[column] = df[column].str.replace(i, j)

  # 8. Stop_word removal
  if remove_stopwords == True:
    df[column] = [stopwrd_removal(i) for i in df[column].values]

  # 9. remove all white-space i.e., \n, \t, and extra_spaces
  df[column] = df[column].str.replace("\n", " ").str.replace("\t", " ").str.rstrip()
  df[column] = [re.sub('  +', ' ', i) for i in df[column].values]




# 1. text preprocessing
df_test['clean_title'] = df_test['question_title']
df_test['clean_body'] = df_test['question_body']
df_test['clean_answer'] = df_test['answer']
text_preprocessor('clean_title',df = df_test,  remove_stopwords = False, remove_specialchar = False)
text_preprocessor('clean_body',df = df_test,  remove_stopwords = False, remove_specialchar = False)
text_preprocessor('clean_answer',df = df_test,  remove_stopwords = False, remove_specialchar = False)




# 2. preparing input data
title_test = df_test['clean_title'].values
body_test = df_test['clean_body'].values
answer_test = df_test['clean_answer'].values

title_body_test = [i+' '+j for i,j in zip(title_test, body_test)]




# 3. tokenizing
title_body_seq_test = title_body_token.texts_to_sequences(title_body_test) # title + body
answer_seq_test = answer_token.texts_to_sequences(answer_test) # answer

# 4. padding
title_body_seq_test = pad_sequences(title_body_seq_test, maxlen = 300, padding = 'post', truncating='post')
answer_seq_test = pad_sequences(answer_seq_test, maxlen = 300, padding = 'post', truncating='post')




# 5. loading best model weights 
best_epoch = np.argmax(print_spearman_fn.spearman_dict['val_spearman'])+1
model.load_weights('./saved model/'+ str(best_epoch))

# 6. predicting unseen test set 
y_pred_test = model.predict([title_body_seq_test, answer_seq_test])




# 7. post_processing : binning
final_pred = binned_out(y_pred_test)




# 8. submission file 
pred_csv =  pd.DataFrame(final_pred, columns = tar_features)
id_df = pd.DataFrame(df_test['qa_id'].values, columns =['qa_id'])
submission_csv = pd.concat([id_df, pred_csv], axis=1)
submission_csv.to_csv('submission.csv', index=False)
submission_csv.head()

