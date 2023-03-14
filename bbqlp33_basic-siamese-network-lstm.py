#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import jieba.posseg as pseg
import os
import keras
print(os.listdir("../input"))
print(os.listdir("../input/fake-news-pair-classification-challenge"))
print(os.listdir("../input/apply-jieba-tokenizer"))

# Any results you write to the current directory are saved as output.




# ! pip install opencc-python-reimplemented




# from opencc import OpenCC
import pandas as pd




if os.path.isdir("../input/fake-news-pair-classification-challenge"):
    TRAIN_CSV_PATH = '../input/fake-news-pair-classification-challenge/train.csv'
    TEST_CSV_PATH = '../input/fake-news-pair-classification-challenge/test.csv'
    TOKENIZED_TRAIN_CSV_PATH = "../input/apply-jieba-tokenizer/tokenized_train.csv"
    TOKENIZED_TEST_CSV_PATH = "../input/apply-jieba-tokenizer/tokenized_test.csv"
else:
    TRAIN_CSV_PATH = '../input/train.csv'
    TEST_CSV_PATH = '../input/test.csv'
    TOKENIZED_TRAIN_CSV_PATH = None




train = pd.read_csv(TRAIN_CSV_PATH, index_col='id')
train.head(3)




cols = ['title1_zh', 
        'title2_zh', 
        'label']
train = train.loc[:, cols]
train.head(3)




text = '我是台中人，但是我在板橋上班'
words = pseg.cut(text)
[word for word in words]




def jieba_tokenizer(text):
    words = pseg.cut(text)
    return ' '.join([
        word for word, flag in words if flag != 'x'])




train.isna().any()




train.title2_zh.fillna('UNKNOWN', inplace=True)
train.isna().any()




def process(data):
    res = data.apply(jieba_tokenizer)
    return res


def check_merge_idx(data, res):
    assert((data.index == res.index).all(), 'Something error when merge data')

def parallelize(data, func):
    from multiprocessing import cpu_count, Pool
    cores = partitions = cpu_count()
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    res = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    check_merge_idx(data, res)
    return res




np.all(train.index == train.title1_zh.index)




if os.path.exists(TOKENIZED_TRAIN_CSV_PATH):
    print("Use prepared tokenized train data")
    train = pd.read_csv(TOKENIZED_TRAIN_CSV_PATH, index_col='id')
else:
    print("start to training")
    train['title1_tokenized'] = parallelize(train.loc[:, 'title1_zh'], process)
    train['title2_tokenized'] = parallelize(train.loc[:, 'title2_zh'], process)
    train.to_csv('tokenized_train.csv',index=True)




train.loc[:, ["title1_zh", "title1_tokenized"]].head(10)




train.loc[:, ["title2_zh", "title2_tokenized"]].head(10)




train.fillna('UNKNOWN', inplace=True)





MAX_NUM_WORDS = 10000
tokenizer = keras     .preprocessing     .text     .Tokenizer(num_words=MAX_NUM_WORDS)




corpus_x1 = train.title1_tokenized
corpus_x2 = train.title2_tokenized
corpus = pd.concat([
    corpus_x1, corpus_x2])
corpus.shape




pd.DataFrame(corpus.iloc[:5],
             columns=['title'])




corpus.isna().any()




tokenizer.fit_on_texts(corpus)
x1_train = tokenizer     .texts_to_sequences(corpus_x1)
x2_train = tokenizer     .texts_to_sequences(corpus_x2)




len(x1_train)




x1_train[:1]




for seq in x1_train[:1]:
    print([tokenizer.index_word[idx] for idx in seq])




MAX_SEQUENCE_LENGTH = 20
x1_train = keras     .preprocessing     .sequence     .pad_sequences(x1_train, 
                   maxlen=MAX_SEQUENCE_LENGTH)

x2_train = keras     .preprocessing     .sequence     .pad_sequences(x2_train, 
                   maxlen=MAX_SEQUENCE_LENGTH)




x1_train[0]




for seq in x1_train + x2_train:
    assert len(seq) == 20
    
print("所有新聞標題的序列長度皆為 20 !")




train.label[:5]




import numpy as np 

# 定義每一個分類對應到的索引數字
label_to_index = {
    'unrelated': 0, 
    'agreed': 1, 
    'disagreed': 2
}

# 將分類標籤對應到剛定義的數字
y_train = train.label.apply(
    lambda x: label_to_index[x])

y_train = np.asarray(y_train)             .astype('float32')

y_train[:5]




# 基本參數設置，有幾個分類
NUM_CLASSES = 3

# 在語料庫裡有多少詞彙
MAX_NUM_WORDS = 10000

# 一個標題最長有幾個詞彙
MAX_SEQUENCE_LENGTH = 20

# 一個詞向量的維度
NUM_EMBEDDING_DIM = 256

# LSTM 輸出的向量維度
NUM_LSTM_UNITS = 128




x1_train[:5]




train.label[:5]




y_train = keras     .utils     .to_categorical(y_train)

y_train[:5]




from sklearn.model_selection     import train_test_split

VALIDATION_RATIO = 0.1
# 小彩蛋
RANDOM_STATE = 9527

x1_train, x1_val, x2_train, x2_val, y_train, y_val =     train_test_split(
        x1_train, x2_train, y_train, 
        test_size=VALIDATION_RATIO, 
        random_state=RANDOM_STATE
)




print("Training Set")
print("-" * 10)
print(f"x1_train: {x1_train.shape}")
print(f"x2_train: {x2_train.shape}")
print(f"y_train : {y_train.shape}")

print("-" * 10)
print(f"x1_val:   {x1_val.shape}")
print(f"x2_val:   {x2_val.shape}")
print(f"y_val :   {y_val.shape}")
print("-" * 10)
print("Test Set")




# 建立孿生 LSTM 架構（Siamese LSTM）
from keras import Input
from keras.layers import Embedding,LSTM, concatenate, Dense
from keras.models import Model

# 分別定義 2 個新聞標題 A & B 為模型輸入
# 兩個標題都是一個長度為 20 的數字序列
top_input = Input(
    shape=(MAX_SEQUENCE_LENGTH, ), 
    dtype='int32')
bm_input = Input(
    shape=(MAX_SEQUENCE_LENGTH, ), 
    dtype='int32')

# 詞嵌入層
# 經過詞嵌入層的轉換，兩個新聞標題都變成
# 一個詞向量的序列，而每個詞向量的維度
# 為 256
embedding_layer = Embedding(
    MAX_NUM_WORDS, NUM_EMBEDDING_DIM)
top_embedded = embedding_layer(
    top_input)
bm_embedded = embedding_layer(
    bm_input)

# LSTM 層
# 兩個新聞標題經過此層後
# 為一個 128 維度向量
shared_lstm = LSTM(NUM_LSTM_UNITS)
top_output = shared_lstm(top_embedded)
bm_output = shared_lstm(bm_embedded)

# 串接層將兩個新聞標題的結果串接單一向量
# 方便跟全連結層相連
merged = concatenate(
    [top_output, bm_output], 
    axis=-1)

# 全連接層搭配 Softmax Activation
# 可以回傳 3 個成對標題
# 屬於各類別的可能機率
dense =  Dense(
    units=NUM_CLASSES, 
    activation='softmax')
predictions = dense(merged)

# 我們的模型就是將數字序列的輸入，轉換
# 成 3 個分類的機率的所有步驟 / 層的總和
model = Model(
    inputs=[top_input, bm_input], 
    outputs=predictions)

model.summary()




from keras.utils import plot_model
import matplotlib.pyplot as plt
plot_model(
    model, 
    to_file='model.png', 
    show_shapes=True, 
    show_layer_names=False, 
    rankdir='LR')

from IPython.display import SVG
from keras.utils import model_to_dot
SVG(model_to_dot(model, rankdir='LR', show_shapes=True, show_layer_names=False,).create(prog='dot', format='svg'))




from keras.optimizers import Adam




lr = 1e-3
opt = Adam(lr=lr, decay=lr/50)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])




x1_train[:9527].shape




# 決定一次要放多少成對標題給模型訓練
BATCH_SIZE = 512

# 決定模型要看整個訓練資料集幾遍
NUM_EPOCHS = 50

# 實際訓練模型
history = model.fit(
    # 輸入是兩個長度為 20 的數字序列
    x=[x1_train, x2_train], 
    y=y_train,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    # 每個 epoch 完後計算驗證資料集
    # 上的 Loss 以及準確度
    validation_data=(
        [x1_val, x2_val], 
        y_val
    ),
    # 每個 epoch 隨機調整訓練資料集
    # 裡頭的數據以讓訓練過程更穩定
    shuffle=True
)




import pandas as pd
if os.path.exists(TOKENIZED_TEST_CSV_PATH):
    print("Use tokenized test csv")
    test = pd.read_csv(TOKENIZED_TEST_CSV_PATH, index_col=0)
else:
    print("Use raw test csv")
    test = pd.read_csv(TEST_CSV_PATH, index_col=0)
    test.fillna('UNKNOWN', inplace=True)
    test['title1_tokenized'] = parallelize(test.loc[:, 'title1_zh'], process)
    test['title2_tokenized'] = parallelize(test.loc[:, 'title2_zh'], process)
    test.fillna('UNKNOWN', inplace=True)
test.head(3)





# 將詞彙序列轉為索引數字的序列
x1_test = tokenizer     .texts_to_sequences(
        test.title1_tokenized)
x2_test = tokenizer     .texts_to_sequences(
        test.title2_tokenized)

# 為數字序列加入 zero padding
x1_test = keras     .preprocessing     .sequence     .pad_sequences(
        x1_test, 
        maxlen=MAX_SEQUENCE_LENGTH)
x2_test = keras     .preprocessing     .sequence     .pad_sequences(
        x2_test, 
        maxlen=MAX_SEQUENCE_LENGTH)    

# 利用已訓練的模型做預測
predictions = model.predict(
    [x1_test, x2_test])




predictions[:5]




index_to_label = {v: k for k, v in label_to_index.items()}

test['Category'] = [index_to_label[idx] for idx in np.argmax(predictions, axis=1)]

submission = test     .loc[:, ['Category']]     .reset_index()

submission.columns = ['Id', 'Category']
submission.to_csv('submission.csv', index=False)
submission.head()

