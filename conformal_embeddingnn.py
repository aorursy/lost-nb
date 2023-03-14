#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt 
plt.style.use('dark_background')
import pandas as pd
import numpy as np
import seaborn as sns
import wordcloud
import pandas
from multiprocessing import Pool
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
import re
import gc
from nltk.util import ngrams
from sklearn import preprocessing
from itertools import combinations
from scipy.sparse import csr_matrix, hstack
import time
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dropout, Dense, concatenate, PReLU, Embedding, Flatten, Activation, BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.models import Model
from keras.initializers import he_uniform
from keras.preprocessing.sequence import pad_sequences
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)


class Clock:
    
    def __enter__(self):
        self.begin_time = time.time()
        
    def __exit__(self, exec_type, exec_value, exec_trace):
        print("time:", time.time() - self.begin_time, "s")


def parallelize_dataframe(df, func, cores=4):
    """
    对一个dataframe并行计算, df是一个dataframe, func是处理一个dataframe的函数, 而不是
    处理某一列的函数, 如果我们要把处理某一列的函数封装成func, 需要在用一个处理dataframe的
    函数对这个处理单列的函数进行封装, 最后会返回这个处理完的dataframe
    """
    # 分割data frame, array_split用来在垂直方向上分割dataframe
    df_split = np.array_split(df, cores)
    pool = Pool(cores)
    
    # 并行计算, 最终结果是按照df_split的顺序返回的, 并在axis=0, 也就是垂直进行连接
    df = pd.concat(pool.map(func, df_split), axis=0)
    pool.close()
    pool.join()
    
    return df


def clean_str(text, max_text_length=60):
    """
    去掉特殊字符, 只保留数字和字母, 并且把数字和字母分开, 然后用空格分割各个单词
    处理如果我们的文本超过了max_text_length, 就会被截断成max_text_length.
    """
    try:
        # 进行切分后合并, 最多取60个词
        text = ' '.join([w for w in text.split()[:max_text_length]] )        
        text = text.lower()
        # 特殊字符
        text = re.sub(u"é", u"e", text)
        text = re.sub(u"ē", u"e", text)
        text = re.sub(u"è", u"e", text)
        text = re.sub(u"ê", u"e", text)
        text = re.sub(u"à", u"a", text)
        text = re.sub(u"â", u"a", text)
        text = re.sub(u"ô", u"o", text)
        text = re.sub(u"ō", u"o", text)
        text = re.sub(u"ü", u"u", text)
        text = re.sub(u"ï", u"i", text)
        text = re.sub(u"ç", u"c", text)
        text = re.sub(u"\u2019", u"'", text)
        text = re.sub(u"\xed", u"i", text)
        # 简写
        text = re.sub(u"w\/", u" with ", text)
        
        # 去掉非数字以及字母符
        text = re.sub(u"[^a-z0-9]", " ", text)
        # 如果pattern中含有括号, 那么括号中匹配的也能被保留下来, 将数字与字母分开
        text = u" ".join(re.split('(\d+)',text))
        # 去掉任何空白字符, 转换为单个空格
        text = re.sub( u"\s+", u" ", text).strip()
    except:
        text = np.NaN
        
    return text

def take(generator):
    try:
        for elem in generator:
            yield elem
    except StopIteration:
        return


def fill_brandname(name, brand_name_set):
    try:
        for size in [4, 3, 2, 1]:
            for x in take(ngrams(name.split(), size)):
                if " ".join(x) in brand_name_set:
                    return " ".join(x)
        return np.NaN
    except:
        return np.NaN

    
def clean_str_df(x):
    return x.apply(lambda y: clean_str(y))


def fill_brandname_df(x):
    # x是dataframe
    global brand_name_set
    # 返回的也应该是一个dataframe
    return x.apply(lambda y: fill_brandname(y, brand_name_set))


def lg(text):
    text = [x for x in text.split() if x != '']
    return len(text)


def tokenize(text):
    return [w for w in text.split()]


def word_doc_counts(words, word_doc_count_dic):
    # 文档内去重
    words = set(words.split())
    for x in words:
        if x in word_doc_count_dic:
            word_doc_count_dic[x] = word_doc_count_dic[x] + 1
        else:
            word_doc_count_dic[x] = 1
            
            
def remove_rare_words(words, word_doc_count_dic):
    return " ".join([x for x in words.split() if x in word_doc_count_dic])


def name_to_seq(df):
    global vocabulary_one
    global MAX_NAME_SEQ
    return df.apply(lambda x: [vocabulary_one[y] for y in x.split()[:MAX_NAME_SEQ]])


def description_to_seq(df):
    global vocabulary_one
    global MAX_ITEM_DESC_SEQ
    ret = df.apply(lambda x: [vocabulary_one[y] for y in x.split()[:MAX_ITEM_DESC_SEQ]])
    return ret


# 为了节约内存, 我们在获取test的时候, 采用分批次的方式
def get_sample(train_file, test_file, max_text_length=60):
    # 读取样本
    print("读取样本")
    with Clock():
        train = pd.read_csv(train_file, sep='\t', encoding='utf-8')
        train.drop(columns=["train_id"], inplace=True)
        train["price"] = np.log1p(train["price"].values)
        print('train shape:', train.shape, "columns:", train.columns)
        test = pd.read_csv(test_file, sep='\t', encoding='utf-8')
        test.drop(columns=["test_id"], inplace=True)
        print('test shape:', test.shape, "columns:", test.columns)
        sample = pd.concat([train.drop(columns=["price"], axis=0), test], axis=0)
        gc.collect()

    with Clock():
        print("处理特征")
        # 对item_description进行填充
        sample["item_description"].fillna("", inplace=True) #na
        # No description yet
        sample["item_description"] = sample["item_description"].apply(lambda x : 
                                                                     x.replace('No description yet',''))

        # 清理字符串
        sample["item_description"] = parallelize_dataframe(sample["item_description"], clean_str_df)
        sample["name"] = parallelize_dataframe(sample["name"], clean_str_df)
        sample["brand_name"] = parallelize_dataframe(sample["brand_name"], clean_str_df)

        # 获取brand_name和name大小的对应关系
        global brand_name_set 
        brand_name_set = sample.groupby("brand_name").size()
        # 在name中抽取brand name进行对brand name的填充
        sample.loc[sample["brand_name"].isnull(), "brand_name"] = parallelize_dataframe(
            sample.loc[sample["brand_name"].isnull(), "name"], 
            fill_brandname_df)
        # 填充na
        sample["brand_name"].fillna("", inplace=True)
        sample["item_condition_id"].fillna(2, inplace=True)

        # 填充category_name
        sample["category_name"].fillna("//", inplace=True)
        sample["catery_name_first"] =  sample["category_name"].apply(lambda x: x.split("/")[0])
        sample["catery_name_second"] = sample["category_name"].apply(lambda x: x.split("/")[1])
        sample["catery_name_third"] = sample["category_name"].apply(lambda x: x.split("/")[2])
        sample['category_name'] = sample['category_name'].apply( lambda x: ' '.join(x.split('/') ).strip())

        sample['nb_words_item_description'] = sample['item_description'].apply(lg).astype(np.uint16)
        sample['nb_words_item_description'] = 1.0 * sample['nb_words_item_description'] / max_text_length

        for x in ['brand_name', 'category_name', 'catery_name_first', 'catery_name_second', 
                  'catery_name_third']:
            le = preprocessing.LabelEncoder()
            sample[x] = le.fit_transform(sample[x])

        sample["item_condition_id"] = 1.0 * sample["item_condition_id"] / 5
        sample["old_name"] = sample["name"].copy()
        sample["brand_cat"] = "cat1_" + sample["catery_name_first"].astype(str) + " " +                     "cat2_" + sample["catery_name_second"].astype(str) + " " +                     "cat3_" +  sample["catery_name_third"].astype(str) + " " +                     "brand_" + sample["brand_name"].astype(str)
        sample["name"] = sample["brand_cat"] + " " + sample["name"]
        # 融合了cate, brand, name, desc
        sample["name_desc"] = sample["name"] + " " + sample["item_description"].apply(lambda x: " ".join(x.split()[:5]))  
        gc.collect()

    print("构造文本特征:")
    with Clock():
        word_doc_count_dic = dict()
        for x in ['name','item_description']:
            sample[x].apply(lambda y: word_doc_counts(y, word_doc_count_dic))
        min_df_one = 5
        rare_words = [x for x in word_doc_count_dic if word_doc_count_dic[x] < min_df_one]
        for x in rare_words:
            del word_doc_count_dic[x]
        for x in ['name','item_description']:
            sample[x] = sample[x].apply(lambda y: remove_rare_words(y, word_doc_count_dic))
        global vocabulary_one 
        vocabulary_one = word_doc_count_dic.copy()
        count = 0
        for x in word_doc_count_dic:
            count += 1
            vocabulary_one[x] = count

    print("构建统计特征:")
    with Clock():
        train_new = sample[:train.shape[0]].copy()
        train_new["price"] = train["price"]
        for x in ['catery_name_first', 'catery_name_second', 'catery_name_third', 'category_name', 'brand_name'  ]:
            tmp = train_new.groupby(x)["price"].mean().astype(np.float32)
            sample['mean_price_' + x] = sample[x].map(tmp).astype(np.float32)
            sample['mean_price_' + x].fillna(tmp.mean(), inplace=True)
        gc.collect()

    print("构造样本:")
    # 定义一些参数
    global MAX_NAME_SEQ 
    MAX_NAME_SEQ = 20
    global MAX_ITEM_DESC_SEQ 
    MAX_ITEM_DESC_SEQ = 30
    # 0是默认值
    MAX_TEXT = len(vocabulary_one) + 1
    # 加1是因为从0开始计算, 这些都是要embedding
    MAX_CATEGORY    = np.max(sample['category_name'].max()) + 1
    MAX_CATEGORY1   = np.max(sample['catery_name_first'].max()) + 1
    MAX_CATEGORY2   = np.max(sample['catery_name_second'].max()) + 1
    MAX_CATEGORY3   = np.max(sample['catery_name_third'].max()) + 1
    MAX_BRAND       = np.max(sample['brand_name'].max()) + 1
    split_num = train.shape[0]
        
 
    with Clock():
        new_train = sample[:split_num]
        new_train["seq_name"] = parallelize_dataframe(new_train["name"], name_to_seq)
        new_train["seq_item_description"] = parallelize_dataframe(new_train["item_description"],
                                                                  description_to_seq)
        seq_name = pad_sequences(new_train["seq_name"], maxlen=MAX_NAME_SEQ)
        seq_item_description = pad_sequences(new_train["seq_item_description"], 
                                             maxlen=MAX_ITEM_DESC_SEQ)
        x = {
            "name": seq_name,
            "item_desc": seq_item_description,
            "brand_name": np.array(new_train["brand_name"]),
            "category_name": np.array(new_train["category_name"]),
            "catery_name_first": np.array(new_train["catery_name_first"]),
            "catery_name_second": np.array(new_train["catery_name_second"]),
            "catery_name_third": np.array(new_train["catery_name_third"]),
            "item_condition": np.array(new_train["item_condition_id"]),
            "shipping": np.array(new_train["shipping"])
            }
        y = train.price.values
        new_test = sample[split_num:]
        
        def test_gen():
            batch = 500000
            remain = new_test.shape[0]
            current = 0
            while remain > 0:
                if remain < batch:
                    tmp = new_test[current:]
                else:
                    tmp = new_test[current:current + batch]

                remain = remain - batch
                current = current + batch

                tmp["seq_name"] = parallelize_dataframe(tmp["name"], name_to_seq)
                tmp["seq_item_description"] = parallelize_dataframe(tmp["item_description"],
                                                                          description_to_seq)
                seq_name = pad_sequences(tmp["seq_name"], maxlen=MAX_NAME_SEQ)
                seq_item_description = pad_sequences(tmp["seq_item_description"], 
                                                     maxlen=MAX_ITEM_DESC_SEQ)
                x = {
                    "name": seq_name,
                    "item_desc": seq_item_description,
                    "brand_name": np.array(tmp["brand_name"]),
                    "category_name": np.array(tmp["category_name"]),
                    "catery_name_first": np.array(tmp["catery_name_first"]),
                    "catery_name_second": np.array(tmp["catery_name_second"]),
                    "catery_name_third": np.array(tmp["catery_name_third"]),
                    "item_condition": np.array(tmp["item_condition_id"]),
                    "shipping": np.array(tmp["shipping"])
                  }
                yield x

        return x, y, test_gen, MAX_CATEGORY, MAX_CATEGORY1, MAX_CATEGORY2, MAX_CATEGORY3, MAX_BRAND, MAX_TEXT, MAX_NAME_SEQ, MAX_ITEM_DESC_SEQ
  
        
def model(MAX_CATEGORY, MAX_CATEGORY1, MAX_CATEGORY2, MAX_CATEGORY3, MAX_BRAND, MAX_TEXT,
         MAX_NAME_SEQ):
    name = Input(shape=[MAX_NAME_SEQ], name="name")
    item_desc = Input(shape=[MAX_ITEM_DESC_SEQ], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name="category_name")
    catery_name_first = Input(shape=[1], name="catery_name_first")
    catery_name_second = Input(shape=[1], name="catery_name_second")
    catery_name_third = Input(shape=[1], name="catery_name_third")
    item_condition = Input(shape=[1], name="item_condition")
    shipping = Input(shape=[1], name="shipping")
    
    
    share_embedding = Embedding(MAX_TEXT, 50)
    # [batch, MAX_NAME_SEQ, 50]
    name_pool = GlobalAveragePooling1D()(share_embedding(name))
    # [batch, MAX_ITEM_DESC_SEQ, 50]
    item_desc_pool = GlobalAveragePooling1D()(share_embedding(item_desc))
    
    brand_name_emb = Flatten()(Embedding(MAX_BRAND, 10)(brand_name))
    category_name_emb = Flatten()(Embedding(MAX_CATEGORY, 10)(category_name))
    catery_name_first_emb = Flatten()(Embedding(MAX_CATEGORY1, 10)(catery_name_first))
    catery_name_second_emb = Flatten()(Embedding(MAX_CATEGORY2, 10)(catery_name_second))
    catery_name_third_emb = Flatten()(Embedding(MAX_CATEGORY3, 10)(catery_name_third))
    x = concatenate([name_pool, item_desc_pool, brand_name_emb, category_name_emb, 
                     catery_name_first_emb, catery_name_second_emb, catery_name_third_emb,
                    item_condition, shipping])
    x = BatchNormalization()(x)
    x = Dense(1024)(x)
    x = Activation("relu")(x)
    x = Dense(1)(x)
    model = Model([name, item_desc, brand_name, category_name, catery_name_first, 
                  catery_name_second, catery_name_third, item_condition, shipping], x)
    model.compile(loss="mse", optimizer=Adam(0.001))
    return model


# In[2]:


path = "../input/"
x, y, test_gen, MAX_CATEGORY, MAX_CATEGORY1, MAX_CATEGORY2, MAX_CATEGORY3, MAX_BRAND, MAX_TEXT, MAX_NAME_SEQ, MAX_ITEM_DESC_SEQ = get_sample(path + "train.tsv", path + "test_stg2.tsv", max_text_length=60)
model = model(MAX_CATEGORY, MAX_CATEGORY1,
              MAX_CATEGORY2, MAX_CATEGORY3, 
              MAX_BRAND, MAX_TEXT, MAX_NAME_SEQ)
model.summary()
model.fit(x, y - np.mean(y), 2048, epochs=3)
del x
gc.collect()

pred = []
for test in test_gen():
    tmp = model.predict(test, 2048) + np.mean(y)
    pred.append(tmp)
    del test
    gc.collect()
pred = np.concatenate(pred)
pred = np.expm1(pred)
pred[pred < 3] = 3
pred[pred > 1000] = 1000
test_id= pd.read_csv(path + "test_stg2.tsv", sep='\t', encoding='utf-8')[["test_id"]]
test_id["price"] = pred
test_id.to_csv("ans.csv", index=False)


# In[ ]:




