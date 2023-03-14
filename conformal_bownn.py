#!/usr/bin/env python
# coding: utf-8



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
from keras.layers import Input, Dropout, Dense, concatenate, PReLU
from keras.optimizers import Adam
from keras.models import Model
from keras.initializers import he_uniform


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


def get_sample(train_file, test_file, max_text_length=60):
    # 读取样本
    print("读取样本")
    with Clock():
        train = pd.read_csv(train_file, sep='\t', encoding='utf-8')
        train.drop(columns=["train_id"], inplace=True)
        print('train shape:', train.shape, "columns:", train.columns)
        test = pd.read_csv(test_file, sep='\t', encoding='utf-8')
        test.drop(columns=["test_id"], inplace=True)
        print('test shape:', test.shape, "columns:", test.columns)
        train['price'] = np.log1p(train['price'].values)
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
    max_features = 200000
    with Clock():
        vectorizer = CountVectorizer(lowercase=True, tokenizer=tokenize, stop_words='english', 
                                     ngram_range=(1, 2), min_df=5, max_features=max_features, 
                                     binary=True, dtype=np.int8)
        vectorizer.fit(sample['name'] + ' ' + sample['item_description'])
        
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
    with Clock():
        split_num = train.shape[0]
        x = {
           'sparse_data': vectorizer.transform(sample[:split_num]['name'] + 
                                               ' ' + sample[:split_num]['item_description']),
           'item_condition': np.array(sample['item_condition_id'])[:split_num],
           'shipping': np.array(sample["shipping"])[:split_num],
           'mean_price_catery_name_second': np.array(sample["mean_price_catery_name_second"])[:split_num],
           'nb_words_item_description': np.array(sample["nb_words_item_description"])[:split_num]
              }
        test = {
           'sparse_data': vectorizer.transform(sample[split_num:]['name'] + 
                                               ' ' + sample[split_num:]['item_description']),
           'item_condition': np.array(sample['item_condition_id'])[split_num:],
           'shipping': np.array(sample["shipping"])[split_num:],
           'mean_price_catery_name_second': np.array(sample["mean_price_catery_name_second"])[split_num:],
           'nb_words_item_description': np.array(sample["nb_words_item_description"])[split_num:]
              }
        y = train.price.values
    return x, y, test 


def model(max_features=200000):
    sparse_data = Input(shape=[max_features], name="sparse_data", dtype='float32', sparse=True)
    item_condition = Input(shape=[1], name="item_condition", dtype='float32')
    shipping = Input(shape=[1], name="shipping", dtype='float32')
    mean_price_catery_name_second = Input(shape=[1], name="mean_price_catery_name_second", 
                                          dtype='float32')
    nb_words_item_description = Input(shape=[1], name="nb_words_item_description", 
                                          dtype='float32')
    x = Dense(200, kernel_initializer=he_uniform(seed=0))(sparse_data)
    x = PReLU()(x)
    x = concatenate([x, item_condition, shipping, mean_price_catery_name_second,
                    nb_words_item_description], axis=1)
    x = Dense(200, kernel_initializer=he_uniform(seed=0))(x)
    x = PReLU()(x)
    x = Dense(100, kernel_initializer=he_uniform(seed=0))(x)
    x = PReLU()(x)
    x = Dense(1, kernel_initializer=he_uniform(seed=0))(x)
    m = Model([sparse_data, item_condition, shipping, mean_price_catery_name_second, 
              nb_words_item_description], x)
    optimizer = Adam(.001)
    m.compile(loss="mse", optimizer=optimizer)
    return m




path = '../input/'
x, y, test = get_sample(path + "train.tsv", path + "test_stg2.tsv", max_text_length=60)




m = model()
m.summary()

m.fit(x, y - np.mean(y), 2048, epochs=3)

pred = m.predict(test, 2048) + np.mean(y)
pred = np.expm1(pred)
pred[pred < 3] = 3
pred[pred > 1000] = 1000
test_id= pd.read_csv(path + "test_stg2.tsv", sep='\t', encoding='utf-8')[["test_id"]]
test_id["price"] = pred
test_id.to_csv("ans.csv", index=False)






