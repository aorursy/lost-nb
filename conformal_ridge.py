#!/usr/bin/env python
# coding: utf-8



import matplotlib.pyplot as plt 
plt.style.use('dark_background')
import pandas as pd
import numpy as np
import seaborn as sns
import wordcloud
import pandas
from multiprocessing import Pool
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
from nltk.corpus import stopwords
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


def fill_brandname(name, brand_name_set):
    try:
        for size in [4, 3, 2, 1]:
            for x in ngrams(name.split(), size):
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


def word_doc_counts(words, word_doc_count_dic):
    # 文档内去重
    words = set(words.split())
    for x in words:
        if x in word_doc_count_dic:
            word_doc_count_dic[x] = word_doc_count_dic[x] + 1
        else:
            word_doc_count_dic[x] = 1

            
def remove_rare_words(words, word_doc_count_dic):
    stops = set(stopwords.words("english"))
    return " ".join([x for x in words.split() if x in word_doc_count_dic and x not in stops])


def lemmatize_all(sentence):
    ans = []
    stops = set(stopwords.words("english"))
    
    wnl = WordNetLemmatizer()
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag.startswith('NN'):
            new_word = wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            new_word = wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            new_word = wnl.lemmatize(word, pos='a')
        elif tag.startswith('R'):
            new_word = wnl.lemmatize(word, pos='r')
        else:
            new_word = word
        # 去除停用词
        if new_word in stops:
            continue
        ans.append(new_word)
    
    return " ".join(ans)


def create_bi_gram(text, word_doc_count_dic):
    ret = []
        
    for x in combinations(np.unique(lemmatize_all(text).split()), 2):
        if (x[0] + x[1]) in word_doc_count_dic:
            new_word = x[0] + x[1]
            if len(new_word) > 0:
                ret.append(new_word)
                continue
                
        if (x[1] + x[0]) in word_doc_count_dic:
            new_word = x[1] + x[0]
            if len(new_word) > 0:
                ret.append(new_word)
                continue
        
        if len(x[0] + x[1]) > 0:
            ret.append(x[0] + "___" + x[1])
        
    return " ".join(ret)     
	
	
def create_bigrams_df(x):
    global word_doc_count_dic
    return x.apply(lambda y: create_bi_gram(y, word_doc_count_dic))


def tokenize(text):
    return [w for w in text.split()]

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

        # 填充category_name
        sample["category_name"].fillna("//", inplace=True)
        sample["catery_name_first"] =  sample["category_name"].apply(lambda x: x.split("/")[0])
        sample["catery_name_second"] = sample["category_name"].apply(lambda x: x.split("/")[1])
        sample["catery_name_third"] = sample["category_name"].apply(lambda x: x.split("/")[2])
        sample['category_name'] = sample['category_name'].apply( lambda x: ' '.join(x.split('/') ).strip())

        sample['nb_words_item_description'] = sample['item_description'].apply(lg).astype(np.uint16)
        sample['nb_words_item_description'] = 1.0 * sample['nb_words_item_description'] / max_text_length

        for x in ['brand_name', 'category_name', 'catery_name_first', 'catery_name_second', 'catery_name_third']:
            le = preprocessing.LabelEncoder()
            sample[x] = le.fit_transform(sample[x])

        sample["old_name"] = sample["name"].copy()
        sample["brand_cat"] = "cat1_" + sample["catery_name_first"].astype(str) + " " +                     "cat2_" + sample["catery_name_second"].astype(str) + " " +                     "cat3_" +  sample["catery_name_third"].astype(str) + " " +                     "brand_" + sample["brand_name"].astype(str)
        sample["name"] = sample["brand_cat"] + " " + sample["name"]
        # 融合了cate, brand, name, desc
        sample["name_desc"] = sample["name"] + " " + sample["item_description"].apply(lambda x: " ".join(x.split()[:5]))  
        gc.collect()
        
    print("构造文本特征:")
    with Clock():
#         global word_doc_count_dic
#         word_doc_count_dic = dict()
#         for x in ['name','item_description']:
#             sample[x].apply(lambda y: word_doc_counts(y, word_doc_count_dic))
#         min_df_one = 5
#         rare_words = [x for x in word_doc_count_dic if word_doc_count_dic[x] < min_df_one]
#         for x in rare_words:
#             del word_doc_count_dic[x]
#         for x in ['name','item_description']:
#             sample[x] = sample[x].apply(lambda y: remove_rare_words(y, word_doc_count_dic))

#         sample['name_bi'] = parallelize_dataframe(sample['name_desc'],  create_bigrams_df)
#         bi_word_doc_count_dic = dict()
#         sample['name_bi'].apply(lambda y: word_doc_counts(y, bi_word_doc_count_dic))
#         rare_words = [x for x in bi_word_doc_count_dic if bi_word_doc_count_dic[x] < min_df_one]
#         for x in rare_words:
#             del bi_word_doc_count_dic[x]
#         sample["name_bi"] = sample["name_bi"].apply(lambda y: remove_rare_words(y, bi_word_doc_count_dic))

        # 构建索引表
#         vocabulary_one = word_doc_count_dic.copy()
#         vocabulary_bi = bi_word_doc_count_dic.copy()
#         for dc in [vocabulary_one,  vocabulary_bi]:
#             cpt = 0
#             for key in dc:
#                 dc[key] = cpt
#                 cpt += 1
#         count = 0
#         for key in word_doc_count_dic:
#             vocabulary_one[key] = count
#             count += 1
        print("train_name_one")
        vect_name_one  = CountVectorizer(stop_words="english",
                                         min_df=5,
                                         dtype=np.uint8,
                                         tokenizer=tokenize,
                                         binary=True) 
        train_name_one  = vect_name_one.fit_transform(sample['name'])

        print("vect_name_one")
        vect_item_one = CountVectorizer(stop_words="english",
                                        ngram_range=(1, 2),
                                        min_df=5,   
                                        dtype=np.uint8, 
                                        tokenizer=tokenize, 
                                        binary=True) 
        print("train_item_one")
        train_item_one = vect_item_one.fit_transform(sample['item_description'])
#         vect_name_bi = CountVectorizer(vocabulary= vocabulary_bi,   
#                                    dtype=np.uint8, 
#                                    tokenizer=tokenize, 
#                                    binary=True) 
#         train_name_bi = vect_name_bi.fit_transform(sample['name_bi'])
        gc.collect()
        
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
        keep = ['item_condition_id', 'shipping', 'nb_words_item_description',
           'mean_price_catery_name_first', 'mean_price_catery_name_second', 'mean_price_catery_name_third',
            'mean_price_category_name', 'mean_price_brand_name']
        y = train.price.values
        x = hstack([sample[keep].values, train_name_one, train_item_one]).tocsr()
        test = x[train.shape[0]:]
        x = x[:train.shape[0]]
        gc.collect()
    
    return x, y, test 

def train(x, y):
    ridge = Ridge(alpha=20, copy_X=True, 
              fit_intercept=True, solver='auto', 
              max_iter=200, normalize=False, 
              random_state=0,  tol=0.0025)
    ridge.fit(x, y)
    return ridge

def submit(model, submit, test):
    path = '../input/'
    pred = model.predict(test)
    pred = np.expm1(pred)
    pred[pred < 3] = 3
    pred[pred > 1000] = 1000
    test_id= pd.read_csv(path + "test_stg2.tsv", sep='\t', encoding='utf-8')[["test_id"]]
    test_id["price"] = pred
    test_id.to_csv(submit, index=False)




path = '../input/'
x, y, test = get_sample(path + "train.tsv", path + "test_stg2.tsv", max_text_length=60)
ridge = train(x, y)
submit(ridge, "ans.csv", test)






