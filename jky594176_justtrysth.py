#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
import glob
import os
import json
import matplotlib.pyplot as plt
import pprint

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from tqdm import tqdm
from PIL import Image

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.display.max_rows = 128
pd.options.display.max_columns = 128

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


# In[2]:


def get_session(gpu_fraction=0.6):
    # 设置允许TF使用的GPU为60%
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(get_session())


# In[3]:


plt.rcParams['figure.figsize'] = (12, 9)


# In[4]:


os.listdir('../input')


# In[5]:


os.listdir('../input/petfinder-adoption-prediction/test/')


# In[6]:


train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
sample_submission = pd.read_csv('../input/petfinder-adoption-prediction/test/sample_submission.csv')


# In[7]:



state_gdp = {
    41336: 116.679,
    41325: 40.596,
    41367: 23.02,
    41401: 190.075,
    41415: 5.984,
    41324: 37.274,
    41332: 42.389,
    41335: 52.452,
    41330: 67.629,
    41380: 5.642,
    41327: 81.284,
    41345: 80.167,
    41342: 121.414,
    41326: 280.698,
    41361: 32.270
}

# 州人口: https://en.wikipedia.org/wiki/Malaysia
state_population = {
    41336: 33.48283,
    41325: 19.47651,
    41367: 15.39601,
    41401: 16.74621,
    41415: 0.86908,
    41324: 8.21110,
    41332: 10.21064,
    41335: 15.00817,
    41330: 23.52743,
    41380: 2.31541,
    41327: 15.61383,
    41345: 32.06742,
    41342: 24.71140,
    41326: 54.62141,
    41361: 10.35977
}

train["state_gdp"] = train['State'].map(state_gdp)
train["state_population"] = train['State'].map(state_population)
test["state_gdp"] = test['State'].map(state_gdp)
test["state_population"] = test['State'].map(state_population)
train["gdp_vs_population"] = train["state_gdp"] / train["state_population"]
test["gdp_vs_population"] = test["state_gdp"] / test["state_population"]


# In[8]:


labels_breed = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')
labels_state = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')
labels_color = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')


# In[9]:


train_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_images/*.jpg'))
train_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_metadata/*.json'))
train_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_sentiment/*.json'))

print('num of train images files: {}'.format(len(train_image_files)))
print('num of train metadata files: {}'.format(len(train_metadata_files)))
print('num of train sentiment files: {}'.format(len(train_sentiment_files)))


test_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_images/*.jpg'))
test_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_metadata/*.json'))
test_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_sentiment/*.json'))

print('num of test images files: {}'.format(len(test_image_files)))
print('num of test metadata files: {}'.format(len(test_metadata_files)))
print('num of test sentiment files: {}'.format(len(test_sentiment_files)))


# In[10]:


plt.rcParams['figure.figsize'] = (12, 9)
plt.style.use('ggplot')


# Images:
train_df_ids = train[['PetID']]
print(train_df_ids.shape)

train_df_imgs = pd.DataFrame(train_image_files)
train_df_imgs.columns = ['image_filename']
train_imgs_pets = train_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
train_df_imgs = train_df_imgs.assign(PetID=train_imgs_pets)
print(len(train_imgs_pets.unique()))

pets_with_images = len(np.intersect1d(train_imgs_pets.unique(), train_df_ids['PetID'].unique()))
print('fraction of pets with images: {:.3f}'.format(pets_with_images / train_df_ids.shape[0]))

# Metadata:
train_df_ids = train[['PetID']]
train_df_metadata = pd.DataFrame(train_metadata_files)
train_df_metadata.columns = ['metadata_filename']
train_metadata_pets = train_df_metadata['metadata_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
train_df_metadata = train_df_metadata.assign(PetID=train_metadata_pets)
print(len(train_metadata_pets.unique()))

pets_with_metadatas = len(np.intersect1d(train_metadata_pets.unique(), train_df_ids['PetID'].unique()))
print('fraction of pets with metadata: {:.3f}'.format(pets_with_metadatas / train_df_ids.shape[0]))

# Sentiment:
train_df_ids = train[['PetID']]
train_df_sentiment = pd.DataFrame(train_sentiment_files)
train_df_sentiment.columns = ['sentiment_filename']
train_sentiment_pets = train_df_sentiment['sentiment_filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
train_df_sentiment = train_df_sentiment.assign(PetID=train_sentiment_pets)
print(len(train_sentiment_pets.unique()))

pets_with_sentiments = len(np.intersect1d(train_sentiment_pets.unique(), train_df_ids['PetID'].unique()))
print('fraction of pets with sentiment: {:.3f}'.format(pets_with_sentiments / train_df_ids.shape[0]))


# In[11]:


# Images:
test_df_ids = test[['PetID']]
print(test_df_ids.shape)

test_df_imgs = pd.DataFrame(test_image_files)
test_df_imgs.columns = ['image_filename']
test_imgs_pets = test_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
test_df_imgs = test_df_imgs.assign(PetID=test_imgs_pets)
print(len(test_imgs_pets.unique()))

pets_with_images = len(np.intersect1d(test_imgs_pets.unique(), test_df_ids['PetID'].unique()))
print('fraction of pets with images: {:.3f}'.format(pets_with_images / test_df_ids.shape[0]))


# Metadata:
test_df_ids = test[['PetID']]
test_df_metadata = pd.DataFrame(test_metadata_files)
test_df_metadata.columns = ['metadata_filename']
test_metadata_pets = test_df_metadata['metadata_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
test_df_metadata = test_df_metadata.assign(PetID=test_metadata_pets)
print(len(test_metadata_pets.unique()))

pets_with_metadatas = len(np.intersect1d(test_metadata_pets.unique(), test_df_ids['PetID'].unique()))
print('fraction of pets with metadata: {:.3f}'.format(pets_with_metadatas / test_df_ids.shape[0]))



# Sentiment:
test_df_ids = test[['PetID']]
test_df_sentiment = pd.DataFrame(test_sentiment_files)
test_df_sentiment.columns = ['sentiment_filename']
test_sentiment_pets = test_df_sentiment['sentiment_filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
test_df_sentiment = test_df_sentiment.assign(PetID=test_sentiment_pets)
print(len(test_sentiment_pets.unique()))

pets_with_sentiments = len(np.intersect1d(test_sentiment_pets.unique(), test_df_ids['PetID'].unique()))
print('fraction of pets with sentiment: {:.3f}'.format(pets_with_sentiments / test_df_ids.shape[0]))


# are distributions the same?
print('images and metadata distributions the same? {}'.format(
    np.all(test_metadata_pets == test_imgs_pets)))


# In[12]:


class PetFinderParser(object):
    def __init__(self, debug=False):
        self.debug = debug
        self.sentence_sep = '; '
        self.extract_sentiment_text = False
        
    def open_metadata_file(self, filename):
        """
        加载元数据文件
        """
        with open(filename, 'r') as f:
            metadata_file = json.load(f)
        return metadata_file
            
    def open_sentiment_file(self, filename):
        """
        加载情感数据文件
        """
        with open(filename, 'r') as f:
            sentiment_file = json.load(f)
        return sentiment_file
            
    def open_image_file(self, filename):
        """
        加载图片文件
        """
        image = np.asarray(Image.open(filename))
        return image
        
    def parse_sentiment_file(self, file):
        """
        爬取情感数据文件，简单来讲，就是将数据通过Pandas读入一个DataFrame，然后将整一个dataframe作为返回值
        """
        # documentSentiment
        ret_val = {}
        ret_val['doc_mag'] = file['documentSentiment']['magnitude']
        ret_val['doc_score']= file['documentSentiment']['score']
        ret_val['doc_language'] = file['language']
        ret_val['doc_stcs_len'] = len(file['sentences'])
        if ret_val['doc_stcs_len']>0:
            ret_val['doc_first_score'] = file['sentences'][0]['sentiment']['score']
            ret_val['doc_first_mag'] = file['sentences'][0]['sentiment']['magnitude']
            ret_val['doc_last_score'] = file['sentences'][-1]['sentiment']['score']
            ret_val['doc_last_mag'] = file['sentences'][-1]['sentiment']['magnitude']
        else:
            ret_val['doc_first_score'] = np.nan
            ret_val['doc_first_mag'] = np.nan
            ret_val['doc_last_score'] = np.nan
            ret_val['doc_last_mag'] = np.nan
        ret_val['doc_ent_num'] = len(file['entities'])
        
        # sentence score
        mags, scores = [], []
        for s in file['sentences']:
            mags.append(s['sentiment']['magnitude'])
            scores.append(s['sentiment']['score'])
        
        if len(scores)==0:
            ret_val['doc_score_sum'] = np.nan
            ret_val['doc_mag_sum'] = np.nan
            ret_val['doc_score_mena'] = np.nan
            ret_val['doc_mag_mean'] = np.nan
            ret_val['doc_score_max'] = np.nan
            ret_val['doc_mag_max'] = np.nan
            ret_val['doc_score_min'] = np.nan
            ret_val['doc_mag_min'] = np.nan
            ret_val['doc_score_std'] = np.nan
            ret_val['doc_mag_std'] = np.nan
        else:
            ret_val['doc_score_sum'] = np.sum(scores)
            ret_val['doc_mag_sum'] = np.sum(mags)
            ret_val['doc_score_mena'] = np.mean(scores)
            ret_val['doc_mag_mean'] = np.mean(mags)
            ret_val['doc_score_max'] = np.max(scores)
            ret_val['doc_mag_max'] = np.max(mags)
            ret_val['doc_score_min'] = np.min(scores)
            ret_val['doc_mag_min'] = np.min(mags)
            ret_val['doc_score_std'] = np.std(scores)
            ret_val['doc_mag_std'] = np.std(mags)

        # entity type
        ret_val['sentiment_entities'] = []
        ret_val['doc_ent_person_count'] = 0
        ret_val['doc_ent_location_count'] = 0
        ret_val['doc_ent_org_count'] = 0
        ret_val['doc_ent_event_count'] = 0
        ret_val['doc_ent_woa_count'] = 0
        ret_val['doc_ent_good_count'] = 0
        ret_val['doc_ent_other_count'] = 0
        key_mapper = {
            'PERSON':'doc_ent_person_count',
            'LOCATION':'doc_ent_location_count',
            'ORGANIZATION':'doc_ent_org_count',
            'EVENT':'doc_ent_event_count',
            'WORK_OF_ART':'doc_ent_woa_count',
            'CONSUMER_GOOD':'doc_ent_good_count',
            'OTHER':'doc_ent_other_count'
        }
        for e in file['entities']:
            ret_val['sentiment_entities'].append(e['name'])
            if e['type'] in key_mapper:
                ret_val[key_mapper[e['type']]]+=1
        
        ret_val['sentiment_entities'] = ' '.join(ret_val['sentiment_entities'])
        return ret_val
    
    def parse_metadata_file(self, file, img):
        """
        爬取元数据文件，返回一个df
        """
        file_keys = list(file.keys())
        if 'textAnnotations' in file_keys:
#             textanno = 1
            textblock_num = len(file['textAnnotations'])
            textlen = np.sum([len(text['description']) for text in file['textAnnotations']])
        else:
#             textanno = 0
            textblock_num = 0
            textlen = 0
        if 'faceAnnotations' in file_keys:
            faceanno = 1
        else:
            faceanno = 0
        if 'labelAnnotations' in file_keys:
            file_annots = file['labelAnnotations']#[:len(file['labelAnnotations'])]
            if len(file_annots)==0:
                file_label_score_mean = np.nan
                file_label_score_max = np.nan
                file_label_score_min = np.nan
            else:
                temp = np.asarray([x['score'] for x in file_annots])
                file_label_score_mean = temp.mean()
                file_label_score_max = temp.max()
                file_label_score_min = temp.min()
            file_top_desc = [x['description'] for x in file_annots]
        else:
            file_label_score_mean = np.nan
            file_label_score_max = np.nan
            file_label_score_min = np.nan
            file_top_desc = ['']
        
        file_colors = file['imagePropertiesAnnotation']['dominantColors']['colors']
        file_crops = file['cropHintsAnnotation']['cropHints']
        if len(file_colors)==0:
            file_color_score = np.nan
            file_color_pixelfrac = np.nan
            color_red_mean = np.nan
            color_green_mean = np.nan
            color_blue_mean = np.nan
            color_red_std = np.nan
            color_green_std = np.nan
            color_blue_std = np.nan
        else:
            file_color_score = np.asarray([x['score'] for x in file_colors]).mean()
            file_color_pixelfrac = np.asarray([x['pixelFraction'] for x in file_colors]).mean()
            file_color_red = np.asarray([x['color']['red'] if 'red' in x['color'] else 0 for x in file_colors])
            file_color_green = np.asarray([x['color']['green'] if 'green' in x['color'] else 0for x in file_colors])
            file_color_blue = np.asarray([x['color']['blue'] if 'blue' in x['color'] else 0 for x in file_colors])
            color_red_mean = file_color_red.mean()
            color_green_mean = file_color_green.mean()
            color_blue_mean = file_color_blue.mean()
            color_red_std = file_color_red.std()
            color_green_std = file_color_green.std()
            color_blue_std = file_color_blue.std()
        
        if len(file_crops)==0:
            file_crop_conf=np.nan
            file_crop_importance = np.nan
            file_crop_fraction_mean = np.nan
            file_crop_fraction_sum = np.nan
            file_crop_fraction_std = np.nan
            file_crop_num = 0
        else:
            file_crop_conf = np.asarray([x['confidence'] for x in file_crops]).mean()
            file_crop_num = len(file_crops)
            if 'importanceFraction' in file_crops[0].keys():
                file_crop_importance = np.asarray([x['importanceFraction'] for x in file_crops]).mean()
            else:
                file_crop_importance = np.nan
            crop_areas = []
            image_area = img.shape[0]*img.shape[1]
            for crophint in file_crops:
                v_x, v_y = [], []
                for vertices in crophint['boundingPoly']['vertices']:
                    if 'x' not in vertices:
                        v_x.append(0)
                    else:
                        v_x.append(vertices['x'])
                    if 'y' not in vertices:
                        v_y.append(0)
                    else:
                        v_y.append(vertices['y'])
                crop_areas.append((max(v_x)-min(v_x))*(max(v_y)-min(v_y))/image_area)
            file_crop_fraction_mean = np.mean(crop_areas)
            file_crop_fraction_sum = np.sum(crop_areas)
            file_crop_fraction_std = np.std(crop_areas)

        df_metadata = {
            'label_score_mean': file_label_score_mean,
            'label_score_max': file_label_score_max,
            'label_score_min': file_label_score_min,
            'color_score': file_color_score,
            'color_pixelfrac': file_color_pixelfrac,
            'crop_conf': file_crop_conf,
            'crop_importance': file_crop_importance,
            'color_red_mean':color_red_mean,
            'color_green_mean':color_green_mean,
            'color_blue_mean':color_blue_mean,
            'color_red_std':color_red_std,
            'color_green_std':color_green_std,
            'color_blue_std':color_blue_std,
#             'crop_area_mean':file_crop_fraction_mean,
            'crop_area_sum':file_crop_fraction_sum,
#             'crop_area_std':file_crop_fraction_std,
            'annots_top_desc': self.sentence_sep.join(file_top_desc),
            'img_aratio':img.shape[0]/img.shape[1],
#             'text_annotation':textanno,
            'text_len':textlen,
            'textblock_num':textblock_num,
            'face_annotation':faceanno
        }
        
        return df_metadata
    
# 读取文件的函数
def extract_additional_features(pet_id, mode='train'):
    sentiment_filename = '../input/petfinder-adoption-prediction/{}_sentiment/{}.json'.format(mode, pet_id)
    try:
        sentiment_file = pet_parser.open_sentiment_file(sentiment_filename)
        df_sentiment = pet_parser.parse_sentiment_file(sentiment_file)
        df_sentiment['PetID'] = pet_id
    except FileNotFoundError:
        df_sentiment = None

    dfs_metadata = []
    for ind in range(1,200):
        metadata_filename = '../input/petfinder-adoption-prediction/{}_metadata/{}-{}.json'.format(mode, pet_id, ind)
        image_filename = '../input/petfinder-adoption-prediction/{}_images/{}-{}.jpg'.format(mode, pet_id, ind)
        try:
            image = cv2.imread(image_filename)
            metadata_file = pet_parser.open_metadata_file(metadata_filename)
            df_metadata = pet_parser.parse_metadata_file(metadata_file, image)
            df_metadata['PetID'] = pet_id
            dfs_metadata.append(df_metadata)
        except FileNotFoundError:
            break
    return [df_sentiment, dfs_metadata]
    
pet_parser = PetFinderParser()


# In[13]:


import pynvml
# 在进入图片处理前，确认一眼我们分配的显存
pynvml.nvmlInit()
# 这里的0是GPU id
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(meminfo.total) #第一块显卡总的显存大小
print(meminfo.used)#这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
print(meminfo.free) #第一块显卡剩余显存大小
print(pynvml.nvmlDeviceGetCount())#显示有几块GPU


# In[14]:


import cv2
import pandas as pd
import numpy as np
import os
from tqdm import tqdm, tqdm_notebook
from keras.applications.densenet import preprocess_input, DenseNet121
import itertools
# 读取宠物的PetID（Unique）
debug = False
train_pet_ids = train.PetID.unique()
test_pet_ids = test.PetID.unique()

if debug:
    train_pet_ids = train_pet_ids[:1000]
    test_pet_ids = test_pet_ids[:500]

n_jobs = 6
# Train set:
# 并行数据读取
dfs_train = Parallel(n_jobs=n_jobs, verbose=1)(
    delayed(extract_additional_features)(i, mode='train') for i in train_pet_ids)

# 将读取得到的信息特征提取，并转换为dataframe
train_dicts_sentiment = [x[0] for x in dfs_train if x[0] is not None]
train_dfs_metadata = [x[1] for x in dfs_train if len([x[1]])>0]

train_dfs_sentiment = pd.DataFrame(train_dicts_sentiment)
train_dfs_metadata = list(itertools.chain.from_iterable(train_dfs_metadata))
train_dfs_metadata = pd.DataFrame(train_dfs_metadata)

print(train_dfs_sentiment.shape, train_dfs_metadata.shape)


# In[15]:


# 测试集读取：
dfs_test = Parallel(n_jobs=n_jobs, verbose=1)(
    delayed(extract_additional_features)(i, mode='test') for i in test_pet_ids)

test_dicts_sentiment = [x[0] for x in dfs_test if x[0] is not None]
test_dfs_metadata = [x[1] for x in dfs_test if len(x[1])>0]

test_dfs_sentiment = pd.DataFrame(test_dicts_sentiment)
test_dfs_metadata = list(itertools.chain.from_iterable(test_dfs_metadata))
test_dfs_metadata = pd.DataFrame(test_dfs_metadata)

print(test_dfs_sentiment.shape, test_dfs_metadata.shape)


# In[16]:


# 再确认一次GPU
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(meminfo.total) #第一块显卡总的显存大小
print(meminfo.used)#这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
print(meminfo.free) #第一块显卡剩余显存大小
print(pynvml.nvmlDeviceGetCount())#显示有几块GPU


# In[17]:


# 分别处理metadata和sentiment，最终会在merge部分和main DFs融合。
meta_df = pd.concat([train_dfs_metadata, test_dfs_metadata], sort=False).reset_index(drop=True)
senti_df = pd.concat([train_dfs_sentiment, test_dfs_sentiment], sort=False).reset_index(drop=True)

# 根据排序读取文件
metadata_desc = meta_df.groupby(['PetID'])['annots_top_desc'].unique().reset_index()
metadata_desc['meta_annots_top_desc'] = metadata_desc['annots_top_desc'].apply(lambda x: '; '.join(x))
metadata_desc.drop('annots_top_desc', axis=1, inplace=True)

possible_annots = set()
for i in range(len(meta_df)):
    possible_annots = possible_annots.union(set(meta_df['annots_top_desc'].iloc[i].split('; ')))
annot_mapper = {}
for idx, a in enumerate(possible_annots):
    annot_mapper[a] = str(idx)
metadata_desc['meta_desc'] = metadata_desc['meta_annots_top_desc'].apply(lambda x: ' '.join(annot_mapper[i] for i in x.split('; ')))


# In[18]:


# 情感特征，包括词长，内容等
senti_df['sentiment_entities'].fillna('', inplace=True)
senti_df['sentiment_entities'] = senti_df['sentiment_entities'].str.lower()
senti_df['sentiment_len'] = senti_df['sentiment_entities'].apply(lambda x:len(x))
senti_df['sentiment_word_len'] = senti_df['sentiment_entities'].apply(lambda x: len(x.replace(';',' ').split(' ')))
senti_df['sentiment_word_unique'] = senti_df['sentiment_entities'].apply(lambda x: len(set(x.replace(';',' ').split(' '))))
senti_df['doc_language'] = pd.factorize(senti_df['doc_language'])[0]


# In[19]:


senti_df.head()


# In[20]:


# metadata, 因为每一个宠物都有着很多个json文件，也就对应着大量的数据。我们需要通过聚合，将特征通过均值、最大值、最小值或者平均值反映出来。
aggregates = {
    'color_blue_mean':['mean','std'],
    'color_blue_std':['mean'],
    'color_green_mean':['mean','std'], 
    'color_green_std':['mean'],
    'color_pixelfrac':['mean','std'],
    'color_red_mean':['mean','std'],
    'color_red_std':['mean'],
    'color_score':['mean','max'], 
    'crop_area_sum':['mean','std','min'], 
    'crop_conf':['mean','std','max'],
    'crop_importance':['mean','std'],
    'label_score_max':['mean','std','max'],
    'label_score_mean':['mean','max','std'],
    'label_score_min':['mean','max','std'],
    'img_aratio':['nunique','std','max','min'],
    'textblock_num':['mean','max'],
    'face_annotation':['mean','nunique']
}

# Train
metadata_gr = meta_df.drop(['annots_top_desc'], axis=1)
for i in metadata_gr.columns:
    if 'PetID' not in i:
        metadata_gr[i] = metadata_gr[i].astype(float)
metadata_gr = metadata_gr.groupby(['PetID']).agg(aggregates)
metadata_gr.columns = pd.Index(['{}_{}_{}'.format('meta', c[0], c[1].upper()) for c in metadata_gr.columns.tolist()])
metadata_gr = metadata_gr.reset_index()


# In[21]:


meta_df = metadata_desc.merge(metadata_gr, on='PetID', how='left')

# annotation feature
meta_df['meta_annots_top_desc'].fillna(' ', inplace=True)

meta_df[meta_df['meta_textblock_num_MEAN']>0.8].head()


# In[22]:


# Train merges:
train_proc = train.copy()
# Test merges:
test_proc = test.copy()


print(train_proc.shape, test_proc.shape)
assert train_proc.shape[0] == train.shape[0]
assert test_proc.shape[0] == test.shape[0]


# In[23]:


train_proc.head()


# In[24]:


# 以breed1为主breed
train_breed_main = train_proc[['Breed1']].merge(
    labels_breed, how='left',
    left_on='Breed1', right_on='BreedID',
    suffixes=('', '_main_breed'))

train_breed_main = train_breed_main.iloc[:, 2:]
train_breed_main = train_breed_main.add_prefix('main_breed_')
# 第二品种
train_breed_second = train_proc[['Breed2']].merge(
    labels_breed, how='left',
    left_on='Breed2', right_on='BreedID',
    suffixes=('', '_second_breed'))

train_breed_second = train_breed_second.iloc[:, 2:]
train_breed_second = train_breed_second.add_prefix('second_breed_')

# 添加两列用来记录breed
train_proc = pd.concat(
    [train_proc, train_breed_main, train_breed_second], axis=1)

# 测试集做同样操作
test_breed_main = test_proc[['Breed1']].merge(
    labels_breed, how='left',
    left_on='Breed1', right_on='BreedID',
    suffixes=('', '_main_breed'))

test_breed_main = test_breed_main.iloc[:, 2:]
test_breed_main = test_breed_main.add_prefix('main_breed_')

test_breed_second = test_proc[['Breed2']].merge(
    labels_breed, how='left',
    left_on='Breed2', right_on='BreedID',
    suffixes=('', '_second_breed'))

test_breed_second = test_breed_second.iloc[:, 2:]
test_breed_second = test_breed_second.add_prefix('second_breed_')


test_proc = pd.concat(
    [test_proc, test_breed_main, test_breed_second], axis=1)

print(train_proc.shape, test_proc.shape)


# In[25]:


train_proc.head()


# In[26]:


# FROM: https://www.kaggle.com/myltykritik/simple-lgbm-image-features
# 图片处理需要定义图片的大小以及每一次打包的文件大小。
img_size = 256
batch_size = 16


# In[27]:


pet_ids = train['PetID'].values
n_batches = len(pet_ids) // batch_size + 1


# In[28]:


def resize_to_square(im):
    # 图片标准化，将读取的图片信息重新编成256*256*3的RGB图片。
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return new_im

def load_image(path, pet_id,i=1):
    # 读取图片的函数。我们将i的默认值设为1，是为了在仅读取一个petID的一张图片时用
    image = cv2.imread(f'{path}{pet_id}-{i}.jpg')
    new_image = resize_to_square(image)
    new_image = preprocess_input(new_image)
    return new_image


# In[29]:


# 通过使用别人训练好的DenseNet分类器分类特征
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K
inp = Input((256,256,3))
backbone = DenseNet121(input_tensor = inp, 
                       weights="../input/densenet/DenseNet-BC-121-32-no-top.h5",
                       include_top = False)
x = backbone.output
x = GlobalAveragePooling2D()(x)
x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)
x = AveragePooling1D(4)(x)
out = Lambda(lambda x: x[:,:,0])(x)

m = Model(inp,out)


# In[30]:


# 图片特征提取，前面注释的部分是遍历每一张图片，最终我们只遍历每一个PetID的第一张图片。
# 在此隆重介绍装逼神奇tqdm!这个可以让长循环生成进度条的工具实在是太舒服了。。。
"""
features = {}
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_pets = pet_ids_list[start:end]
    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))
    for i,pet_id in enumerate(batch_pets):
        for j in range(int(train[train.PetID==pet_id]['PhotoAmt'])):
            try:
                batch_images[i] = load_image("../input/petfinder-adoption-prediction/train_images/", pet_id,j)
            except:
                pass
    batch_preds = m.predict(batch_images)
    for i,pet_id in enumerate(batch_pets):
        if pet_id in features.keys():
            sum += batch_preds[i]
            count += 1
        else:
            sum = batch_preds[i]
            count = 0
        if count == 0:
            count = 0.1
        features[pet_id] = sum/count
        
"""
features = {}
# 将工作分为n个batch。不知道有什么用。。。可能是为了让进度条看起来好看一点
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_pets = pet_ids[start:end]
    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))
    # 对每个petID进行特征提取
    for i,pet_id in enumerate(batch_pets):
        try:
            batch_images[i] = load_image("../input/petfinder-adoption-prediction/train_images/", pet_id)
        except:
            pass
    batch_preds = m.predict(batch_images)
    for i,pet_id in enumerate(batch_pets):
        features[pet_id] = batch_preds[i]


# In[31]:


train_feats = pd.DataFrame.from_dict(features, orient='index')
train_feats.columns = ['pic_'+str(i) for i in range(train_feats.shape[1])]


# In[32]:


# 测试集遍历
"""
sum = 0
pet_ids = test_proc['PetID'].values
pet_ids_list = []
for k in range(len(pet_ids)):
    sum += int(test[test.PetID==pet_ids[k]]['PhotoAmt'])
    for j in range(int(test[test.PetID==pet_ids[k]]['PhotoAmt'])):
        pet_ids_list.append(pet_ids[k])
n_batches = sum // batch_size + 1
print(pet_ids_list[0:20])


features = {}
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_pets = pet_ids_list[start:end]
    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))
    for i,pet_id in enumerate(batch_pets):
        for j in range(int(test[test.PetID==pet_id]['PhotoAmt'])):
            try:
                batch_images[i] = load_image("../input/petfinder-adoption-prediction/test_images/", pet_id,j)
            except:
                pass
    batch_preds = m.predict(batch_images)
    for i,pet_id in enumerate(batch_pets):
        if pet_id in features.keys():
            sum += batch_preds[i]
            count += 1
        else:
            sum = batch_preds[i]
            count = 0
        if count == 0:
            count = 0.1
        features[pet_id] = sum/count
        
"""
pet_ids = test['PetID'].values
n_batches = len(pet_ids) // batch_size + 1

features = {}
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_pets = pet_ids[start:end]
    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))
    for i,pet_id in enumerate(batch_pets):
        try:
            batch_images[i] = load_image("../input/petfinder-adoption-prediction/test_images/", pet_id)
        except:
            pass
    batch_preds = m.predict(batch_images)
    for i,pet_id in enumerate(batch_pets):
        features[pet_id] = batch_preds[i]


# In[33]:


# 再一次查看GPU

pynvml.nvmlInit()
# 这里的1是GPU id
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(meminfo.total) #第1块显卡总的显存大小
print(meminfo.used)#这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
print(meminfo.free) #第1块显卡剩余显存大小
print(pynvml.nvmlDeviceGetCount())#显示有几块GPU


# In[34]:


test_feats = pd.DataFrame.from_dict(features, orient='index')
test_feats.columns = ['pic_'+str(i) for i in range(test_feats.shape[1])]


# In[35]:


test_feats = test_feats.reset_index()
test_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)

train_feats = train_feats.reset_index()
train_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)

train = pd.merge(train, train_feats, how='left', on='PetID')
test = pd.merge(test, test_feats, how='left', on='PetID')


# In[36]:


train_proc = pd.merge(train_proc, train_feats, how='left', on='PetID')
test_proc = pd.merge(test_proc, test_feats, how='left', on='PetID')


# In[37]:


X = pd.concat([train_proc, test_proc], ignore_index=True, sort=False)
print('NaN structure:\n{}'.format(np.sum(pd.isnull(X))))


# In[38]:


X.head()


# In[39]:


column_types = X.dtypes

int_cols = column_types[column_types == 'int']
float_cols = column_types[column_types == 'float']
cat_cols = column_types[column_types == 'object']

print('\tinteger columns:\n{}'.format(int_cols))
print('\n\tfloat columns:\n{}'.format(float_cols))
print('\n\tto encode categorical columns:\n{}'.format(cat_cols))


# In[40]:


# 首先，我们先定义一个to_drop_columns，这个list用于保存不需要进入模型训练的数据列名
to_drop_columns = ['PetID', 'Name', 'RescuerID']
# 创建X_temp作为特征工程的操作列
# 这样可以避免一些误操作
X_temp = X.copy()
rescuer_ids = X_temp['RescuerID'].values

# 通过将3列color和2列breed合并，重新编码进行降维
X_temp['Breed_full'] = X_temp['Breed1'].astype(str)+'_'+X_temp['Breed2'].astype(str)
X_temp['Color_full'] = X_temp['Color1'].astype(str)+'_'+X_temp['Color2'].astype(str)+'_'+X_temp['Color3'].astype(str)
X_temp['Breed_full'],_ = pd.factorize(X_temp['Breed_full'])
X_temp['Color_full'],_ = pd.factorize(X_temp['Color_full'])

to_drop_columns.extend(['Breed1','Breed2','Color1','Color2','Color3'])


# In[41]:


# 添加姓名特征，因为姓名本身的意义并不大，但是可以通过操作来得到一些有用的特征
import re
pattern = re.compile(r"[0-9\.:!]")
X_temp['empty_name'] = X_temp['Name'].isnull().astype(np.int8)
X_temp['Name'] =X_temp['Name'].fillna('')
X_temp['name_len'] = X_temp['Name'].apply(lambda x: len(x))
X_temp['strange_name'] = X_temp['Name'].apply(lambda x: len(pattern.findall(x))>0).astype(np.int8)


# In[42]:


# 重新编码Vacinated, Dewormed, Sterilized等特征。目的也是降维

X_temp['hard_interaction'] = X_temp['Type'].astype(str)+X_temp['Gender'].astype(str)+                               X_temp['Vaccinated'].astype(str)+'_'+                               X_temp['Dewormed'].astype(str)+'_'+X_temp['Sterilized'].astype(str)
X_temp['hard_interaction'],_ = pd.factorize(X_temp['hard_interaction'])

X_temp['MaturitySize'] = X_temp['MaturitySize'].replace(0, np.nan)
X_temp['FurLength'] = X_temp['FurLength'].replace(0, np.nan)

X_temp['Vaccinated'] = X_temp['Vaccinated'].replace(3, np.nan)
X_temp['Vaccinated'] = X_temp['Vaccinated'].replace(2, 0)

X_temp['Dewormed'] = X_temp['Dewormed'].replace(3, np.nan)
X_temp['Dewormed'] = X_temp['Dewormed'].replace(2, 0)

X_temp['Sterilized'] = X_temp['Sterilized'].replace(3, np.nan)
X_temp['Sterilized'] = X_temp['Sterilized'].replace(2, 0)


X_temp['Health'] = X_temp['Health'].replace(0, np.nan)
to_drop_columns.extend


# In[43]:


to_drop_columns.extend(['Type','Gender','Vaccinated','Dewormed','Sterilized'])


# In[44]:


# nltk是常用的自然语言处理包

from nltk.tokenize import TweetTokenizer
import nltk
isascii = lambda s: len(s) == len(s.encode())
# Tokenizer是一个分词器，他可以将文本转换为向量形式。
tknzr = TweetTokenizer()
# jieba是一个中文分词器（不要问我为什么还有中文，这个组织是马来西亚的）
import jieba

# PorterStemmer和SnowballStemmer两者都是提取词干的算法
from nltk.stem import PorterStemmer

ps = PorterStemmer()
from nltk.stem.lancaster import LancasterStemmer
lc = LancasterStemmer()
from nltk.stem import SnowballStemmer
sb = SnowballStemmer("english")

def custom_tokenizer(text):
    init_doc = tknzr.tokenize(text)
    retval = []
    for t in init_doc:
        if isascii(t): 
            retval.append(t)
        else:
            for w in t:
                retval.append(w)
    return retval


# In[45]:


X_temp['Description'] = X_temp['Description'].fillna(' ')

# 分别对中文和英文进行处理
english_desc, chinese_desc = [], []
tokens = set()
word_dict = {}
pos_count, word_count = 1, 1 
pos_dict = {}
eng_sequences = []
pos_sequences = []
for i in range(len(X_temp)):
    e_d, c_d, eng_seq, pos_seq = [], [], [], []
    doc = custom_tokenizer(X_temp['Description'].iloc[i])
    for token in doc:
        if not isascii(token):
            c_d.append(token)
        else:
            e_d.append(token)
            if token not in word_dict:
                word_dict[token] = word_count
                word_count +=1
    english_desc.append(' '.join(e_d))
    chinese_desc.append(' '.join(c_d))
    pos_seq = nltk.pos_tag(e_d)
    for t in pos_seq:
        if t[1] not in pos_dict:
            pos_dict[t[1]] = pos_count
            pos_count += 1
    pos_seq = [pos_dict[t[1]] for t in pos_seq]
    eng_seq = [word_dict[t] for t in e_d]
    if len(eng_seq)==0:
        eng_seq.append(0)
        pos_seq.append(0)
    eng_sequences.append(eng_seq)
    pos_sequences.append(pos_seq)


# In[46]:


# 从所提取的文本信息中提取特征，特征主要有文本长度，单词长度，单词数量等等。
# 这些特征不一定有用，我们最后会判断这些特征是否有用，这里先提取出来即可。

X_temp['English_desc'] = english_desc
X_temp['Chinese_desc'] = chinese_desc

X_temp['e_description_len'] = X_temp['English_desc'].apply(lambda x:len(x))
X_temp['e_description_word_len'] = X_temp['English_desc'].apply(lambda x: len(x.split(' ')))
X_temp['e_description_word_unique'] = X_temp['English_desc'].apply(lambda x: len(set(x.split(' '))))

X_temp['c_description_len'] = X_temp['Chinese_desc'].apply(lambda x:len(x))
X_temp['c_description_word_len'] = X_temp['Chinese_desc'].apply(lambda x:len(x.split(' ')))
X_temp['c_description_word_unique'] = X_temp['Chinese_desc'].apply(lambda x: len(set(x)))

X_temp['description_len'] = X_temp['Description'].apply(lambda x:len(x))
X_temp['description_word_len'] = X_temp['Description'].apply(lambda x: len(x.split(' ')))

to_drop_columns.extend(['English_desc','Description','Chinese_desc'])


# In[47]:


# 选择需要处理的列
text_columns = ['Description','English_desc','Chinese_desc']
categorical_columns = ['main_breed_BreedName', 'second_breed_BreedName']


# In[48]:


# 和name和PetID都该被处理， RescuerID被处理后也会被丢掉

# 计算出现次数
rescuer_count = X.groupby(['RescuerID'])['PetID'].count().reset_index()
rescuer_count.columns = ['RescuerID', 'RescuerID_COUNT']

# 添加特征到目标DF
X_temp = X_temp.merge(rescuer_count, how='left', on='RescuerID')


# In[49]:


feat_df = X_temp[['PetID','Color1','Breed1','State','RescuerID','Name','Breed_full','Color_full','hard_interaction']]


# In[50]:


# 继续处理其他特征
agg = {
    'Fee':['mean','std','max'],
    'Breed1':['nunique'],
    #'Gender':['nunique'],
    'Age':['mean','std','max'], #,'min'
    'Quantity':['std'],#'mean',,'min','max'
    'PetID':['nunique']
}
# 处理颜色特征
feat = X_temp.groupby('Color1').agg(agg)
feat.columns = pd.Index(['COLOR_' + e[0] + "_" + e[1].upper() for e in feat.columns.tolist()])
feat.reset_index(inplace=True)
feat_df = feat_df.merge(feat, on='Color1', how='left')

agg = {
    'Fee':['mean','std','max'],
    'Breed_full':['nunique'],
    'Quantity':['sum'],
}
feat = X_temp.groupby('Color_full').agg(agg)
feat.columns = pd.Index(['COLORfull_' + e[0] + "_" + e[1].upper() for e in feat.columns.tolist()])
feat.reset_index(inplace=True)
feat_df = feat_df.merge(feat, on='Color_full', how='left')


# In[51]:


# Breed feature
agg = {
    'Color_full':['nunique'],
    'Breed2':['nunique'],
    'FurLength':['nunique'],
    'Fee':['mean','max'],#,'min'
    'Age':['mean','std','min','max'],
    'Quantity':['mean','std','max','sum'],#'min'
    'PetID':['nunique'],
    'FurLength':['mean'],
    'Health':['mean'],
    'MaturitySize':['mean','std','min','max'],
    'Vaccinated':['mean'],
    'Dewormed':['mean'],
    'Sterilized':['mean']
}
feat = X_temp.groupby('Breed1').agg(agg)
feat.columns = pd.Index(['BREED1_' + e[0] + "_" + e[1].upper() for e in feat.columns.tolist()])
feat.reset_index(inplace=True)
feat_df = feat_df.merge(feat, on='Breed1', how='left')

# Breed 特征
agg = {
    'Color_full':['nunique'],
    'Fee':['mean','min','max'],
    'Quantity':['sum'],
    'PetID':['nunique']
}
feat = X_temp.groupby('Breed_full').agg(agg)
feat.columns = pd.Index(['BREEDfull_' + e[0] + "_" + e[1].upper() for e in feat.columns.tolist()])
feat.reset_index(inplace=True)
feat_df = feat_df.merge(feat, on='Breed_full', how='left')


# In[52]:


# State 特征
agg = {
    'Color_full':['nunique'],
    'Breed_full':['nunique'],
    'PetID':['nunique'],
    'RescuerID':['nunique'],
    'Fee':['mean','max'],
    'Age':['mean','std','max'],
    'Quantity':['mean','std','max'],#,'min','sum'
    'FurLength':['mean','std'],
    'Health':['mean'],
    'MaturitySize':['mean','std'],
    'Vaccinated':['mean'],
    'Dewormed':['mean'],
    'Sterilized':['mean'],
    'VideoAmt':['mean','std'],
    'PhotoAmt':['mean','std']
}
feat = X_temp.groupby('State').agg(agg)
feat.columns = pd.Index(['STATE_' + e[0] + "_" + e[1].upper() for e in feat.columns.tolist()])
feat.reset_index(inplace=True)
feat_df = feat_df.merge(feat, on='State', how='left')


# In[53]:


agg = {
    'Fee':['mean','min','max'],
    'Age':['mean','std','min','max'],
    'Quantity':['mean','std','sum'],
    'PetID':['nunique'],
    'FurLength':['mean'],
    'Health':['mean'],
    'MaturitySize':['mean','std'],
    'Vaccinated':['mean'],
    'Dewormed':['mean'],
    'Sterilized':['mean']
}
feat = X_temp.groupby(['State','Breed1','Color1']).agg(agg)
feat.columns = pd.Index(['MULTI_' + e[0] + "_" + e[1].upper() for e in feat.columns.tolist()])
feat.reset_index(inplace=True)
feat_df = feat_df.merge(feat, on=['State','Breed1','Color1'], how='left')

agg = {
    'Fee':['mean','min','max'],
    'Age':['mean','std','min','max'],
    'Quantity':['mean','std','sum'],
    'PetID':['nunique'],
}
feat = X_temp.groupby(['State','Breed_full','Color_full']).agg(agg)
feat.columns = pd.Index(['MULTI2_' + e[0] + "_" + e[1].upper() for e in feat.columns.tolist()])
feat.reset_index(inplace=True)
feat_df = feat_df.merge(feat, on=['State','Breed_full','Color_full'], how='left')


# In[54]:


# name feature
feat = X_temp.groupby('Name')['PetID'].agg({'name_count':'nunique'}).reset_index()
feat_df = feat_df.merge(feat, on='Name', how='left')


# In[55]:


# 继续添加特征
agg = {

    'PetID':['nunique'],
    'Breed_full':['nunique'],
    'VideoAmt':['mean','std'],
    'PhotoAmt':['mean','std'],
    'Sterilized':['mean'],
    'Dewormed':['mean'],
    'Vaccinated':['mean']
}
rescuer_count = X_temp.groupby(['RescuerID']).agg(agg)
rescuer_count.columns = pd.Index(['RESCUER_' + e[0] + "_" + e[1].upper() for e in rescuer_count.columns.tolist()])
rescuer_count.reset_index(inplace=True)
feat_df = feat_df.merge(rescuer_count, how='left', on='RescuerID')


# In[56]:


# State 特征
agg = {
    'Fee':['mean','min','max']
}
feat = X_temp.groupby('hard_interaction').agg(agg)
feat.columns = pd.Index(['INTERACTION_' + e[0] + "_" + e[1].upper() for e in feat.columns.tolist()])
feat.reset_index(inplace=True)
feat_df = feat_df.merge(feat, on='hard_interaction', how='left')


# In[57]:


feat_df.drop(['Color1','Breed1','State','RescuerID','Name','Breed_full','Color_full','hard_interaction'], axis=1, inplace=True)


# In[58]:


# 将特征型的列重新编码
for i in categorical_columns:
    X_temp.loc[:, i] = pd.factorize(X_temp.loc[:, i])[0]


# In[59]:


X_temp.head()


# In[60]:


# 建立text数据子集
X_text = X_temp[text_columns]

for i in X_text.columns:
    X_text.loc[:, i] = X_text.loc[:, i].fillna('<MISSING>')


# In[61]:


X_text = X_text.drop(['Chinese_desc'],axis =1)


# In[62]:


# 通过TFIDF处理文本特征
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import SparsePCA, TruncatedSVD, LatentDirichletAllocation, NMF

n_components = 5
text_features = []

# 之前也提到了，其实TFIDF就是文本特征
# 生成文本特征
for i in X_text.columns:
    
    # Initialize decomposition methods:
    print('generating features from: {}'.format(i))
    svd_ = TruncatedSVD(
        n_components=n_components, random_state=1337)
    nmf_ = NMF(
        n_components=n_components, random_state=1337)
    
    tfidf_col = TfidfVectorizer().fit_transform(X_text.loc[:, i].values)
    svd_col = svd_.fit_transform(tfidf_col)
    svd_col = pd.DataFrame(svd_col)
    svd_col = svd_col.add_prefix('SVD_{}_'.format(i))
    
    nmf_col = nmf_.fit_transform(tfidf_col)
    nmf_col = pd.DataFrame(nmf_col)
    nmf_col = nmf_col.add_prefix('NMF_{}_'.format(i))
    
    text_features.append(svd_col)
    text_features.append(nmf_col)

    
# 结合所有文本数据
text_features = pd.concat(text_features, axis=1)

# 接合文本数据到主DF
X_temp = pd.concat([X_temp, text_features], axis=1)


# In[63]:


# 这一步本来是为了删除一些不必要的列的。。后来没删，但是后面的代码懒得改了，于是就多出来了一个X_proc
X_proc = X_temp.copy()
#for item in to_drop_columns:
#    X_temp = X_temp.drop([item], axis=1)

# Check final df shape:
#print('X shape: {}'.format(X_temp.shape))


# In[64]:


# 在merge之前观察一下我们的列
column_types = X.dtypes

int_cols = column_types[column_types == 'int']
float_cols = column_types[column_types == 'float']
cat_cols = column_types[column_types == 'object']

print('\tinteger columns:\n{}'.format(int_cols))
print('\n\tfloat columns:\n{}'.format(float_cols))
print('\n\tto encode categorical columns:\n{}'.format(cat_cols))


# In[65]:


# merge
X_temp = X_temp.merge(senti_df, how='left', on='PetID')
X_temp = X_temp.merge(meta_df, how='left', on='PetID')
X_temp = X_temp.merge(feat_df, how='left', on='PetID')

print(X_temp.shape)


# In[66]:


X_temp.head()


# In[67]:


# 将训练集和测试集分开
X_train = X_temp.loc[np.isfinite(X_temp.AdoptionSpeed), :]
X_test = X_temp.loc[~np.isfinite(X_temp.AdoptionSpeed), :]

# 将测试集中的AdoptionSpeed列删除（理论上全部都是NaN）
X_test = X_test.drop(['AdoptionSpeed'], axis=1)


print('X_train shape: {}'.format(X_train.shape))
print('X_test shape: {}'.format(X_test.shape))

assert X_train.shape[0] == train.shape[0]
assert X_test.shape[0] == test.shape[0]


# 确认长度，保证分割正确性
train_cols = X_train.columns.tolist()
train_cols.remove('AdoptionSpeed')

test_cols = X_test.columns.tolist()

assert np.all(train_cols == test_cols)
rescuer_ids = rescuer_ids[:len(X_train)]
assert len(rescuer_ids) == len(X_train)


# In[68]:


np.sum(pd.isnull(X_train))


# In[69]:


np.sum(pd.isnull(X_test))


# In[70]:


import scipy as sp

from collections import Counter
from functools import partial
from math import sqrt

from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix

# 这以下三个函数都是拿来计算QWK的。QWK的计算过程我们在报告中有写，可以参考。
# taken from Ben Hamner's github repository： https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    返回两个rater产生的混淆矩阵
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    直方图统计
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    # 计算并返回QWK
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


def to_bins(x, borders):
    # 封装函数
    for i in range(len(borders)):
        if x <= borders[i]:
            return i
    return len(borders)

class OptimizedRounder(object):
    # 这个类是在模型输出结果的时候用的。
    # 每个模型训练后，都会输出一个numpy的矩阵，而这几个矩阵通过这个函数结合起来后，可以得到我们最终的stacking结果
    # 这个函数可以在单个模型中用，也可以在stacking后的模型用
    def __init__(self):
        self.coef_ = 0

    def _loss(self, coef, X, y, idx):
        # 定义损失函数
        X_p = np.array([to_bins(pred, coef) for pred in X])
        ll = -quadratic_weighted_kappa(y, X_p)
        return ll
    
    def _kappa_loss(self, coef, X, y):
        # 定义卡帕损失系数
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        # 定义拟合
        coef = [1.5, 2.0, 2.5, 3.0]
        golden1 = 0.618
        golden2 = 1 - golden1
        ab_start = [(1, 2), (1.5, 2.5), (2, 3), (2.5, 3.5)]
        for it1 in range(10):
            for idx in range(4):
                # golden section search
                a, b = ab_start[idx]
                # calc losses
                coef[idx] = a
                la = self._loss(coef, X, y, idx)
                coef[idx] = b
                lb = self._loss(coef, X, y, idx)
                for it in range(20):
                    # choose value
                    if la > lb:
                        a = b - (b - a) * golden1
                        coef[idx] = a
                        la = self._loss(coef, X, y, idx)
                    else:
                        b = b - (b - a) * golden2
                        coef[idx] = b
                        lb = self._loss(coef, X, y, idx)
        self.coef_ = {'x': coef}

    def predict(self, X, coef):
        X_p = np.array([to_bins(pred, coef) for pred in X])
        return X_p

    def coefficients(self):
        return self.coef_['x']
    
def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# In[71]:


# 以下几个函数，是用来调参的时候查看输出结果的。

def val_kappa(preds, train_data):
    labels = train_data.get_label()
    preds = np.argmax(preds.reshape((-1,5)), axis=1)
    
    return 'qwk', quadratic_weighted_kappa(labels, preds), True

def val_kappa_reg(preds, train_data, cdf):
    labels = train_data.get_label()
    preds = getTestScore2(preds, cdf)
    return 'qwk', quadratic_weighted_kappa(labels, preds), True

def get_cdf(hist):
    return np.cumsum(hist/np.sum(hist))

def getScore(pred, cdf, valid=False):
    num = pred.shape[0]
    output = np.asarray([4]*num, dtype=int)
    rank = pred.argsort()
    output[rank[:int(num*cdf[0]-1)]] = 0
    output[rank[int(num*cdf[0]):int(num*cdf[1]-1)]] = 1
    output[rank[int(num*cdf[1]):int(num*cdf[2]-1)]] = 2
    output[rank[int(num*cdf[2]):int(num*cdf[3]-1)]] = 3
    if valid:
        cutoff = [ pred[rank[int(num*cdf[i]-1)]] for i in range(4) ]
        return output, cutoff
    return output

def getTestScore(pred, cutoff):
    num = pred.shape[0]
    output = np.asarray([4]*num, dtype=int)
    for i in range(num):
        if pred[i] <= cutoff[0]:
            output[i] = 0
        elif pred[i] <= cutoff[1]:
            output[i] = 1
        elif pred[i] <= cutoff[2]:
            output[i] = 2
        elif pred[i] <= cutoff[3]:
            output[i] = 3
    return output

def getTestScore2(pred, cdf):
    num = pred.shape[0]
    rank = pred.argsort()
    output = np.asarray([4]*num, dtype=int)
    output[rank[:int(num*cdf[0]-1)]] = 0
    output[rank[int(num*cdf[0]):int(num*cdf[1]-1)]] = 1
    output[rank[int(num*cdf[1]):int(num*cdf[2]-1)]] = 2
    output[rank[int(num*cdf[2]):int(num*cdf[3]-1)]] = 3
    return output

def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

isascii = lambda s: len(s) == len(s.encode())

def custom_tokenizer(text):
    init_doc = tknzr.tokenize(text)
    retval = []
    for t in init_doc:
        if isascii(t): 
            retval.append(t)
        else:
            for w in t:
                retval.append(w)
    return retval

def build_emb_matrix(word_dict, emb_dict):
    embed_size = 300
    nb_words = len(word_dict)+1000
    nb_oov = 0
    embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)
    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.
    for key in tqdm(word_dict):
        word = key
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.upper()
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = ps.stem(key)
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lc.stem(key)
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = sb.stem(key)
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        nb_oov+=1
        embedding_matrix[word_dict[key]] = unknown_vector                    
    return embedding_matrix, nb_words, nb_oov

def _init_esim_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
        if isinstance(module, nn.LSTM):
            hidden_size = module.bias_hh_l0.data.shape[0] // 4
        else:
            hidden_size = module.bias_hh_l0.data.shape[0] // 3
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif '_ih_' in name:
                nn.init.xavier_normal_(param)
            elif '_hh_' in name:
                nn.init.orthogonal_(param)
                param.data[hidden_size:(2 * hidden_size)] = 1.0


# In[72]:


# K交叉验证算法需要的函数

def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


# In[73]:


def get_oof(clf, X, y, X_test, groups):   
    # 将训练的预测结果转换成可以stacking的函数
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(stratified_group_k_fold(X, y, rescuer_ids, NFOLDS, 1337)):
        print('Training for fold: ', i + 1)
        
        x_tr = X.iloc[train_index, :]
        y_tr = y[train_index]
        x_te = X.iloc[test_index, :]
        y_te = y[test_index]

        clf.train(x_tr, y_tr, x_val=x_te, y_val=y_te)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(X_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# In[74]:


# 这一块是几个模型训练定义的类。当然，最后我们并没有用到全部的模型，例如sklearn和xgb我们就没有用。

class SklearnWrapper(object):
    # Sklearn类，包含constructor,两个函数：训练和预测函数
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train, **kwargs):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    
class XgbWrapper(object):
    # XGB类，同样包含constructor,两个函数：训练和预测函数
    def __init__(self, params=None):
        self.param = params
        self.nrounds = params.pop('nrounds', 60000)
        self.early_stop_rounds = params.pop('early_stop_rounds', 2000)

    def train(self, x_train, y_train, **kwargs):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dvalid = xgb.DMatrix(data=kwargs['x_val'], label=kwargs['y_val'])
        
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        
        self.model = xgb.train(dtrain=dtrain, num_boost_round=self.nrounds, evals=watchlist, early_stopping_rounds=self.early_stop_rounds, 
                               verbose_eval=1000, params=self.param)

    def predict(self, x):
        return self.model.predict(xgb.DMatrix(x), ntree_limit=self.model.best_ntree_limit)

    
class LGBWrapper(object):
    # LGB类，同样包含constructor,两个函数：训练和预测函数，但不同的是多了一个importance函数，用于输出特征的重要性
    def __init__(self, params=None):
        self.param = params
        self.num_rounds = params.pop('nrounds', 60000)
        self.early_stop_rounds = params.pop('early_stop_rounds', 2000)

    def train(self, x_train, y_train, **kwargs):
        dtrain = lgb.Dataset(x_train, label=y_train)
        dvalid = lgb.Dataset(kwargs['x_val'], label=kwargs['y_val'])

        watchlist = [dtrain, dvalid]
        
        print('training LightGBM with params: ', self.param)
        self.model = lgb.train(
                  self.param,
                  train_set=dtrain,
                  num_boost_round=self.num_rounds,
                  valid_sets=watchlist,
                  verbose_eval=1000,
                  early_stopping_rounds=self.early_stop_rounds
        )

    def predict(self, x):
        return self.model.predict(x, num_iteration=self.model.best_iteration)
    
    def importance(self):
        lgb.plot_importance(self.model, max_num_features=300)
        return 0
    


# In[75]:


from contextlib import contextmanager
import time
@contextmanager
# 定义一个计时器，用于计算我们训练的时间
def timer(task_name="timer"):
    # a timer cm from https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    print("----{} started".format(task_name))
    t0 = time.time()
    yield
    print("----{} done in {:.0f} seconds".format(task_name, time.time() - t0))


# In[76]:


# 这一段预先训练一次lightGBM，是为了使用LGB中的feature importance函数来确定哪些特征可以被剔除
# 当然在结束之后为了节省资源，我们就不需要再用这个了，注释掉即可
from collections import defaultdict
import random
# LightGBM
"""

lgbm_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'nrounds': 50000,
    'early_stop_rounds': 2000,
    # trainable params
    'max_depth': 9,
    'num_leaves': 70,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.9,
    'bagging_freq': 8,
    'learning_rate': 0.019,
    'verbose': 0
}
lgb_wrapper = LGBWrapper(lgbm_params)

with timer('Training LightGBM'):
    lgb_oof_train, lgb_oof_test = get_oof(lgb_wrapper, X_train_non_null.drop(['AdoptionSpeed'], axis=1), X_train_non_null['AdoptionSpeed'].values.astype(int), X_test_non_null, groups=rescuer_ids)

lgb_wrapper.importance()
"""


# In[77]:


"""
importance = lgb_wrapper.model.feature_importance()
names = lgb_wrapper.model.feature_name()
feature_importance = pd.DataFrame({'feature_name':names,'importance':importance} )
"""


# In[78]:


# feature_importance


# In[79]:


# feature_no_result = feature_importance[feature_importance['importance']<130]


# In[80]:


# list_column=feature_no_result['feature_name'].tolist()


# In[81]:


X_train_drop = X_train.fillna(-1)
X_test_drop = X_test.fillna(-1)
# for index in list_column:
#     X_train_drop = X_train_drop.drop(columns=[index])


# In[82]:


X_train_drop.to_csv('X_train_drop.csv', index = True)


# In[83]:


from sklearn.model_selection import StratifiedKFold, GroupKFold

# 通过KCV，将训练集分为5份，分别进行训练。
n_splits = 5
# kfold = GroupKFold(n_splits=n_splits)
split_index = []
# for train_idx, valid_idx in kfold.split(train, train['AdoptionSpeed'], train['RescuerID']):
#     split_index.append((train_idx, valid_idx))

kfold = StratifiedKFold(n_splits=n_splits, random_state=1991)
for train_idx, valid_idx in kfold.split(X_train_drop, X_train_drop['AdoptionSpeed']):
    split_index.append((train_idx, valid_idx))


# In[84]:


# 定义一部分几个模型通用的超参数
early_stop = 300
verbose_eval = 100
num_rounds = 10000

# 再添加一部分不需要被训练的列，
drop_columns = ['PetID', 'Name', 'RescuerID', 'AdoptionSpeed',  
                   'main_breed_Type', 'main_breed_BreedName', 'second_breed_Type', 'second_breed_BreedName',
                   'Description', 'sentiment_entities', 'meta_annots_top_desc','meta_desc',
                   'Chinese_desc', 'English_desc']

to_drop_columns.extend(drop_columns)


# In[85]:


# 去除重复的列
to_drop_columns = list(set(to_drop_columns))


# In[86]:


# torch imports
from torch import nn
import torch
from torch.nn import functional as F
from torchvision.models import resnet50, resnet34, densenet201, densenet121
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import TensorDataset

torch.manual_seed(1991)
torch.cuda.manual_seed(1991)
torch.backends.cudnn.deterministic = True


# In[87]:


# 在NN模型中需要预处理的列
fm_cols = ['Type','Age','Breed1','Breed2','Gender','Color1','Color2','Color3','MaturitySize',
           'FurLength','Vaccinated','Dewormed','Sterilized','Health','State','Breed_full',
           'Color_full', 'hard_interaction']
fm_data = X_temp[fm_cols]
fm_values = []
for c in fm_cols:
    fm_data.loc[:,c] = fm_data[c].fillna(0)
    fm_data.loc[:,c] = c+'_'+fm_data[c].astype(str)
    fm_values+=fm_data[c].unique().tolist()


# In[88]:


from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()
lbe.fit(fm_values)
for c in fm_cols:
    fm_data.loc[:,c] = lbe.transform(fm_data[c])


# In[89]:


numerical_cols = [x for x in X_train_drop.columns if x not in to_drop_columns+fm_cols]


# In[90]:


# 将numerical feature单独拿出
numerical_feats = []
for c in numerical_cols:
    numerical_feats.append(X_temp[c].fillna(0))
    
# for c in range(1920):
#     numerical_feats.append(raw_img_features['resnet50_%d'%c].fillna(0))

numerical_feats = np.vstack(numerical_feats).T
# numerical_feats = stdscaler.fit_transform(numerical_feats)


# In[91]:


numerical_feats.shape


# In[92]:


from torch.utils.data import Dataset, DataLoader

MAX_LEN = 400
class PetDesDataset(Dataset):
    def __init__(self, sentences, pos, fm_data, numerical_feat,
                 mode='train', target=None):
        super(PetDesDataset, self).__init__()
        self.data = sentences
        self.pos = pos
        self.target = target
        self.mode = mode
        self.fm_data = fm_data
        self.fm_dim = fm_data.shape[1]
        self.numerical_feat = numerical_feat

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index not in range(0, self.__len__()):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        sentence_len = min(MAX_LEN, len(self.data[index]))
        sentence = torch.tensor(self.data[index][:sentence_len])
        fm_data = self.fm_data[index,:]
        pos = torch.tensor(self.pos[index][:sentence_len])

        if self.mode != 'test':  # , pos, tag
            return sentence, pos, sentence_len, fm_data, self.numerical_feat[index], self.target[index]  # , clf_label
        else:
            return sentence, pos, sentence_len, fm_data, self.numerical_feat[index]
        
def nn_collate(batch):
    has_label = len(batch[0]) == 6
    if has_label:
        sentences, poses, lengths, fm_data, numerical_feats, label = zip(*batch)
        sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True).type(torch.LongTensor)
        poses = nn.utils.rnn.pad_sequence(poses, batch_first=True).type(torch.LongTensor)
        lengths = torch.LongTensor(lengths)
        fm_data = torch.LongTensor(fm_data)
        numerical_feats = torch.FloatTensor(numerical_feats)
        label = torch.FloatTensor(label)
        return sentences, poses, lengths, fm_data, numerical_feats, label
    else:
        sentences, poses, lengths, fm_data, numerical_feats = zip(*batch)
        sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True).type(torch.LongTensor)
        poses = nn.utils.rnn.pad_sequence(poses, batch_first=True).type(torch.LongTensor)
        lengths = torch.LongTensor(lengths)
        fm_data = torch.LongTensor(fm_data)
        numerical_feats = torch.FloatTensor(numerical_feats)
        return sentences, poses, lengths, fm_data, numerical_feats


# In[93]:



def get_mask(sequences_batch, sequences_lengths, cpu=False):
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    if cpu:
        return mask
    else:
        return mask.cuda()
class Attention(nn.Module):
    def __init__(self, feature_dim, bias=True, head_num=1, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.head_num = head_num
        weight = torch.zeros(feature_dim, self.head_num)
        bias = torch.zeros((1, 1, self.head_num))
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        self.b = nn.Parameter(bias)

    def forward(self, x, mask=None):
        batch_size, step_dim, feature_dim = x.size()
        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),  # B*L*H
            self.weight  # B*H*1
        ).view(-1, step_dim, self.head_num)  # B*L*head
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        if mask is not None:
            eij = eij * mask - 99999.9 * (1 - mask)
        a = torch.softmax(eij, dim=1)

        weighted_input = torch.bmm(x.permute((0,2,1)),
                                   a).view(batch_size, -1)
        return weighted_input


# In[94]:



embed_size = 300
class FM(nn.Module):

    def __init__(self, max_features, feat_len, embed_size):
        super(FM, self).__init__()
        self.bias_emb = nn.Embedding(max_features, 1)
        self.fm_emb = nn.Embedding(max_features, embed_size)
        self.feat_len = feat_len
        self.embed_size = embed_size

    def forward(self, x):
        bias = self.bias_emb(x)
        bias = torch.sum(bias,1) # N * 1

        # second order term
        # square of sum
        emb = self.fm_emb(x)
        sum_feature_emb = torch.sum(emb, 1) # N * k
        square_sum_feature_emb = sum_feature_emb*sum_feature_emb

        # sum of square
        square_feature_emb = emb * emb
        sum_square_feature_emb = torch.sum(square_feature_emb, 1) # N * k

        second_order = 0.5*(square_sum_feature_emb-sum_square_feature_emb) # N *k
        return bias+second_order, emb.view(-1, self.feat_len*self.embed_size)


# In[95]:



class FmNlpModel(nn.Module):
    def turn_on_embedding(self):
        self.embedding.weight.requires_grad = True

    def __init__(self, hidden_size=64, init_embedding=None, head_num=3,
                 fm_embed_size=8, fm_feat_len=10, fm_max_feature = 300, numerical_dim = 300,
                 nb_word = 40000, nb_pos = 200, pos_emb_size = 10):
        super(FmNlpModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(nb_word, 300, padding_idx=0)
        self.pos_embedding = nn.Embedding(nb_pos+100, pos_emb_size, padding_idx=0)
        
        if init_embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(init_embedding))
        self.embedding.weight.requires_grad = False
        
        self.fm = FM(fm_max_feature, fm_feat_len, fm_embed_size)

        self.dropout = nn.Dropout(0.1)
        self.attention_gru = Attention(feature_dim=self.hidden_size * 2, head_num=head_num)
        self.gru = nn.GRU(embed_size+pos_emb_size, hidden_size, bidirectional=True, batch_first=True) #
        self.gru2 = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        
        self.dnn = nn.Sequential(
            nn.BatchNorm1d(numerical_dim),
            nn.Dropout(0.1),
            nn.Linear(numerical_dim, 256),
            nn.ELU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ELU(inplace=True),
        )
        
        self.rnn_dnn = nn.Sequential(
            nn.BatchNorm1d(fm_embed_size+fm_feat_len*fm_embed_size +2*head_num * hidden_size+128), #
            nn.Dropout(0.3),
            nn.Linear(fm_embed_size+fm_feat_len*fm_embed_size+2*head_num * hidden_size+128, 32),
            nn.ELU(inplace=True),
        )
        self.logit = nn.Sequential(
            nn.Linear(32,1)
        )
#         self.apply(_init_esim_weights)

    def forward(self, x, pos_x, len_x, fm_x, numerical_x):
        
        fm_result, fm_embed = self.fm(fm_x)
        
        sentence_mask = get_mask(x, len_x)
        x = x * sentence_mask.long()
        sentence_mask = torch.unsqueeze(sentence_mask, -1)

        h_embedding = self.embedding(x)
        h_pos_embedding = self.pos_embedding(pos_x)
        h_embedding = torch.cat([h_embedding, h_pos_embedding],2)
        
        h_embedding = self.dropout(h_embedding)
        
        sorted_seq_lengths, indices = torch.sort(len_x, descending=True)
        # 排序前的顺序是对下标再排一次序
        _, desorted_indices = torch.sort(indices, descending=False)
        h_embedding = h_embedding[indices]
        packed_inputs = nn.utils.rnn.pack_padded_sequence(h_embedding, sorted_seq_lengths, batch_first=True)
        
        h_gru, _ = self.gru(packed_inputs)
        h_gru2, _ = self.gru2(h_gru)  # sentence_mask.expand_as(h_lstm)
        
        h_gru2, _ = nn.utils.rnn.pad_packed_sequence(h_gru2, batch_first=True)
        h_gru2 = h_gru2[desorted_indices]
        att_pool_gru = self.attention_gru(h_gru2, sentence_mask)
        
        numerical_x = self.dnn(numerical_x)

        x = torch.cat([att_pool_gru,fm_result,fm_embed,numerical_x],1) # 
        feat = self.rnn_dnn(x)
        out = self.logit(feat)

        return out, feat


# In[96]:



X_train_numerical = numerical_feats[0:len(train)]
X_test_numerical = numerical_feats[len(train):]

X_train_seq = pd.Series(eng_sequences[0:len(train)])
X_test_seq = pd.Series(eng_sequences[len(train):])

X_train_pos_seq = pd.Series(pos_sequences[0:len(train)])
X_test_pos_seq = pd.Series(pos_sequences[len(train):])

X_train_fm = fm_data.iloc[0:len(train)].values
X_test_fm = fm_data.iloc[len(train):].values

Y_train = X_temp.iloc[0:len(train)]['AdoptionSpeed'].values


# In[97]:


# 通过Glove建立文本模型
def load_glove():
    EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'

    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in (open(EMBEDDING_FILE)))
    return embeddings_index

glove_emb = load_glove()

embedding_matrix, nb_words, nb_oov = build_emb_matrix(word_dict, glove_emb)
print(nb_words, nb_oov)
del glove_emb
gc.collect()


# In[98]:


nb_pos = len(pos_dict)


# In[99]:


# 训练
train_epochs = 6
loss_fn = torch.nn.MSELoss().cuda()
oof_train_nlp = np.zeros((X_train_drop.shape[0], 32+1))
oof_test_nlp = []

test_set = PetDesDataset(X_test_seq.tolist(), X_test_pos_seq.tolist(), X_test_fm, X_test_numerical, mode='test')
test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=1, pin_memory=True,
                                collate_fn=nn_collate)
qwks = []
rmses = []

for n_fold, (train_idx, valid_idx) in enumerate(split_index): 
        
    print('fold:', n_fold)
    hist = histogram(Y_train[train_idx].astype(int), 
                     int(np.min(X_train_drop['AdoptionSpeed'])), 
                     int(np.max(X_train_drop['AdoptionSpeed'])))
    tr_cdf = get_cdf(hist)
    
    training_set = PetDesDataset(X_train_seq[train_idx].tolist(), 
                                 X_train_pos_seq[train_idx].tolist(),
                                 X_train_fm[train_idx], 
                                 X_train_numerical[train_idx], target = Y_train[train_idx])
    
    validation_set = PetDesDataset(X_train_seq[valid_idx].tolist(), 
                                   X_train_pos_seq[valid_idx].tolist(),
                                   X_train_fm[valid_idx], 
                                  X_train_numerical[valid_idx],target = Y_train[valid_idx])
    
    training_loader = DataLoader(training_set, batch_size=512, shuffle=True, num_workers=1,
                                collate_fn=nn_collate)
    validation_loader = DataLoader(validation_set, batch_size=512, shuffle=False, num_workers=1,
                                collate_fn=nn_collate)
    
    model = FmNlpModel(hidden_size=48, init_embedding=embedding_matrix, head_num=10, 
                      fm_embed_size=10, fm_feat_len=X_train_fm.shape[1], fm_max_feature=len(fm_values),
                      numerical_dim=X_train_numerical.shape[1],
                      nb_word=nb_words, nb_pos=nb_pos, pos_emb_size=10)
    model.cuda()
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=0.0001)
    
    iteration = 0
    min_val_loss = 100
    since = time.time()
    
    for epoch in range(train_epochs):       
        scheduler.step()
        model.train()
        for sentences, poses, lengths, x_fm, x_numerical, labels in training_loader:
            iteration += 1
            sentences = sentences.cuda()
            poses = poses.cuda()
            lengths = lengths.cuda()
            x_fm = x_fm.cuda()
            x_numerical = x_numerical.cuda()
            labels = labels.type(torch.FloatTensor).cuda().view(-1, 1)

            pred,_ = model(sentences, poses, lengths, x_fm, x_numerical)
            loss = loss_fn(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_predicts = []
        val_feats = []
        with torch.no_grad():
            for sentences, poses, lengths, x_fm, x_numerical, labels in validation_loader:
                sentences = sentences.cuda()
                poses = poses.cuda()
                lengths = lengths.cuda()
                x_fm = x_fm.cuda()
                x_numerical = x_numerical.cuda()
                labels = labels.type(torch.FloatTensor).cuda()#.view(-1, 1)
                v_pred, v_feat = model(sentences, poses, lengths, x_fm, x_numerical)
                val_predicts.append(v_pred.cpu().numpy())
                val_feats.append(v_feat.cpu().numpy())

        val_predicts = np.concatenate(val_predicts)
        val_feats = np.vstack(val_feats)
        val_loss = rmse(Y_train[valid_idx], val_predicts)
        if val_loss<min_val_loss:
            min_val_loss = val_loss
            oof_train_nlp[valid_idx,:] = np.hstack([val_feats, val_predicts])
            test_feats = []
            test_preds = []
            with torch.no_grad():
                for sentences, poses, lengths, x_fm, x_numerical in test_loader:
                    sentences = sentences.cuda()
                    poses = poses.cuda()
                    lengths = lengths.cuda()
                    x_fm = x_fm.cuda()
                    x_numerical = x_numerical.cuda()
                    v_pred, feat = model(sentences, poses, lengths, x_fm, x_numerical)
                    test_preds.append(v_pred.cpu().numpy())
                    test_feats.append(feat.cpu().numpy())
            test_feats = np.hstack([np.vstack(test_feats), np.concatenate(test_preds)])
#             pred_test_y_k = getTestScore2(val_predicts.flatten(), tr_cdf)
#             qwk = quadratic_weighted_kappa(Y_train[valid_idx], pred_test_y_k)
#             print(epoch, "val loss:", val_loss, "val QWK_2 = ", qwk, "elapsed time:", time.time()-since)
    oof_test_nlp.append(test_feats)
    del model
    del training_set
    del validation_set 
    del sentences
    del lengths
    del x_fm
    del x_numerical
    del poses
    gc.collect()
    torch.cuda.empty_cache()
    
#     qwks.append(qwk)
#     rmses.append(min_val_loss)


# In[100]:


pynvml.nvmlInit()
# 这里的1是GPU id
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(meminfo.total) #第二块显卡总的显存大小
print(meminfo.used)#这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
print(meminfo.free) #第二块显卡剩余显存大小
print(pynvml.nvmlDeviceGetCount())#显示有几块GPU


# In[101]:


oof_test_nlp = np.mean(oof_test_nlp, axis=0)


# In[102]:


print(oof_train_nlp)


# In[103]:


oof_test_nlp[:,-1]


# In[104]:


#for item in to_drop_columns:
#    X_temp = X_temp.drop([item], axis=1)

# Check final df shape:
print('X shape: {}'.format(X_temp.shape))


# In[105]:


# 再一次将训练集和测试集分开
X_train = X_temp.loc[np.isfinite(X_temp.AdoptionSpeed), :]
X_test = X_temp.loc[~np.isfinite(X_temp.AdoptionSpeed), :]

# 剔除AdoptionSpeed列
X_test = X_test.drop(['AdoptionSpeed'], axis=1)


print('X_train shape: {}'.format(X_train.shape))
print('X_test shape: {}'.format(X_test.shape))

assert X_train.shape[0] == train.shape[0]
assert X_test.shape[0] == test.shape[0]


# 确认两者长度是否一致
train_cols = X_train.columns.tolist()
train_cols.remove('AdoptionSpeed')

test_cols = X_test.columns.tolist()

assert np.all(train_cols == test_cols)
rescuer_ids = rescuer_ids[:len(X_train)]
assert len(rescuer_ids) == len(X_train)


# In[106]:


X_train_drop = X_train.fillna(-1)
X_test_drop = X_test.fillna(-1)
ntrain = X_train_drop.shape[0]
ntest = X_test_drop.shape[0]


# In[107]:


import lightgbm as lgb
import xgboost as xgb


# In[108]:


features = [x for x in X_train_drop.columns if x not in to_drop_columns]


# In[109]:


# features.remove('AdoptionSpeed')


# In[110]:


# Additional parameters:
early_stop = 300
verbose_eval = 100
num_rounds = 10000


# In[111]:


# LGB训练，这一段在report中有介绍，可以看那边
params = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 37,
          'max_depth': 6,
          'learning_rate': 0.01,
          'subsample': 0.85,
          'feature_fraction': 0.7,
          'lambda_l1':0.01,
          'verbosity': -1,
         }

oof_train_lgb = np.zeros((X_train_drop.shape[0]))
oof_test_lgb = []
qwks = []
rmses = []

for n_fold, (train_idx, valid_idx) in enumerate(split_index):
    since = time.time()
    X_tr = X_train_drop.iloc[train_idx]
    X_val = X_train_drop.iloc[valid_idx]

    y_tr = X_tr['AdoptionSpeed'].values    
    y_val = X_val['AdoptionSpeed'].values
        
    d_train = lgb.Dataset(X_tr[features], label=y_tr,
#                          categorical_feature=['Breed1','Color1','Breed2','State','Breed_full','Color_full']
                         )
    d_valid = lgb.Dataset(X_val[features], label=y_val, reference=d_train)
    watchlist = [d_valid]
    
    print('training LGB:')
    lgb_model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=500,
                      early_stopping_rounds=100,
                     )
    
    val_pred = lgb_model.predict(X_val[features])
    test_pred = lgb_model.predict(X_test_drop[features])
    train_pred = lgb_model.predict(X_tr[features])
    
    oof_train_lgb[valid_idx] = val_pred
    oof_test_lgb.append(test_pred)
               
    hist = histogram(X_tr['AdoptionSpeed'].astype(int), 
                    int(np.min(X_train['AdoptionSpeed'])), 
                    int(np.max(X_train['AdoptionSpeed'])))
    tr_cdf = get_cdf(hist)
    _, cutoff = getScore(train_pred, tr_cdf, True)
    pred_test_y_k = getTestScore(val_pred, cutoff)
    qwk = quadratic_weighted_kappa(X_val['AdoptionSpeed'].values, pred_test_y_k)
    print("QWK_1 = ", qwk)
    qwks.append(qwk)
    rmses.append(rmse(X_val['AdoptionSpeed'].values, val_pred))
    
#     pred_test_y_k = getTestScore2(val_pred, tr_cdf)
#     qwk = quadratic_weighted_kappa(X_val['AdoptionSpeed'].values, pred_test_y_k)
#     qwks.append(qwk)
#     rmses.append(rmse(X_val['AdoptionSpeed'].values, val_pred))
#     print("QWK_2 = ", qwk,'elapsed time:',time.time()-since)

print('overall rmse: %.5f'%rmse(oof_train_lgb, X_train_drop['AdoptionSpeed']))
print('mean rmse =', np.mean(rmses), 'rmse std =', np.std(rmses))
print('mean QWK =', np.mean(qwks), 'std QWK =', np.std(qwks))


# In[112]:


"""
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

features.extend(['AdoptionSpeed'])
train_data = X_train_drop[features]  # 读取数据
y = train_data.pop('AdoptionSpeed').values   # 用pop方式将训练数据中的标签值y取出来，作为训练目标，这里的‘30’是标签的列名

col = train_data.columns
x = train_data[col].values  # 剩下的列作为训练数据
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2, random_state=0)   # 分训练集和验证集
train = lgb.Dataset(train_x, train_y)
valid = lgb.Dataset(valid_x, valid_y, reference=train)

# since = time.time()
# X_tr = train_data.iloc[train_idx]
# X_val = train_data.iloc[valid_idx]

# y_tr = X_tr['AdoptionSpeed'].values    
# y_val = X_val['AdoptionSpeed'].values

watchlist = [valid]

oof_train_lgb = np.zeros((X_train_drop.shape[0]))
oof_test_lgb = []
qwks = []
rmses = []

params = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 70,
          'max_depth': 9,
          'learning_rate': 0.01,
          'subsample': 0.85,
          'feature_fraction': 0.7,
          'lambda_l1':0.01,
          'verbosity': -1,
         }



parameters = {
              'max_depth': [15, 20, 25, 30, 35],
              'num_leaves':[20, 30, 40, 50, 60],
              'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
              'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              'bagging_freq': [2, 4, 5, 6, 8],
              'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
              'lambda_l2': [0, 10, 15, 35, 40],
              'cat_smooth': [1, 10, 15, 20, 35]
}

gbm = lgb.train(params,
                      train_set=train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=500,
                      early_stopping_rounds=100,
                     )

# 有了gridsearch我们便不需要fit函数

gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch.fit(train_x, train_y)

print("Best score: %0.3f" % gsearch.best_score_)
print("Best parameters set:")
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

"""


# In[113]:


"""
from hyperopt import fmin, tpe, hp, partial
from sklearn.model_selection import train_test_split

features.extend(['AdoptionSpeed'])
train_data = X_train_drop[features]  # 读取数据
y = train_data.pop('AdoptionSpeed').values   # 用pop方式将训练数据中的标签值y取出来，作为训练目标，这里的‘30’是标签的列名

col = train_data.columns
x = train_data[col].values  # 剩下的列作为训练数据
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2, random_state=0)   # 分训练集和验证集
train = lgb.Dataset(train_x, train_y)
valid = lgb.Dataset(valid_x, valid_y, reference=train)
watchlist = [valid]

oof_train_lgb = np.zeros((X_train_drop.shape[0]))
oof_test_lgb = []
qwks = []
rmses = []
# 自定义hyperopt的参数空间
space = {"max_depth": hp.randint("max_depth", 15),
         'learning_rate': hp.uniform('learning_rate', 1e-3, 5e-1),
         "bagging_fraction": hp.randint("bagging_fraction", 5),
         "num_leaves": hp.randint("num_leaves", 6),
         }

def argsDict_tranform(argsDict, isPrint=False):
    argsDict["max_depth"] = argsDict["max_depth"] + 5
    argsDict["learning_rate"] = argsDict["learning_rate"] * 0.02 + 0.05
    argsDict["bagging_fraction"] = argsDict["bagging_fraction"] * 0.1 + 0.5
    argsDict["num_leaves"] = argsDict["num_leaves"] * 3 + 10
    if isPrint:
        print(argsDict)
    else:
        pass

    return argsDict
"""


# In[114]:


"""
from sklearn.metrics import mean_squared_error

def lightgbm_factory(argsDict):
    argsDict = argsDict_tranform(argsDict)

    params = {'max_depth': argsDict['max_depth'],  # 最大深度
              'learning_rate': argsDict['learning_rate'],  # 学习率
              'bagging_fraction': argsDict['bagging_fraction'],  # bagging采样数
              'num_leaves': argsDict['num_leaves'],  # 终点节点最小样本占比的和
              'application': 'regression',
              'subsample': 0.85,
              'feature_fraction': 0.7,  # 样本列采样
              'lambda_l1': 0.01,  # L1 正则化
              'bagging_seed': 100,  # 随机种子,light中默认为100
              }
    #rmse
    params['metric'] = ['rmse']

    model_lgb = lgb.train(params,
                      train_set=train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=500,
                      early_stopping_rounds=100,
                     )

    return get_tranformer_score(model_lgb)

def get_tranformer_score(tranformer):

    model = tranformer
    prediction = model.predict(valid_x, num_iteration=model.best_iteration)

    return mean_squared_error(valid_y, prediction)
    
"""


# In[115]:


# 开始使用hyperopt进行自动调参
# algo = partial(tpe.suggest, n_startup_jobs=1)
# best = fmin(lightgbm_factory, space, algo=algo, max_evals=20, pass_expr_memo_ctrl=None)

# print('best :', best)


# In[116]:


# oof_train_lgb


# In[117]:


"""
hist = histogram(X_tr['AdoptionSpeed'].astype(int), 
                int(np.min(X_train['AdoptionSpeed'])), 
                int(np.max(X_train['AdoptionSpeed'])))
tr_cdf = get_cdf(hist)
_, cutoff = getScore(train_pred, tr_cdf, True)
pred_test_y_k = getTestScore(val_pred, cutoff)
qwk = quadratic_weighted_kappa(X_val['AdoptionSpeed'].values, pred_test_y_k)
print("QWK_1 = ", qwk)
"""


# In[118]:


# 接下来训练catboost
from catboost import CatBoostRegressor, Pool
import time

features = [x for x in X_train_drop.columns if x not in to_drop_columns]

cat_index = []
for idx, c in enumerate(features):
    if c in ['Type','Breed1','Breed2','Gender','Color1','Color2','Color3','State','Breed_full',
           'Color_full', 'hard_interaction','img_CLUSTER_0']:
        cat_index.append(idx)
        
oof_train_cat = np.zeros((X_train_drop.shape[0]))
oof_test_cat = []
qwks = []
rmses = []

for n_fold, (train_idx, valid_idx) in enumerate(split_index):
    since = time.time()
    X_tr = X_train_drop.iloc[train_idx]
    X_val = X_train_drop.iloc[valid_idx]

    y_tr = X_tr['AdoptionSpeed'].values 
    y_val = X_val['AdoptionSpeed'].values 
        
    
    eval_dataset = Pool(X_val[features].values,
                    y_val,
                   cat_index)
    print('training Catboost:')
    model = CatBoostRegressor(learning_rate=0.01,  depth=6, task_type = "GPU", l2_leaf_reg=2)
    model.fit(X_tr[features].values,
              y_tr,
              eval_set=eval_dataset,
              cat_features= cat_index,
              verbose=False)
    
    val_pred = model.predict(eval_dataset)
    test_pred = model.predict(X_test_drop[features])
    
    oof_train_cat[valid_idx] = val_pred
    oof_test_cat.append(test_pred)
               
    hist = histogram(X_tr['AdoptionSpeed'].astype(int), 
                      int(np.min(X_train['AdoptionSpeed'])), 
                      int(np.max(X_train['AdoptionSpeed'])))
    tr_cdf = get_cdf(hist)
    
    pred_test_y_k = getTestScore2(val_pred, tr_cdf)
    qwk = quadratic_weighted_kappa(X_val['AdoptionSpeed'].values, pred_test_y_k)
    qwks.append(qwk)
    rmses.append(rmse(X_val['AdoptionSpeed'].values, val_pred))
    print('rmse=',rmses[-1],"QWK_2 = ", qwk,'elapsed time:',time.time()-since)


# In[119]:


print('overall rmse: %.5f'%rmse(oof_train_cat, X_train_drop['AdoptionSpeed']))
print('mean rmse =', np.mean(rmses), 'rmse std =', np.std(rmses))
print('mean QWK =', np.mean(qwks), 'std QWK =', np.std(qwks))


# In[120]:


features = [x for x in X_train_drop.columns if x not in to_drop_columns]
# features.remove('AdoptionSpeed')
xgb_features = features


# In[121]:


"""
params = {
        'objective': 'reg:linear', #huber
        'eval_metric':'rmse',
        'eta': 0.01,
        'tree_method':'gpu_hist',
        'max_depth': 9,  
        'subsample': 0.85,  
        'colsample_bytree': 0.7,     
        'alpha': 0.01,  
    } 

oof_train_xgb = np.zeros((X_train_drop.shape[0]))
oof_test_xgb = []
qwks = []

i = 0
test_set = xgb.DMatrix(X_test_drop[xgb_features])

for n_fold, (train_idx, valid_idx) in enumerate(split_index):  
    X_tr = X_train_drop.iloc[train_idx]
    X_val = X_train_drop.iloc[valid_idx]
    
    y_tr = X_tr['AdoptionSpeed'].values    
    y_val = X_val['AdoptionSpeed'].values
        
    d_train = xgb.DMatrix(X_tr[xgb_features], y_tr)
    d_valid = xgb.DMatrix(X_val[xgb_features], y_val)
    watchlist = [d_valid]
    since = time.time()
    print('training XGB:')
    model = xgb.train(params, d_train, num_boost_round = 10000, evals=[(d_valid,'val')],
                     early_stopping_rounds=100, #feval=xgb_eval_kappa,
                     verbose_eval=500)
    
    val_pred = model.predict(d_valid)
    test_pred = model.predict(test_set)
    
    oof_train_xgb[valid_idx] = val_pred
    oof_test_xgb.append(test_pred)
    
#     hist = histogram(X_tr['AdoptionSpeed'].astype(int), 
#                      int(np.min(X_train['AdoptionSpeed'])), 
#                      int(np.max(X_train['AdoptionSpeed'])))
#     tr_cdf = get_cdf(hist)
    
#     pred_test_y_k = getTestScore2(val_pred, tr_cdf)
#     qwk = quadratic_weighted_kappa(X_val['AdoptionSpeed'].values, pred_test_y_k)
#     qwks.append(qwk)
#     print("QWK_2 = ", qwk,'elapsed time:',time.time()-since)
"""


# In[122]:


"""
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

train_data = X_train_drop  # 读取数据
y = train_data.pop('AdoptionSpeed').values   # 用pop方式将训练数据中的标签值y取出来，作为训练目标

col = train_data.columns   
x = train_data[col].values  # 剩下的列作为训练数据
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.333, random_state=0)   # 分训练集和验证集

parameters = {
              'max_depth': [5, 10, 15, 20, 25],
              'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
              'n_estimators': [500, 1000, 2000, 3000, 5000],
              'min_child_weight': [0, 2, 5, 10, 20],
              'max_delta_step': [0, 0.2, 0.6, 1, 2],
              'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
              'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
              'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
              'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
              'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]

}

xlf = xgb.XGBRegressor(
                    objective = 'reg:linear',
                    eval_metric='rmse',
                    tree_method='gpu_hist',
                    device='gpu',
                    silent= 1,
                    # seed= 1337,
                    num_boost_round= 10000,
                    verbose_eval=100,
                    # trainable params
                    subsample= 0.85,
                    colsample_bytree= 0.6,
                    gamma= 0.65,
                    eta = 0.01,
                    max_depth= 4,
                    min_child_weight= 5.0,
                    n_estimators= 1000,)
            
# 有了gridsearch我们便不需要fit函数
gsearch = GridSearchCV(xlf, param_grid=parameters, scoring='accuracy', cv=3)
gsearch.fit(train_x, train_y)

print("Best score: %0.3f" % gsearch.best_score_)
print("Best parameters set:")
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

"""


# In[123]:


np.corrcoef([np.mean(oof_test_lgb,axis=0), 
             #np.mean(oof_test_lgb2,axis=0),
             np.mean(oof_test_cat,axis=0),
             #np.mean(oof_test_xgb,axis=0),
             oof_test_nlp[:,-1]
            ])


# In[124]:


from sklearn.linear_model import Ridge

Y_train = X_train_drop.iloc[0:len(train)]['AdoptionSpeed'].values
# 生成训练融合矩阵
X_train_stacking = np.vstack([oof_train_lgb,  
                              oof_train_cat,
                             oof_train_nlp[:,-1]
                             ]).T
#  生成测试融合矩阵
X_test_stacking = np.vstack([np.mean(oof_test_lgb, axis=0),

                             np.mean(oof_test_cat,axis=0),
                            oof_test_nlp[:,-1]
                            ]).T

stacking_train = np.zeros((X_train_drop.shape[0]))
stacking_test = []
rmses, qwks = [], []
# 对每一个fold分别进行岭回归训练
for n_fold, (train_idx, valid_idx) in enumerate(split_index):
    
    X_tr = X_train_stacking[train_idx]
    X_val = X_train_stacking[valid_idx]
    
    y_tr = X_train_drop.iloc[train_idx]['AdoptionSpeed'].values    
    y_val = X_train_drop.iloc[valid_idx]['AdoptionSpeed'].values
        
    since = time.time()
    
    print('training Ridge:')
    model = Ridge(alpha=1)
    model.fit(X_tr, y_tr)
    print(model.coef_)
    
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test_stacking)
    
    stacking_train[valid_idx] = val_pred
    stacking_test.append(test_pred)
    loss = rmse(Y_train[valid_idx], val_pred)
    hist = histogram(y_tr.astype(int), 
                     int(np.min(X_train['AdoptionSpeed'])), 
                     int(np.max(X_train['AdoptionSpeed'])))
    tr_cdf = get_cdf(hist)
    
    pred_test_y_k = getTestScore2(val_pred, tr_cdf)
    qwk = quadratic_weighted_kappa(y_val, pred_test_y_k)
    qwks.append(qwk)
    rmses.append(loss)
    print("RMSE=",loss, "QWK_2 = ", qwk,'elapsed time:',time.time()-since)
stacking_test = np.mean(stacking_test, axis=0)
print('mean rmse:',np.mean(rmses), 'rmse std:', np.std(rmses))
print('mean qwk:', np.mean(qwks), 'qwk std:', np.std(qwks))


# In[125]:


# 计算融合之后的QWK
hist = histogram(X_train['AdoptionSpeed'].astype(int), 
                 int(np.min(X_train['AdoptionSpeed'])), 
                 int(np.max(X_train['AdoptionSpeed'])))
tr_cdf = get_cdf(hist)
train_predictions = getTestScore2(stacking_train, tr_cdf)
test_predictions = getTestScore2(stacking_test, tr_cdf)
qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, train_predictions)
print("QWK = ", qwk)


# In[126]:


# 最终生成结果文件，上传

submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions.astype(np.int32)})
submission.to_csv('submission.csv', index=False)


# In[127]:


submission.head()

