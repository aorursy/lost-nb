#!/usr/bin/env python
# coding: utf-8



# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
import warnings
import json
import os

# Options
warnings.filterwarnings(action='ignore')
sns.set_palette('YlGnBu')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")




def reset_dfs():
    global train_df, test_df
    train_df = pd.read_csv('../input/train/train.csv')
    test_df = pd.read_csv('../input/test/test.csv')
reset_dfs()




'''
Score:  0.07917
   AdoptionSpeed  Type_2
0              2       1
1              0       1
''';




reset_dfs()
plt.figure(figsize=(8,5))
sns.countplot(train_df['AdoptionSpeed']);




g = sns.catplot("AdoptionSpeed", col="Type",
                hue="Gender", data=train_df,
                kind="count", height=5, aspect=.8);




def sample(df, target_val_sets):
    for target_val_set in target_val_sets:
        df_class_0 = df[df['AdoptionSpeed'] == target_val_set[0]]
        count_1 = df['AdoptionSpeed'].value_counts()[target_val_set[1]]
        df_class_0_sampled = df_class_0.sample(count_1,replace='True')
        df = pd.merge(df.drop(df_class_0.index),
                      df_class_0_sampled, how='outer')
    return df

# Over sample where AdoptionSpeed == 0, to match the count of AdoptionSpeed == 1
train_df = sample(train_df, [[0, 1]])




# Note how counts of 0 and 1 now match:
sns.countplot(train_df['AdoptionSpeed']);




g = sns.catplot("AdoptionSpeed", col="Type",
                hue="Gender", data=train_df,
                kind="count", height=5, aspect=.8);




'''
Score:  0.09349
   AdoptionSpeed  Type_2
0              2       1
1              3       0
''';




X = train_df.drop('AdoptionSpeed', axis=1)
y = train_df['AdoptionSpeed']
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(sampling_strategy='minority',
                                    random_state=1)
X_ros, y_ros = ros.fit_sample(X, y)

print(X_ros.shape[0] - X.shape[0], 'new randomly picked points')




resampled_df = pd.DataFrame(list(X_ros), columns=train_df.drop('AdoptionSpeed', axis=1).columns)
resampled_df['AdoptionSpeed'] = list(y_ros)
resampled_df.head(2)




sns.countplot(resampled_df['AdoptionSpeed']);




sns.catplot("AdoptionSpeed", col="Type",
                hue="Gender", data=resampled_df,
                kind="count", height=5, aspect=.8);




'''
Score:  0.12413
   AdoptionSpeed  Type_2
0              2       1
1              0       1

''';




score = 0
with open('../input/train_metadata/0a9f9f178-7.json') as f:
    good_image = json.load(f)
for label in good_image['labelAnnotations']:
    if label['description'] == 'dog':
        score = label['score']
print(score)




score = 0
with open('../input/train_metadata/0b3deeb66-2.json') as f:
    good_image = json.load(f)
for label in good_image['labelAnnotations']:
    if label['description'] == 'dog':
        score = label['score']
print(score)




{
    "description": "dog like mammal",
    "score": 0.77083683,
};




def get_photo_score(x, match='exact', start=1, stop=2, multiple=False):
    pet_id = x
    pet_type = train_df[train_df['PetID'] == pet_id]['Type'].values[0]
    pet_type_dict = {1: 'dog', 2: 'cat'}
    pet_type = pet_type_dict[pet_type]
    scores = []
    score = 0
    i = start
    while (i > 0) & (i < stop):
        json_file = '../input/train_metadata/' + pet_id + '-' + str(i) + '.json'
        if os.path.isfile(json_file):
            with open(json_file) as f:
                try:
                    image_data = False
                    image_data = pd.DataFrame(json.load(f)['labelAnnotations'])
                except Exception:
                    pass
            try:
                if match == 'exact':
                    scores.append(image_data[image_data['description'] == pet_type]['score'].values[0])
                if match == 'contains':
                    scores.append(image_data[image_data['description'].str.contains(pet_type)]['score'].values.max())
            except Exception:
                scores.append(.0)
                pass
            i += 1
        else:
            break
    try:
        if not multiple:
            if (stop-start) > 1:
                score = np.array(scores).mean()
            if (stop-start) == 1:
                score = np.array(scores).max()
        if multiple:
            score = np.array(scores)
    except Exception:
        pass
    return score

train_df['FirstPhotoScore'] = train_df['PetID'].apply(lambda x: get_photo_score(x, match='exact', start=1, stop=2, multiple=False))




train_df['FirstPhotoScore > 0'] = train_df['FirstPhotoScore'] > 0

sns.catplot("AdoptionSpeed", col="Type",
                hue="FirstPhotoScore > 0", data=train_df,
                kind="count", height=5, aspect=.8);




def group_photo_score(x):
    score = x['FirstPhotoScore']
    pet_type = x['Type']
    if pet_type == 1:
        good_threshold = 0.96
    if pet_type == 2:
        good_threshold = 0.99
    if score > good_threshold: 
        return 'Good'
    if (score < good_threshold) & (score > .5): 
        return 'Okay'
    return 'Not Great'
train_df['FirstPhotoScoreRange'] = train_df[['Type','FirstPhotoScore']].apply(lambda x: group_photo_score(x), axis=1)

sns.catplot("AdoptionSpeed", col="FirstPhotoScoreRange",
             data=train_df,
                kind="count", height=5, aspect=.8);




train_df['AllPhotoScores'] = train_df['PetID'].apply(lambda x: get_photo_score(x, match='exact', start=1, stop=99, multiple=False))




def mean_photo_score(x):
    mean_score = x['AllPhotoScore'] / x['PhotoAmt']
    pet_type = x['Type']
    if pet_type == 1:
        good_threshold = 0.96
    if pet_type == 2:
        good_threshold = 0.99
    if mean_score > good_threshold: 
        return 'Good'
    if (mean_score < good_threshold) & (mean_score > 0): 
        return 'Okay'
    return 'Not Great'

train_df['AllPhotoScoreRange'] = train_df[['AllPhotoScore', 'PhotoAmt', 'Type']].apply(lambda x: mean_photo_score(x), axis=1)

sns.catplot("AdoptionSpeed", col="AllPhotoScoreRange",
                 data=train_df,
                kind="count", height=5, aspect=.8);




train_df['AllPhotoScoresList'] = train_df['PetID']    .apply(lambda x: get_photo_score(x, match='exact', start=1, stop=99, multiple=True))




train_df['AllPhotoScoresList'].head()




def count_good_photos(x):
    count = 0
    pet_type = x['Type']
    if pet_type == 1:
        good_threshold = 0.96
    if pet_type == 2:
        good_threshold = 0.99
    try:
        count = len(x[x > good_threshold])
    except Exception:
        pass
    if count > 2:
        count = '> 3'
    return count

train_df['GoodPhotos'] = train_df[['Type', 'AllPhotoScoresList']].apply(lambda x: count_good_photos(x), axis=1)

sns.catplot("AdoptionSpeed", col="GoodPhotos",
                 data=train_df,
                kind="count", height=5, aspect=.8);




def count_secondary_good_photos(x):
    count = 0
    pet_type = x['Type']
    scores = x['AllPhotoScoresList']
    if pet_type == 1:
        good_threshold = 0.96
    if pet_type == 2:
        good_threshold = 0.99
    try:
        scores = scores[1:]
        count = len(scores[scores > good_threshold])
    except Exception:
        pass
    if count > 2:
        return 'Good'
    if count > 0:
        return 'Okay'
    return 'Not Great'

train_df['GoodSecondaryPhotos'] = train_df[['AllPhotoScoresList', 'Type']].apply(lambda x: count_secondary_good_photos(x), axis=1)

sns.catplot("AdoptionSpeed", col="GoodSecondaryPhotos",
                 data=train_df,
                kind="count", height=5, aspect=.8);




train_df['FirstAndSecondaryPhotos'] =  train_df['FirstPhotoScoreRange'] + '__' + train_df['GoodSecondaryPhotos']




sns.catplot("AdoptionSpeed", col="FirstAndSecondaryPhotos",col_wrap=4,
                 data=train_df[train_df['FirstPhotoScoreRange'] == 'Good'],
                kind="count", height=5, aspect=.8);




sns.catplot("AdoptionSpeed", col="FirstAndSecondaryPhotos",col_wrap=4,
                 data=train_df[train_df['FirstPhotoScoreRange'] == 'Okay'],
                kind="count", height=5, aspect=.8);




sns.catplot("AdoptionSpeed", col="FirstAndSecondaryPhotos",col_wrap=4,
                 data=train_df[train_df['FirstPhotoScoreRange'] == 'Not Great'],
                kind="count", height=5, aspect=.8);




scores = np.array([34,33,23234,12,45,456,454,6,6])




with open('../input/train_metadata/0a9f9f178-7.json') as json_file:
    json_text = json.load(json_file)




image_data = pd.DataFrame(json_text['labelAnnotations']).drop(['mid', 'topicality'], axis=1).rename({'description': 'Description', 'score': 'Score'}, axis=1)




image_data['PetID'] = '0a9f9f178'




image_data['ImageID'] = 1




image_data




file_name = '0a9f9f178-7.json'




file_name.split('-')[0]




int(file_name.split('-')[1].split('.')[0])




def image_data_matches(df, images_df):
    return len(set(images_df['PetID'].unique()) - set(train_df['PetID'].unique())) == 0

def create_image_df(df, images_df_file_name):
    print('Building new image data csv...', df.name)
    json_folder_path = path + '/input/' + df.name + '_metadata/'
    json_files = [f_name for f_name in os.listdir(json_folder_path)
                  if f_name.endswith('.json')]
    pet_type_dict = {1: 'dog', 2: 'cat'}
    all_images_list = []
    for index, f_name in enumerate(json_files):
        with open(os.path.join(json_folder_path, f_name)) as json_file:
            json_text = json.load(json_file)
            try:
                label_annotations = json_text['labelAnnotations']
            except:
                continue
            image_data = pd.DataFrame(label_annotations)                .drop(['mid', 'topicality'], axis=1)                .rename({'description': 'Description',
                         'score': 'Score'},
                        axis=1)
            pet_id = f_name.split('-')[0]
            image_data['PetID'] = pet_id
            image_data['ImageID'] = int(f_name.split('-')[1].split('.')[0])
            image_data['PetLabel'] = pet_type_dict[df[df['PetID'] == pet_id]['Type'].values[0]]
        all_images_list.append(image_data)
    images_df = pd.concat(all_images_list)
    images_df.to_csv(images_df_file_name, index=False)
    return images_df

def get_image_data(df, force_new_csv=False):
    images_df_file_name = path + '/' + df.name + '_image_data.csv'
    try:
        images_df = pd.read_csv(images_df_file_name)
        no_image_file = False
    except Exception:
        no_image_file = True
    if no_image_file or force_new_csv or not image_data_matches(df, images_df):
        images_df = create_image_df(df, images_df_file_name)
    return images_df

def rate_image(x):
    pet_label = x['PetLabel']
    score = x['Score']
    if pet_label == 'dog':
        good_threshold = 0.96
    if pet_label == 'cat':
        good_threshold = 0.99
    if score > good_threshold:
        return 2
    return 1
    
def cap_max_image_rating(x):
    if x > 2:
        return 2
    return x
        
def append_image_data(df):
    images_df = get_image_data(df)
    images = images_df[(images_df['PetLabel'] == images_df['Description'])]                             [['PetID','Score','PetLabel', 'ImageID']]
    images['ImageRating'] = images[['PetLabel', 'Score']].apply(lambda x: rate_image(x), axis=1)
    first_images = images[images['ImageID'] == 1][['PetID', 'ImageRating']]
    first_images.rename({'ImageRating': 'FirstImageRating'}, axis=1, inplace=True)
    second_images = images[(images['ImageID'] > 1) & (images['ImageRating'] > 1)].groupby('PetID')['ImageRating'].count().reset_index()
    second_images.rename({'ImageRating': 'SecondImageRating'}, axis=1, inplace=True)
    second_images['SecondImageRating'] = second_images['SecondImageRating'].apply(lambda x: cap_max_image_rating(x))
    image_ratings = pd.merge(first_images, second_images, on='PetID', how='left')
    df = pd.merge(df, image_ratings[['PetID', 'FirstImageRating', 'SecondImageRating']], on='PetID', how='left')
    df['FirstImageRating'].fillna(0, inplace=True)
    df['SecondImageRating'].fillna(0, inplace=True)
    df['TotalImageRating'] = df['FirstImageRating'] + (df['SecondImageRating'] * .1)
    df.loc[df['TotalImageRating'] == 1.0, 'TotalImageRating'] = 0.0
    df.loc[(df['TotalImageRating'] == 1.2) | 
           (df['TotalImageRating'] == 1.1), 'TotalImageRating'] = 1.0
    df.loc[df['TotalImageRating'] == 2.1, 'TotalImageRating'] = 2.0
    df.loc[df['TotalImageRating'] == 2.2, 'TotalImageRating'] = 3.0
    return df

reset_dfs()

train_df.name = 'train'
test_df.name = 'test'

train_df = append_image_data(train_df)
test_df = append_image_data(test_df)

train_df.columns




images_df = pd.read_csv('../train_image_data.csv')
len(set(images_df['PetID'].unique()) - set(train_df['PetID'].unique()))

train_df_copy = train_df.copy()




image_ratings = pd.DataFrame(columns=['PetID', 'ImageRating'])
image_ratings




reset_dfs()
train_df.shape




sns.catplot("AdoptionSpeed", col="FirstImageRating",col_wrap=4,
                 data=train_df.log,
                kind="count", height=5, aspect=.8);




sns.catplot("AdoptionSpeed", col="SecondImageRating",col_wrap=4,
                 data=train_df,
                kind="count", height=5, aspect=.8);




train_df['TotalImageRating'] = train_df['FirstImageRating'] + (train_df['SecondImageRating'] * .01)
sns.catplot("AdoptionSpeed", col="FirstImageRating",col_wrap=4,
                 data=train_df,
                kind="count", height=5, aspect=.8);




sns.catplot("AdoptionSpeed", col="SecondImageRating",col_wrap=4,
                 data=train_df,
                kind="count", height=5, aspect=.8);




train_df['TotalImageRating'] = train_df['FirstImageRating'] + (train_df['SecondImageRating'] * .1)
train_df.loc[train_df['TotalImageRating'] == 1.0, 'TotalImageRating'] = 0.0
train_df.loc[(train_df['TotalImageRating'] == 1.2) | (train_df['TotalImageRating'] == 1.1), 'TotalImageRating'] = 1.0
train_df.loc[train_df['TotalImageRating'] == 2.1, 'TotalImageRating'] = 2.0
train_df.loc[train_df['TotalImageRating'] == 2.2, 'TotalImageRating'] = 3.0
sns.catplot("AdoptionSpeed", col="TotalImageRating",col_wrap=4,
                 data=train_df,
                kind="count", height=5, aspect=.8);




images_df = pd.read_csv('../train_image_data.csv')
def get_adoptionspeed(x):
    return train_df[train_df['PetID'] == x['PetID']]['AdoptionSpeed']
images_df['AdoptionSpeed'] = images_df[['PetID']].apply(lambda x: get_adoptionspeed(x), axis=1)
images_df.head()
# images = images_df[(images_df['PetLabel'] == 'dog') & (images_df['ImageID'] == 1)].groupby('Description')[['PetLabel']].count().sort_values('PetLabel', ascending=False)
# images.head(50)






