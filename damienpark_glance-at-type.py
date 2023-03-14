#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np

import json

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

import keras

import pprint
import gc
import matplotlib.pyplot as plt
import seaborn as sns

# pandas display option
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_row', 1500)
pd.set_option('max_colwidth', 1500)
pd.set_option('display.float_format', '{:.2f}'.format)




train = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv")
# test = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")
label = pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv")
# sample = pd.read_csv("/kaggle/input/data-science-bowl-2019/sample_submission.csv")
specs = pd.read_csv("/kaggle/input/data-science-bowl-2019/specs.csv")




# df = train.merge(label, how="left", on=["installation_id", "game_session", "title"])
# df.timestamp = pd.to_datetime(df.timestamp)
# df.sort_values(["timestamp", "event_count"], ascending=True, inplace=True)
# del train, label
# gc.collect()




train.timestamp = pd.to_datetime(train.timestamp)
df = train.sort_values(["timestamp", "event_count"], ascending=True)
df = df.merge(label, how="left", on=["installation_id", "game_session", "title"])
df = df.merge(specs, how="left", on=["event_id"])
del train
gc.collect()
pd.unique(df.type)




# ['Scrub-A-Dub', 'Bubble Bath', 'All Star Sorting', 'Chow Time',
#  'Dino Dive', 'Happy Camel', 'Leaf Leader', 'Pan Balance',
#  'Dino Drink', 'Crystals Rule', 'Air Show']




df.head()




title_select = "Scrub-A-Dub"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data", "event_id"]]
for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




title_select = "Scrub-A-Dub"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x: json.loads(x)["round"])
_["levels"] = _.event_data.apply(lambda x: (json.loads(x)["level"] if "level" in json.loads(x).keys() else -999))

bins_rounds = len(np.unique(_.groupby("game_session").rounds.max()))
bins_levels = len(np.unique(_.groupby("game_session").levels.max()))

plt.figure(figsize=(25, 7))
plt.subplot(1, 2, 1)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)
plt.subplot(1, 2, 2)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)

plt.show()




title_select = "Bubble Bath"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data", "event_id"]]
for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




title_select = "Bubble Bath"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x: json.loads(x)["round"])
_["levels"] = _.event_data.apply(lambda x: (json.loads(x)["level"] if "level" in json.loads(x).keys() else -999))

bins_rounds = len(np.unique(_.groupby("game_session").rounds.max()))
bins_levels = len(np.unique(_.groupby("game_session").levels.max()))

plt.figure(figsize=(25, 7))
plt.subplot(1, 2, 1)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)
plt.subplot(1, 2, 2)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)

plt.show()




title_select = "All Star Sorting"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data", "event_id"]]
for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




title_select = "All Star Sorting"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x: json.loads(x)["round"])
_["levels"] = _.event_data.apply(lambda x: (json.loads(x)["level"] if "level" in json.loads(x).keys() else -999))

bins_rounds = len(np.unique(_.groupby("game_session").rounds.max()))
bins_levels = len(np.unique(_.groupby("game_session").levels.max()))

plt.figure(figsize=(25, 7))
plt.subplot(1, 2, 1)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)
plt.subplot(1, 2, 2)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)

plt.show()




title_select = "Chow Time"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data", "event_id"]]
for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




title_select = "Chow Time"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x: json.loads(x)["round"])
_["levels"] = _.event_data.apply(lambda x: (json.loads(x)["level"] if "level" in json.loads(x).keys() else -999))

bins_rounds = len(np.unique(_.groupby("game_session").rounds.max()))
bins_levels = len(np.unique(_.groupby("game_session").levels.max()))

plt.figure(figsize=(25, 7))
plt.subplot(1, 2, 1)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)
plt.subplot(1, 2, 2)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)

plt.show()




title_select = "Dino Dive"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data", "event_id"]]
for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




title_select = "Dino Dive"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x: json.loads(x)["round"])
_["levels"] = _.event_data.apply(lambda x: (json.loads(x)["level"] if "level" in json.loads(x).keys() else -999))

bins_rounds = len(np.unique(_.groupby("game_session").rounds.max()))
bins_levels = len(np.unique(_.groupby("game_session").levels.max()))

plt.figure(figsize=(25, 7))
plt.subplot(1, 2, 1)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)
plt.subplot(1, 2, 2)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)

plt.show()




title_select = "Happy Camel"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data", "event_id"]]
for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




title_select = "Happy Camel"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x: json.loads(x)["round"])
_["levels"] = _.event_data.apply(lambda x: (json.loads(x)["level"] if "level" in json.loads(x).keys() else -999))

bins_rounds = len(np.unique(_.groupby("game_session").rounds.max()))
bins_levels = len(np.unique(_.groupby("game_session").levels.max()))

plt.figure(figsize=(25, 7))
plt.subplot(1, 2, 1)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)
plt.subplot(1, 2, 2)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)

plt.show()




title_select = "Leaf Leader"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data", "event_id"]]
for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




title_select = "Leaf Leader"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x: json.loads(x)["round"])
_["levels"] = _.event_data.apply(lambda x: (json.loads(x)["level"] if "level" in json.loads(x).keys() else -999))

bins_rounds = len(np.unique(_.groupby("game_session").rounds.max()))
bins_levels = len(np.unique(_.groupby("game_session").levels.max()))

plt.figure(figsize=(25, 7))
plt.subplot(1, 2, 1)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)
plt.subplot(1, 2, 2)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)

plt.show()




title_select = "Pan Balance"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data", "event_id"]]
for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




title_select = "Pan Balance"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x: json.loads(x)["round"])
_["levels"] = _.event_data.apply(lambda x: (json.loads(x)["level"] if "level" in json.loads(x).keys() else -999))

bins_rounds = len(np.unique(_.groupby("game_session").rounds.max()))
bins_levels = len(np.unique(_.groupby("game_session").levels.max()))

plt.figure(figsize=(25, 7))
plt.subplot(1, 2, 1)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)
plt.subplot(1, 2, 2)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)

plt.show()




title_select = "Dino Drink"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data", "event_id"]]
for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




title_select = "Dino Drink"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x: json.loads(x)["round"])
_["levels"] = _.event_data.apply(lambda x: (json.loads(x)["level"] if "level" in json.loads(x).keys() else -999))

bins_rounds = len(np.unique(_.groupby("game_session").rounds.max()))
bins_levels = len(np.unique(_.groupby("game_session").levels.max()))

plt.figure(figsize=(25, 7))
plt.subplot(1, 2, 1)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)
plt.subplot(1, 2, 2)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)

plt.show()




title_select = "Crystals Rule"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data", "event_id"]]
for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




title_select = "Crystals Rule"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x: json.loads(x)["round"])
_["levels"] = _.event_data.apply(lambda x: (json.loads(x)["level"] if "level" in json.loads(x).keys() else -999))

bins_rounds = len(np.unique(_.groupby("game_session").rounds.max()))
bins_levels = len(np.unique(_.groupby("game_session").levels.max()))

plt.figure(figsize=(25, 7))
plt.subplot(1, 2, 1)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)
plt.subplot(1, 2, 2)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)

plt.show()




title_select = "Air Show"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data", "event_id"]]
for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




title_select = "Air Show"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x: json.loads(x)["round"])
_["levels"] = _.event_data.apply(lambda x: (json.loads(x)["level"] if "level" in json.loads(x).keys() else -999))

bins_rounds = len(np.unique(_.groupby("game_session").rounds.max()))
bins_levels = len(np.unique(_.groupby("game_session").levels.max()))

plt.figure(figsize=(25, 7))
plt.subplot(1, 2, 1)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)
plt.subplot(1, 2, 2)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)

plt.show()




np.unique(_.levels)




class accuracy:
    def __init__(self, df):
        self.df = df

        
    # Assessment evaluation-Cart Balancer (Assessment)
    def cart_assessment(self):
        _ = self.df.query("title=='Cart Balancer (Assessment)' and event_id=='d122731b'")
        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]
        _["correct"] = _.event_data.apply(lambda x:(json.loads(x)["correct"] if "correct" in json.loads(x).keys() else -999))
        _["num_correct_"]=0
        _["num_incorrect_"]=0
        _.loc[_.correct==True, "num_correct_"] = 1
        _.loc[_.correct==False, "num_incorrect_"] = 1
        _ = _.groupby(["installation_id", "game_session"]).sum().reset_index()
        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["num_incorrect_"])
        _["accuracy_group_"] = _["num_incorrect_"].apply(lambda x : 3 if x==0 else (2 if x==1 else 1))*_["num_correct_"]

        return _.loc[:, ["installation_id", "game_session", "num_correct_", "num_incorrect_", "accuracy_", "accuracy_group_"]]

    def cart_assessment_2(self):
        _ = self.df.query("title=='Cart Balancer (Assessment)' and event_id=='b74258a0'")
        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]
        _["misses"] = _.event_data.apply(lambda x:(json.loads(x)["misses"] if "misses" in json.loads(x).keys() else -999))
        _["num_correct_"]=1
        _ = _.groupby("game_session").sum().reset_index()
        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["misses"])

        return _.loc[:, ["game_session", "num_correct_", "misses", "accuracy_"]]
    
    
    # Assessment evaluation-Chest Sorter (Assessment)
    def chest_assessment(self):
        _ = self.df.query("title=='Chest Sorter (Assessment)' and event_id=='93b353f2'")
        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]
        _["correct"] = _.event_data.apply(lambda x:(json.loads(x)["correct"] if "correct" in json.loads(x).keys() else -999))
        _["num_correct_"]=0
        _["num_incorrect_"]=0
        _.loc[_.correct==True, "num_correct_"] = 1
        _.loc[_.correct==False, "num_incorrect_"] = 1
        _ = _.groupby(["installation_id", "game_session"]).sum().reset_index()
        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["num_incorrect_"])
        _["accuracy_group_"] = _["num_incorrect_"].apply(lambda x : 3 if x==0 else (2 if x==1 else 1))*_["num_correct_"]

        return _.loc[:, ["installation_id", "game_session", "num_correct_", "num_incorrect_", "accuracy_", "accuracy_group_"]]
    
    def chest_assessment_2(self):
        _ = self.df.query("title=='Chest Sorter (Assessment)' and event_id=='38074c54'")
        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]
        _["misses"] = _.event_data.apply(lambda x:(json.loads(x)["misses"] if "misses" in json.loads(x).keys() else -999))
        _["num_correct_"]=1
        _ = _.groupby("game_session").sum().reset_index()
        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["misses"])

        return _.loc[:, ["game_session", "num_correct_", "misses", "accuracy_"]]
    
    
    # Assessment evaluation-Cauldron Filler (Assessment)
    def cauldron_assessment(self):
        _ = self.df.query("title=='Cauldron Filler (Assessment)' and event_id=='392e14df'")
        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]
        _["correct"] = _.event_data.apply(lambda x:(json.loads(x)["correct"] if "correct" in json.loads(x).keys() else -999))
        _["num_correct_"] = 0
        _["num_incorrect_"] = 0
        _.loc[_.correct==True, "num_correct_"] = 1
        _.loc[_.correct==False, "num_incorrect_"] = 1
        _ = _.groupby(["installation_id", "game_session"]).sum().reset_index()
        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["num_incorrect_"])
        _["accuracy_group_"] = _["num_incorrect_"].apply(lambda x : 3 if x==0 else (2 if x==1 else 1))*_["num_correct_"]

        return _.loc[:, ["installation_id", "game_session", "num_correct_", "num_incorrect_", "accuracy_", "accuracy_group_"]]

    def cauldron_assessment_2(self):
        _ = self.df.query("title=='Cauldron Filler (Assessment)' and event_id=='28520915'")
        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]
        _["misses"] = _.event_data.apply(lambda x:(json.loads(x)["misses"] if "misses" in json.loads(x).keys() else -999))
        _["num_correct_"] = 1
        _ = _.groupby("game_session").sum().reset_index()
        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["misses"])

        return _.loc[:, ["game_session", "num_correct_", "misses", "accuracy_"]]
    
    
    # Assessment evaluation-Mushroom Sorter (Assessment)
    def mushroom_assessment(self):
        _ = self.df.query("title=='Mushroom Sorter (Assessment)' and event_id=='25fa8af4'")
        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]
        _["correct"] = _.event_data.apply(lambda x:(json.loads(x)["correct"] if "correct" in json.loads(x).keys() else -999))
        _["num_correct_"]=0
        _["num_incorrect_"]=0
        _.loc[_.correct==True, "num_correct_"] = 1
        _.loc[_.correct==False, "num_incorrect_"] = 1
        _ = _.groupby(["installation_id", "game_session"]).sum().reset_index()
        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["num_incorrect_"])
        _["accuracy_group_"] = _["num_incorrect_"].apply(lambda x : 3 if x==0 else (2 if x==1 else 1))*_["num_correct_"]

        return _.loc[:, ["installation_id", "game_session", "num_correct_", "num_incorrect_", "accuracy_", "accuracy_group_"]]
    
    def mushroom_assessment_2(self):
        _ = self.df.query("title=='Mushroom Sorter (Assessment)' and event_id=='6c930e6e'")
        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]
        _["misses"] = _.event_data.apply(lambda x:(json.loads(x)["misses"] if "misses" in json.loads(x).keys() else -999))
        _["num_correct_"] = 1
        _ = _.groupby("game_session").sum().reset_index()
        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["misses"])

        return _.loc[:, ["game_session", "num_correct_", "misses", "accuracy_"]]
    
    
    # Assessment evaluation-Bird Measurer (Assessment)
    def bird_assessment(self):
        _ = self.df.query("title=='Bird Measurer (Assessment)' and event_id=='17113b36'")
        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]
        _["correct"] = _.event_data.apply(lambda x:(json.loads(x)["correct"] if "correct" in json.loads(x).keys() else -999))
        _["num_correct_"]=0
        _["num_incorrect_"]=0
        _.loc[_.correct==True, "num_correct_"] = 1
        _.loc[_.correct==False, "num_incorrect_"] = 1
        _ = _.groupby(["installation_id", "game_session"]).sum().reset_index()
        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["num_incorrect_"])
        _["accuracy_group_"] = _["num_incorrect_"].apply(lambda x : 3 if x==0 else (2 if x==1 else 1))*_["num_correct_"]

        return _.loc[:, ["installation_id", "game_session", "num_correct_", "num_incorrect_", "accuracy_", "accuracy_group_"]]
    
    def bird_assessment_2(self):
        _ = self.df.query("title=='Bird Measurer (Assessment)' and event_id=='f6947f54'")
        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]
        _["misses"] = _.event_data.apply(lambda x:(json.loads(x)["misses"] if "misses" in json.loads(x).keys() else -999))
        _["num_correct_"] = 1
        _ = _.groupby("game_session").sum().reset_index()
        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["misses"])

        return _.loc[:, ["game_session", "num_correct_", "misses", "accuracy_"]]




_ = df.query("type=='Assessment'")
pd.unique(_.event_code)
_.loc[:, ["event_code", "event_id", "info", "title"]].drop_duplicates(["event_code", "event_id", "info", "title"]).sort_values("event_code").reset_index(drop=True).groupby(["event_code"]).size()




temp_title = _.loc[:, ["event_code", "event_id", "info", "title"]].drop_duplicates(["event_code", "event_id", "info", "title"])
temp = pd.DataFrame(_.loc[:, ["event_code", "event_id", "info", "title"]].drop_duplicates(["event_code", "event_id", "info", "title"])                    .sort_values("event_code").reset_index(drop=True).groupby(["event_code", "info"]).size()).reset_index()




temp.merge(temp_title)




temp.merge(temp_title).query("title=='Cart Balancer (Assessment)'")




json.loads(specs.query("event_id=='65a38bf7'").reset_index(drop=True)["args"][0])




json.loads(df.query("event_id=='65a38bf7'").reset_index(drop=True)["event_data"][0])




json.loads(df.query("event_id=='65a38bf7'").reset_index(drop=True)["event_data"][4000])




json.loads(specs.query("event_id=='b74258a0'").reset_index(drop=True)["args"][0])




json.loads(df.query("event_id=='b74258a0'").reset_index(drop=True)["event_data"][0])




json.loads(df.query("event_id=='b74258a0'").reset_index(drop=True)["event_data"][1298])




json.loads(specs.query("event_id=='795e4a37'").reset_index(drop=True)["args"][0])




json.loads(df.query("event_id=='795e4a37'").reset_index(drop=True)["event_data"][0])




json.loads(df.query("event_id=='795e4a37'").reset_index(drop=True)["event_data"][3])




desc = df.query("event_id=='795e4a37'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["description"]))
iden = df.query("event_id=='795e4a37'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["identifier"]))
medi = df.query("event_id=='795e4a37'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["media_type"]))
dura = df.query("event_id=='795e4a37'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["total_duration"]))
pd.unique(desc), pd.unique(iden), pd.unique(medi), pd.unique(dura)




json.loads(specs.query("event_id=='5de79a6a'").reset_index(drop=True)["args"][0])




json.loads(df.query("event_id=='5de79a6a'").reset_index(drop=True)["event_data"][0])




desc = df.query("event_id=='5de79a6a'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["description"]))
iden = df.query("event_id=='5de79a6a'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["identifier"]))
medi = df.query("event_id=='5de79a6a'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["media_type"]))
dura = df.query("event_id=='5de79a6a'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["total_duration"]))
pd.unique(desc), pd.unique(iden), pd.unique(medi), pd.unique(dura)




np.unique(dura)




json.loads(specs.query("event_id=='a8876db3'").reset_index(drop=True)["args"][0])




json.loads(df.query("event_id=='a8876db3'").reset_index(drop=True)["event_data"][0])




desc = df.query("event_id=='a8876db3'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["description"]))
iden = df.query("event_id=='a8876db3'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["identifier"]))
medi = df.query("event_id=='a8876db3'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["media_type"]))
dura = df.query("event_id=='a8876db3'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["total_duration"]))
pd.unique(desc), pd.unique(iden), pd.unique(medi), pd.unique(dura)




json.loads(specs.query("event_id=='828e68f9'").reset_index(drop=True)["args"][0])




json.loads(df.query("event_id=='828e68f9'").reset_index(drop=True)["event_data"][0])




desc = df.query("event_id=='828e68f9'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["description"]))
iden = df.query("event_id=='828e68f9'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["identifier"]))
medi = df.query("event_id=='828e68f9'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["media_type"]))
dura = df.query("event_id=='828e68f9'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["duration"]))
pd.unique(desc), pd.unique(iden), pd.unique(medi)




np.unique(dura)




plt.figure(figsize=(20, 10))
plt.hist(dura[dura<20000], bins=100)
plt.show()




json.loads(specs.query("event_id=='31973d56'").reset_index(drop=True)["args"][0])




json.loads(df.query("event_id=='31973d56'").reset_index(drop=True)["event_data"][0])




desc = df.query("event_id=='31973d56'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["description"]))
iden = df.query("event_id=='31973d56'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["identifier"]))
medi = df.query("event_id=='31973d56'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["media_type"]))
dura = df.query("event_id=='31973d56'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["duration"]))
pd.unique(desc), pd.unique(iden), pd.unique(medi)




plt.figure(figsize=(20, 10))
plt.hist(dura[dura<5000], bins=100)
plt.show()




json.loads(specs.query("event_id=='ecaab346'").reset_index(drop=True)["args"][0])




json.loads(df.query("event_id=='ecaab346'").reset_index(drop=True)["event_data"][0])




desc = df.query("event_id=='ecaab346'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["description"]))
iden = df.query("event_id=='ecaab346'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["identifier"]))
medi = df.query("event_id=='ecaab346'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["media_type"]))
dura = df.query("event_id=='ecaab346'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["duration"]))
pd.unique(desc), pd.unique(iden), pd.unique(medi)




plt.figure(figsize=(20, 10))
plt.hist(dura[dura<10000], bins=100)
plt.show()




json.loads(specs.query("event_id=='5c2f29ca'").reset_index(drop=True)["args"][0])




json.loads(df.query("event_id=='5c2f29ca'").reset_index(drop=True)["event_data"][0])




coor = df.query("event_id=='5c2f29ca'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["coordinates"]))
side = df.query("event_id=='5c2f29ca'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["side"]))
sour = df.query("event_id=='5c2f29ca'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["source"]))
dura = df.query("event_id=='5c2f29ca'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["duration"]))
pd.unique(side), pd.unique(sour), pd.unique(dura)




pd.unique(coor.apply(lambda x : x["stage_width"])), pd.unique(coor.apply(lambda x : x["stage_height"]))




a = coor.apply(lambda x : x["stage_width"])
b = coor.apply(lambda x : x["stage_height"])
c = zip(a, b)
set(c)




plt.figure(figsize=(20, 7))
plt.subplot(1, 3, 1)
x_cor = coor.apply(lambda x : x["x"] if x["stage_width"]==1015 and x["stage_height"]==748 else None)
y_cor = coor.apply(lambda x : x["y"] if x["stage_width"]==1015 and x["stage_height"]==748 else None)
plt.scatter(x=x_cor, y=y_cor)

plt.subplot(1, 3, 2)
x_cor = coor.apply(lambda x : x["x"] if x["stage_width"]==1015 and x["stage_height"]==762 else None)
y_cor = coor.apply(lambda x : x["y"] if x["stage_width"]==1015 and x["stage_height"]==762 else None)
plt.scatter(x=x_cor, y=y_cor)

plt.subplot(1, 3, 3)
x_cor = coor.apply(lambda x : x["x"] if x["stage_width"]==1267 and x["stage_height"]==800 else None)
y_cor = coor.apply(lambda x : x["y"] if x["stage_width"]==1267 and x["stage_height"]==800 else None)
plt.scatter(x=x_cor, y=y_cor)

plt.show()




plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.hist(coor.apply(lambda x : x["x"]))
plt.subplot(1, 2, 2)
plt.hist(coor.apply(lambda x : x["y"]))
plt.show()




json.loads(specs.query("event_id=='5e109ec3'").reset_index(drop=True)["args"][0])




json.loads(df.query("event_id=='5e109ec3'").reset_index(drop=True)["event_data"][1])




coor = df.query("event_id=='5e109ec3'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["coordinates"]))
# side = df.query("event_id=='5e109ec3'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["side"]))
sour = df.query("event_id=='5e109ec3'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["source"]))
# dura = df.query("event_id=='5e109ec3'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["duration"]))
pd.unique(sour)




pd.unique(coor.apply(lambda x : x["stage_width"])), pd.unique(coor.apply(lambda x : x["stage_height"]))




a = coor.apply(lambda x : x["stage_width"])
b = coor.apply(lambda x : x["stage_height"])
c = zip(a, b)
set(c)




plt.figure(figsize=(20, 7))
plt.subplot(1, 3, 1)
x_cor = coor.apply(lambda x : x["x"] if x["stage_width"]==1015 and x["stage_height"]==748 else None)
y_cor = coor.apply(lambda x : x["y"] if x["stage_width"]==1015 and x["stage_height"]==748 else None)
plt.scatter(x=x_cor, y=y_cor)

plt.subplot(1, 3, 2)
x_cor = coor.apply(lambda x : x["x"] if x["stage_width"]==1015 and x["stage_height"]==762 else None)
y_cor = coor.apply(lambda x : x["y"] if x["stage_width"]==1015 and x["stage_height"]==762 else None)
plt.scatter(x=x_cor, y=y_cor)

plt.subplot(1, 3, 3)
x_cor = coor.apply(lambda x : x["x"] if x["stage_width"]==1267 and x["stage_height"]==800 else None)
y_cor = coor.apply(lambda x : x["y"] if x["stage_width"]==1267 and x["stage_height"]==800 else None)
plt.scatter(x=x_cor, y=y_cor)

plt.show()




plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.hist(coor.apply(lambda x : x["x"]))
plt.subplot(1, 2, 2)
plt.hist(coor.apply(lambda x : x["y"]))
plt.show()




json.loads(specs.query("event_id=='3d63345e'").reset_index(drop=True)["args"][0])




json.loads(df.query("event_id=='3d63345e'").reset_index(drop=True)["event_data"][1])




coor = df.query("event_id=='3d63345e'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["coordinates"]))
# side = df.query("event_id=='3d63345e'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["side"]))
sour = df.query("event_id=='3d63345e'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["source"]))
dura = df.query("event_id=='3d63345e'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["duration"]))
pd.unique(sour), pd.unique(dura)




pd.unique(coor.apply(lambda x : x["stage_width"])), pd.unique(coor.apply(lambda x : x["stage_height"]))




a = coor.apply(lambda x : x["stage_width"])
b = coor.apply(lambda x : x["stage_height"])
c = zip(a, b)
set(c)




plt.figure(figsize=(20, 7))
plt.subplot(1, 3, 1)
x_cor = coor.apply(lambda x : x["x"] if x["stage_width"]==1015 and x["stage_height"]==748 else None)
y_cor = coor.apply(lambda x : x["y"] if x["stage_width"]==1015 and x["stage_height"]==748 else None)
plt.scatter(x=x_cor, y=y_cor)

plt.subplot(1, 3, 2)
x_cor = coor.apply(lambda x : x["x"] if x["stage_width"]==1015 and x["stage_height"]==762 else None)
y_cor = coor.apply(lambda x : x["y"] if x["stage_width"]==1015 and x["stage_height"]==762 else None)
plt.scatter(x=x_cor, y=y_cor)

plt.subplot(1, 3, 3)
x_cor = coor.apply(lambda x : x["x"] if x["stage_width"]==1267 and x["stage_height"]==800 else None)
y_cor = coor.apply(lambda x : x["y"] if x["stage_width"]==1267 and x["stage_height"]==800 else None)
plt.scatter(x=x_cor, y=y_cor)

plt.show()




plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.hist(coor.apply(lambda x : x["x"]))
plt.subplot(1, 2, 2)
plt.hist(coor.apply(lambda x : x["y"]))
plt.show()




plt.figure(figsize=(20, 10))
plt.hist(dura[dura<5000], bins=100)
plt.show()




json.loads(specs.query("event_id=='9d4e7b25'").reset_index(drop=True)["args"][0])




json.loads(df.query("event_id=='9d4e7b25'").reset_index(drop=True)["event_data"][1])




coor = df.query("event_id=='9d4e7b25'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["coordinates"]))
sour = df.query("event_id=='9d4e7b25'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["source"]))
dura = df.query("event_id=='9d4e7b25'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["duration"]))
pd.unique(sour), pd.unique(dura)




pd.unique(coor.apply(lambda x : x["stage_width"])), pd.unique(coor.apply(lambda x : x["stage_height"]))




a = coor.apply(lambda x : x["stage_width"])
b = coor.apply(lambda x : x["stage_height"])
c = zip(a, b)
set(c)




plt.figure(figsize=(20, 7))
plt.subplot(1, 3, 1)
x_cor = coor.apply(lambda x : x["x"] if x["stage_width"]==1015 and x["stage_height"]==748 else None)
y_cor = coor.apply(lambda x : x["y"] if x["stage_width"]==1015 and x["stage_height"]==748 else None)
plt.scatter(x=x_cor, y=y_cor)

plt.subplot(1, 3, 2)
x_cor = coor.apply(lambda x : x["x"] if x["stage_width"]==1015 and x["stage_height"]==762 else None)
y_cor = coor.apply(lambda x : x["y"] if x["stage_width"]==1015 and x["stage_height"]==762 else None)
plt.scatter(x=x_cor, y=y_cor)

plt.subplot(1, 3, 3)
x_cor = coor.apply(lambda x : x["x"] if x["stage_width"]==1267 and x["stage_height"]==800 else None)
y_cor = coor.apply(lambda x : x["y"] if x["stage_width"]==1267 and x["stage_height"]==800 else None)
plt.scatter(x=x_cor, y=y_cor)

plt.show()




plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.hist(coor.apply(lambda x : x["x"]))
plt.subplot(1, 2, 2)
plt.hist(coor.apply(lambda x : x["y"]))
plt.show()




plt.figure(figsize=(20, 10))
plt.hist(dura[dura<5000], bins=100)
plt.show()




json.loads(specs.query("event_id=='acf5c23f'").reset_index(drop=True)["args"][0])




json.loads(df.query("event_id=='acf5c23f'").reset_index(drop=True)["event_data"][1])




coor = df.query("event_id=='acf5c23f'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["coordinates"]))
# sour = df.query("event_id=='acf5c23f'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["source"]))
# dura = df.query("event_id=='acf5c23f'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["duration"]))
# pd.unique(sour), pd.unique(dura)




pd.unique(coor.apply(lambda x : x["stage_width"])), pd.unique(coor.apply(lambda x : x["stage_height"]))




a = coor.apply(lambda x : x["stage_width"])
b = coor.apply(lambda x : x["stage_height"])
c = zip(a, b)
set(c)




plt.figure(figsize=(20, 7))
plt.subplot(1, 3, 1)
x_cor = coor.apply(lambda x : x["x"] if x["stage_width"]==1015 and x["stage_height"]==748 else None)
y_cor = coor.apply(lambda x : x["y"] if x["stage_width"]==1015 and x["stage_height"]==748 else None)
plt.scatter(x=x_cor, y=y_cor)

plt.subplot(1, 3, 2)
x_cor = coor.apply(lambda x : x["x"] if x["stage_width"]==1015 and x["stage_height"]==762 else None)
y_cor = coor.apply(lambda x : x["y"] if x["stage_width"]==1015 and x["stage_height"]==762 else None)
plt.scatter(x=x_cor, y=y_cor)

plt.subplot(1, 3, 3)
x_cor = coor.apply(lambda x : x["x"] if x["stage_width"]==1267 and x["stage_height"]==800 else None)
y_cor = coor.apply(lambda x : x["y"] if x["stage_width"]==1267 and x["stage_height"]==800 else None)
plt.scatter(x=x_cor, y=y_cor)

plt.show()




df.query("event_id=='acf5c23f'").reset_index(drop=True).accuracy_group




plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.hist(coor.apply(lambda x : x["x"]))
plt.subplot(1, 2, 2)
plt.hist(coor.apply(lambda x : x["y"]))
plt.show()




plt.figure(figsize=(15, 15))
plt.scatter(x=coor.apply(lambda x : x["x"]), y=coor.apply(lambda x : x["y"]))
plt.show()




json.loads(specs.query("event_id=='ecc6157f'").reset_index(drop=True)["args"][0])




json.loads(df.query("event_id=='ecc6157f'").reset_index(drop=True)["event_data"][1])




dwell = df.query("event_id=='ecc6157f'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["dwell_time"]))
obj = df.query("event_id=='ecc6157f'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["object"]))
np.unique(dwell, return_counts=True), pd.unique(obj)




len(df.query("event_id=='ecc6157f'"))




json.loads(specs.query("event_id=='4e5fc6f5'").reset_index(drop=True)["args"][0])




json.loads(df.query("event_id=='4e5fc6f5'").reset_index(drop=True)["event_data"][0])




df.query("game_session=='f20ba87c8f78fffd'").iloc[:, :-3]




json.loads(specs.query("event_id=='d122731b'").reset_index(drop=True)["args"][0])




json.loads(df.query("event_id=='d122731b'").reset_index(drop=True)["event_data"][0])




cor = df.query("event_id=='d122731b'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["correct"]))
# obj = df.query("event_id=='d122731b'").reset_index(drop=True)["event_data"].apply((lambda x: json.loads(x)["object"]))
np.unique(cor, return_counts=True)




_ = df.query("type=='Assessment' and title=='Cart Balancer (Assessment)'")
pd.unique(_.title)




pd.unique(_.event_id), pd.unique(_.event_code)




_ = df.query("type=='Assessment'")




_




_.loc[:, ["event_code", "event_id", "info", "title"]].drop_duplicates(["event_code", "event_id", "info", "title"]).sort_values("event_code").reset_index(drop=True)




# for i in pd.unique(_.event_id):
#     pprint.pprint([i, list(specs.query("event_id==@i")["info"])], width=150, indent=0, depth=2)




_.loc[:, ["event_code", "event_id"]].drop_duplicates(["event_code", "event_id"]).sort_values("event_code").reset_index(drop=True)




specs.query("event_id=='7ad3efc6'")




specs.query("info=='The start game event is triggered at the very beginning of the level (after the game finishes loading, don\\'t wait for intro movie to finish). This is used to compute things like time spent in game.'")




# Assessment evaluation-Cart Balancer (Assessment)
_ = accuracy(df).cart_assessment()
_




temp = label.merge(_, on=["game_session"])




len(temp), sum(temp["accuracy_group"] != temp["accuracy_group_"])




temp[temp["accuracy_group"] != temp["accuracy_group_"]]




temp.merge(temp_title).query("title=='Chest Sorter (Assessment)'")




_ = df.query("type=='Assessment' and title=='Chest Sorter (Assessment)'")
pd.unique(_.title)




pd.unique(_.event_id), pd.unique(_.event_code)




# for i in pd.unique(_.event_id):
#     pprint.pprint([i, list(specs.query("event_id==@i")["info"])], width=150, indent=0, depth=2)




_.loc[:, ["event_code", "event_id"]].drop_duplicates(["event_code", "event_id"]).sort_values("event_code").reset_index(drop=True)




# Assessment evaluation-Chest Sorter (Assessment)
_ = accuracy(df).chest_assessment()
_




plt.hist(temp.accuracy_)




plt.hist(temp.accuracy)




temp = label.merge(_, on=["game_session"])




len(temp), sum(temp["accuracy_group"] != temp["accuracy_group_"])




temp[temp["accuracy_group"] != temp["accuracy_group_"]]














# Assessment evaluation-Cauldron Filler (Assessment)
def cauldron_assessment(df):
    _ = df.query("title=='Cauldron Filler (Assessment)' and event_id=='392e14df'")
    _ = _.loc[:, ["game_session", "installation_id", "event_data"]]
    _["correct"] = _.event_data.apply(lambda x:(json.loads(x)["correct"] if "correct" in json.loads(x).keys() else -999))
    _["num_correct_"]=0
    _["num_incorrect_"]=0
    _.loc[_.correct==True, "num_correct_"] = 1
    _.loc[_.correct==False, "num_incorrect_"] = 1
    _ = _.groupby("game_session").sum().reset_index()
    _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["num_incorrect_"])
    
    return _




_ = cauldron_assessment(df)
temp = label.merge(_, on=["game_session"])




len(temp), sum(temp["accuracy"] != temp["accuracy_"])




temp[temp["accuracy"] != temp["accuracy_"]]




# Assessment evaluation-Mushroom Sorter (Assessment)
def mushroom_assessment(df):
    _ = df.query("title=='Mushroom Sorter (Assessment)' and event_id=='25fa8af4'")
    _ = _.loc[:, ["game_session", "installation_id", "event_data"]]
    _["correct"] = _.event_data.apply(lambda x:(json.loads(x)["correct"] if "correct" in json.loads(x).keys() else -999))
    _["num_correct_"]=0
    _["num_incorrect_"]=0
    _.loc[_.correct==True, "num_correct_"] = 1
    _.loc[_.correct==False, "num_incorrect_"] = 1
    _ = _.groupby("game_session").sum().reset_index()
    _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["num_incorrect_"])
    
    return _




_ = mushroom_assessment(df)
temp = label.merge(_, on=["game_session"])




len(temp), sum(temp["accuracy"] != temp["accuracy_"])




temp[temp["accuracy"] != temp["accuracy_"]]




# Assessment evaluation-Bird Measurer (Assessment)
def bird_assessment(df):
    _ = df.query("title=='Bird Measurer (Assessment)' and event_id=='17113b36'")
    _ = _.loc[:, ["game_session", "installation_id", "event_data"]]
    _["correct"] = _.event_data.apply(lambda x:(json.loads(x)["correct"] if "correct" in json.loads(x).keys() else -999))
    _["num_correct_"]=0
    _["num_incorrect_"]=0
    _.loc[_.correct==True, "num_correct_"] = 1
    _.loc[_.correct==False, "num_incorrect_"] = 1
    _ = _.groupby("game_session").sum().reset_index()
    _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["num_incorrect_"])
    
    return _




_ = bird_assessment(df)
temp = label.merge(_, on=["game_session"])




len(temp), sum(temp["accuracy"] != temp["accuracy_"])




temp[temp["accuracy"] != temp["accuracy_"]]




sns.distplot(label.accuracy)
plt.show()




_ = cart_assessment(df)
_ = _.append(chest_assessment(df))
_ = _.append(cauldron_assessment(df))
_ = _.append(mushroom_assessment(df))
_ = _.append(bird_assessment(df))




title_select = "Cart Balancer (Assessment)"
_ = df.query("type=='Assessment' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_id", "event_data"]]




for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




_ = df.query("type=='Assessment' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_id", "event_data", "event_code"]]
_["crystals"] = _.event_data.apply(lambda x: (json.loads(x)["crystals"] if "crystals" in json.loads(x).keys() else -999))
_["description"] = _.event_data.apply(lambda x: (json.loads(x)["description"] if "description" in json.loads(x).keys() else -999))
_["correct"] = _.event_data.apply(lambda x: (json.loads(x)["correct"] if "correct" in json.loads(x).keys() else -999))




# _ = df.query("type=='Assessment' and title==@title_select").reset_index(drop=True)
# _ = _.loc[:, ["installation_id", "game_session", "event_id", "event_data"]]
# # _["rounds"] = _.event_data.apply(lambda x: json.loads(x)["round"])
# _["crystals"] = _.event_data.apply(lambda x: (json.loads(x)["crystals"] if "crystals" in json.loads(x).keys() else -999))

# # bins_rounds = len(np.unique(_.groupby("game_session").rounds.max()))
# bins_levels = len(np.unique(_.groupby("game_session").crystals.max()))

# plt.figure(figsize=(25, 7))
# # sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").crystals.max(), bins=bins_levels, label="levels")
# plt.legend()
# plt.title(title_select)

# plt.show()




title_select = "Chest Sorter (Assessment)"
_ = df.query("type=='Assessment' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_id", "event_data"]]




for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




title_select = "Cauldron Filler (Assessment)"
_ = df.query("type=='Assessment' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_id", "event_data"]]




for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




title_select = "Mushroom Sorter (Assessment)"
_ = df.query("type=='Assessment' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_id", "event_data"]]




for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




title_select = "Bird Measurer (Assessment)"
_ = df.query("type=='Assessment' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_id", "event_data"]]




for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())










# _["rounds"] = _.event_data.apply(lambda x: json.loads(x)["round"])
# _["levels"] = _.event_data.apply(lambda x: (json.loads(x)["level"] if "level" in json.loads(x).keys() else -999))

# bins_rounds = len(np.unique(_.groupby("game_session").rounds.max()))
# bins_levels = len(np.unique(_.groupby("game_session").levels.max()))

# plt.figure(figsize=(25, 7))
# plt.subplot(1, 2, 1)
# sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
# plt.legend()
# plt.title(title_select)
# plt.subplot(1, 2, 2)
# sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
# sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
# plt.legend()
# plt.title(title_select)

# plt.show()




# 하나의 game_session에는 하나의 title
_ = df.groupby("game_session").title.unique()
_ = _.reset_index().groupby("game_session").size()
__ = _.reset_index().rename(columns={0:"counts"})
__[__.counts!=1]




df.query("game_session=='000050630c4b081b'")




df.head()




_ = df.query("type=='Clip'")




pd.unique(_.title)




_ = df.query("type=='Clip'")
for i in pd.unique(_.title):
    _ = df.query("type=='Clip' and title==@i").reset_index(drop=True)
    _ = _.loc[:, ["installation_id", "game_session", "event_id", "event_data"]]
    for j in pd.unique(_.event_id):
        print(i, j, json.loads(_.query("event_id==@j").reset_index(drop=True).loc[0, "event_data"]).keys())









_ = df.query("type=='Activity'")
pd.unique(_.title)




title_select = "Sandcastle Builder (Activity)"
_ = df.query("type=='Activity' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_id", "event_data"]]




for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




# _.query("game_session=='905b19967a1974d7'")




title_select = "Watering Hole (Activity)"
_ = df.query("type=='Activity' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_id", "event_data"]]




for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




title_select = "Bottle Filler (Activity)"
_ = df.query("type=='Activity' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_id", "event_data"]]




for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




title_select = "Chicken Balancer (Activity)"
_ = df.query("type=='Activity' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_id", "event_data"]]




for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




title_select = "Fireworks (Activity)"
_ = df.query("type=='Activity' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_id", "event_data"]]




for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())




title_select = "Flower Waterer (Activity)"
_ = df.query("type=='Activity' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_id", "event_data"]]




for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())









title_select = "Egg Dropper (Activity)"
_ = df.query("type=='Activity' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_id", "event_data"]]




for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())









title_select = "Bug Measurer (Activity)"
_ = df.query("type=='Activity' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_id", "event_data"]]




for i in pd.unique(_.event_id):
    print(i, json.loads(_.query("event_id==@i").reset_index(drop=True).loc[0, "event_data"]).keys())





























# accuracy = _.query("((event_code==2000) or (event_code==4100 and title!='Bird Measurer (Assessment)') or \
#                      (event_code==4110 and title=='Bird Measurer (Assessment)')) and (type=='Assessment')").reset_index(drop=True)

# accuracy["event_data_json"] = accuracy["event_data"].apply(lambda x: json.loads(x))

# accuracy["num_incorrect"] = accuracy["event_data_json"].apply(lambda x: (0 if x["correct"] else 1) if "correct" in x  else 0)
# accuracy["num_correct"] = accuracy["event_data_json"].apply(lambda x: (1 if x["correct"] else 0)  if "correct" in x  else 0)

# accuracy = accuracy.groupby(["installation_id", "game_session"]).agg(num_correct_pred = ("num_correct", "max"), num_incorrect_pred = ("num_incorrect", "sum"), ).reset_index()
# accuracy["accuracy_group_pred"] = accuracy["num_incorrect_pred"].apply(lambda x: 3 if x == 0 else (2 if x == 1 else 1)) * accuracy["num_correct_pred"]

# accuracy = accuracy.groupby(["installation_id"]).last().reset_index()
# accuracy.drop("game_session", axis=1, inplace=True)

