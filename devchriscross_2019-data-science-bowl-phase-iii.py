#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import json
import gc

import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress
pbar = ProgressBar()
pbar.register()

import statsmodels
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import numexpr 
numexpr.set_num_threads(1)

import os
def print_log(string):
    os.system(f'echo \"{string}\"')
    print(string)


# In[2]:


def load_dataset():
    print_log("Loading dataset ...")
    data_dtype = {"world": "category", "type": "category", "title": "category", "event_code": "category", "game_session": "category",
               "event_count": "uint8", "game_time": "uint32",
               "event_data": "str", "event_id": "category", "installation_id": "category"}
    df_train = pd.read_csv("../input/data-science-bowl-2019/train.csv", parse_dates=["timestamp"], dtype=data_dtype)
    
    assessment_attempt_query = "((event_code=='4100' & title!='Bird Measurer (Assessment)') | (event_code=='4110' & title=='Bird Measurer (Assessment)'))"
    attempts_query = "type=='Assessment' & " +  assessment_attempt_query
    ids = df_train.query(attempts_query).installation_id.unique().tolist()
    df_train = df_train[df_train.installation_id.isin(ids)].sort_values(["installation_id", "timestamp"])
    gc.collect()
    
    print_log("Grouping each game session by Assessment ...")
    # List of Assessment game sessions with an accuracy group
    df_train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")
    assessment_gs_labels = df_train_labels.game_session.tolist()

    # Construct a map of game_session to index (e.g 0,1,2...,n)
    df_train = df_train.reset_index().drop(columns=["index"])
    gs_index_map = df_train[df_train.game_session.isin(assessment_gs_labels)]        .groupby(["game_session"]).tail(1).reset_index()        .set_index("game_session")["index"].to_dict()
    gs_index_map.update({0: 0})
    
    # Create a map of game_session to labeled accuracy group (e.g 0,1,2,3)
    gs_accgroup_map = df_train_labels.set_index("game_session")
    
    # Same with assessment_gs_labels but preserves ordering of game_session per installation_id
    game_session_list = list(gs_index_map.keys())
    game_session_list.insert(0, 0)
    
    # Construct a dataframe which tells what group of indexes in the dataset will be used to evaluate a given Assessment
    df_gs_map = pd.Series(game_session_list).to_frame().rename(columns={0: "start_gs"})
    df_gs_map["game_session"] = df_gs_map.start_gs.shift(-1)
    df_gs_map["accuracy_group"] = df_gs_map.game_session.map(gs_accgroup_map.accuracy_group.to_dict())
    df_gs_map["start_gs"] = df_gs_map.start_gs.map(gs_index_map)
    df_gs_map["end_gs"] = df_gs_map.start_gs.shift(-1)
    df_gs_map = df_gs_map[:-2]
    
    # Construct a series for which Assessment game session each index belongs to 
    # Counts how many game session for each group of indexes to be created
    print_log("Creating gs_to_eval variable ...")
    gs_list = list()
    for i, row in enumerate(df_gs_map.itertuples()):
        quantity = int(row.end_gs-row.start_gs)
        quantity = quantity + (1 if i==0 else 0)
        gs_list.extend([row.game_session]*quantity)
    df_gs_to_eval = pd.Series(gs_list).to_frame()
    gc.collect()
    
    # Concatenate the series into the dataset and map the accuracy group accordingly
    df_train = pd.concat([df_train, df_gs_to_eval], axis=1).rename(columns={0: "gs_to_eval"})
    df_train["accuracy_group"] = df_train["gs_to_eval"].map(gs_accgroup_map.accuracy_group.to_dict()).astype("category")

    return df_train


# In[3]:


import re
class PropExtractor():
    def __init__(self, prop):
        self.prop = prop
    
    def extract_bool(self, event_data):
        prop = re.findall(r"(?<=" + self.prop + "\"\:)\s?\w+", event_data)
        if len(prop) > 0:
            return 1 if prop[0].strip()=="true" else 0
        else:
            return -1
        
    def extract_int(self, event_data):
        prop = re.findall(r"(?<=" + self.prop + "\"\:)\s?\d+", event_data)
        if len(prop) > 0:
            return int(prop[0].strip())
        else:
            return -1
        
    def extract_string(self, event_data):
        prop = re.findall(r"(?<=" + self.prop + "\"\:\")\s?\w+", event_data)
        if len(prop) > 0:
            return prop[0].strip()
        else:
            return ""


# In[4]:


def extract_args(df_train):
    print_log("Extracting media_type ...")
    argsint = ["level", "misses", "dwell_time", "round", "duration", "total_duration"]
    df_train["media_type"] = df_train.event_data.map(PropExtractor("media_type").extract_string).astype("category")
    
    print_log("Extracting correct ...")
    df_train["correct"] = df_train.event_data.map(PropExtractor("correct").extract_bool).astype("int8")
    
    for args in argsint:
        print_log(" ".join(["Extracting", str(args), "..."]))
        df_train[args] = df_train.event_data.map(PropExtractor(args).extract_int)
        
    return df_train


# In[5]:


def add_prev_assessment(df_train):
    gs_list = df_train.gs_to_eval.unique().tolist()
    gs_with_prev_assess = df_train[(df_train["type"]=="Assessment") & ~df_train.game_session.isin(gs_list)].groupby(["gs_to_eval"]).head(1).gs_to_eval.tolist()
    df_train["has_prev_assessment"] = False
    df_train.loc[df_train.gs_to_eval.isin(gs_with_prev_assess), "has_prev_assessment"] = True
    
    return df_train

def add_difficult(df_train):
    df_specs = pd.read_csv("../input/data-science-bowl-2019/specs.csv")
    df_specs["is_difficult"] = False
    df_specs.loc[df_specs["info"].str.contains("difficult"), "is_difficult"] = True

    df_train["is_difficult"] = False
    event_id_difficult = df_specs[df_specs.is_difficult].event_id.tolist()
    df_train.loc[df_train.event_id.isin(event_id_difficult), "is_difficult"] = True
    
    return df_train 

def add_clip_duration(df_train):
    df_media = pd.read_csv("../input/data-science-bowl-2019-media-sequence/media_sequence.csv")
    clip_duration_map = df_media.set_index("title").duration.to_dict()

    df_train.loc[df_train["type"]=='Clip', "duration"] = df_train[df_train["type"]=='Clip'].title.map(clip_duration_map)
    df_train["prev_ts"] = df_train.timestamp.shift(-1)
    df_train["clip_runtime"] = (df_train.prev_ts - df_train.timestamp).dt.total_seconds()

    completed_clips = (df_train["type"]=="Clip") & ((df_train.clip_runtime > df_train.duration) | (df_train.clip_runtime <= 0))
    df_train.loc[completed_clips, "clip_runtime"] =  df_train[completed_clips].duration
    df_train["is_completed"] =  False
    df_train.loc[completed_clips, "is_completed"] =  True
    
    return df_train


# In[6]:


outlier_cap_val = dict()
def remove_outliers(df, col, extreme=None):
#     q1 = df[col].quantile(0.25)
#     q3 = df[col].quantile(0.75)
    
#     iqr = q3 - q1
#     iqr_multiplier = extreme if extreme else 1.5
#     lower_limit = q1 - (iqr_multiplier*iqr)
#     upper_limit = q3 + (iqr_multiplier*iqr)
    
#     iqr_lower_filter = df[col] >= lower_limit
#     iqr_upper_filter = df[col] <= upper_limit
#     iqr_filter = iqr_lower_filter & iqr_upper_filter
#     outliers = df[~iqr_filter]
    
#     df = df.copy()
#     df.loc[df[col] < lower_limit, col] = lower_limit
#     df.loc[df[col] > upper_limit, col] = upper_limit
#     outlier_cap_val[col] = (lower_limit, upper_limit)
    
#     outlier_percent = len(outliers)/(len(df))*100
#     print("Percentage of oultiers for", col, ":", outlier_percent)
    
    return df


# In[7]:


def handle_outliers(df_train):
    
    df_correct = remove_outliers(df_correct, "correct")
    df_correct = remove_outliers(df_correct, "round")
#     df_level = remove_outliers(df_level, "level_min")
#     df_level = remove_outliers(df_level, "level_max")
    df_level = remove_outliers(df_level, "level_mean")
    
#     df_misses = remove_outliers(df_misses, "misses_sum")
#     df_misses = remove_outliers(df_misses, "misses_mean")
#     df_misses = remove_outliers(df_misses, "misses_max")
    df_misses = remove_outliers(df_misses, "miss_rate_gs")
#     df_misses = remove_outliers(df_misses, "miss_rate_round")

#     df_dwell = remove_outliers(df_dwell, "dwell_time_max")
#     df_dwell = remove_outliers(df_dwell, "dwell_time_sum")
    df_dwell = remove_outliers(df_dwell, "dwell_time_mean")
    
    df_correct = remove_outliers(df_correct, "drag_time")
    df_correct = remove_outliers(df_correct, "has_prev_assessment")
#     df_correct = remove_outliers(df_correct, "correct_ratio_drag_duration")

    df_media_playback = remove_outliers(df_media_playback, "total_duration")
    df_media_playback = remove_outliers(df_media_playback, "is_interrupted")
    
#     df_clip = remove_outliers(df_clip, "clip_runtime")
#     df_clip = remove_outliers(df_clip, "is_completed")

#     for col in df_eventcode.columns:
#         df_eventcode = remove_outliers(df_eventcode, col, extreme=3.0)

#     df_correct = remove_outliers(df_correct, "duration")


# In[8]:


def accumulate(df, col, op):
    gs_to_eval = df["gs_to_eval"].to_dict()
    if op == "mean":
        df = df.groupby(["installation_id"])[[col]].expanding().mean().reset_index()
    elif op == "sum":
        df = df.groupby(["installation_id"])[[col]].expanding().sum().reset_index()
    df["gs_to_eval"] = df["level_1"].map(gs_to_eval)
    df = df.drop(columns=["level_1"]).groupby(["gs_to_eval"]).tail(1).set_index("gs_to_eval")[[col]]
    return df


# In[9]:


# Capped outliers
def correct_game_agg(df_train):
    df_correct = df_train[(df_train["type"]=="Game") & (df_train.correct!=-1)]
    
    max_rounds = df_correct.groupby(["installation_id", "title", "game_session"]).tail(1)
    max_rounds = max_rounds.groupby(["installation_id"]).mean()["round"]
    df_correct = df_correct.groupby(["installation_id"]).mean().drop(columns=["round"])
    df_correct = df_correct.join(max_rounds)[["correct", "round"]]
    
    return df_correct

# Capped outliers
def level_agg(df_train):
    df_level = df_train[df_train.level != -1]
    
    df_level = df_level.groupby(["installation_id", "game_session"]).tail(1)        .groupby(["installation_id"]).agg(["min", "max", "mean"])
    df_level.columns = ["_".join(col) for col in df_level.columns]
    df_level = df_level.reset_index().set_index("installation_id")[["level_mean"]]
    
    return df_level

# Capped outliers
def media_playback_agg(df_train):
    df_media_playback = df_train[(df_train.media_type!="") & ((df_train["type"]=="Game") | (df_train["type"]=="Activity"))]

    df_media_playback["is_interrupted"] = df_media_playback.total_duration == -1
    df_media_playback.loc[df_media_playback.is_interrupted, "total_duration"] = df_media_playback[df_media_playback.is_interrupted].duration
    
    num_isinterrupted = df_media_playback.groupby(["installation_id"]).mean().is_interrupted
    df_media_playback = df_media_playback.groupby(["installation_id", "title", "media_type", "game_session"], observed=True).sum().reset_index()
    df_media_playback = df_media_playback.groupby(["installation_id", "title", "media_type"], observed=True).mean().reset_index()
    df_media_playback = df_media_playback.groupby(["installation_id"], observed=True).sum()
    df_media_playback = df_media_playback.drop(columns=["is_interrupted"]).join(num_isinterrupted)
    df_media_playback = df_media_playback[["total_duration", "is_interrupted"]]
    
    return df_media_playback

def misses_assess_agg(df_train):
    df_misses = df_train[(df_train.misses!=-1) & (df_train["type"]=="Assessment")]
    return misses_agg(df_misses, "assess")
    
def misses_game_agg(df_train):
    df_misses = df_train[(df_train.misses!=-1) & ((df_train["type"]=="Game") | (df_train["type"]=="Activity"))]
    return misses_agg(df_misses, "game")
    
# Capped outliers
def misses_agg(df_misses, prefix):
    df_misses = df_misses.sort_values(["installation_id", "game_session"])

    avg_miss_per_round = df_misses.groupby(["installation_id", "round"], observed=True).mean()[["misses"]].reset_index()
    total_avg_miss_per_round = avg_miss_per_round.groupby(["installation_id"]).sum()[["misses"]]
    max_round_per_gs = avg_miss_per_round.groupby(["installation_id"]).tail(1).set_index(["installation_id"])[["round"]]
    miss_rate_per_max_round = total_avg_miss_per_round.join(max_round_per_gs)
    miss_rate_per_max_round["miss_rate_round"] = miss_rate_per_max_round["misses"] / miss_rate_per_max_round["round"]
    miss_rate_per_max_round = miss_rate_per_max_round.drop(columns=["misses", "round"])

    total_miss_per_gs = df_misses.groupby(["installation_id", "game_session"], observed=True).sum()[["misses"]]
    max_round_per_local_gs = df_misses.groupby(["installation_id", "game_session"], observed=True).tail(1).set_index(["installation_id", "game_session"])[["round"]]
    avg_miss_per_local_gs_round = total_miss_per_gs.join(max_round_per_local_gs)
    avg_miss_per_local_gs_round["miss_rate_gs"] = avg_miss_per_local_gs_round["misses"] / avg_miss_per_local_gs_round["round"]
    avg_miss_per_local_gs_round = avg_miss_per_local_gs_round.groupby(["installation_id"]).mean().drop(columns=["misses", "round"])
    
    df_misses = df_misses.groupby(["installation_id"], observed=True).agg(["sum", "mean", "max"])[["misses"]]
    df_misses.columns = ['_'.join(col) for col in df_misses.columns]
    df_misses = df_misses.join(avg_miss_per_local_gs_round).join(miss_rate_per_max_round)[["miss_rate_gs"]]
    
    df_misses.columns = ["_".join([prefix, col]) for col in df_misses.columns]
    
    return df_misses

# Capped outliers
def dwell_time_agg(df_train):
    df_dwell = df_train[df_train.dwell_time!=-1]
    
    df_dwell = df_dwell.groupby(["installation_id"], observed=True).agg(["max", "sum", "mean"])
    df_dwell.columns = ["_".join(col) for col in df_dwell.columns]
    df_dwell = df_dwell[["dwell_time_mean"]]
    
    return df_dwell

def clip_runtime_agg(df_train):
    df_clip = df_train[df_train["type"]=='Clip']

    num_completed = df_clip.groupby(["installation_id"]).mean().is_completed
    df_clip = df_clip.groupby(["installation_id", "title"], observed=True).mean()
    df_clip = df_clip.groupby(["installation_id"]).sum()
    df_clip = df_clip.drop(columns=["is_completed"]).join(num_completed)
    df_clip = df_clip[["clip_runtime", "is_completed"]]
    
    return df_clip

def is_difficult_agg(df_train): 
    df_difficult = df_train.groupby(["installation_id"]).mean()[["is_difficult"]]
    return df_difficult

def event_code_agg(df_train):
    df_eventcode = df_train.groupby(["installation_id", "event_code"]).count()[["event_count"]].fillna(np.nan).reset_index()            .pivot(columns="event_code", index="installation_id", values=["event_count"])
    df_eventcode.columns = ["_".join(col) for col in df_eventcode.columns]
    for col in df_eventcode.columns:
        df_eventcode = remove_outliers(df_eventcode, col, extreme=3.0)
    
    return df_eventcode

# Capped outliers
def assessment_agg(df_train):
    gs_list = df_train.gs_to_eval.unique().tolist()
    df_assessment = df_train[(df_train["type"]=="Assessment") & ~df_train.game_session.isin(gs_list)]
    
    df_assessment_count = df_assessment.groupby(["installation_id", "game_session"]).head(1)
    df_assessment_count = df_assessment_count.groupby(["installation_id"]).sum()[["has_prev_assessment"]]
    
    df_correct = df_assessment[(df_train.correct!=-1)]
    df_correct = remove_outliers(df_correct, "duration")
    drag_time = df_correct[df_correct.duration!=-1].groupby(["installation_id"]).mean()[["duration"]].rename(columns={"duration":"drag_time"})
    df_correct = df_correct.groupby(["installation_id"]).mean().rename(columns={"correct": "correct_assessment"})
    df_correct = df_correct.join(drag_time)[["correct_assessment", "drag_time"]]
    
    df_correct = df_correct.join(df_assessment_count)
    
    return df_correct


# In[10]:


def aggregate_data(df_train):
    print_log("Aggregating variables by player ...")
    df_train_data = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")
    df_train_data = df_train_data[["game_session", "installation_id", "accuracy_group"]]
#     df_train_data = df_train.groupby(["gs_to_eval"]).head(1).set_index("gs_to_eval")[["accuracy_group", "installation_id"]]
    df_train_data = df_train_data.merge(correct_game_agg(df_train), on='installation_id', how='left')
    df_train_data = df_train_data.merge(level_agg(df_train), on='installation_id', how='left')
    df_train_data = df_train_data.merge(media_playback_agg(df_train), on='installation_id', how='left')
    df_train_data = df_train_data.merge(misses_game_agg(df_train), on='installation_id', how='left')
    df_train_data = df_train_data.merge(misses_assess_agg(df_train), on='installation_id', how='left')
    df_train_data = df_train_data.merge(dwell_time_agg(df_train), on='installation_id', how='left')
    df_train_data = df_train_data.merge(clip_runtime_agg(df_train), on='installation_id', how='left')
    df_train_data = df_train_data.merge(is_difficult_agg(df_train), on='installation_id', how='left')
    df_train_data = df_train_data.merge(event_code_agg(df_train), on='installation_id', how='left')
    df_train_data = df_train_data.merge(assessment_agg(df_train).fillna(0), on='installation_id', how='left')
    
    df_train_data = df_train_data.drop(columns=["event_count_2010", "event_count_2035", "event_count_2075", 
                                                "event_count_3010", "event_count_3020", "event_count_3021",
                                                "event_count_2040", "event_count_2050", "event_count_2030",
                                               "event_count_2020", "event_count_4235", "event_count_5010",
                                               "event_count_2000"])
    return df_train_data


# In[11]:


def plot_distribution_mean(df_train_data):
    df_train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")
    class_min = df_train_labels.accuracy_group.value_counts().min()
    sampled_gs = df_train_labels.groupby(["accuracy_group"]).apply(lambda x: x.sample(class_min, random_state=1)).game_session.tolist()
    df_train_sample = df_train_data.reset_index()[df_train_data.reset_index().gs_to_eval.isin(sampled_gs)]

    variable_num = len(df_train_sample.columns)-2
    f, ax = plt.subplots(variable_num,2,figsize=(25,7*variable_num))
    for i, col in enumerate(df_train_sample.columns[2:]):
        df_plot = df_train_sample[~df_train_sample[col].isnull()]
        sns.boxenplot(x=col, y="accuracy_group", data=df_plot, ax=ax[i, 0])
        sns.pointplot(x=col, y="accuracy_group", data=df_plot, ax=ax[i, 1])


# In[12]:


from pandas.api.types import is_numeric_dtype
def print_missing_percent(df_train_data):
    for col in df_train_data.columns:
        if(is_numeric_dtype(df_train_data[col])):
            print(col, "Percentage of missing values:", df_train_data[col].isnull().mean(), "====","Minimum value:", df_train_data[col].min())


# In[13]:


def fill_missing_values(df_train_data):
    df_train_data["correct"] = df_train_data["correct"].fillna(-1)
#     df_train_data["correct_ratio_round"] = df_train_data["correct_ratio_round"].fillna(-1)

#     df_train_data["level_min"] = df_train_data["level_min"].fillna(-1)
#     df_train_data["level_max"] = df_train_data["level_max"].fillna(-1)
    df_train_data["level_mean"] = df_train_data["level_mean"].fillna(-1)

#     df_train_data["game_misses_sum"] = df_train_data["game_misses_sum"].fillna(-1)
#     df_train_data["game_misses_mean"] = df_train_data["game_misses_mean"].fillna(-1)
#     df_train_data["game_misses_max"] = df_train_data["game_misses_max"].fillna(-1)
    df_train_data["game_miss_rate_gs"] = df_train_data["game_miss_rate_gs"].fillna(-1)
#     df_train_data["game_miss_rate_round"] = df_train_data["game_miss_rate_round"].fillna(-1)

#     df_train_data["assess_misses_sum"] = df_train_data["assess_misses_sum"].fillna(-1)
#     df_train_data["assess_misses_mean"] = df_train_data["assess_misses_mean"].fillna(-1)
#     df_train_data["assess_misses_max"] = df_train_data["assess_misses_max"].fillna(-1)
    df_train_data["assess_miss_rate_gs"] = df_train_data["assess_miss_rate_gs"].fillna(-1)
#     df_train_data["assess_miss_rate_round"] = df_train_data["assess_miss_rate_round"].fillna(-1)

#     df_train_data["dwell_time_max"] = df_train_data["dwell_time_max"].fillna(-1)
    df_train_data["dwell_time_mean"] = df_train_data["dwell_time_mean"].fillna(-1)
#     df_train_data["dwell_time_sum"] = df_train_data["dwell_time_sum"].fillna(-1)

    df_train_data["is_completed"] = df_train_data["is_completed"].fillna(-1)
    df_train_data["correct_assessment"] = df_train_data["correct_assessment"].fillna(-1)
    df_train_data["drag_time"] = df_train_data["drag_time"].fillna(-1)
#     df_train_data["correct_ratio_drag_duration"] = df_train_data["correct_ratio_drag_duration"].fillna(-1)

    df_train_data["total_duration"] = df_train_data["total_duration"].fillna(0)
    df_train_data["is_interrupted"] = df_train_data["is_interrupted"].fillna(0)
    df_train_data["round"] = df_train_data["round"].fillna(0)
    df_train_data["clip_runtime"] = df_train_data["clip_runtime"].fillna(0)
    df_train_data["has_prev_assessment"] = df_train_data["has_prev_assessment"].fillna(0)

    event_code_cols = [col for col in df_train_data.columns if "event_count" in col] 
    df_train_data[event_code_cols] = df_train_data[event_code_cols].fillna(0)
    
    plt.figure(figsize=(30, 30))
    sns.heatmap(df_train_data.corr(), annot=True, fmt=".1f").set_title("Correlational Matrix")
    
    return df_train_data


# In[14]:


def compile_train_data():
    df_train = load_dataset()
    gc.collect()
    df_train = extract_args(df_train)
    gc.collect()
    print_log("Adding additional variables")
    df_train = add_clip_duration(df_train)
    gc.collect()
    df_train = add_difficult(df_train)
    gc.collect()
    df_train = add_prev_assessment(df_train)
    gc.collect()
    df_train_data = aggregate_data(df_train)
    gc.collect()
#     plot_distribution_mean(df_train_data)
#     print_missing_percent(df_train_data)
#     df_train_data = fill_missing_values(df_train_data)
#     gc.collect()
    
    return df_train_data

df_train_data = compile_train_data()


# In[15]:


gs_index_map = df_train[df_train.game_session==df_train.gs_to_eval].groupby(["gs_to_eval"]).head(1).reset_index().set_index("gs_to_eval")[["index"]]
gs_index_map["index"] = gs_index_map["index"] - 1
gs_index_map = gs_index_map["index"].to_dict()
id_index_map = df_train.groupby(["installation_id"]).head(1).reset_index().set_index("installation_id")["index"].to_dict()


# In[16]:


plt.figure(figsize=(30, 30))
sns.heatmap(df_train_data.drop(columns=["installation_id"]).corr(), annot=True, fmt=".1f").set_title("Correlational Matrix")


# In[17]:


from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, learning_curve, GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, cohen_kappa_score, make_scorer
from sklearn.model_selection import train_test_split

def validate_model(rfc, X_train, y_train, groups):
    scoring_list = {"accuracy": make_scorer(accuracy_score), 
                    "precision_weighted": make_scorer(precision_score, average="weighted"), 
                    "recall_weighted": make_scorer(recall_score, average="weighted"), 
                    "f1_weighted": make_scorer(f1_score, average="weighted"), 
                    "quadratic_kappa": make_scorer(cohen_kappa_score, weights="quadratic")}
    scores = cross_validate(rfc, X_train, y_train, 
                            cv=GroupKFold(3).split(X_train, y_train, groups=groups),
                            scoring=scoring_list, return_train_score=True, n_jobs=-1)
#     print("Accuracy score (train set):", np.mean(scores["train_accuracy"]))
#     print("Accuracy score (validation set):", np.mean(scores["test_accuracy"]), "\n")
    
#     print("Precision score (train set):", np.mean(scores["train_precision_weighted"]))
#     print("Precision score (validation set):", np.mean(scores["test_precision_weighted"]), "\n")
    
#     print("Recall score (train set):", np.mean(scores["train_recall_weighted"]))
#     print("Recall score (validation set):", np.mean(scores["test_recall_weighted"]), "\n")
    
#     print("F1 score (train set):", np.mean(scores["train_f1_weighted"]))
#     print("F1 score (validation set):", np.mean(scores["test_f1_weighted"]), "\n")
    
    print("Quadratic Kappa score (train set):", np.mean(scores["train_quadratic_kappa"]))
    print("Quadratic Kappa score (validation set):", np.mean(scores["test_quadratic_kappa"]))

dfx_train = df_train_data.drop(columns=["accuracy_group", "game_session", "installation_id"]).to_numpy()
dfy_train = df_train_data["accuracy_group"].to_numpy().ravel()
cweights = dict(zip(range(4), compute_class_weight(class_weight="balanced", classes=np.unique(dfy_train), y=dfy_train)))
kappa_score = make_scorer(cohen_kappa_score, weights="quadratic")

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(df_train_data.installation_id.unique().tolist())
id_groups = encoder.transform(df_train_data.installation_id.values)

for train_index, test_index in GroupKFold().split(dfx_train, dfy_train, groups=id_groups):
    X_train, X_test = dfx_train[train_index], dfx_train[train_index]
    y_train, y_test = dfy_train[train_index], dfy_train[train_index]
    rfc = HistGradientBoostingClassifier(loss="categorical_crossentropy", learning_rate=0.1, max_iter=100, max_depth=None, l2_regularization=1,
                                        max_bins=255, scoring=kappa_score, validation_fraction=0.25, verbose=0, random_state=1)
    validate_model(rfc, X_train, y_train, id_groups[train_index])
    
    rfc.fit(X_train, y_train)
    preds = rfc.predict(X_test)
    print("Quadratic Kappa score: (test set)", cohen_kappa_score(y_test, preds, weights="quadratic"), "====\n")


# In[18]:


from sklearn.model_selection import GroupKFold
def plot_learning_curve(rfc, X, y, groups, scoring):
    sns.set()
    train_sizes = list(range(707, int(len(y)*0.8), 707))
    rfc_curve = learning_curve(rfc, X, y, train_sizes=train_sizes, n_jobs=-1, cv=GroupKFold(5).split(X, y, groups=groups), random_state=1, scoring=scoring, verbose=4)

    df_lcurve = pd.DataFrame(0, columns=["training_size", "score", "type"], index=range(len(train_sizes)*2))
    iteration_run = list(rfc_curve[0])
    iteration_run.extend(rfc_curve[0])
    df_lcurve["training_size"] = iteration_run

    score_run = [np.mean(col) for col in rfc_curve[1]]
    score_run.extend([np.mean(col) for col in rfc_curve[2]])
    df_lcurve["score"] = score_run
    df_lcurve["type"] = "train"
    df_lcurve.loc[len(train_sizes):,"type"] = "validation"

    g = sns.relplot(kind="line", y="score", x="training_size", data=df_lcurve, hue="type", style="type", markers=True, dashes=False)
    g.fig.suptitle(" ".join([str(scoring), "Performance over Training size"]))
    g.fig.set_figwidth(20)


# In[19]:


################ Grid Search ################
from sklearn.model_selection import GridSearchCV
rfc = HistGradientBoostingClassifier(loss="categorical_crossentropy", max_iter=100, n_iter_no_change=25, scoring=kappa_score, validation_fraction=0.25, verbose=4, random_state=1)
param_grid = {"max_bins": [9, 10, 11], "l2_regularization": [0, 0.01, 0.015], "max_bins": [9, 10, 11], "max_depth": [5, 6, 7]}
gsc = GridSearchCV(estimator=rfc, param_grid=param_grid, scoring=kappa_score, n_jobs=-1, verbose=4, cv=GroupKFold(3).split(dfx_train, dfy_train, groups=id_groups))
grid_result = gsc.fit(dfx_train, dfy_train)
grid_result.best_params_


# In[20]:


rfc = HistGradientBoostingClassifier(loss="categorical_crossentropy", learning_rate=0.1, max_iter=100, n_iter_no_change=25, max_depth=6, l2_regularization=0.01,
                                        max_bins=9, scoring=kappa_score, validation_fraction=0.25, verbose=4, random_state=1)
validate_model(rfc, dfx_train, dfy_train, id_groups)
plot_learning_curve(rfc, dfx_train, dfy_train, id_groups, kappa_score)


# In[21]:


from sklearn.metrics import roc_auc_score
for train_index, test_index in GroupKFold(5).split(dfx_train, dfy_train, groups=id_groups):
    X_train, X_test = dfx_train[train_index], dfx_train[train_index]
    y_train, y_test = dfy_train[train_index], dfy_train[train_index]
    rfc = HistGradientBoostingClassifier(loss="categorical_crossentropy", learning_rate=0.1, max_iter=100, n_iter_no_change=25,
                                         max_depth=6, l2_regularization=0.01, max_bins=9, scoring=kappa_score, 
                                         validation_fraction=0.25, verbose=0, random_state=1)
    rfc.fit(X_train, y_train)
    preds = rfc.predict(X_test)
    print("Quadratic Kappa score: (test set)", cohen_kappa_score(y_test, preds, weights="quadratic"))
    print("ROC Area Under Curve Score: ", roc_auc_score(y_test, rfc.predict_proba(X_test), multi_class="ovo"), "\n")


# In[22]:


from joblib import dump
rfc = HistGradientBoostingClassifier(loss="categorical_crossentropy", learning_rate=0.1, max_iter=100, n_iter_no_change=25,
                                         max_depth=6, l2_regularization=0.01, max_bins=9, scoring=kappa_score, validation_fraction=0.25, 
                                         verbose=4, random_state=1)
rfc.fit(dfx_train, dfy_train)
dump(rfc, "rfc_accgroup_model.joblib")

