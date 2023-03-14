#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import gc
import os

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress
pbar = ProgressBar()
pbar.register()

from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype, is_string_dtype

def print_log(string):
    os.system(f'echo \"{string}\"')


# In[2]:


def load_dataset():
    print_log("Reading dataset ...")

    df_train = dask.compute(dd.read_parquet("../input/2019-data-science-bowl-phase-i-computation-only/df_train.parquet"))[0]
    df_train = df_train.drop(columns=["event_data", "args_encoded", "argsint_encoded", "argsstring_encoded", "argsobject_encoded", "argsarray_encoded"])            .sort_values(["installation_id", "timestamp"])
    gc.collect()

    print_log("Filling NA values ...")
    numeric_cols = []
    category_cols = []

    for col in df_train.columns:
        if is_numeric_dtype(df_train[col]):
            df_train[col] = df_train[col].fillna(-1)
            numeric_cols.append(col)
        if is_string_dtype(df_train[col]):
            df_train[col] = df_train[col].replace("", "NaN")
            df_train[col] = df_train[col].astype("category")
            category_cols.append(col)
    gc.collect()
    
    print_log("Optimizing column dtypes ...")
    max_values = df_train[numeric_cols].max()
    min_values = df_train[numeric_cols].min()

    column_items = []
    for col in numeric_cols:
        cast_to_int = np.array_equal(df_train[col].to_numpy(), df_train[col].to_numpy().astype(int))
        column_items.append([col, min_values[col], max_values[col], cast_to_int])
    df_column = pd.DataFrame(column_items, index=range(len(numeric_cols)), columns=["col_name", "min", "max", "cast_to_int"])
    
    df_column["dtype"] = "int64"
    df_column.loc[df_column.cast_to_int & (df_column["max"] < np.iinfo(np.int32).max), "dtype"] = "int32"
    df_column.loc[df_column.cast_to_int & (df_column["max"] < np.iinfo(np.int16).max), "dtype"] = "int16"
    df_column.loc[df_column.cast_to_int & (df_column["max"] < np.iinfo(np.int8).max), "dtype"] = "int8"

    int32_cols = df_column[df_column["dtype"]=="int32"].col_name.tolist()
    int16_cols = df_column[df_column["dtype"]=="int16"].col_name.tolist()
    int8_cols = df_column[df_column["dtype"]=="int8"].col_name.tolist()

    df_train[int32_cols] = df_train[int32_cols].astype("int32")
    df_train[int16_cols] = df_train[int16_cols].astype("int16")
    df_train[int8_cols] = df_train[int8_cols].astype("int8")
    gc.collect()
    
    return df_train


# In[3]:


def group_game_session_by_assessment(df_train, df_train_labels):
    print_log("Grouping each game session by Assessment ...")
    # List of Assessment game sessions with an accuracy group
    assessment_gs_labels = df_train_labels.game_session.tolist()

    # Construct a map of game_session to index (e.g 0,1,2...,n)
    # The original index contains concatenated values of columns from game_session, timestamp, etc. so index needs to be reset twice
    df_train = df_train.reset_index()
    gs_index_map = df_train[df_train.game_session.isin(assessment_gs_labels)]        .groupby(["game_session"]).tail(1).reset_index()        .set_index("game_session")["level_0"].to_dict()
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
    gs_list = list()
    for i, row in enumerate(df_gs_map.itertuples()):
        quantity = int(row.end_gs-row.start_gs)
        quantity = quantity + (1 if i==0 else 0)
        gs_list.extend([row.game_session]*quantity)
    df_gs_to_eval = pd.Series(gs_list).to_frame()
    
    # Concatenate the series into the dataset and map the accuracy group accordingly
    df_train = pd.concat([df_train, df_gs_to_eval], axis=1).rename(columns={0: "gs_to_eval"})
    df_train["accuracy_group"] = df_train["gs_to_eval"].map(gs_accgroup_map.accuracy_group.to_dict()).astype("category")
    df_train["accuracy"] = df_train["gs_to_eval"].map(gs_accgroup_map.accuracy.to_dict()).astype("float32")
    df_train["num_correct"] = df_train["gs_to_eval"].map(gs_accgroup_map.num_correct.to_dict()).astype("uint8")
    df_train["num_incorrect"] = df_train["gs_to_eval"].map(gs_accgroup_map.num_incorrect.to_dict()).astype("uint8")
    return df_train


# In[4]:


def remove_outliers(df, col, extreme=None):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    
    iqr = q3 - q1
    iqr_multiplier = extreme if extreme else 1.5
    lower_limit = q1 - (iqr_multiplier*iqr)
    upper_limit = q3 + (iqr_multiplier*iqr)
    
    iqr_lower_filter = df[col] >= lower_limit
    iqr_upper_filter = df[col] <= upper_limit
    iqr_filter = iqr_lower_filter & iqr_upper_filter
    outliers = df[~iqr_filter]
    
    df = df.copy()
    df.loc[df[col] < lower_limit, col] = lower_limit
    df.loc[df[col] > upper_limit, col] = upper_limit
    
    
    outlier_percent = len(outliers)/(len(df))*100
    print("Percentage of oultiers for", col, ":", outlier_percent)
    
    return df, outliers


# In[5]:


df_train = load_dataset()
df_specs = pd.read_parquet("../input/2019-data-science-bowl-phase-i-computation-only/df_specs.parquet")

# Classes are highly imbalanced especially for Accuracy Group 3
df_train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")
df_train = group_game_session_by_assessment(df_train, df_train_labels)

class_min = df_train_labels.accuracy_group.value_counts().min()
sampled_gs = df_train_labels.groupby(["accuracy_group"]).apply(lambda x: x.sample(class_min, random_state=1)).game_session.tolist()
df_train = df_train[df_train.gs_to_eval.isin(sampled_gs)]

gc.collect()
df_train.sample(5)


# In[6]:


base_cols = ['gs_to_eval', 'event_id', 'game_session', 'timestamp', 'installation_id', 'title', 'type']
event_data_cols = ['level', 'total_duration',
       'scale_weights', 'misses', 'target_containers', 'containers',
       'hole_position', 'duration', 'weights', 'dinosaur_weight', 'bowl_id',
       'position', 'shell_size', 'dinosaur_count', 'size', 'scale_weight',
       'dwell_time', 'distance', 'water_level', 'weight', 'round',
       'table_weights', 'target_weight', 'group', 'correct', 'mode',
       'description', 'animal', 'item_type', 'object_type', 'object',
       'media_type', 'toy_earned', 'identifier', 'dinosaur', 'source']
numeric_cols = [col for col in event_data_cols if is_numeric_dtype(df_train[col])]
string_cols = [col for col in event_data_cols if is_string_dtype(df_train[col])]
sns.set()


# In[7]:


event_data_cols


# In[8]:


df_train_ag = df_train[(df_train["type"]=="Activity") | (df_train["type"]=="Game")]
df_train_ag.loc[:,"type"] = df_train_ag["type"].astype("str").astype("category")
df_train_ag.loc[:,"title"] = df_train_ag["title"].astype("str").astype("category")
gs_accgroup_map = df_train_ag.groupby(["gs_to_eval"]).head(1).set_index("gs_to_eval")


# In[9]:


df_train_level = df_train_ag[df_train_ag.level!=-1]
df_train_level, outliers = remove_outliers(df_train_level, "level")
df_train_level = df_train_level.groupby(["gs_to_eval", "game_session"]).tail(1)        .groupby(["gs_to_eval"]).agg(["min", "max", "mean"])
df_train_level.columns = ["_".join(col) for col in df_train_level.columns]
df_train_level = df_train_level.reset_index()
df_train_level["accuracy_group"] = df_train_level.gs_to_eval.map(gs_accgroup_map.accuracy_group.to_dict()).astype("category")

f, ax = plt.subplots(1,2,figsize=(25,5))
sns.boxplot(y="accuracy_group", x="level_min", data=df_train_level, ax=ax[0]).set_title("Distribution of Min Levels")
sns.boxplot(y="accuracy_group", x="level_max", data=df_train_level, ax=ax[1]).set_title("Distribution of Max Levels")

g = sns.catplot(kind="count", x="level_max", data=df_train_level, col="accuracy_group")
g.fig.suptitle("Number of Max Levels per Accuracy Group", y=1.05)
g.set_xticklabels(rotation=45)


# In[10]:


g = sns.FacetGrid(df_train_level, col="accuracy_group")
g.map(sns.distplot, "level_mean", rug=True)
g.fig.suptitle("Distribution of Mean Levels per Accuracy Group")
g.fig.set_figwidth(25)
g.fig.set_figheight(5)


# In[11]:


feedback_duration = df_specs[df_specs.feedback_media_playback_duration].event_id.tolist()
df_media_playback = df_train_ag[df_train_ag.event_id.isin(feedback_duration)]
df_media_playback["is_interrupted"] = df_media_playback.total_duration == -1
df_media_playback.loc[df_media_playback.is_interrupted, "total_duration"] = df_media_playback[df_media_playback.is_interrupted].duration
df_media_playback.loc[df_media_playback.is_interrupted, "duration"] = -1
df_media_playback = df_media_playback[df_media_playback.total_duration!=-1].drop(columns=["duration"])
df_media_playback["media_type"] = df_media_playback["media_type"].astype("str").astype("category")

num_isinterrupted = df_media_playback.groupby(["gs_to_eval"]).mean().is_interrupted
df_media_playback = df_media_playback.groupby(["gs_to_eval", "title", "media_type", "game_session"], observed=True).sum().reset_index()
df_media_playback = df_media_playback.groupby(["gs_to_eval", "title", "media_type"], observed=True).mean().reset_index()


# In[12]:


df_media_playback_, outlier = remove_outliers(df_media_playback, "total_duration", extreme=3.0)
f = plt.figure(figsize=(20,5))
f.suptitle("Number of Total Interruptions vs. Total Duration of Media Playback (ms) ")
sns.regplot(x="total_duration", y="is_interrupted", data=df_media_playback_)

g = sns.catplot(kind="count", x="title", col="media_type", data=df_media_playback)
g.fig.suptitle("Number of Media Playback per Title", y=1.05)
g.fig.set_figwidth(20)
g.set_xticklabels(rotation=90)


# In[13]:


df_media_playback = df_media_playback.groupby(["gs_to_eval"], observed=True).sum()
df_media_playback = df_media_playback.drop(columns=["is_interrupted"]).join(num_isinterrupted).reset_index()
df_media_playback["accuracy_group"] = df_media_playback.gs_to_eval.map(gs_accgroup_map.accuracy_group.to_dict()).astype("category")

df_media_playback, outlier = remove_outliers(df_media_playback, "total_duration")
f, ax = plt.subplots(1,2,figsize=(20,5))
sns.boxplot(y="total_duration", x="accuracy_group", data=df_media_playback, ax=ax[0]).set_title("Distribution of Total Duration of Media Playback (ms)")
sns.pointplot(y="total_duration", x="accuracy_group", data=df_media_playback, ax=ax[1]).set_title("Mean of Total Duration of Media Playback (ms)")

df_media_playback, outlier = remove_outliers(df_media_playback, "is_interrupted")
f, ax = plt.subplots(1,2,figsize=(20,5))
sns.boxplot(y="is_interrupted", x="accuracy_group", data=df_media_playback, ax=ax[0]).set_title("Distribution of Interruption Ratio")
sns.pointplot(y="is_interrupted", x="accuracy_group", data=df_media_playback, ax=ax[1]).set_title("Mean of Interruption Ratio")


# In[14]:


g = sns.FacetGrid(df_media_playback, hue="accuracy_group", legend_out=False)
g.map(sns.kdeplot, "total_duration").add_legend()
g.fig.suptitle("Distribution of Total Duration of Media Playback (ms)")
g.fig.set_figwidth(25)
g.fig.set_figheight(5)


# In[15]:


df_misses = df_train_ag[df_train_ag.misses!=-1]
df_misses = df_misses.sort_values(["gs_to_eval", "game_session"])

# Computes the mean of misses across all rounds in a 'gs_to_eval', then sums up those means
# then divides that summation of means to the maximum (or last round) ever reached by a 'gs_to_eval'
avg_miss_per_round = df_misses.groupby(["gs_to_eval", "round"], observed=True).mean()[["misses"]].reset_index()
total_avg_miss_per_round = avg_miss_per_round.groupby(["gs_to_eval"]).sum()[["misses"]]
max_round_per_gs = avg_miss_per_round.groupby(["gs_to_eval"]).tail(1).set_index(["gs_to_eval"])[["round"]]
miss_rate_per_max_round = total_avg_miss_per_round.join(max_round_per_gs)
miss_rate_per_max_round["miss_rate_round"] = miss_rate_per_max_round["misses"] / miss_rate_per_max_round["round"]
miss_rate_per_max_round = miss_rate_per_max_round.drop(columns=["misses", "round"])

# Computes the sum of all misses in a 'game_session', then divides that sum to the maximum (or last round) reached in that same 'game_session'
# then computes the mean of across those quotients computed in each 'game_session' in a 'gs_to_eval'
total_miss_per_gs = df_misses.groupby(["gs_to_eval", "game_session"], observed=True).sum()[["misses"]]
max_round_per_local_gs = df_misses.groupby(["gs_to_eval", "game_session"], observed=True).tail(1).set_index(["gs_to_eval", "game_session"])[["round"]]
avg_miss_per_local_gs_round = total_miss_per_gs.join(max_round_per_local_gs)
avg_miss_per_local_gs_round["miss_rate_gs"] = avg_miss_per_local_gs_round["misses"] / avg_miss_per_local_gs_round["round"]
avg_miss_per_local_gs_round = avg_miss_per_local_gs_round.groupby(["gs_to_eval"]).mean().drop(columns=["misses", "round"])


# In[16]:


df_misses = avg_miss_per_local_gs_round.join(miss_rate_per_max_round).reset_index()
df_misses["accuracy_group"] = df_misses.gs_to_eval.map(gs_accgroup_map.accuracy_group.to_dict()).astype("category")
df_misses_, outlier = remove_outliers(df_misses, "miss_rate_round", extreme=3.0)
g = sns.relplot(x="miss_rate_round", y="miss_rate_gs", data=df_misses_)
g.fig.suptitle("Average Miss Rate per Game Session vs Average Miss Rate per Round", y=1.05)
g.fig.set_figwidth(20)


# In[17]:


df_misses, outlier = remove_outliers(df_misses, "miss_rate_gs", extreme=3.0)
f, ax = plt.subplots(1,2,figsize=(20,5))
sns.kdeplot(data=df_misses.query("accuracy_group==0")["miss_rate_gs"], ax=ax[0], shade=True, label="Accuracy Group 0").set_title("Distribution of Average Miss Rate per Game Session")
sns.kdeplot(data=df_misses.query("accuracy_group==1")["miss_rate_gs"], ax=ax[0], label="Accuracy Group 1")
sns.kdeplot(data=df_misses.query("accuracy_group==2")["miss_rate_gs"], ax=ax[0], label="Accuracy Group 2")
sns.kdeplot(data=df_misses.query("accuracy_group==3")["miss_rate_gs"], ax=ax[0], shade=True, label="Accuracy Group 3")
sns.pointplot(x="accuracy_group", y="miss_rate_gs", data=df_misses, ax=ax[1]).set_title("Mean of Average Miss Rate per Game Session")

df_misses, outlier = remove_outliers(df_misses, "miss_rate_round", extreme=3.0)
f, ax = plt.subplots(1,2,figsize=(20,5))
sns.kdeplot(data=df_misses.query("accuracy_group==0")["miss_rate_round"], ax=ax[0], shade=True, label="Accuracy Group 0").set_title("Distribution of Average Miss Rate per Round")
sns.kdeplot(data=df_misses.query("accuracy_group==1")["miss_rate_round"], ax=ax[0], label="Accuracy Group 1")
sns.kdeplot(data=df_misses.query("accuracy_group==2")["miss_rate_round"], ax=ax[0], label="Accuracy Group 2")
sns.kdeplot(data=df_misses.query("accuracy_group==3")["miss_rate_round"], ax=ax[0], shade=True, label="Accuracy Group 3")
sns.pointplot(x="accuracy_group", y="miss_rate_gs", data=df_misses, ax=ax[1]).set_title("Mean of Average Miss Rate per Round")


# In[18]:


df_misses = df_train_ag[df_train_ag.misses!=-1]
df_misses = df_misses.groupby(["gs_to_eval"], observed=True).agg(["sum", "mean", "min", "max"])[["round", "duration", "misses"]]
df_misses.columns = ['_'.join(col) for col in df_misses.columns]
df_misses = df_misses.reset_index()
df_misses["accuracy_group"] = df_misses.gs_to_eval.map(gs_accgroup_map.accuracy_group.to_dict()).astype("category")

df_misses, outlier = remove_outliers(df_misses, "misses_sum", extreme=3.0)
df_misses, outlier = remove_outliers(df_misses, "misses_mean", extreme=3.0)
# df_misses, outlier = remove_outliers(df_misses, "misses_min")
df_misses, outlier = remove_outliers(df_misses, "misses_max", extreme=3.0)

f, ax = plt.subplots(1,4,figsize=(25,5))
f.suptitle("Mean of Misses (Sum, Mean, Min, Max) per Accuracy Group")
sns.pointplot(y="misses_sum", x="accuracy_group", data=df_misses, ax=ax[0])
sns.pointplot(y="misses_mean", x="accuracy_group", data=df_misses, ax=ax[1])
sns.pointplot(y="misses_max", x="accuracy_group", data=df_misses, ax=ax[2])
sns.pointplot(y="misses_min", x="accuracy_group", data=df_misses, ax=ax[3])

f, ax = plt.subplots(2,2, figsize=(20,10))
f.suptitle("Distribution of Misses (Sum, Mean, Min, Max) per Accuracy Group")
sns.boxenplot(x="misses_sum", y="accuracy_group", ax=ax[0, 0], data=df_misses)
sns.boxenplot(x="misses_mean", y="accuracy_group", ax=ax[0, 1], data=df_misses)
sns.boxenplot(x="misses_min", y="accuracy_group", ax=ax[1, 0], data=df_misses)
sns.boxenplot(x="misses_max", y="accuracy_group", ax=ax[1, 1], data=df_misses)


# In[19]:


df_train_dwell = df_train_ag[df_train_ag.dwell_time!=-1]
df_train_dwell["title"] = df_train_dwell["title"].astype("str").astype("category")

df_train_dwell, _ = remove_outliers(df_train_dwell, "dwell_time", extreme=3.0)
f = plt.figure(figsize=(20, 5))
f.suptitle("Average Dwell Time per Round (ms)")
sns.lineplot(x="round", y="dwell_time", ci="sd", hue="accuracy_group", data=df_train_dwell.query("type=='Game'"), style="accuracy_group", markers=True, dashes=False)
sns.scatterplot(x="round", y="dwell_time", hue="accuracy_group", data=df_train_dwell.query("type=='Game'"))


# In[20]:


df_train_dwell, _ = remove_outliers(df_train_dwell, "game_time", extreme=3.0)
f, ax = plt.subplots(1,2,figsize=(25, 5))
sns.scatterplot(x="game_time", y="dwell_time", hue="accuracy_group", data=df_train_dwell, s=100, style="type", alpha=0.75, edgecolor="black", ax=ax[0]).set_title("Dwell Time (ms) vs Game Time (ms)")
sns.countplot(y="accuracy_group", hue="type", data=df_train_dwell, ax=ax[1]).set_title("Number of Titles per Type")


# In[21]:


df_train_dwell = df_train_dwell.groupby(["gs_to_eval"], observed=True).agg(["min", "max", "sum", "mean"])
df_train_dwell.columns = ["_".join(col) for col in df_train_dwell.columns]
df_train_dwell = df_train_dwell.reset_index()

df_train_dwell["accuracy_group"] = df_train_dwell.gs_to_eval.map(gs_accgroup_map.accuracy_group.to_dict()).astype("category")

df_train_dwell, outlier = remove_outliers(df_train_dwell, "dwell_time_sum")
df_train_dwell, outlier = remove_outliers(df_train_dwell, "dwell_time_mean")
df_train_dwell, outlier = remove_outliers(df_train_dwell, "dwell_time_min")
df_train_dwell, outlier = remove_outliers(df_train_dwell, "dwell_time_max")

f, ax = plt.subplots(1,4,figsize=(25,5))
f.suptitle("Mean of Dwell Time (Sum, Min, Max, Mean) per Accuracy Group (ms)")
sns.pointplot(x="accuracy_group",y="dwell_time_sum", data=df_train_dwell, ax=ax[0])
sns.pointplot(x="accuracy_group",y="dwell_time_min", data=df_train_dwell, ax=ax[1])
sns.pointplot(x="accuracy_group",y="dwell_time_max", data=df_train_dwell, ax=ax[2])
sns.pointplot(x="accuracy_group",y="dwell_time_mean", data=df_train_dwell, ax=ax[3])

f, ax = plt.subplots(1,4,figsize=(25,5))
f.suptitle("Distribution of Dwell Time (Sum, Min, Max, Mean) per Accuracy Group (ms)")
sns.boxplot(y="accuracy_group",x="dwell_time_sum", data=df_train_dwell, ax=ax[0])
sns.boxplot(y="accuracy_group",x="dwell_time_min", data=df_train_dwell, ax=ax[1])
sns.boxplot(y="accuracy_group",x="dwell_time_max", data=df_train_dwell, ax=ax[2])
sns.boxplot(y="accuracy_group",x="dwell_time_mean", data=df_train_dwell, ax=ax[3])


# In[22]:


df_correct = df_train_ag[df_train_ag.correct!=-1]
df_correct_round = df_correct.groupby(["gs_to_eval", "round"]).mean().reset_index()
df_correct_round["accuracy_group"] =  df_correct_round.gs_to_eval.map(gs_accgroup_map.accuracy_group.to_dict()).astype("category")

f = plt.figure(figsize=(25, 7))
sns.lineplot(x="round", y="correct", hue="accuracy_group", data=df_correct_round, ci=None, markers=True, dashes=False).set_title("Average Correct Ratio per Round")
g = sns.relplot(kind="line", x="round", y="correct", row="accuracy_group", data=df_correct_round, ci="sd", markers=True, dashes=False)
g.fig.suptitle("Average Correct Ratio per Round per Accuracy Group", y=1.05)
g.fig.set_figwidth(25)


# In[23]:


max_rounds = df_correct.groupby(["gs_to_eval", "title", "game_session"]).tail(1)
max_rounds = max_rounds.groupby(["gs_to_eval"]).mean()["round"]
df_correct = df_correct.groupby(["gs_to_eval"]).mean().drop(columns=["round"])
df_correct = df_correct.join(max_rounds).reset_index()
df_correct["correct_ratio_round"] = df_correct["round"] * df_correct["correct"]

df_correct["accuracy_group"] =  df_correct.gs_to_eval.map(gs_accgroup_map.accuracy_group.to_dict()).astype("category")

df_correct, _ = remove_outliers(df_correct, "round")
f, ax = plt.subplots(1,2,figsize=(20, 5))
f.suptitle("Distribution of Correct Ratio")
sns.kdeplot(data=df_correct.query("accuracy_group==0").correct, ax=ax[0], label="Accuracy Group 0")
sns.kdeplot(data=df_correct.query("accuracy_group==1").correct, ax=ax[0], label="Accuracy Group 1")
sns.kdeplot(data=df_correct.query("accuracy_group==2").correct, ax=ax[0], label="Accuracy Group 2")
sns.kdeplot(data=df_correct.query("accuracy_group==3").correct, ax=ax[0], label="Accuracy Group 3")
sns.boxplot(x="correct", y="accuracy_group", data=df_correct, ax=ax[1])

df_correct, _ = remove_outliers(df_correct, "correct_ratio_round")
f, ax = plt.subplots(2,2,figsize=(20, 12))
sns.boxplot(x="round", y="accuracy_group", data=df_correct, ax=ax[0, 0]).set_title("Distribution of Round per Accuracy Group")
sns.pointplot(x="round", y="accuracy_group", data=df_correct, ax=ax[0, 1]).set_title("Mean of Round per Accuracy Group")
sns.boxplot(x="correct_ratio_round", y="accuracy_group", data=df_correct, ax=ax[1, 0]).set_title("Distribution of Correct Ratio per Round")
sns.pointplot(x="correct_ratio_round", y="accuracy_group", data=df_correct, ax=ax[1, 1]).set_title("Mean of Correct Ratio per Round")


# In[24]:


df_media = pd.read_parquet("../input/2019-data-science-bowl-phase-i-computation-only/df_media.parquet")
clip_duration_map = df_media.set_index("title").duration.to_dict()

df_train.loc[df_train["type"]=='Clip', "duration"] = df_train[df_train["type"]=='Clip'].title.map(clip_duration_map)
df_train["prev_ts"] = df_train.timestamp.shift(-1)
df_train["clip_runtime"] = (df_train.prev_ts - df_train.timestamp).dt.total_seconds()

completed_clips = (df_train["type"]=="Clip") & ((df_train.clip_runtime > df_train.duration) | (df_train.clip_runtime <= 0))
df_train.loc[completed_clips, "clip_runtime"] =  df_train[completed_clips].duration
df_train["is_completed"] =  False
df_train.loc[completed_clips, "is_completed"] =  True

df_clip = df_train[df_train["type"]=='Clip']

num_completed = df_clip.groupby(["gs_to_eval"]).mean().is_completed
df_clip = df_clip.groupby(["gs_to_eval", "title"], observed=True).mean()
df_clip = df_clip.groupby(["gs_to_eval"]).sum()
df_clip = df_clip.drop(columns=["is_completed"]).join(num_completed).reset_index()
df_clip["accuracy_group"] =  df_clip.gs_to_eval.map(gs_accgroup_map.accuracy_group.to_dict()).astype("category")

f, ax = plt.subplots(1,2,figsize=(20, 5))
df_clip, _ = remove_outliers(df_clip, "clip_runtime")
sns.boxplot(x="clip_runtime", y="accuracy_group", data=df_clip, ax=ax[0]).set_title("Distribution of Clip Runtime per Accuracy Group")
sns.pointplot(x="clip_runtime", y="accuracy_group", data=df_clip, ax=ax[1]).set_title("Mean of Clip Runtime per Accuracy Group")

g = sns.FacetGrid(df_clip, hue="accuracy_group", legend_out=False)
g.map(sns.kdeplot, "is_completed").add_legend()
g.fig.suptitle("Distribution of Clip Completeness Ratio per Accuracy Group")
g.fig.set_figwidth(20)
g.fig.set_figheight(5)


# In[25]:


event_code_map = df_train.groupby(["event_code"]).head(1).reset_index().drop(columns=["level_0"]).reset_index().set_index("event_code")["level_0"].to_dict()
df_train["event_code_encoded"] = df_train.event_code.map(event_code_map)
df_train_eventcode = df_train.groupby(["gs_to_eval", "event_code"], observed=True).head(1).reset_index()
df_train_eventcode["accuracy_group"] =  df_train_eventcode.gs_to_eval.map(gs_accgroup_map.accuracy_group.to_dict()).astype("category")
df_train_eventcode["event_code"] = df_train_eventcode["event_code"].astype('str').astype("category")

g = sns.catplot(kind="count", y="event_code", data=df_train_eventcode, col="accuracy_group", col_wrap=2)
g.fig.suptitle("Number of Event Code per Accuracy Group", y=1.05)
g.fig.set_figwidth(20)


# In[26]:


df_eventcode_minmax = df_train_eventcode.groupby(["accuracy_group", "event_code"], observed=True).count().reset_index()
event_code_trend = list()

for ec in df_train.event_code.unique().tolist():
    count_trend = df_eventcode_minmax[df_eventcode_minmax.event_code==str(ec)].set_index("accuracy_group").event_count.sort_index()
    count_min = count_trend.min()
    count_max = count_trend.max()
    count_list = count_trend.tolist()
    
    if ((count_list[0] == count_min and count_list[-1] == count_max) or (count_list[0] == count_max and count_list[-1] == count_min)) and count_max-count_min >= 200:
        print(ec, count_list)
        event_code_trend.append(str(ec))


# In[27]:


df_train_eventcode_ = df_train_eventcode[df_train_eventcode.event_code.isin(event_code_trend)]
df_train_eventcode_["event_code"] = df_train_eventcode_["event_code"].astype("str").astype("category")
g = sns.FacetGrid(df_train_eventcode_, col="event_code", hue="accuracy_group", col_wrap=4)
g.map(sns.countplot, "accuracy_group")
g.fig.suptitle("Number of Event Codes per Accuracy Group", y=1.05)
g.fig.set_figwidth(20)


# In[28]:


g = sns.FacetGrid(df_train_eventcode, hue="accuracy_group", legend_out=False)
g.map(sns.kdeplot, "event_code_encoded").add_legend()
g.fig.suptitle("Distribution of present Event Codes per Accuracy Group")
g.fig.set_figwidth(25)
g.fig.set_figheight(5)

g = sns.FacetGrid(df_train, hue="accuracy_group", legend_out=False)
g.map(sns.kdeplot, "event_code_encoded").add_legend()
g.fig.suptitle("Distribution of all Event Codes per Accuracy Group")
g.fig.set_figwidth(25)
g.fig.set_figheight(5)


# In[29]:


eventid_map = df_specs.reset_index().set_index("event_id")["index"].to_dict()
df_train["event_id_encoded"] = df_train.event_id.map(eventid_map)
df_train_eventid = df_train.groupby(["gs_to_eval", "event_id"]).head(1)

g = sns.FacetGrid(df_train_eventid, hue="accuracy_group", legend_out=False)
g.map(sns.kdeplot, "event_id_encoded").add_legend()
g.fig.suptitle("Distribution of present Event IDs per Accuracy Group")
g.fig.set_figwidth(25)
g.fig.set_figheight(7)

g = sns.FacetGrid(df_train, hue="accuracy_group", legend_out=False)
g.map(sns.kdeplot, "event_id_encoded").add_legend()
g.fig.suptitle("Distribution of all Event IDs per Accuracy Group")
g.fig.set_figwidth(25)
g.fig.set_figheight(7)


# In[30]:


type_map = df_media.groupby(["type"]).head(1).reset_index().reset_index().set_index("type")["level_0"].to_dict()
title_map = df_media.reset_index().set_index("title")["index"].to_dict()
df_train["type_encoded"] = df_train["type"].map(type_map)
df_train["title_encoded"] = df_train.title.map(title_map)

g = sns.FacetGrid(df_train, hue="accuracy_group", legend_out=False)
g.map(sns.kdeplot, "type_encoded").add_legend()
g.fig.suptitle("Distribution of all Title Types per Accuracy Group")
g.fig.set_figwidth(20)
g.fig.set_figheight(7)

g = sns.FacetGrid(df_train, hue="accuracy_group", legend_out=False)
g.map(sns.kdeplot, "title_encoded").add_legend()
g.fig.suptitle("Distribution of all Titles per Accuracy Group")
g.fig.set_figwidth(25)
g.fig.set_figheight(7)


# In[31]:


df_train["is_difficult"] = False
event_id_difficult = df_specs[df_specs.is_difficult].event_id.tolist()
df_train.loc[df_train.event_id.isin(event_id_difficult), "is_difficult"] = True

df_difficult = df_train.groupby(["gs_to_eval"]).mean()[["is_difficult"]].reset_index()
df_difficult["accuracy_group"] =  df_difficult.gs_to_eval.map(gs_accgroup_map.accuracy_group.to_dict()).astype("category")

df_difficult, _ = remove_outliers(df_difficult, "is_difficult")
f, ax = plt.subplots(1,2,figsize=(20,5))
sns.pointplot(x="accuracy_group", y="is_difficult", data=df_difficult, ax=ax[0])
sns.boxplot(x="accuracy_group", y="is_difficult", data=df_difficult, ax=ax[1])


# In[32]:


g = sns.FacetGrid(df_difficult, hue="accuracy_group")
g.map(sns.kdeplot, "is_difficult")
g.fig.set_figheight(7)
g.fig.set_figwidth(25)

