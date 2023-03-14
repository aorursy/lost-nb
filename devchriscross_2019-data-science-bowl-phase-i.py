#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !cat /proc/meminfo # Inquire system memory
# !df -h / | awk '{print $4}' # Inquire system disk
# !lscpu  # Inquire system processor


# In[2]:


import numpy as np 
import pandas as pd 
import gc
import ast

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

get_ipython().system('pip install --upgrade orjson')
import orjson

import os
import sys
def print_log(string):
    os.system(f'echo \"{string}\"')

import dask
import dask.dataframe as dd
from dask.distributed import Client, progress, wait
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
import numexpr 
numexpr.set_num_threads(1)


# In[3]:


def load_dataset():
    data_dtypes = {"type": "category", "title": "category", "event_data": "str", "event_id": "str", "installation_id": "str", "event_code": "category",
                    "world": "category", "str": "category","event_count": "uint8", "game_time": "uint32"}

    print_log("Reading training data ...")
    df_train = dd.read_csv("../input/data-science-bowl-2019/train.csv", parse_dates=["timestamp"], dtype=data_dtypes)
    
    media_dtypes = {"title": "category","type": "category", "duration": "float64"}
    print_log("Reading media sequence ...")
    df_media = pd.read_csv("../input/data-science-bowl-2019-media-sequence/media_sequence.csv", dtype=media_dtypes)

    print_log("Reading specifications for event ids ...")
    df_specs = pd.read_csv("../input/data-science-bowl-2019/specs.csv")
    
    return df_train, df_media, df_specs


# In[4]:


df_train, df_media, df_specs = load_dataset()
df_media.duration = df_media.duration.fillna(0).astype('float32')
df_titles = df_media.title.unique().tolist()
assessment_attempt_event_code_query = "((event_code=='4100' & title!='Bird Measurer (Assessment)') | (event_code=='4110' & title=='Bird Measurer (Assessment)'))"
attempts_query = "type=='Assessment' & " +  assessment_attempt_event_code_query
print_log("Finised loading datasets")


# In[5]:


df_train.head()


# In[6]:


df_specs.head()


# In[7]:


df_specs.query("event_id=='2b9272f4'")["info"].values


# In[8]:


df_specs.query("event_id=='2b9272f4'")["args"].values


# In[9]:


df_specs.describe()


# In[10]:


print_log("Encoding args column ...")
args_to_int_map = {args: i for i, args in enumerate(df_specs["args"].unique())}
df_specs["args_encoded"] = df_specs["args"].map(args_to_int_map)

eventid_to_argsint_map = df_specs.set_index("event_id")["args_encoded"].to_dict()
df_train = df_train.assign(**{"args_encoded": df_train["event_id"].map(eventid_to_argsint_map)})
df_train.head()


# In[11]:


def generate_argstype_map(argstype_encoded, use_client=False):
    unique_args = df_specs[argstype_encoded].unique()
    dask_with_args = [df_train[df_train[argstype_encoded]==i].title.unique() for i in range(len(unique_args))]
    if use_client:
        dask_with_args = client.compute(dask_with_args)
        progress(dask_with_args, notebook=False)
        titles_with_args = client.gather(dask_with_args)
    else:
        titles_with_args = dask.compute(dask_with_args)[0]
    titles_with_args_i = {i: titles_with_args[i].tolist() for i in range(len(unique_args))}
    return titles_with_args_i


# In[12]:


def generate_titleargs_table(titles_with_args):
    # Create a dataframe with all the titles and the corresponding args an event id might represent
    # The dataframe is binary, 1 if present, 0 otherwise
    args_length = len(titles_with_args)

    title_args_table = list()
    for title in df_titles:
        title_args_row = [0.0 for i in range(args_length)]
        title_args_table.append(title_args_row)

    args_encoded_col = list(range(args_length))
    title_args_table = pd.DataFrame(title_args_table, 
                                    index=df_titles,
                                    columns=args_encoded_col)\
        .reset_index().rename(columns={"index":"title"}).set_index("title")

    # Change the value to 1 if a given args can be represented by a given title
    for i in range(args_length):
        titles_with_args_i = titles_with_args[i]
        title_args_table.loc[titles_with_args_i, i] = 1

    # Add the type (e.g Game) of each title and group them accordingly
    title_args_table = title_args_table.join(df_media.set_index("title")["type"])            .sort_values(["type", "title"]).reset_index()
    title_args_table["title"] = title_args_table["title"].str.cat(title_args_table["type"], sep=" ")
    title_args_table = title_args_table.set_index("title").drop(columns=["type"])
    
    return title_args_table


# In[13]:


def clear_figure(f):
    f.clf()
    plt.clf()
    plt.close(f)
    gc.collect()


# In[14]:


args_encoded_map = generate_argstype_map("args_encoded")
title_args_table = generate_titleargs_table(args_encoded_map)

f = plt.figure(figsize=(20,20))
sns.heatmap(title_args_table, linewidths=.25).set_title("Encoded Args per Title")
f.savefig("encoded_args_vs_title.png")


# In[15]:


clear_figure(f)
f = plt.figure(figsize=(20,20))
sns.heatmap(title_args_table[~title_args_table.index.str.contains("Clip")], linewidths=.25).set_title("Encoded Args per Title (excluding Clip Titles)")
f.savefig("encoded_args_vs_title2.png")


# In[16]:


clear_figure(f)
print_log("Extracting and encoding different args type ...")
int_to_args_map = dict()
type_list = ["int", "string", "object", "array"]
for _type in type_list:
    int_to_args_map[_type] = dict()

# Extract numeric attributes of a given args
for i, args in enumerate(df_specs["args"].unique()):
    args_json = orjson.loads(args)
    args_list = dict()
    for _type in type_list:
        args_list[_type] = list()
    
    for args in args_json:
        if (args["name"] != "event_code" and args["name"] != "game_time" and args["name"] != "event_count"):
            # Include numeric types and the "correct" attribute and don't consider those that are already extracted (e.g game_time)
            if (args["type"] == "int" or args["type"] == "number" or args["type"] == "boolean"):  
                args_list["int"].append(args["name"])
            elif ("array" in args["type"]):
                args_list["array"].append(args["name"])
            else:
                args_list[args["type"]].append(args["name"])
    
    for _type in type_list:
        int_to_args_map[_type][i] = args_list[_type]

# Encode corresponding numeric_args
for _type in type_list:
    args_name = "_".join(["args", _type])
    args_encode_name = "".join(["args", _type]) + "_encoded"
    
    args_to_encode_map = {args: int_to_args_map[_type][i] for i, args in enumerate(df_specs["args"].unique())}
    df_specs[args_name] = df_specs["args"].map(args_to_encode_map).astype(str)

    argstype_to_int_map = {args: i for i, args in enumerate(df_specs[args_name].unique())}
    df_specs[args_encode_name] = df_specs[args_name].map(argstype_to_int_map)

    # Set the encoding in the dataset
    eventid_to_args_map = df_specs.set_index("event_id")[args_encode_name].to_dict()
    df_train = df_train.assign(**{args_encode_name: df_train["event_id"].map(eventid_to_args_map)})

df_train.head()


# In[17]:


print_log("Plotting heatmap for encoded args of type int ...")
argsint_encoded_map = generate_argstype_map("argsint_encoded")
title_args_table = generate_titleargs_table(argsint_encoded_map)

# plot heatmap
f = plt.figure(figsize=(20,10))
sns.heatmap(title_args_table[~title_args_table.index.str.contains("Clip")], linewidths=.25).set_title("Int type Encoded Args per Title")
f.savefig("int_encoded_args_vs_title.png")


# In[18]:


clear_figure(f)
print_log("Plotting heatmap title per int args ...")
title_allargs = set()
title_args_map = dict()
title_singleargs_map = dict()

for title in title_args_table.itertuples():
    args_list = list()
    args_set = set()
    title = pd.Series(title)
    
    if "Clip" not in title.get(0) and "Assessment" not in title.get(0):
        for i in title_args_table.columns:
            # add +1 since 0 represents the title of the series
            if title.get(i+1) == 1:
                args = ast.literal_eval(df_specs[df_specs.argsint_encoded==i].head(1).args_int.values[0])
                title_allargs.update(args)
                args_set.update(args)
                args_list.append((i, args))
                
        title_args_map[title.get(0)] = args_list
        title_singleargs_map[title.get(0)] = args_set    


# In[19]:


title_singleargs_table = [[1 if arg in args else 0 for arg in title_allargs] for title, args in title_singleargs_map.items()]
title_singleargs_table = pd.DataFrame(title_singleargs_table, index=title_singleargs_map.keys(), columns=title_allargs)

f = plt.figure(figsize=(20,10))
sns.heatmap(title_singleargs_table, linewidths=.25).set_title("Individual Int Args per Title")
f.savefig("individual_int_args_vs_title.png")


# In[20]:


df_specs.sample(10, random_state=1)["info"].values


# In[21]:


# Looking for phrases that contains "It contains information"
import random
random.sample(set([info[info.find("It contains"):info.find("We")] for info in df_specs["info"].unique().tolist()]), 10)


# In[22]:


# Looking for phrases that contains "players feel are too difficult"
set(info for info in df_specs["info"].unique().tolist() if "difficult" in info)


# In[23]:


# Looking for phrases that contains "We can answer questions like"
set([info[info.find("We can"):] for info in df_specs["info"].unique().tolist()])


# In[24]:


clear_figure(f)
print_log("Classifying event ids based on info column ...")

df_specs["info_str"] = df_specs["info"].map(lambda info: "".join(e for e in info if e.isalnum()))
df_specs["contains_diagnosis"] = df_specs.info_str.str.contains("diagnose")
df_specs["contains_answer"] =  df_specs.info_str.str.contains("answerquestions") 
df_specs["is_difficult"] = df_specs.info_str.str.contains("difficult") 
df_specs["info_type"] = "none"
df_specs.loc[df_specs.contains_diagnosis, "info_type"] = "contains_diagnosis"
df_specs.loc[df_specs.contains_answer, "info_type"] = "contains_answer"
df_specs.loc[df_specs.is_difficult, "info_type"] = "is_difficult"

f, axes = plt.subplots(1, 4, figsize=(20,5))
axes[0].set_title("Event ID count per Info Type")
sns.countplot(y="info_type", data=df_specs, ax=axes[0], order=["contains_answer", "contains_diagnosis", "is_difficult", "none"])
axes[1].set_title("Event ID count per Diagnosis Info Type")
sns.countplot(x="contains_answer", data=df_specs, hue="contains_diagnosis", ax=axes[1])
axes[2].set_title("Event ID count per Difficult Info Type")
sns.countplot(x="contains_diagnosis", data=df_specs, hue="is_difficult", ax=axes[2])
axes[3].set_title("Event ID count per Answer Info Type")
sns.countplot(x="is_difficult", data=df_specs, hue="contains_answer", ax=axes[3])
f.savefig("event_id_count_per_info.png")


# In[25]:


clear_figure(f)
def extract_singleargs_per_infotype(args_type, title_args_table):
    reduced_all_args = dict()
    title_singleargs_map = dict()

    args_name = "_".join(["args", args_type])
    args_encode_name = "".join(["args", args_type]) + "_encoded"
    
    info_types = df_specs.info_type.unique().tolist()
    for info_t in info_types:
        reduced_all_args[info_t] = set()

    for title in title_args_table.itertuples():
        args_list = dict()
        args_set = dict()
        for info_t in info_types:
            args_list[info_t] = list()
            args_set[info_t] = set()

        title = pd.Series(title)
        if "Clip" not in title.get(0) and "Assessment" not in title.get(0):
            for i in title_args_table.columns:
                # add +1 since 0 represents the title of the series
                if title.get(i+1) == 1:
                    args_row = df_specs[(df_specs[args_encode_name]==i)].groupby(["info_type"]).head(1)
                    for args_by_type in args_row.itertuples():
                        args = ast.literal_eval(getattr(args_by_type, args_name))
                        info_type = args_by_type.info_type
                        args_set[info_type].update(args)
                        args_list[info_type].append((i, args))
                        reduced_all_args[info_type].update(args)
            title_singleargs_map[title.get(0)] = args_set 
    return title_singleargs_map, reduced_all_args


# In[26]:


import math
def plot_eventids_per_infotype(title_singleargs_map, reduced_all_args, title):
    info_types = df_specs.info_type.unique().tolist()
    row_plot_count = math.ceil(len(info_types))
    f, axes = plt.subplots(row_plot_count, 1, figsize=(20,15*row_plot_count))
    i = j = 0
    
    for info_t in info_types:
        title_singleargs_table = [[1 if arg in args[info_t] else 0 for arg in reduced_all_args[info_t]] for title, args in title_singleargs_map.items()]
        title_singleargs_table = pd.DataFrame(title_singleargs_table, index=title_singleargs_map.keys(), columns=reduced_all_args[info_t])
        
        
#         i = (i+1) if j==2 else i
#         j = 0 if j==2 else j
        axes[i].title.set_text("Event IDs with " + info_t + " info type")
        if len(reduced_all_args[info_t]) == 0:
            title_singleargs_table = pd.DataFrame(np.zeros((len(title_singleargs_map.keys()), 1)), index=title_singleargs_map.keys(), columns=["none"])
        sns.heatmap(title_singleargs_table, linewidths=.25, ax=axes[i])
        i = i + 1
        
    f.savefig(title)


# In[27]:


print_log("Plotting and processing all args type ...")
plot_eventids_per_infotype(*extract_singleargs_per_infotype("int", title_args_table), "individual_args_vs_title_per_info_type.png")


# In[28]:


df_specs[df_specs.args_int.str.contains("misses")].head()


# In[29]:


df_specs[df_specs.args_int.str.contains("misses")]["info"].unique()


# In[30]:


def find_definition(args_list, info_type, args_type):
    args_list_info = dict()

    for args in args_list:
        args_with_diagnosis_args = df_specs[(df_specs.info_type==info_type) & (df_specs[args_type].str.contains(args))].groupby(["args_encoded"]).head(1)
        args_list_info[args] = set()

        for _args in args_with_diagnosis_args.itertuples():
            for args_comp in orjson.loads(_args.args):
                if args_comp["name"] == args:
                    args_list_info[args].add(args_comp["info"])

    return args_list_info                


# In[31]:


# Context usage for Diagnosis Info Type of event_id(s)
diagnosis_args = ["round", "correct", "dwell_time"]
find_definition(diagnosis_args, "contains_diagnosis", "args_int")              


# In[32]:


# Context usage for Answer Info Type of event_id(s)
answer_args = ["round", "duration", "total_duration"]
find_definition(answer_args, "contains_answer", "args_int")


# In[33]:


# Context usage for Difficult Info Type of event_id(s)
difficult_args = ["round"]
find_definition(difficult_args, "is_difficult", "args_int")


# In[34]:


# Context usage for None Info Type of event_id(s)
none_args = ["round", "misses", "duration"]
find_definition(none_args, "none", "args_int")


# In[35]:


# Created the new Missed Info Type
df_specs["is_missed"] = df_specs.args_int.str.contains("misses")
df_specs.loc[df_specs.args_int.str.contains("misses"), "info_type"] = "is_missed"
missed_args = ["round", "misses", "duration"]
find_definition(missed_args, "is_missed", "args_int")


# In[36]:


df_specs["current_round"] = df_specs.args.str.contains("number of the current round")
df_specs["completed_round"] = df_specs.args.str.contains("number of the round that has just been completed")
df_specs["feedback_media_playback_duration"] = df_specs.args.str.contains("media playback in milliseconds")
df_specs["total_fbmedia_playback_duration"] = df_specs.args.str.contains("media playback in milliseconds (if it ran uninterrupted)")
df_specs["level_duration"] = df_specs.args.str.contains("duration of the level in milliseconds")
df_specs["round_duration"] = df_specs.args.str.contains("duration of the round in milliseconds")
df_specs.head()


# In[37]:


plot_eventids_per_infotype(*extract_singleargs_per_infotype("int", title_args_table), "individual_args_vs_title_per_info_type2.png")


# In[38]:


argsstring_encoded_map = generate_argstype_map("argsstring_encoded")
title_args_table = generate_titleargs_table(argsstring_encoded_map)
plot_eventids_per_infotype(*extract_singleargs_per_infotype("string", title_args_table), "individual_stringargs_vs_title_per_info_type.png")


# In[39]:


answer_args = ["identifier", "media_type", "description"]
find_definition(answer_args, "contains_answer", "args_string")


# In[40]:


find_definition(["object"], "contains_diagnosis", "args_string")


# In[41]:


find_definition(["object"], "none", "args_string")


# In[42]:


df_specs.loc[df_specs.contains_answer & df_specs.args_string.str.contains("media_type"), df_specs.columns.str.contains("args")].head(1).args.values


# In[43]:


df_specs.loc[df_specs.contains_diagnosis & df_specs.args_string.str.contains("object"), df_specs.columns.str.contains("args")].head(1).args.values


# In[44]:


argsobject_encoded_map = generate_argstype_map("argsobject_encoded")
title_args_table = generate_titleargs_table(argsobject_encoded_map)
plot_eventids_per_infotype(*extract_singleargs_per_infotype("object", title_args_table), "individual_objectargs_vs_title_per_info_type.png")


# In[45]:


find_definition(["coordinates"], "contains_diagnosis", "args_object")


# In[46]:


find_definition(["coordinates"], "none", "args_object")


# In[47]:


find_definition(["coordinates"], "is_difficult", "args_object")


# In[48]:


df_specs[df_specs.contains_diagnosis & df_specs.args_object.str.contains("coordinates") & df_specs.args_int.str.contains("correct")].head(1).args.values


# In[49]:


df_specs[df_specs.is_difficult & df_specs.args_object.str.contains("coordinates") & df_specs.args_int.str.contains("round")].head(1).args.values


# In[50]:


argsarray_encoded_map = generate_argstype_map("argsarray_encoded")
title_args_table = generate_titleargs_table(argsarray_encoded_map)
plot_eventids_per_infotype(*extract_singleargs_per_infotype("array", title_args_table), "individual_arrayargs_vs_title_per_info_type.png")


# In[51]:


print_log("Filtering dataset based on selected event ids ...")
diagnosis_args_int = ["round", "correct", "dwell_time"]
answer_args_int = ["round", "duration", "total_duration"]
missed_args_int = ["round", "misses", "duration"]
difficult_args_int = ["round"]

answer_args_string = ["identifier", "media_type", "description"]

diagnosis_filter = df_specs.contains_diagnosis &         (df_specs.args_string.str.contains("object") |          df_specs.args_int.str.contains(diagnosis_args_int[1]) | df_specs.args_int.str.contains(diagnosis_args_int[2]))

answer_filter = df_specs.contains_answer &            (df_specs.args_int.str.contains(answer_args_int[1]) |            df_specs.args_int.str.contains(answer_args_int[2]) | df_specs.args_string.str.contains(answer_args_string[0]) |            df_specs.args_string.str.contains(answer_args_string[1]) | df_specs.args_string.str.contains(answer_args_string[2]))

difficult_filter = df_specs.is_difficult & (df_specs.args_object.str.contains("coordinates"))

missed_filter = df_specs.is_missed &         (df_specs.args_int.str.contains(missed_args_int[1]) |          df_specs.args_int.str.contains(missed_args_int[2]))

df_specs["filtered_id"] = diagnosis_filter | answer_filter | missed_filter
event_id_filter = df_specs[df_specs.filtered_id].event_id.values

df_specs.to_parquet("df_specs.parquet")


# In[52]:


dask.compute(df_train.shape)


# In[53]:


df_train.head()


# In[54]:


print_log("Removing installation ids without attempts ...")
ids = dask.compute(df_train.query(attempts_query).installation_id.unique())[0]  # ids who took assessments
df_train = df_train[df_train.installation_id.isin(ids.tolist())]
df_train = df_train.assign(**{"index":df_train.installation_id.str.cat([
                    df_train.game_session, 
                    df_train.event_count.astype("str"), 
                    df_train.game_time.astype("str"), 
                    df_train.timestamp.dt.strftime("%y%m%d%H%M%S")])})\
                .set_index("index")

df_event_id_filter = df_train[df_train.event_id.isin(event_id_filter) & (df_train["type"]!="Clip") & (df_train["type"]!="Assessment")]
df_event_id_filter.head()


# In[55]:


all_args_int = set()
all_args_string = set()
# all_args_object = dict()

df_event_id_filter_list = dask.compute(df_event_id_filter.event_id.unique())[0].tolist()
selected_event_ids = df_specs[df_specs.event_id.isin(df_event_id_filter_list)]
for event_id in selected_event_ids.itertuples():
    all_args_int.update(ast.literal_eval(event_id.args_int))
    all_args_string.update(ast.literal_eval(event_id.args_string))

#     all_args.update(ast.literal_eval(event_id.args_object))
#     event_args = orjson.loads(event_id.args)
#     for arg_obj in ast.literal_eval(event_id.args_object):
#         for event_arg in event_args:
#             if event_arg["name"] == arg_obj:
#                 event_arg_items = event_arg["info"][event_arg["info"].find("{") : event_arg["info"].find("}")+1]
#                 event_arg_items = re.findall(r"(?!\")\w*(?=\")", event_arg_items)
#                 if len(event_arg_items) > 0:
#                     all_args_object[event_arg["name"]] = event_arg_items
# import pickle
# with open("all_args.pkl", "wb") as write_args:
#     pickle.dump(all_args, write_args)
# all_args


# In[56]:


len(all_args_int) + len(all_args_string)


# In[57]:


filter_columns = [col for col in df_train.columns if "encoded" in col]
df_train_filter = df_train[df_train.event_id.isin(df_event_id_filter_list)]
df_train_filter = df_train_filter.assign(**{args: -1 for args in all_args_int})    .assign(**{args: "" for args in all_args_string})    .drop(columns=["event_id", "installation_id", "event_code", "title", "type", "world", "game_session", "event_count", "game_time"])    .drop(columns=filter_columns)
df_train_filter.head()


# In[58]:


def extract_event(df):
    index_list = df.index.tolist()
    event_object_list = df.event_data.map(lambda x: orjson.loads(x)).tolist()
    event_object_map = dict(zip(index_list, event_object_list))
    for args in all_args_int:
        df[args] = df.index.map(lambda x: event_object_map[x].get(args, -1))
        df[args] = df[args].astype("int") 
    for args in all_args_string:
        df[args] = df.index.map(lambda x: event_object_map[x].get(args, ""))
        df[args] = df[args].astype("category") 
    return df
print("Extracting json args in the dataset ...")
client = Client()
df_train_filter = df_train_filter.map_partitions(extract_event, meta=df_train_filter).drop(columns=["event_data", "timestamp"])
df_train_join = df_train.join(df_train_filter)
dd_train_future = client.persist(df_train_join)
progress(dd_train_future, notebook=False)   
dd_train_future.to_parquet("df_train.parquet")

