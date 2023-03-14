#!/usr/bin/env python
# coding: utf-8



from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print(tf.__version__)




import sys, os
import pandas as pd
import numpy as np

rename_col_dict = {
    'area1': 'zona_urbana',
    'area2': 'zona_rural',
    'v2a1': 'monthly_rent',
    'lugar1': 'region_central',
    'lugar2': 'region_chorotega',
    'lugar3': 'region_pacifico_central',
    'lugar4': 'region_brunca',
    'lugar5': 'region_huetar_atlantica',
    'lugar6': 'region_huetar_norte'}


def get_filter_by_row(input, columns):
    filter_data = {}
    for col in columns:
        if "columns" in col:
            for df_column in col["columns"]:
                if input[df_column] == 1:
                    filter_data[df_column] = 1
        elif "name" in col:
            filter_data[col["name"]] = input[col["name"]]
    return filter_data


techo_columns = ['techozinc', 'techoentrepiso', 'techocane', 'techootro']

techo_variables_check = [
    {"columns": ["lugar1", "lugar2", "lugar3", "lugar4", "lugar5", "lugar6"]},
    {"columns": ["area1", "area2"]},
    {"columns": ["paredblolad", "paredzocalo", "paredpreb", "pareddes", "paredmad", "paredzinc",
                 "paredfibras", "paredother"]},
    {"columns": ["pisomoscer", "pisocemento", "pisoother", "pisonatur", "pisonotiene",
                 "pisomadera"]},
]


def techo_mode(input, house_parent, list_houses_techo_issue):
    if input.idhogar not in list_houses_techo_issue:
        return input
    filter_data = {}
    for n in range(len(techo_variables_check)):
        filter_data = None
        if n == 0:
            filter_data = get_filter_by_row(input, techo_variables_check)
        elif n > 0:
            filter_data = get_filter_by_row(input, techo_variables_check[:-n])
        filtered_ds = house_parent.loc[
            (house_parent[list(filter_data)] == pd.Series(filter_data)).all(axis=1)]
        if filtered_ds.shape[0] > 0:
            input[filtered_ds[techo_columns].sum().idxmax()] = 1
            return input


def fix_techo(ds):
    idhogar_list = pd.Series(ds['idhogar'].unique())
    techo_ds = ds[['idhogar', 'techozinc', 'techoentrepiso', 'techocane', 'techootro']]
    id = idhogar_list.apply(
        lambda x: x if techo_ds[techo_ds.idhogar == x].all().value_counts()[True] != 2 else None)
    print("Cantidad de familias sin caracteristicas comunes: ", len(id[id.notnull()]))
    list_houses_techo_issue = list(id[id.notnull()])
    house_parent = ds[(ds.parentesco1 == 1) & ~(ds.idhogar.isin(list_houses_techo_issue))]
    return ds.apply(lambda x: techo_mode(x, house_parent, list_houses_techo_issue), axis=1)


electricity_variables = [
    {"columns": ["lugar1", "lugar2", "lugar3", "lugar4", "lugar5", "lugar6"]},
    {"columns": ["area1", "area2"]},
    {"columns": ["energcocinar1", "energcocinar2", "energcocinar3", "energcocinar4"]},
    {"name": "cielorazo"},
    {"columns": ["eviv1", "eviv2", "eviv3"]},
    {"columns": ["etecho1", "etecho2", "etecho3"]},
    {"columns": ["epared1", "epared2", "epared3"]},

]

electricity_columns = ["public", "planpri", "noelec", "coopele"]


def electricity_mode(input, house_parent, list_houses_issue):
    if input.idhogar not in list_houses_issue:
        return input
    for n in range(len(electricity_variables)):
        filter_data = {}
        if n == 0:
            filter_data = get_filter_by_row(input, electricity_variables)
        elif n > 0:
            filter_data = get_filter_by_row(input, electricity_variables[:-n])
        filtered_ds = house_parent.loc[
            (house_parent[list(filter_data)] == pd.Series(filter_data)).all(axis=1)]
        if filtered_ds.shape[0] > 0:
            input[filtered_ds[electricity_columns].sum().idxmax()] = 1
            return input


def fix_electricity(ds):
    idhogar_list = pd.Series(ds['idhogar'].unique())
    elect_ds = ds[['idhogar', "public", "planpri", "noelec", "coopele"]]
    id = idhogar_list.apply(
        lambda x: x if elect_ds[elect_ds.idhogar == x].all().value_counts()[True] != 2 else None)

    print("Cantidad de familias sin caracteristicas comunes: ", len(id[id.notnull()]))
    list_houses_issue = list(id[id.notnull()])
    house_parent = ds[(ds.parentesco1 == 1) & ~(ds.idhogar.isin(list_houses_issue))]
    return ds.apply(lambda x: electricity_mode(x, house_parent, list_houses_issue), axis=1)


def fix_v18q1(ds):
    ds.loc[ds.v18q1.isna(), 'v18q1'] = 0
    return ds


costo_oportunidad_check_columns = [
    {"columns": ["region_central", "region_chorotega", "region_pacifico_central", "region_brunca",
                 "region_huetar_atlantica", "region_huetar_norte"]},
    {"columns": ["zona_urbana", "zona_rural"]},
    {"name": "cielorazo"},
    {"columns": ["eviv1", "eviv2", "eviv3"]},
    {"name": "rooms"},
    {"columns": ["etecho1", "etecho2", "etecho3"]},
    {"columns": ["epared1", "epared2", "epared3"]},
    {"columns": ["paredblolad", "paredzocalo", "paredpreb", "pareddes", "paredmad", "paredzinc",
                 "paredfibras", "paredother"]},
    {"columns": ["pisomoscer", "pisocemento", "pisoother", "pisonatur", "pisonotiene",
                 "pisomadera"]},
    {"columns": ["techozinc", "techoentrepiso", "techocane", "techootro"]},
]


def get_costo_de_oportunidad(input, ds_paid_rent):
    if input.monthly_rent > 0:
        return input
    for n in range(len(costo_oportunidad_check_columns)):
        filter_data = {}
        if n == 0:
            filter_data = get_filter_by_row(input, costo_oportunidad_check_columns)
        elif n > 0:
            filter_data = get_filter_by_row(input, costo_oportunidad_check_columns[:-n])
        filtered_ds = ds_paid_rent.loc[
            (ds_paid_rent[list(filter_data)] == pd.Series(filter_data)).all(axis=1)]
        if filtered_ds.shape[0] > 0:
            input["monthly_rent"] = filtered_ds.monthly_rent.mean()
            return input


def get_educacion_jefe(x, _ds):
    try:
        x['edu_jefe'] = (_ds.loc[(_ds['parentesco1'] == 1) & (
                    _ds['idhogar'] == x['idhogar']), 'escolari'].item()) ** 2
    except ValueError:
        x['edu_jefe'] = x.escolari
    return x


def add_synthetic_features(_ds):
    _ds['tech_individuo'] = (_ds['mobilephone'] + _ds['v18q']) ** 2
    _ds['tech_hogar'] = (_ds['television'] + _ds['qmobilephone'] +
                         _ds['computer'] + _ds['v18q1']) ** 2
    _ds['monthly_rent_log'] = np.log(_ds['monthly_rent'])
    _ds = _ds.apply(lambda x: get_educacion_jefe(x, _ds), axis=1)
    _ds['bedrooms_to_rooms'] = _ds['bedrooms'] / _ds['rooms']
    _ds['rent_to_rooms'] = _ds['monthly_rent'] / _ds['rooms']
    _ds['SQBage'] = _ds['age'] ** 2
    _ds['SQBhogar_total'] = (_ds['hogar_nin'] + _ds['hogar_mayor'] + _ds['hogar_adul']) ** 2
    _ds['child_dependency'] = _ds['hogar_nin'] / (
                _ds['hogar_nin'] + _ds['hogar_mayor'] + _ds['hogar_adul'])
    _ds['rooms_per_person'] = (_ds['hogar_nin'] + _ds['hogar_mayor'] + _ds['hogar_adul']) / (
    _ds['rooms'])
    _ds['female_weight'] = ((_ds['r4m1'] + _ds['r4m2']) / _ds['tamhog']) ** 2
    _ds['male_weight'] = ((_ds['r4h1'] + _ds['r4h2']) / _ds['tamhog']) ** 2
    
    _ds["no_work"] = _ds["hogar_nin"] + _ds["hogar_mayor"]
    
    _ds["work"] = 0
    _ds["overcrowding"] = 0
    _ds["edjefe"] = 0
    _ds["edjefa"] = 0
    _ds["dependency"] = 0
    for hogar in _ds.idhogar.unique():
        _ds.loc[(_ds.idhogar==hogar), "overcrowding" ] = _ds[ (_ds.idhogar==hogar)].shape[0] / _ds[ (_ds.idhogar==hogar)].iloc[0].bedrooms
        _ds.loc[(_ds.idhogar==hogar), "work" ] = _ds[ (_ds.idhogar==hogar) & (_ds.age<65) & (_ds.age > 18) ].shape[0]
        if _ds[ (_ds.idhogar==hogar)].iloc[0].work > 0:
            _ds.loc[(_ds.idhogar==hogar), "dependency" ] = _ds[ (_ds.idhogar==hogar)].shape[0] / _ds[ (_ds.idhogar==hogar)].iloc[0].work
        else:
            _ds.loc[(_ds.idhogar==hogar), "dependency" ] = 10
        try:
            _ds.loc[(_ds.idhogar==hogar), "edjefe" ] = _ds[ (_ds.idhogar==hogar) & (_ds.parentesco1==1)  & (_ds.male==1) ].iloc[0].age
        except:
            pass
        try:
            _ds.loc[(_ds.idhogar==hogar), "edjefa" ] = _ds[ (_ds.idhogar==hogar) & (_ds.parentesco1==1)  & (_ds.female==1) ].iloc[0].age
        except:
            pass   
    return _ds


def clean(ds, drop_hogares_miss=True):
    # Step 1.1
    _calc_feat = ds.loc[:, 'SQBescolari':'agesq'].columns
    print('Columnas eliminadas: ', _calc_feat.values)
    ds.drop(columns=_calc_feat, inplace=True)
    ds.drop(columns=["edjefe", "edjefa", "dependency", "meaneduc", "elimbasu5"], inplace=True)
    print(
        "Columnas eliminadas: edjefe, edjefa, dependency, meaneduc, rez_esc, hhsize, r4t1, r4t2, r4t3,r4m3, r4h3, hogar_total, elimbasu5")
    # Step 2.2
    ds.drop(columns=["rez_esc"], inplace=True)

    # Step 2.5
    ds.drop(columns=["hhsize", 'r4t1', 'r4t2', 'r4t3', 'r4m3', 'r4h3', "hogar_total"], inplace=True)

    ds = fix_techo(ds)
    ds = fix_electricity(ds)
    ds = fix_v18q1(ds)

    if drop_hogares_miss:
        hogares = ds[["parentesco1", "idhogar"]].groupby(['idhogar']).sum()
        array_hogares = hogares[hogares.parentesco1 != 1].index.values
        ds = ds[ds.idhogar.isin(list(array_hogares)) == False]

        # Step 2.6
        v2a1_max = ds.v2a1.std() * 3 + ds.v2a1.mean()
        ds = ds[(ds.v2a1 < v2a1_max) | (ds.v2a1.isnull())]

    # Step 3.3
    rename_col_dict = {
        'area1': 'zona_urbana',
        'area2': 'zona_rural',
        'v2a1': 'monthly_rent',
        'lugar1': 'region_central',
        'lugar2': 'region_chorotega',
        'lugar3': 'region_pacifico_central',
        'lugar4': 'region_brunca',
        'lugar5': 'region_huetar_atlantica',
        'lugar6': 'region_huetar_norte'}
    ds.rename(columns=rename_col_dict, inplace=True)

    # Step 5
    ds_paid_rent = ds[(ds.monthly_rent > 0) & (ds.parentesco1 == 1)]
    ds = ds.apply(lambda x: get_costo_de_oportunidad(x, ds_paid_rent), axis=1)

    # step 6 add new feautures
    ds = add_synthetic_features(ds)

    cat = len(ds.select_dtypes(include=['object']).columns)
    num = len(ds.select_dtypes(include=['int64', 'float64']).columns)
    print('Total Features: ', cat, 'objetos', '+',
          num, 'numerical', '=', cat + num, 'features')

    return ds




df = pd.read_csv('../input/costa-rican-household-poverty-prediction/train.csv')




df = clean(df)




df_test = pd.read_csv('../input/costa-rican-household-poverty-prediction/test.csv')
df_test = clean(df_test, False)




#df_test.head()
df_test[df_test.idhogar == "d14b3e03a"]




df.drop(columns=["Id","idhogar"], inplace=True)
df_test.drop(columns=["idhogar"], inplace=True) # "Id"




train_dataset = df.sample(frac=0.8,random_state=0)
test_dataset = df.drop(train_dataset.index)




train_labels = tf.keras.utils.to_categorical(train_dataset.pop('Target'))
test_labels = tf.keras.utils.to_categorical(test_dataset.pop('Target'))




train_stats = train_dataset.describe(include="all")
train_stats = train_stats.transpose()




def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
normed_submit_data = norm(df_test.drop(columns=["Id"]))




def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(normed_train_data.keys())]),
    layers.Dropout(.25),
    layers.Dense(64, activation='relu'),
    layers.Dropout(.25),
    layers.Dense(64, activation='relu'),
    layers.Dense(5, activation='softmax')
  ])
  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
  #optimizer = tf.keras.optimizers.SGD(lr=0.001)
  model.compile(optimizer=optimizer, 
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
  return model
model = build_model()




model.summary()




early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
history = model.fit(normed_train_data, train_labels, shuffle=True, epochs=600, validation_data=(normed_test_data, test_labels), verbose=1, callbacks=[early_stop,])




def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['loss'],
           label='Loss')
  plt.plot(hist['epoch'], hist['accuracy'],
           label = 'Accuracy')

  plt.plot(hist['epoch'], hist['val_loss'],
           label='Validation Loss')
  plt.plot(hist['epoch'], hist['val_accuracy'],
           label = 'Validation Accuracy')
  plt.legend()

plot_history(history)




evaluation = model.evaluate(normed_test_data, test_labels, verbose=2)

print(evaluation)




test_predictions = model.predict_classes(normed_submit_data).flatten()
test_predictions




predictions = model.predict_classes(normed_submit_data,verbose=0)
predictions=predictions.flatten()
results = pd.Series(predictions,name="Target")
submission = pd.concat([pd.Series(df_test.Id,name = "Id"),results],axis = 1)
submission.to_csv("submission.csv",index=False)

