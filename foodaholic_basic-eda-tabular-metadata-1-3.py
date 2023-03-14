#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install missingpy')




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import Image as show_gif
from sklearn.cluster import KMeans
from skimage.transform import resize
import scipy.ndimage as ndimage
from plotly.tools import FigureFactory as FF
from skimage import measure, morphology, segmentation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from missingpy import MissForest
import lightgbm as lgb
from scipy import stats
import copy
import pydicom
import glob
import re
import os
import scipy




train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')




train




train.describe()




train[train.duplicated(subset=['Patient','Weeks'], keep=False)]




train = train.groupby(['Patient', 'Weeks']).agg({ 
    'FVC': 'mean', 
    'Percent': 'mean', 
    'Age': 'first',
    'Sex': 'first',
    'SmokingStatus': 'first'
}).reset_index()




print('Number of duplicates: %s'%len(train[train.duplicated(subset=['Patient','Weeks'], keep=False)]))




train.isna().any()




print("Number of unique patients: %s"%(train.Patient.nunique()))




num_obs_per_patient = train.groupby('Patient').count()['Weeks'].sort_values()

fig, axs = plt.subplots(2, 1, figsize=(15, 10))
dense = sns.kdeplot(num_obs_per_patient, bw=.5, ax=axs[0])
dense.get_legend().remove()
dense.set_xlabel("Number of Observations")
bar = sns.barplot(x=list(range(len(num_obs_per_patient))), y=num_obs_per_patient, ax=axs[1])
bar.axes.get_xaxis().set_visible(False);
bar.set_ylabel("Oberservations");




fig, axs = plt.subplots(2, 2, figsize=(15, 10))

axs[0,0].set_title('Weeks', fontsize=18)
sns.kdeplot(train.Weeks, shade=True, ax=axs[0,0])
axs[0,0].get_legend().remove()

axs[0,1].set_title('FVC', fontsize=18)
sns.kdeplot(train.FVC, shade=True, ax=axs[0,1])
axs[0,1].get_legend().remove()

axs[1,0].set_title('Percent', fontsize=18)
sns.kdeplot(train.Percent, shade=True, ax=axs[1,0])
axs[1,0].get_legend().remove()

axs[1,1].set_title('Age', fontsize=18)
sns.kdeplot(train.Age, shade=True, ax=axs[1,1])
axs[1,1].get_legend().remove()




def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)

g = sns.PairGrid(train, palette=["red"])
g.map_upper(plt.scatter, s=10)
g.map_diag(sns.distplot, kde=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_lower(corrfunc)




id_patients_most_weeks = train.groupby('Patient').Weeks.count().sort_values(ascending=False).iloc[:5].index
train_patients = train[train.Patient.isin(id_patients_most_weeks)].sort_values('Weeks')

fig, ax = plt.subplots(figsize=(16,6))

for name, group in train_patients.groupby('Patient'):
    color = next(ax._get_lines.prop_cycler)['color']
    group.sort_values('Weeks').plot(x='Weeks', y='FVC', ax=ax, label=name, color=color)
    reg = LinearRegression().fit(np.array(group.Weeks).reshape(-1, 1), np.array(group.FVC))
    ax.plot(group.Weeks,reg.predict(np.array(group.Weeks).reshape(-1, 1)),'--', color=color)
    
ax.set_title('Progress of patients with the most FVC measurements', fontsize=18)
ax.set_xlabel('Weeks', fontsize=12)
ax.set_ylabel('FVC', fontsize=12);




example_patient_file_path = "../input/osic-pulmonary-fibrosis-progression/train/ID00014637202177757139317/"




num_scans = {}
for patient_id in train.Patient.unique():
    files = glob.glob("../input/osic-pulmonary-fibrosis-progression/train/%s/*.dcm"%patient_id)
    num_scans[patient_id] = len(files)
df_scans = pd.DataFrame.from_dict(num_scans, orient='index', columns=['num_scans'])
df_scans.index = df_scans.index.rename('Patient_id')
df_scans = df_scans.sort_values('num_scans')

fig, axs = plt.subplots(2, 1, figsize=(15, 10))
fig.suptitle('Number of CT-Scans for every Patient id', fontsize=16)
bar = sns.barplot(x=df_scans.index, y=df_scans.num_scans, ax=axs[0])
bar.axes.get_xaxis().set_visible(False)
bar.set_ylabel("Number of CT-Scan Files")

dense = sns.kdeplot(df_scans.num_scans, bw=50, ax=axs[1])
dense.get_legend().remove()
dense.set_xlabel('Numober of CT-Scan Files');




df_scans.num_scans.describe()




#https://github.com/pydicom/pydicom/issues/319
def dictify(ds):
    """Turn a pydicom Dataset into a dict with keys derived from the Element tags.

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        The Dataset to dictify

    Returns
    -------
    output : dict
    """
    output = dict()
    for elem in ds:
        # skip the image data
        if elem.name=='Pixel Data':
            continue
        if elem.VR != 'SQ':
            output[elem.name] = elem.value
        else:
            output[elem.name] = [dictify(item) for item in elem]
    return output




ds1 = dictify(pydicom.filereader.dcmread(example_patient_file_path+"1.dcm"))
ds2 = dictify(pydicom.filereader.dcmread(example_patient_file_path+"2.dcm"))
ds1




def get_diff(ds1, ds2):
    diff_keys = []
    for key in ds1.keys():
        if ds1[key]!=ds2[key]:
            diff_keys.append(key)
    return diff_keys

changing_keys = get_diff(ds1, ds2)
stable_keys = ds1.keys()-changing_keys




stable_keys




additional_metadata = []
for patient_id in tqdm(train.Patient.unique()):
    file = glob.glob("../input/osic-pulmonary-fibrosis-progression/train/%s/*.dcm"%patient_id)[0]
    ds_dict = dictify(pydicom.filereader.dcmread(file))
    metadata = {k:ds_dict[k] for k in stable_keys if k in ds_dict}
    additional_metadata.append(metadata)
    
df_add_meta = pd.DataFrame(additional_metadata)




for column in df_add_meta.columns:
    if df_add_meta[column].dtypes!='float64':
        try:
            print(column, ":", df_add_meta[column].unique()[:5], "-", df_add_meta[column].dtypes)
        except:
            print(column, ":", df_add_meta[column].iloc[:10].values, "-", df_add_meta[column].dtypes)
        print("-"*100)




df_add_meta['Convolution Kernel'].values




column = 'Convolution Kernel'
# strange way to get the single faulty value but it works 
df_add_meta[column][(pd.isna(df_add_meta[column].str.contains('I50f'))).values] ='I50f'




numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
columns_to_keep = ["Patient Position", "Patient ID", "Manufacturer", "Convolution Kernel"]
columns_to_keep = list(df_add_meta.select_dtypes(include=numerics).columns.values) + columns_to_keep
df_add_meta = df_add_meta[columns_to_keep]




perc_missing_cols = (df_add_meta.isna().sum()/len(df_add_meta)).sort_values(ascending=False)
print("Percentage of missing values in each column:")
print("-"*50)
print(perc_missing_cols[perc_missing_cols!=0])




perc_missing_cols = perc_missing_cols[perc_missing_cols<0.1]
df_add_meta = df_add_meta[df_add_meta.columns.intersection(list(perc_missing_cols.index))]




df_train_join = pd.merge(train, df_add_meta, left_on='Patient', right_on="Patient ID", how='inner')




perc_missing_cols = (df_train_join.isna().sum()/len(df_train_join)).sort_values(ascending=False)
print("Percentage of missing values in each column:")
print("-"*50)
print(perc_missing_cols[perc_missing_cols!=0])




imputer = MissForest()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# create a new dataframe containing only numeric values which can be imputed
df_train_join_numeric = df_train_join.select_dtypes(include=numerics)
imputed_matrix = imputer.fit_transform(df_train_join_numeric)
# replace imputed columns in the df_add_meta dataframe
df_train_join_numeric = pd.DataFrame(imputed_matrix, index=df_train_join_numeric.index, columns=df_train_join_numeric.columns)
for column in df_train_join_numeric.columns:
    df_train_join[column] = df_train_join_numeric[column]




for column in df_train_join.select_dtypes(include=numerics).columns:
    if(df_train_join[column].apply(float.is_integer).all()):
        df_train_join[column] = df_train_join[column].astype(int)




def evaluate_features(X_train, y_train, X_test, y_test, params, metrices):
    """
    Trains a simple gradient boosting model and evaluates its feature importances (if multiple columns provided).
    Furthermore the trained model is evaluated with the provided metric(es).
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param params:
    :param metrices:
    :return:
    """

    for col in X_train.select_dtypes(include='object').columns:
        le = LabelEncoder()
        le.fit(list(X_train[col].astype(str).values) + list(X_test[col].astype(str).values))
        X_train[col] = le.transform(list(X_train[col].astype(str).values))
        X_test[col] = le.transform(list(X_test[col].astype(str).values))


    clf = lgb.LGBMRegressor(**params)
    clf.fit(X_train.values, y_train)

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    features_to_show = len(X_train.columns)

    plt.figure(figsize=(15,10))
    plt.title("Feature importances")
    plt.bar(range(features_to_show), importances[indices][:features_to_show],
            color="r", align="center")
    feature_names = [X_train.columns[indices[f]] for f in range(features_to_show)]
    plt.xticks(range(features_to_show), feature_names, rotation='vertical')
    plt.xlim([-1, features_to_show])
    plt.show()

    scores = get_model_scores(clf, X_train, y_train, X_test, y_test, metrices, True)

    df_feature_importance = pd.DataFrame({'column':X_train.columns[indices], 'importance':importances[indices]})
    return (df_feature_importance, scores)

def get_model_scores(model, x_train, y_train, x_test, y_test, metrices, print_values=True):
    scores = {}
    for metric in metrices:
        try:
            score_train = metric(model.predict(x_train), y_train)
            score_test = metric(model.predict(x_test), y_test)
            if print_values:
                print(metric.__name__, "(train):", score_train)
                print(metric.__name__, "(test):", score_test)
                print("------------------------------------------------------------")
            scores[metric.__name__] = [score_train, score_test]
        except:
            print("Could not calculate score", metric.__name__)
            print("------------------------------------------------------------")
            scores[metric.__name__] = [None, None]
    return scores




X_train, X_test, y_train, y_test = train_test_split(df_train_join.drop(columns=['FVC']), df_train_join.FVC)




evaluate_features(X_train, y_train, X_test, y_test, {}, [mean_squared_error])




df_train_join.to_csv('train_merged_and_cleaned.csv', index=False)




def center_crop(img, new_width=512, new_height=512):        

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img

def slices_as_gif(slices, fps=10):
    fig = plt.figure()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    plt.axis('off')
    slices = [[plt.imshow(img, cmap='gray')] for img in slices]
    ani = animation.ArtistAnimation(fig, slices, interval=200, repeat_delay=0)
    ani.save('test_anim.gif', writer='imagemagick', fps=fps)
    plt.close()
    return 'test_anim.gif'
    
def read_dicoms(file, with_mask=False):
    img = pydicom.filereader.dcmread(file).pixel_array
    img = center_crop(img)
    if with_mask:
        img = make_lungmask(img)
    return img




files = glob.glob(example_patient_file_path+"*.dcm")
files = sorted(files, key=lambda x:float(re.findall("(\d+)",x)[-1]))
fig=plt.figure(figsize=(18, 6))
columns = 10
rows = 3
for i in range(30):
    ds = pydicom.filereader.dcmread(files[i])
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(center_crop(ds.pixel_array), cmap='gray')
    plt.title(os.path.basename(files[i]))
    plt.axis('off')
    plt.grid(b=None)




slices = [read_dicoms(file) for file in files]
show_gif(filename=slices_as_gif(slices), format='png', width=512, height=512)






