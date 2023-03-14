#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py     # 绘图的函数


import plotly.graph_objs as go              # 可用于绘制不同图型，如 go.bar()
import plotly.express as px                 # 可用于绘制不同图型，如 px.bar()
from plotly.subplots import make_subplots   # 创建子图
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 
#NOTEBOOK WHILE KERNEL IS RUNNING


# In[2]:


train = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv')
test = pd.read_csv('../input/porto-seguro-safe-driver-prediction/test.csv')


# In[3]:


train.head()


# In[4]:


# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")

        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[5]:


train_copy = train
train_copy = train_copy.replace(-1, np.NaN)


# In[6]:


# Missing values statistics
missing_values = missing_values_table(train_copy)  # train_copy 是一个 dataframe
missing_values.head(20)


# In[7]:


# train_counts = train.target.value_counts()
# train_counts = pd.DataFrame(train_counts)

# fig = px.bar(train_counts,x=train_counts.index,y='target',barmode='group',color='target')
# fig.update_traces(textposition='outside')
# fig.update_layout(template='seaborn',title='target (counts)')
# fig.show()


# In[8]:


train_counts = train.target.value_counts()
train_counts = pd.DataFrame(train_counts)

fig = px.bar(train_counts,x=train_counts.index,y='target',barmode='group',color='target',text='target') # text 可以标上数值
fig.update_traces(textposition='outside')
fig.update_layout(yaxis_title='counts',xaxis_title='target',template='seaborn',title='target (counts)')
fig.show()


# In[9]:


bin_col = [col for col in train.columns if '_bin' in col]
zero_list = []
one_list = []
for col in bin_col:
    zero_list.append((train[col]==0).sum())
    one_list.append((train[col]==1).sum())


# In[10]:


trace1 = go.Bar(
    x=bin_col,
    y=zero_list ,
    name='Zero count'
)
trace2 = go.Bar(
    x=bin_col,
    y=one_list,
    name='One count'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title='Count of 1 and 0 in binary variables'
)

fig = go.Figure(data=data, layout=layout)
fig.show()


# In[11]:


from sklearn.model_selection import train_test_split

import lightgbm as lgb
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix
import itertools

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score


# In[12]:


train_target = train['target']
# train_feature = train.drop(columns='target')
train_feature = train.drop(columns = ['target','id'])


x_train,x_test,y_train,y_test = train_test_split(train_feature,train_target,test_size= 0.2,random_state=10)


# In[13]:


# model = XGBClassifier(n_estimators=1000)
model = XGBClassifier()


model.fit(x_train,y_train)
y_pred = model.predict(x_test)

acc = accuracy_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)

print('Accuracy: {:.3f}'.format(acc* 100.0))
print('recall: {:.3f}'.format(recall* 100.0))


# In[14]:


def plot_confusion_matrix(cm, classes,
                                normalize=False,
                                title='Confusion matrix',
                                cmap=plt.cm.Blues):
        """
            此函数打印并绘制混淆矩阵。
            可以通过设置“ normalize = True”来应用归一化。
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


# In[15]:


classes = ['target_0','target_1'] # 顺序别搞错
np.set_printoptions(precision=2)

cm = confusion_matrix(y_test,y_pred)
plt.figure()
plot_confusion_matrix(cm,classes)
plt.show()

# plt.figure()
# plot_confusion_matrix(cm,classes,normalize=True)
# plt.show()


# In[16]:


from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


# In[17]:


# 定义绘图函数
def plot_2d_space(X, y, label='Classes'):   
            colors = ['#1F77B4', '#FF7F0E']
            markers = ['o', 's']
            for l, c, m in zip(np.unique(y), colors, markers):
                plt.scatter(
                    X[y==l, 0],
                    X[y==l, 1],
                    c=c, label=l, marker=m
                )
            plt.title(label)
            plt.legend(loc='upper right')
            plt.show()
            
print("label0: ",len(x_train[y_train==0]))
print("label1: ",len(x_train[y_train==1]))

ss = StandardScaler()
X = ss.fit_transform(x_train)

# `2、`如果数据存在多维特征可使用PCA来降维，使其能在2D图中展示

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X = pca.fit_transform(X)

plot_2d_space(X, y_train, 'Imbalanced dataset (2 PCA components)')


# In[18]:


oversampler=SMOTE(random_state=0)
# 开始人工合成数据
os_features,os_labels=oversampler.fit_sample(x_train,y_train)

# 查看生成结果
print("label1: ",len(os_labels[os_labels==1]))
print("label0: ",len(os_labels[os_labels==0]))


# In[19]:


oversampler=SMOTE(random_state=0)
# 开始人工合成数据
os_features_test,os_labels_test=oversampler.fit_sample(x_test,y_test)

# 查看生成结果
print("label1: ",len(os_labels_test[os_labels_test==1]))
print("label0: ",len(os_labels_test[os_labels_test==0]))


# In[20]:


ss = StandardScaler()
X = ss.fit_transform(os_features)

pca = PCA(n_components=2)
X = pca.fit_transform(X)

plot_2d_space(X, os_labels, 'Imbalanced dataset (2 PCA components)')


# In[21]:


new_os_features = os_features.copy()
new_os_features['target'] = os_labels

# Find correlations with the target and sort
corrs = new_os_features.corr()['target'].sort_values(ascending=False)
correlations = pd.DataFrame(corrs)

# Display correlations
print('Most Positive Correlations:\n')
correlations.head()


# In[22]:


print('Most Negative Correlations:\n')
correlations.tail()


# In[23]:


correlations.loc[correlations.index.isin(['ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin'])]


# In[24]:


np.abs(corrs).sort_values(ascending=False).tail(15)


# In[25]:


corrs = os_features.corr()


# In[26]:


# 设置阈值
threshold = 0.8

# 创建一个空字典以容纳相关变量
above_threshold_vars = {}

# 对于每一列，记录index行中的那个值高于阈值的变量
for col in corrs:
    above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])


# In[27]:


# above_threshold_vars


# In[28]:


# 跟踪要删除的列和已检查的列
cols_to_remove = []
cols_seen = []
cols_to_remove_pair = []

# 遍历列和相关列
for key, value in above_threshold_vars.items():
    # 跟踪已检查的列
    cols_seen.append(key)
    for x in value:
        if x == key:
            next
        else:
            # 如果存在高相关的特征，只保留一个
            if x not in cols_seen:                  # 如果该特征在之前的 key 数据中没有出现过。
                cols_to_remove.append(x)            # 存在高度相关，将高度相关的特征放入 cols_to_remove 中。
                cols_to_remove_pair.append(key)     # cols_to_remove 和 cols_to_remove_pair 得到的结果一致。

cols_to_remove = list(set(cols_to_remove))
print('Name of columns to remove: ', cols_to_remove)
print('Number of columns to remove: ', len(cols_to_remove))


# In[29]:


train_corrs_removed = os_features.drop(columns = cols_to_remove)
test_corrs_removed = os_features_test.drop(columns = cols_to_remove)

print('Training Corrs Removed Shape: ', train_corrs_removed.shape)
print('Testing Corrs Removed Shape: ', test_corrs_removed.shape)


# In[30]:


train_corrs_removed.to_csv('train_corrs_removed.csv', index = False)
test_corrs_removed.to_csv('test_corrs_removed.csv', index = False)


# In[31]:


# `1、定义基尼系数：`

def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

# `2、定义 xgb gini 系数：`

# 返回一个 normalized 后的 gini 分数
def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred) / gini(y, y)


# In[32]:


from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb


# In[33]:


params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 
          'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}

nrounds=200  
kfold = 2
skf = StratifiedKFold(n_splits=kfold, random_state=0)

for i, (train_index, test_index) in enumerate(skf.split(train_feature, train_target)):
    print(' xgb kfold: {}  of  {} : '.format(i+1, kfold))
    X_train, X_valid = train_feature.loc[train_index], train_feature.loc[test_index]
    y_train, y_valid = train_target.loc[train_index], train_target.loc[test_index]
    d_train = xgb.DMatrix(X_train, y_train) 
    d_valid = xgb.DMatrix(X_valid, y_valid) 
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    xgb_model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=100, 
                          feval=gini_xgb, maximize=True, verbose_eval=100)
    
# xgb_y_pred = xgb_model.predict(xgb.DMatrix(test[features].values), 
#                         ntree_limit=xgb_model.best_ntree_limit+50)


# In[34]:


features = train_feature.columns
test_ids = test['id']


# In[35]:


xgb_test_predictions1 = xgb_model.predict(xgb.DMatrix(test[features]),ntree_limit=xgb_model.best_ntree_limit+50)

submission = pd.DataFrame({'id': test_ids, 'target': xgb_test_predictions1})
submission.to_csv('before_oversampling_xgb.csv', index = False, float_format='%.5f')


# In[36]:


def model_feature_importances(model):
    trace = go.Scatter(
        y = np.array(list(model.get_fscore().values())),
        x = np.array(list(model.get_fscore().keys())),
        mode='markers',
        marker=dict(
            sizemode = 'diameter',
            sizeref = 1,
            size = 13,
            #size= model.feature_importances_,
            #color = np.random.randn(500), #set color equal to a variable
            color =  np.array(list(model.get_fscore().values())),
            colorscale='Portland',
            showscale=True
        ),
        text = np.array(list(model.get_fscore().keys()))
    )
    data = [trace]

    layout= go.Layout(
        autosize= True,
        title= 'xgb Feature Importance',
        hovermode= 'closest',
         xaxis= dict(
             ticklen= 5,
             showgrid=False,
            zeroline=False,
            showline=False
         ),
        yaxis=dict(
            title= 'Feature Importance',
            showgrid=False,
            zeroline=False,
            ticklen= 5,
            gridwidth= 2
        ),
        showlegend= False
    )
    fig = go.Figure(data=data, layout=layout)
    fig.show()


# In[37]:


model_feature_importances(xgb_model)


# In[38]:


xgb_importance = np.array(list(xgb_model.get_fscore().values()))
xgb_features = np.array(list(xgb_model.get_fscore().keys()))

x, y = (list(x) for x in zip(*sorted(zip(xgb_importance,xgb_features), reverse = False)))
trace2 = go.Bar(
    x=x ,
    y=y,
    marker=dict(
        color=x,
        colorscale = 'Viridis',
        reversescale = True
    ),
    name='Random Forest Feature importance',
    orientation='h',
)

layout = dict(
    title='Barplot of Feature importances',
    width = 900, height = 2000,
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
#         domain=[0, 0.85],
    ))

fig1 = go.Figure(data=[trace2])
fig1['layout'].update(layout)
py.iplot(fig1, filename='plots')


# In[39]:


new_features = pd.DataFrame(xgb_model.get_fscore(),index=['features_importance']).T                        .sort_values(by='features_importance',ascending=False)                        .iloc[0:38]
new_features 


# In[40]:


train_feature.shape


# In[41]:


new_train_feature = train_feature[new_features.index]
new_train_feature.head()


# In[42]:


new_train_feature.shape


# In[43]:


params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 
          'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}

nrounds=200  
kfold = 2
skf = StratifiedKFold(n_splits=kfold, random_state=0)

for i, (train_index, test_index) in enumerate(skf.split(new_train_feature, train_target)):
    print(' xgb kfold: {}  of  {} : '.format(i+1, kfold))
    X_train, X_valid = new_train_feature.loc[train_index], new_train_feature.loc[test_index]
    y_train, y_valid = train_target.loc[train_index], train_target.loc[test_index]
    d_train = xgb.DMatrix(X_train, y_train) 
    d_valid = xgb.DMatrix(X_valid, y_valid) 
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    xgb_model2 = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=100, 
                          feval=gini_xgb, maximize=True, verbose_eval=100)


# In[44]:


xgb_test_predictions2 = xgb_model2.predict(xgb.DMatrix(test[new_features.index]),ntree_limit=xgb_model2.best_ntree_limit+50)

submission = pd.DataFrame({'id': test_ids, 'target': xgb_test_predictions2})
submission.to_csv('before_oversampling_after_feature_choose_xgb.csv', index = False, float_format='%.5f')


# In[45]:


def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True


# In[46]:


# https://www.kaggle.com/rshally/porto-xgb-lgb-kfold-lb-0-282

# xgb
# params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 
#         'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}

# submission=test['id'].to_frame()
# submission['target']=0

# nrounds=200  # need to change to 2000
# kfold = 2  # need to change to 5
# skf = StratifiedKFold(n_splits=kfold, random_state=0)
# for i, (train_index, test_index) in enumerate(skf.split(new_train_feature, train_target)):
#     print(' xgb kfold: {}  of  {} : '.format(i+1, kfold))
#     X_train, X_valid = new_train_feature.loc[train_index], new_train_feature.loc[test_index]
#     y_train, y_valid = train_target.loc[train_index], train_target.loc[test_index]
#     d_train = xgb.DMatrix(X_train, y_train) 
#     d_valid = xgb.DMatrix(X_valid, y_valid) 
#     watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#     xgb_model3 = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=100, 
#                         feval=gini_xgb, maximize=True, verbose_eval=100)

#     # 结尾 除以 (2*kfold)，是因为要将 xgb 和 lgb 去平均然后将结果相加合并
#     submission['target'] += xgb_model3.predict(xgb.DMatrix(test[new_features.index]), 
#                         ntree_limit=xgb_model3.best_ntree_limit+50) / (2*kfold)
    
# submission.head(2)

# # lgb
# params = {'metric': 'auc', 'learning_rate' : 0.01, 'max_depth':10, 'max_bin':10,  'objective': 'binary', 
#         'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':10,  'min_data': 500}

# skf = StratifiedKFold(n_splits=kfold, random_state=1)
# for i, (train_index, test_index) in enumerate(skf.split(new_train_feature, os_labels)):
#     print(' lgb kfold: {}  of  {} : '.format(i+1, kfold))
#     X_train, X_eval = new_train_feature.loc[train_index], new_train_feature.loc[test_index]
#     y_train, y_eval = train_target.loc[train_index], train_target.loc[test_index]
#     lgb_model = lgb.train(params, lgb.Dataset(X_train, label=y_train), nrounds, 
#                 lgb.Dataset(X_eval, label=y_eval), verbose_eval=100, 
#                 feval=gini_lgb, early_stopping_rounds=100)

#     # 结尾 除以 (2*kfold)，是因为要将 xgb 和 lgb 去平均然后将结果相加合并
#     submission['target'] += lgb_model.predict(test[new_features.index], 
#                         num_iteration=lgb_model.best_iteration) / (2*kfold)

# submission.to_csv('before_oversampling_lgb+xgb.csv', index=False, float_format='%.5f') 

# submission.head(2)


# In[47]:


# submission.to_csv('before_oversampling_after_feature_choose_xgb.csv', index = False, float_format='%.5f')


# In[48]:


# params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 
#           'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}

# nrounds=200  
# kfold = 2  
# skf = StratifiedKFold(n_splits=kfold, random_state=0)

# for i, (train_index, test_index) in enumerate(skf.split(os_features, os_labels)):
#     print(' xgb kfold: {}  of  {} : '.format(i+1, kfold))
#     X_train, X_valid = os_features.loc[train_index], os_features.loc[test_index]
#     y_train, y_valid = os_labels.loc[train_index], os_labels.loc[test_index]
#     d_train = xgb.DMatrix(X_train, y_train) 
#     d_valid = xgb.DMatrix(X_valid, y_valid) 
#     watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#     xgb_model2 = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=100, 
#                           feval=gini_xgb, maximize=True, verbose_eval=100)


# In[49]:


# xgb_test_predictions = xgb_model2.predict(xgb.DMatrix(test[features]), 
#                         ntree_limit=xgb_model2.best_ntree_limit+50)


# In[50]:


# submission = pd.DataFrame({'id': test_ids, 'target': xgb_test_predictions})
# submission.to_csv('after_oversampling_xgb.csv', index = False, float_format='%.5f')


# In[51]:


# from sklearn.ensemble import RandomForestClassifier


# In[52]:


# os_features.drop(['id'],axis=1,inplace=True)
# os_features_test.drop(['id'],axis=1,inplace=True)


# rdf_clf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
# rdf_clf.fit(os_features, os_labels)
# features = os_features.columns.values
# print("----- Training Done -----")


# In[53]:


# RandomForestClassifier feature importances Scatter plot 

# def rdf_feature_importances(model):
#     trace = go.Scatter(
#         y = model.feature_importances_,
#         x = features,
#         mode='markers',
#         marker=dict(
#             sizemode = 'diameter',
#             sizeref = 1,
#             size = 13,
#             #size= model.feature_importances_,
#             #color = np.random.randn(500), #set color equal to a variable
#             color = model.feature_importances_,
#             colorscale='Portland',
#             showscale=True
#         ),
#         text = features
#     )
#     data = [trace]

#     layout= go.Layout(
#         autosize= True,
#         title= 'Random Forest Feature Importance',
#         hovermode= 'closest',
#          xaxis= dict(
#              ticklen= 5,
#              showgrid=False,
#             zeroline=False,
#             showline=False
#          ),
#         yaxis=dict(
#             title= 'Feature Importance',
#             showgrid=False,
#             zeroline=False,
#             ticklen= 5,
#             gridwidth= 2
#         ),
#         showlegend= False
#     )
#     fig = go.Figure(data=data, layout=layout)
#     fig.show()

# rdf_feature_importances(model=rdf_clf)


# In[54]:


# x, y = (list(x) for x in zip(*sorted(zip(rdf_clf.feature_importances_, features), 
#                                                             reverse = False)))
# trace2 = go.Bar(
#     x=x ,
#     y=y,
#     marker=dict(
#         color=x,
#         colorscale = 'Viridis',
#         reversescale = True
#     ),
#     name='Random Forest Feature importance',
#     orientation='h',
# )

# layout = dict(
#     title='Barplot of Feature importances',
#      width = 900, height = 2000,
#     yaxis=dict(
#         showgrid=False,
#         showline=False,
#         showticklabels=True,
# #         domain=[0, 0.85],
#     ))

# fig1 = go.Figure(data=[trace2])
# fig1['layout'].update(layout)
# py.iplot(fig1, filename='plots')


# In[55]:


# y_pred = rdf_clf.predict(os_features_test)

# acc = accuracy_score(os_labels_test,y_pred)
# recall = recall_score(os_labels_test,y_pred)

# print('Accuracy: {:.3f}'.format(acc* 100.0))
# print('recall: {:.3f}'.format(recall* 100.0))


# In[56]:


# model = XGBClassifier()


# model.fit(os_features,os_labels)
# y_pred = model.predict(os_features_test)

# acc = accuracy_score(os_labels_test,y_pred)
# recall = recall_score(os_labels_test,y_pred)

# print('Accuracy: {:.3f}'.format(acc* 100.0))
# print('recall: {:.3f}'.format(recall* 100.0))


# In[ ]:





# In[57]:


# test_predictions = rdf_clf.predict(new_test)

# submission = pd.DataFrame({'id': test_ids, 'target': test_predictions})
# submission.to_csv('RandomForestClassifier_predict_1.csv', index = False)


# In[58]:


# submission.head()


# In[ ]:




