#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Standard plotly imports
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import iplot, init_notebook_mode
import cufflinks
import cufflinks as cf
import plotly.figure_factory as ff

# Using plotly + cufflinks in offline mode
init_notebook_mode(connected=True)
cufflinks.go_offline(connected=True)

import os
print(os.listdir("../input"))




df_train_id = pd.read_csv("../input/train_identity.csv", nrows=50000)
df_train_trans = pd.read_csv("../input/train_transaction.csv", nrows=50000)
#df_test_id = pd.read_csv("../input/test_identity.csv")
#df_test_trans = pd.read_csv("../input/test_transaction.csv")




def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

def plot_distribution(df, var_select=None, title=None, bins=1.0): 
    # Calculate the correlation coefficient between the new variable and the target
    tmp_fraud = df[df['isFraud'] == 1]
    tmp_no_fraud = df[df['isFraud'] == 0]    
    corr = df['isFraud'].corr(df[var_select])
    corr = np.round(corr,3)
    tmp1 = tmp_fraud[var_select].dropna()
    tmp2 = tmp_no_fraud[var_select].dropna()
    hist_data = [tmp1, tmp2]
    
    group_labels = ['Fraud', 'No Fraud']
    colors = ['seagreen','indianred', ]

    fig = ff.create_distplot(hist_data,
                             group_labels,
                             colors = colors, 
                             show_hist = True,
                             curve_type='kde', 
                             bin_size = bins
                            )
    
    fig['layout'].update(title = title+' '+'(corr target ='+ str(corr)+')')

    iplot(fig, filename = 'Density plot')
    
def plot_dist_churn(df, col, binary=None):
    tmp_churn = df[df[binary] == 1]
    tmp_no_churn = df[df[binary] == 0]
    tmp_attr = round(tmp_churn[col].value_counts().sort_index() / df[col].value_counts().sort_index(),2)*100
    print(f'Distribution of {col}: ')
    trace1 = go.Bar(
        x=tmp_churn[col].value_counts().sort_index().index,
        y=tmp_churn[col].value_counts().sort_index().values, 
        name='Fraud',opacity = 0.8, marker=dict(
            color='seagreen',
            line=dict(color='#000000',width=1)))

    trace2 = go.Bar(
        x=tmp_no_churn[col].value_counts().sort_index().index,
        y=tmp_no_churn[col].value_counts().sort_index().values,
        name='No Fraud', opacity = 0.8, 
        marker=dict(
            color='indianred',
            line=dict(color='#000000',
                      width=1)
        )
    )

    trace3 =  go.Scatter(   
        x=tmp_attr.sort_index().index,
        y=tmp_attr.sort_index().values,
        yaxis = 'y2', 
        name='% Fraud', opacity = 0.6, 
        marker=dict(
            color='black',
            line=dict(color='#000000',
                      width=2 )
        )
    )
    
    layout = dict(title =  f'Distribution of {str(col)} feature by %Fraud',
              xaxis=dict(type='category'), 
              yaxis=dict(title= 'Count'), 
              yaxis2=dict(range= [0, 15], 
                          overlaying= 'y', 
                          anchor= 'x', 
                          side= 'right',
                          zeroline=False,
                          showgrid= False, 
                          title= 'Percentual Fraud Transactions'
                         ))

    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    iplot(fig)
    
## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

## REducing memory
df_train_trans = reduce_mem_usage(df_train_trans)
df_train_id = reduce_mem_usage(df_train_id)




## REducing memory
df_train_trans = reduce_mem_usage(df_train_trans)
df_train_id = reduce_mem_usage(df_train_id)




resumetable(df_train_id)




resumetable(df_train_trans)[:25]




print("Transactions % Fraud:")
print(round(df_train_trans[['isFraud', 'TransactionID']]['isFraud'].value_counts(normalize=True) * 100,2))
# df_train.groupby('Churn')['customerID'].count().iplot(kind='bar', title='Churn (Target) Distribution', 
#                                                      xTitle='Customer Churn?', yTitle='Count')

trace0 = go.Bar(
    x=df_train_trans[['isFraud', 'TransactionID']].groupby('isFraud')['TransactionID'].count().index,
    y=df_train_trans[['isFraud', 'TransactionID']].groupby('isFraud')['TransactionID'].count().values,
    marker=dict(
        color=['indianred', 'seagreen']),
)

data = [trace0] 
layout = go.Layout(
    title='Fraud (Target) Distribution <br>## 0: No Fraud | 1: Is Fraud ##', 
    xaxis=dict(
        title='Transaction is fraud', 
        type='category'),
    yaxis=dict(
        title='Count')
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)









def print_trans(tmp, num_col='TransactionAmt'):
    print(f"The mininum value in Transaction Amount is {tmp[num_col].min()}, median is {round(tmp[num_col].median(),2)}, and the maximum is {df_train_trans[num_col].max()}")
    print(f"The mean Transaction Amount of Fraudulent Transactions is {round(tmp[tmp['isFraud'] == 1][num_col].median(),2)}          \nThe mean Transaction Amount of No-Fraudulent Transactions is {round(tmp[tmp['isFraud'] == 0][num_col].median(),2)}")
    
print_trans(df_train_trans[['isFraud', 'TransactionAmt']], 'TransactionAmt')




print("Transaction Amount Quantiles: ")
print(df_train_trans['TransactionAmt'].quantile([0.01, .025, .1, .25, .5, .75, .975, .99]))




# df_train_trans['TransactionAmt_log'] = df_train_trans['TransactionAmt'].apply(np.log)
tmp = df_train_trans[['TransactionAmt', 'isFraud']]
tmp['TransactionAmt_log'] = tmp['TransactionAmt'].apply(np.log)
## Calling the function
plot_distribution(tmp[(tmp['TransactionAmt'] <= 800)], 'TransactionAmt', 'Transaction Amount Distribution', bins=10.0,)
plot_distribution(tmp[(tmp['TransactionAmt'] <= 800)], 'TransactionAmt_log', 'Transaction Amount Log Distribution', bins=0.1)




plot_dist_churn(df_train_trans[['ProductCD', 'isFraud']], 'ProductCD', 'isFraud')




for col in ['card4', 'card6']:
    df_train_trans[col] = df_train_trans[col].fillna('NoInf')
    plot_dist_churn(df_train_trans, col, 'isFraud')




print("Card Features Quantiles: ")
print(df_train_trans[['card1', 'card2', 'card3', 'card5']].quantile([0.01, .025, .1, .25, .5, .75, .975, .99]))




for col in ['card1', 'card2', 'card3', 'card5']:
    df_train_trans[str(col)+'_log'] = np.log(df_train_trans[col])




## Calling the function
plot_distribution(df_train_trans[['isFraud','card1_log']], 'card1_log', 'Card 1 Feature Log Distribution by Target', bins=0.05,)




## Calling the function
plot_distribution(df_train_trans[['isFraud','card2_log']], 'card2_log', 'Card 2 Feature Log Distribution by Target', bins=0.05)




df_train_trans.loc[df_train_trans.card3.isin(df_train_trans['card3'].value_counts()[df_train_trans['card3'].value_counts() < 10].index), 'card3'] = -99




plot_dist_churn(df_train_trans[['card3', 'isFraud']], 'card3', 'isFraud')




df_train_trans.loc[df_train_trans.card5.isin(df_train_trans['card5']                                             .value_counts()                                             [df_train_trans['card5']                                              .value_counts() < 20]                                             .index), 'card5'] = -99

plot_dist_churn(df_train_trans[['card5', 'isFraud']], 'card5', 'isFraud')




tmp = df_train_trans[['M1','M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'isFraud']]
for col in ['M1','M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']:
    tmp[col] = tmp[col].fillna('NoInf')
    plot_dist_churn(tmp, col, 'isFraud')






