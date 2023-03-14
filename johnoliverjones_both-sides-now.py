#!/usr/bin/env python
# coding: utf-8



import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
import pandas as pd
import os
import lightgbm as lgb
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
DATA_PATH = "../input/liverpool-ion-switching"

train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

#submission_df = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))




x = train_df[['time','signal']]

a1 = 0.99
b0 = 1
b1 = -1
x['f_DC'] = 0
x['f_DC'] = a1 * x['f_DC'].shift(1) + b0 * x['signal'] + b1 * x['signal'].shift(1)
x.loc[0,'f_DC'] = x.loc[0,'signal']




r = 0.995
fs = 10000
f0 = 50

a1 = -2*r*np.cos(2*np.pi*f0/fs)
a2 = r**2
b0 = 1
b1 = -2*np.cos(2*np.pi*f0/fs)
b2 = 1

x['f_50Hz'] = 0
x['f_50Hz'] = a1 * x['f_50Hz'].shift(1) + a2*x['f_50Hz'].shift(2) +              b0 * x['signal'] + b1*x['signal'].shift(1) + b2*x['signal'].shift(2)
x.loc[0:1,'f_50Hz'] = x.loc[0:1,'signal']




examples = ['signal']
bs = 500000
start = 3650000
end = start + 100000
fs=10000.
fig, ax = plt.subplots(nrows=len(examples), ncols=1, figsize=(25, 5*len(examples)))
fig.subplots_adjust(hspace = .5)
#ax = ax.ravel()
colors = plt.rcParams["axes.prop_cycle"]()
print("How is this odd noise composed?")

fft = np.fft.fft(x.loc[start:end,'signal'])
psd = np.abs(fft) ** 2
fftfreq = np.fft.fftfreq(len(psd),1/fs)
    
i = abs(fftfreq) < 1000
ax.grid()
ax.plot(fftfreq[i], 20*np.log10(psd[i]), linewidth=.5)
ax.set_title('Signal - 365-375 seconds')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD (dB)')
ax.set_ylim(100,150)




fs = 10000
f0 = 100

K = np.int(fs/f0)
a1 = -0.99 
x['f_n100Hz'] = 0.
x['f_n100Hz'] = x['signal'] + a1 * x['signal'].shift(K)
x.loc[0:K-1,'f_n100Hz'] = x.loc[0:K-1,'signal']

x['f_2xn100Hz'] = 0.
x['f_2xn100Hz'] = x['f_n100Hz'] + a1 * x['f_n100Hz'].shift(K)
x.loc[0:K-1,'f_2xn100Hz'] = x.loc[0:K-1,'f_n100Hz']

x['f_3xn100Hz'] = 0.
x['f_3xn100Hz'] = x['f_2xn100Hz'] + a1 * x['f_2xn100Hz'].shift(K)
x.loc[0:K-1,'f_3xn100Hz'] = x.loc[0:K-1,'f_2xn100Hz']




examples = ['signal','f_DC','f_50Hz', 'f_n100Hz','f_2xn100Hz', 'f_3xn100Hz']

fig, ax = plt.subplots(nrows=len(examples), ncols=1, figsize=(25, 4*len(examples)))
fig.subplots_adjust(hspace = .5)
ax = ax.ravel()
colors = plt.rcParams["axes.prop_cycle"]()

for i in range(len(examples)):
    
    c = next(colors)["color"]
    ax[i].grid()
    if examples[i] in ['dt_50Hz_energy_floor','50Hz_energy_floor']:
        ax[i].plot(x['time'], x[examples[i]],color=c, linewidth= 2)
        ax[i].set_ylim(0,4)

    ax[i].plot(x['time'], x[examples[i]],color=c, linewidth=.5)
    ax[i].set_title(examples[i], fontsize=24)
    ax[i].set_xlabel('Time (seconds)', fontsize=18)
    #ax[i].set_ylabel('current (pA)', fontsize=24)
    #ax[i].set_ylim(0,5)




def make_features(x,list_f):
    for name in list_f:
    
        x[name + '_power'] = x[name]**2
        
        x[name + "_rel_power_sum"] = np.sqrt(x[name + "_power"]**2 + (x[name + "_power"].mean())**2)
        
        x[name + '_rel_power_XL'] = x[name + '_power'] - x[name + '_power'].mean()
        x[name + '_rel_power_L'] = x[name + '_power'] - x[name + '_power'].rolling(window=7500,min_periods=5).mean()
        x.loc[0:4,name + '_rel_power_L'] = x.loc[0:4,name + '_power']
        x[name + '_rel_power_S'] = x[name + '_power'] - x[name + '_power'].shift(1)
        x[name + '_rel_power_S'][0] = x[name + '_power'][0]
        
    x['energy_floor'] = x['f_50Hz_rel_power_sum'].rolling(window=100, min_periods=1).min()
    
    x[name + '_energy_floor_XL'] = x[name + '_rel_power_XL'].rolling(window=100, min_periods=5).min()
    x.loc[0:4,name + '_energy_floor_XL'] = x.loc[0:4,name + '_rel_power_XL']
    
    x[name + '_energy_floor_S'] = x[name + '_rel_power_S'].rolling(window=100, min_periods=5).min()
    x.loc[0:4,name + '_energy_floor'] = x.loc[0:4,name + '_rel_power_S']
    x[name + '_energy_floor_L'] = x[name + '_rel_power_L'].rolling(window=7500, min_periods=5).min()
    x.loc[0:4,name + '_energy_floor_L'] = x.loc[0:4,name + '_rel_power_L']
    
    return x

x = make_features(x[['signal','f_50Hz', 'f_n100Hz']],
                  list_f = ['signal','f_50Hz', 'f_n100Hz'])




test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
y = test_df[['signal']]

# Perform two filters used in model;
# 50Hz notch filter
r = 0.995
fs = 10000
f0 = 50
a1 = -2*r*np.cos(2*np.pi*f0/fs)
a2 = r**2
b0 = 1
b1 = -2*np.cos(2*np.pi*f0/fs)
b2 = 1
y['f_50Hz'] = 0
y['f_50Hz'] = a1 * y['f_50Hz'].shift(1) + a2*y['f_50Hz'].shift(2) +              b0 * y['signal'] + b1*y['signal'].shift(1) + b2*y['signal'].shift(2)
y.loc[0:1,'f_50Hz'] = y.loc[0:1,'signal']

# 100Hz comb filter
fs = 10000
f0 = 100
K = np.int(fs/f0)
a1 = -0.99 
y['f_n100Hz'] = 0.
y['f_n100Hz'] = y['signal'] + a1 * y['signal'].shift(K)
y.loc[0:K-1,'f_n100Hz'] = y.loc[0:K-1,'signal']

y = make_features(y[['signal','f_50Hz', 'f_n100Hz']],
                  list_f = ['signal','f_50Hz', 'f_n100Hz'])

y = y.replace([np.inf, -np.inf], np.nan)    
y.fillna(0, inplace=True)





# Reference https://www.kaggle.com/teejmahal20/single-model-lgbm-kalman-filter
# simple lgbm with 5 stratified KFold (function has the option for a return for bayesian optimization, just ignore this)
def run_lgb(pre_train, pre_test, features, params):
    
    kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    target = 'open_channels'
    oof_pred = np.zeros(len(pre_train))
    y_pred = np.zeros(len(pre_test))
     
    for fold, (tr_ind, val_ind) in enumerate(kf.split(pre_train, pre_train[target])):
        x_train, x_val = pre_train[features].iloc[tr_ind], pre_train[features].iloc[val_ind]
        y_train, y_val = pre_train[target][tr_ind], pre_train[target][val_ind]
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)
        
        model = lgb.train(params, train_set, num_boost_round = 10000, early_stopping_rounds = 50, 
                         valid_sets = [train_set, val_set], verbose_eval = 500)
        
        oof_pred[val_ind] = model.predict(x_val)
        
        y_pred += model.predict(pre_test[features]) / kf.n_splits
        
    rmse_score = np.sqrt(mean_squared_error(pre_train[target], oof_pred))
    # want to clip and then round predictions (you can get a better performance using optimization to found the best cuts)
    oof_pred = np.round(np.clip(oof_pred, 0, 10)).astype(int)
    round_y_pred = np.round(np.clip(y_pred, 0, 10)).astype(int)
    f1 = f1_score(pre_train[target], oof_pred, average = 'macro')
    print(f'Our oof rmse score is {rmse_score}')
    print(f'Our oof macro f1 score is {f1}')
    
    return round_y_pred, model




# lgb modeling
# reference: https://www.kaggle.com/teejmahal20/single-model-lgbm-kalman-filter
# define hyperparammeter (using bayesian optimization extracted with 151 features)

x.loc[:,'open_channels'] = train_df['open_channels']

features = [col for col in x.columns if col not in ['open_channels', 'time']]

params = {'boosting_type': 'gbdt',
          'metric': 'rmse',
          'objective': 'regression',
          'n_jobs': -1,
          'seed': 236,
          'num_leaves': 280,
          'learning_rate': 0.026623466966581126,
          'max_depth': 73,
          'lambda_l1': 2.959759088169741,
          'lambda_l2': 1.331172832164913,
          'bagging_fraction': 0.9655406551472153,
          'bagging_freq': 9,
          'colsample_bytree': 0.6867118652742716}

y_pred_lgb, lgb_model = run_lgb(x, y, features, params)




lgb.plot_importance(lgb_model)




submission_df = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
submission_df['open_channels'] = y_pred_lgb
submission_df.to_csv('submit.csv', index = False,float_format='%.4f')




x.loc[:,'time'] = test_df['time']
examples = ['signal','f_n100Hz','f_50Hz']

fig, ax = plt.subplots(nrows=len(examples), ncols=1, figsize=(25, 4*len(examples)))
fig.subplots_adjust(hspace = .5)
ax = ax.ravel()
colors = plt.rcParams["axes.prop_cycle"]()

for i in range(len(examples)):
    
    c = next(colors)["color"]
    ax[i].grid()
    if examples[i] in ['dt_50Hz_energy_floor','50Hz_energy_floor']:
        ax[i].plot(x['time'], x[examples[i]],color=c, linewidth= 2)
        ax[i].set_ylim(0,4)

    ax[i].plot(x['time'], x[examples[i]],color=c, linewidth=.5)
    ax[i].set_title(examples[i], fontsize=24)
    ax[i].set_xlabel('Time (seconds)', fontsize=18)
    #ax[i].set_ylabel('current (pA)', fontsize=24)
    #ax[i].set_ylim(0,5)






