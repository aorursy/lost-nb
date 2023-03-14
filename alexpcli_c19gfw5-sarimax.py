#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.seasonal import STL
import itertools
from tqdm.auto import tqdm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('error')
import traceback
import logging




# adapted from https://github.com/endolith/waveform_analysis
def freq_from_crossings(sig):
    """
    Estimate frequency by counting zero crossings
    
    The MIT License (MIT)

    Copyright (c) 2016 endolith@gmail.com

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    
    """
    # Find all indices right before a rising-edge zero crossing
    indices = np.nonzero((sig[1:] >= 0) & (sig[:-1] < 0))[0]
    crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]
    return np.mean(np.diff(crossings))




get_ipython().system('date')
get_ipython().system('ls -ltrh ../input/covid19-global-forecasting-week-5/')




test = pd.read_csv(
    '/kaggle/input/covid19-global-forecasting-week-5/test.csv',
    parse_dates=['Date']
).fillna('')

train = pd.read_csv(
    '/kaggle/input/covid19-global-forecasting-week-5/train.csv',
    parse_dates=['Date']
).fillna('')




def prepare_dataset(X):
    X['Location'] = X[['Country_Region', 'Province_State', 'County']]        .apply(lambda x: f'{x.Country_Region} - {x.Province_State} - {x.County}' if x.County != ''
                    else f'{x.Country_Region} - {x.Province_State} ' if x.Province_State != ''
                    else f'{x.Country_Region}',
               axis='columns')
    return X.drop(columns=['Country_Region', 'Province_State', 'County'])




get_ipython().run_cell_magic('time', '', "train = prepare_dataset(train)\ntrain = train.set_index(['Location', 'Date', 'Target']).TargetValue.unstack('Target')")




train.head()




get_ipython().run_cell_magic('time', '', "test = prepare_dataset(test).set_index(['Location', 'Date', 'Target'])[['ForecastId']]")




test.head()




locations = train.index.get_level_values('Location').unique()
targets = train.columns.unique()




R = []
for location, target in tqdm(itertools.product(locations, targets),
                             total=len(locations)*len(targets)):
    X_train = train[train.index.get_level_values('Location') == location]        .reset_index().set_index('Date')[target]
    if X_train.sum() == 0:
        print(f"zero training set; skipping {location}, {target}")
        continue
    X_train.index.freq = 'D'

    loess = STL(X_train).fit()

    try:
        freq = freq_from_crossings(loess.seasonal.values)
    except:
        X_train.plot(figsize=(15,5))
        plt.show()
        loess.plot()
        plt.show()
        loess.seasonal.plot(figsize=(15,5))
        plt.show()
        print(f"error on freq_from_crossings(loess.seasonal.values); skipping {location}, {target}")
        continue

    try:
        mod = sm.tsa.statespace.SARIMAX(X_train, trend=[1, 1], seasonal_order=(1,1,1,freq), freq='D')
        res = mod.fit(disp=False)
    except:
        try:
            mod = sm.tsa.statespace.SARIMAX(X_train, trend=[1, 1], order=(1,1,1), freq='D')
            res = mod.fit(disp=False)
        except:
            try:
                mod = sm.tsa.statespace.SARIMAX(X_train, trend=[1, 1], freq='D')
                res = mod.fit(disp=False)
            except Exception as e:
                print(res.summary())
                logging.error(traceback.format_exc())
                print(f"SARIMAX error; skipping {location}, {target}")
                continue

    ## In-sample one-step-ahead predictions
    #predict = res.get_prediction()
    #predict_ci = predict.conf_int(alpha=0.1)

    X_test = test[test.index.get_level_values('Location') == location]        .reset_index().set_index('Date')['ForecastId']

    #X_test.head()

    # Dynamic predictions
    start = str(X_test.index.min().date())
    end = str(X_test.index.max().date())
    predict_dy = res.get_prediction(start=start, end=end)
    predict_dy_ci = predict_dy.conf_int(alpha=0.1)

    predict_dy_ci.index.name = 'Date'

    result = test        [(test.index.get_level_values('Location') == location)
         & (test.index.get_level_values('Target') == target)]\
        .join(predict_dy_ci.rename(columns={f'lower {target}': '0.05',
                                            f'upper {target}': '0.95'}))\
        .join(predict_dy.predicted_mean.to_frame('0.5'))

    result = result.melt('ForecastId',
                         value_vars=['0.05', '0.5', '0.95'],
                         var_name='Quantile',
                         value_name='TargetValue')

    result['ForecastId_Quantile']= result['ForecastId'].astype(str).str.cat(result['Quantile'], sep='_')

    result = result.drop(columns=['ForecastId', 'Quantile']).set_index('ForecastId_Quantile')

    R.append(result)




# Graph
fig, ax = plt.subplots(figsize=(9,4))

X_train.plot(ax=ax, style='o', label='Observed')

#predict.predicted_mean.plot(ax=ax, color='r', marker='x', linestyle='--', label='One-step-ahead forecast')

#ci = predict_ci[start:end]
#ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)

predict_dy.predicted_mean.plot(ax=ax, style='g', marker='.', label='Dynamic forecast')

ci = predict_dy_ci[start:end]
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='g', alpha=0.1)

ax.legend(loc='best');




submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv',
                         index_col=0)




submission.head()




submission[[]].join(pd.concat(R)).fillna(0).clip(lower=0, upper=None).to_csv('submission.csv')




get_ipython().system('ls -ltrh submission.csv')
get_ipython().system('head submission.csv')

