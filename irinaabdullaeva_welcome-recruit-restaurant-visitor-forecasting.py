#!/usr/bin/env python
# coding: utf-8



from __future__ import absolute_import, division, print_function, unicode_literals

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
sns.set(rc={'figure.figsize' : (12, 6)})
sns.set_style("darkgrid", {'axes.grid' : True})
# plt.style.use('seaborn-whitegrid')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from statsmodels.iolib.table import SimpleTable

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product                    # some useful functions
from tqdm import tqdm_notebook




air_res = pd.read_csv('../input/recruit-restaurant-visitor-forecasting/air_reserve.csv')
air_store = pd.read_csv('../input/recruit-restaurant-visitor-forecasting/air_store_info.csv')
hpg_res = pd.read_csv('../input/recruit-restaurant-visitor-forecasting/hpg_reserve.csv')
hpg_store = pd.read_csv('../input/recruit-restaurant-visitor-forecasting/hpg_store_info.csv')
air_visit = pd.read_csv('../input/recruit-restaurant-visitor-forecasting/air_visit_data.csv')
id_rel = pd.read_csv('../input/recruit-restaurant-visitor-forecasting/store_id_relation.csv')




air_res.head()




air_res.info()




air_store.head()




air_store.info()




air_visit.head()




air_visit.info()




hpg_res.info()




id_rel.head()




air_res.air_store_id.nunique() # number of unique restaurants in air system




hpg_res.hpg_store_id.nunique() # number of unique restaurants in hpg system




id_rel.air_store_id.nunique() # number of unique restaurants that are in both systems at once




# Rename some columns before merging
# air_store.rename(columns={"air_genre_name": "genre_name", " B": "c"})




# Merge tables
air = pd.merge(air_res, air_store, on='air_store_id')
hpg = pd.merge(hpg_res, hpg_store, on='hpg_store_id')




air_rel = pd.merge(air, id_rel, how='left', on='air_store_id')
hpg_rel = pd.merge(hpg, id_rel, how='left', on='hpg_store_id')
full = pd.merge(air_rel, hpg_rel, how='outer')




print("In air reservations are: %d \nIn hpg reservations are: %d \nIn both systems at once there are: %d" %       (air.shape[0], hpg.shape[0], id_rel.shape[0]))
print("So, totally must be: %d reservations."       % (full.shape[0]))




# Then we need to convert columns 'visit_datetime' and 'reserve_datetime' from object type -> to data/time type
full['visit_datetime'] = pd.to_datetime(full['visit_datetime'])
full['reserve_datetime'] = pd.to_datetime(full['reserve_datetime'])




full.info()




# Split converted date-time columns to year, month, date, day of week and time separate coluns of dataset
full['visit_year'] = pd.Series(full.visit_datetime.dt.year)
full['visit_month']  = pd.Series(full.visit_datetime.dt.month)
full['visit_date'] = pd.Series(full.visit_datetime.dt.day)
full['visit_weekday'] = pd.Series(full.visit_datetime.dt.weekday)
full['visit_time'] = pd.Series(full.visit_datetime.dt.time)




full['reserve_year'] = pd.Series(full.reserve_datetime.dt.year)
full['reserve_month']  = pd.Series(full.reserve_datetime.dt.month)
full['reserve_date'] = pd.Series(full.reserve_datetime.dt.day)
full['reserve_weekday'] = pd.Series(full.reserve_datetime.dt.weekday)
full['reserve_time'] = pd.Series(full.reserve_datetime.dt.time)




full.head()




# Fill NaNs to ease operations with ids and creation new columns
full['air_store_id'] = full['air_store_id'].fillna('0')
full['hpg_store_id'] = full['hpg_store_id'].fillna('0')
full['air_genre_name'] = full['air_genre_name'].fillna('0')
full['hpg_genre_name'] = full['hpg_genre_name'].fillna('0')




# Now lets put our data in order
# Create column 'store_id', where all ids from two sources will be collected together
full.loc[(full['air_genre_name'] != '0'), 'store_id'] = full['air_store_id']
full.loc[(full['air_genre_name'] == '0'), 'store_id'] = full['hpg_store_id']

# Create column 'store_genre_name', where all genres of restaurants will be collected together
full.loc[(full['air_genre_name'] != '0'), 'store_genre_name'] = full['air_genre_name']
full.loc[(full['air_genre_name'] == '0'), 'store_genre_name'] = full['hpg_genre_name']

# Create column 'air_hpg_link', to save connection between restaurants that are in both sources
full.loc[(full['air_store_id'] != '0') & (full['hpg_genre_name'] != '0'), 'air_hpg_link'] = full['air_store_id']
full.loc[(full['air_store_id'] != '0') & (full['hpg_genre_name'] == '0'), 'air_hpg_link'] = full['hpg_store_id']

# Create column 'store_genre_name', where all genres of restaurants will be collected together
full.loc[(full['air_genre_name'] != '0'), 'area_name'] = full['air_area_name']
full.loc[(full['air_genre_name'] == '0'), 'area_name'] = full['hpg_area_name']

full['air_hpg_link'] = full['air_hpg_link'].fillna('0')




full.head()




fullhist = full.groupby(['visit_datetime'],as_index=False).count().sort_values(by=['visit_datetime'])
fullhist_mnth = fullhist.loc[fullhist['visit_datetime'] <= pd.to_datetime('2016-02-01 23:59:00')]
fullhist_week = fullhist.loc[(fullhist['visit_datetime'] >= pd.to_datetime('2016-01-04 00:00:00'))                             & (fullhist['visit_datetime'] <= pd.to_datetime('2016-01-10 23:59:00'))]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(22,25))
ax1.plot(fullhist.visit_datetime, fullhist.store_id)
ax1.set_title("Visits during full period")
plt.ylabel("Number of visits")
plt.grid(True)

ax2.plot(fullhist_mnth.visit_datetime, fullhist_mnth.store_id)
ax2.set_title("Visits during one month")
plt.ylabel("Number of visits")
plt.grid(True)

ax3.plot(fullhist_week.visit_datetime, fullhist_week.store_id)
ax3.set_title("Visits during one week")
plt.ylabel("Number of visits")

plt.xlabel("Period")
plt.grid(True)
plt.show()




monthshist = full.groupby(['visit_month'],as_index=False).count().sort_values(by=['visit_month'])
weekshist = full.groupby(['visit_weekday'],as_index=False).count().sort_values(by=['visit_weekday'])
dayhist = full.groupby(['visit_time'],as_index=False).count().sort_values(by=['visit_time'])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,10))

plt.ylabel("Number of visits")
ax1.bar(monthshist.visit_month, monthshist.store_id)
ax1.set_title("Visits by months")
plt.xlabel("Months")

ax2.bar(weekshist.visit_weekday, weekshist.store_id, color='C2')
ax2.set_title("Visits by one week")
plt.xlabel("Week days")

ax3.plot(dayhist.visit_time, dayhist.store_id, color='C1')
ax3.set_title("Visits during one day")
plt.xlabel("Time")

plt.show()




full.head(1)




# Create dataset combined by visit date from full data
# Note: that in air data the last datetime of visit is near 05.2017 and in hpg data is ended at 04.2017
# So on last month of period (t.e. May 2017) joined data for both sited is not full!
datehist = full.loc[full['visit_datetime'] < pd.to_datetime('2017-05-01')]
datehist['visit_date_full'] = pd.Series(datehist.visit_datetime.dt.date)
datehist = datehist.groupby(['visit_date_full'],as_index=False).count().sort_values(by=['visit_date_full'])
datehist.tail()




# Create dataset combined by visit date from months data
date_mnth_hist = datehist.loc[datehist['visit_date_full'] < pd.to_datetime('2016-02-01')]
date_mnth_hist.tail()




# Importing everything from above
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error




# Calculate average of last n observations
def moving_average(data, n):
    """
    data - series type, which need to be smoothed
    n - size of the moving window
    """
    return np.average(data[-n:])

moving_average(datehist.store_id, 10)




moving_average(date_mnth_hist.store_id, 10)




# Or use Pandas implementation - DataFrame.rolling(window).mean(), that provides rolling window calculations.
# As main parameters to this function you should pass: 
# * window - Size of the moving window. This is the number of observations used for calculating the statistic.
# * win_type - Provide a window type.
# Also, note that by default, the result is set to the right edge of the window. 
# This can be changed to the center of the window by setting center=True.
def plotMovingAverage(data, window_size, plot_intervals=False, scale=1.96, plot_anomalies=False):
    """
        data - dataframe with timeseries
        window_size - rolling window size 
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 
    """
    rolling_mean = data.rolling(window=window_size).mean()

    plt.figure(figsize=(15,5))
    plt.title("Moving average\n window size = {}".format(window_size))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(data[window_size:], rolling_mean[window_size:])
        deviation = np.std(data[window_size:] - rolling_mean[window_size:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
        
        # Having the intervals, find abnormal values
        if plot_anomalies:           # Fix it!!!
            anomalies = pd.DataFrame(index=data.index, columns=[data.name])
            anomalies[data<lower_bond] = data[data<lower_bond]
            anomalies[data>upper_bond] = data[data>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
        
    plt.plot(data[window_size:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)




rolling_mean = date_mnth_hist.rolling(window=4).mean()
rolling_mean.tail()




# Let's smooth by the previous 7 days so we get weekly trend more clearly.
plotMovingAverage(date_mnth_hist.store_id, 7)




# Let's smooth by the previous 2 days so we get trend more clearly without loosing extra high values of visitors on weekends.
plotMovingAverage(date_mnth_hist.store_id, 2)




# Plot confidence intervals for our smoothed values for a full period
plotMovingAverage(datehist.store_id, 7, plot_intervals=True)




def weighted_average(data, weights):
    """
        Calculate weighter average on series
        data - series type, which need to be smoothed
        weights - weights of samples
    """
    result = 0.0
    weights.reverse()
    for n in range(len(weights)):
        result += data.iloc[-n-1] * weights[n]
    return float(result)

weighted_average(date_mnth_hist.store_id, [0.6, 0.4, 0.2, 0.1])




def exponential_smoothing(data, alpha):
    """
        data - dataset with timestamps
        alpha - float [0.0, 1.0], smoothing parameter
    """
    result = [data[0]] # first value is same as series
    for n in range(1, len(data)):
        result.append(alpha * data[n] + (1 - alpha) * result[n-1])
    return result




def plotExponentialSmoothing(data, alphas):
    """
        Plots exponential smoothing with different alphas
        data - dataset with timestamps
        alphas - list of floats, smoothing parameters
        
    """
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=(28, 10))
        for alpha in alphas:
            plt.plot(exponential_smoothing(data, alpha), label="Alpha {}".format(alpha))
        plt.plot(data.values, "c", label = "Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Exponential Smoothing")
        plt.grid(True);




plotExponentialSmoothing(date_mnth_hist.store_id, [0.5, 0.3, 0.05])




def double_exponential_smoothing(data, alpha, beta):
    """
        data - dataset with timeseries
        alpha - float [0.0, 1.0], smoothing parameter for level
        beta - float [0.0, 1.0], smoothing parameter for trend
    """
    # first value is same as series
    result = [data[0]]
    for n in range(1, len(data)+1):
        if n == 1:
            level, trend = data[0], data[1] - data[0]
        if n >= len(data): # forecasting
            value = result[-1]
        else:
            value = data[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result




def plotDoubleExponentialSmoothing(data, alphas, betas):
    """
        Plots double exponential smoothing with different alphas and betas
        data - dataset with timestamps
        alphas - list of floats, smoothing parameters for level
        betas - list of floats, smoothing parameters for trend
    """
    
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=(20, 8))
        for alpha in alphas:
            for beta in betas:
                plt.plot(double_exponential_smoothing(data, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
        plt.plot(data.values, label = "Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Double Exponential Smoothing")
        plt.grid(True)




# Let's smooth full data with different settings of parameters (alpha and beta) so we cas see which of them performs the best.
plotDoubleExponentialSmoothing(datehist.store_id, alphas=[0.9, 0.02], betas=[0.9, 0.02])




# Then try to smooth weekly data with different settings of parameters (alpha and beta).
plotDoubleExponentialSmoothing(fullhist_week.reset_index().store_id, alphas=[0.9, 0.02], betas=[0.9, 0.02])




def initial_trend(series, season_len):
    """
    series - initial time series
    season_len - length of a season
    """
    summ = 0.0
    for i in range(season_len):
        summ += float(series[i + season_len] - series[i]) / season_len
    return summ / season_len  




def initial_seasonal_components(series, season_len):
    """
    series - initial time series
    season_len - length of a season
    """
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series) / season_len)
#     print("n_seasons=%d, len(series)=%d, season_len=%d" % (n_seasons, len(series), season_len))
    # let's calculate season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[season_len*j : season_len*j+season_len]) / float(season_len))
    # let's calculate initial values
    for i in range(season_len):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[season_len*j+i] - season_averages[j]
        seasonals[i] = sum_of_vals_over_avg / n_seasons
    return seasonals   




def triple_exponential_smoothing(series, season_len, alpha, beta, gamma, n_preds, scaling_factor=1.96):
    """
    Holt-Winters model with the anomalies detection using Brutlag method
    series - initial time series
    season_len - length of a season
    alpha, beta, gamma - Holt-Winters model coefficients
    n_preds - predictions horizon
    scaling_factor - sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)   
    """
    result = []
    Smooth = []
    Season = []
    Trend = []
    PredictedDeviation = []
    UpperBond = []
    LowerBond = []  
    seasonals = initial_seasonal_components(series, season_len)
    
    for i in range(len(series) + n_preds):
        if i == 0: # components initialization
            smooth = series[0]
            trend = initial_trend(series, season_len)
            result.append(series[0])
            Smooth.append(smooth)
            Trend.append(trend)
            Season.append(seasonals[i%season_len])
            
            PredictedDeviation.append(0)
            UpperBond.append(result[0] + scaling_factor * PredictedDeviation[0])
            LowerBond.append(result[0] - scaling_factor * PredictedDeviation[0])                   
            continue
                
        if i >= len(series): # predicting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%season_len])
            # when predicting we increase uncertainty on each step
            PredictedDeviation.append(PredictedDeviation[-1]*1.01)  
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha * (val-seasonals[i%season_len]) + (1-alpha) * (smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%season_len] = gamma*(val-smooth) + (1-gamma)*seasonals[i%season_len]
            result.append(smooth+trend+seasonals[i%season_len])
            # Deviation is calculated according to Brutlag algorithm.
            PredictedDeviation.append(gamma * np.abs(series[i] - result[i]) + (1-gamma)*PredictedDeviation[-1])         
            UpperBond.append(result[-1] + scaling_factor * PredictedDeviation[-1])
            LowerBond.append(result[-1] - scaling_factor * PredictedDeviation[-1])
            Smooth.append(smooth)
            Trend.append(trend)
            Season.append(seasonals[i%season_len])
    return[result, LowerBond, UpperBond]




with plt.style.context('seaborn-white'): 
        plt.figure(figsize=(20, 8))
        plt.plot(triple_exponential_smoothing(date_mnth_hist.store_id, 7, 0.4, 0.5, 0.5, 7)[0], label = "Smoothed")
        plt.plot(date_mnth_hist.store_id, label = "Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Holt-Winters model Smoothing")
        plt.grid(True)




from sklearn.model_selection import TimeSeriesSplit # you have time seriaes splitting already done for you

def timeseriesCVscore(params, series, loss_function=mean_squared_error, season_len=7):
    """
        Returns error on CV  
        params - vector of parameters for optimization
        series - dataset with timeseries
        season_len - season length for Holt-Winters model
    """
    # errors array
    errors = []
    
    values = series.values
    alpha, beta, gamma = params
    
    # set the number of folds for cross-validation
    tscv = TimeSeriesSplit(n_splits=5) 
    
    # iterating over folds, train model on each, forecast and calculate error
    for train, test in tscv.split(values):
        res = triple_exponential_smoothing(series=values[train], season_len=season_len, alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))[0]
        predictions = res[-len(test):]
        actual = values[test]
        error = loss_function(predictions, actual)
        errors.append(error)
        
    return np.mean(np.array(errors))




# leave some data for testing = one month
test_period_len = 30




from scipy.optimize import minimize              # for function minimization

get_ipython().run_line_magic('time', '')
data = datehist.store_id[:-test_period_len] # leave some data for testing = one month

# initializing model parameters alpha, beta and gamma
x = [0, 0, 0] 

# Minimization of loss function of one or more variables.
opt = minimize(timeseriesCVscore, x0=x, args=(data, median_absolute_error), method="TNC", bounds = ((0, 1), (0, 1), (0, 1)))

# Take optimal values
alpha_final, beta_final, gamma_final = opt.x
print("Best alpha=%f, best beta=%f, best gamma=%f" % (alpha_final, beta_final, gamma_final))

# and train the model with them, forecasting for the next 7 days
ret = triple_exponential_smoothing(data, season_len = 7, alpha = alpha_final, beta = beta_final, gamma = gamma_final, n_preds = 7, scaling_factor = 3) 




def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100




def plotHoltWinters(series, returned, plot_intervals=False, plot_anomalies=False):
    """
        series - dataset with timeseries
        returned - list, returned from triple_exponential_smoothing func: (result, LowerBond, UpperBond)
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 
    """
    [result, LowerBond, UpperBond] = returned
    plt.figure(figsize=(28, 10))
    plt.plot(result, label = "Model")
    plt.plot(series.values, label = "Actual")
    error = mean_absolute_percentage_error(series.values, result[:len(series)])
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    
    if plot_anomalies:
        anomalies = np.array([np.NaN]*len(series))
        anomalies[series.values < LowerBond[:len(series)]] =             series.values[series.values < LowerBond[:len(series)]]
        anomalies[series.values > UpperBond[:len(series)]] =             series.values[series.values > UpperBond[:len(series)]]
        plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
    if plot_intervals:
        plt.plot(UpperBond, "r--", alpha=0.5, label = "Up/Low confidence")
        plt.plot(LowerBond, "r--", alpha=0.5)
        plt.fill_between(x=range(0,len(result[:-7])), y1=UpperBond, y2=LowerBond, alpha=0.2, color = "grey") 
        
    plt.vlines(len(series), ymin=min(LowerBond), ymax=max(UpperBond), linestyles='dashed')
    plt.axvspan(len(series)-test_period_len, len(result), alpha=0.3, color='lightgrey')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc="best", fontsize=13);




plotHoltWinters(datehist.store_id[:-test_period_len], ret, plot_intervals=True, plot_anomalies=True)




def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)

        print("Dickey-Fuller test criterion: p=%f" % sm.tsa.stattools.adfuller(y)[1])

        plt.tight_layout()
        
    test = sm.tsa.adfuller(y)
    print('adf: ', test[0])  
    print ('p-value: ', test[1])
    print('Critical values: ', test[4])
    if test[0]> test[4]['5%']: 
        print('A unit root is present, series are not stationary') 
    else:
        print('No unit roots are present, series are stationary') 
    return 

tsplot(datehist.store_id, lags=30)




data_diff1 = datehist.store_id.diff(periods=1).dropna()
tsplot(data_diff1, lags=30)




m = data_diff1.index[len(data_diff1.index)//2+1]
r1 = sm.stats.DescrStatsW(data_diff1[m:])
r2 = sm.stats.DescrStatsW(data_diff1[:m])
print('p-value: ', sm.stats.CompareMeans(r1,r2).ttest_ind()[1]) 




tsplot(data_diff1, lags=30)




# setting initial values and some bounds for them
ps = range(3, 6)
d=1 
qs = range(3, 6)
Ps = range(3, 5)
D=1 
Qs = range(3, 5)
s = 7 # season length is still 7

# creating list with all the possible combinations of parameters
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)




def optimizeSARIMA(parameters_list, d, D, s):
    """
        Return dataframe with parameters and corresponding AIC
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order 
        s - length of season
    """
    
    results = []
    best_aic = float("inf")

    for param in tqdm_notebook(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model=sm.tsa.statespace.SARIMAX(data_diff1, order=(param[0], d, param[1]), 
                                            seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table




# %time
# result_table = optimizeSARIMA(parameters_list, d, D, s)




# set the parameters that give the lowest AIC
# p, q, P, Q = result_table.parameters[0]
p = 5
q = 5
P = 3
Q = 3
print("p = %f, q = %f, P = %f, Q = %f" % (p, q, P, Q))




best_model=sm.tsa.statespace.SARIMAX(data_diff1, order=(p, d, q), 
                                        seasonal_order=(P, D, Q, s)).fit(disp=-1)
print(best_model.summary())




# Firstly let's inspect the residuals of the model.
tsplot(best_model.resid[7+1:], lags=30)




#  resid, хранит остатки модели, qstat=True, означает что применяем указынный тест к коэф-ам
# Calculate the autocorrelation function.
q_test = sm.tsa.stattools.acf(best_model.resid[7+1:], qstat=True) 
acf_scores = pd.DataFrame({'Q-stat':q_test[1], 'p-value':q_test[2]})
acf_scores




data = pd.DataFrame.from_dict({'actual':data_diff1})
data['sarima_model'] = best_model.fittedvalues
forecast = best_model.predict(start = data.shape[0], end = data.shape[0]+ 14)
forecast = data.sarima_model.append(forecast)
# forecast.tail()
data.sarima_model.shape




def plotSARIMA(series, model, n_steps):
    """
        Plots model vs predicted values
        series - dataset with timeseries
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future
        
    """
    # adding model values
    data = pd.DataFrame.from_dict({'actual':data_diff1})
    data['sarima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data['sarima_model'][:s+d] = np.NaN
    
    # forecasting on n_steps forward 
    forecast = model.predict(start = data.shape[0], end = data.shape[0]+n_steps)
    forecast = data.sarima_model.append(forecast)
    
    # calculate estimations of the last 2 weeks of predictions (which are still registered in actual data) 
    true = data.actual[data.shape[0]-n_steps:data.shape[0]]
    pred = data.sarima_model[data.shape[0]-n_steps:data.shape[0]]
    r2 = r2_score(true, pred)
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    print("R2 = %f, MSE = %f, MAE = %f" % (r2, mse, mae))

    plt.figure(figsize=(22, 7))
    plt.title("SARIMA Model Predictions Plot")
    plt.plot(forecast, color='r', label="model")
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label="actual")
    plt.legend()
    plt.grid(True);

plotSARIMA(data_diff1, best_model, 14)




# Create dataset combined by visit date from full data
# Note: that in air data the last datetime of visit is near 05.2017 and in hpg data is ended at 04.2017
# So on last month of period (t.e. May 2017) joined data for both sited is not full!
# dthist = full.loc[full['visit_datetime'] < pd.to_datetime('2017-05-01')]
# dthist['visit_date_full'] = pd.Series(datehist.visit_datetime.dt.date)
# dthist = dthist.groupby(['visit_date_full'],as_index=False).count().sort_values(by=['visit_date_full'])




# Creating a copy of the initial datagrame to make various transformations 
datehist_timeidx = datehist.set_index('visit_date_full')
data = pd.DataFrame(datehist_timeidx.store_id.copy())
data.columns = ["y"]
# Adding the lag of the target variable from 7 steps back up to 21
for i in range(7, 22):
    data["lag_{}".format(i)] = data.y.shift(i)
data.tail()




def timeseries_train_test_split(X, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """
    # get the index after which test set starts
    test_index = int(len(X)*(1-test_size))
    
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    return X_train, X_test, y_train, y_test




from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# for time-series cross-validation set 5 folds 
tscv = TimeSeriesSplit(n_splits=5)

y = data.dropna().y
X = data.dropna().drop(['y'], axis=1)

# reserve 30% of data for testing
X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

# Create and train linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)




def plotModelResults(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):
    """
        Plots modelled vs actual values, prediction intervals and anomalies
    """
#     Make prediction
    prediction = model.predict(X_test)
    with plt.style.context('seaborn-white'): 
        plt.figure(figsize=(22, 7))
        plt.plot(prediction, "g", label="prediction", linewidth=2.0)
        plt.plot(y_test.values, label="actual", linewidth=2.0)
    
        if plot_intervals:
            cv = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")                         
            mae = cv.mean() * (-1)
            deviation = cv.std()
        
            scale = 1.96
            lower = prediction - (mae + scale * deviation)
            upper = prediction + (mae + scale * deviation)
        
            plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
            plt.plot(upper, "r--", alpha=0.5)
        
            if plot_anomalies:
                anomalies = np.array([np.NaN]*len(y_test))
                anomalies[y_test<lower] = y_test[y_test<lower]
                anomalies[y_test>upper] = y_test[y_test>upper]
                plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
        error = mean_absolute_percentage_error(prediction, y_test)
        plt.title("Mean absolute percentage error {0:.2f}%".format(error))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)




def plotCoefficients(model):
    """
        Plots sorted coefficient values of the model
    """
    with plt.style.context('seaborn-white'): 
        coefs = pd.DataFrame(model.coef_, X_train.columns)
        coefs.columns = ["coef"]
        coefs["abs"] = coefs.coef.apply(np.abs)
        coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    
        plt.figure(figsize=(22, 7))
        coefs.coef.plot(kind='bar')
        plt.grid(True, axis='y')
        plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed')




plotModelResults(lr, plot_intervals=True)
plotCoefficients(lr)




data.index = pd.to_datetime(data.index)
data["weekday"] = data.index.weekday
data['is_weekend'] = data.weekday.isin([5,6])*1
# List of public and common local holidays in Japan
holidays = pd.to_datetime(pd.Series(['01.01.2016', '02.01.2016', '03.01.2016', '11.01.2016', '11.02.2016', '03.03.2016', '20.03.2016', '21.03.2016',            '29.04.2016', '03.05.2016', '04.05.2016', '05.05.2016', '07.07.2016', '18.07.2016', '11.08.2016', '19.09.2016',            '22.09.2016', '10.10.2016', '03.11.2016', '15.11.2016', '23.11.2016', '23.12.2016', '25.12.2016', '31.12.2016',            '01.01.2017', '02.01.2017', '03.01.2017', '09.01.2017', '11.02.2017', '03.03.2017', '20.03.2017', '29.04.2017']))
data['is_holiday'] = data.index.isin(holidays)*1
data.tail()




def plotFeatures(df, features):
    """
    Visualizing features of data
    df - dataframe,from which featuresare taken
    features - list of features
    """
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(22, 7))
        plt.title("Encoded features")
        for i in range(0, len(features)):
            plt.plot(data[features[i]], label=features[i], linewidth=2.0)
        plt.grid(True)
        plt.legend(loc="best")




ftrs = ['weekday', 'is_weekend', 'is_holiday']
plotFeatures(data, ftrs)




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Create and train linear regression again
y = data.dropna().y
X = data.dropna().drop(['y'], axis=1)

X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

# But this time perform normalization on the data 
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True)
plotCoefficients(lr)




def code_mean(data, cat_feature, real_feature):
    """
    Returns a dictionary where keys are unique categories of the cat_feature,
    and values are means over real_feature
    """
    return dict(data.groupby(cat_feature)[real_feature].mean())




average_wd = code_mean(data, 'weekday', "y")
plt.figure(figsize=(7, 5))
plt.title("Weekday averages")
pd.DataFrame.from_dict(average_wd, orient='index')[0].plot()
plt.grid(True);




# Finally, let's put all the transformations together in a single function .
def prepareData(series, lag_start, lag_end, test_size, target_encoding=False):
    """
        series: pd.DataFrame
            dataframe with timeseries

        lag_start: int
            initial step back in time to slice target variable 
            example - lag_start = 1 means that the model 
                      will see yesterday's values to predict today

        lag_end: int
            final step back in time to slice target variable
            example - lag_end = 4 means that the model 
                      will see up to 4 days back in time to predict today

        test_size: float
            size of the test dataset after train/test split as percentage of dataset

        target_encoding: boolean
            if True - add target averages to the dataset
        
    """
    
    # copy of the initial dataset
#     data_timeidx = datehist.set_index('visit_date_full')    
    data = pd.DataFrame(series.copy())
    data.columns = ["y"]
    
    # lags of series
    for i in range(lag_start, lag_end):
        data["lag_{}".format(i)] = data.y.shift(i)
    
    # datetime features
    data.index = pd.to_datetime(data.index)
    data["weekday"] = data.index.weekday
    data['is_weekend'] = data.weekday.isin([5,6])*1
    # List of public and common local holidays in Japan
    holidays = pd.to_datetime(pd.Series(['01.01.2016', '02.01.2016', '03.01.2016', '11.01.2016', '11.02.2016', '03.03.2016', '20.03.2016', '21.03.2016',                '29.04.2016', '03.05.2016', '04.05.2016', '05.05.2016', '07.07.2016', '18.07.2016', '11.08.2016', '19.09.2016',                '22.09.2016', '10.10.2016', '03.11.2016', '15.11.2016', '23.11.2016', '23.12.2016', '25.12.2016', '31.12.2016',                '01.01.2017', '02.01.2017', '03.01.2017', '09.01.2017', '11.02.2017', '03.03.2017', '20.03.2017', '29.04.2017']))
    data['is_holiday'] = data.index.isin(holidays)*1
    
    if target_encoding:
        # calculate averages on train set only
        test_index = int(len(data.dropna())*(1-test_size))
        data['weekday_average'] = list(map(code_mean(data[:test_index], 'weekday', "y").get, data.weekday))

        # drop encoded variables 
        data.drop(["weekday"], axis=1, inplace=True)
    
    # train-test split
    y = data.dropna().y
    X = data.dropna().drop(['y'], axis=1)
    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test




datehist_timeidx = datehist.set_index('visit_date_full')
X_train, X_test, y_train, y_test = prepareData(datehist_timeidx.store_id, lag_start=7, lag_end=22, test_size=0.3, target_encoding=True)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True, plot_anomalies=True)
plotCoefficients(lr)




plt.figure(figsize=(15, 10))
sns.heatmap(X_train.corr());




from sklearn.linear_model import LassoCV, RidgeCV

# Perform Ridge (L2) regression
ridge = RidgeCV(cv=tscv)
ridge.fit(X_train_scaled, y_train)

plotModelResults(ridge, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True, plot_anomalies=True)
plotCoefficients(ridge)




ridge.alpha_




lasso = LassoCV(cv=tscv)
lasso.fit(X_train_scaled, y_train)

plotModelResults(lasso,  X_train=X_train_scaled, X_test=X_test_scaled,plot_intervals=True, plot_anomalies=True)
plotCoefficients(lasso)




lasso.alpha_ 




from xgboost import XGBRegressor 
xgb = XGBRegressor(booster='gblinear', alpha=11, updater='coord_descent', feature_selector='greedy', top_k=7)
xgb.fit(X_train_scaled, y_train)
plotModelResults(xgb,  X_train=X_train_scaled, X_test=X_test_scaled,plot_intervals=True, plot_anomalies=True)

