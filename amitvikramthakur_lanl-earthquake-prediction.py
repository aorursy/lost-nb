#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, tqdm_notebook

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR
from sklearn import linear_model
from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import pearsonr
from scipy.stats import mode
from scipy.signal import find_peaks
from sklearn.model_selection import KFold, cross_validate, cross_val_predict, TimeSeriesSplit

#import MiniSom
#print(sklearn.__version__) # => '0.20.3'


# In[3]:


print("reading the file...")
df_trainset = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
print("reading done")
print(df_trainset.shape)
pd.options.display.precision = 15 # pandas doesn't show us all the decimals
df_trainset.head()


# In[4]:


df_trainset.head()


# In[5]:


df_trainset[0:150000].describe()


# In[6]:


def PlotLinearGraph_Actual_vs_Pred_absolute(y_actual,y_pred):
    # Create linear regression object to plot best fit line across y_actual vs y_pred
    linearRegressor = linear_model.LinearRegression()
    linearRegressor.fit(y_actual.reshape(-1,1), y_pred)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_actual, y_pred)
    plt.plot(y_actual, linearRegressor.predict(y_actual.reshape(-1,1)), color = 'black')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.xlabel('actual', fontsize=12)
    plt.ylabel('predicted', fontsize=12)
    plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
    plt.show()
    print("Line - Intercept: "+str(linearRegressor.intercept_))
    print("Line - Slope: "+str(linearRegressor.coef_))
    print()
    
    score_mae = mean_absolute_error(y_actual, y_pred)
    score_r2 = r2_score(y_actual, y_pred)
    score_mse = mean_squared_error(y_actual, y_pred)
    print(f'MAE (Score): {score_mae:0.3f}')
    print(f'r2_score: {score_r2:0.3f}')
    print(f'MSE: {score_mse:0.3f}')


# In[7]:


#Preparation of Training and Validation datasets ("segment_150000")
'''
//Analysis of Data Visualisation//

Parameters of interest: std,max,
print(np.diff(y_time_to_failure[0:30]).mean()) # -0.03896207078551724 = -0.0390
In order of importance:-
 - mean_Hdistance_bw_higherthan_0_peak_pts 
 - mean_Hdistance_bw_higherthan_10_peak_pts
 - mean_Hdistance_bw_lowerthan_n0_valley_pts
 - no_of_peaks
 - no_of_valleys_total
 - no_of_peaks_total
 - highest_count_density_vertically
 - lowest_count_density_vertically
 - count_0_9
 - count_10_19
 - count_n10_n19
 - q95
 - q05
 - mean_change_rate
 - amount_of_silence_total
 - std>=10 => ttf<=10
 - count_20_29
 - max>=3000=>ttf=0-0.5,max>=500=>ttf=0-10
 - count_30_39
 - count_40_49
 - min (similar approach as that for 'max')
 - Mean>=4.8 => ttf<=13;Mean<=3.9=>ttf=10-17,1-8
 
Parameters to consider for each segment of 150000 rows:-
- threshold1crossed
- max
- min
- avg
- std
- no. of peaks
- last_30000_no_of_peaks
- stress-increase-rate = sum of peak values

columns_X_150000 = [
                    'threshold1crossed',
                    'max/min',
                    'avg',
                    'std',
                    'kurtosis',
                    'skewness',
                    'no_of_peaks',
                    'last_30000_no_of_peaks',
                    'last_30000_max',
                    'first_30000_max',
                   ]

'''


columns_X_150000 = []



def addParameters_150000(seg_150000,X_150000,segment_150000):
    
    
    x = seg_150000['acoustic_data'].values
    #use np.log() to reduce fluctuations in acoustic data
    x_log10 = np.log10(np.abs(x)+np.array([1]*np.size(x))) 
    
    
    ##/2/ok/##
    X_150000.loc[segment_150000, 'mean'] = x.mean()
    X_150000.loc[segment_150000, 'std'] = x.std()
    X_150000.loc[segment_150000, 'max'] = x.max()
    X_150000.loc[segment_150000, 'min'] = x.min()
    
    '''
    ##/1/No/##
    # Measure of amount of energy released
    s = x.sum(); s_sign = int(s/abs(s))
    energy_released = s_sign*np.log10(max(s**4,1e-10));
    X_150000.loc[segment_150000, 'energy_released'] = energy_released
    '''
    
    # count of signal value from 0 to 9
    x_div_10 = x//10
    x_div_100 = x//100
    ##/3/Yes/##
    X_150000.loc[segment_150000, 'count_0_9'] = (x_div_10 == 0).sum()
    n = max((x_div_10 == 1).sum(),1e-10);s = 1/n;d = -np.log10(max((1-s),1e-10))
    X_150000.loc[segment_150000, 'count_10_19'] = d#/3/Yes/#
    '''##/3/ok/##
    n = max((x_div_10 == 2).sum(),1e-10);s = 1/n;d = -np.log10(max((1-s),1e-10))
    X_150000.loc[segment_150000, 'count_20_29'] = d
    n = max((x_div_10 == 3).sum(),1e-10);s = 1/n;d = -np.log10(max((1-s),1e-10))
    X_150000.loc[segment_150000, 'count_30_39'] = d
    n = max((x_div_10 == 4).sum(),1e-10);s = 1/n;d = -np.log10(max((1-s),1e-10))
    X_150000.loc[segment_150000, 'count_40_49'] = d
    n = max((x_div_10 == 5).sum(),1e-10);s = 1/n;d = -np.log10(max((1-s),1e-10))
    X_150000.loc[segment_150000, 'count_50_59'] = d
    n = max((x_div_10 >= 6).sum(),1e-10);s = 1/n;d = -np.log10(max((1-s),1e-10))
    X_150000.loc[segment_150000, 'count_60_above'] = d
    '''
    
    ##/3/Yes/##
    n = max((x_div_10 == -1).sum(),1e-10);s = 1/n;d = -np.log10(max((1-s),1e-10))
    X_150000.loc[segment_150000, 'count_n10_n19'] = d
    '''##/3/ok/##
    n = max((x_div_10 == -2).sum(),1e-10);s = 1/n;d = -np.log10(max((1-s),1e-10))
    X_150000.loc[segment_150000, 'count_n20_n29'] = d
    n = max((x_div_10 == -3).sum(),1e-10);s = 1/n;d = -np.log10(max((1-s),1e-10))
    X_150000.loc[segment_150000, 'count_n30_n39'] = d
    n = max((x_div_10 == -4).sum(),1e-10);s = 1/n;d = -np.log10(max((1-s),1e-10))
    X_150000.loc[segment_150000, 'count_n40_n49'] = d
    n = max((x_div_10 == -5).sum(),1e-10);s = 1/n;d = -np.log10(max((1-s),1e-10))
    X_150000.loc[segment_150000, 'count_n50_n59'] = d
    n = max((x_div_10 <= -6).sum(),1e-10);s = 1/n;d = -np.log10(max((1-s),1e-10))
    X_150000.loc[segment_150000, 'count_n60_below'] = d
    '''
    
    '''
    ##/4/ok/##
    # analysis of indices of count_i_j data points
    
    def indices_count_i_j_calc(indices_count_i_j,indices_count_i_j_title):
        if len(indices_count_i_j)!=0:
            X_150000.loc[segment_150000, indices_count_i_j_title+'_mean'] =  indices_count_i_j.mean()
            X_150000.loc[segment_150000, indices_count_i_j_title+'_std'] =  indices_count_i_j.std()
            X_150000.loc[segment_150000, indices_count_i_j_title+'_max'] =  indices_count_i_j.max()
            X_150000.loc[segment_150000, indices_count_i_j_title+'_min'] =  indices_count_i_j.min()
        else:
            X_150000.loc[segment_150000, indices_count_i_j_title+'_mean'] =  -1
            X_150000.loc[segment_150000, indices_count_i_j_title+'_std'] =  -1
            X_150000.loc[segment_150000, indices_count_i_j_title+'_max'] =  -1
            X_150000.loc[segment_150000, indices_count_i_j_title+'_min'] =  -1
        
    indices_count_i_j_calc(np.where(x_div_10 == 0)[0],'indices_count_0_9')    
    indices_count_i_j_calc(np.where(x_div_10 == 1)[0],'indices_count_10_19')    
    indices_count_i_j_calc(np.where(x_div_10 == 2)[0],'indices_count_20_29')     
    indices_count_i_j_calc(np.where(x_div_10 == 3)[0],'indices_count_30_39')    
    indices_count_i_j_calc(np.where(x_div_10 == 4)[0],'indices_count_40_49')    
    indices_count_i_j_calc(np.where(x_div_100 >= 2)[0],'indices_count_200_above')
    
    indices_count_i_j_calc(np.where(x_div_10 == -1)[0],'indices_count_n10_n19')    
    indices_count_i_j_calc(np.where(x_div_10 == -2)[0],'indices_count_n20_n29')    
    indices_count_i_j_calc(np.where(x_div_10 == -3)[0],'indices_count_n30_n39')    
    indices_count_i_j_calc(np.where(x_div_10 == -4)[0],'indices_count_n40_n49')    
    indices_count_i_j_calc(np.where(x_div_100 <= -2)[0],'indices_count_n200_below')
    '''
    
    ##/5/Yes/##
    # mean_Hdistance_bw_pts
    def mean_Hdistance_bw_pts_calc(i,mean_Hdistance_bw_pts_title,comparison_op):
        if comparison_op == '>':
            Hdistance_arr = np.diff(np.where((x>=i)!=False)[0],n=1)
        elif comparison_op == '<':
            Hdistance_arr = np.diff(np.where((x<=i)!=False)[0],n=1)
        else:
            Hdistance_arr = np.diff(np.where((x==i)!=False)[0],n=1)
        mean_Hdistance=150_000
        if np.size(Hdistance_arr)>0:
            mean_Hdistance = np.mean(Hdistance_arr)
        X_150000.loc[segment_150000,mean_Hdistance_bw_pts_title] = mean_Hdistance
    
    mean_Hdistance_bw_pts_calc(0,'mean_Hdistance_bw_higherthan_0_peak_pts','>')
    mean_Hdistance_bw_pts_calc(10,'mean_Hdistance_bw_higherthan_10_peak_pts','>')
    '''mean_Hdistance_bw_pts_calc(20,'mean_Hdistance_bw_higherthan_20_peak_pts','>')
    mean_Hdistance_bw_pts_calc(30,'mean_Hdistance_bw_higherthan_30_peak_pts','>')
    mean_Hdistance_bw_pts_calc(40,'mean_Hdistance_bw_higherthan_40_peak_pts','>')
    mean_Hdistance_bw_pts_calc(50,'mean_Hdistance_bw_higherthan_50_peak_pts','>')
    mean_Hdistance_bw_pts_calc(60,'mean_Hdistance_bw_higherthan_60_peak_pts','>')
    mean_Hdistance_bw_pts_calc(1000,'mean_Hdistance_bw_higherthan_1000_peak_pts','>')
    '''
    
    mean_Hdistance_bw_pts_calc(0,'mean_Hdistance_bw_lowerthan_n0_valley_pts','<')
    '''mean_Hdistance_bw_pts_calc(-10,'mean_Hdistance_bw_lowerthan_n10_valley_pts','<')
    mean_Hdistance_bw_pts_calc(-20,'mean_Hdistance_bw_lowerthan_n20_valley_pts','<')
    mean_Hdistance_bw_pts_calc(-30,'mean_Hdistance_bw_lowerthan_n30_valley_pts','<')
    mean_Hdistance_bw_pts_calc(-40,'mean_Hdistance_bw_lowerthan_n40_valley_pts','<')
    mean_Hdistance_bw_pts_calc(-50,'mean_Hdistance_bw_lowerthan_n50_valley_pts','<')
    mean_Hdistance_bw_pts_calc(-60,'mean_Hdistance_bw_lowerthan_n60_valley_pts','<')
    mean_Hdistance_bw_pts_calc(-1000,'mean_Hdistance_bw_lowerthan_n1000_valley_pts','<')
    '''
    
    
    
    #idx = np.arange(0,x.size)
    #sum_of_idx = idx.sum()+1
    #X_150000.loc[segment_150000,'idx_weightage1_above_zero'] = (x.clip(min=0)*idx).sum()/sum_of_idx
    #X_150000.loc[segment_150000,'idx_weightage1_below_zero'] = (x.clip(max=0)*idx).sum()/sum_of_idx
    
    
    # Highest and lowest densities and their locations
    density_max_vpos = -1
    density_max_v = -1
    density_min_vpos = -1
    density_min_v = 150000
    
    def density_max_comparison(density,pos,density_max,density_max_pos):
        if  density > density_max:
            density_max_pos = pos
            density_max = density
        
        return density_max,density_max_pos
    
    def density_min_comparison(density,pos,density_min,density_min_pos):
        if  density < density_min:
            density_min_pos = pos
            density_min = density
        
        return density_min,density_min_pos
    
    #Highest Density Calculation
    density_v = (x_div_10 == 0).sum()/20
    density_max_v,density_max_vpos = density_max_comparison(density_v,0,density_max_v,density_max_vpos)
    
    density_v = (x_div_10 >= 6).sum()/(np.max([x.max(),70])-60)
    density_max_v,density_max_vpos = density_max_comparison(density_v,6,density_max_v,density_max_vpos)
    
    density_v = (x_div_10 <= -6).sum()/(-np.min([x.min(),-70])-60)
    density_max_v,density_max_vpos = density_max_comparison(density_v,-6,density_max_v,density_max_vpos)
    
    for i in range(1,6):
        density_v = (x_div_10 == i).sum()/10
        density_max_v,density_max_vpos =             density_max_comparison(density_v,i,density_max_v,density_max_vpos)
        
        density_v = (x_div_10 == -i).sum()/10
        density_max_v,density_max_vpos =             density_max_comparison(density_v,-i,density_max_v,density_max_vpos)

    
    #Lowest Density Calculation  
    density_v = (x_div_10 == 0).sum()/20
    density_min_v,density_min_vpos = density_min_comparison(density_v,0,density_min_v,density_min_vpos)
    
    density_v = (x_div_10 >= 6).sum()/(np.max([x.max(),70])-60)
    density_min_v,density_min_vpos = density_min_comparison(density_v,6,density_min_v,density_min_vpos)
    
    density_v = (x_div_10 <= -6).sum()/(-np.min([x.min(),-70])-60)
    density_min_v,density_min_vpos = density_min_comparison(density_v,-6,density_min_v,density_min_vpos)
    
    for i in range(1,6):
        density_v = (x_div_10 == i).sum()/10
        density_min_v,density_min_vpos =             density_min_comparison(density_v,i,density_min_v,density_min_vpos)
        
        density_v = (x_div_10 == -i).sum()/10
        density_min_v,density_min_vpos =             density_min_comparison(density_v,-i,density_min_v,density_min_vpos)
    
    
    #X_150000.loc[segment_150000, 'highest_count_density_pos_vertically'] = density_max_vpos
    X_150000.loc[segment_150000, 'highest_count_density_vertically'] = density_max_v 
    #X_150000.loc[segment_150000, 'lowest_count_density_pos_vertically'] = density_min_vpos
    X_150000.loc[segment_150000, 'lowest_count_density_vertically'] = density_min_v 
    
    
    
    '''
    max_by_min = 1
    if x.min() != 0:
        max_by_min = x.max()/(x.min())
    else:
        max_by_min = x.max()
    X_150000.loc[segment_150000, 'max/min'] = max_by_min
    '''
    X_150000.loc[segment_150000, 'kurtosis'] = kurtosis(x)
    X_150000.loc[segment_150000, 'skewness'] = skew(x)
    
    
    peaks, _ = find_peaks(x, height=0)
    no_of_peaks = len(peaks)
    X_150000.loc[segment_150000, 'no_of_peaks'] = no_of_peaks
    X_150000.loc[segment_150000, 'no_of_peaks_total'] = (x>0).sum()
    X_150000.loc[segment_150000, 'no_of_valleys_total'] = (x<0).sum()
    X_150000.loc[segment_150000, 'amount_of_silence_total'] = (x==0).sum()
    #X_150000.loc[segment_150000, 'stress_increase_rate'] = x.clip(max=0).sum()    
    
    
    
    xc = pd.Series(seg_150000['acoustic_data'].values)   
    zc = np.fft.fft(xc)
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    #X_150000.loc[segment_150000, 'sum'] = xc.sum()
    #X_150000.loc[segment_150000, 'mad'] = xc.mad()
    #X_150000.loc[segment_150000, 'med'] = xc.median()
    #X_150000.loc[segment_150000, 'abs_mean'] = np.abs(xc).mean()
    X_150000.loc[segment_150000, 'q95'] = np.quantile(xc, 0.95)
    #X_150000.loc[segment_150000, 'q99'] = np.quantile(xc, 0.99)
    X_150000.loc[segment_150000, 'q05'] = np.quantile(xc, 0.05)
    #X_150000.loc[segment_150000, 'q01'] = np.quantile(xc, 0.01)
    #X_150000.loc[segment_150000, 'Rmean'] = realFFT.mean()
    #X_150000.loc[segment_150000, 'Rstd'] = realFFT.std()
    #X_150000.loc[segment_150000, 'Rmax'] = realFFT.max()
    #X_150000.loc[segment_150000, 'Rmin'] = realFFT.min()
    #X_150000.loc[segment_150000, 'Imean'] = imagFFT.mean()
    #X_150000.loc[segment_150000, 'Istd'] = imagFFT.std()
    #X_150000.loc[segment_150000, 'Imax'] = imagFFT.max()
    #X_150000.loc[segment_150000, 'Imin'] = imagFFT.min()
    #peaks, _ = find_peaks(realFFT, height=0)
    #no_of_peaks = len(peaks)
    #X_150000.loc[segment_150000, 'Rno_of_peaks'] = no_of_peaks
    #peaks, _ = find_peaks(imagFFT, height=0)
    #no_of_peaks = len(peaks)
    #X_150000.loc[segment_150000, 'Ino_of_peaks'] = no_of_peaks
    X_150000.loc[segment_150000, 'std_first_50000'] = xc[:50000].std()
    X_150000.loc[segment_150000, 'std_last_50000'] = xc[-50000:].std()
    X_150000.loc[segment_150000, 'std_first_25000'] = xc[:25000].std()
    X_150000.loc[segment_150000, 'std_last_25000'] = xc[-25000:].std()
    X_150000.loc[segment_150000, 'std_first_10000'] = xc[:10000].std()
    X_150000.loc[segment_150000, 'std_last_10000'] = xc[-10000:].std()
    
    def calc_change_rate(x):
        change = (np.diff(x) / x[:-1])#.values
        change = change[np.nonzero(change)[0]]
        change = change[~np.isnan(change)]
        change = change[change != -np.inf]
        change = change[change != np.inf]
        return np.mean(change)
    
    #X_150000.loc[segment_150000, 'sum_abs_total'] = np.sum(np.abs(x))
    #X_150000.loc[segment_150000, 'mean_change_abs'] = np.mean(np.diff(x))
    X_150000.loc[segment_150000, 'mean_change_rate'] = calc_change_rate(x)
    #X_150000.loc[segment_150000, 'abs_max'] = np.abs(x).max()
    #X_150000.loc[segment_150000, 'abs_min'] = np.abs(x).min()
    


# In[8]:



#Preparation of Training dataset ("segment_150000")
#df_trainset_training_150000 = df_trainset.iloc[0:int(0.7*df_trainset.shape[0])]
#data1 = df_trainset.iloc[0:150_000*20]
#data2 = df_trainset.iloc[150_000*50:150_000*100]
#df_trainset_training_150000 = pd.concat([data1, data2], ignore_index=True)
df_trainset_training_150000 = df_trainset.iloc[0:int(int(0.7*df_trainset.shape[0])/150_000)*150_000]
#df_trainset_training_150000 = df_trainset.iloc[0:150_000*30]
#df_trainset_training_150000 = df_trainset.iloc[0:int(int(0.3*df_trainset.shape[0])/150_000)*150_000]

rows_150000 = 150000
segments_150000 = int(np.floor(df_trainset_training_150000.shape[0] / rows_150000))

X_train_150000 = pd.DataFrame(index=range(segments_150000),
                              dtype=np.float64,
                              columns=columns_X_150000
                             )
y_train_150000 = pd.DataFrame(index=range(segments_150000),
                              dtype=np.float64,
                              columns=['time_to_failure']
                             )

for segment_150000 in tqdm_notebook(range(segments_150000)):
    seg_150000 = df_trainset_training_150000.iloc[
        segment_150000*rows_150000:segment_150000*rows_150000+rows_150000
    ]
    y = seg_150000['time_to_failure'].values[-1]
    
    y_train_150000.loc[segment_150000, 'time_to_failure'] = y
    
    addParameters_150000(seg_150000,X_train_150000,segment_150000)


# In[9]:


## #Preparation of Validation dataset ("segment_150000")
df_trainset_validation_150000 = df_trainset.iloc[int(0.7*df_trainset.shape[0]):df_trainset.shape[0]]
#part=2000;df_trainset_validation_150000 = df_trainset.iloc[150_000*part:150_000*(part+100)]
#part=100;df_trainset_validation_150000 = \
#    df_trainset.iloc[int(0.7*df_trainset.shape[0]):int(0.7*df_trainset.shape[0])+150_000*(part)]

'''
#DON'T FORGET TO fill the EMPTY ROWS
#var_new = pd.DataFrame(var).fillna(method='pad')
#var_new = np.mat(var_new)
'''

rows_150000 = 150000
segments_150000 = int(np.floor(df_trainset_validation_150000.shape[0] / rows_150000))    

X_validate_150000 = pd.DataFrame(index=range(segments_150000),
                                 dtype=np.float64,
                                 columns=columns_X_150000
                                )
y_validate_150000 = pd.DataFrame(index=range(segments_150000),
                                 dtype=np.float64,
                                 columns=['time_to_failure']
                                )

for segment_150000 in tqdm_notebook(range(segments_150000)):
    seg_150000 = df_trainset_validation_150000.iloc[
        segment_150000*rows_150000:segment_150000*rows_150000+rows_150000
    ]
    y = seg_150000['time_to_failure'].values[-1]
    
    y_validate_150000.loc[segment_150000, 'time_to_failure'] = y
    
    addParameters_150000(seg_150000,X_validate_150000,segment_150000)


# In[10]:



# Process the X_train_150000 and X_validate_150000 using different Unsupervised ML algorithms

X_train_150000_2 = X_train_150000.copy()
y_train_150000_2 = y_train_150000.copy()
X_validate_150000_2 = X_validate_150000.copy()
y_validate_150000_2 = y_validate_150000.copy()

'''
#Processing the 'count_10_19' data
data = X_train_150000_2['count_10_19'].values
data = 2.0e5*(data - data.min())
X_train_150000_2['count_10_19'] = data

#Processing the 'count_12.5_25' data
data = X_validate_150000_2['count_12.5_25'].values
data = 2.1e4*(data - data.min())
X_validate_150000_2['count_12.5_25'] = data
'''
columns_initial = X_train_150000.copy().columns

default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 10,
                'min_samples': 20,
                'xi': 0.05,
                'min_cluster_size': 0.1}
params = default_base.copy()

# estimate bandwidth for mean shift
bandwidth = cluster.estimate_bandwidth(X_train_150000.copy(), quantile=params['quantile'])

## connectivity matrix for structured Ward
#connectivity = kneighbors_graph(
#    X_train_150000.copy(), n_neighbors=params['n_neighbors'], include_self=False)
## make connectivity symmetric
#connectivity = 0.5 * (connectivity + connectivity.T)

##======= Clustering Algorithms =======##
kmeans_150000 = KMeans(n_clusters=10, random_state=0)
dbscan_150000 = DBSCAN(eps=3, min_samples=2)
gmm_150000 = GaussianMixture(n_components=10)
ms_150000 = MeanShift(bandwidth=bandwidth, bin_seeding=True)
birch_150000 = cluster.Birch(n_clusters=params['n_clusters'])


def preprocess_function_train(X_150000):
    
    
    # KMeans
    print('Processing using KMeans...')
    for i in tqdm_notebook(range(0,len(columns_initial))):
        X_kmeans_150000 = X_150000[columns_initial[i]].values.reshape(-1,1)
        X_150000['kmeans_cluster_'+columns_initial[i]] = kmeans_150000.fit_predict(X_kmeans_150000)
    
    '''
    # DBSCAN
    print('Processing using DBSCAN...')
    for i in tqdm_notebook(range(0,len(columns_initial))):
        X_dbscan_150000 = X_150000[columns_initial[i]].values.reshape(-1,1)
        X_150000['dbscan_cluster_'+columns_initial[i]] = dbscan_150000.fit_predict(X_dbscan_150000)
    '''
    
    # GaussianMixture
    print('Processing using GaussianMixture...')
    for i in tqdm_notebook(range(0,len(columns_initial))):
        X_gmm_150000 = X_150000[columns_initial[i]].values.reshape(-1,1)
        X_150000['gmm_mixture_'+columns_initial[i]] = gmm_150000.fit_predict(X_gmm_150000)
    
    # MeanShift
    print('Processing using MeanShift...')
    for i in tqdm_notebook(range(0,len(columns_initial))):
        X_ms_150000 = X_150000[columns_initial[i]].values.reshape(-1,1)
        X_150000['ms_cluster_'+columns_initial[i]] = ms_150000.fit_predict(X_ms_150000)
    
    '''
    # AgglomerativeClustering - ward
    print('Processing using AgglomerativeClustering - ward...')
    for i in tqdm_notebook(range(0,len(columns_initial))):
        X_ward_150000 = X_150000[columns_initial[i]].values.reshape(-1,1)
        ward_150000 = cluster.AgglomerativeClustering(
            n_clusters=params['n_clusters'], linkage='ward',
            connectivity=connectivity)
        X_150000['agg_ward_cluster_'+columns_initial[i]] = ward_150000.fit_predict(X_ward_150000)
    '''
    
    '''
    # AgglomerativeClustering - average_linkage
    print('Processing using AgglomerativeClustering - average_linkage...')
    for i in tqdm_notebook(range(0,len(columns_initial))):
        X_average_linkage_150000 = X_150000[columns_initial[i]].values.reshape(-1,1)
        average_linkage_150000 = cluster.AgglomerativeClustering(
            linkage="average", affinity="cityblock",
            n_clusters=params['n_clusters'], connectivity=connectivity)
        X_150000['agg_average_linkage_cluster_'+columns_initial[i]] = \
            average_linkage_150000.fit_predict(X_average_linkage_150000)
    '''

    '''
    # SpectralClustering
    print('Processing using SpectralClustering...')
    for i in tqdm_notebook(range(0,len(columns_initial))):
        X_spectral_150000 = X_150000[columns_initial[i]].values.reshape(-1,1)
        spectral_150000 = cluster.SpectralClustering(
            n_clusters=params['n_clusters'], eigen_solver='arpack',
            affinity="nearest_neighbors")
        X_150000['spectral_cluster_'+columns_initial[i]] = \
            spectral_150000.fit_predict(X_spectral_150000)
    '''
   
    '''
    # MiniBatchKMeans
    print('Processing using MiniBatchKMeans...')
    for i in tqdm_notebook(range(0,len(columns_initial))):
        X_two_means_150000 = X_150000[columns_initial[i]].values.reshape(-1,1)
        two_means_150000 = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
        X_150000['two_means_cluster_'+columns_initial[i]] = \
            two_means_150000.fit_predict(X_two_means_150000)
    '''

    ''' 
    # OPTICS #--> AttributeError: module 'sklearn.cluster' has no attribute 'OPTICS'
    print('Processing using OPTICS...')
    for i in tqdm_notebook(range(0,len(columns_initial))):
        X_optics_150000 = X_150000[columns_initial[i]].values.reshape(-1,1)
        optics_150000 = cluster.OPTICS(min_samples=params['min_samples'],
                                         xi=params['xi'],
                                        min_cluster_size=params['min_cluster_size'])
        X_150000['optics_cluster_'+columns_initial[i]] = \
            optics_150000.fit_predict(X_optics_150000)
    '''

    '''
    # AffinityPropagation # extremely slow!!
    print('Processing using AffinityPropagation...')
    for i in tqdm_notebook(range(0,len(columns_initial))):
        X_affinity_propagation_150000 = X_150000[columns_initial[i]].values.reshape(-1,1)
       affinity_propagation_150000 = cluster.AffinityPropagation(
            damping=params['damping'], preference=params['preference'])
        X_150000['affinity_propagation_cluster_'+columns_initial[i]] = \
            affinity_propagation_150000.fit_predict(X_affinity_propagation_150000)
    '''
    
    
    # Birch
    print('Processing using Birch...')
    for i in tqdm_notebook(range(0,len(columns_initial))):
        X_birch_150000 = X_150000[columns_initial[i]].values.reshape(-1,1)
        X_150000['birch_cluster_'+columns_initial[i]] = birch_150000.fit_predict(X_birch_150000)
    
    
    # replacing na values
    print("Filling in places with 'NaN'...")
    for i in tqdm_notebook(range(0,len(X_150000.columns))):
        X_150000[X_150000.columns[i]].fillna( method ='ffill', inplace = True)


def preprocess_function_test(X_150000):
    
     # KMeans
    print('Processing using KMeans...')
    for i in tqdm_notebook(range(0,len(columns_initial))):
        X_kmeans_150000 = X_150000[columns_initial[i]].values.reshape(-1,1)
        X_150000['kmeans_cluster_'+columns_initial[i]] = kmeans_150000.predict(X_kmeans_150000)
    
    '''
    # DBSCAN
    print('Processing using DBSCAN...')
    for i in tqdm_notebook(range(0,len(columns_initial))):
        X_dbscan_150000 = X_150000[columns_initial[i]].values.reshape(-1,1)
        X_150000['dbscan_cluster_'+columns_initial[i]] = dbscan_150000.fit_predict(X_dbscan_150000)
    '''
    
    # GaussianMixture
    print('Processing using GaussianMixture...')
    for i in tqdm_notebook(range(0,len(columns_initial))):
        X_gmm_150000 = X_150000[columns_initial[i]].values.reshape(-1,1)
        X_150000['gmm_mixture_'+columns_initial[i]] = gmm_150000.predict(X_gmm_150000)
    
    # MeanShift
    print('Processing using MeanShift...')
    for i in tqdm_notebook(range(0,len(columns_initial))):
        X_ms_150000 = X_150000[columns_initial[i]].values.reshape(-1,1)
        X_150000['ms_cluster_'+columns_initial[i]] = ms_150000.predict(X_ms_150000)
    
    # Birch
    print('Processing using Birch...')
    for i in tqdm_notebook(range(0,len(columns_initial))):
        X_birch_150000 = X_150000[columns_initial[i]].values.reshape(-1,1)
        X_150000['birch_cluster_'+columns_initial[i]] = birch_150000.predict(X_birch_150000)
    
    # replacing na values
    print("Filling in places with 'NaN'...")
    for i in tqdm_notebook(range(0,len(X_150000.columns))):
        X_150000[X_150000.columns[i]].fillna( method ='ffill', inplace = True)


        
preprocess_function_train(X_train_150000_2)
preprocess_function_test(X_validate_150000_2) 



# In[11]:


#X_train_150000_2['kmeans_cluster'] = kmeans_150000_2.labels_
#X_validate_150000_2['kmeans_cluster'] = kmeans_150000_2.predict(X_validate_kmeans_150000_2)


# In[12]:



X_train = X_train_150000_2.copy()
y_train = y_train_150000_2.copy()
X_validate = X_validate_150000_2.copy()
y_validate = y_validate_150000_2.copy()


# In[13]:


X_train.head()


# In[14]:


X_validate.head()


# In[15]:


y_train.head()


# In[16]:


'''
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
'''
'''
# Import the model we are using
from sklearn.tree import export_graphviz
from IPython import display
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 2, random_state = 42, verbose=1)
#m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)


# Train the model on training data
rf.fit(X_train, y_train.values.flatten())
'''


# In[17]:


'''
from IPython.display import SVG
from graphviz import Source
#graph = Source( tree.export_graphviz(dtreg, out_file=None, feature_names=X.columns))
#SVG(graph.pipe(format='svg'))

# Display a tree
str_tree = export_graphviz(rf.estimators_[0], #random_forest.estimators_ => decision trees
   out_file=None, 
   feature_names=X_train.columns, # column names
   filled=True,        
   special_characters=True, 
   rotate=True, 
   precision=6
   )
graph = Source(str_tree)
SVG(graph.pipe(format='svg'))

#display.display(str_tree)
'''


# In[18]:



'''
# Visualisation of Features' Importance
importance = rf.feature_importances_
print(importance)
print(len(list(X_train)))
x = [i for i in range(0,len(list(X_train)))]
y = importance
plt.bar(x,y)
plt.show()
'''


# In[19]:


'''
svm = NuSVR()
svm.fit(X_train_scaled, y_train.values.flatten())
y_pred = svm.predict(X_train_scaled)
'''
'''
y_train_pred_model_rf = rf.predict(X_train)
y_pred_model_rf = rf.predict(X_validate)
'''


# In[20]:


'''
PlotLinearGraph_Actual_vs_Pred_absolute(y_train.values.flatten(),y_train_pred_model_rf)
PlotLinearGraph_Actual_vs_Pred_absolute(y_validate.values.flatten(),y_pred_model_rf)
'''


# In[21]:


# CatBoost
from catboost import CatBoostRegressor

model_CatBoost = CatBoostRegressor(iterations=3000,
                                   silent=True,
                                   loss_function='MAE',
                                   boosting_type='Ordered',
                                   #depth=10, #max_depth=16 -> 'Memory Allocation Problem'
                                   #verbose=1,
                                  )

#num_folds = 3
#tss = TimeSeriesSplit(n_splits=10)
#cross_validate_results = cross_validate(model_CatBoost, X_train, y_train.values.flatten(), n_jobs=-1, return_estimator= True, return_train_score=True, cv=tss, verbose=50, scoring="neg_mean_absolute_error" )

model_CatBoost.fit(X_train, y_train.values.flatten(),
                   eval_set=(X_validate, y_validate.values.flatten()),plot=True)
#print(model_CatBoost.best_score_)
'''
OUTPUT = {'learn': {'MAE': 6.035374141450229}} #for segments_150000 - max,min,avg,std parameters
Validation MAE Score for 1000 iterations = 2.636
'''


# In[22]:


#cross_validate_results.keys()


# In[23]:


#num_folds=10
#for i in range(num_folds):
#    print("Fold: {} Training MAE: {}".format(i,-cross_validate_results['train_score'][i]))
#    print("Fold: {} Validation MAE: {}".format(i,-cross_validate_results['test_score'][i]))


# In[24]:


#model_CatBoost_new = cross_validate_results['estimator'][9]


# In[25]:


print(model_CatBoost.best_score_)
print(model_CatBoost.tree_count_)
print(model_CatBoost.get_params())


# In[26]:


import catboost
importance = model_CatBoost.get_feature_importance()
x = np.array([list(X_train)[i] for i in range(0,len(list(X_train)))])
y = importance
plt.bar(x,y)
plt.show()

#print(y[np.where(x=='stress_increase_rate_total')[0][0]])
#print(y[np.where(x=="mean_Hdistance_bw_higherthan_10_peak_pts")[0][0]])
#print(y[np.where(x=="mean_Hdistance_bw_lowerthan_n10_valley_pts")[0][0]])


# In[27]:



y_train_pred_model_CatBoost = model_CatBoost.predict(X_train)
y_pred_model_CatBoost = model_CatBoost.predict(X_validate)


# In[28]:


# Plot Actual vs Prediction Graph (Absolute) before fine-tuning
#PlotLinearGraph_Actual_vs_Pred(y_validate.values.flatten(),y_pred_model_CatBoost)
PlotLinearGraph_Actual_vs_Pred_absolute(y_train.values.flatten(),y_train_pred_model_CatBoost)
PlotLinearGraph_Actual_vs_Pred_absolute(y_validate.values.flatten(),y_pred_model_CatBoost)


# In[29]:


# Linear Model's Regression Methods

#model_regressor = linear_model.Lasso(alpha=0.1)
model_regressor = linear_model.LassoLarsCV(cv=5)
#model_regressor = linear_model.Ridge(alpha=.5)
model_regressor.fit(X_train, y_train.values.flatten())


# In[30]:


y_train_pred_model_regressor = model_regressor.predict(X_train)
y_pred_model_regressor = model_regressor.predict(X_validate)


# In[31]:


PlotLinearGraph_Actual_vs_Pred_absolute(y_train.values.flatten(),y_train_pred_model_regressor)
PlotLinearGraph_Actual_vs_Pred_absolute(y_validate.values.flatten(),y_pred_model_regressor)


# In[32]:



#SVR
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_validate_scaled = scaler.transform(X_validate)
model_SVM = NuSVR()
model_SVM.fit(X_train_scaled, y_train.values.flatten())


# In[33]:


X_train_scaled


# In[34]:


y_train_pred_model_SVM = model_SVM.predict(X_train_scaled)
y_pred_model_SVM = model_SVM.predict(X_validate_scaled)


# In[35]:



PlotLinearGraph_Actual_vs_Pred_absolute(y_train.values.flatten(),y_train_pred_model_SVM)
PlotLinearGraph_Actual_vs_Pred_absolute(y_validate.values.flatten(),y_pred_model_SVM)


# In[36]:


'''
## Neural Network Begins ##

#Libraries for neural net
import keras
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

#Model parameters

kernel_init = 'he_normal'
input_size = len(X_train.columns)

### Neural Network ###

# Model architecture: A very simple shallow Neural Network 
model_NN = Sequential()
model_NN.add(Dense(16, input_dim = input_size)) 
model_NN.add(Activation('linear'))
model_NN.add(BatchNormalization())
model_NN.add(Dropout(0.5))
model_NN.add(Dense(32))    
model_NN.add(Activation('tanh'))
model_NN.add(BatchNormalization())
model_NN.add(Dropout(0.5))
model_NN.add(Dense(1))    
model_NN.add(Activation('linear'))

#compile the model
optim = optimizers.Adam(lr = 0.001)
model_NN.compile(loss = 'mean_absolute_error', optimizer = optim)

#Callbacks
#csv_logger = CSVLogger('log.csv', append=True, separator=';')
#best_model = ModelCheckpoint("model.hdf5", save_best_only=True, period=3)
#restore_best = EarlyStopping(monitor='val_loss', verbose=2, patience=100, restore_best_weights=True)

model_NN.fit(x=X_train, y=y_train.values.flatten(), batch_size=64, epochs=200, verbose=2,\
          validation_data=(X_validate,y_validate.values.flatten())) #callbacks=[csv_logger, best_model],
### Neural Network End ###

#nn_predictions = model.predict(Xtest, verbose = 2, batch_size = 64)
#submission['time_to_failure'] = nn_predictions
#submission.to_csv('submission.csv')
'''


# In[37]:


'''
y_train_pred_model_NN = model_NN.predict(X_train, verbose = 2, batch_size = 64)
y_pred_model_NN = model_NN.predict(X_validate, verbose = 2, batch_size = 64)
'''


# In[38]:


'''
PlotLinearGraph_Actual_vs_Pred_absolute(y_train.values.flatten(),y_train_pred_model_NN)
PlotLinearGraph_Actual_vs_Pred_absolute(y_validate.values.flatten(),y_pred_model_NN)
'''


# In[39]:


'''
def format_to_lstm(df):
    X = np.array(df)
    return np.reshape(X, (X.shape[0], 1, X.shape[1]))
train_X = X_train.copy()#y_train_pred_model_CatBoost.reshape(-1,1).copy()
train_y = y_train.copy()
# reshape input to be 3D [samples, timesteps, features] (batch_size, timesteps, input_dim)
train_X = format_to_lstm(train_X)
#train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
#time_step=50
#train_X = np.stack(np.split(train_X, time_step), axis=1)
#test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

validate_X = X_validate.copy()#y_pred_model_CatBoost.reshape(-1,1).copy()
validate_y = y_validate.copy()
validate_X = format_to_lstm(validate_X)
'''


# In[40]:


'''
print(X_train.shape)
print(train_X.shape)
'''


# In[41]:


'''
# LSTM using Keras

import keras
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.models import Sequential
from keras import optimizers
# design network
model_LSTM = Sequential()
model_LSTM.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model_LSTM.add(Dense(1))
model_LSTM.compile(loss='mae', optimizer='adam')

# fit network
def my_lstm_fit(X,y,my_epochs,my_batch_size,model):
    #history = model.fit(train_X, train_y, epochs=50, batch_size=150_000, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    history = model.fit(X, y, epochs=my_epochs, batch_size=my_batch_size, verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    #pyplot.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

my_lstm_fit(train_X,train_y,1000,50,model_LSTM)
'''


# In[42]:


'''
y_train_pred_model_LSTM = model_LSTM.predict(train_X)
y_pred_model_LSTM = model_LSTM.predict(validate_X)
'''


# In[43]:


'''
PlotLinearGraph_Actual_vs_Pred_absolute(y_train.values.flatten(),y_train_pred_model_LSTM)
PlotLinearGraph_Actual_vs_Pred_absolute(y_validate.values.flatten(),y_pred_model_LSTM)
'''


# In[44]:


'''
# Random ttf generation model
y_train_pred_model_RANDOM1 = np.random.choice(y_train.values.flatten(),len(X_train))
y_pred_model_RANDOM1 = np.random.choice(y_train.values.flatten(),len(X_validate))
'''


# In[45]:


'''
PlotLinearGraph_Actual_vs_Pred_absolute(y_train.values.flatten(),y_train_pred_model_RANDOM1)
PlotLinearGraph_Actual_vs_Pred_absolute(y_validate.values.flatten(),y_pred_model_RANDOM1)
'''


# In[46]:


'''
# Blending of Models - 1

# Model Catboost
#y_train_pred_model_CatBoost
#y_pred_model_CatBoost

#y_train_pred_model_regressor
#y_pred_model_regressor

c1=0.6;c2=0.1;c3=0.3;c4=0;c5=0.1
correction = 0#5.75;
y_train_pred_blend1 = c1*y_train_pred_model_CatBoost.copy()\
                      + c2*y_train_pred_model_regressor.copy()\
                      + c3*y_train_pred_model_LSTM.flatten().copy()
                      #+ c4*y_train_pred_model_SVM.copy()\
                      #+ c5*y_train_pred_model_RANDOM1.copy()

y_pred_blend1 = c1*y_pred_model_CatBoost.copy()\
                + c2*y_pred_model_regressor.copy()\
                + c3*y_pred_model_LSTM.flatten().copy()
                #+ c4*y_pred_model_SVM.copy()\
                #+ c5*y_pred_model_RANDOM1.copy()

y_train_pred_blend1 = y_train_pred_blend1 - np.array([correction]*len(y_train_pred_blend1))
y_pred_blend1 = y_pred_blend1 - np.array([correction]*len(y_pred_blend1))
'''


# In[47]:


'''
PlotLinearGraph_Actual_vs_Pred_absolute(y_train.values.flatten(),y_train_pred_blend1)
PlotLinearGraph_Actual_vs_Pred_absolute(y_validate.values.flatten(),y_pred_blend1)
#PlotLinearGraph_Actual_vs_Pred_absolute(np.arange(len(y_validate.values.flatten())),y_validate.values.flatten())
#PlotLinearGraph_Actual_vs_Pred_absolute(np.arange(len(y_validate.values.flatten())),y_pred_blend1)
'''


# In[48]:


'''
score_arr = []
score_min = 9999
c1_arr = []
c1_min = -1
for i in range(0,11):
    c1=0.0 + 0.1*i;
    c2=1.0 - 0.1*i;
    #y_train_pred_blend1 = c1*y_train_pred_model_CatBoost.copy()+c2*y_train_pred_model_regressor.copy()
    y_pred_blend1 = c1*y_pred_model_CatBoost.copy()+c2*y_pred_model_regressor.copy()
    #y_actual = y_train.values.flatten().copy()
    y_actual = y_validate.values.flatten().copy()
    y_pred = y_pred_blend1.copy()
    score_mae = mean_absolute_error(y_actual, y_pred)
    #score_r2 = r2_score(y_actual, y_pred)
    #score_mse = mean_squared_error(y_actual, y_pred)
    score_arr.append(score_mae)
    c1_arr.append(c1)
    if score_mae<score_min:
        score_min = score_mae
        c1_min = c1
        
x = np.array(c1_arr)
y = np.array(score_arr)
plt.plot(x,y,'g.')
plt.show()
print('c1_min: '+str(c1_min)+', score_min: '+str(score_min))
#TRAIN - c1_min: 1.0, score_min: 1.974
#VALIDATE - c1_min: 1.0, score_min: 1.974
'''


# In[49]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')


# In[50]:


X_test = pd.DataFrame(dtype=np.float64, index=submission.index) #columns=X_train.columns,)


# In[51]:


for seg_id in tqdm_notebook(X_test.index):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
        
    addParameters_150000(seg,X_test,seg_id)

preprocess_function_test(X_test)


# In[52]:


X_test.head()


# In[53]:


#X_test_scaled = scaler.transform(X_test)
#submission['time_to_failure'] = svm.predict(X_test_scaled)
#submission['time_to_failure'] = rf.predict(X_test)
submission['time_to_failure'] = model_CatBoost.predict(X_test)
submission.to_csv('submission.csv')


# In[54]:


# visualise missing values
#import missingno as msno
#msno.matrix(df_trainset)


# In[55]:


'''
n=150_000
df_chosen = df_trainset1[0:n].copy()
x = [x for x in df_chosen['time_elapsed'].values]
#y_acoustic_data = X_validate['var/time_segment'].values
#y_time_to_failure = y_train.values.flatten()
y_acoustic_data = df_chosen['acoustic_data'].values
y_time_to_failure = df_chosen['time_to_failure'].values
#y_pred_time_to_failure = y_train_pred_model_CatBoost
plt.plot(x,[1*y for y in y_acoustic_data],'b-')
plt.show()
plt.plot(x,[1*t for t in y_time_to_failure],'r-') #magnify the time_to_failure 100 times for visualisation purpose
#plt.plot(x,[1*t for t in y_pred_time_to_failure],'g-') #magnify the time_to_failure 100 times for visualisation purpose
plt.show()
'''


# In[56]:



#VISUALISATION OF df_trainset - 3
part=35;start=150_000*part;finish=150_000*part+150_000*4
df_chosen = df_trainset[start:finish]
#x = np.array([x for x in df_chosen['time_to_failure'].values])
x = np.array([x for x in range(0,len(df_chosen))])
y_acoustic_data = df_chosen['acoustic_data'].values
y_time_to_failure = df_chosen['time_to_failure'].values
plt.plot(x,[1*y for y in y_acoustic_data],'b-')
plt.show()
plt.plot(x,[1*t for t in y_time_to_failure],'r-') #magnify the time_to_failure 100 times for visualisation purpose
plt.show()
#print('Slope ttf vs S.No.: '+str((y_time_to_failure[-1]-y_time_to_failure[0])/(len(y_time_to_failure))))


# In[57]:


# Display the list of column names of X_train
X_train.columns


# In[58]:



## Data Visualisation of X_train
dfx_chosen = X_train.copy()
dfy_chosen = y_train.copy()
#x = np.array([x for x in dfy_chosen['time_to_failure'].values])
#x = np.array([x for x in range(0,len(dfx_chosen))])
for i in range(0,len(X_train.columns)):
    x_title=str(X_train.columns[i])
    y_acoustic_data = dfx_chosen[x_title].values #dfx_chosen['mode_count'].values
    y_time_to_failure = dfy_chosen['time_to_failure'].values
    plt.plot(y_acoustic_data,[1*y for y in y_time_to_failure],'b.')
    plt.xlabel(x_title)
    plt.ylabel('time_to_failure')
    plt.show()
    #plt.plot(x,[1*y for y in y_acoustic_data],'b-')
    #plt.show()
    #plt.plot(x,[1*t for t in y_time_to_failure],'r-') #magnify the time_to_failure 100 times for visualisation purpose
    #plt.show()
#x = dfx_chosen['count_0_9'].values
#y = dfy_chosen['time_to_failure'].values
#plt.scatter(x, y, c=kmeans_150000_2.labels_)

#print(y_time_to_failure.max())
#print(round(-np.diff(y_time_to_failure[0:10]).mean(),4))


# In[59]:


'''
//Analysis of above Data Visualisation//

Parameters of interest: std,max,
print(np.diff(y_time_to_failure[0:30]).mean()) # -0.03896207078551724 = -0.0390
In order of importance:-
 - mean_Hdistance_bw_higherthan_0_peak_pts 
 - mean_Hdistance_bw_higherthan_10_peak_pts
 - mean_Hdistance_bw_lowerthan_n0_valley_pts
 - no_of_peaks
 - no_of_valleys_total
 - no_of_peaks_total
 - highest_count_density_vertically
 - lowest_count_density_vertically
 - count_0_9
 - count_10_19
 - count_n10_n19
 - q95
 - q05
 - mean_change_rate
 - amount_of_silence_total
 - std>=10 => ttf<=10
 - count_20_29
 - max>=3000=>ttf=0-0.5,max>=500=>ttf=0-10
 - count_30_39
 - count_40_49
 - min (similar approach as that for 'max')
 - Mean>=4.8 => ttf<=13;Mean<=3.9=>ttf=10-17,1-8
'''


# In[60]:




## Data Visualisation of X_test
y_test_values = model_CatBoost.predict(X_test).copy()[0:30]
dfx_chosen = X_test[0:30].copy()
#dfy_chosen = y_test.copy()
#x = np.array([x for x in dfy_chosen['time_to_failure'].values])
x = np.array([x for x in range(0,len(dfx_chosen))])
#y_acoustic_data = dfx_chosen['count_0_9'].values #dfx_chosen['mode_count'].values
y_time_to_failure = y_test_values
#plt.plot(x,[1*y for y in y_acoustic_data],'b-')
#plt.show()
plt.plot(x,[1*t for t in y_time_to_failure],'r-') #magnify the time_to_failure 100 times for visualisation purpose
plt.show()
#x = dfx_chosen['count_0_9'].values
#y = dfy_chosen['time_to_failure'].values
#plt.scatter(x, y, c=kmeans_150000_2.labels_)

#print(y_time_to_failure.max())
#print(round(-np.diff(y_time_to_failure[0:10]).mean(),4))


# In[61]:


## Important Concepts (ROUGH WORK)
#========================================#
#print(skew([1,1,2,2,2,3]))
#print(kurtosis([1,1,2,2,2,3]))
#f = [1,4,9,16,25,36]
#print(np.gradient(f,edge_order=2))
#print(np.gradient(np.gradient(f,edge_order=2)))
'''
v = [-1,2,3]
print(np.abs(v))
v = np.add(np.array([1,3,5]),np.array([-1]*len(v)))
print(v)
print(v*3)
print(6/17)
print(np.max(np.array([1,2,8,4,7])))
print(1e7)
np.std([1,2,1])
a = [1,2,3,4,5]
print(a[0:10])
print(int(629145480/150_000)) # = 4194
y = x.clip(min=0)
print(y)
x = np.array([0,1,2,-1,-3,5])
a = np.mean(x)
print(a)
print(a*len(x))
z = np.sum(x)
print(z)
df_experiment = pd.DataFrame(list(zip(x,y)), columns=['col1','col2'])
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = x*2
print(y)
x = np.array([20,3,12,-10,34,56,15])
a = x>15
print(a.sum())
r,pvalue = pearsonr(x,y)
print(r)
#avg 	std 	min 	kurtosis 	skewness 	no_of_peaks 	stress_increase_rate
r,_ = pearsonr(X_train['skewness'],y_train.values.flatten())
print(r)
r,_ = pearsonr(X_train['stress_increase_rate'],y_train.values.flatten())
print(r)
r,_ = pearsonr(X_train['kurtosis'],y_train.values.flatten())
print(r)
r,_ = pearsonr(X_train['no_of_peaks_total'],y_train.values.flatten())
print(r)
x = np.array([20,3,12,-10,34,56,15])
print(x.clip(max=10))
B=150_000;b=4096
print(int(B/b))
print(x[6])
x[6]=30
print(x[6])
print(x)
print(np.minimum([1,2,3],[0,-3,5]))
print(mode(x).mode[0])
A = np.array([20,3,12,-10,34,56,15])
np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=A)
numpy.fromiter((<some_func>(x) for x in <something>),<dtype>,<size of something>)
Mode = 3Median - 2Mean

A = np.array([20,3,12,-10,34,56,15])
print(A*10)
A = np.array([20,3,12,-10,34,56,15])
A_by_10 = A//10
print(A_by_10)
print(mode(A_by_10).count[0])
print((A_by_10 == 1).sum())
A = np.array([20,3,12,-10,34,56,15])
A_by_10 = A//10
print(A_by_10)
print(A_by_10 == 1)
print(np.where(A_by_10 == 1)[0])
print('len([]): '+str(len(np.array([]))))
a=b=c=9
print(a)
print(b)
print(c)
a = np.array([-1,3,4,-2,10,0,-9,4,5,7])
a1 = a.clip(min=0)
print(a1)
b = np.arange(0,a1.size)
sum_of_b = b.sum() + 1
print(b)
print(a1.size)
print(a1*b)
print(sum_of_b)
print((a1*b).sum()/sum_of_b)
print(182/46)
print(np.max([1,2,7,30]))
x = [1,3,2,10,4]
np.abs(np.diff(x, n=1)) # array([2, 1, 8, 6])
np.sum(np.abs(np.diff(x,n=1)))
print((x>9))
print(np.where((x>9)!=False)[0])
print(np.diff(np.where((x>9)!=False)[0],n=1))
print(np.size(np.diff(np.where((x>9)!=False)[0],n=1)))
print(np.mean(np.diff(np.where((x>9)!=False)[0],n=1)))
indices_count_i_j_title = 'indices_count_0_9'
print(indices_count_i_j_title+'_mean')
df = pd.DataFrame(data=np.array([1,2,3,2,7,10]),columns=[indices_count_i_j_title+'_mean',])
df
print((x*x).sum())
print(np.log(x))
x = np.array([10000,-10000])
#use np.log() to reduce fluctuations in acoustic data
x = np.log10(np.abs(x)+np.array([1]*np.size(x)))
print(x)
x = np.array([1,3,2,10,4])
print((x<4) * (x>1))
0.0390*413
print(x-1)
df_expt['value'] = np.array([1]*len(x))
#Processing the y_acoustic_data
n = y_acoustic_data
s = 1/n
d = -np.log10(1-s)
#y_acoustic_data = y_acoustic_data - y_acoustic_data.min()
#y_acoustic_data = y_acoustic_data*2e5
d = d - d.min()
d = d*2.1e4
print(1/x)
def calc_change_rate_expt(x):
    change = (np.diff(x) / x[:-1])#.values
    change = change[np.nonzero(change)[0]]
    change = change[~np.isnan(change)]
    change = change[change != -np.inf]
    change = change[change != np.inf]
    return np.mean(change)

x = np.array([1,3,0,2,10,4])
#print(calc_change_rate_expt(x))
print(np.diff(x))
print(x[:-1])
change_expt = (np.diff(x) / x[:-1])
print(change_expt)
print(np.nonzero(change_expt)[0])
print(~np.isnan(change_expt))
print(change_expt != np.inf)
print(change_expt[change_expt != np.inf])

# Import the model we are using
from sklearn.tree import export_graphviz
from IPython import display
from IPython.display import SVG
from graphviz import Source
from sklearn.ensemble import RandomForestRegressor
x_expt =np.array([36,4,9,16,25, 12,23,34,45,56, -1,-2,-3,-4,-5]).reshape(-1,1)
y_expt =np.array([1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3])
# Instantiate model with 1000 decision trees
m = RandomForestRegressor(n_estimators=1000, bootstrap=False, n_jobs=-1)
# Train the model on training data
m.fit(x_expt, y_expt)
##graph = Source( tree.export_graphviz(dtreg, out_file=None, feature_names=X.columns))
##SVG(graph.pipe(format='svg'))
# Display a tree
str_tree = export_graphviz(m.estimators_[0], #random_forest.estimators_ => decision trees
   out_file=None, 
   #feature_names=x_expt.columns, # column names
   filled=True,        
   special_characters=True, 
   rotate=True, 
   precision=6
   )
graph = Source(str_tree)
##SVG(graph.pipe(format='svg'))
##display.display(str_tree)
m.predict(np.array([49]).reshape(-1,1))
'''


# In[62]:


'''  
# estimate bandwidth for mean shift
bandwidth_train = cluster.estimate_bandwidth(X_train_150000, quantile=0.3)
bandwidth_validate = cluster.estimate_bandwidth(X_validate_150000, quantile=0.3)

# connectivity matrix for structured Ward
connectivity_train = kneighbors_graph(
    X_train_150000, n_neighbors=params['n_neighbors'], include_self=False)
connectivity_validate = kneighbors_graph(
    X_validate_150000, n_neighbors=params['n_neighbors'], include_self=False)
# make connectivity symmetric
connectivity_train = 0.5 * (connectivity_train + connectivity_train.T)
connectivity_validate = 0.5 * (connectivity_validate + connectivity_validate.T)
'''

'''
# KMeans
print('Processing using KMeans...')
for i in tqdm_notebook(range(0,len(columns_initial))):
    X_train_kmeans_150000_2 = X_train_150000_2[columns_initial[i]].values.reshape(-1,1)
    X_validate_kmeans_150000_2 = X_validate_150000_2[columns_initial[i]].values.reshape(-1,1)
    kmeans_150000_2 = KMeans(n_clusters=10, random_state=0)
    X_train_150000_2['kmeans_cluster_'+columns_initial[i]] = kmeans_150000_2.fit_predict(X_train_kmeans_150000_2)
    X_validate_150000_2['kmeans_cluster_'+columns_initial[i]] = kmeans_150000_2.fit_predict(X_validate_kmeans_150000_2)
'''
'''
# DBSCAN
print('Processing using DBSCAN...')
for i in tqdm_notebook(range(0,len(columns_initial))):
    X_train_dbscan_150000_2 = X_train_150000_2[columns_initial[i]].values.reshape(-1,1)
    X_validate_dbscan_150000_2 = X_validate_150000_2[columns_initial[i]].values.reshape(-1,1)
    dbscan_150000_2 = DBSCAN(eps=3, min_samples=2)
    X_train_150000_2['dbscan_cluster_'+columns_initial[i]] = dbscan_150000_2.fit_predict(X_train_dbscan_150000_2)
    X_validate_150000_2['dbscan_cluster_'+columns_initial[i]] = dbscan_150000_2.fit_predict(X_validate_dbscan_150000_2)

# GaussianMixture
print('Processing using GaussianMixture...')
for i in tqdm_notebook(range(0,len(columns_initial))):
    X_train_gmm_150000_2 = X_train_150000_2[columns_initial[i]].values.reshape(-1,1)
    X_validate_gmm_150000_2 = X_validate_150000_2[columns_initial[i]].values.reshape(-1,1)
    gmm_150000_2 = GaussianMixture(n_components=10)
    X_train_150000_2['gmm_mixture_'+columns_initial[i]] = gmm_150000_2.fit_predict(X_train_gmm_150000_2)
    X_validate_150000_2['gmm_mixture_'+columns_initial[i]] = gmm_150000_2.fit_predict(X_validate_gmm_150000_2)

# MeanShift
print('Processing using MeanShift...')
for i in tqdm_notebook(range(0,len(columns_initial))):
    X_train_ms_150000_2 = X_train_150000_2[columns_initial[i]].values.reshape(-1,1)
    X_validate_ms_150000_2 = X_validate_150000_2[columns_initial[i]].values.reshape(-1,1)
    ms_150000_2 = MeanShift(bandwidth=bandwidth_train, bin_seeding=True)
    X_train_150000_2['ms_cluster_'+columns_initial[i]] = ms_150000_2.fit_predict(X_train_ms_150000_2)
    ms_150000_2 = MeanShift(bandwidth=bandwidth_validate, bin_seeding=True)
    X_validate_150000_2['ms_cluster_'+columns_initial[i]] = ms_150000_2.fit_predict(X_validate_ms_150000_2)


# AgglomerativeClustering - ward
print('Processing using AgglomerativeClustering - ward...')
for i in tqdm_notebook(range(0,len(columns_initial))):
    X_train_ward_150000_2 = X_train_150000_2[columns_initial[i]].values.reshape(-1,1)
    X_validate_ward_150000_2 = X_validate_150000_2[columns_initial[i]].values.reshape(-1,1)
    ward_150000_2 = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity_train)
    X_train_150000_2['agg_ward_cluster_'+columns_initial[i]] = ward_150000_2.fit_predict(X_train_ward_150000_2)
    ward_150000_2 = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity_validate)
    X_validate_150000_2['agg_ward_cluster_'+columns_initial[i]] = ward_150000_2.fit_predict(X_validate_ward_150000_2)
'''
    
'''
# AgglomerativeClustering - average_linkage
print('Processing using AgglomerativeClustering - average_linkage...')
for i in tqdm_notebook(range(0,len(columns_initial))):
    X_train_average_linkage_150000_2 = X_train_150000_2[columns_initial[i]].values.reshape(-1,1)
    X_validate_average_linkage_150000_2 = X_validate_150000_2[columns_initial[i]].values.reshape(-1,1)
    average_linkage_150000_2 = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity_train)
    X_train_150000_2['agg_average_linkage_cluster_'+columns_initial[i]] = \
        average_linkage_150000_2.fit_predict(X_train_average_linkage_150000_2)
    average_linkage_150000_2 = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity_validate)
    X_validate_150000_2['agg_average_linkage_cluster_'+columns_initial[i]] = \
        average_linkage_150000_2.fit_predict(X_validate_average_linkage_150000_2)
'''

'''
# SpectralClustering
print('Processing using SpectralClustering...')
for i in tqdm_notebook(range(0,len(columns_initial))):
    X_train_spectral_150000_2 = X_train_150000_2[columns_initial[i]].values.reshape(-1,1)
    X_validate_spectral_150000_2 = X_validate_150000_2[columns_initial[i]].values.reshape(-1,1)
    spectral_150000_2 = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
    X_train_150000_2['spectral_cluster_'+columns_initial[i]] = \
        spectral_150000_2.fit_predict(X_train_spectral_150000_2)
    X_validate_150000_2['spectral_cluster_'+columns_initial[i]] = \
        spectral_150000_2.fit_predict(X_validate_spectral_150000_2)
'''
'''
# MiniBatchKMeans
print('Processing using MiniBatchKMeans...')
for i in tqdm_notebook(range(0,len(columns_initial))):
    X_train_two_means_150000_2 = X_train_150000_2[columns_initial[i]].values.reshape(-1,1)
    X_validate_two_means_150000_2 = X_validate_150000_2[columns_initial[i]].values.reshape(-1,1)
    two_means_150000_2 = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    X_train_150000_2['two_means_cluster_'+columns_initial[i]] = \
        two_means_150000_2.fit_predict(X_train_two_means_150000_2)
    X_validate_150000_2['two_means_cluster_'+columns_initial[i]] = \
        two_means_150000_2.fit_predict(X_validate_two_means_150000_2)
'''

''' 
# OPTICS #--> AttributeError: module 'sklearn.cluster' has no attribute 'OPTICS'
print('Processing using OPTICS...')
for i in tqdm_notebook(range(0,len(columns_initial))):
    X_train_optics_150000_2 = X_train_150000_2[columns_initial[i]].values.reshape(-1,1)
    X_validate_optics_150000_2 = X_validate_150000_2[columns_initial[i]].values.reshape(-1,1)
    optics_150000_2 = cluster.OPTICS(min_samples=params['min_samples'],
                                     xi=params['xi'],
                                    min_cluster_size=params['min_cluster_size'])
    X_train_150000_2['optics_cluster_'+columns_initial[i]] = \
        optics_150000_2.fit_predict(X_train_optics_150000_2)
    X_validate_150000_2['optics_cluster_'+columns_initial[i]] = \
        optics_150000_2.fit_predict(X_validate_optics_150000_2)
'''

'''
# AffinityPropagation # extremely slow!!
print('Processing using AffinityPropagation...')
for i in tqdm_notebook(range(0,len(columns_initial))):
    X_train_affinity_propagation_150000_2 = X_train_150000_2[columns_initial[i]].values.reshape(-1,1)
    X_validate_affinity_propagation_150000_2 = X_validate_150000_2[columns_initial[i]].values.reshape(-1,1)
    affinity_propagation_150000_2 = cluster.AffinityPropagation(
        damping=params['damping'], preference=params['preference'])
    X_train_150000_2['affinity_propagation_cluster_'+columns_initial[i]] = \
        affinity_propagation_150000_2.fit_predict(X_train_affinity_propagation_150000_2)
    X_validate_150000_2['affinity_propagation_cluster_'+columns_initial[i]] = \
        affinity_propagation_150000_2.fit_predict(X_validate_affinity_propagation_150000_2)
'''
'''
# Birch
print('Processing using Birch...')
for i in tqdm_notebook(range(0,len(columns_initial))):
    X_train_birch_150000_2 = X_train_150000_2[columns_initial[i]].values.reshape(-1,1)
    X_validate_birch_150000_2 = X_validate_150000_2[columns_initial[i]].values.reshape(-1,1)
    birch_150000_2 = cluster.Birch(n_clusters=params['n_clusters'])
    X_train_150000_2['birch_cluster_'+columns_initial[i]] = \
        birch_150000_2.fit_predict(X_train_birch_150000_2)
    X_validate_150000_2['birch_cluster_'+columns_initial[i]] = \
        birch_150000_2.fit_predict(X_validate_birch_150000_2)
'''


#print(kmeans_150000_2.labels_)
#print(kmeans_150000_2.predict(X_validate_150000_2))


# In[63]:


'''
columns_X = ['-std','kurtosis','skewness',
             'max/min','no_of_peaks'] 
=> LB Score = 1.748, My Local Score = 2.190

LB Score=1.952, My Local Score=2.161
LB Score=1.605, My Local Score=2.146 (used the same column 'Column_X', model=CatBoostRegressor)
LB Score=1.587, My Local Score (Validation) = 2.035, r2 = 42.9% (used loss_function='MAE' 
                                                                 in CatBoostRegressor)
LB Score=1.547, My Local Score (Validation) = 1.983, r2 = 45.3% (used count_0_9 etc.)
LB Score=1.541, My Local Score (Training): 2.286 => r2=37.5%, My Local Score (Validation): 1.869 => r2=47.1% \
(used ONLY ONE FEATURE "Processed 'count_10_19'")
'''

