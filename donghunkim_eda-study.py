#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




import matplotlib.pyplot as plt
import seaborn as sns




train = pd.read_csv('../input/liverpool-ion-switching/train.csv')




train.head()




train.info()




train.describe()




train.open_channels.value_counts()




plt.figure(figsize=(20,5)); res = 1000 #간격을 1000 만큼씩 띄어서 표시
x = range(0,train.shape[0],res)
y = train.signal[0::res]
plt.plot(x,y,'b',alpha=0.7)
for i in range(11): 
    plt.plot([i*500000,i*500000],[-5,12.5],'r')
for j in range(10): 
    plt.text(j*500000+200000,10,str(j+1),size=16)
plt.xlabel('Row',size=16); plt.ylabel('Signal & Ion channels',size=16); 
plt.title('Training Data Signal - 10 batches',size=20)

#plt.figure(figsize=(20,5))
y2 = train.open_channels[0::res]
plt.plot(x,y2,'r',alpha=0.3) #Ion 채널 수 표시

plt.show()




train.corr()




#각 배치를 10개 데이터 리스트로 구분
batch = []
gap = 500000
batch.append([]) #dummy
for i in range(10):
    start = i*gap
    batch.append(train[start:start+gap])




group1 = [batch[1],batch[2]]
group2 = [batch[3],batch[7]]
group3 = [batch[4],batch[8]]
group4 = [batch[6],batch[9]]
group5 = [batch[5],batch[10]]




#plot작성용 helper 
def plot_batch(batch_no=0, start=0, end=500000):
    plt.figure(figsize=(20,5))
    x = range(end-start)
    y1 = batch[batch_no].signal[start:end]
    y2 = batch[batch_no].open_channels[start:end]
    plt.plot(x,y1)
    plt.plot(x,y2,'r',alpha=0.5)
    plt.legend()
    plt.show()




plot_batch(6) #6번 배치를 출력




plot_batch(6,0,100) # 6번 배치에서 0~100




plot_batch(6,40,60) #6번 배치에서 40~60




plot_batch(10,0,500000)




batch[10].describe()




batch[10].open_channels.value_counts()




plot_batch(1,100,20000) #1번 배치에서 100~200 




plot_batch(1,15200,15400)




train.describe()




# 데이터 값 실수. 소수점 자리 설정
pd.options.display.float_format = '{:.2f}'.format

sig_mean = []
sig_std = []

for i in range(11):
    print('='*10, i , '='*40)
    df_desc = train[train.open_channels == i][['signal','open_channels']].describe().T[['count','mean','std','min','max']]
    sig_mean.append(df_desc['mean'][0] )
    sig_std.append(df_desc['std'][0])
    print(df_desc)




import scipy.stats as stats




x = np.linspace(-15, 15, 200) #x축 값, 200개 값을 -15에서 15사이에 균일간격으로 생성  




legend = []

def norm (mu, var) :
    legend.append(str(i) + ": N(" + str(round(mu,3)) + ", " + str(round(var,3)) + ")")
    return stats.norm(mu, var).pdf(x)        

plt.figure(figsize=(20, 6))          # 플롯 사이즈 지정

for i in range(len(sig_mean)):
    plt.plot(x, norm(sig_mean[i], sig_std[i]))          # plot 추가       
    
plt.xlabel("Signal")                      # x축 레이블 지정
plt.ylabel("Open ION Channel")                      # y축 레이블 지정
plt.grid()                           # 플롯에 격자 보이기
plt.title("[ Signal distribution per open channel ]")     # 타이틀 표시
plt.legend(legend)                   # 범례 표시
plt.show()                           # 플롯 보이기





plt.figure(figsize=(20, 5))
sns.boxplot(x="open_channels", y="signal", data=train) #실용적, 2초 정도 걸림, violinplot (11초) 보다 빠름, stipplot (120초)
plt.show() 




plt.figure(figsize=(20,5))
sns.violinplot(x="open_channels",y="signal",data=train)
plt.show() #violinplot은 나오는데 boxplot보다 시간이 더 오래걸림, 11초 이상




#120초 정도 걸려서 보기 힘들고, 점이 너무 많아서 구분이 되지 않음 
#plt.figure(figsize=(20,5))
#sns.stripplot(x='open_channels', y='signal', data=train)
#plt.show()




#rolling 사용법 
# rolling(window=데이터몇개를 묶어서 rolling 처리할지, min_periods= window사이즈가 되지 않아도 처리하는 최소크기, 
#       center=기본 false로, true이면 window의 중간 중심으로 앞뒤 window/2씩 rolling 됨 )

plt.figure(figsize=(20,5))
batch[2].signal.rolling(500).mean().plot()
plt.show()




plot_batch(3,0,500000)




plot_batch(7,0,500000)




plot_batch(3,650,850)




plot_batch(7,650,850)




#plot작성용 helper, rolling추가 
def plot_rolling(batch_no=0, start=0, end=500000, window=10, rtype='mean'):
    plt.figure(figsize=(20,5))
    x = range(end-start)
    y1 = batch[batch_no].signal[start:end]
    y2 = batch[batch_no].open_channels[start:end]
    if rtype == 'min':
        y3 = batch[batch_no].signal[start:end].rolling(window=window,min_periods=1,center=False).min()
    elif rtype == 'max':
        y3 = batch[batch_no].signal[start:end].rolling(window=window,min_periods=1,center=False).max()
    else:
        y3 = batch[batch_no].signal[start:end].rolling(window=window,min_periods=1,center=False).mean()
    
    plt.plot(x,y1)
    plt.plot(x,y2,'y',alpha=0.5)
    plt.plot(x,y3,'r',alpha=0.5)
    plt.legend()
    plt.grid()
    plt.show()




plot_rolling(7,650,750, window=10)




plot_rolling(7,650,750, window=3)




plot_rolling(7,650,750, window=2, rtype='min')




batch[1].head(3)




sns.lineplot(x='time',y='signal',data=batch[1][100:200])




get_ipython().run_line_magic('pinfo', 'np.round')




def transit_matrix(data):
    channel_range = np.unique(data) #채널오픈 수량 중복제거 추출, 0,1,2,.... 10
    channel_bins = np.append(channel_range, 11) # bins 는 0,1,2,3,11 의미는 0~1, 1~2, 2~3, 3~11 의 구간임
    data_next = np.roll(data, -1)    # 1칸 앞으로 말기, 뒤에 가서 붙음
    matrix = []
    for i in channel_range:
        current_row = np.histogram(data_next[data == i], bins=channel_bins)[0]    #2개 리턴되는데, 첫번째가 각 bind 에 들어있는 수량 
        current_row = current_row / np.sum(current_row)     # i=2 일 경우,  현재 오픈된 채널별 갯수  / 직전 오픈채널이 2인 것의 총 갯수
        matrix.append(current_row) 
    return np.array(matrix)  # 행: 직전 오픈채널 값, 열: 현재 오픈채널 값 , 데이터: 해당 행,열 기준의 대상 수량




T = transit_matrix(train.open_channels)




np.round(T,3)




plt.figure(figsize=(11,11))
sns.heatmap( T,annot=True, fmt='.3f', cmap='Reds', vmin=0, vmax=0.5, linewidths=3)
plt.show()




eig_values, eig_vectors = np.linalg.eig(np.transpose(T))
print("Eigenvalues :", eig_values)




np.round(eig_vectors,3) # console출력이 길게 나오게하는 옵션이 있는지?? 중간에 짤리니 보기 불편함 




T1 = transit_matrix(batch[1].open_channels)




plt.figure(figsize=(11,11))
sns.heatmap( T1,annot=True, fmt='.3f', cmap='Reds', vmin=0, vmax=0.5, linewidths=3)
plt.show()




eig_values, eig_vectors = np.linalg.eig(np.transpose(T1))
print("Eigen values :", eig_values)
print("Eigen vectors :\n", np.round(eig_vectors,3))




dist01 = eig_vectors[:,0] / np.sum(eig_vectors[:,0])
print("Probability distribution for sequence 3 :", dist01)




np.histogram(batch[1].open_channels, bins=[0,1,2,3], density=True)[0]




def create_axes_grid(numplots_x, numplots_y, plotsize_x=6, plotsize_y=3):
    fig, axes = plt.subplots(numplots_y, numplots_x)
    fig.set_size_inches(plotsize_x * numplots_x, plotsize_y * numplots_y)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig, axes
    
def set_axes(axes, use_grid=True, x_val = [0,100,10,5], y_val = [-50,50,10,5]):
    axes.grid(use_grid)
    axes.tick_params(which='both', direction='inout', top=True, right=True, labelbottom=True, labelleft=True)
    axes.set_xlim(x_val[0], x_val[1])
    axes.set_ylim(y_val[0], y_val[1])
    axes.set_xticks(np.linspace(x_val[0], x_val[1], np.around((x_val[1] - x_val[0]) / x_val[2] + 1).astype(int)))
    axes.set_xticks(np.linspace(x_val[0], x_val[1], np.around((x_val[1] - x_val[0]) / x_val[3] + 1).astype(int)), minor=True)
    axes.set_yticks(np.linspace(y_val[0], y_val[1], np.around((y_val[1] - y_val[0]) / y_val[2] + 1).astype(int)))
    axes.set_yticks(np.linspace(y_val[0], y_val[1], np.around((y_val[1] - y_val[0]) / y_val[3] + 1).astype(int)), minor=True)
    




data = batch[5].signal

fig, axes = create_axes_grid(1,1,10,10)
set_axes(axes, x_val=[-4,8,1,.1], y_val=[-4,8,1,.1])

axes.set_aspect('equal')
axes.scatter(np.roll(data,-1), data, s=.01);




data = batch[5].signal
data_true = batch[5].open_channels

fig, axes = create_axes_grid(1,1,10,10)
set_axes(axes, x_val=[-4,8,1,.1], y_val=[-4,8,1,.1])

axes.set_aspect('equal')
for i in range(11):
    axes.scatter(np.roll(data,-1)[data_true==i], data[data_true==i], s=.01);




from statsmodels.tsa.seasonal import seasonal_decompose




train.head()




decompose_01 = seasonal_decompose(batch[1].signal, model='additive', period=1)




decompose_01.plot()




def plot_decompose(result):
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(20,10))
    result.observed.plot(legend=False, ax=ax1)
    ax1.set_ylabel('Observed')
    result.trend.plot(legend=False, ax=ax2)
    ax2.set_ylabel('Trend')
    result.seasonal.plot(legend=False, ax=ax3)
    ax3.set_ylabel('Seasonal')
    result.resid.plot(legend=False, ax=ax4)
    ax4.set_ylabel('Resid')




plot_decompose(decompose_01)




decompose = []
decompose.append([])
for i in range(10):
    decompose.append(seasonal_decompose(batch[i+1].signal, model='additive', period=1))




plot_decompose(decompose[1])




plot_decompose(decompose[2])




plot_decompose(decompose[3])




plot_decompose(decompose[4])




plot_decompose(decompose[5])




plot_decompose(decompose[6])




train_clean = pd.read_csv('../input/ion-channel-without-drift/train_clean.csv')
test_clean = pd.read_csv('../input/ion-channel-without-drift/train_clean.csv')




plt.figure(figsize=(20,5)); res = 1000 #간격을 1000 만큼씩 띄어서 표시
x = range(0,train.shape[0],res)
y = train.signal[0::res]
plt.plot(x,y,'b',alpha=0.7)
for i in range(11): 
    plt.plot([i*500000,i*500000],[-5,12.5],'r')
for j in range(10): 
    plt.text(j*500000+200000,10,str(j+1),size=16)
plt.xlabel('Row',size=16); plt.ylabel('Signal & Ion channels',size=16); 
plt.title('Training Data Signal - 10 batches',size=20)

#plt.figure(figsize=(20,5))
y2 = train.open_channels[0::res]
plt.plot(x,y2,'r',alpha=0.3) #Ion 채널 수 표시

plt.show()




plt.figure(figsize=(20,5)); res = 1000 #간격을 1000 만큼씩 띄어서 표시
x = range(0,train_clean.shape[0],res)
y = train_clean.signal[0::res]
plt.plot(x,y,'black',alpha=0.7)
for i in range(11): 
    plt.plot([i*500000,i*500000],[-5,12.5],'r')
for j in range(10): 
    plt.text(j*500000+200000,10,str(j+1),size=16)
plt.xlabel('Row',size=16); plt.ylabel('Signal & Ion channels',size=16); 
plt.title('Training Data Signal without drift - 10 batches',size=20)

#plt.figure(figsize=(20,5))
y2 = train_clean.open_channels[0::res]
plt.plot(x,y2,'r',alpha=0.3) #Ion 채널 수 표시

plt.show()




train_clean['group'] = -1
x = [(0,500000),        # 0
     (1000000,1500000), # 1
     (1500000,2000000), # 2
     (2500000,3000000), # 3
     (2000000,2500000)  # 4 
    ]

for k in range(5): 
    train_clean.iloc[x[k][0]:x[k][1],3] = k
    
res = 1000
plt.figure(figsize=(20,5))
plt.plot(train_clean.time[::res],train_clean.signal[::res], color='gray')
plt.plot(train_clean.time,train_clean.group,color='blue')
plt.title('Clean Train Data. Blue line is signal. Black line is group number.')
plt.xlabel('time'); plt.ylabel('signal')
plt.show()




from scipy.stats import mode




step = 0.2
pt = [[],[],[],[],[]]
cuts = [[],[],[],[],[]]
for g in range(5):
    mn = train_clean.loc[train_clean.group==g].signal.min()
    mx = train_clean.loc[train_clean.group==g].signal.max()
    old = 0
    for x in np.arange(mn,mx+step,step):
        sg = train_clean.loc[(train_clean.group==g)&(train_clean.signal>x-step/2)&(train_clean.signal<x+step/2)].open_channels.values
        if len(sg)>100:
            m = mode(sg)[0][0]
            pt[g].append((x,m))
            if m!=old: cuts[g].append(x-step/2)
            old = m
    pt[g] = np.vstack(pt[g])
    
models = ['1 channel low prob',
          '1 channel high prob',
          '3 channel',
          '5 channel',
          '10 channel']
plt.figure(figsize=(15,8))
for g in range(5):
    plt.plot(pt[g][:,0],pt[g][:,1],'-o',label='Group %i (%s model)'%(g,models[g]))
plt.legend()
plt.title('Traing Data Open Channels versus Clean Signal Value',size=16)
plt.xlabel('Clean Signal Value',size=16)
plt.ylabel('Open Channels',size=16)
plt.show()






