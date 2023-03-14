#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import os

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
style.use('ggplot')

import seaborn as sns
sns.set()

from IPython.display import HTML
display(HTML("<style>.container { width:98% !important; }</style>"))

import timeit
from tqdm import tqdm

from ipywidgets import interact
import ipywidgets as widgets

from scipy import fftpack

from os import listdir
print(listdir("../input"))


# In[2]:


train_nrows = get_ipython().getoutput('wc -l ../input/train.csv')
train_nrows_val = int(train_nrows[0].split()[0])
print('train.csv contains {:,} rows'.format(train_nrows_val))


# In[3]:


get_ipython().system('head ../input/train.csv')


# In[4]:


start_time = timeit.default_timer()
max_precision = 0
count = 0
with open('../input/train.csv', 'r') as f:
    while count<10: #True: #count <10:
        line = f.readline()
        if not line: 
            break
        else: 
            print(line.rstrip())
            if count > 0:
                print(line[:-1].split('.')[1])
                if '.' in line: 
                    str_len = len(line[:-1].split('.')[1])
                    print(str_len)
                    if max_precision < str_len:
                        print(line)
                    max_precision = max_precision if max_precision > str_len else str_len
                print(line)
                print(line.split('.')[1])
        count +=1
print (max_precision)
print('elapsed time: {:.2f} sec'.format(timeit.default_timer()-start_time))


# In[5]:


column_names = get_ipython().getoutput('head -n1 ../input/train.csv')
print(column_names[0].split(','))


# In[6]:


df_train_sample = pd.read_csv('../input/train.csv', skiprows = 0, nrows=100,
                       dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64}) 


# In[7]:


def display_df_with_preset_precision(df, precision):
    curr_precision = pd.get_option("display.precision")
    pd.set_option("display.precision", precision)
    display(df)
    pd.set_option("display.precision", curr_precision)
    
display_df_with_preset_precision(df_train_sample.head(9), max_precision)


# In[8]:


try:
    del(df_train_sample)    
except NameError:
    pass


# In[9]:


start_time = timeit.default_timer()
from collections import Counter
diff_ttf_2_counter_dict = Counter()
try:
    del(df_train_iter)    
except NameError:
    pass
df_train_iter = pd.read_csv('../input/train.csv', chunksize=train_nrows_val//100,
                       dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64},iterator=True)
time_to_failure_diffs_set = set()
df_after_jumping_up_points = pd.DataFrame()
df_after_long_jumps_down_points=pd.DataFrame()
for df in df_train_iter:
    df['diff_in_time_to_failure']=df['time_to_failure'].diff()
    nparr_ = df['diff_in_time_to_failure'].values
    nparr_ = nparr_[~np.isnan(nparr_)]
    diff_ttf_2_counter_dict += Counter(nparr_)

    df_jumps_up = df.loc[(df['diff_in_time_to_failure'] > 0)]
    df_long_jumps_down = df.loc[(df['diff_in_time_to_failure'] < -0.0001)]
    df_after_jumping_up_points=df_after_jumping_up_points.append(df_jumps_up)
    df_after_long_jumps_down_points=df_after_long_jumps_down_points.append(df_long_jumps_down)
print('elapsed time: {:.2f} sec'.format(timeit.default_timer()-start_time))


# In[10]:


len(list(diff_ttf_2_counter_dict.keys()))


# In[11]:


diff_ttf_2_counter_dict


# In[12]:


# positive jumps
{k: v for k, v in diff_ttf_2_counter_dict.items() if k>0}


# In[13]:


10.**max_precision


# In[14]:


negative_diff_tff_2_counter_dict = Counter()


# In[15]:


for k, v in diff_ttf_2_counter_dict.items():
    if k<=0:
        negative_diff_tff_2_counter_dict[int(-k*(10.**max_precision))] += v 
negative_diff_tff_2_counter_dict


# In[16]:


#time diff within sampling frames
time_diff_within_sampling_frames_2_count_dict = {k: v for k, v in negative_diff_tff_2_counter_dict.items() if k<100}
time_diff_within_sampling_frames_2_count_dict


# In[17]:


#time diff between sampling frames
time_diff_between_sampling_frames_2_count_dict = {round(k/(10.**max_precision),6): v for k, v in negative_diff_tff_2_counter_dict.items() if k>=100}
time_diff_between_sampling_frames_2_count_dict


# In[18]:


print(df_after_jumping_up_points.shape)
display_df_with_preset_precision(df_after_jumping_up_points, max_precision)


# In[19]:


display_df_with_preset_precision(df_after_long_jumps_down_points.head(),max_precision)


# In[20]:


np.unique(np.diff(df_after_long_jumps_down_points.index))


# In[21]:


len(np.where(np.diff(df_after_long_jumps_down_points.index)==8192)[0])


# In[22]:


len(np.where(np.diff(df_after_long_jumps_down_points.index)==4095)[0])


# In[23]:


len(np.where(np.diff(df_after_long_jumps_down_points.index)==4096)[0])


# In[24]:


max(time_diff_within_sampling_frames_2_count_dict.keys())/(10.**max_precision)*8192


# In[25]:


np.mean(list(time_diff_between_sampling_frames_2_count_dict.keys()))


# In[26]:


(max(time_diff_within_sampling_frames_2_count_dict.keys())/(10.**max_precision)*8192)/np.mean(list(time_diff_between_sampling_frames_2_count_dict.keys()))


# In[27]:


start_time = timeit.default_timer()
try:
    del(df_train_iter)    
except NameError:
    pass

df_train_iter = pd.read_csv('../input/train.csv', chunksize=train_nrows_val//100,
                       dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64},iterator=True) #use chunksize to iterate
df_before_jumping_up_points = pd.DataFrame()
for df in df_train_iter:
    if len(df.index.intersection(df_after_jumping_up_points.index-1)) > 0:
        try:
            df_before_jumping_up_points=df_before_jumping_up_points.append(df.loc[df.index.intersection(df_after_jumping_up_points.index-1),:])
        except KeyError:
            print('KeyError')
            pass
print('elapsed time: {:.2f} sec'.format(timeit.default_timer()-start_time))


# In[28]:


print(df_before_jumping_up_points.shape)
df_before_jumping_up_points


# In[29]:


start_time = timeit.default_timer()

try:
    del(df_train_tail)    
except NameError:
    pass
df_train_tail = pd.read_csv('../input/train.csv', skiprows = train_nrows_val-100000, iterator=False, names=column_names[0].split(','))
df_train_tail['acoustic_data'] = df_train_tail['acoustic_data'].astype(np.int16)
df_train_tail['time_to_failure'] = df_train_tail['time_to_failure'].astype(np.float64)
print('elapsed time: {:.2f} sec'.format(timeit.default_timer()-start_time))


# In[30]:


df_train_tail.tail()


# In[31]:


# TTF steps
df_train_tail.tail(20000)['time_to_failure'].plot();


# In[32]:


start_time = timeit.default_timer()

try:
    del(df_train_head)    
except NameError:
    pass
df_train_head = pd.read_csv('../input/train.csv', skiprows = 0, nrows = 100000, iterator=False)

print('elapsed time: {:.2f} sec'.format(timeit.default_timer()-start_time))


# In[33]:


# TTF step in the first section
df_train_head.head(20000)[8192:8192+4096]['time_to_failure'].plot();


# In[34]:


set(df_train_head.head(20000)[8192:8192+4095]['time_to_failure'].diff())


# In[35]:


n=0
set(df_train_head.head(2000000)['time_to_failure'].diff())


# In[36]:


index_ranges = [(ent[0],ent[1]) for ent in zip([0]+list(df_after_jumping_up_points.index)[:-1],list(df_before_jumping_up_points.index))]
index_ranges


# In[37]:


train_set_lengths =np.array([ent[1]-ent[0] for ent in zip([0]+list(df_before_jumping_up_points.index)[:-1],list(df_before_jumping_up_points.index))])
train_set_lengths


# In[38]:


train_set_lengths.mean(), train_set_lengths.std()


# In[39]:


range_index = 3
window_size = 15000
window_offset = -window_size
start_time = timeit.default_timer()
try:
    del(df_sample)    
except NameError:
    pass
df_sample = pd.read_csv('../input/train.csv', skiprows = index_ranges[range_index][0], nrows= index_ranges[range_index][1]-index_ranges[range_index][0],
                       dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
df_sample.columns=['acoustic_data','time_to_failure']
print(df_sample.index)

fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False)
fig.set_size_inches(32,4)
df_sample['acoustic_data'].plot(ax=axs[0]);
plt.show()
df_sample['time_to_failure'].plot(ax=axs[1]);
plt.show()
print('elapsed time: {:.2f} sec'.format(timeit.default_timer()-start_time))


# In[40]:


max_time_to_failure_points = pd.read_csv('../input/train.csv', skiprows = 0, nrows= 1, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})['time_to_failure'].append(df_after_jumping_up_points['time_to_failure'])
max_time_to_failure_points


# In[41]:


max_time_to_failure_points.values[:-1]


# In[42]:


decline_angle_tangents = np.array([ent[0]/ent[1] for ent in zip(max_time_to_failure_points.values[:-1], max_time_to_failure_points.index[1:])])
print(decline_angle_tangents.mean())
print(decline_angle_tangents.std())


# In[43]:


try:
    del(df_sample)    
except NameError:
    pass


# In[44]:


test_seg_files = listdir("../input/test")
test_seg_files[:5]


# In[45]:


len(test_seg_files)


# In[46]:


os.path.join("../input/test",test_seg_files[0])


# In[47]:


get_ipython().system('wc -l {os.path.join("../input/test",test_seg_files[0])}')


# In[48]:


get_ipython().system('head {os.path.join("../input/test",test_seg_files[0])}')


# In[49]:



def plot_test_seg_by_index(idx):
    df_test_seg = pd.read_csv(os.path.join("../input/test",test_seg_files[idx]), dtype={'acoustic_data': np.int16})
    (df_test_seg['acoustic_data']-df_test_seg['acoustic_data'].mean()).plot();


# In[50]:


interact(plot_test_seg_by_index, idx=widgets.IntSlider(min=0,max=len(test_seg_files)-1,step=1,value=0));


# In[51]:


# make sure that all seg files have the same length: 150,000 samples:
seg_files_lengths = get_ipython().getoutput('for filename in ../input/test/*; do wc -l $filename; done')
{ent.split(' ')[0] for ent in seg_files_lengths}


# In[52]:


(max(time_diff_within_sampling_frames_2_count_dict.keys())/(10.**max_precision)*150000)


# In[53]:


np.mean(list(time_diff_between_sampling_frames_2_count_dict.keys()))*(150000/4096)


# In[54]:


start_time = timeit.default_timer()
try:
    del(df_sample)    
except NameError:
    pass

range_index=0 #first training sequence is the shortest one
df_sample = pd.read_csv('../input/train.csv', skiprows = index_ranges[range_index][0], nrows= index_ranges[range_index][1]-index_ranges[range_index][0],
                       dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
df_sample.columns=['acoustic_data','time_to_failure']

fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False)
fig.set_size_inches(32,4)
df_sample['acoustic_data'].plot(ax=axs[0]);
plt.show()
print ('time_to_failure decline rate = {:.16f}'.format(df_sample['time_to_failure'][0]/df_sample['time_to_failure'].shape[0]))
df_sample['time_to_failure'].plot(ax=axs[1]);
plt.show()
print('elapsed time: {:.2f} sec'.format(timeit.default_timer()-start_time))


# In[55]:


df_sample['time_to_failure'][0]/df_sample['time_to_failure'].shape[0]


# In[56]:


df_sample['time_to_failure'].shape[0]


# In[57]:


df_sample['acoustic_data'].head()


# In[58]:


df_sample['acoustic_data'].mean()


# In[59]:


train_values = (df_sample['acoustic_data']-df_sample['acoustic_data'].mean()).values
train_values


# In[60]:


df_test_seg = pd.read_csv(os.path.join("../input/test",test_seg_files[0]), dtype={'acoustic_data': np.int16})
df_test_seg.head()


# In[61]:


df_test_seg['acoustic_data'].mean()


# In[62]:


test_values = (df_test_seg['acoustic_data']-df_test_seg['acoustic_data'].mean()).values


# In[63]:


print(train_values.shape, test_values.shape)


# In[64]:


from scipy import signal
signal_corr = signal.correlate(np.square(train_values), np.square(test_values),mode='valid', method='fft')


# In[65]:


signal_corr.shape


# In[66]:


pd.DataFrame(signal_corr).plot()


# In[67]:


def correlation_with_test_seg_idx(idx):
    df_test_seg = pd.read_csv(os.path.join("../input/test",test_seg_files[idx]), dtype={'acoustic_data': np.int16})
    test_values = (df_test_seg['acoustic_data']-df_test_seg['acoustic_data'].mean()).values
    signal_corr = signal.correlate(np.square(train_values), np.square(test_values),mode='same', method='fft')
    pd.DataFrame(signal_corr).plot();


# In[68]:


interact(correlation_with_test_seg_idx, idx=widgets.IntSlider(min=0,max=len(test_seg_files)-1,step=1,value=0));


# In[69]:


from scipy.signal import spectrogram

M = 1024
N = 1024
freqs, times, Sx = signal.spectrogram(df_test_seg['acoustic_data'].values, fs=1, window='hanning',
                                      nperseg=N, noverlap=M - 100,
                                      detrend=False, scaling='spectrum')

f, ax = plt.subplots(figsize=(4.8, 2.4))
ax.pcolormesh(times, freqs / 1000, 10 * np.log10(Sx), cmap='viridis')
ax.set_ylabel('Frequency [kHz]')
ax.set_xlabel('Time [s]');


# In[70]:


f, t, Sxx = spectrogram(df_test_seg['acoustic_data'].values)
plt.pcolormesh(t, f, 10 * np.log10(Sxx))
plt.show()
plt.plot(Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


# In[71]:


from skimage import util

M = 1024

slices = util.view_as_windows(df_test_seg['acoustic_data'].values, window_shape=(M,), step=100)
print(f'data shape: {df_test_seg["acoustic_data"].values.shape}, Sliced data shape: {slices.shape}')


# In[72]:


win = np.hanning(M + 1)[:-1]
slices = slices * win
slices = slices.T
print('Shape of `slices`:', slices.shape)
spectrum = np.fft.fft(slices, axis=0)[:M // 2 + 1:-1]
spectrum = np.abs(spectrum)


# In[73]:


L=df_test_seg['acoustic_data'].values.shape[0]


# In[74]:


rate = 10
f, ax = plt.subplots(figsize=(4.8, 2.4))

S = np.abs(spectrum)
S = 20 * np.log10(S / np.max(S))

ax.imshow(S, origin='lower', cmap='viridis',
          extent=(0, L, 0, rate / 2 / 1000))
ax.axis('tight')
ax.set_ylabel('Frequency [kHz]')
ax.set_xlabel('Time [s]');


# In[75]:


yf = fftpack.fft(df_test_seg['acoustic_data'].values)
plt.plot(np.abs(yf))
plt.grid()
plt.show()


# In[76]:


display_df_with_preset_precision(df_after_long_jumps_down_points.head(),max_precision)


# In[77]:


range_index=15 #first training sequence is the shortest one
start_time = timeit.default_timer()
try:
    del(df_sample)    
except NameError:
    pass
df_sample = pd.read_csv('../input/train.csv', skiprows = index_ranges[range_index][0], nrows= index_ranges[range_index][1]-index_ranges[range_index][0],
                       dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
df_sample.columns=['acoustic_data','time_to_failure']
print('elapsed time: {:.2f} sec'.format(timeit.default_timer()-start_time))


# In[78]:


df_sample.index
index_ranges[range_index]
ixs_of_indexes = np.where(np.all([df_after_long_jumps_down_points.index <= index_ranges[range_index][1],df_after_long_jumps_down_points.index >= index_ranges[range_index][0]-1],axis=0))[0]
ixs_of_indexes1 = np.where(np.logical_and(df_after_long_jumps_down_points.index <= index_ranges[range_index][1],
                        df_after_long_jumps_down_points.index >= index_ranges[range_index][0]))[0]
ixs_of_indexes,ixs_of_indexes1


# In[79]:


np.array_equal(ixs_of_indexes1,ixs_of_indexes)


# In[80]:


indexes = df_after_long_jumps_down_points.index[ixs_of_indexes].union(index_ranges[range_index])


# In[81]:


np.unique(np.diff(indexes))


# In[82]:


frame_indexes = list(zip(indexes[:-1],indexes[1:]))


# In[83]:


len(list(frame_indexes))


# In[84]:


#MEAN (Bias) is removed 
low_path_filter_n_freqs = 2048
avg_len=1
frame_sequence_offset = -frame_indexes[0][0]
def show_frame_and_fft(sequence_idx):
    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=False)
    the_df = df_sample[frame_sequence_offset+frame_indexes[sequence_idx][0]:frame_sequence_offset+frame_indexes[sequence_idx][1]]
    print(the_df.shape)
    diff_ttf = the_df['time_to_failure'].diff()[1:]    
    acustic_data_series = the_df['acoustic_data']
    
    print('ttf = {:.9f}'.format(the_df['time_to_failure'].mean()))
    print('diff_ttf.mean = {:.9f}, diff_ttf.std = {:.9f}'.format(diff_ttf.mean(), diff_ttf.std()))
    print('acustic.mean = {:.9f}, acustic.std = {:.9f}'.format(acustic_data_series.mean(), acustic_data_series.std()))
    
    (acustic_data_series - acustic_data_series.mean()).plot(ax=axs[0]);
    fig.set_size_inches(32,4)
    
    acustic_data_rft = fftpack.rfft(acustic_data_series-acustic_data_series.mean())
    acustic_data_ft = fftpack.fft(acustic_data_series-acustic_data_series.mean())
    
    rfreqs = fftpack.rfftfreq(acustic_data_rft.size,diff_ttf.mean())
    freqs = fftpack.fftfreq(acustic_data_ft.size, diff_ttf.mean())    
    rfreqs = -rfreqs
    
    the_dict = {}
    for idx in range(len(acustic_data_rft)):
        if rfreqs[idx] in the_dict:
            the_dict[rfreqs[idx]] = (the_dict[rfreqs[idx]]+np.abs(acustic_data_rft[idx]))/2.
        else: 
            the_dict[rfreqs[idx]] = np.abs(acustic_data_rft[idx])
    unique_rfreqs = np.unique(rfreqs)
    print("arrays are equal is {}".format(np.array_equal(sorted(unique_rfreqs),unique_rfreqs)))
    
    pd.DataFrame.from_dict({'acustic_data_rft_amp': [the_dict[ent] for ent in unique_rfreqs][:low_path_filter_n_freqs], 
                            'rfreqs':unique_rfreqs[:low_path_filter_n_freqs]}).set_index('rfreqs').plot(ax=axs[1])
    pd.DataFrame.from_dict({'acustic_data_ft_amp': (np.abs(acustic_data_ft)[len(freqs)//2:]), 
                            'freqs':freqs[len(freqs)//2:]}).set_index('freqs').plot(ax=axs[2])


# In[85]:


print('range_index = {}'.format(range_index))
interact(show_frame_and_fft, sequence_idx=widgets.IntSlider(min=0,max=len(list(frame_indexes))-1,step=1,value=0));


# In[86]:


def calc_fft_amp_per_sequence_index(sequence_idx):
    the_df = df_sample[frame_sequence_offset+frame_indexes[sequence_idx][0]:frame_sequence_offset+frame_indexes[sequence_idx][1]]

    diff_ttf = the_df['time_to_failure'].diff()[1:]   
    acustic_data_series = the_df['acoustic_data']
    
    print('ttf = {:.9f}'.format(the_df['time_to_failure'].mean()))
    print('diff_ttf.mean = {:.9f}, diff_ttf.std = {:.9f}'.format(diff_ttf.mean(), diff_ttf.std()))
    print('acustic.mean = {:.9f}, acustic.std = {:.9f}'.format(acustic_data_series.mean(), acustic_data_series.std()))    
    
    acustic_data_rft = fftpack.rfft(acustic_data_series-acustic_data_series.mean())
       
    rfreqs = -fftpack.rfftfreq(len(acustic_data_rft),diff_ttf.mean())   
    
    the_dict = {}
    for idx in range(len(acustic_data_rft)):
        if rfreqs[idx] in the_dict:
            the_dict[rfreqs[idx]] = (the_dict[rfreqs[idx]]+np.abs(acustic_data_rft[idx]))/2.
        else: 
            the_dict[rfreqs[idx]] = np.abs(acustic_data_rft[idx])
    
    unique_rfreqs = np.unique(rfreqs)
    
    return pd.DataFrame.from_dict({'acustic_data_rft_amp': [the_dict[ent] for ent in unique_rfreqs][:low_path_filter_n_freqs], 
                                   'rfreqs':unique_rfreqs[:low_path_filter_n_freqs]}).set_index('rfreqs')


# In[87]:


(calc_fft_amp_per_sequence_index(0)['acustic_data_rft_amp']).values[:low_path_filter_n_freqs]


# In[88]:


def features_row_per_sequence_index(sequence_idx):
    the_df = df_sample[frame_sequence_offset+frame_indexes[sequence_idx][0]:frame_sequence_offset+frame_indexes[sequence_idx][1]]

    diff_ttf = the_df['time_to_failure'].diff()[1:]   
    acustic_data_series = the_df['acoustic_data']   
    
    acustic_data_rft = fftpack.rfft(acustic_data_series-acustic_data_series.mean())
       
    rfreqs = -fftpack.rfftfreq(len(acustic_data_rft),diff_ttf.mean())   
    
    the_dict = {}
    for idx in range(len(acustic_data_rft)):
        if rfreqs[idx] in the_dict:
            the_dict[rfreqs[idx]] = (the_dict[rfreqs[idx]]+np.abs(acustic_data_rft[idx]))/2.
        else: 
            the_dict[rfreqs[idx]] = np.abs(acustic_data_rft[idx])
    
    unique_rfreqs = np.unique(rfreqs)
    
    return [acustic_data_series.mean()]+[the_dict[ent] for ent in unique_rfreqs][:low_path_filter_n_freqs][1:]+[the_df['time_to_failure'].mean()]


# In[89]:


start_time = timeit.default_timer()
pd.DataFrame(np.array([features_row_per_sequence_index(idx) for idx in range(len(list(frame_indexes)))])).to_csv('../working/feat_rfft_data_for_range_index_{}_nfreqs_{}.csv'.format(range_index, low_path_filter_n_freqs))
print('elapsed time: {:.2f} sec'.format(timeit.default_timer()-start_time))


# In[90]:


def plot_test_seg_by_index(idx):
    df_test_seg = pd.read_csv(os.path.join("../input/test",test_seg_files[idx]), dtype={'acoustic_data': np.int16})
    (df_test_seg['acoustic_data']-df_test_seg['acoustic_data'].mean()).plot();


# In[91]:


#MEAN (Bias) is removed 
diff_ttf_mean = -0.000000001
avg_len=1
frame_sequence_offset = -frame_indexes[0][0]
def show_test_frame_and_fft_by_idx(idx):
    df_test_seg = pd.read_csv(os.path.join("../input/test",test_seg_files[idx]), dtype={'acoustic_data': np.int16})
    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=False)

    print(df_test_seg.shape)
    
    acustic_data_series = df_test_seg['acoustic_data']
   
    print('acustic.mean = {:.9f}, acustic.std = {:.9f}'.format(acustic_data_series.mean(), acustic_data_series.std()))
    
    (acustic_data_series - acustic_data_series.mean()).plot(ax=axs[0]);

    fig.set_size_inches(32,4)
    
    acustic_data_rft = fftpack.rfft(acustic_data_series-acustic_data_series.mean())
    acustic_data_ft = fftpack.fft(acustic_data_series-acustic_data_series.mean())
    
    rfreqs = fftpack.rfftfreq(acustic_data_rft.size,diff_ttf_mean)
    freqs = fftpack.fftfreq(acustic_data_ft.size, diff_ttf_mean)
        
    rfreqs = -rfreqs
    
    the_dict = {}
    for idx in range(len(acustic_data_rft)):
        if rfreqs[idx] in the_dict:
            the_dict[rfreqs[idx]] = (the_dict[rfreqs[idx]]+np.abs(acustic_data_rft[idx]))/2.
        else: 
            the_dict[rfreqs[idx]] = np.abs(acustic_data_rft[idx])
    unique_rfreqs = np.unique(rfreqs)
    print("arrays are equal is {}".format(np.array_equal(sorted(unique_rfreqs),unique_rfreqs)))
    
    pd.DataFrame.from_dict({'acustic_data_rft_amp': [the_dict[ent] for ent in unique_rfreqs][:low_path_filter_n_freqs], 
                            'rfreqs':unique_rfreqs[:low_path_filter_n_freqs]}).set_index('rfreqs').plot(ax=axs[1])
    pd.DataFrame.from_dict({'acustic_data_ft_amp': (np.abs(acustic_data_ft)[len(freqs)//2:]), 
                            'freqs':freqs[len(freqs)//2:]}).set_index('freqs').plot(ax=axs[2])


# In[92]:


interact(show_test_frame_and_fft_by_idx, idx=widgets.IntSlider(min=0,max=len(test_seg_files)-1,step=1,value=0));


# In[93]:


def test_features_row_per_sequence_index(idx):
    df_test_seg = pd.read_csv(os.path.join("../input/test",test_seg_files[idx]), dtype={'acoustic_data': np.int16}) 
    acustic_data_series = df_test_seg['acoustic_data'] 
    
    acustic_data_rft = fftpack.rfft(acustic_data_series-acustic_data_series.mean())
       
    rfreqs = -fftpack.rfftfreq(len(acustic_data_rft),diff_ttf_mean)   
    
    the_dict = {}
    for idx in range(len(acustic_data_rft)):
        if rfreqs[idx] in the_dict:
            the_dict[rfreqs[idx]] = (the_dict[rfreqs[idx]]+np.abs(acustic_data_rft[idx]))/2.
        else: 
            the_dict[rfreqs[idx]] = np.abs(acustic_data_rft[idx])
    
    unique_rfreqs = np.unique(rfreqs)
    
    return [acustic_data_series.mean()]+[the_dict[ent] for ent in unique_rfreqs][:low_path_filter_n_freqs][1:]#+[the_df['time_to_failure'].mean()


# In[94]:


len(test_features_row_per_sequence_index(0))


# In[95]:


start_time = timeit.default_timer()
pd.DataFrame(np.array([test_features_row_per_sequence_index(idx) for idx in range(len(test_seg_files))])).to_csv('../working/feat_test_rfft_data_nfeatures_{}.csv'.format(low_path_filter_n_freqs))
print('elapsed time: {:.2f} sec'.format(timeit.default_timer()-start_time))


# In[96]:


range_index=12 
sequence_length = 4096
start_time = timeit.default_timer()
try:
    del(df_sample)    
except NameError:
    pass
df_sample = pd.read_csv('../input/train.csv', skiprows = index_ranges[range_index][0], nrows= index_ranges[range_index][1]-index_ranges[range_index][0],
                       dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
df_sample.columns=['acoustic_data','time_to_failure']
print(df_sample.shape[0]/sequence_length)
print('elapsed time: {:.2f} sec'.format(timeit.default_timer()-start_time))


# In[97]:


#MEAN (Bias) is removed 
avg_len=2
sequence_offset = 0
def show_frame_and_cwt(sequence_idx):
    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=False)

    print(df_sample[sequence_offset+sequence_length*sequence_idx:sequence_offset+sequence_length*sequence_idx+sequence_length].shape)
    df_sample[sequence_offset+sequence_length*sequence_idx:sequence_offset+sequence_length*sequence_idx+sequence_length]['time_to_failure'].plot(ax=axs[0]);
    
    acustic_data_series = df_sample[sequence_offset+sequence_length*sequence_idx:sequence_offset+sequence_length*sequence_idx+sequence_length]['acoustic_data']
    acustic_data_series_zero_mean = (acustic_data_series - acustic_data_series.mean())
    acustic_data_series_zero_mean.plot(ax=axs[1])
    fig.set_size_inches(32,4)
    
    acustic_data_cwt = signal.cwt(acustic_data_series_zero_mean, signal.morlet, #signal.morlet, signal.ricker
                                  np.arange(1,31))
    axs[2].imshow(acustic_data_cwt, cmap='PRGn', aspect='auto',)
    print(len(acustic_data_cwt))
    print(acustic_data_cwt.shape)


# In[98]:


interact(show_frame_and_cwt, sequence_idx=widgets.IntSlider(min=0,max=df_sample.shape[0]/sequence_length,step=1,value=0));


# In[99]:


import pywt 
from pywt import wavedec
#MEAN (Bias) is removed 
avg_len=2
sequence_offset = 0
def show_frame_and_dwt(sequence_idx):
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False)

    print(df_sample[sequence_offset+sequence_length*sequence_idx:sequence_offset+sequence_length*sequence_idx+sequence_length].shape)
    df_sample[sequence_offset+sequence_length*sequence_idx:sequence_offset+sequence_length*sequence_idx+sequence_length]['time_to_failure'].plot(ax=axs[0]);
    
    acustic_data_series = df_sample[sequence_offset+sequence_length*sequence_idx:sequence_offset+sequence_length*sequence_idx+sequence_length]['acoustic_data']
    acustic_data_series_zero_mean = (acustic_data_series - acustic_data_series.mean())
    acustic_data_series_zero_mean.plot(ax=axs[1])

    fig.set_size_inches(32,4)
    
    acustic_data_dwt_coeffs = wavedec(acustic_data_series_zero_mean,'db1', level=1)
    print(len(acustic_data_dwt_coeffs))
    print((acustic_data_dwt_coeffs[0]).shape)
    print(acustic_data_dwt_coeffs)


# In[100]:


interact(show_frame_and_dwt, sequence_idx=widgets.IntSlider(min=0,max=df_sample.shape[0]/sequence_length,step=1,value=0));

