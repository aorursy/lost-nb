#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

size=16
params = {'legend.fontsize': 'large',
          'figure.figsize': (16,4),
          'axes.labelsize': size*1.1,
          'axes.titlesize': size*1.3,
          'xtick.labelsize': size*0.9,
          'ytick.labelsize': size*0.9,
          'axes.titlepad': 25}
plt.rcParams.update(params)




signal_batch_size = 500_000

df_train = pd.read_csv('../input/liverpool-ion-switching/train.csv')

# Add a signal_batch number, for ease of grouping
df_train['signal_batch'] = np.arange(len(df_train)) // signal_batch_size




fig, ax = plt.subplots(1,1,figsize=(12,6))

df_train    .groupby('signal_batch')['open_channels']    .apply(lambda x: len(set(x)))    .value_counts()    .sort_index()    .plot(kind='bar', ax=ax, width=0.8)

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', va='bottom', 
                color='black', fontsize=14, 
                #fontweight='heavy',
                xytext=(0,5), 
                textcoords='offset points')

ax.set_yticks([0,1,2,3,4])
ax.set_yticklabels([0,1,2,3,4])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_xlabel('No. of Labels per Signal Batch')
ax.set_ylabel('No. of Signal Batch')
ax.set_title('Distribution of No. of Labels Per Signal Batch '+'$(n = 10)$')

for loc in ['right','top']:
    ax.spines[loc].set_visible(False)




def plot_signal_and_label(segment_size=200):
    fig, ax = plt.subplots(1,1, figsize=(14,6))

    sample = np.random.randint(0,9)
    segment = np.random.randint(0,500_000 - segment_size)
    
    df_segment = df_train.query('signal_batch == @sample')
    
    df_segment['signal'].iloc[segment:segment+segment_size]        .plot(ax=ax, label='Signal', alpha=0.8, linewidth=2)
    
    ax_2nd = ax.twinx()
    df_segment['open_channels'].iloc[segment:segment+segment_size]        .plot(ax=ax_2nd, label='Open Channels (Ground Truth)', color='C1', linewidth=2)

    time_start = df_segment['time'].iloc[segment]
    time_end = df_segment['time'].iloc[segment + segment_size-1]
    
    xticklabels = [val for i, val in enumerate(df_segment['time'].iloc[segment:segment + segment_size + 1]) if i%(segment_size//10) == 0]
    xtickloc = [val for i, val in enumerate(df_segment.iloc[segment:segment + segment_size + 1].index) if i%(segment_size//10) == 0]
    
    ax.set_xticks(xtickloc)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('Timestamp')
    
    ax.set_ylabel('Signal')
    ax_2nd.set_ylabel('Open Channels')
    
    ax.set_title(f'Signal Batch #{sample} \n('
                 r'$t_{start} = $' + f'${time_start} s, $'
                 r'$t_{end} = $' + f'${time_end} s$' + ')')
    fig.legend(bbox_to_anchor=(1.03,0.5), loc='center left')
    
    ax.spines['top'].set_visible(False)
    ax_2nd.spines['top'].set_visible(False)
    ax.grid(which='major',axis='x', linestyle='--')

    plt.tight_layout()
    plt.show()
    




for i in range(10):
    plot_signal_and_label(segment_size=200)






