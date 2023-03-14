#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import itertools
from collections import Counter
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




tournies = pd.read_csv('../input/TourneyCompactResults.csv')




## read in seeds data, put seeds as columns in tournies frame
seeds = pd.read_csv('../input/TourneySeeds.csv')
seeds['Season/Team'] = [(seas,team) for seas,team in zip(seeds.Season,seeds.Team)]
seeds = seeds.set_index('Season/Team').drop(['Season','Team'],axis=1).squeeze().to_dict()
tournies['Wteam_seed'] = [seeds[(year,team)] for year,team in zip(tournies.Season,tournies.Wteam)]
tournies['Lteam_seed'] = [seeds[(year,team)] for year,team in zip(tournies.Season,tournies.Lteam)]




first_round = set([(1,16),(2,15),(3,14),(4,13),
                   (5,12),(6,11),(7,10),(8,9)])
third_round = set([tuple(sorted(tup)) 
                   for tup in itertools.product([1,8,9,16],[4,5,12,13])]) \
              .union(set([tuple(sorted(tup)) 
                          for tup in itertools.product([2,15,7,10],
                                                       [3,6,11,14])]))
fourth_round = set([tuple(sorted(tup)) 
                    for tup in itertools.product([1,4,5,8,9,12,13,16],
                                                 [2,3,6,7,10,11,14,15])])

def get_round_from_seeds(seed1,seed2):
    if len(seed1)==len(seed2)==4:
        return 0
    seed1,seed2 = sorted([seed1,seed2])
    bracket1,seed1 = seed1[0],int(seed1[1:3])
    bracket2,seed2 = seed2[0],int(seed2[1:3])
    if bracket1 in 'WX' and bracket2 in 'YZ':
        return 6
    if bracket1!=bracket2:
        return 5
    seed1,seed2 = sorted([seed1,seed2])
    if (seed1,seed2) in first_round:
        return 1
    if (seed1,seed2) in third_round:
        return 3
    if (seed1,seed2) in fourth_round:
        return 4
    return 2




## put round and seed numbers in
tournies['Round'] = tournies[['Wteam_seed','Lteam_seed']].apply(lambda x: get_round_from_seeds(x.iloc[0],x.iloc[1]),axis=1)
tournies['Wteam_seed_num'] = [int(seed[1:3]) for seed in tournies.Wteam_seed]
tournies['Lteam_seed_num'] = [int(seed[1:3]) for seed in tournies.Lteam_seed]




for round_,df in tournies.groupby('Round'):
    seeds = Counter(np.concatenate([df.Wteam_seed_num.values,df.Lteam_seed_num.values]))
    fig,ax = plt.subplots(1,1,figsize=(12,5))
    plt.bar(range(1,17),[seeds[seed] if seed in seeds else 0 for seed in range(1,17)])
    plt.title(round_)
    plt.xticks(range(1,17),range(1,17))
    plt.show()

