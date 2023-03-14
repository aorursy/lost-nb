#!/usr/bin/env python
# coding: utf-8



import numpy as np 
import pandas as pd 




gift_pref = pd.read_csv('../input/child_wishlist.csv',header=None).drop(0, 1).values
child_pref = pd.read_csv('../input/gift_goodkids.csv',header=None).drop(0, 1).values
random_sub = pd.read_csv('../input/sample_submission_random.csv').values.tolist()




# https://www.kaggle.com/wendykan/average-normalized-happiness-demo

from collections import Counter

n_children = 1000000 # n children to give
n_gift_type = 1000 # n types of gifts available
n_gift_quantity = 1000 # each type of gifts are limited to this quantity
n_gift_pref = 10 # number of gifts a child ranks
n_child_pref = 1000 # number of children a gift ranks
twins = int(0.004 * n_children)    # 0.4% of all population, rounded to the closest even number
ratio_gift_happiness = 2
ratio_child_happiness = 2


def avg_normalized_happiness(pred, child_pref, gift_pref):
    
    # check if number of each gift exceeds n_gift_quantity
    gift_counts = Counter(elem[1] for elem in pred)
    for count in gift_counts.values():
        assert count <= n_gift_quantity
                
    # check if twins have the same gift
    for t1 in range(0,twins,2):
        twin1 = pred[t1]
        twin2 = pred[t1+1]
        assert twin1[1] == twin2[1]
    
    max_child_happiness = n_gift_pref * ratio_child_happiness
    max_gift_happiness = n_child_pref * ratio_gift_happiness
    total_child_happiness = 0
    total_gift_happiness = np.zeros(n_gift_type)
    
    for row in pred:
        child_id = row[0]
        gift_id = row[1]
        
        # check if child_id and gift_id exist
        assert child_id < n_children
        assert gift_id < n_gift_type
        assert child_id >= 0 
        assert gift_id >= 0
        child_happiness = (n_gift_pref - np.where(gift_pref[child_id]==gift_id)[0]) * ratio_child_happiness
        if not child_happiness:
            child_happiness = -1

        gift_happiness = ( n_child_pref - np.where(child_pref[gift_id]==child_id)[0]) * ratio_gift_happiness
        if not gift_happiness:
            gift_happiness = -1

        total_child_happiness += child_happiness
        total_gift_happiness[gift_id] += gift_happiness
    
    # print(max_child_happiness, max_gift_happiness 
    print('normalized child happiness=',float(total_child_happiness)/(float(n_children)*float(max_child_happiness)) ,         ', normalized gift happiness',np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity))
    return float(total_child_happiness)/(float(n_children)*float(max_child_happiness)) + np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity)




from numba import jit

@jit(nopython=True)
def avg_normalized_happiness_fast(pred, child_pref, gift_pref):
    
    n_children = 1000000 # n children to give
    n_gift_type = 1000 # n types of gifts available
    n_gift_quantity = 1000 # each type of gifts are limited to this quantity
    n_gift_pref = 10 # number of gifts a child ranks
    n_child_pref = 1000 # number of children a gift ranks
    twins = int(0.004 * n_children)    # 0.4% of all population, rounded to the closest even number
    ratio_gift_happiness = 2
    ratio_child_happiness = 2

    # check if number of each gift exceeds n_gift_quantity
    tmp_dict = np.zeros(n_gift_quantity, dtype=np.uint16)
    for i in np.arange(len(pred)):
        tmp_dict[pred[i][1]] += 1
    for count in np.arange(n_gift_quantity):
        assert count <= n_gift_quantity    
                
    # check if twins have the same gift
    for t1 in np.arange(0,twins,2):
        twin1 = pred[t1]
        twin2 = pred[t1+1]
        assert twin1[1] == twin2[1]
    
    max_child_happiness = n_gift_pref * ratio_child_happiness
    max_gift_happiness = n_child_pref * ratio_gift_happiness
    total_child_happiness = 0
    total_gift_happiness = np.zeros(n_gift_type, dtype=np.float32)

    for i in np.arange(len(pred)):
        row = pred[i]
        child_id = row[0]
        gift_id = row[1]
        
        # check if child_id and gift_id exist
        assert child_id < n_children
        assert gift_id < n_gift_type
        assert child_id >= 0 
        assert gift_id >= 0
        
        child_happiness = (n_gift_pref - np.where(gift_pref[child_id]==gift_id)[0]) * ratio_child_happiness
        if (len(child_happiness) == 0):
            tmp_child_happiness = -1
        else:
            tmp_child_happiness = child_happiness[0]

        gift_happiness = ( n_child_pref - np.where(child_pref[gift_id]==child_id)[0]) * ratio_gift_happiness
        if (len(gift_happiness) == 0):
            tmp_gift_happiness = -1
        else:
            tmp_gift_happiness = gift_happiness[0]
            
        total_child_happiness += tmp_child_happiness    
        total_gift_happiness[gift_id] += tmp_gift_happiness    
        
    # print(max_child_happiness, max_gift_happiness  
    print('normalized child happiness=',float(total_child_happiness)/(float(n_children)*float(max_child_happiness)) ,         ', normalized gift happiness',np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity))
    return float(total_child_happiness)/(float(n_children)*float(max_child_happiness)) + np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity)
            




time_original = get_ipython().run_line_magic('timeit', '-r 10 -o avg_normalized_happiness(random_sub, child_pref, gift_pref)')




time_fast = get_ipython().run_line_magic('timeit', '-r 10 -o avg_normalized_happiness_fast(np.array(random_sub), child_pref, gift_pref)')




import matplotlib.pyplot as plt
import seaborn as sns

time_result = pd.DataFrame({'time_original': time_original.all_runs,
                            'time_fast': time_fast.all_runs})
time_result.describe()




time_result.plot.box(figsize=(12,10))
plt.show()




time_result.plot(figsize=(12,10))
plt.show()




avg_normalized_happiness(np.array(random_sub), child_pref, gift_pref)




avg_normalized_happiness_fast(np.array(random_sub), child_pref, gift_pref)

