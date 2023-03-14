#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# user local module stochastic product search
#import stochprodsearch_03 # DOESNT WORK SO FAR

from datetime import datetime
import time 
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# to save 
import pickle
from sklearn.externals import joblib
#import joblib
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) 
# will list all files under the input directory

import os
from numba import njit, jit, prange

# to get computer name
import platform
import re

# For Figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns
sns.set(color_codes=True, font_scale=1.33)


# In[2]:


###
# Paths to folders/files
###

# KAGGLE
PATH_INPUT_KAGGLE = '/kaggle/input' #for kaggle
PATH_TO_SAVE_DATA_KAGGLE = '/kaggle/working' #for kaggle
PATH_TO_EXPLORE_DATA_KAGGLE = '/kaggle/input/santa-2019-for-my-exploration' # for kaggle

# LOCAL
PATH_INPUT = 'kaggle/input'
PATH_TO_SAVE_DATA = "../../data"
PATH_TO_EXPLORE_DATA = 'kaggle/input/santa_2019_for_my_exploration'

# GOOGLE COLAB
PATH_GOOGLE_COLAB = "/content/drive/My Drive/OpenClassRooms/IML_projet_8"  +              "/code/santa-workshop-tour-2019"

# get computer name
COMPUTERNAME = platform.node()
# select current platform
MY_PLATFORM = platform.system()
# check if Google colab need Drive ?
if re.match("^/content", os.getcwd()):
    print("GOOGLE COLAB MODE")
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    os.chdir(PATH_GOOGLE_COLAB)
elif re.match("^/kaggle", os.getcwd()):
    print("KAGGLE COLAB MODE")
    PATH_INPUT = PATH_INPUT_KAGGLE 
    PATH_TO_SAVE_DATA = PATH_TO_SAVE_DATA_KAGGLE 
    PATH_TO_EXPLORE_DATA = PATH_TO_EXPLORE_DATA_KAGGLE 
    
# PROB FILE PATH
PATH_SAVE_PROB_FAM = PATH_TO_SAVE_DATA + '/df_prob_fam.pkl'

## POP PATH : to generate first pop, used 'RAMDOM_PATH' instead of None
#SAVE_POP = None
#SAVE_POP = '10R' # 10 ranges method
SAVE_POP = 'RANDOM_PATH' # random ranges method
#SAVE_POP = 'RANDOM_CHOICE' # first random choices method

# path to pop df file id save pop is none.
#PATH_DF_POP = PATH_TO_SAVE_DATA + '/' + \
#   "df_pop_choices_10R_1000_fs10_rfm0.05_dc2.pkl"
PATH_DF_POP = PATH_TO_SAVE_DATA + '/' + "df_pop_choices_RANDOM_PATH_1000.pkl"
#PATH_DF_POP = PATH_TO_SAVE_DATA + '/' + \
#    "df_pop_choices_RANDOM_CHOICE_1000_dcr1.pkl"
#PATH_DF_POP = PATH_TO_SAVE_DATA + '/' + \
#    "df_pop1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen200000_s84089.67769691352.pkl"
# DAY information (useless ?)
DAY_RANGE = list(range(1, 101)) # day before Christmas
DAY_RANGE_MIN = np.min(DAY_RANGE) 
DAY_RANGE_MAX = np.max(DAY_RANGE) 

##########
## Hyper-parameters: DEFAULT

# from data 
CHOICE_RANGE_MIN = 0 # minimum choice number
CHOICE_RANGE_MAX = 4 # maximum choice number

# for POP 
NB_FIRST_SEED = 10 # best parent to create mutated first population
NB_FIRST_POP = 1000 # number of first population of choices 
DELTA_CHOICE_FIRST_POP = 2 # +/- delta choice of mutated first population 
R_FIRST_MUT = 0.05 # RATIO of mutation for first population
DELTA_CHOICE_RANDOM_POP = 1 # delta for first random choice pop
DELTA_RANDOM_MUT_POP = 1 # delta for first random mut pop
R_FIRST_RANDOM_MUT = 0.2 # RATIO of mutation for first population in random mut
# for all generations
R_POP_MUT = 0.05 # RATIO of population mutation after first generation
R_MUT = 0.01 # RATIO of number of family choices mutated
DELTA_DAYS = 1 # delta of days around previous best day for generation/mutation
R_POP_LOST = 0.01 # Ratio of lost individuals in population 
POW_SELECTION = 0.3 # power for slection during crossing
NB_BEST_KEEP = 10 # number of best indiv to keep at each epoch
NB_MAX_EPOCHS = 1000
DELTA_CHOICE = 1 # +/- delta choice of mutated for generation population
R_CROSSOVER = 1 # Crossover Ratio of pop for next generation 
    
# check DATA input folder
for dirname, _, filenames in os.walk(PATH_INPUT):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


def find_choice_range(choice_curr):
    '''
    select range of choices 
    from random between +/- delta choice around choice_curr
    
    return a range
    '''
    choice_min = np.min([np.max([CHOICE_RANGE_MIN, choice_curr - DELTA_CHOICE]), 
                          CHOICE_RANGE_MAX - 2 * DELTA_CHOICE])
    choice_max = np.min([CHOICE_RANGE_MAX, 
                        np.max([CHOICE_RANGE_MIN + 2 * DELTA_CHOICE, 
                        choice_curr + DELTA_CHOICE])]) + 1
    range_choice = range(choice_min, choice_max)
    
    return range_choice

@njit
def find_choice_range_optim(choice_curr, delta_choice=DELTA_CHOICE):
    '''
    select range of choices 
    from random between +/- delta choice around choice_curr
    
    return a range
    
    use external constant var : CHOICE_RANGE_MIN & CHOICE_RANGE_MAX
    '''
    #return max(choice_curr, CHOICE_RANGE_MIN)
    
    
    choice_min = min(max(CHOICE_RANGE_MIN, choice_curr - delta_choice), 
                          CHOICE_RANGE_MAX - 2 * delta_choice)
    choice_max = min(CHOICE_RANGE_MAX, 
                        max(CHOICE_RANGE_MIN + 2 * delta_choice, 
                        choice_curr + delta_choice)) + 1
    range_choice = np.arange(choice_min, choice_max)
    
    return range_choice


def find_choice_from_day(day_curr, fam_id):
    '''
    find current choice according to day_curr and family id
    '''
    fam_days = data_choice.loc[fam_id]
    
    idx_choice_curr = fam_days[day_curr == fam_days].index
    
    if idx_choice_curr.shape[0] == 0:
        return 10 
    else:
        return idx_choice_curr[0]

def find_choice_from_day_arr(day_curr, fam_id):
    '''
    find current choice according to day_curr and family id
    
    use external constant var : arr_choice
    '''
    idx_choice_curr = np.nonzero(arr_choice[fam_id,:] == day_curr)[0]

    if idx_choice_curr.shape[0] == 0:
        return 10 
    else:
        return idx_choice_curr[0]

@njit
def find_choice_from_day_optim(day_curr, fam_id):
    '''
    find current choice according to day_curr and family id
    
    use external constant var : arr_choice
    '''
    idx_choice_curr = np.nonzero(arr_choice[fam_id,:] == day_curr)[0]

    if idx_choice_curr.shape[0] == 0:
        return 10 
    else:
        return idx_choice_curr[0]
    
    
def choose_day_prob(choice_curr, fam_id):
    '''
    find or choose the day of given choice number according to family choices
    
    if choice 10, then find a day randomly
    
    return a day
    '''
  
    choice_curr = np.array(choice_curr)
    
    vect_days_choice = data_choice.loc[fam_id, np.minimum(9, choice_curr)]
    vect_days_choice = np.array(vect_days_choice)
    
    idx_choice_10 = np.nonzero(choice_curr == 10)[0]
    if idx_choice_10.shape[0] > 0:
        nb_10 = idx_choice_10.shape[0]
        vect_all_days = np.array(range(1, 101))
        vect_prob_10 = np.array(df_prob_fam.astype("float").loc[fam_id])/             df_prob_fam.loc[fam_id].sum()
        #vect_prob_10 = np.ones(vect_all_days.shape[0])
        vect_prob_10[idx_choice_10] = 0
        vect_prob_10 = vect_prob_10 / np.sum(vect_prob_10)
        days_10 = np.random.choice(vect_all_days, size=nb_10,
                                  p=vect_prob_10)
        #print("idx_choice_10: ", idx_choice_10)
        for i_day, indice in enumerate(idx_choice_10):
            
            if vect_days_choice.shape:
                
                ''' print("days_10: ", days_10)
                print("vect_days_choice: ", np.array(vect_days_choice))
                print("vect_days_choice[indice]: ", 
                      np.array(vect_days_choice)[indice])
                print("i_day: ", i_day)
                print("days_10[i_day]: ", days_10[i_day])
                print("indice : ", indice)'''

                vect_days_choice[indice] = days_10[i_day]
            else:
                vect_days_choice = np.array(days_10[i_day])
            
        return vect_days_choice
    else:
        return vect_days_choice
    
def choose_day_prob_arr(choice_curr, fam_id):
    '''
    find or choose the day of given choice number according to family choices
    
    if choice 10, then find a day randomly
    
    return a day
    '''
    choice_curr = np.array(choice_curr)
    
    vect_days_choice = arr_choice[fam_id, np.minimum(9, choice_curr)]
    #vect_days_choice = np.array(vect_days_choice)
    
    idx_choice_10 = np.nonzero(choice_curr == 10)[0]
    if idx_choice_10.shape[0] > 0:
        nb_10 = idx_choice_10.shape[0]
        vect_all_days = np.array(range(1, 101))
        vect_prob_10 = arr_prob_fam[fam_id]/arr_prob_fam[fam_id].sum()
        #np.array(df_prob_fam.astype("float").loc[fam_id])/ \
        #    df_prob_fam.loc[fam_id].sum()
        #vect_prob_10 = np.ones(vect_all_days.shape[0])
        vect_prob_10[idx_choice_10] = 0
        vect_prob_10 = vect_prob_10 / np.sum(vect_prob_10)
        days_10 = np.random.choice(vect_all_days, size=nb_10,
                                  p=vect_prob_10)
        #print("idx_choice_10: ", idx_choice_10)
        for i_day, indice in enumerate(idx_choice_10):
            
            if vect_days_choice.shape:
                
                vect_days_choice[indice] = days_10[i_day]
            else:
                vect_days_choice = np.array(days_10[i_day])
            
        return vect_days_choice
    else:
        return vect_days_choice    

@njit
def choose_day_prob_optim(choice_curr, fam_id):
    #def choose_day_prob_optim(choice_curr, fam_id, arr_choice=arr_choice):
    '''
    V1.1 : correction about no selection of days choices for choice 10 
    
    find or choose the day of given choice number according to family choices
    
    if choice 10, then find a day randomly
    
    return a day
    
    use external constant vars : arr_choice & arr_prob_fam
    '''
    #choice_curr = np.array(choice_curr)
        
    idx_choice_10 = []
    for I in np.arange(choice_curr.shape[0]):
        if choice_curr[I] == 10:
            idx_choice_10.append(I)
            
    idx_choice_10 = np.array(idx_choice_10)
    
    vect_days_choice = []
    for J in np.arange(choice_curr.shape[0]):
        vect_days_choice.append(arr_choice[fam_id, 
                                           np.minimum(9, choice_curr[J])])
        
    vect_days_choice = np.array(vect_days_choice)
    #print("idx_choice_10 " , idx_choice_10)
    if idx_choice_10.shape[0] > 0:
        nb_10 = idx_choice_10.shape[0]
        vect_all_days = np.arange(1, 101)
        vect_prob_10 = arr_prob_fam[fam_id].copy()
        vect_prob_10[arr_choice[fam_id]] = 0
        vect_prob_10 = vect_prob_10 / np.sum(vect_prob_10)
        #print("vect_prob_10 ",vect_prob_10)
        days_10 = rand_choice_nb(vect_all_days, size=nb_10,
                                  prob=vect_prob_10)
        #print("days_10 ", days_10)
        for i_day, indice in enumerate(idx_choice_10):
            vect_days_choice[indice] = days_10[i_day]         
        return vect_days_choice
    else:
        return vect_days_choice 

def mutation_day(day_curr, fam_id, nb_mut=1, flag_prob=False):
    # function mutation of days : around choices
    # day_curr -> choice_curr -> range choices -> 
    # choose randomly 1 choice -> 1 day 
    # fam_id = 0
    # day_curr = 100
    # mutation_day(day_curr, fam_id, 10)
    choice_curr = find_choice_from_day_arr(day_curr, fam_id)
    #print("choice_curr: ", choice_curr)
    range_choices = find_choice_range(choice_curr)
    #print("range_choices: ", np.array(range_choices))
    
    if flag_prob:
        vect_prob = arr_prob[family_size_dict[fam_id]-2, range_choices]
        #np.array(df_prob.loc[data.loc[fam_id,"n_people"], 
        #                     ['choice_{}'.format(i) for i in range_choices]])
        vect_prob = vect_prob / vect_prob.sum()
        #print("vect_prob: ", vect_prob)
        choice_new = np.random.choice(range_choices, size=nb_mut,
                                     p=vect_prob)
    else:
        choice_new = np.random.choice(range_choices, size=nb_mut)
        
    #print("choice_new: ", choice_new)
    return choose_day_prob_arr(choice_new, fam_id)


def find_day_range(day_curr):
    '''
    select range of days from random between +/- delta day around day_curr
    
    return a range
    '''
    day_min = np.min([np.max([DAY_RANGE_MIN, day_curr-DELTA_DAYS]), 
                      DAY_RANGE_MAX - 2 * DELTA_DAYS])
    day_max = np.min([DAY_RANGE_MAX, np.max([DAY_RANGE_MIN + 2 * DELTA_DAYS, 
                                             day_curr + DELTA_DAYS])]) + 1
    range_day = range(day_min, day_max)

    return range_day


def generate_pop(seed_indiv=None, nb_pop=None, r_mut=None):
    '''
    Generate first polulation from one seed individual
    - seed_indiv # best submission 
    - nb_pop # number of individual
    - r_mut # ratio of individual who mutate for each family
    
    return dataFrame population
    '''
    t_fit_0 = time.time()
    # GENERATION OF FIRST POP:
    print("Generating population : ")
    # definitions:
    if seed_indiv is None:
        seed_indiv = submission
    if nb_pop is None:
        nb_pop = NB_FIRST_POP
    if r_mut is None:
        r_mut = R_FIRST_MUT
    #print("seed_indiv: ", seed_indiv)
    print("nb_pop: ", nb_pop)
    print("r_mut: ", r_mut)
    # initialize with same previous best indiv.
    df_pop = pd.DataFrame(index=range(0, nb_pop), 
                          columns=seed_indiv["family_id"])

    for fam_id in seed_indiv["family_id"]:
        df_pop[fam_id] = seed_indiv.at[fam_id,'assigned_day']

    # create nb_pop family choices from baseline :
    # use probabilies
    # df_pop : contains number choices of all the population of 5000 families
    # df_pop = f(indiv., family)
    # for each family, create n choice among their first choice
    # use day probabilities specific for each family
    for fam_id in data.index: # data = f(family, num choice)
        # day current is the old best seed_indiv day for this family
        day_curr = seed_indiv.at[fam_id, 'assigned_day']
        # find range around day curr +/- DELTA
        range_curr = find_day_range(day_curr)
        # retrict probabilities to range. use df_prob_fam = f(fam_id, day)
        day_prob = df_prob_fam.astype("float").loc[fam_id, range_curr]/             df_prob_fam.loc[fam_id, range_curr].sum()
        # choose randomly with probabilities days around old best day
        vect_pop_mutated = np.random.choice(np.array(range_curr), 
                                size=nb_pop, 
                                p=np.array(day_prob))

        # apply the new days only a part of pop : r_mut [-]
        range_mut = np.random.choice(range(0, nb_pop), 
                                     size=int(r_mut*nb_pop))

        df_pop.loc[range_mut, fam_id] = vect_pop_mutated[range_mut]

    # keep the best : 
    df_pop.loc[0] = seed_indiv['assigned_day']
    
    print("Generation population is done.")
    t_fit_1 = time.time()
    print("Timing : ", t_fit_1 - t_fit_0)
    
    return df_pop

@njit
def generate_crossing(arr_pop_in):
    '''
    function to generate crossing
    2 parents give 2 children
    Crossing point is randomly chosen
    
    input the current poulation array
    
    return new array of whole population
    '''
    # Do the Crossover between pair indiv.
    # 1 Cross point is ramdomly choosen (prob uniform)
    # example : 
    # 1-2-3\  /5-8-9-1-3-4-9  
    #       \/
    # 5-6-5/ \4-5-6-7-8-9-10
    #
    # give : 
    #
    # 1-2-3--4-5-6-7-8-9-10
    # 5-6-5--5-8-9-1-3-4-9 
    # create pairs : ramdomly
    arr_pop = arr_pop_in.copy()
    vect_indiv = np.arange(arr_pop.shape[0])
    vect_fam = np.arange(arr_pop.shape[1])
    vect_fam = vect_fam[2:-2]
    
    # method 1 : each 2 parents create 2 children (by replacement)
    arr_pairs = np.random.choice(vect_indiv, replace=False,
                        size=(int(arr_pop.shape[0]/2), 2))
    
    
    # loop over pairs of indiv.
    for indice in np.arange(arr_pairs.shape[0]):
        id_0 = arr_pairs[indice, 0]
        id_1 = arr_pairs[indice, 1]
        
        # random point of crossover (among families)
        fam_id_cross = np.random.choice(vect_fam)
        
        # find parts of first new indiv    
        vect_id_0_part_0 = arr_pop[id_0].take(np.arange(fam_id_cross))
        vect_id_0_part_1 = arr_pop[id_1].take(np.arange(fam_id_cross, 
                                                   arr_pop.shape[1]))
    
        # find parts of second new indiv
        vect_id_1_part_0 = arr_pop[id_1].take(np.arange(fam_id_cross))
        vect_id_1_part_1 = arr_pop[id_0].take(np.arange(fam_id_cross, 
                                                  arr_pop.shape[1])) 
    
        # replace 2 parents by 2 children
        arr_pop[id_0,:] = np.concatenate((vect_id_0_part_0, vect_id_0_part_1))
        arr_pop[id_1,:] = np.concatenate((vect_id_1_part_0, vect_id_1_part_1))
      
    return arr_pop



def create_df_prob_day_fam(df_prob_day, df_prob):
    df_prob_fam = pd.DataFrame(index = data.index, columns=df_prob_day.index)
    for fam_id in df_prob_fam.index:
        # give at first to each families the same day probabilities
        df_prob_fam.loc[fam_id] = df_prob_day["prob"]
        # and add prob for each day choosen by families
        for choice in list_choice_all: 
            prob_curr = df_prob.at[data.at[fam_id, "n_people"], choice]
            day_curr = data.at[fam_id, choice]
            # add this prob of these days to family into df_prob_fam
            df_prob_fam.loc[fam_id, day_curr] += prob_curr
            #print(day_curr)
            #print(prob_curr)
        df_prob_fam.loc[fam_id] =             df_prob_fam.loc[fam_id] / df_prob_fam.loc[fam_id].sum()
    return df_prob_fam

def create_df_prob_day_fam_optim(df_prob_day, df_prob):
    '''
    Creation of probabilities for each families and each days
    We give at first to each families the same day probabilities df_prob_day.
    (df_prob_day is inversely proportional to sum of all choices of this day)
    And we add for each family their probability df_prob 
    (df_prob depends to )
    info : Optimized version
    
    input : df_prob_day, df_prob
    ouput : df_prob_fam
    '''
    
    arr_prob_fam = np.zeros([data.shape[0], df_prob_day.shape[0]])
    arr_prob = np.array(df_prob)
    list_choice_all = ['choice_{}'.format(n) for n in range(0, 10)]
    arr_data = np.array(data.filter(items=list_choice_all))
    for fam_id in np.arange(data.shape[0]):
        # give at first to each families the same day probabilities
        
        #df_prob_fam.loc[fam_id] = df_prob_day["prob"]
        arr_prob_fam[fam_id] = df_prob_day["prob"].values
        # and add prob for each day choosen by families
        for choice in np.arange(df_prob.shape[1]-1): 
            prob_curr = arr_prob[family_size_dict[fam_id]-2, choice]
            day_curr = arr_data[fam_id, choice]
            # add this prob of these days to family into df_prob_fam
            arr_prob_fam[fam_id, day_curr-1] += prob_curr
            #print(day_curr)
            #print(prob_curr)
        arr_prob_fam[fam_id] =             arr_prob_fam[fam_id] / arr_prob_fam[fam_id].sum()
        
    df_prob_fam = pd.DataFrame(index = data.index, columns=df_prob_day.index, 
                              data = arr_prob_fam)
    
    return df_prob_fam



@njit#(parallel=True, fastmath=True)
def rand_choice_nb(arr, size=1, prob=None):
    """
    numba compatible vesrion of np.random.choice(arr, size=size, prob=prob)
    
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    
    source : https://github.com/numba/numba/issues/2539
    """
    if prob is not None:
        list_value=[]
        for I in prange(size):
            list_value.append(arr[np.searchsorted(np.cumsum(prob), 
                                                  np.random.random(), 
                                                  side="right")])
        return np.array(list_value)
    else:
        return np.random.choice(arr, size=size)




def find_max_same_indiv(arr_pop):
    '''
    Counts max number of same indiv among population
    return only max number of same indiv
    '''
    arr_unique, arr_counts = np.unique(arr_pop, axis=0, return_counts=True)
    nb_same_indiv = np.max(arr_counts) - 1
    return nb_same_indiv



def selection_prob(df_cost, df_pop=None, pow_selection=0.3, flag_ouput=False,
                   nb_best_keep = NB_BEST_KEEP):
    '''
    Calculate prob for selection of best indiv. among pop
    Can return also best individuates : df_best, and their cost : df_cost_best
    '''
    df_cost_sort = df_cost.sort_values(by="cost")
    df_cost_sort["rank"] = range(df_cost.shape[0]+1,1,-1 )
    arr_select_prob = np.zeros(df_cost_sort.shape[0])
    arr_select_prob[df_cost_sort.index] = df_cost_sort["rank"].values
    arr_select_prob = (arr_select_prob)**POW_SELECTION
    arr_select_prob = arr_select_prob / np.sum(arr_select_prob)

    if flag_ouput:
        list_index_best = df_cost_sort.iloc[range(0,nb_best_keep)].index
        df_best = df_pop.loc[list_index_best]
        df_cost_best = df_cost.loc[list_index_best]
        return arr_select_prob, df_best, df_cost_best
    else:
        return arr_select_prob

@njit
def selection_prob_arr(arr_cost, arr_pop=None, pow_selection=0.3, 
                       flag_ouput=False,
                   nb_best_keep = NB_BEST_KEEP):
    '''
    Calculate prob for selection of best indiv. among pop
    return also best individuates : arr_best, and their cost : arr_cost_best
    '''
    indice_cost = np.argsort(arr_cost)
    rank_range = np.arange(arr_cost.shape[0]+1,1,-1)
    arr_select_prob = np.zeros(arr_cost.shape[0])
    K=0
    for indice in indice_cost:
        arr_select_prob[indice] = rank_range[K]
        K=K+1
    arr_select_prob = (arr_select_prob)**POW_SELECTION
    arr_select_prob = arr_select_prob / np.sum(arr_select_prob)

    list_index_best = indice_cost[np.arange(0, nb_best_keep)]
    arr_best = arr_pop[list_index_best].copy()
    arr_cost_best = arr_cost[list_index_best].copy()
    return arr_select_prob, arr_best, arr_cost_best
    

def pop_choices_info(df_pop):
    '''
    
    Show information about population df_pop
    
    outputs :
    - df_choices  : assignation day for each families for all pop
    - df_des_choices : describe of df_dhoices
    -  std_mean = Mean Standard deviation over families for whole population 
    
    '''
    
    @njit
    def find_pop_choices(arr_pop):
        '''
        Determine all choices of the population, from days
        '''
        arr_choices = np.zeros((arr_pop.shape[0], arr_pop.shape[1]))*np.nan
        for fam_id in range(arr_pop.shape[1]):
            for indice in range(arr_pop.shape[0]):
                arr_choices[indice, fam_id] =                     find_choice_from_day_optim(arr_pop[indice, fam_id], fam_id)
        return arr_choices
    
    arr_choices = find_pop_choices(df_pop.values)   
    df_choices = pd.DataFrame(arr_choices.astype(np.int64), 
                              index = df_pop.index,
                              columns=df_pop.columns) 
    arr_choices = find_pop_choices(df_pop.values)   
    df_choices = pd.DataFrame(arr_choices.astype(np.int64), 
                              index = df_pop.index,
                              columns=df_pop.columns) 
    df_des_choices = df_choices.describe()
    std_mean = df_des_choices.loc["std"].mean()
    print("Mean Standard deviation over families for whole population : ", 
         std_mean)
    print("Info about std: ",df_des_choices.loc["std"].describe())
    return df_choices, df_des_choices, std_mean

def create_seek_ranges(nb_first_seed=NB_FIRST_SEED):
    
    '''create df_range : contains all path to seek optimum'''
    
    df_range = pd.DataFrame(index=range(0, nb_first_seed), 
                            columns=range(0, submission.shape[0]))

    df_range.loc[0] = np.array(range(0, submission.shape[0]))

    df_range.loc[1] = np.array(range(submission.index.max(), 
                                     submission.index.min()-1, -1))  
    # generate start points
    start_pt = np.linspace(0, submission.shape[0], 
                           num=int((NB_FIRST_SEED)/2)+1, dtype="int")
    start_pt = start_pt[1:-1]
    start_pt

    # create range order for seeking
    indice = 2
    for st_id, _ in enumerate(start_pt):
        df_range.loc[indice+st_id] = np.concatenate((np.array(range(start_pt[st_id], 
                                                    submission.shape[0])), 
                   np.array(range(0, start_pt[st_id]))))

    indice = 6
    for st_id, _ in enumerate(start_pt):
        df_range.loc[indice+st_id] =             np.concatenate((range(start_pt[st_id], 0-1, -1), 
                    range(submission.shape[0]-1, start_pt[st_id], -1)))

    return df_range 


# In[4]:


@njit(parallel=True, fastmath=True)
def generate_crossing_prob(arr_pop_in, p=None , n_indiv=None, r_cross=1):
    '''
    function to generate crossing indiv  (version with probabilities)
    V1.1 : add force create new child if same parents by changing parent
           and limitation to start - 1 end end - 1 for crossing point.
           
    time exec : 3.5ms 1000 children
    
    2 parents give 1 child
    
    Crossing point is randomly chosen
    
    input the current population array
    
    return new array of whole population
    
    r_cross is more a target of ratio of crossover individuates.
    if not enough children, then generate more than the ratio
    
    EXAMPLE : 
    arr_test = np.array([[1,2,3,4,5,6,7,8,9,1], [2,5,8,9,5,5,5,5,5,2],
                [3,1,1,1,5,5,5,5,5,3], [4,6,6,6,6,9,9,9,9,4], 
                [5,6,6,6,6,9,9,9,9,5]])
    arr_prob_test = 1 / np.array([1,2,3,4,5])
    arr_prob_test = arr_prob_test / np.sum(arr_prob_test)
    arr_test_new = generate_crossing_prob(arr_test, p=arr_prob_test, n_indiv=11)
    plt.plot(arr_test_new[0]-arr_test[1])
    
    '''
    # Do the Crossover between pair indiv.
    # 1 Cross point is ramdomly choosen (prob uniform)
    # example : 
    # 1-2-3\  /5-8-9-1-3-4-9  
    #       \/
    # 5-6-5/ \4-5-6-7-8-9-10
    #
    # give : 
    #
    # 1-2-3--4-5-6-7-8-9-10
    # 5-6-5--5-8-9-1-3-4-9 
    # create pairs : ramdomly

    # check number of indiv : 
    nb_pop = arr_pop_in.shape[0]
    # calculate number of new children
    n_indiv_cross = int(n_indiv*r_cross)
    if n_indiv_cross == 0:
        n_indiv_cross = 1
    # if not enough to create n_indiv, add crossover indiv:
    if n_indiv > nb_pop + n_indiv_cross:
        n_indiv_cross = n_indiv - nb_pop
    #print("n_indiv_cross : ", n_indiv_cross) 
    # initialize output
    arr_pop = np.zeros((int(n_indiv), arr_pop_in.shape[1]), dtype=np.int64)
    #print('arr_pop.shape :', arr_pop.shape)
    
    # preparation for loop over pairs : 
    vect_parents = np.arange(arr_pop_in.shape[0])
    vect_fam = np.arange(arr_pop.shape[1])
    # NOT replacing all part of parents : limit range
    vect_fam = vect_fam[1:-1]
    arr_pairs = np.zeros((int(n_indiv_cross), 2), dtype=np.int64)
    # create pairs : select best ones more frequently first
    for I in prange(int(arr_pairs.shape[0])):
        arr_pairs[I] = rand_choice_nb(vect_parents, size=2, prob=p)
        #arr_pairs[I] = np.random.choice(vect_parents, size=2)
        K=0
        # check and force to new children from same parents
        # patch dirty ! but njit doenst work with random choice prob 
        # & non replace...
        while (arr_pairs[I,0] == arr_pairs[I,1]) & (K < 1000):
            arr_pairs[I] = rand_choice_nb(vect_parents, size=2, prob=p)
            # test if same indiv : 
            while not(np.any(arr_pop_in[arr_pairs[I,0]] -                              arr_pop_in[arr_pairs[I,1]])) & (K < 1000):
                arr_pairs[I] = rand_choice_nb(vect_parents, size=2, prob=p)
                K=K+1
            K=K+1
            
    # for all pairs wanted as output  
    for indice in prange(int(arr_pairs.shape[0])):
        # indice of 2 parents
        id_0 = arr_pairs[indice, 0]
        id_1 = arr_pairs[indice, 1]
        # random point of crossover (among families)
        fam_id_cross = np.random.choice(vect_fam)
        # find parts of first new indiv    
        vect_id_0_part_0 = arr_pop_in[id_0].take(np.arange(fam_id_cross))
        vect_id_0_part_1 = arr_pop_in[id_1].take(np.arange(fam_id_cross, 
                                                   arr_pop_in.shape[1]))
        # create 1 children 
        arr_pop[indice] = np.concatenate((vect_id_0_part_0, 
                                            vect_id_0_part_1))
    # if crossing not all pop
    if r_cross < 1:
        # keep some of best parents 
        nb_parents_keep = n_indiv - n_indiv_cross
        #print("nb_parents_keep: ", nb_parents_keep)
        #vect_parent_keep = rand_choice_nb(vect_parents, size=nb_parents_keep, 
        #                                  prob=p)
        # we keep only the best parents
        inv_ind_best = np.argsort(p)
        arr_pop_ranked = arr_pop_in[inv_ind_best, :]
        
        #print('inv_ind_best.shape: ', inv_ind_best.shape)
        #print('arr_pop_in red: ', arr_pop[n_indiv_cross:, :].shape)
        #print('arr_pop_ranked.shape red: ',  arr_pop_ranked[-nb_parents_keep:, :].shape)
        #print('arr_pop_ranked.shape: ', arr_pop_ranked.shape)
        arr_pop[n_indiv_cross:, :] =             arr_pop_ranked[-nb_parents_keep:, :]
        
        return arr_pop
    else:  
        return arr_pop
    
def boost_diff_browsing(arr_choice, best, arr_range):
    '''
    Boosting simple by seeking by ranges 
    Simple baseline optimisation following different path range of families.
    Forward/Backward
    
    input : arr_choice : array of choice days by families
            best : best submission
            arr_range : ranges of paths into families
    return :  arr_sub : array of submissions
            arr_score : array of their score/cost
    
    example :
    arr_sub, arr_score = boost_diff_browsing(arr_choice, best, arr_range)
    
    '''
    # Create baselines # optimized version
    t_fit_0 = time.time()
    # Create baselines # optimized version
    start_cost = cost_function_optim(best)
    print("Start cost: ", start_cost)
    
    # prepare output : best submission seeking in different range walk around
    arr_sub = np.zeros((arr_range.shape[0], best.shape[0])).astype(np.int64)
    arr_score = np.zeros(arr_range.shape[0])
    
    #new = best.copy()
    
    for indice in np.arange(arr_range.shape[0]):
    #for indice in df_range.index:
        # choose current range in df_range
        range_optim = arr_range[indice]
        #range_optim = df_range.loc[indice]

        # initiate first inviduate 
        new = best.copy()
        cost_best = start_cost

        # loop over each family with this current range       
        for fam_id in range_optim:
            # loop over each family choice
            for pick in range(10):
                day = arr_choice[fam_id, pick]
                temp = new.copy()
                temp[fam_id] = day # add in the new pick
                cost_curr = cost_function_optim(temp) # test cost
                if cost_curr < cost_best:
                    new = temp.copy()
                    cost_best = cost_curr
                    #print(f'...Baseline #{indice} current best Score: {cost_best}')
                
        print(f'Baseline #{indice} Score: {cost_best}')
        arr_sub[indice] = new
        arr_score[indice] = cost_best
        
        
    # timing
    t_fit_1 = time.time()
    print("Timing: ", t_fit_1 - t_fit_0)
        
    return arr_sub, arr_score

def generate_pop_choices(seed_indiv=None, nb_pop=None, r_mut=None, 
                         delta_choice=DELTA_CHOICE_FIRST_POP):
    '''
    Generate first polulation from one seed individual by family choices
    - seed_indiv # best submission 
    - nb_pop # number of individual
    - r_mut # ratio of individual who mutate for each family
    
    return dataFrame population
    
    external argument : data & 
    '''
    t_fit_0 = time.time()
    # GENERATION OF FIRST POP:
    print("Generating population : ")
    # definitions:
    #if seed_indiv is None:
    #    seed_indiv = submission
    if nb_pop is None:
        nb_pop = NB_FIRST_POP
    if r_mut is None:
        r_mut = R_FIRST_MUT
    #print("seed_indiv: ", seed_indiv)
    print("nb_pop: ", nb_pop)
    print("r_mut: ", r_mut)
    # initialize with same previous best indiv.
    #df_pop = pd.DataFrame(index=range(0, nb_pop), 
    #                      columns=seed_indiv["family_id"])
    df_pop = pd.DataFrame(index=range(0, nb_pop), 
                          columns=range(seed_indiv.shape[0]))
    #print("df_pop.shape ", df_pop.shape)
    #for fam_id in seed_indiv["family_id"]:
    for fam_id in range(seed_indiv.shape[0]):
        #df_pop[fam_id] = seed_indiv.at[fam_id,'assigned_day']
        df_pop[fam_id] = seed_indiv[fam_id]

    # create nb_pop family choices from baseline :
    # use probabilies
    # df_pop : contains number choices of all the population of 5000 families
    # df_pop = f(indiv., family)
    # for each family, create n choice among their first choice
    # use day probabilities specific for each family
    for fam_id in data.index: # data = f(family, num choice)
        # day current is the old best seed_indiv day for this family
        #day_curr = seed_indiv.at[fam_id, 'assigned_day']
        day_curr = seed_indiv[fam_id]
        vect_pop_mutated = mutation_day_optim(day_curr, fam_id, nb_mut=nb_pop, 
                                        flag_prob=True, 
                                        delta_choice=delta_choice)
        
        # apply the new days only a part of pop : r_mut [-]
        range_mut = np.random.choice(range(0, nb_pop), 
                                     size=int(r_mut*nb_pop))
        #range_mut = rand_choice_nb(range(0, nb_pop), size=int(r_mut*nb_pop))
        #print("df_pop.shape: ", df_pop.shape)
        #print("range_mut.shape: ", range_mut.shape)
        #print("fam_id ", fam_id)
        df_pop.loc[range_mut, fam_id] = vect_pop_mutated[range_mut]

    # keep the best : 
    #df_pop.loc[0] = seed_indiv['assigned_day']
    df_pop.loc[0] = seed_indiv
    print("Generation population is done.")
    t_fit_1 = time.time()
    print("Timing : ", t_fit_1 - t_fit_0)
    
    return df_pop

@njit
def generate_pop_choices_optim(seed_indiv=None, 
                               nb_pop=NB_FIRST_POP, 
                               r_mut=R_FIRST_MUT, 
                               delta_choice=DELTA_CHOICE_FIRST_POP):
    '''
    Generate first polulation from one seed individual by family choices
    - seed_indiv # best submission 
    - nb_pop # number of individual
    - r_mut # ratio of individual who mutate for each family
    
    return array population
    '''
    # GENERATION OF FIRST POP:
    print("Generating population : ")
    # definitions:
    print("nb_pop: ", nb_pop)
    print("r_mut: ", r_mut)
    print("delta choice: ", delta_choice)
    # initialize with same previous best indiv.
    arr_pop = np.zeros((nb_pop, seed_indiv.shape[0]), dtype=np.int64)
    
    for fam_id in range(seed_indiv.shape[0]):
        arr_pop[:, fam_id] = seed_indiv[fam_id]
        
    # create nb_pop family choices from baseline :
    # use probabilies
    # df_pop : contains number choices of all the population of 5000 families
    # df_pop = f(indiv., family)
    # for each family, create n choice among their first choice
    # use day probabilities specific for each family
    for fam_id in range(seed_indiv.shape[0]): # data = f(family, num choice)
        # day current is the old best seed_indiv day for this family
        day_curr = seed_indiv[fam_id]
        vect_pop_mutated = mutation_day_optim(day_curr, fam_id, nb_mut=nb_pop, 
                                        flag_prob=True, 
                                        delta_choice=delta_choice)
        
        # apply the new days only a part of pop : r_mut [-]
        range_mut = np.random.choice(np.arange(nb_pop), 
                                     size=int(r_mut*nb_pop))
        arr_pop[range_mut, fam_id] = vect_pop_mutated[range_mut]

    # keep the best : 
    arr_pop[0] = seed_indiv
    
    print("Generation population is done.")
    
    return arr_pop

@njit
def removeDups(arr): 
    # Python 3 program to remove the 
    # duplicates from the array
    
    # example : removeDups(np.array([9, 8, 8, 8, 4, 5, 6, 7, 8, 9]))
    # >> (array([4, 5, 6, 7, 1, 0]), array([4, 5, 6, 7, 8, 9]))
    arr_u = np.unique(arr)
    nb_arr = arr.shape[0]
    nb_arr_u = arr_u.shape[0]
    if nb_arr == nb_arr_u:
        indices = np.arange(nb_arr)
        return indices, arr

    #indices = np.arange(arr.shape[0])
    indices = np.empty(nb_arr_u, dtype=np.int64)
    K=0
    for u_val_curr in arr_u:
        i_arr = 0
        for val_curr in arr:
            if val_curr == u_val_curr:
                indices[K] = i_arr
                K=K+1
                break
            i_arr = i_arr + 1
    
    return indices, arr[indices]
            


# In[5]:


def fun_find_choices_sub(my_days):
    nb_fam = my_days.shape[0]
    my_choices = np.empty(nb_fam)
    for fam_id in range(nb_fam):
        my_choices[fam_id] =             find_choice_from_day_optim(my_days[fam_id], fam_id)
    return my_choices

def plot_std_choice_pop(df_pop, df_des_choices):
    ax = []
    d_plot = 1000
    for I in range(int(df_pop.shape[1]/d_plot)):
        fig = plt.figure(figsize=(16, 4))
        ax_curr = fig.gca() 
        ax.append(ax_curr)

        error_margin =             1.96*df_des_choices.loc["std",range(I*d_plot, 
                                                I*d_plot+d_plot)]/(df_pop.shape[0])**0.5

        plt.plot(range(I*d_plot, I*d_plot+d_plot), error_margin,'.')
            
    


def plot_delta_choice_pop(df_pop, df_des_choices):
    ax = []
    d_plot = 1000
    for I in range(int(df_pop.shape[1]/d_plot)):
        fig = plt.figure(figsize=(16, 4))
        ax_curr = fig.gca() 
        ax.append(ax_curr)

        error_margin =             1.96*df_des_choices.loc["std",range(I*d_plot, 
                                                I*d_plot+d_plot)]/(df_pop.shape[0])**0.5

        plt.plot(range(I*d_plot, I*d_plot+d_plot), 
                 df_des_choices.loc["mean", range(I*d_plot, 
                                                  I*d_plot+d_plot)], 'o-', alpha=0.25)
        plt.plot(range(I*d_plot, I*d_plot+d_plot), 
                 df_des_choices.loc["mean", range(I*d_plot, 
                                                  I*d_plot+d_plot)] + error_margin,'.')
        plt.plot(range(I*d_plot, I*d_plot+d_plot), 
                 df_des_choices.loc["mean", range(I*d_plot, 
                                                  I*d_plot+d_plot)] - error_margin,'.')


# In[6]:


fpath = PATH_INPUT + '/santa-2019-workshop-scheduling/family_data.csv'
data = pd.read_csv(fpath, index_col='family_id')

fpath = PATH_INPUT + '/santa-2019-workshop-scheduling/sample_submission.csv'
submission = pd.read_csv(fpath, index_col='family_id')

data_choice = data.iloc[:,range(0,10)]
data_choice.columns = range(0,10)
# patch for optimization
arr_choice = np.array(data_choice)
data_choice.head()


# In[7]:


data.head()


# In[8]:


submission.head()


# In[9]:


#family_size_dict[fam_id]
arr_n_people = data["n_people"].values

@njit
def mutation_day_optim(day_curr, fam_id, nb_mut=1, flag_prob=False, 
                       arr_n_people=arr_n_people, delta_choice=DELTA_CHOICE):
    # function mutation of days : around choices
    # day_curr -> choice_curr -> range choices -> 
    # choose randomly 1 choice -> 1 day 
    # fam_id = 0
    # day_curr = 100
    # mutation_day(day_curr, fam_id, 10)
    # time cpu = #1=4us & 1000=191us
    choice_curr = find_choice_from_day_optim(day_curr, fam_id)
    #print("choice_curr: ", choice_curr)
    range_choices = find_choice_range_optim(choice_curr, 
                                            delta_choice=delta_choice)
    #print("range_choices: ", np.array(range_choices))
    
    if flag_prob:
        #vect_prob = arr_prob[family_size_dict[fam_id]-2, range_choices]
        vect_prob = arr_prob[arr_n_people[fam_id]-2].take(range_choices)
        #np.array(df_prob.loc[data.loc[fam_id,"n_people"], 
        #                     ['choice_{}'.format(i) for i in range_choices]])
        vect_prob = vect_prob / vect_prob.sum()
        #print("vect_prob: ", vect_prob)
        choice_new = rand_choice_nb(range_choices, size=nb_mut, 
                                    prob=vect_prob)
                        #np.random.choice(range_choices, size=nb_mut,
                        #             p=vect_prob)
    else:
        choice_new = np.random.choice(range_choices, size=nb_mut)
        
    #print("type choice_new: ", type(choice_new))
    return choose_day_prob_optim(choice_new, fam_id)


# In[10]:


@njit#(parallel=True, fastmath=True)
def fun_vect_mut(arr_pop_in, r_pop_mut=R_POP_MUT, r_mut=R_MUT, 
                 delta_choice=DELTA_CHOICE):
    '''
    Mutation of all population
    
    input arr_pop
    output new arr_pop
    
    Example : 
    
    arr_test = np.array([[1,2,3,4,5,6,7,8,9,10], [4,5,8,9,5,5,5,5,5,5],
                    [1,1,1,1,5,5,5,5,5,5], [6,6,6,6,6,9,9,9,9,9]])
    arr_test = np.concatenate((arr_test, np.minimum(10,arr_test+1)))

    R_POP_MUT =1
    R_MUT =1
    DELTA_CHOICE = 2
    print("R_POP_MUT ", R_POP_MUT)
    print("R_MUT ", R_MUT)
    print("DELTA_CHOICE ", DELTA_CHOICE)
    %timeit arr_test_mut = fun_vect_mut(arr_test, r_pop_mut=R_POP_MUT, r_mut=R_MUT, delta_choice=DELTA_CHOICE)
    
    # timeit : 342 µs ± 31.9 µs
    
    # timeit : 32 ms on pop = 1000,  R_POP=0.1 R_MUT=0.01 DELTA_CHOICE=2
    '''
    #print("fun_vect_mut : r_pop_mut: ", r_pop_mut)
    #print("fun_vect_mut : r_pop_mut: ", r_mut)
    #print("fun_vect_mut : delta_choice: ", delta_choice)
    arr_pop = arr_pop_in.copy()
    nb_fam = arr_pop.shape[1]
    np_pop = arr_pop.shape[0]
    nb_mut = int(r_pop_mut*arr_pop.shape[0])
    # indice of mutated indiv.
    indice_mut = np.random.choice(np.arange(np_pop), size=nb_mut, 
                                  replace=False)
    #indice_mut = np.random.permutation(arr_pop.shape[0])
    #indice_mut = indice_mut[0:nb_mut]
    
    # number of family who mutate for each mutated indiv. : R_MUT * nb families
    nb_fam_mut = int(r_mut*nb_fam)
    # loop over indice of mutated indiv to apply mutation to number of family
    # who mutated :
    #print("nb_mut: ", nb_mut)
    #print("nb_fam_mut: ", nb_fam_mut)
    # for each indiv to mutate, select a random group of families to mutate
    for idx_mut in prange(indice_mut.shape[0]):
        indice = indice_mut[idx_mut]
        # faster version : multiple mutation of same fam is possible
        #fam_mut = np.random.choice(np.arange(arr_pop.shape[1]), size=nb_fam_mut)
        # slower version : one familly can mutate only once
        #fam_mut = np.random.permutation(nb_fam) # better but slower
        #fam_mut = fam_mut[0:nb_fam_mut] # better but slower
        # new version
        fam_mut = np.random.choice(np.arange(nb_fam), replace=False, 
                                   size=nb_fam_mut)
        # for each family to mutate, find a new day among their choices
        for idx_fam in np.arange(nb_fam_mut):
            fam_id = fam_mut[idx_fam]
            arr_pop[indice, fam_id] = mutation_day_optim(
                            arr_pop[indice, fam_id], fam_id, nb_mut=1, 
                            flag_prob=True, delta_choice=delta_choice)[0]

    return arr_pop


# In[11]:


family_size_dict = data[['n_people']].to_dict()['n_people']

cols = [f'choice_{i}' for i in range(10)]
choice_dict = data[cols].to_dict()

N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125

# from 100 to 1
days = list(range(N_DAYS,0,-1))


# In[12]:


def cost_function(prediction, flag_prompt=False):

    penalty = 0

    # We'll use this to count the number of people scheduled each day
    daily_occupancy = {k:0 for k in days}
    
    # Looping over each family; d is the day for each family f
    for f, d in enumerate(prediction):

        # Using our lookup dictionaries to make simpler variable names
        n = family_size_dict[f]
        choice_0 = choice_dict['choice_0'][f]
        choice_1 = choice_dict['choice_1'][f]
        choice_2 = choice_dict['choice_2'][f]
        choice_3 = choice_dict['choice_3'][f]
        choice_4 = choice_dict['choice_4'][f]
        choice_5 = choice_dict['choice_5'][f]
        choice_6 = choice_dict['choice_6'][f]
        choice_7 = choice_dict['choice_7'][f]
        choice_8 = choice_dict['choice_8'][f]
        choice_9 = choice_dict['choice_9'][f]

        # add the family member count to the daily occupancy
        daily_occupancy[d] += n

        # Calculate the penalty for not getting top preference
        if d == choice_0:
            penalty += 0
        elif d == choice_1:
            penalty += 50
        elif d == choice_2:
            penalty += 50 + 9 * n
        elif d == choice_3:
            penalty += 100 + 9 * n
        elif d == choice_4:
            penalty += 200 + 9 * n
        elif d == choice_5:
            penalty += 200 + 18 * n
        elif d == choice_6:
            penalty += 300 + 18 * n
        elif d == choice_7:
            penalty += 300 + 36 * n
        elif d == choice_8:
            penalty += 400 + 36 * n
        elif d == choice_9:
            penalty += 500 + 36 * n + 199 * n
        else:
            penalty += 500 + 36 * n + 398 * n
    if flag_prompt:
        print("penalty for only families: ", penalty)
    # for each date, check total occupancy
    #  (using soft constraints instead of hard constraints)
    for _, v in daily_occupancy.items():
        if (v > MAX_OCCUPANCY) or (v < MIN_OCCUPANCY):
            penalty += 100000000
    if flag_prompt:
        print("daily_occupancy: ", daily_occupancy)
    # Calculate the accounting cost
    # The first day (day 100) is treated special
    accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)
    
    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = daily_occupancy[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))
        yesterday_count = today_count
    if flag_prompt:
        print("accounting_cost: ", accounting_cost)
    penalty += accounting_cost
    
    if flag_prompt:
        return penalty, accounting_cost, daily_occupancy
    else:
        return penalty


# In[13]:


"""
V 2.0 : 17/01/2020 : G.LANG : limitation when use  accounting_matrix
# About this kernel

The `cost_function` in this kernel is roughly 600x faster compared to the original kernel. 
Each function call takes roughly 24 µs.

## Quick Start

1. Import this utility file: File > Add utility script > Search Notebooks > *Type this notebook name*

2. Copy the code below to get started:
```
# Imports
import pandas as pd
import numpy as np

# The name of the kernel might change, so update this if needed
from santa_s_2019_faster_cost_function_24_s import build_cost_function

# Load Data
base_path = '/kaggle/input/santa-workshop-tour-2019/'
data = pd.read_csv(base_path + 'family_data.csv', index_col='family_id')
submission = pd.read_csv(base_path + 'sample_submission.csv', index_col='family_id')

# Build your "cost_function"
cost_function = build_cost_function(data)

# Run it on default submission file
best = submission['assigned_day'].values
start_score = cost_function(best)
```

A longer example is provided at the end.


## Note

Starting in V12, I decided to make this an utility script instead of a regular notebook.
I think this is a better use of this kernel, since you can now directly import this into
your project and use it just like an API, instead of copy-pasting the lengthy code.

I think that make this into a script forces me to keep the code cleaner.

## Reference

* (Excellent) Original Kernel: https://www.kaggle.com/inversion/santa-s-2019-starter-notebook
* First kernel that had the idea to use Numba: https://www.kaggle.com/nickel/250x-faster-cost-function-with-numba-jit
* Another great cost function optimization: https://www.kaggle.com/sekrier/fast-scoring-using-c-52-usec
* More modular output for intermediate function: https://www.kaggle.com/nickel/santa-s-2019-fast-pythonic-cost-23-s
"""

from functools import partial

## Intermediate Helper Functions
def _build_choice_array(data, n_days):
    choice_matrix = data.loc[:, 'choice_0': 'choice_9'].values
    choice_array_num = np.full((data.shape[0], n_days + 1), -1)

    for i, choice in enumerate(choice_matrix):
        for d, day in enumerate(choice):
            choice_array_num[i, day] = d
    
    return choice_array_num


def _precompute_accounting(max_day_count, max_diff):
    accounting_matrix = np.zeros((max_day_count+1, max_diff+1))
    # Start day count at 1 in order to avoid division by 0
    for today_count in range(1, max_day_count+1):
        for diff in range(max_diff+1):
            accounting_cost = (today_count - 125.0) / 400.0 * today_count**(0.5 + diff / 50.0)
            accounting_matrix[today_count, diff] = max(0, accounting_cost)
    
    return accounting_matrix


def _precompute_penalties(choice_array_num, family_size):
    penalties_array = np.array([
        [
            0,
            50,
            50 + 9 * n,
            100 + 9 * n,
            200 + 9 * n,
            200 + 18 * n,
            300 + 18 * n,
            300 + 36 * n,
            400 + 36 * n,
            500 + 36 * n + 199 * n,
            500 + 36 * n + 398 * n
        ]
        for n in range(family_size.max() + 1)
    ])
    
    penalty_matrix = np.zeros(choice_array_num.shape)
    N = family_size.shape[0]
    for i in range(N):
        choice = choice_array_num[i]
        n = family_size[i]
        
        for j in range(penalty_matrix.shape[1]):
            penalty_matrix[i, j] = penalties_array[n, choice[j]]
    
    return penalty_matrix


@njit
def _compute_cost_fast(prediction, family_size, days_array, 
                       penalty_matrix, accounting_matrix, 
                       MAX_OCCUPANCY, MIN_OCCUPANCY, N_DAYS):
    """
    Do not use this function. Please use `build_cost_function` instead to 
    build your own "cost_function".
    """
    N = family_size.shape[0]
    # We'll use this to count the number of people scheduled each day
    daily_occupancy = np.zeros(len(days_array)+1, dtype=np.int64)
    penalty = 0
    
    # Looping over each family; d is the day, n is size of that family
    for i in range(N):
        n = family_size[i]
        d = prediction[i]
        
        daily_occupancy[d] += n
        penalty += penalty_matrix[i, d]

    # for each date, check total occupancy 
    # (using soft constraints instead of hard constraints)
    # Day 0 does not exist, so we do not count it
    relevant_occupancy = daily_occupancy[1:]
    
    # patch : G.L. 12/01/2020 - begins
    incorrect_occupancy =  (relevant_occupancy > MAX_OCCUPANCY) |         (relevant_occupancy < MIN_OCCUPANCY)
    for inc_curr in incorrect_occupancy:
        if inc_curr:
            #print("inc_curr", inc_curr)
            penalty += 100000000
    #print(incorrect_occupancy)
    # patch : G.L. 12/01/2020 - ends

    # Calculate the accounting cost
    # The first day (day 100) is treated special
    init_occupancy = daily_occupancy[days_array[0]]
    accounting_cost = (init_occupancy - 125.0) / 400.0 * init_occupancy**(0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)
    
    # Loop over the rest of the days_array, keeping track of previous count
    yesterday_count = init_occupancy
    for day in days_array[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        # patch G.L. : 17/01/2020 : limit inputs
        today_count_lim = max(MIN_OCCUPANCY, min(MAX_OCCUPANCY, today_count))
        diff_lim = max(0, min(N_DAYS, diff))
        accounting_cost += accounting_matrix[today_count_lim, diff_lim]

        yesterday_count = today_count
    #print("penalty: ", penalty)   
    #print("accounting_cost: ", accounting_cost)
    return penalty, accounting_cost, daily_occupancy

def build_cost_function(data, N_DAYS=100, MAX_OCCUPANCY=300, MIN_OCCUPANCY=125):
    """
    data (pd.DataFrame): 
        should be the df that contains family information. Preferably load it from "family_data.csv".
    """
    family_size = data.n_people.values
    days_array = np.arange(N_DAYS, 0, -1)

    # Precompute matrices needed for our cost function
    choice_array_num = _build_choice_array(data, N_DAYS)
    penalty_matrix = _precompute_penalties(choice_array_num, family_size)
    # patch G.L. 12/01/2020 - begins
    accounting_matrix = _precompute_accounting(max_day_count=MAX_OCCUPANCY, 
                                               max_diff=MAX_OCCUPANCY)
    # patch G.L. 12/01/2020 - ends
    
    # Partially apply `_compute_cost_fast` so that the resulting partially applied
    # function only requires prediction as input. E.g.
    # Non partial applied: score = _compute_cost_fast(prediction, family_size, days_array, ...)
    # Partially applied: score = cost_function(prediction)
    def cost_function(prediction: np.ndarray) -> float:
        penalty, accounting_cost, daily_occupancy = _compute_cost_fast(
            prediction=prediction,
            family_size=family_size, 
            days_array=days_array, 
            penalty_matrix=penalty_matrix, 
            accounting_matrix=accounting_matrix,
            MAX_OCCUPANCY=MAX_OCCUPANCY,
            MIN_OCCUPANCY=MIN_OCCUPANCY,
            N_DAYS=N_DAYS
        )
        #print('penalty', penalty)
        #print('accounting_cost', accounting_cost)
        return penalty + accounting_cost
    
    return cost_function  

# Build your "cost_function"
cost_function_optim = build_cost_function(data)


# In[14]:


# version build : AVEC les parametres dans le build 

@njit(parallel=True, fastmath=False)
def _eval_cost_vect_optim(arr_pop, family_size, days_array, 
                       penalty_matrix, accounting_matrix, 
                       MAX_OCCUPANCY, MIN_OCCUPANCY, N_DAYS):
    '''
    Boosting simple by seeking by ranges 
    Simple baseline optimisation following different path range of families.
    
    info : speed up to max prop to cost optim  : 24e-6s by cost calculation.
    
    input : arr_choice : array of choice days by families
            best : best submission
            arr_range : ranges of paths into families
    return :  arr_sub : array of submissions
            arr_score : array of their score/cost
    
    example :
    arr_sub, arr_score = boost_diff_browsing(arr_choice, best, arr_range)
    
    # time execution : parallel 17 ms ald 24ms for 1000 cost evaluation 

    '''
    #@njit
    def _compute_cost_fast_intern(prediction):
        """
        Do not use this function. Please use `build_cost_function` instead to 
        build your own "cost_function".
        """
        N = family_size.shape[0]
        # We'll use this to count the number of people scheduled each day
        daily_occupancy = np.zeros(len(days_array)+1, dtype=np.int64)
        penalty = 0

        # Looping over each family; d is the day, n is size of that family
        for i in range(N):
            n = family_size[i]
            d = prediction[i]

            daily_occupancy[d] += n
            penalty += penalty_matrix[i, d]

        # for each date, check total occupancy 
        # (using soft constraints instead of hard constraints)
        # Day 0 does not exist, so we do not count it
        relevant_occupancy = daily_occupancy[1:]

        # patch : G.L. 12/01/2020 - begins
        a = (relevant_occupancy > MAX_OCCUPANCY)
        b = (relevant_occupancy < MIN_OCCUPANCY)
        incorrect_occupancy = a | b 
        for inc_curr in incorrect_occupancy:
            if inc_curr:
                #print("inc_curr", inc_curr)
                penalty += 100000000
        #print(incorrect_occupancy)
        # patch : G.L. 12/01/2020 - ends

        # Calculate the accounting cost
        # The first day (day 100) is treated special
        init_occupancy = daily_occupancy[days_array[0]]
        accounting_cost = (init_occupancy - 125.0) /             400.0 * init_occupancy**(0.5)
        # using the max function because the soft constraints 
        # might allow occupancy to dip below 125
        accounting_cost = max(0, accounting_cost)

        # Loop over the rest of the days_array, keeping track of previous count
        yesterday_count = init_occupancy
        for day in days_array[1:]:
            today_count = daily_occupancy[day]
            diff = abs(today_count - yesterday_count)
            # patch G.L. : 17/01/2020 : limit inputs
            today_count_lim = max(MIN_OCCUPANCY, 
                                  min(MAX_OCCUPANCY, today_count))
            diff_lim = max(0, min(N_DAYS, diff))
            accounting_cost += accounting_matrix[today_count_lim, diff_lim]

            yesterday_count = today_count
        #print("penalty: ", penalty)   
        #print("accounting_cost: ", accounting_cost)
        return penalty, accounting_cost, daily_occupancy
    
    
    arr_score = np.zeros(arr_pop.shape[0])
    
    for indice in prange(arr_pop.shape[0]):
        # patch to accelerate _compute_cost_fast_intern fct (do a copy)
        arr_curr = arr_pop[indice].copy()
        penalty, accounting_cost, daily_occupancy  =             _compute_cost_fast_intern(arr_curr)
        arr_score[indice] = penalty + accounting_cost
         
    return  arr_score


def build_eval_cost_vect_optim(data, N_DAYS=100, MAX_OCCUPANCY=300, 
                                    MIN_OCCUPANCY=125):
    family_size = data.n_people.values
    days_array = np.arange(N_DAYS, 0, -1)
    # Precompute matrices needed for our cost function
    choice_array_num = _build_choice_array(data, N_DAYS)
    penalty_matrix = _precompute_penalties(choice_array_num, family_size)
    # patch G.L. 12/01/2020 - begins
    accounting_matrix = _precompute_accounting(max_day_count=MAX_OCCUPANCY,
                                       max_diff=MAX_OCCUPANCY)
    
    def my_eval_cost_vect_optim(arr_pop):
        
        arr_score = _eval_cost_vect_optim(arr_pop, 
            family_size=family_size, 
            days_array=days_array, 
            penalty_matrix=penalty_matrix, 
            accounting_matrix=accounting_matrix,
            MAX_OCCUPANCY=MAX_OCCUPANCY,
            MIN_OCCUPANCY=MIN_OCCUPANCY,
            N_DAYS=N_DAYS)
    
        return arr_score
    
    return my_eval_cost_vect_optim

eval_cost_vect_optim = build_eval_cost_vect_optim(data)


# In[15]:


# version build : AVEC les parametres dans le build 

@njit
def _boost_diff_browsing_optim(best, arr_range, family_size, days_array, 
                               penalty_matrix, accounting_matrix, 
                               MAX_OCCUPANCY, MIN_OCCUPANCY, N_DAYS, flag_seq,
                               flag_prompt):
    '''
    Boosting simple by seeking by ranges 
    Simple baseline optimisation following different path range of families.
    
    info : speed up to max prop to cost optim  : 24e-6s by cost calculation.
    
    input : arr_choice : array of choice days by families
            best : best submission
            arr_range : ranges of paths into families
    return :  arr_sub : array of submissions
            arr_score : array of their score/cost
    
    example :
    arr_sub, arr_score = boost_diff_browsing(arr_choice, best, arr_range)
    
    speed exec : 10.8 s for 10 ranges.

    '''
    #@njit
    def _compute_cost_fast_intern(prediction):
        """
        Do not use this function. Please use `build_cost_function` instead to 
        build your own "cost_function".
        """
        N = family_size.shape[0]
        # We'll use this to count the number of people scheduled each day
        daily_occupancy = np.zeros(len(days_array)+1, dtype=np.int64)
        penalty = 0

        # Looping over each family; d is the day, n is size of that family
        for i in range(N):
            n = family_size[i]
            d = prediction[i]

            daily_occupancy[d] += n
            penalty += penalty_matrix[i, d]

        # for each date, check total occupancy 
        # (using soft constraints instead of hard constraints)
        # Day 0 does not exist, so we do not count it
        relevant_occupancy = daily_occupancy[1:]

        # patch : G.L. 12/01/2020 - begins
        a = (relevant_occupancy > MAX_OCCUPANCY)
        b = (relevant_occupancy < MIN_OCCUPANCY)
        incorrect_occupancy = a | b 
        for inc_curr in incorrect_occupancy:
            if inc_curr:
                #print("inc_curr", inc_curr)
                penalty += 100000000
        #print(incorrect_occupancy)
        # patch : G.L. 12/01/2020 - ends

        # Calculate the accounting cost
        # The first day (day 100) is treated special
        init_occupancy = daily_occupancy[days_array[0]]
        accounting_cost = (init_occupancy - 125.0) /             400.0 * init_occupancy**(0.5)
        # using the max function because the soft constraints 
        # might allow occupancy to dip below 125
        accounting_cost = max(0, accounting_cost)

        # Loop over the rest of the days_array, keeping track of previous count
        yesterday_count = init_occupancy
        for day in days_array[1:]:
            today_count = daily_occupancy[day]
            diff = abs(today_count - yesterday_count)
            # patch G.L. : 17/01/2020 : limit inputs
            today_count_lim = max(MIN_OCCUPANCY, 
                                  min(MAX_OCCUPANCY, today_count))
            diff_lim = max(0, min(N_DAYS, diff))
            accounting_cost += accounting_matrix[today_count_lim, diff_lim]

            yesterday_count = today_count
        #print("penalty: ", penalty)   
        #print("accounting_cost: ", accounting_cost)
        return penalty, accounting_cost, daily_occupancy
    
    
    # Create baselines # optimized version
    penalty, accounting_cost, daily_occupancy  = _compute_cost_fast_intern(best)
    start_cost = penalty + accounting_cost
    #start_cost = cost_function_optim(best)
    if flag_prompt:
        print("Start cost: ", start_cost)
    
    # prepare output : best submission seeking in different range walk around
    arr_sub = np.zeros((arr_range.shape[0], best.shape[0])).astype(np.int64)
    arr_score = np.zeros(arr_range.shape[0])
    new = best.copy() # TEST
    for indice in prange(arr_range.shape[0]):

        # initiate first inviduate
        if flag_seq == False:
            # if mode where each range is treated independently (no seq)
            # we reset the best to the first best as input arg.
            new = best.copy() 
        
        cost_best = start_cost
        
        # choose current range in df_range
        range_optim = arr_range[indice]
        # loop over each family with this current range       
        for fam_id in range_optim:
            # loop over each family choice
            for pick in range(10):
                day = arr_choice[fam_id, pick]
                temp = new.copy()
                temp[fam_id] = day # add in the new pick
        
                penalty, accounting_cost, daily_occupancy =                     _compute_cost_fast_intern(temp) #test cost
                cost_curr =  penalty + accounting_cost
                #cost_curr = cost_function_optim(temp)
                # if best cost found save it
                if cost_curr < cost_best:
                    new = temp.copy()
                    cost_best = cost_curr
                    #print("Current best cost: ", cost_best)

        arr_sub[indice] = new
        arr_score[indice] = cost_best
        if flag_prompt:
            print("Score: ", cost_best)
          
    return arr_sub, arr_score


def build_boost_diff_browsing_optim(data, flag_seq=True, flag_prompt=True,
                                    N_DAYS=100, 
                                    MAX_OCCUPANCY=300, 
                                    MIN_OCCUPANCY=125):
    family_size = data.n_people.values
    days_array = np.arange(N_DAYS, 0, -1)
    # Precompute matrices needed for our cost function
    choice_array_num = _build_choice_array(data, N_DAYS)
    penalty_matrix = _precompute_penalties(choice_array_num, family_size)
    # patch G.L. 12/01/2020 - begins
    accounting_matrix = _precompute_accounting(max_day_count=MAX_OCCUPANCY,
                                       max_diff=MAX_OCCUPANCY)
    
    def my_boost_diff_browsing_optim(best, arr_range):
        
        arr_sub, arr_score = _boost_diff_browsing_optim(best, arr_range, 
                                                        family_size=family_size, 
            days_array=days_array, 
            penalty_matrix=penalty_matrix, 
            accounting_matrix=accounting_matrix,
            MAX_OCCUPANCY=MAX_OCCUPANCY,
            MIN_OCCUPANCY=MIN_OCCUPANCY,
            N_DAYS=N_DAYS,
            flag_seq=flag_seq,
            flag_prompt=flag_prompt)
    
        return arr_sub, arr_score
    
    return my_boost_diff_browsing_optim
# boost seq from one best along several range as input 
boost_diff_browsing_optim = build_boost_diff_browsing_optim(data)


# In[16]:


family_size = data.n_people.values
days_array = np.arange(N_DAYS, 0, -1)
# Precompute matrices needed for our cost function
choice_array_num = _build_choice_array(data, N_DAYS)
penalty_matrix = _precompute_penalties(choice_array_num, family_size)
# patch G.L. 12/01/2020 - begins
accounting_matrix = _precompute_accounting(max_day_count=MAX_OCCUPANCY,
                                   max_diff=MAX_OCCUPANCY)

@njit(parallel=True, fastmath=True)
def boost_optim_one_by_one(best, arr_range):
    '''
    Calculation optim cost from one best indiv (submission) seeking along 
    several ranges (arr_range) but not sequential.
    '''
    arr_pop = np.zeros((arr_range.shape[0], best.shape[0]), dtype=np.int64)
    arr_score = np.zeros(arr_range.shape[0], dtype=np.float64)
    for indice in prange(arr_range.shape[0]):
        arr_pop_curr, arr_score_curr =             _boost_diff_browsing_optim(best, 
                                       arr_range[indice:indice+1], 
                                       family_size, 
                                       days_array, 
                                       penalty_matrix, 
                                       accounting_matrix, 
                       MAX_OCCUPANCY, MIN_OCCUPANCY, N_DAYS, flag_seq=False, 
                                      flag_prompt=False)
        arr_pop[indice] = arr_pop_curr
        arr_score[indice] = arr_score_curr[0]
    return arr_pop, arr_score

@njit(parallel=True, fastmath=True)
def boost_optim_one_by_one_multi(arr_pop_in, arr_range):
    '''
    Calculation optim cost from several best indiv (arr_pop_in) seeking along 
    arr_range not sequential.
    '''
    arr_pop = arr_pop_in.copy()#np.zeros((arr_pop_in.shape[0], best.shape[0]), dtype=np.int64)
    arr_score = np.zeros(arr_pop.shape[0], dtype=np.float64)
    for indice in prange(arr_pop_in.shape[0]):
        arr_pop_curr, arr_score_curr =             _boost_diff_browsing_optim(arr_pop_in[indice], 
                                       arr_range, 
                                       family_size, 
                                       days_array, 
                                       penalty_matrix, 
                                       accounting_matrix, 
                       MAX_OCCUPANCY, MIN_OCCUPANCY, N_DAYS, flag_seq=False, 
                                      flag_prompt=False)
        arr_pop[indice] = arr_pop_curr
        arr_score[indice] = arr_score_curr[0]
    return arr_pop, arr_score

@njit(parallel=True, fastmath=True)
def boost_optim_one_by_one_epochs(arr_pop_in, n_epochs=100, nb_epoch_check=10,
                                  nb_try_not_best_max=2):
    '''
    Calculation simple optim cost from several best indiv (arr_pop_in) seeking 
    along random range several times sequentially (n_epochs).
    '''
    arr_pop = arr_pop_in.copy()#np.zeros((arr_pop_in.shape[0], best.shape[0]), dtype=np.int64)
    nb_pop = arr_pop.shape[0]
    nb_fam = arr_pop.shape[1]
    arr_score = np.zeros(nb_pop, dtype=np.float64)
    if n_epochs < nb_epoch_check:
        nb_epoch_check = n_epochs
    nb_check = int(np.ceil(n_epochs/nb_epoch_check))   
    print("nb_check: ", nb_check)
    for indice in prange(nb_pop):
        # For each check
        best = arr_pop_in[indice].copy()
        nb_try_not_best = 0
        for i_check in np.arange(nb_check):
            print("Indiv #", indice)
            print("         check #", i_check)
            # calculate number of ranges to test
            if (i_check + 1)* nb_epoch_check > n_epochs:
                nb_epochs_curr = n_epochs - (i_check * nb_epoch_check)
            else:
                nb_epochs_curr = nb_epoch_check
            #print("nb_epochs_curr: ", nb_epochs_curr)    
            # create nb_epochs_curr ranges    
            arr_range = np.empty((nb_epochs_curr, nb_fam), dtype=np.int64) 
            for i_epoch in np.arange(nb_epochs_curr):
                arr_range[i_epoch] = np.random.permutation(np.arange(nb_fam))
                
            # optimize sequentially to find best for one indiv
            arr_pop_curr, arr_score_curr =                 _boost_diff_browsing_optim(best, 
                                           arr_range, 
                                           family_size, 
                                           days_array, 
                                           penalty_matrix, 
                                           accounting_matrix, 
                           MAX_OCCUPANCY, MIN_OCCUPANCY, N_DAYS, flag_seq=True, 
                                          flag_prompt=False)
            #print("arr_score_curr: ", arr_score_curr)
            
            best_score_curr = arr_score_curr.min()
            
            if i_check > 0:
                if best_score_curr >= arr_score[indice]:
                    print("Not better for Indiv #", indice)
                    nb_try_not_best +=1 
                    if nb_try_not_best > nb_try_not_best_max:
                        print("Early stop for Indiv #", indice)
                        break
            arr_score[indice] = arr_score_curr.min()
            arr_pop[indice] = arr_pop_curr[np.argmin(arr_score_curr)].copy()
            best = arr_pop[indice].copy()
        print("Best cost for indiv #", indice, ":", arr_score[indice])
    return arr_pop, arr_score


# In[17]:


'''
Data preparation
'''
# for KAGGLE, GOOGLE COLAB & LOCAL compatibility
# copy of familly data 
# Source path 
source = PATH_INPUT + '/santa-2019-workshop-scheduling/family_data.csv'
  
# Destination path 
destination = os.getcwd() + "/family_data.csv"
  
# Copy the content of 
# source to destination 
if not os.path.exists(destination):
    import shutil
    dest = shutil.copyfile(source, destination) 
    print("Copy ok")
else:
    print("file exists")


# In[18]:


get_ipython().run_cell_magic('writefile', 'stochprodsearch_03.cpp', '#include <array>\n#include <cassert>\n#include <algorithm>\n#include <cmath>\n#include <fstream>\n#include <iostream>\n#include <vector>\n#include <thread>\n#include <atomic>\n#include <random>\n#include <string.h>\nusing namespace std;\n#include <chrono>\nusing namespace std::chrono;\n\n// V1.1 : 16/02/2020 : extend range to choice 10\n\n//int N_JOBS = 4;\n//int END_TIME = 1*N_JOBS;//in minutes\n\nauto START_TIME = high_resolution_clock::now();\n//constexpr array<uint8_t, 4> DISTRIBUTION{2, 2, 3, 5}; // 82400 -> 72172.8 in 8h (KAGGLE V4)\nconstexpr array<uint8_t, 6> DISTRIBUTION{2, 2, 2, 2, 3, 5}; // 82400 -> 72060.8 in 8h (KAGGLE V5)\n// constexpr array<uint8_t, 15> DISTRIBUTION{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5};  // You can setup how many families you need for swaps and what best choice use for each family\n// {2, 5} it\'s mean the first random family will brute force for choices 1-2 and the second random family will brute force for choices 1-5\n\nconstexpr int MAX_OCCUPANCY = 300;\nconstexpr int MIN_OCCUPANCY = 125;\nconstexpr int BEST_N = 10;\narray<uint8_t, 5000> n_people;\narray<array<uint8_t, 11>, 5000> choices;\narray<array<uint16_t, 11>, 5000> PCOSTM;\narray<array<double, 176>, 176> ACOSTM;\n\nstruct Index {\n    Index(array<int, 5000> assigned_days_) : assigned_days(assigned_days_)  {\n        setup();\n    }\n    array<int, 5000> assigned_days;\n    array<uint16_t, 100> daily_occupancy_{};\n    int preference_cost_ = 0;\n    void setup() {\n        preference_cost_ = 0;\n        daily_occupancy_.fill(0);\n        for (int j = 0; j < assigned_days.size(); ++j) {\n            daily_occupancy_[choices[j][assigned_days[j]]] += n_people[j];\n            preference_cost_ += PCOSTM[j][assigned_days[j]];\n        }\n    }\n    double calc(const array<uint16_t, 5000>& indices, const array<uint8_t, DISTRIBUTION.size()>& change) {\n        double accounting_penalty = 0.0;\n        auto daily_occupancy = daily_occupancy_;\n        int preference_cost = preference_cost_;\n        for (int i = 0; i < DISTRIBUTION.size(); ++i) {\n            int j = indices[i];\n            daily_occupancy[choices[j][assigned_days[j]]] -= n_people[j];\n            daily_occupancy[choices[j][       change[i]]] += n_people[j];\n            \n            preference_cost += PCOSTM[j][change[i]] - PCOSTM[j][assigned_days[j]];\n        }\n\n        for (auto occupancy : daily_occupancy)\n            if (occupancy < MIN_OCCUPANCY) \n                return 1e12*(MIN_OCCUPANCY-occupancy);\n            else if (occupancy > MAX_OCCUPANCY)\n                return 1e12*(occupancy - MAX_OCCUPANCY);\n\n        for (int day = 0; day < 99; ++day)\n            accounting_penalty += ACOSTM[daily_occupancy[day]-125][daily_occupancy[day+1]-125];\n\n        accounting_penalty += ACOSTM[daily_occupancy[99]-125][daily_occupancy[99]-125];\n        return preference_cost + accounting_penalty;\n    }\n    void reindex(const array<uint16_t, DISTRIBUTION.size()>& indices, const array<uint8_t, DISTRIBUTION.size()>& change) {\n        for (int i = 0; i < DISTRIBUTION.size(); ++i) {\n            assigned_days[indices[i]] = change[i];\n        }\n        setup();\n    }\n};\n\nstatic std::atomic<bool> flag(false);\nstatic Index global_index({});\n\nbool time_exit_fn(int end_time){\n    return duration_cast<seconds>(high_resolution_clock::now()-START_TIME).count() < end_time;\n}\n\nvoid init_data() {\n    ifstream in("family_data.csv");\n    \n    assert(in && "family_data.csv");\n    string header;\n    int n,x;\n    char comma;\n    getline(in, header);\n    for (int fam_id = 0; fam_id < choices.size(); ++fam_id) {\n        in >> x >> comma;\n        for (int n_choice = 0; n_choice < 10; ++n_choice) {\n            in >> x >> comma;\n            choices[fam_id][n_choice] = x-1;\n        }\n        in >> n;\n        n_people[fam_id] = n;\n        //std::cout << fam_id << ", " << (int)n_people[fam_id] << endl;\n    }\n    array<int, 11> pc{0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500};\n    array<int, 11> pn{0,  0,  9,   9,   9,  18,  18,  36,  36, 235, 434};\n    //cout << endl << "PCOSTM:";\n    for (int j = 0; j < PCOSTM.size(); ++j) {\n        //cout << endl << j << ": ";\n        for (int i = 0; i < 11; ++i) {\n            PCOSTM[j][i] = pc[i] + pn[i] * n_people[j];\n            //cout << PCOSTM[j][i] << " ";\n        }\n    }\n    \n    for (int i = 0; i < 176; ++i)\n        for (int j = 0; j < 176; ++j)\n            ACOSTM[i][j] = i * pow(i+125, 0.5 + abs(i-j) / 50.0) / 400.0;\n}\n// not used (keep for later add-on ?)\narray<int, 5000> read_submission(string filename) {\n    ifstream in(filename);\n    assert(in && "submission.csv");\n    array<int, 5000> assigned_day{};\n    string header;\n    int id, x;\n    char comma;\n    getline(in, header);\n    for (int j = 0; j < choices.size(); ++j) {\n        in >> id >> comma >> x;\n        assigned_day[j] = x-1;\n        auto it = find(begin(choices[j]), end(choices[j]), assigned_day[j]);\n        if (it != end(choices[j]))\n            assigned_day[j] = distance(begin(choices[j]), it);\n    }\n    return assigned_day;\n}\n\n\ndouble calc(const array<int, 5000>& assigned_days, bool print=false) {\n    int preference_cost = 0;\n    double accounting_penalty = 0.0;\n    array<uint16_t, 100> daily_occupancy{};\n    for (int j = 0; j < assigned_days.size(); ++j) {\n        preference_cost += PCOSTM[j][assigned_days[j]];\n        daily_occupancy[choices[j][assigned_days[j]]] += n_people[j];\n        //cout << j  << ", " << (int)assigned_days[j] << ", " << (int)n_people[j] << endl;\n        //cout << j << ", " << n_people[j] << endl;\n    }\n    int K=1;\n    for (auto occupancy : daily_occupancy) {\n        if (occupancy < MIN_OCCUPANCY) {\n            //std::cout << "occ. day " << K << "=" << occupancy << " ! MIN_OCC reached" << endl;\n            return 1e12*(MIN_OCCUPANCY-occupancy);\n        } else if (occupancy > MAX_OCCUPANCY) {\n            //std::cout << "occ. day " << K << "=" << occupancy << " ! MAX_OCC reached" << endl;\n            return 1e12*(occupancy - MAX_OCCUPANCY);\n        }\n        K = K+1;\n    }\n\n    for (int day = 0; day < 99; ++day)\n        accounting_penalty += ACOSTM[daily_occupancy[day]-125][daily_occupancy[day+1]-125];\n\n    accounting_penalty += ACOSTM[daily_occupancy[99]-125][daily_occupancy[99]-125];\n    if (print) {\n        cout << preference_cost+accounting_penalty << ": pc: " << preference_cost << " + ac: " << accounting_penalty << endl;\n    }\n    return preference_cost + accounting_penalty;\n}\n\nvoid save_sub(const array<int, 5000>& assigned_day) {\n    ofstream out("submission.csv");\n    out << "family_id,assigned_day" << endl;\n    for (int i = 0; i < assigned_day.size(); ++i)\n        out << i << "," << choices[i][assigned_day[i]]+1 << endl;\n}\n        \nconst vector<array<uint8_t, DISTRIBUTION.size()>> changes = []() {\n    vector<array<uint8_t, DISTRIBUTION.size()>> arr;\n    array<uint8_t, DISTRIBUTION.size()> tmp{};\n    for (int i = 0; true; ++i) {\n        arr.push_back(tmp);\n        tmp[0] += 1;\n        for (int j = 0; j < DISTRIBUTION.size(); ++j)\n            if (tmp[j] >= DISTRIBUTION[j]) {\n                if (j >= DISTRIBUTION.size()-1)\n                    return arr;\n                tmp[j] = 0;\n                ++tmp[j+1];\n            }\n    }\n    return arr;\n}();\n\n//template<class ExitFunction>\nvoid stochastic_product_search(Index index, int end_time) { // 15\'360\'000it/s  65ns/it  0.065µs/it\n    double best_local_score = calc(index.assigned_days);\n    thread_local std::mt19937 gen(std::random_device{}());\n    uniform_int_distribution<> dis(0, 4999);\n    array<uint16_t, 5000> indices;\n    iota(begin(indices), end(indices), 0);\n    array<uint16_t, DISTRIBUTION.size()> best_indices{};\n    array<uint8_t, DISTRIBUTION.size()> best_change{};\n    for (;time_exit_fn(end_time);) {\n        bool found_better = false;\n        for (int k = 0; k < BEST_N; ++k) {\n            for (int i = 0; i < DISTRIBUTION.size(); ++i) //random swap for n first families\n                swap(indices[i], indices[dis(gen)]);\n            for (const auto& change : changes) {\n                auto score = index.calc(indices, change);\n                if (score < best_local_score) {\n                    found_better = true;\n                    best_local_score = score;\n                    best_change = change;\n                    copy_n(begin(indices), DISTRIBUTION.size(), begin(best_indices));\n                }\n            }\n        }\n\n        if (flag.load() == true){\n            return;\n        }\n\n        if (found_better && flag.load() == false) { // reindex from N best if found better\n            flag = true;\n\n            index.reindex(best_indices, best_change);\n            global_index = index;\n            return;\n        }\n    }\n}\n\n// output function : for python ctypes, needed to add extern "C"\nextern "C" int* sps(int* days, int duration=6, int n_jobs=4) {\n    \n    int N_JOBS = n_jobs;\n    int END_TIME = duration*N_JOBS; //in seconds\n    std::cout << "duration = " << duration << endl; // debug\n    std::cout << "END_TIME = " << END_TIME << endl; // debug\n    \n    init_data();\n    //auto assigned_day = read_submission("../input/first-simple-optimization-santa-submission/submission_535295.5188186927.csv");\n    //auto assigned_day = read_submission("../input/best-result-with-algo-genetic/submission_85181.22055864273.csv");\n    //auto assigned_day = read_submission("../input/santa-workshop-tour-2019/sample_submission.csv");\n    //auto assigned_day = read_submission("submission_82477.85928353199.csv");\n    //auto assigned_day = read_submission("submission_100135.53956452094.csv");\n    //auto assigned_day = read_submission("submission_535295.5188186927.csv");\n    \n    // read input from python\n    array<int, 5000> assigned_day{};\n    // int pointer instead of array format for output to python : it is the best solution found\n    int *assigned_days_out = new int[5000];\n    \n    // read python input pointer days to good format for C++\n    //std::cout << "current choices:";\n    int K = 0;\n    for (int i = 0; i < 2*5000; i+=2) {\n        //std::cout << " " << days[i] << ", ";\n        assigned_day[K] = days[i] - 1;\n        //auto it = find(begin(choices[K]), end(choices[K]), assigned_day[K]);\n        auto it = find(begin(choices[K]), end(choices[K])-1, assigned_day[K]);\n        if (it != end(choices[K])-1) {\n            assigned_day[K] = distance(begin(choices[K]), it);\n            //cout << " " << K << ": " << (int)assigned_day[K];\n        } else {\n            //cout << " " << K << ": " << (int)assigned_day[K]; \n            choices[K][10] = days[i] - 1;\n            assigned_day[K] = 10;\n        }\n        K = K + 1;\n    }\n    \n    Index index(assigned_day);\n    global_index = index;\n    std::cout << "current cost: ";\n    calc(index.assigned_days, true);\n    std::cout << endl << "Start s.p.s V1.1..." << endl;\n    for(;time_exit_fn(END_TIME);){\n\n        std::thread threads[N_JOBS];\n        for(int i = 0; i < N_JOBS; i++){\n            threads[i] = std::thread(stochastic_product_search, index, END_TIME);\n        }\n        for(int i = 0; i < N_JOBS; i++){\n            threads[i].join();\n        }\n\n        auto best_score = calc(global_index.assigned_days, false);\n\n        flag = false;\n        index = global_index;      \n    }\n    \n    // display best cost : \n    std::cout << "Best cost found: ";\n    calc(global_index.assigned_days, true);\n    // creation of output for python\n    //std::cout << endl; // for debug\n    for (int i = 0; i < global_index.assigned_days.size(); ++i) {\n        assigned_days_out[i] = choices[i][global_index.assigned_days[i]]+1;\n        //std::cout << assigned_days_out[i] << ", "; // for debug\n    }\n    \n    return assigned_days_out;\n    //return 0;\n}')


# In[19]:


if MY_PLATFORM == "Linux":
    print("Linux compilation...")
    get_ipython().system('g++ -c -pthread -O3 -std=c++17 -fPIC stochprodsearch_03.cpp -o stochprodsearch_03_linux.o')
    get_ipython().system('g++ -shared -Wl,-soname,libstochprodsearch_03.so -o libstochprodsearch_03.so  stochprodsearch_03_linux.o')
elif MY_PLATFORM == "Darwin":
    get_ipython().system('g++ -c -pthread -O3 -std=c++17 -fPIC stochprodsearch_03.cpp -o stochprodsearch_03.o')
    get_ipython().system('g++ -dynamiclib -undefined suppress -flat_namespace stochprodsearch_03.o -o libstochprodsearch_03.dylib')
elif MY_PLATFORM == "Windows":
  # TODO
  pass
else:
    print("Unknow platform ! ")


# In[20]:


get_ipython().run_cell_magic('writefile', 'stochprodsearch_03.py', 'import platform\nimport pandas as pd\nimport numpy as np\nfrom numpy.ctypeslib import ndpointer\nimport ctypes\nfrom ctypes import cdll\nfrom ctypes import c_char_p\nfrom ctypes import c_double\nfrom ctypes import c_int\n\'\'\'\nThis script link C++ executable as a ctypes library in python.\nBut because lib ctypes doesnt execute well twice when used directly in another\npython script, it is needed to used a dirty csv file as output. \n \nHow to use examples : \n- From command line: 6 sec and 4 threads:\n>> python stochprodsearch_03.py my_submission.csv 6 4 \n- From notebooks : run_stochprodsearch(arr_best_curr, end_time=6, nb_jobs=4)\n\'\'\'\n# DEFINTION : \nOUTPOUT_FILE_NAME = "submission_from_sps.csv" # TO BE CHANGED !!!\n# select current platform\nMY_PLATFORM = platform.system()\n#Linux: Linux\n#Mac: Darwin\n#Windows: Windows\n# load lib\nif MY_PLATFORM == "Darwin":\n    lib = cdll.LoadLibrary(\'./libstochprodsearch_03.dylib\')\nelif MY_PLATFORM == "Linux":\n    lib = cdll.LoadLibrary(\'./libstochprodsearch_03.so\')\nelif MY_PLATFORM == "Windows":\n    lib = cdll.LoadLibrary(\'./libstochprodsearch_03.dll\') \nelse:\n    print("Unknow platform ! ")\n    lib = None\n# declare ctypes pointer format for output from C++\nc_int_p = ctypes.POINTER(ctypes.c_int)\n# prepare output from C++ to numpy array of size 5000 \nif lib is not None:\n    lib.sps.restype = ndpointer(dtype=ctypes.c_int, shape=(5000,))\n\ndef sps(arr_in, end_time=6, nb_jobs=4):\n    \'\'\'\n    Stochastic product search ctypes function to link with C++ code\n    \n    arr_in : initial days assigned numpy array\n    end_time : duration of search in seconds\n    nb_jobs : number of threads used for searching\n    \'\'\'\n\n    # cast to integer (security : if needed)\n    arr_in = arr_in.astype(np.int)\n\n    # prepare input for C++ to integer pointer \n    arr_in_p = arr_in.ctypes.data_as(c_int_p)\n    \n    # execute the optimisation product search with inital value arr_in_p\n    arr_best_sps = lib.sps(arr_in_p, c_int(end_time), c_int(nb_jobs))\n    \n    return arr_best_sps\n    #return lib.sps(arr_in_p)\n    \nif __name__ == "__main__":\n    import sys\n    try:\n        csv = sys.argv[1]\n    except:\n        csv = \'submission_100135.53956452094.csv\' \n    try:\n        end_time = int(sys.argv[2])\n    except:\n        end_time= 6\n    try:\n        nb_jobs = int(sys.argv[3])\n    except:\n        nb_jobs = 4\n        \n    submission_test = pd.read_csv(csv, \n                         index_col=\'family_id\')\n    arr_curr = submission_test["assigned_day"].values\n    \n    print("arr_curr : ", arr_curr)\n    # run stochastic product search\n    arr_best = sps(arr_curr, end_time, nb_jobs)\n    print("Days : ", arr_best)\n    # prepare ouput data\n    submission_final = submission_test.copy()\n    submission_final["assigned_day"] = arr_best\n    # FALLBACK : Export res in CSV format (to be read by notebook)\n    submission_final.to_csv(OUTPOUT_FILE_NAME)')


# In[21]:


get_ipython().run_cell_magic('time', '', 'try:\n    !python stochprodsearch_03.py ../input/santa-2019-for-my-exploration/submission_71447.87946293628_for_sps.csv 6 4\nexcept:\n    print("Module test failed : Try to change csv input file!")')


# In[22]:


def run_stochprodsearch(arr_best_curr, end_time=6, nb_jobs=4):
    '''
    Run stochastic product search from python
    (fallback solution because problem with ctypes)
    '''
    
    cost_best = cost_function_optim(arr_best_curr)
    
    csv_in = "submission_{}_for_sps.csv".format(cost_best)
    csv_out = "submission_from_sps.csv"
    
    submission_in = pd.DataFrame(columns=["assigned_day"])
    submission_in["assigned_day"] = arr_best_curr
    submission_in.index.name = 'family_id'
    submission_in.to_csv(csv_in)    
    
    os.system("python stochprodsearch_03.py {} {} {}".format(csv_in ,end_time,
                                                            nb_jobs))
    csv_out = "submission_from_sps.csv"
    submission_from_sps = pd.read_csv(csv_out, index_col='family_id')
    
    return submission_from_sps["assigned_day"].values


# In[23]:


# show how people choose days : 
# 5 first choices 
# 5 last choices
#data
df_day = pd.DataFrame(index=range(1,101))
df_day["all_choices"] = 0
df_day["first_choices"] = 0
df_day["mid_choices"] = 0
df_day["last_choices"] = 0
list_choice_all = ['choice_{}'.format(n) for n in range(0, 10)]
list_choice_first = ['choice_{}'.format(n) for n in range(0, 3)]
list_choice_mid = ['choice_{}'.format(n) for n in range(3, 7)]
list_choice_last = ['choice_{}'.format(n) for n in range(7, 10)]

# for each first choices, add to each days the number of people
for choice in list_choice_all:
    for indice in data.index:
        df_day.loc[data.at[indice, choice], 
                   "all_choices"] += data.at[indice, "n_people"]
        
for choice in list_choice_first:
    for indice in data.index:
        df_day.loc[data.at[indice, choice], 
                   "first_choices"] += data.at[indice, "n_people"]
        
for choice in list_choice_mid:
    for indice in data.index:
        df_day.loc[data.at[indice, choice], 
                   "mid_choices"] += data.at[indice, "n_people"]
        
for choice in list_choice_last:
    for indice in data.index:
        df_day.loc[data.at[indice, choice], 
                   "last_choices"] += data.at[indice, "n_people"]
        
df_day.head()


# In[24]:


fig = plt.figure(figsize=(12, 22)) 
ax1 = fig.gca()
df_day["all_choices"].plot.barh(ax=ax1)
ax1.set_xlabel("max nb people")
ax1.set_ylabel("number of days before Christmas [days]");
ax1.set_title("all choices");


# In[25]:


fig = plt.figure(figsize=(12, 22)) 
#plt.title("potential people vs choices")

ax1 = fig.add_subplot(1,3,1)
df_day["first_choices"].plot.barh(ax=ax1)
ax1.set_xlabel("max nb people")
ax1.set_ylabel("number of days before Christmas [days]");
ax1.set_title("first choices")

ax2 = fig.add_subplot(1,3,2)
df_day["mid_choices"].plot.barh(ax=ax2)
ax2.set_xlabel("max nb people")
ax2.set_title("mid choices")

ax3 = fig.add_subplot(1,3,3)
df_day["last_choices"].plot.barh(ax=ax3)
ax3.set_xlabel("max nb people")
ax3.set_title("last choices");


# In[26]:


fig = plt.figure(figsize=(12, 6)) 

df_day["first_choices"].plot(label="first_choices [1-3]")

df_day["mid_choices"].plot(label="mid_choices [4-7]")

df_day["last_choices"].plot(label="last_choices [8-10]")
plt.legend(loc='upper right');
ax=fig.gca()
ax.set_ylabel("attendance")
ax.set_xlabel("number of days before Christmas [days]");
ax.set_title("Maximum of attendance by days")


# In[27]:


df_prob_day = pd.DataFrame(df_day["all_choices"])
df_prob_day["prob"] = 1/df_prob_day["all_choices"]
df_prob_day["prob"] = df_prob_day["prob"] / df_prob_day["prob"].sum()
df_prob_day["prob"].sum()


# In[28]:


df_prob_day.head()


# In[29]:


#df_prob_day["prob"].plot.barh()
fig = plt.figure(figsize=(12, 22)) 
ax1 = fig.gca()
df_prob_day["prob"].plot.barh(ax=ax1)
ax1.set_xlabel("prob [-]")
ax1.set_ylabel("number of days before Christmas [days]");
ax1.set_title("Probabilities for each days");


# In[30]:


def cost_family(n=1, choice=0):
    # Calculate the penalty for not getting top preference
    penalty = 0
    if choice == 0:
        penalty += 0
    elif choice == 1:
        penalty += 50
    elif choice == 2:
        penalty += 50 + 9 * n
    elif choice == 3:
        penalty += 100 + 9 * n
    elif choice == 4:
        penalty += 200 + 9 * n
    elif choice == 5:
        penalty += 200 + 18 * n
    elif choice == 6:
        penalty += 300 + 18 * n
    elif choice == 7:
        penalty += 300 + 36 * n
    elif choice == 8:
        penalty += 400 + 36 * n
    elif choice == 9:
        penalty += 500 + 36 * n + 199 * n
    else:
        penalty += 500 + 36 * n + 398 * n
        
    return penalty


# In[31]:


ax = sns.boxplot(x=data["n_people"])


# In[32]:


ax = sns.boxplot(data)


# In[33]:


df_fam_cost = pd.DataFrame(index = np.array(range(np.min(data["n_people"]),
                            np.max(data["n_people"]) + 1)),
            columns=['choice_{}'.format(n_choice) for n_choice in range(0, 11)])
df_fam_cost["n"] = df_fam_cost.index
df_fam_cost


# In[34]:


df_fam_cost["choice_0"] = df_fam_cost["n"].apply(cost_family, args=(0,))
df_fam_cost["choice_1"] = df_fam_cost["n"].apply(cost_family, args=(1,))
df_fam_cost["choice_2"] = df_fam_cost["n"].apply(cost_family, args=(2,))
df_fam_cost["choice_3"] = df_fam_cost["n"].apply(cost_family, args=(3,))
df_fam_cost["choice_4"] = df_fam_cost["n"].apply(cost_family, args=(4,))
df_fam_cost["choice_5"] = df_fam_cost["n"].apply(cost_family, args=(5,))
df_fam_cost["choice_6"] = df_fam_cost["n"].apply(cost_family, args=(6,))
df_fam_cost["choice_7"] = df_fam_cost["n"].apply(cost_family, args=(7,))
df_fam_cost["choice_8"] = df_fam_cost["n"].apply(cost_family, args=(8,))
df_fam_cost["choice_9"] = df_fam_cost["n"].apply(cost_family, args=(9,))
df_fam_cost["choice_10"] = df_fam_cost["n"].apply(cost_family, args=(10,))
df_fam_cost


# In[35]:


fig = plt.figure(figsize=(12, 8))
plt.title("Cost family [$]")
list_choice = ['choice_{}'.format(n_choice) for n_choice in range(0, 11)]

for choice in list_choice:
    plt.plot(df_fam_cost["n"], df_fam_cost[choice], '-o', label=choice)
plt.legend(loc='upper left');

ax = fig.gca()
ax.set_xlabel("number of people [-]");


# In[36]:


fig = plt.figure(figsize=(12, 8))
plt.title("Cost for family by number of people [$]")
list_choice = ['choice_{}'.format(n_choice) for n_choice in range(0, 11)]

for n in df_fam_cost["n"]:
    plt.plot(range(0, df_fam_cost.filter(items=list_choice).shape[1]), 
        np.array(df_fam_cost.filter(items=list_choice).filter(items=[n], 
                                                              axis=0))[0],
             '-o', label=n)
plt.legend(loc='upper left');

ax = fig.gca()
ax.set_xlabel("choice number [-]");


# In[37]:


def create_df_fam_cost_prob(df_fam_cost, p_min=0.03, p_max=0.1):
    # For genetic algo, for start population or mutation,
    # try to assign probabilities of choice for each possibilities : 
    # from  choice 0 to 10.
    list_choice = ['choice_{}'.format(n_choice) for n_choice in range(0, 
                                                        CHOICE_RANGE_MAX + 1)]
    df_prob = df_fam_cost.filter(items=list_choice)
    vect_penalty = [0, np.max(np.max(df_prob))]
    print("vect_penalty: ", vect_penalty)
    vect_prob = [p_max, p_min]
    print("vect_prob", vect_prob)
    # family : number of people
    df_prob = df_prob.applymap(lambda x: np.interp(x, vect_penalty, vect_prob))
    for indice in df_prob.index:
        df_prob.loc[indice] = df_prob.loc[indice]/df_prob.loc[indice].sum()
    return df_prob  


# In[38]:


df_prob = create_df_fam_cost_prob(df_fam_cost, p_min=0.01, p_max=1)
df_prob


# In[39]:


df_prob
fig = plt.figure(figsize=(8, 5))
plt.title("probabilities during mutation [-]")
ax = sns.heatmap(df_prob)
ax.set_ylabel("nb people [-]")


# In[40]:


# save
joblib.dump(df_prob, PATH_TO_SAVE_DATA + '/df_prob.pkl')


# In[41]:


df_prob_fam = create_df_prob_day_fam_optim(df_prob_day, df_prob)


# In[42]:


df_prob_fam.head()


# In[43]:


# save
joblib.dump(df_prob_fam, PATH_SAVE_PROB_FAM)


# In[44]:


PATH_SAVE_PROB_FAM


# In[45]:


df_prob_fam = joblib.load(PATH_SAVE_PROB_FAM)
df_prob = joblib.load(PATH_TO_SAVE_DATA + '/df_prob.pkl')
# patch to optimize mutation fonction:
arr_prob = np.array(df_prob)
arr_prob_fam = np.array(df_prob_fam.astype("float"))


# In[46]:


#SAVE_POP = 'RANDOM_MUT'
#NB_FIRST_POP = 3
#DELTA_RANDOM_MUT_POP = 1 # delta for first random mut pop
#R_FIRST_RANDOM_MUT = 0.2 # RATIO of mutation for first population in random mut


# In[47]:


if SAVE_POP == 'RANDOM_MUT':
    print("Generate Random Mutation")
    # Create ranges
    submission = pd.read_csv(fpath, index_col='family_id')
    
    # create normal range 
    #arr_range = np.array([np.arange(submission.shape[0])])
    #arr_range = np.empty((NB_FIRST_POP, submission.shape[0]), dtype=np.int64)
    
    vect_pop_first_0 = submission["assigned_day"].values
        
    # generate first first choice for every families : 
    arr_pop_first_0 = np.empty((NB_FIRST_POP, vect_pop_first_0.shape[0]), 
                               dtype=np.int64)
    for indice in range(NB_FIRST_POP):
        arr_pop_first_0[indice] = vect_pop_first_0
    
    # generated random choice for first pop around 0 + DELTA_CHOICE_RANDOM_POP
    arr_pop_first = fun_vect_mut(arr_pop_first_0, 
                                 r_pop_mut=1, 
                                 r_mut=R_FIRST_RANDOM_MUT, 
                                 delta_choice=DELTA_RANDOM_MUT_POP)
    # replace first line by sample
    arr_pop_first[0] = vect_pop_first_0
    
    # Optimize first pop along some random ranges 
    t_fit_0 = time.time()
    print("Optimizing one by one indiv along random ranges ...")
    arr_pop, arr_score = boost_optim_one_by_one_epochs(arr_pop_first,
                                                       n_epochs=40, 
                                                       nb_epoch_check=10,
                                                       nb_try_not_best_max=2)
    t_fit_1 = time.time()
    print("Timing : ", t_fit_1 - t_fit_0)
    df_cost = pd.DataFrame(data=arr_score, columns=["cost"])
    df_cost.boxplot()
    print("df_cost: ", df_cost.sort_values(by="cost").head(10))
    df_pop = pd.DataFrame(arr_pop)
    df_pop


# In[48]:


#SAVE_POP = 'RANDOM_CHOICE'


# In[49]:


if SAVE_POP == 'RANDOM_CHOICE':
    print("Generate random Choices")
    # Create ranges
    submission = pd.read_csv(fpath, index_col='family_id')
    
    # create normal range 
    #arr_range = np.array([np.arange(submission.shape[0])])
    #arr_range = np.empty((NB_FIRST_POP, submission.shape[0]), dtype=np.int64)
    
    arr_pop_first_0 =         np.empty((NB_FIRST_POP, submission.shape[0]), dtype=np.int64)
    # generate first first choice for every families : 
    
    for fam_id in range(submission.shape[0]):
        arr_pop_first_0[0,fam_id] = choose_day_prob_optim(np.array([0]), fam_id)
    for indice in range(NB_FIRST_POP):
        arr_pop_first_0[indice] = arr_pop_first_0[0]
    
    # generated random choice for first pop around 0 + DELTA_CHOICE_RANDOM_POP
    arr_pop_first = fun_vect_mut(arr_pop_first_0, 
                                 r_pop_mut=1, 
                                 r_mut=1, 
                                 delta_choice=DELTA_CHOICE_RANDOM_POP)
    
    # Optimize first pop along one range only
    t_fit_0 = time.time()
    #best = submission['assigned_day'].values
    print("Optimizing one by one indiv along random range ...")
    arr_pop, arr_score = boost_optim_one_by_one_epochs(arr_pop_first,
                                                       n_epochs=100, 
                                                       nb_epoch_check=10,
                                                       nb_try_not_best_max=2)
    t_fit_1 = time.time()
    print("Timing : ", t_fit_1 - t_fit_0)
    df_cost = pd.DataFrame(data=arr_score, columns=["cost"])
    df_cost.boxplot()
    print("df_cost: ", df_cost.sort_values(by="cost").head(10))
    df_pop = pd.DataFrame(arr_pop)
    df_pop


# In[50]:


#SAVE_POP = 'RANDOM_PATH' 


# In[51]:


if SAVE_POP == 'RANDOM_PATH':
    print("Generate random paths")
    # Create ranges
    submission = pd.read_csv(fpath, index_col='family_id')
    
    # create NB_FIRST_POP random path to seek optimum
    arr_range = np.empty((NB_FIRST_POP, submission.shape[0]), dtype=np.int64)
    
    for indice in range(NB_FIRST_POP):
        arr_range[indice] = np.random.permutation(submission.shape[0])
    
    df_range = pd.DataFrame(data=arr_range)
    print(df_range.head(10))
    fig = plt.figure(figsize=(8, 8))
    plt.title("ranges")
    #for indice in range(0, df_range.shape[0]):
    plt.plot(df_range.loc[0])
    
    # create pop from submision by seeking along random paths
    t_fit_0 = time.time()
    best = submission['assigned_day'].values
    arr_pop, arr_score = boost_optim_one_by_one(best, arr_range=arr_range)
    t_fit_1 = time.time()
    print("Timing : ", t_fit_1 - t_fit_0)
    df_cost = pd.DataFrame(data=arr_score, columns=["cost"])
    df_cost.boxplot()
    print("df_cost: ", df_cost.sort_values(by="cost").head(10))
    df_pop = pd.DataFrame(arr_pop)
    df_pop


# In[52]:


#SAVE_POP = '10R'


# In[53]:


# OPTIM VERSION 
# std pop choices : 0.578
if SAVE_POP == '10R':
    # Create ranges
    submission = pd.read_csv(fpath, index_col='family_id')
    df_range = create_seek_ranges(nb_first_seed=NB_FIRST_SEED)
    fig = plt.figure(figsize=(8, 8))
    plt.title("ranges")
    for indice in range(0, df_range.shape[0]):
        plt.plot(df_range.loc[indice])
        
    # Create baselines : optimized version
    t_fit_0 = time.time()
    # Start with the sample submission values
    submission = pd.read_csv(fpath, index_col='family_id')
    best = submission['assigned_day'].values
    arr_range = df_range.values.astype(np.int64)
    arr_sub, arr_score = boost_diff_browsing_optim(best, arr_range=arr_range)
    t_fit_1 = time.time()
    print("Timing: ", t_fit_1 - t_fit_0)
    print("Info first pop of 10 :")
    _, df_des_choices_0, _ = pop_choices_info(pd.DataFrame(arr_sub))

    
    t_tot_0 = time.time()
    nb_indiv_done = 0
    nb_range = df_range.index.shape[0]
    nb_indiv_curr = np.floor(NB_FIRST_POP/df_range.shape[0])
    for i_seed in df_range.index:
        # choose number of indiv.
        nb_indiv_done += nb_indiv_curr
        if i_seed == nb_range-1:
            if NB_FIRST_POP % nb_range != 0:
                nb_indiv_curr += NB_FIRST_POP % nb_range
        nb_indiv_curr = int(nb_indiv_curr)
        print("# {} / nb_indiv_curr: {} / done: {}".format(i_seed, nb_indiv_curr, 
                                                           nb_indiv_done))
        # load best indiv in range #i_seed
        #seed_indiv = pd.read_csv(f'submission_range{i_seed}.csv')
        seed_indiv = arr_sub[i_seed]
        #print("seed_indiv.shape: ", seed_indiv.shape)
        # generate sub-pop 
        t_fit_0 = time.time()
        arr_pop_curr = generate_pop_choices_optim(seed_indiv=seed_indiv, 
                                           nb_pop=nb_indiv_curr, 
                                           r_mut=R_FIRST_MUT, 
                                           delta_choice=DELTA_CHOICE_FIRST_POP)
        t_fit_1 = time.time()
        print("Timing : ", t_fit_1 - t_fit_0)
        df_pop_curr = pd.DataFrame(arr_pop_curr)
        
        # add sub-pop to pop
        if i_seed == 0:
            df_pop = df_pop_curr
        else:
            df_pop = df_pop.append(df_pop_curr, ignore_index=True)
            
    t_tot_1 = time.time()
    print("Timing TOTAL: ", t_tot_1 - t_tot_0)
    print("df_pop.shape: ", df_pop.shape) 
    df_choices_0, df_des_choices_0, std_mean_0 = pop_choices_info(df_pop)
    print("Info for all pop: ")
    df_des_choices_0


# In[54]:


if SAVE_POP == "10R": 
    #plt.plot(df_pop.columns,df_pop.loc[0]-df_pop.loc[1])
    
    
    path_df_pop_saved = PATH_TO_SAVE_DATA +         '/df_pop_choices_{}_{}_fs{}_rfm{}_dc{}.pkl'.format(
                    SAVE_POP,
                    NB_FIRST_POP,
                    NB_FIRST_SEED, 
                    R_FIRST_MUT, 
                    DELTA_CHOICE_FIRST_POP)
    # check file already exist : 
    if os.path.isfile(path_df_pop_saved):
        path_df_pop_saved_old = PATH_TO_SAVE_DATA +         '/df_pop_choices_{}_{}_fs{}_rfm{}_dc{}_{}.pkl'.format(
                    SAVE_POP,
                    NB_FIRST_POP,
                    NB_FIRST_SEED, 
                    R_FIRST_MUT, 
                    DELTA_CHOICE_FIRST_POP,
                    datetime.today().strftime('%Y_%m_%d_%H_%M_%S'))
        os.rename(path_df_pop_saved, path_df_pop_saved_old)
    # save
    joblib.dump(df_pop, path_df_pop_saved, compress=True)


# In[55]:


if SAVE_POP == "RANDOM_PATH": 
    path_df_pop_saved = PATH_TO_SAVE_DATA +         '/df_pop_choices_{}_{}.pkl'.format(
                    SAVE_POP,
                    NB_FIRST_POP)
    # check file already exist : 
    if os.path.isfile(path_df_pop_saved):
        path_df_pop_saved_old = PATH_TO_SAVE_DATA +         '/df_pop_choices_{}_{}_{}.pkl'.format(
                    SAVE_POP,
                    NB_FIRST_POP,
                    datetime.today().strftime('%Y_%m_%d_%H_%M_%S'))
        os.rename(path_df_pop_saved, path_df_pop_saved_old)
    # save
    joblib.dump(df_pop, path_df_pop_saved, compress=True)
    print(path_df_pop_saved)


# In[56]:


if SAVE_POP == "RANDOM_CHOICE": 
    path_df_pop_saved = PATH_TO_SAVE_DATA +         '/df_pop_choices_{}_{}_dcr{}.pkl'.format(
                    SAVE_POP,
                    NB_FIRST_POP,
                    DELTA_CHOICE_RANDOM_POP)
    # check file already exist : 
    if os.path.isfile(path_df_pop_saved):
        path_df_pop_saved_old = PATH_TO_SAVE_DATA +         '/df_pop_choices_{}_{}_dcr{}_{}.pkl'.format(
                    SAVE_POP,
                    NB_FIRST_POP,
                    DELTA_CHOICE_RANDOM_POP,
                    datetime.today().strftime('%Y_%m_%d_%H_%M_%S'))
        os.rename(path_df_pop_saved, path_df_pop_saved_old)
    # save
    joblib.dump(df_pop, path_df_pop_saved, compress=True)
    print(path_df_pop_saved)


# In[57]:


if SAVE_POP == "RANDOM_MUT": 
    path_df_pop_saved = PATH_TO_SAVE_DATA +         '/df_pop_choices_{}_{}_rfrm{}_dcr{}.pkl'.format(
                    SAVE_POP,
                    NB_FIRST_POP,
                    R_FIRST_RANDOM_MUT,
                    DELTA_RANDOM_MUT_POP)
    # check file already exist : 
    if os.path.isfile(path_df_pop_saved):
        path_df_pop_saved_old = PATH_TO_SAVE_DATA +         '/df_pop_choices_{}_{}_rfrm{}_dcr{}_{}.pkl'.format(
                    SAVE_POP,
                    NB_FIRST_POP,
                    R_FIRST_RANDOM_MUT,
                    DELTA_RANDOM_MUT_POP,
                    datetime.today().strftime('%Y_%m_%d_%H_%M_%S'))
        os.rename(path_df_pop_saved, path_df_pop_saved_old)
    # save
    joblib.dump(df_pop, path_df_pop_saved, compress=True)
    print(path_df_pop_saved)


# In[58]:


df_range = create_seek_ranges(nb_first_seed=NB_FIRST_SEED)
arr_range = df_range.values.astype(np.int64) # f(num range, families)
df_range


# In[59]:


df_prob_fam = joblib.load(PATH_SAVE_PROB_FAM)
df_prob = joblib.load(PATH_TO_SAVE_DATA + '/df_prob.pkl')
# patch to optimize mutation fonction:
arr_prob = np.array(df_prob)
arr_prob_fam = np.array(df_prob_fam.astype("float"))


if (SAVE_POP is None) & os.path.isfile(PATH_DF_POP):
    print("Loading: ", PATH_DF_POP)
    df_pop = joblib.load(PATH_DF_POP)
else:
    print("Loading: ", path_df_pop_saved)
    df_pop = joblib.load(path_df_pop_saved)
    
# info about first pop
print("Infos about population: ")
plt.plot(df_pop.columns,df_pop.loc[0]-df_pop.loc[1])
plt.title("Example : indiv_0 - indiv_1")
print("Max Same indiv nb : ", find_max_same_indiv(df_pop.values))
_, df_des_choices_0, _ = pop_choices_info(df_pop)
df_des_choices_0


# In[60]:


# create cost dataFrame for all population
t_0 = time.time()
arr_pop = np.array(df_pop)
arr_score = eval_cost_vect_optim(arr_pop)
df_cost = pd.DataFrame(data=arr_score, columns=["cost"])
print("Timing: ", time.time()-t_0)
df_cost.boxplot()
df_cost.sort_values(by="cost").head(10)


# In[61]:


POW_SELECTION = 0.3


# In[62]:


# Prob for indiv = inverse rank * POW_SELECTION
arr_select_prob = selection_prob(df_cost, pow_selection=POW_SELECTION)
plt.plot(np.sort(arr_select_prob))
plt.title("Selection probabilities");


# In[63]:


arr_pop = np.array(df_pop) # df_pop = f(indiv., families)
arr_cost = df_cost["cost"].values


# In[64]:


## HYPERPARAMETERS
NB_MAX_EPOCHS = 125000
R_POP_MUT = 0.1
R_MUT = 0.01 
DELTA_CHOICE = 2
NB_BEST_KEEP = 20
POW_SELECTION = 0.3
flag_boost = True
boost_freq = 2000
R_CROSSOVER = 1
boost_sps_freq = 1000 # epochs frequency to run Stochastic Product Search 
thr_sps_cost = 0 # delta cost over boost_sps_freq epochs : -1 to disable

## DISPLAY PARAM
flag_prompt = False # timing information to each steps 
prompt_freq = 500 #100 # frequency info about cost & timing 

## PREPARE LOOP
# prepare data 
FIRST_COST = arr_cost.min()
nb_indiv_boost = 0
nb_sps_not_found = 0
#list_best_cost = []
list_best_cost = np.empty(0)
indice_cost = np.argsort(arr_cost)
t_fit_0 = time.time()

## LOOP OVER GENERATIONS (MAIN ALGO GEN)
for gen_id in range(0, NB_MAX_EPOCHS):
    
    t_epoch_0 = time.time()
    if flag_prompt:
        t_lost_0 = time.time()

    # add current epoch best cost among population:
    list_best_cost = np.append(list_best_cost, arr_cost.min())
    
    ############
    # STOCHASTIC PRODUCT SEARCH 
    # boost by stochastic product search if critera is reached:
    # critera = if no better improvement of cost during boost_sps_freq epochs
    if (gen_id % boost_sps_freq == 0):
        if (list_best_cost.shape[0] > boost_sps_freq):
            if (list_best_cost[-boost_sps_freq] - list_best_cost[-1]) <=             thr_sps_cost:
                print("GEN. #{} / Stochastic Product Search...".format(gen_id))
                # select best
                arr_best_curr = arr_pop[np.argmin(arr_cost)]
                # search time
                duration_sps = max(6,min(6*nb_sps_not_found, 300)) # max 5 min 
                # run s.p.s
                #arr_best_sps=stochprodsearch_03.sps(arr_best_curr)# BUG ctypes
                arr_best_sps = run_stochprodsearch(arr_best_curr, 
                                                   end_time=duration_sps, 
                                                   nb_jobs=4)
                cost_best_sps = cost_function_optim(arr_best_sps)
                # if better found we replace it into population
                if cost_best_sps < cost_function_optim(arr_best_curr):
                    arr_pop[np.argmin(arr_cost)] = arr_best_sps
                    arr_cost[np.argmin(arr_cost)] = cost_best_sps
                    print("Stochastic Product Search : Best cost: ", 
                          cost_best_sps)
                    nb_sps_not_found = 0
                else:
                    nb_sps_not_found = nb_sps_not_found + 1

    ############
    # SELECTION  
    # calculation of probabilities for crossing next generation 
    # prob =  (1/rank)^POW_SELECTION

    # Keep the NB_BEST_KEEP best indiv.s
    arr_select_prob, arr_best, arr_cost_best = selection_prob_arr(arr_cost, 
      pow_selection=POW_SELECTION, arr_pop=arr_pop, flag_ouput=True,
      nb_best_keep=NB_BEST_KEEP)
    
    if flag_prompt:
        t_lost_1 = time.time()
        print("Timing lost: ", t_lost_1 - t_lost_0)
        
    ############
    # CROSSOVER 
    #
    # Do the Crossover between pair indiv.
    # 1 Cross point is ramdomly choosen (prob uniform)
    # example : 
    # 1-2-3\  /5-8-9-1-3-4-9  
    #       \/
    # 5-6-5/ \4-5-6-7-8-9-10
    #
    # give : 
    #
    # 1-2-3--4-5-6-7-8-9-10
    # 5-6-5--5-8-9-1-3-4-9 
    # create pairs : ramdomly
    if flag_prompt:
        t_cross_0 = time.time()

    # crossing with more prob for best indiv.
    # number of new children = N pop - N best to keep same nb of indiv each gen.
    nb_cross = int(NB_FIRST_POP - NB_BEST_KEEP - nb_indiv_boost)
    # reset nb boost indiv
    if nb_indiv_boost > 0:
        nb_indiv_boost = 0 

    arr_pop = generate_crossing_prob(arr_pop.copy(), p=arr_select_prob, 
                                     n_indiv=nb_cross, r_cross=R_CROSSOVER)
    if flag_prompt:
        t_cross_1 = time.time()
        print("Timing cross: ", t_cross_1 - t_cross_0)

    ############
    # MUTATION
    # among pop, number of mutation = R_POP_MUT * number of indiv
    # arr_pop or df_pop = f(indiv, family)
    if flag_prompt:
        t_mut_0 = time.time()
    arr_pop = fun_vect_mut(arr_pop, r_pop_mut=R_POP_MUT, r_mut=R_MUT, 
                delta_choice=DELTA_CHOICE)
    
    ############    
    ## ADD the best ones
    #
    #arr_pop = np.append(arr_pop, np.array(df_best), axis=0)
    arr_pop = np.append(arr_pop, arr_best, axis=0)
    # AVOID DUPLICATE INDIV 
    if flag_prompt:
        t_mut_1 = time.time()
        print("Timing mutation: ", t_mut_1 - t_mut_0)

    ############
    # EVALUATION
    # create cost dataFrame for all population
    if flag_prompt:
        t_eval_0 = time.time()
    # optim
    arr_cost = eval_cost_vect_optim(arr_pop)
    
    ##########
    # BOOSTING
    #
    # cost of last best submission

    if flag_boost & (gen_id % boost_freq == 0) & (gen_id > 0) :
        # the 1st best indiv only 
        
        best = arr_pop[np.argmin(arr_cost)]
    
        arr_sub, arr_score = boost_diff_browsing_optim(best=best, 
                                                       arr_range=arr_range)
    
        nb_indiv_boost = arr_sub.shape[0]
        
        arr_pop = np.append(arr_pop, arr_sub, axis=0)
        
        arr_cost = np.append(arr_cost, arr_score, axis=0)
    
    ##########
    # AVOID DUPLICATES BY SAME COST : very FASTER
    # but potential diff indiv elimitated...
    #
    
    indices_cost, arr_cost = removeDups(arr_cost)
    arr_pop = arr_pop[indices_cost]
    
    
    if flag_prompt:
        t_eval_1 = time.time()
        print("Timing eval: ", t_eval_1 - t_eval_0)
    
    ##########
    # DISPLAY
    #
    t_epoch_1 = time.time()
    if (flag_prompt) | (gen_id % prompt_freq == 0):    
        print("GEN. #{} / cost: {} / nb. pop: {} / timing: {}".format(gen_id, 
                        arr_cost.min(), arr_pop.shape[0],
                                                      t_epoch_1 - t_epoch_0))        

# timing 
t_fit_1 = time.time()
print("END:")

print("LAST GEN. #{} / cost: {} / nb. pop: {} / timing Total: {}".format(gen_id, 
                        arr_cost.min(), arr_pop.shape[0], t_fit_1 - t_fit_0)) 


df_cost = pd.DataFrame(data=arr_cost, columns=["cost"])
df_pop = pd.DataFrame(arr_pop)

# figure
fig = plt.figure(figsize=(12, 8))
plt.plot(list_best_cost)
plt.title("Cost over generations")
ax = fig.gca()
ax.set_xlabel("epochs [-]")
ax.set_ylabel("cost [$]")


# In[65]:


arr_cost.min()


# In[66]:


print("How many same indiv. into population at the end : ")
find_max_same_indiv(df_pop)


# In[67]:


print("Pop Info :")
df_choices, df_des_choices, std_mean = pop_choices_info(df_pop)
df_des_choices


# In[68]:


fig = plt.figure(figsize=(12, 6))
plt.plot(df_cost.sort_values(by="cost").head(NB_BEST_KEEP+300).values)
ax = fig.gca()
ax.set_title("Best Cost")
ax.set_ylabel("Cost $")
ax.set_xlabel('assignement n#');


# In[69]:


# POP
joblib.dump(df_pop,
    PATH_TO_SAVE_DATA + '/df_pop{}_fs{}_rfm{}_dc{}_rm{}_nk_{}_gen{}_s{}.pkl'.format(
                NB_FIRST_POP,
                NB_FIRST_SEED, 
                R_FIRST_MUT,
                DELTA_CHOICE,
                R_POP_MUT,
                NB_BEST_KEEP,
                NB_MAX_EPOCHS,
                df_cost.sort_values(by="cost").iloc[0,0]),
           compress=True)
# COST
joblib.dump(df_cost,
    PATH_TO_SAVE_DATA + \
            '/df_cost_pop{}_fs{}_rfm{}_dc{}_rm{}_nk_{}_gen{}_s{}.pkl'.format(
                NB_FIRST_POP,
                NB_FIRST_SEED, 
                R_FIRST_MUT,
                DELTA_CHOICE,
                R_POP_MUT,
                NB_BEST_KEEP,
                NB_MAX_EPOCHS,
                df_cost.sort_values(by="cost").iloc[0,0]),
           compress=True)

# submission csv
submission_final = pd.DataFrame(columns=["assigned_day"])
submission_final["assigned_day"] =     df_pop.loc[df_cost.sort_values(by="cost").iloc[0].name]
submission_final.index.name = 'family_id'
submission_final.to_csv("submission_{}.csv".format(
    df_cost.sort_values(by="cost").iloc[0,0]))           
print("Submission saved here :", "submission_{}.csv".format(
    df_cost.sort_values(by="cost").iloc[0,0]))

# SUBMISSION pickle
joblib.dump(submission_final,
    PATH_TO_SAVE_DATA + \
            '/submission_pop{}_fs{}_rfm{}_dc{}_rm{}_nk_{}_gen{}_s{}.pkl'.format(
                NB_FIRST_POP,
                NB_FIRST_SEED, 
                R_FIRST_MUT,
                DELTA_CHOICE,
                R_POP_MUT,
                NB_BEST_KEEP,
                NB_MAX_EPOCHS,
                df_cost.sort_values(by="cost").iloc[0,0]),
           compress=True)
# list cost vs epochs 
joblib.dump(list_best_cost,
    PATH_TO_SAVE_DATA + \
            '/list_best_cost{}_fs{}_rfm{}_dc{}_rm{}_nk_{}_gen{}_s{}.pkl'.format(
                NB_FIRST_POP,
                NB_FIRST_SEED, 
                R_FIRST_MUT,
                DELTA_CHOICE,
                R_POP_MUT, 
                NB_BEST_KEEP,
                NB_MAX_EPOCHS,
                df_cost.sort_values(by="cost").iloc[0,0]),
           compress=True)



# In[70]:


BEST_COST = df_cost.sort_values(by="cost")["cost"].iloc[0]
# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
try:
    timing = t_fit_1 - t_fit_0
except:
    timing = t_epoch_1 - t_fit_0 

nb_pop = df_pop.shape[0]
df_res = pd.DataFrame(columns=["date", "COMPUTERNAME", "FIRST_COST", 
                               "BEST_COST", "NB_MAX_EPOCHS", "timing", "nb_pop",
                              "NB_FIRST_SEED", "DELTA_CHOICE_FIRST_POP",
                              "R_FIRST_MUT", "R_POP_MUT", "R_MUT", 
                               "DELTA_CHOICE", "NB_BEST_KEEP", "POW_SELECTION",
                              "flag_boost", "boost_freq"], index=[0])

df_res["date"] = dt_string
df_res["COMPUTERNAME"] = COMPUTERNAME
df_res["FIRST_COST"] = FIRST_COST
df_res["BEST_COST"] = BEST_COST
df_res["NB_MAX_EPOCHS"] = NB_MAX_EPOCHS
df_res["timing"] = timing
df_res["nb_pop"] = df_pop.shape[0]
df_res["NB_FIRST_SEED"] = NB_FIRST_SEED
df_res["DELTA_CHOICE_FIRST_POP"] = DELTA_CHOICE_FIRST_POP
df_res["R_FIRST_MUT"] = R_FIRST_MUT
df_res["R_POP_MUT"] = R_POP_MUT
df_res["R_MUT"] = R_MUT
df_res["DELTA_CHOICE"] = DELTA_CHOICE
df_res["NB_BEST_KEEP"] = NB_BEST_KEEP
df_res["POW_SELECTION"] = POW_SELECTION
df_res["flag_boost"] = flag_boost
df_res["boost_freq"] = boost_freq
df_res["pop_path"] = PATH_DF_POP
df_res.index.name = 'job'
df_res.to_csv(PATH_TO_SAVE_DATA +               '/res_pop{}_fs{}_rfm{}_dc{}_rm{}_nk_{}_gen{}_s{}.csv'.format(
                NB_FIRST_POP,
                NB_FIRST_SEED, 
                R_FIRST_MUT,
                DELTA_CHOICE,
                R_POP_MUT,
                NB_BEST_KEEP,
                NB_MAX_EPOCHS,
                df_cost.sort_values(by="cost").iloc[0,0]))
df_res


# In[71]:


df_pop_0 = joblib.load(PATH_DF_POP)

df_choices_0, df_des_choices_0, std_mean_0 = pop_choices_info(df_pop_0)
df_des_choices_0


# In[72]:


df_choices_0.loc[0].value_counts()


# In[73]:


best = df_pop.loc[df_cost["cost"].idxmin()].values


# In[74]:


cost_function_optim(best)


# In[75]:


penalty, accounting_cost , daily_occupancy = cost_function(best, flag_prompt=True)
df_daily = pd.DataFrame(index=daily_occupancy.keys(), data=list(daily_occupancy.values()), 
             columns=['nb_people'])


# In[76]:


df_daily.describe()


# In[77]:


submission_opti = pd.read_csv(PATH_TO_EXPLORE_DATA + '/submission_68888.04.csv', 
                         index_col='family_id')
# find daily occ
penalty_opti, accounting_cost_opti , daily_occupancy_opti =     cost_function(submission_opti["assigned_day"].values, flag_prompt=True)

df_daily_opti = pd.DataFrame(index=daily_occupancy_opti.keys(), 
                             data=list(daily_occupancy_opti.values()), 
             columns=['nb_people'])
df_daily["nb_opti"] = df_daily_opti['nb_people']
df_daily["delta_nb_opti"] = df_daily["nb_opti"] - df_daily['nb_people']


# In[78]:


fig = plt.figure(figsize=(14, 6))
plt.title("Occupancy : compare my best / optimal")
plt.plot(df_daily.index, df_daily["nb_people"],'+-', label="my_best")
plt.plot(df_daily.index, df_daily["nb_opti"],'.-', label="opti")
plt.legend(loc='upper right');
ax = fig.gca()
ax.set_xlabel("Days before Christmas")
ax.set_ylabel("nb people");


# In[79]:


arr_choice_best = fun_find_choices_sub(best)
arr_choice_opti = fun_find_choices_sub(submission_opti["assigned_day"].values)

fig = plt.figure(figsize=(14, 8))
plt.title("choices")
plt.plot(arr_choice_best, '+', label="best")
plt.plot(arr_choice_opti, '.', label="opti")
plt.legend(loc='best');


# In[80]:


fig = plt.figure(figsize=(14, 8))
plt.title("delta choices : best-opti")
plt.plot(arr_choice_best-arr_choice_opti, '.')


# In[81]:


arr_choices = fun_find_choices_sub(best)
df_best = pd.DataFrame(arr_choices.astype(np.int64), columns=["choice"]) 
df_best["day"] = best
df_best["day_opti"] = submission_opti["assigned_day"].values
df_best["choice_opti"] = arr_choice_opti.astype(np.int64)
df_best.describe()


# In[82]:


df_best["choice"].value_counts()


# In[83]:


df_best["choice_opti"].value_counts()


# In[84]:


fig = plt.figure(figsize=(14, 6)) 
plt.title("Count of choices # : compare current best & optimal")
ax = sns.countplot(y="value", hue="variable", 
                   data=df_best.melt(value_vars=['choice', 'choice_opti']))
ax.set_ylabel("choice #");


# In[85]:


df_des_choices


# In[86]:


df_des_choices.loc["std"].max()


# In[87]:


plot_delta_choice_pop(df_pop, df_des_choices)


# In[88]:


plot_std_choice_pop(df_pop, df_des_choices)


# In[89]:


sns.boxplot(df_des_choices.loc["std"])


# In[90]:


sub_my_best = pd.read_csv(PATH_TO_EXPLORE_DATA +                           '/submission_71447.87946293628_for_sps.csv', 
                   index_col='family_id')
sub_opti = pd.read_csv(PATH_TO_EXPLORE_DATA +                           '/submission_68888.04.csv', 
                   index_col='family_id')


# In[91]:



arr_choice_best = fun_find_choices_sub(sub_my_best["assigned_day"].values)
arr_choice_opti = fun_find_choices_sub(sub_opti["assigned_day"].values)

fig = plt.figure(figsize=(16, 8))
plt.title("Compare my best with optimum choices")
plt.plot(arr_choice_best, '+', label="best")
plt.plot(arr_choice_opti, '.', label="opti")
plt.legend(loc='upper left');


# In[92]:


fig = plt.figure(figsize=(12, 8))
plt.title("delta choices : best-opti")
plt.plot(arr_choice_best-arr_choice_opti, '.')


# In[93]:


penalty, accounting_cost , daily_occupancy =     cost_function(sub_my_best["assigned_day"].values, flag_prompt=True)

df_daily = pd.DataFrame(index=daily_occupancy.keys(), 
                        data=list(daily_occupancy.values()), 
                        columns=['nb_people'])
#df_daily.plot()

# find daily occ
penalty_opti, accounting_cost_opti , daily_occupancy_opti =     cost_function(sub_opti["assigned_day"].values, flag_prompt=True)

df_daily_opti = pd.DataFrame(index=daily_occupancy_opti.keys(), 
                             data=list(daily_occupancy_opti.values()), 
             columns=['nb_people'])
df_daily["nb_opti"] = df_daily_opti['nb_people']
df_daily["delta_nb_opti"] = df_daily["nb_opti"] - df_daily['nb_people']


# In[94]:


df_daily


# In[95]:


fig = plt.figure(figsize=(12, 6))
plt.title("Occupancy : compare my best / optimal")
plt.plot(df_daily.index, df_daily["nb_people"], label="my_best")
plt.plot(df_daily.index, df_daily["nb_opti"], label="opti")
plt.legend(loc='upper right');
ax = fig.gca()
ax.set_xlabel("Days before Christmas")
ax.set_ylabel("nb people")


# In[96]:


df_daily["delta_nb_opti"].plot()


# In[97]:


arr_choices = fun_find_choices_sub(sub_my_best["assigned_day"].values)
df_best = pd.DataFrame(arr_choices.astype(np.int64), columns=["choice"]) 
df_best["day"] = best
df_best["day_opti"] = sub_opti["assigned_day"].values
df_best["choice_opti"] = arr_choice_opti.astype(np.int64)
df_best.describe()


# In[98]:


df_best["choice"].value_counts()


# In[99]:


df_best["choice_opti"].value_counts()


# In[100]:


fig = plt.figure(figsize=(14, 6)) 
plt.title("Count of choices # : compare current best & optimal")
ax = sns.countplot(y="value", hue="variable", 
                   data=df_best.melt(value_vars=['choice', 'choice_opti']))
ax.set_ylabel("choice #");


# In[101]:


list_best_cost200_1 = joblib.load(PATH_TO_EXPLORE_DATA + '/' +
    'list_best_cost200_fs10_rfm0.05_dc2_rm0.2_nk_3_gen1000000_s426993.2726332134.pkl')
list_best_cost1000_1 = joblib.load(PATH_TO_EXPLORE_DATA + '/' +
    'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen200000_s89924.27707458043.pkl')
list_best_cost2000_1 = joblib.load(PATH_TO_EXPLORE_DATA + '/' +
    'list_best_cost2000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen100000_s98547.53260584155.pkl')
list_best_cost2000_1 = np.array(list_best_cost2000_1)
list_best_cost2000_2 = joblib.load(PATH_TO_EXPLORE_DATA + '/' +
    'list_best_cost2000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen200000_s90591.5811766185.pkl')
list_best_cost2000_2 = np.array(list_best_cost2000_2)

fig = plt.figure(figsize=(12, 8))
plt.plot(list_best_cost200_1, label="200")
plt.plot(list_best_cost1000_1, label="1000 #1")
plt.plot(list_best_cost2000_1, label="2000 #1")
plt.plot(list_best_cost2000_2, label="2000")
plt.legend(loc='upper right');
ax = fig.gca()
ax.set_xlabel("epochs [-]")
ax.set_ylabel("cost [$]")
ax.set_title("Population size impact");


# In[102]:


fig = plt.figure(figsize=(12, 8))
plt.plot(np.arange(list_best_cost200_1.shape[0])*200/1000,list_best_cost200_1, 
         label="200")
plt.plot(list_best_cost1000_1, label="1000 #1")
plt.plot(np.arange(list_best_cost2000_1.shape[0])*2000/1000, 
         list_best_cost2000_1, label="2000 #1")
plt.plot(np.arange(list_best_cost2000_2.shape[0])*2000/1000, 
         list_best_cost2000_2, label="2000 #2")
plt.legend(loc='upper right');
ax = fig.gca()
ax.set_xlabel("equivalent computation time [-]")
ax.set_ylabel("cost [$]")
ax.set_title("Population size impact");


# In[103]:


fig = plt.figure(figsize=(12, 8))
plt.plot(np.arange(list_best_cost200_1.shape[0])*200/1000,list_best_cost200_1, 
         label="200")
plt.plot(list_best_cost1000_1, label="1000 #1")
plt.plot(np.arange(list_best_cost2000_1.shape[0])*2000/1000, 
         list_best_cost2000_1, label="2000 #1")
plt.plot(np.arange(list_best_cost2000_2.shape[0])*2000/1000, 
         list_best_cost2000_2, label="2000 #2")
plt.legend(loc='upper right');
ax = fig.gca()
ax.set_xlabel("equivalent computation time [-]")
ax.set_ylabel("cost [$]")
ax.set_title("Population size impact");
ax.set_xlim([0,25000])


# In[104]:


path_RPM1 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen100000_s99225.99788819549.pkl'
path_RPM2 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.2_nk_20_gen50000_s381309.48944465496.pkl'
path_RPM3 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.3_nk_20_gen50000_s421919.51684736356.pkl'
list_best_cost_RPM1 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_RPM1)
list_best_cost_RPM2 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_RPM2)
list_best_cost_RPM3 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_RPM3)
fig = plt.figure(figsize=(12, 8))
plt.plot(list_best_cost_RPM1, label="R_POP_MUT = 0.1")
plt.plot(list_best_cost_RPM2, label="R_POP_MUT = 0.2")
plt.plot(list_best_cost_RPM3, label="R_POP_MUT = 0.3")
plt.legend(loc='upper right');
ax = fig.gca()
ax.set_xlabel("epochs [-]")
ax.set_ylabel("cost [$]")
ax.set_title("R_POP_MUT impact");
ax.set_xlim([0 ,50000]);


# In[105]:


path_RM1 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen100000_s99225.99788819549.pkl'
path_RM2 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen50000_s109860.65058950568.pkl'
path_RM5 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen50000_s214233.05776615342.pkl'
list_best_cost_RPM1 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_RM1)
list_best_cost_RPM2 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_RM2)
list_best_cost_RPM5 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_RM5)
fig = plt.figure(figsize=(12, 8))
plt.plot(list_best_cost_RPM1, label="R_MUT = 0.01")
plt.plot(list_best_cost_RPM2, label="R_MUT = 0.02")
plt.plot(list_best_cost_RPM5, label="R_MUT = 0.05")
plt.legend(loc='upper right');
ax = fig.gca()
ax.set_xlabel("epochs [-]")
ax.set_ylabel("cost [$]")
ax.set_title("R_MUT impact");
ax.set_xlim([0 ,50000]);


# In[106]:


path_NB1 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen200000_s89924.27707458043.pkl'
path_NB2_1 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen100000_s96208.60596179293.pkl'
path_NB2_2 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen100000_s94453.49753916638.pkl'
path_NB2_3 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen100000_s91042.12644135197.pkl'

path_B1_1 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen50000_s319490.03112310503.pkl'
path_B1_2 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen150000_s290524.2660717894.pkl'
path_B1_3 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen100000_s270105.47270251444.pkl'
path_B1_4 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen100000_s92151.41696552017.pkl'
path_B1_5 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen100000_s89450.1477974278.pkl'
path_B1_6 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen100000_s88419.49004815941.pkl'

path_B2_1 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen50000_s91072.03135739548.pkl'
path_B2_2 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen150000_s89893.01980541403.pkl'
path_B2_3 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen100000_s89027.70096301885.pkl'
path_B2_4 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen100000_s87875.04506597946.pkl'
path_B2_5 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen100000_s86868.67258295594.pkl'
path_B2_6 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen100000_s86508.96967131883.pkl'


# In[107]:


list_best_cost_NB1 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_NB1)

list_best_cost_NB2_1 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_NB2_1)
list_best_cost_NB2_2 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_NB2_2)
list_best_cost_NB2_3 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_NB2_3)
list_best_cost_NB2 = np.concatenate((list_best_cost_NB2_1, list_best_cost_NB2_2,
                                     list_best_cost_NB2_3))
list_best_cost_B1_1 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_B1_1)
list_best_cost_B1_2 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_B1_2)
list_best_cost_B1_3 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_B1_3)
list_best_cost_B1_4 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_B1_4)
list_best_cost_B1_5 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_B1_5)
list_best_cost_B1_6 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_B1_6)
list_best_cost_B1 = np.concatenate((list_best_cost_B1_1, list_best_cost_B1_2,
                                     list_best_cost_B1_3, list_best_cost_B1_4,
                                    list_best_cost_B1_5, list_best_cost_B1_6))

list_best_cost_B2_1 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_B2_1)
list_best_cost_B2_2 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_B2_2)
list_best_cost_B2_3 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_B2_3)
list_best_cost_B2_4 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_B2_4)
list_best_cost_B2_5 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_B2_5)
list_best_cost_B2_6 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_B2_6)
list_best_cost_B2 = np.concatenate((list_best_cost_B2_1, list_best_cost_B2_2,
                                     list_best_cost_B2_3, list_best_cost_B2_4,
                                    list_best_cost_B2_5, list_best_cost_B2_6))

fig = plt.figure(figsize=(12, 8))
plt.plot(list_best_cost_NB1, label="no boost #1")
plt.plot(list_best_cost_NB2, label="no boost #2")
plt.plot(list_best_cost_B1, label="boost #1")
plt.plot(list_best_cost_B2, label="boost #2")
plt.legend(loc='upper right');
ax = fig.gca()
ax.set_xlabel("epochs [-]")
ax.set_ylabel("cost [$]")
ax.set_title("Simple seq. Boost impact");
#ax.set_xlim([0 ,50000]);


# In[108]:


fig = plt.figure(figsize=(12, 8))
plt.plot(list_best_cost_NB1, label="no boost #1")
plt.plot(list_best_cost_NB2, label="no boost #2")
plt.plot(list_best_cost_B1, label="boost #1")
plt.plot(list_best_cost_B2, label="boost #2")
plt.legend(loc='upper right');
ax = fig.gca()
ax.set_xlabel("epochs [-]")
ax.set_ylabel("cost [$]")
ax.set_title("Simple seq. Boost impact [Start Zoom.]");
ax.set_xlim([0 ,50000]);


# In[109]:


fig = plt.figure(figsize=(12, 8))
plt.plot(list_best_cost_NB1, label="no boost #1")
plt.plot(list_best_cost_NB2, label="no boost #2")
plt.plot(list_best_cost_B1, label="boost #1")
plt.plot(list_best_cost_B2, label="boost #2")
plt.legend(loc='upper right');
ax = fig.gca()
ax.set_xlabel("epochs [-]")
ax.set_ylabel("cost [$]")
ax.set_title("Simple seq. Boost impact [End Zoom.]");
ax.set_xlim([280000, 600000]);
ax.set_ylim([80000, 100000])


# In[110]:


path_cr4_1 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen200000_s84928.91135928089.pkl'
path_cr4_2 = 'list_best_cost1000_fs10_rfm0.05_dc2_rm0.1_nk_20_gen200000_s84089.67769691352.pkl'
list_best_cost_cr4_1 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_cr4_1)
list_best_cost_cr4_2 = joblib.load(PATH_TO_EXPLORE_DATA + '/' + path_cr4_2)
list_best_cost_cr4 = np.concatenate((list_best_cost_cr4_1, 
                                     list_best_cost_cr4_2))


# In[111]:


fig = plt.figure(figsize=(12, 8))
plt.plot(list_best_cost_cr4, label="CH RANGE=4 & boost")
plt.plot(list_best_cost_B2, label="CH RANGE=10 & boost")
plt.legend(loc='upper right');
ax = fig.gca()
ax.set_xlabel("epochs [-]")
ax.set_ylabel("cost [$]")
ax.set_title("CHOICE_RANGE_MAX impact");
ax.set_xlim([0 ,50000]);


# In[112]:


fig = plt.figure(figsize=(12, 8))
plt.plot(list_best_cost_cr4, label="CH RANGE=4 & boost")
plt.plot(list_best_cost_B2, label="CH RANGE=10 & boost")
plt.legend(loc='upper right');
ax = fig.gca()
ax.set_xlabel("epochs [-]")
ax.set_ylabel("cost [$]")
ax.set_title("CHOICE_RANGE_MAX impact");
ax.set_xlim([15000, 50000]);
ax.set_ylim([80000, 120000]);


# In[113]:


fig = plt.figure(figsize=(12, 8))
plt.plot(list_best_cost_cr4, label="CH RANGE=4 & boost")
plt.plot(list_best_cost_B2, label="CH RANGE=10 & boost")
plt.legend(loc='upper right');
ax = fig.gca()
ax.set_xlabel("epochs [-]")
ax.set_ylabel("cost [$]")
ax.set_title("CHOICE_RANGE_MAX impact [Zoom end.]");
#ax.set_xlim([50000, 50000]);
ax.set_ylim([80000, 100000]);


# In[ ]:





# In[114]:


submission_final


# In[ ]:




