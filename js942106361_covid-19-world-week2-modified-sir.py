#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import matplotlib.pyplot as plt

# PEP 484 type hints are easier with this
from typing import Tuple


# In[3]:


fname_train = '../input/covid19-global-forecasting-week-2/train.csv'
fname_test = '../input/covid19-global-forecasting-week-2/test.csv'
fname_sub = '../input/covid19-global-forecasting-week-2/submission.csv'


# In[4]:


df_train = pd.read_csv(fname_train,parse_dates=True)
df_test  = pd.read_csv(fname_test, parse_dates=True)
df_sub   = pd.read_csv(fname_sub)


# In[5]:


# Week 2 globals
# Training dates are from 2020-01-22 to 2020-03-25 ... but the end date can extend each day as more data is added
#                                                  ... possibly up to 2020-03-31
# Week 2: Training index goes 29364 and contains 294 locations
START_TRAIN_DATE = df_train['Date'].min()   # Week 2: '2020-01-22'
END_TRAIN_DATE   = df_train['Date'].max()   # Week 2: '2020-03-25'

# Prediction dates are from 3/19/2020 to 4/30/2020 (43 dates per region)
# Prediction index goes to 12642
START_PREDICTION_DATE = df_test['Date'].min() # Week 2: '2020-03-19'
END_PREDICTION_DATE   = df_test['Date'].max() # Week 2: '2020-04-30'


# In[6]:


# Tanu's population by country
fname_pop = '../input/population-by-country-2020/population_by_country_2020.csv'
df_pop = pd.read_csv(fname_pop)


# In[7]:


df_pop.head()


# In[8]:


GLOBAL_FILL_STRING = 'not_provided'


# In[9]:


prediction_scores = {}  # create a dictionary of prediction scores so that I know where to consider improvements


# In[10]:


def check_first_ConfirmedCase_exists(df_in:pd.DataFrame,
                                     region_list:set)->int:
    '''Confirm that the data for the region contains at least the first ConfirmedCase
    '''
    count_no_cases = 0
    
    for sname in sname_set:       
        df1 = df_in[df_in['SimpleName'] == sname]   # grab the entries for this location only        
        total_cc = df1['ConfirmedCases'].max()        
        if (total_cc < 1):
            print("{} as 0 ConfirmedCases".format(sname))
            count_no_cases += 1
            
    return count_no_cases


# In[11]:


def evaluate_predictions(df_golden:pd.DataFrame,df_predictions:pd.DataFrame,                         num_wh=0, num_extended=0,                         model_name="no-name",verbose=False)->Tuple[(float,float)]:
    '''Compare the predicted values in df_predictions to the values in df_golden for where the data was withheld.
        Data in the extended region is ignored.
        
       Note: DataFrames MUST only contain a single prediction series for ONE location.
    '''
    round_places = 4
    
    # changed names df_train  --> df_golden
    #               df_pred   --> df_predictions
    
    last_golden_date = df_golden['DateTime'].max()
    
    last_predicted_date = df_predictions['DateTime'].max()
    first_withheld_date = last_golden_date - (num_wh)*pd.to_timedelta('1D')
    if (verbose):
        print("last_golden_date: {}".format(last_golden_date))
        print("last_predicted_date: {}".format(last_predicted_date))
        print("first_withheld_date: {}".format(first_withheld_date))
    # comparison period spans the first_withheld_date to the last_golden_date inclusive
    
    start_eval_time = first_withheld_date
    end_eval_time   = last_golden_date
     
    df_pred_for_eval = df_predictions[(df_predictions['DateTime'] >= start_eval_time) & 
                                      (df_predictions['DateTime'] <= end_eval_time)]
    df_answers       = df_golden[(df_golden['DateTime'] >= start_eval_time) & 
                                 (df_golden['DateTime'] <= end_eval_time)]
    
    if (verbose):
        print("df_answers is {}".format(df_answers))
        print("df_pred_for_eval is {}".format(df_pred_for_eval))
        
    
    c_rmsle = round(rmsle(list(df_answers['ConfirmedCases']),
                          list(df_pred_for_eval['ConfirmedCases'])),
                    round_places)
    f_rmsle = round(rmsle(list(df_answers['Fatalities']),
                          list(df_pred_for_eval['Fatalities'])),
                    round_places)
    
    if (verbose):
        print("Model {} for {} through {}".format(model_name,start_eval_time, end_eval_time))
        print("   evaluation (ideal --> 0):\n    RMSLE confirmed: {}\n    RMSLE fatalities: {}".format(c_rmsle,f_rmsle))
    
    return c_rmsle, f_rmsle


# In[12]:


def evaluate_predictions2(df_golden:pd.DataFrame,df_predictions:pd.DataFrame,                         num_wh=0, num_extended=0,                         model_name="no-name",verbose=False)->Tuple[(float,float)]:
    '''Compare the predicted values in df_predictions to the values in df_golden for where the data was withheld.
        Data in the extended region is ignored.
        
       Note: DataFrames MUST only contain a single prediction series for ONE location.
    '''
    round_places = 4
    
    # changed names df_train  --> df_golden
    #               df_pred   --> df_predictions
    
    last_golden_date = df_golden['DateTime'].max()
    
    last_predicted_date = df_predictions['DateTime'].max()
    first_withheld_date = last_golden_date - (num_wh)*pd.to_timedelta('1D')
    if (verbose):
        print("last_golden_date: {}".format(last_golden_date))
        print("last_predicted_date: {}".format(last_predicted_date))
        print("first_withheld_date: {}".format(first_withheld_date))
    # comparison period spans the first_withheld_date to the last_golden_date inclusive
    
    start_eval_time = first_withheld_date
    end_eval_time   = last_golden_date
     
    df_pred_for_eval = df_predictions[(df_predictions['DateTime'] >= start_eval_time) & 
                                      (df_predictions['DateTime'] <= end_eval_time)]
    df_answers       = df_golden[(df_golden['DateTime'] >= start_eval_time) & 
                                 (df_golden['DateTime'] <= end_eval_time)]
    
    if (verbose):
        print("df_answers is {}".format(df_answers))
        print("df_pred_for_eval is {}".format(df_pred_for_eval))
        
    
    c_rmsle = round(rmsle(list(df_answers['ConfirmedCases']),
                          list(df_pred_for_eval['maxConfirmedCases'])),
                    round_places)
    f_rmsle = round(rmsle(list(df_answers['Fatalities']),
                          list(df_pred_for_eval['Fatalities'])),
                    round_places)
    
    if (verbose):
        print("Model {} for {} through {}".format(model_name,start_eval_time, end_eval_time))
        print("   evaluation 2 (ideal --> 0):\n    RMSLE confirmed: {}\n    RMSLE fatalities: {}".format(c_rmsle,f_rmsle))
    
    return c_rmsle, f_rmsle


# In[13]:


# From https://www.kaggle.com/marknagelberg/rmsle-function
import math

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5


# In[14]:


# unit test for evaluate_predictions, withheld = 1, extended = 2
golden_data    = {'DateTime':pd.to_datetime(['2020-06-01','2020-06-02','2020-06-03']),
                  'ConfirmedCases':[2,4,8],
                  'Fatalities':[0,0,1],
                  'Province_State':['PS_TBD','PS_TBD','PS_TBD'],
                  'Country_Region':['CR_TBD','CR_TBD','CR_TBD'],
                  'Id':[201,202,203],
                  'SimpleName':['TBD','TBD','TBD']}
predicted_data = {'DateTime':pd.to_datetime(['2020-06-01','2020-06-02','2020-06-03','2020-06-04','2020-06-05']),
                  'ConfirmedCases':[2,4,6,8,10],
                  'Fatalities':[0,0,0,1,2],
                  'SimpleName':['TBD','TBD','TBD','TBD','TBD']}
df_golden = pd.DataFrame.from_dict(golden_data)
df_predicted = pd.DataFrame.from_dict(predicted_data)


unit_c, unit_f = evaluate_predictions(df_golden,df_predicted,1,2,'unit test',False)
assert((unit_c == 0.1777) & (unit_f == 0.4901)),'Failed unit test for evaluate_predictions'


# In[15]:


def create_parameter_database():
    '''Creates a database of parameters for use in the model.
       Note that this uses 
           global parameter sname_set 
           default values   for all parameters (intent to replace later)
    '''
    # Create a database of BETA, GAMMA, i_factor, r_factor for each SimpleName
    #  Initially, just use the same values I used for CA
    #  Next step will be to customize based on the region, e.g. temperature, proximity to international airports, etc.


    # defaults   beta=0.38  gamma=0.14   ifx=30   rfx=1000
    ones_column = np.ones((len(sname_set),))
    beta_column = 0.38 * ones_column
    gamma_column = 0.14 * ones_column
    ifx_column = 30.0 * ones_column
    rfx_column = 1000.0 * ones_column  

    pop_column = 1000000 * ones_column  ## population - default to 1 million

    data = {'beta':beta_column,'gamma':gamma_column,'ifx':ifx_column,'rfx':rfx_column,'population':pop_column}
    return data


# In[16]:


def plot_train_predictions(df_pred_in,df_train_in,title_str,ylim_factor=1.2):
    fig, axes = plt.subplots(3,2,figsize=(12,6))
    plt.tight_layout()    
    
    title_cc = title_str + ' ConfirmedCases'
    axes[0,0].set_title(title_cc)
    df_pred_in.plot(ax=axes[0,0],legend=True,x='DateTime',y='ConfirmedCases',label='p_Confirmed',c='lightgreen')
    df_train_in.plot(ax=axes[0,0],legend=True,x='DateTime',y='ConfirmedCases',label='Confirmed',c='orange')

    title_f = title_str + ' Fatalities'
    axes[0,1].set_title(title_f)
    df_pred_in.plot(ax=axes[0,1],legend=True,x='DateTime',y='Fatalities',label='p_Fatalities',c='green')
    df_train_in.plot(ax=axes[0,1],legend=True,x='DateTime',y='Fatalities',label='Fatalities',c='darkblue')
    
    #Scale y-axis to be the known ConfirmedCases + 20% to ensure model dovetails with known data
    ylim_cc = ylim_factor * df_train_in['ConfirmedCases'].max()
    axes[1,0].set_title(title_cc)
    axes[1,0].set_ylim(0,ylim_cc)
    df_pred_in.plot(ax=axes[1,0],legend=True,x='DateTime',y='ConfirmedCases',label='p_Confirmed',c='lightgreen')
    df_train_in.plot(ax=axes[1,0],legend=True,x='DateTime',y='ConfirmedCases',label='Confirmed',c='orange')
    
    #title_f = title_str + ' Fatalities'
    ylim_f = ylim_factor * df_train_in['Fatalities'].max()
    axes[1,1].set_title(title_f)
    axes[1,1].set_ylim(0,ylim_f)
    df_pred_in.plot(ax=axes[1,1],legend=True,x='DateTime',y='Fatalities',label='p_Fatalities',c='green')
    df_train_in.plot(ax=axes[1,1],legend=True,x='DateTime',y='Fatalities',label='Fatalities',c='darkblue')
    
    try:
        df_pred_in['maxConfirmedCases'].shape
        title_cc = title_str + ' maxConfirmedCases'
        axes[2,0].set_title(title_cc)
        df_pred_in.plot(ax=axes[2,0],legend=True,x='DateTime',y='maxConfirmedCases',label='p_maxConfirmed',c='lightgreen')
        df_train_in.plot(ax=axes[2,0],legend=True,x='DateTime',y='ConfirmedCases',label='Confirmed',c='orange')
    except:
        pass


# In[17]:


try: 
    df_sname['maxConfirmedCases'].shape
    print('yeah')
except:
    print('nah')


# In[18]:


df_train.head(n=2)


# In[19]:


df_train.tail(n=2)


# In[20]:


df_test.head(n=2)


# In[21]:


df_test.tail(n=2)


# In[22]:


df_sub.tail(n=2)


# In[ ]:





# In[23]:


df_train[['Province_State','Country_Region']] = df_train[['Province_State','Country_Region']].fillna(GLOBAL_FILL_STRING)
df_test[['Province_State','Country_Region']] = df_test[['Province_State','Country_Region']].fillna(GLOBAL_FILL_STRING)


# In[24]:


df_train['SimpleName'] = df_train['Province_State'] + '__' + df_train['Country_Region']   # for fetching data
df_test['SimpleName'] = df_test['Province_State'] + '__' + df_test['Country_Region']      # for storing predictions


# In[25]:


df_train['DateTime'] = pd.to_datetime(df_train['Date'])   #format = '%Y-%m-%d'
df_test['DateTime']  = pd.to_datetime(df_test['Date'])   #format = '%Y-%m-%d'


# In[26]:


df_test.tail(n=3)


# In[27]:


df_train.tail(n=3)


# In[28]:


df_test.dtypes


# In[29]:


df_train['SimpleName'].nunique()


# In[30]:


sname_set = list(set(df_train['SimpleName']))    # another handy global


# In[31]:


# Make sure there aren't any locations with no ConfirmedCases reported yet
assert(0 == check_first_ConfirmedCase_exists(df_train,sname_set)),"Data contains locations with NO ConfirmedCases!"


# In[32]:


# Start with default values for all locations

parameter_data = create_parameter_database()
db_parameters = pd.DataFrame(parameter_data, index=sname_set,                            columns=['beta','gamma','ifx','rfx','population'])


# In[33]:


db_parameters.head(n=2)


# In[34]:


ctry_col = r'Country (or dependency)'
pop_col  = r'Population (2020)'


# In[35]:


df_pop[df_pop[ctry_col] == 'France'][pop_col].values[0]  # test


# In[36]:


df_pop.head(n=2)


# In[37]:


# TODO: Once the basic functionality generates a valid submission, this is where the creative model tweaks will happen
# UPDATE the db_parameters pd.DataFrame

# populate the parameters with the data from df_pop


for index,row in db_parameters.iterrows():
    state, ctry = index.split('__')   # ctry is ['not_provided', 'Jersey']
    if (state == GLOBAL_FILL_STRING):
        try:
            pop = df_pop[df_pop[ctry_col] == ctry][pop_col].values[0]
        except:
            # 8 or so locations use other names in the population database
            if (ctry == 'Czechia'):
                pop = df_pop[df_pop[ctry_col] == r'Czech Republic (Czechia)'][pop_col].values[0]
            elif (ctry == r'Congo (Kinshasa)'):
                pop = df_pop[df_pop[ctry_col] == r'DR Congo'][pop_col].values[0]
            elif (ctry == r'Congo (Brazzaville)'):
                pop = df_pop[df_pop[ctry_col] == r'Congo'][pop_col].values[0]
            elif (ctry == r'Saint Vincent and the Grenadines'):
                pop = df_pop[df_pop[ctry_col] == r'St. Vincent & Grenadines'][pop_col].values[0]                
            elif (ctry == r'Korea, South'):
                pop = df_pop[df_pop[ctry_col] == r'South Korea'][pop_col].values[0]                                
            elif (ctry == r"Cote d'Ivoire"):
                pop = df_pop[df_pop[ctry_col] == r"Côte d'Ivoire"][pop_col].values[0]   
            elif (ctry == r'Saint Kitts and Nevis'):
                pop = df_pop[df_pop[ctry_col] == r'Saint Kitts & Nevis'][pop_col].values[0]    
            elif (ctry == r'Taiwan*'):
                pop = df_pop[df_pop[ctry_col] == r'Taiwan'][pop_col].values[0]   
            elif (ctry == r'Diamond Princess'):
                pop = 3711   # from Wikipedia  
            else:
                pop=500000   # set a default population of 500k
                print("No population data available for {}, {}".format(state,ctry))
        #print('population is {}'.format(pop))
        #print("{} has population {}".format(ctry,pop))
        db_parameters.loc[index,'population'] = pop
    else:
        # Need to figure out how to handle state populations!
        print("State missing population data: {}, {}".format(state,ctry))


# In[38]:


# Create placeholders for ConfirmedCases and Fatalities
col_len = df_test.shape[0]
zeros_column = np.zeros((col_len,))
df_test['ConfirmedCases'] = zeros_column
df_test['Fatalities'] = zeros_column

# Create a placeholder for max_confirmed -- dataset does not reduce the number of ConfirmedCases to reflect
#   only the active cases
df_test['maxConfirmedCases'] = zeros_column


# In[39]:


df_train_preserve = df_train.copy()  # for debugging and testing
df_test_preserve  = df_test.copy()


# In[40]:


# df_train ready to accept predictions
df_test.tail(n=4)


# In[41]:


# check on France data in df_test -- sanity check of randomly selected location to confirm all 0
df_test[df_test['SimpleName'] == 'not_provided__France'].head(n=22)

Visual representation of how the DataFrames compare:
    Train - this is the input data from Kaggle and represents golden data
         n n n
         n n n 
         n n n
         m m m
         m m m
        
def prepare_prediction_template
         This function takes parameters for the user to decide 
             how many days to withhold
             to what date to extend
     
                                 in (golden_data from Train)               out (template_for_predictions)
                                         n n n                                       n n n
                                         n n n                                       n n n
                                         n n n                                       n n n
                                         m m m                       withheld |      0 0 0        
                                         m m m                       withheld |      0 0 0
                                                                     extended |      0 0 0
           
def sir_model
    This function runs the input template through the model and fills in the missing predictions
              in - template_for_predictions
             out - predictions_all_filled_in
        
def evaluate_model
              in - golden_data from Train
              in - predictions_all_filled_in
               w - number withheld
               e - number extended
      Do the evaluation ONLY on the data that was withheld
             -- data prior to that will be 100% correct
             -- data in the extended section cannot be evaluated
# In[42]:


def prepare_prediction_template(df_in,withhold=0,extend_to_date = END_PREDICTION_DATE,
                                verbose=False):
    '''Prepare a pd.DataFrame that has the known data filled in and the dates ready for the predictions.
        prepare_prediction_template(df_training,withhold=0)   <-- use ALL data, then extend dates
        prepare_prediction_template(df_training,withhold=13)  <-- withhold 13 days, then extend dates
        
       Use withhold=0 for final submission only.
    '''
    # GLOBAL reminders
    # start_train_date = '2020-01-22'
    # end_train_date   = '2020-03-25'
    # start_prediction_date = '2020-03-19'
    # end_prediction_date   = '2020-04-30'
    
    # avoid stupid errors with + or -
    withhold = abs(withhold)
    
    if (withhold != 0):
        # calculate the last date in the training set, then calculate which date to go back to in order
        #    withhold info for training
        # df_train['DateTime'].max() - pd.to_timedelta('13D')     # pattern
        wth_str = str(withhold) + 'D'
        last_date = df_in['DateTime'].max() - pd.to_timedelta(wth_str)
        df_local = df_in[df_in['DateTime']<=last_date].copy()
        if (verbose):
            print("pptemp withhold={} so last_date to copy was {}".format(withhold,last_date))
    else:
        df_local = df_in.copy()
        if (verbose):
            print("pptemp withhold=0 so made complete copy before extension.")
        
    # When extending data, extend these info columns, too
    keep_ps =  df_in.iloc[1]['Province_State']
    keep_cr =  df_in.iloc[1]['Country_Region']
    keep_id =  df_in.iloc[1]['Id'] # not likely to be needed, TBD
    keep_sn =  df_in.iloc[1]['SimpleName']
    
    # Extend the predictions through end_prediction_date; first, find out how far it goes right now
    max_date = df_local['DateTime'].max()

    num_extensions = (pd.to_datetime(extend_to_date) - max_date).days
    if (verbose):
        print("Max date was {} so we need {} extensions.".format(max_date,num_extensions))
    
    # Prepare to extend the dates
    next_date = max_date + pd.to_timedelta('1D')
    #print("types are {} and {}".format(type(next_date),type(end_prediction_date)))
    end_date_datetime = pd.to_datetime(extend_to_date)
    while (next_date <= end_date_datetime):
        #print("Adding a row for new date {}".format(next_date))
        df_local = df_local.append({'Id':0,
                                    'Province_State':keep_ps,
                                    'Country_Region':keep_cr,
                                    'DateTime':next_date,
                                    'ConfirmedCases':0,
                                    'maxConfirmedCases':0,
                                    'Fatalities':0,
                                    'SimpleName':keep_sn},
                                   ignore_index=True)
        next_date = next_date + pd.to_timedelta('1D')
    
    return df_local,num_extensions


# In[43]:


# unit test for prepare_predictions_template
df_golden


# In[44]:


df_1, num_e = prepare_prediction_template(df_golden,withhold=1,extend_to_date='2020-06-09',verbose=True)
print("Output num_e={} and df is \n {}".format(num_e,df_1))


# In[45]:


def calculate_SIR_next_step(S,I,R,N,beta,gamma):
    '''Given the current status of S,I,R, N the total population, and the transition factors, 
        the next step for each category is calculated and returned.
    '''   
    # Governing equations:
    #  dS/dt = (-1*BETA*S*I)/N
    #  dI/dt = (BETA*S*I)/N - GAMMA*I
    #  dR/dt = GAMMA*I
    
    drdt = gamma*I
    dsdt = -1*beta*S*I/N
    didt = -1*dsdt - drdt
    
    # print("dsdt {}  didt {}  drdt {}".format(dsdt,didt,drdt))
    
    S = S + dsdt
    I = I + didt
    R = R + drdt
    
    # Saturate the values
    S = max(S,0)  # max because it's heading to 0
    R = min(R,N)  # min because it's heading to N
    
    if (I<0):
        I = 0
    elif (I>N):
        I = N
    
    return S,I,R  


# In[46]:


def sir_model(df_in,N=100000,beta=.2,gamma=.4,i_factor=1,r_factor=1,verbose=False):
    '''Simple SIR model assuming 
            S = population, N
            I = ConfirmedCases
            R = Fatalities
    '''
    df_local = df_in.copy()   
    first_row = True
    first_CC = False
    
    for index, data_row in df_local.iterrows(): 
        r          = index
        r_previous = index - 1     
        
        if (verbose):
            print("Working on index {} and DateTime {}".format(index,df_local.loc[r,'DateTime']))
            
        c = data_row['ConfirmedCases']
        f = data_row['Fatalities']

        # No changes needed for the first timestamp
        if (first_row):
            first_row = False
            df_local.loc[r,'maxConfirmedCases'] = df_local.loc[r,'ConfirmedCases'] # copy over
        else:
            I = df_local.loc[r_previous,'ConfirmedCases'] * i_factor  # factor to guess at under-estimate
            R = df_local.loc[r_previous]['Fatalities']    * r_factor  # factor to guess at under-estimate
            S = N - (I + R) 
        
            # Start predictions only after the first ConfirmedCase has been found
            if (c > 0):
                first_CC = True   # next time we get c==0, we'll start predicting
                df_local.loc[r,'maxConfirmedCases'] = df_local.loc[r,'ConfirmedCases'] # copy over
            if (first_CC & (c == 0)):    # no prediction for ConfirmedCases
                nS,nI,nR = calculate_SIR_next_step(S,I,R,N,beta,gamma)

                # Resolving the chained indexing issue
                df_local.loc[r,'ConfirmedCases'] = round(nI/i_factor)
                df_local.loc[r,'Fatalities'] = round(nR/r_factor)
                
                # Keep track of the maxConfirmedCases
                df_local.loc[r,'maxConfirmedCases'] = max(df_local.loc[r,'ConfirmedCases'],
                                                          df_local.loc[r_previous,'maxConfirmedCases'])
                if (verbose):
                    print("{} SIR previous {} {} {} is changing to {} {} {}".format(df_local.loc[r,'DateTime'],S,I,R,
                                                    round(nS),round(nI),round(nR)))
            else:       
                if (verbose):
                    print("{} SIR current {} {} {} no prediction needed".format(df_local.loc[r,'DateTime'],S,I,R))
    return df_local


# In[47]:


# Week 2's dates are
# start_prediction_date = '2020-03-19'
# end_prediction_date   = '2020-04-30'

mdates = pd.date_range('2020-03-19', '2020-04-30', freq='D')
date_list = list(mdates.strftime('%Y-%m-%d'))

date_list2 = ['2020-03-19','2020-03-20','2020-03-21','2020-03-22','2020-03-23','2020-03-24','2020-03-25',             '2020-03-26','2020-03-27',             '2020-03-28','2020-03-29','2020-03-30','2020-03-31','2020-04-01','2020-04-02','2020-04-03','2020-04-04',             '2020-04-05','2020-04-06','2020-04-07','2020-04-08','2020-04-09','2020-04-10','2020-04-11','2020-04-12',             '2020-04-13','2020-04-14','2020-04-15','2020-04-16','2020-04-17','2020-04-18','2020-04-19','2020-04-20',             '2020-04-21','2020-04-22','2020-04-23','2020-04-24','2020-04-24','2020-04-25','2020-04-26','2020-04-27',             '2020-04-28','2020-04-29','2020-04-30'
            ]
    
def copy_preditions_to_final_list(location, df_predictions, df_final_list,verbose=False):
    '''Copy the prediction results into the final predictions DataFrame'''
    
    if (verbose):
        print("Copying predictions for {}:".format(location))
        
    assert(100==df_predictions.shape[0]),'Error 1 - Passed in an incorrect predictions DataFrame'    
    df_small = df_predictions[df_predictions['SimpleName'] == location]
    assert(df_small.shape[0]==df_predictions.shape[0]),'Error 2 - Passed in an incorrect predictions DataFrame'  
    # in theory, the length of df_predictions and the length of df_small should be the same  
    
    for date1 in date_list:

        date1_dt = pd.to_datetime(date1)
        
        # create masks to get the correct items lined up
        mask_location = df_final_list['SimpleName'] == location
        mask_date = df_final_list['DateTime'] == date1_dt        
        idx_final = df_final_list[mask_location & mask_date].index[0]
        
        mask_date2 = df_small['DateTime'] == date1_dt
        idx_small = df_small[mask_date2].index[0]        
        
        if (verbose):
            print('idx: final={}  small={}'.format(idx_final, idx_small))
        df_final_list.at[idx_final,'ConfirmedCases'] = df_small.at[idx_small,'ConfirmedCases']
        df_final_list.at[idx_final,'maxConfirmedCases'] = df_small.at[idx_small,'maxConfirmedCases']
        df_final_list.at[idx_final,'Fatalities']     = df_small.at[idx_small,'Fatalities']
        
        #break


# In[48]:


# for each SimpleName, do an SIR forecast with a fixed set of parameters

    # GLOBAL reminders
    # start_train_date = '2020-01-22'
    # end_train_date   = '2020-03-25'
    # start_prediction_date = '2020-03-19'
    # end_prediction_date   = '2020-04-30'
    
def single_model(location, practice_withhold=0, verbose=False):

    this_name = location
    # practice_withhold 
        #  0 means withhold  0 days of data -- use only for final submission!
        # 13 means withhold 13 days of data

    df_sname = df_train[df_train['SimpleName'] == this_name].copy()
    
    df_sname['maxConfirmedCases'] = 0

    df_sname_template,num_extended = prepare_prediction_template(df_sname,(-1*practice_withhold),verbose=verbose)     

    
    df_sname.head()

    # defaults   beta=0.38  gamma=0.14   ifx=30   rfx=1000 N = 500000
    this_beta, this_gamma, this_ifx, this_rfx, this_pop = db_parameters.loc[this_name]

    df_SIR_predictions = sir_model(df_sname_template,N=this_pop,                                    beta=this_beta,gamma=this_gamma,                                    i_factor = this_ifx, r_factor = this_rfx)

    # Remember that we can ONLY evaluate versus the original training data, so it can only go from start to
    #    end of training data at most.
    if (verbose):
        starter = practice_withhold + num_extended
        print("ends to compare are \n {} \n ***AND*** \n {}".format(df_sname,
                                                                      df_SIR_predictions.iloc[:(-1*num_extended)]))
    rmsle_c, rmsle_f = evaluate_predictions(df_sname,df_SIR_predictions,
                                          practice_withhold,num_extended,
                                          'simple SIR',verbose)
    rmsle_c2, rmsle_f2 = evaluate_predictions2(df_sname,df_SIR_predictions,
                                          practice_withhold,num_extended,
                                          'simple SIR',verbose)
    plot_title = this_name + "  v1 SIR plots"
    #plot_train_predictions(df_SIR_practice,df_sname,plot_title,ylim_factor = 6)

    print("Finished evaluating...{}: {} {}".format(this_name,rmsle_c,rmsle_f))
    print("Finished evaluating 2...{}: {} {}".format(this_name,rmsle_c2,rmsle_f2))
    copy_preditions_to_final_list(this_name, df_SIR_predictions, df_test)
    
    # update dictionary of prediction scores
    prediction_scores[this_name] = [rmsle_c2, rmsle_f2]
    
    return df_SIR_predictions, df_sname, plot_title


# In[49]:


mdf_SIR_practice, mdf_sname, mplot_title = single_model('Hubei__China',practice_withhold=24,verbose=False)


# In[50]:


prediction_scores


# In[51]:


practice_withhold = 7
verbose = True
this_name = 'Hubei__China'
    # practice_withhold 
    #  0 means withhold  0 days of data -- use only for final submission!
    # 13 means withhold 13 days of data

df_sname = df_train[df_train['SimpleName'] == this_name].copy()
    
df_sname['maxConfirmedCases'] = 0

df_sname_template,num_extended = prepare_prediction_template(df_sname,(-1*practice_withhold),verbose=verbose)     


df_sname.head()

# defaults   beta=0.38  gamma=0.14   ifx=30   rfx=1000 N = 500000
this_beta, this_gamma, this_ifx, this_rfx, this_pop = db_parameters.loc[this_name]

df_SIR_predictions = sir_model(df_sname_template,N=this_pop,                                beta=this_beta,gamma=this_gamma,                                i_factor = this_ifx, r_factor = this_rfx)


# In[52]:


df_sname


# In[53]:


df_sname_template


# In[54]:


df_SIR_predictions


# In[55]:


rmsle_c, rmsle_f = evaluate_predictions(df_sname,df_SIR_predictions,
                                      practice_withhold,num_extended,
                                      'simple SIR',verbose)
rmsle_c2, rmsle_f2 = evaluate_predictions2(df_sname,df_SIR_predictions,
                                      practice_withhold,num_extended,
                                      'simple SIR',verbose)
plot_title = this_name + "  v1 SIR plots"
#plot_train_predictions(df_SIR_practice,df_sname,plot_title,ylim_factor = 6)

print("Finished evaluating...{}: {} {}".format(this_name,rmsle_c,rmsle_f))
print("Finished evaluating 2...{}: {} {}".format(this_name,rmsle_c2,rmsle_f2))


# In[56]:


##### END single country investigation ###################


# In[57]:


df_sname = df_train[df_train['SimpleName'] == 'not_provided__Germany']


# In[58]:


df_sname_template_1,num_extended = prepare_prediction_template(df_sname,-5)


# In[59]:


num_extended


# In[60]:


# df_sname_template_1.tail(n=40)


# In[61]:


df_SIR_practice = sir_model(df_sname_template_1) # use all defaults


# In[62]:


# df_SIR_practice.tail(n=40)


# In[63]:


# df_sname.tail(n=40)


# In[ ]:





# In[64]:



# mdf_SIR_practice, mdf_sname, mplot_title = single_model('Alberta__Canada',practice_withhold=24,verbose=True)
# mdf_SIR_practice, mdf_sname, mplot_title = single_model('not_provided__Italy',practice_withhold=24,verbose=True)
mdf_SIR_practice, mdf_sname, mplot_title = single_model('Hubei__China',practice_withhold=24,verbose=False)


# In[65]:


mdf_sname.head()


# In[66]:


plot_train_predictions(mdf_SIR_practice,mdf_sname,mplot_title,ylim_factor = 4)


# In[67]:


selected_country_region = "US"
selected_province_state = "Washington"     # can also use GLOBAL_FILL_STRING

sname = selected_province_state + '__' + selected_country_region
print("Sanity check for {}".format(sname))


# In[68]:


mdf_SIR_practice, mdf_sname, mplot_title = single_model(sname,7)


# In[69]:


plot_train_predictions(mdf_SIR_practice,mdf_sname,mplot_title,ylim_factor = 2)


# In[70]:


db_parameters.loc['Isle of Man__United Kingdom']


# In[71]:


mdf_SIR_practice, mdf_sname, mplot_title = single_model('Isle of Man__United Kingdom',7)
plot_train_predictions(mdf_SIR_practice,mdf_sname,mplot_title,ylim_factor = 2)


# In[72]:


mdf_sname.tail(n=15)


# In[ ]:





# In[ ]:





# In[ ]:





# In[73]:


sname_set
mini_set = ['not_provided__Togo',
 'not_provided__Cambodia',
 'Connecticut__US',
 'Jilin__China',
 'Greenland__Denmark',
 'Maine__US',
 'Pennsylvania__US',
 'Quebec__Canada',
 'New Jersey__US',
 'Manitoba__Canada',
 'not_provided__Haiti']


# In[74]:


# full set runs about 3 minutes
# start with clean copies
df_train = df_train_preserve.copy()
df_test = df_test_preserve.copy()


# In[75]:


# Run all models 
for sname in sname_set:
    mdf_SIR_practice, mdf_sname, mplot_title = single_model(sname,0)   # omit last 7 days to check score


# In[76]:


# Sanity check that all models are complete
last_day = df_test[df_test['DateTime'] == pd.to_datetime('2020-04-30')]
print("Final prediction for 2020-04-30 has {} ConfirmedCases and {} Fatalities".format(last_day['ConfirmedCases'].sum(),                                                                                          last_day['Fatalities'].sum()))


# In[77]:


last_day


# In[78]:


prediction_scores


# In[79]:


df_scores = pd.DataFrame.from_dict(prediction_scores,orient='index', columns=['cc_score','f_score'])


# In[80]:


df_scores['cc_score'].idxmax()


# In[81]:


df_scores['f_score'].idxmax()


# In[82]:


df_test.shape

# sanity check with plots
df_cc = df_test.copy()
df_cc = df_cc.drop(['ForecastId','Country_Region','Province_State','Date','Fatalities'],axis=1)
df_f  = df_test.copy()
df_f = df_f.drop(['ForecastId','Country_Region','Province_State','Date','ConfirmedCases'],axis=1)

df_ccp = df_cc.pivot(index='SimpleName',columns='DateTime')
df_ccp.iloc[6].plot()
# In[83]:


df_test.head(n=3)


# In[84]:


# write out my results so that I can find issues
# df_test.to_csv('df_test_results2.csv')


# In[85]:


df_sub = df_test.drop(['Province_State','SimpleName','Country_Region','Date','DateTime'],axis=1)


# In[86]:


# final dates 3/19/2020 to 4/30/2020


# In[87]:


df_sub.head()


# In[88]:


df_sub_mod = df_sub.drop('ConfirmedCases', axis=1)


# In[89]:


df_sub_mod.head()


# In[90]:


df_sub_mod.columns = ['ForecastId','Fatalities','ConfirmedCases']
df_sub_mod.head()


# In[91]:


df_sub = df_sub_mod


# In[92]:


df_sub.tail()


# In[93]:


df_sub.to_csv('submission.csv',index=False)

## turned out to have 12642 index lines, which is correct


# In[ ]:




