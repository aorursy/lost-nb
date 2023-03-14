#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting (when it can't be done under the pandas hood)

# Input data files are available in the "../input/" directory.
import os
data_directory = '../input/datafiles'

df_seeds = pd.read_csv(os.path.join(data_directory,'NCAATourneySeeds.csv'))
df_seeds.head()


# In[2]:


# Merge the seeds file with itself on Season.  This creates every combination of two teams by season.
df_sub = df_seeds.merge(df_seeds, how='inner', on='Season')

# We want a little less than half the records in this data frame.  
# Every game appears twice, once with the lower id team in the TeamID_x column, and 
# once in the TeamID_y column.  We also have the impossible matchups of teams with themselves.
# To fix this, we keep only the games where the lower team ID is in the TeamID_x columns.
df_sub = df_sub[df_sub['TeamID_x'] < df_sub['TeamID_y']]

df_sub.head()


# In[3]:


df_sub['ID'] = df_sub['Season'].astype(str) + '_'               + df_sub['TeamID_x'].astype(str) + '_'               + df_sub['TeamID_y'].astype(str)

df_sub['SeedInt_x'] = [int(x[1:3]) for x in df_sub['Seed_x']]
df_sub['SeedInt_y'] = [int(x[1:3]) for x in df_sub['Seed_y']]

df_sub['Pred'] = 0.5 + 0.03*(df_sub['SeedInt_y'] - df_sub['SeedInt_x'])

df_sub.head()


# In[4]:


# save out the 2014-2018 predictions for later submission
df_sub.loc[(df_sub['Season'] >= 2014) & (df_sub['Season'] <= 2018), ['ID', 'Pred']].to_csv('./Submission.csv',index=False)

# now pare down existing df_sub
df_sub = df_sub[['ID','Pred']]
df_sub.head()


# In[5]:


df_tc = pd.read_csv(os.path.join(data_directory,'NCAATourneyCompactResults.csv'))
df_tc.head()


# In[6]:


df_tc['ID'] = df_tc['Season'].astype(str) + '_'               + (np.minimum(df_tc['WTeamID'],df_tc['LTeamID'])).astype(str) + '_'               + (np.maximum(df_tc['WTeamID'],df_tc['LTeamID'])).astype(str)

df_tc['Result'] = 1*(df_tc['WTeamID'] < df_tc['LTeamID'])
df_tc.head(10)


# In[7]:


df_tc = df_tc.merge(df_seeds.rename(columns={'Seed':'WSeed','TeamID':'WTeamID'}), 
                    how='inner', on=['Season', 'WTeamID'])
df_tc = df_tc.merge(df_seeds.rename(columns={'Seed':'LSeed','TeamID':'LTeamID'}), 
                    how='inner', on=['Season', 'LTeamID'])
df_tc.head()


# In[8]:


df_playin = df_tc[df_tc['WSeed'].str[0:3] == df_tc['LSeed'].str[0:3]]
df_playin.head()


# In[9]:


df_tc = df_tc[df_tc['WSeed'].str[0:3] != df_tc['LSeed'].str[0:3]]
df_tc.head()


# In[10]:


def kaggle_clip_log(x):
    '''
    Calculates the natural logarithm, but with the argument clipped within [1e-15, 1 - 1e-15]
    '''
    return np.log(np.clip(x,1.0e-15, 1.0 - 1.0e-15))

def kaggle_log_loss(pred, result):
    '''
    Calculates the kaggle log loss for prediction pred given result result
    '''
    return -(result*kaggle_clip_log(pred) + (1-result)*kaggle_clip_log(1.0 - pred))
    
def score_submission(df_sub, df_results, on_season=None, return_df_analysis=True):
    '''
    Scores a submission against relevant tournament results
    
    Parameters
    ==========
    df_sub: Pandas dataframe containing predictions to be scored (must contain a column called 'ID' and 
            a column called 'Pred')
            
    df_results: Pandas dataframe containing results to be compared against (must contain a column 
            called 'ID' and a column called 'Result')
            
    on_season: array-like or None.  If array, should contain the seasons for which a score should
            be calculated.  If None, will use all seasons present in df_results
            
    return_df_analysis: Bool.  If True, will return the dataframe used for calculations.  This is useful
            for future analysis
            
    Returns
    =======
    df_score: pandas dataframe containing the average score over predictions that were scorable per season
           as well as the number of obvious errors encountered
    df_analysis:  pandas dataframe containing information about all results used in scoring
                  Only provided if return_df_analysis=True
    '''
    
    df_analysis = df_results.copy()
    
    # this will overwrite if there's already a season column but it should be the same
    df_analysis['Season'] = [int(x.split('_')[0]) for x in df_results['ID']]
    
    if not on_season is None:
        df_analysis = df_analysis[np.in1d(df_analysis['Season'], on_season)]
        
    # left merge with the submission.  This will keep all games for which there
    # are results regardless of whether there is a prediction
    df_analysis = df_analysis.merge(df_sub, how='left', on='ID')
    
    # check to see if there are obvious errors in the predictions:
    # Obvious errors include predictions that are less than 0, greater than 1, or nan
    # You can add more if you like
    df_analysis['ObviousError'] = 1*((df_analysis['Pred'] < 0.0)                                   | (df_analysis['Pred'] > 1.0)                                   | (df_analysis['Pred'].isnull()))
    
    df_analysis['LogLoss'] = kaggle_log_loss(df_analysis['Pred'], df_analysis['Result'])
    
    df_score = df_analysis.groupby('Season').agg({'LogLoss' : 'mean', 'ObviousError': 'sum'})
    
    if return_df_analysis:
        return df_score, df_analysis
    else:
        return df_score
    


# In[11]:


df_score, df_analysis =     score_submission(df_sub, df_tc, on_season = np.arange(2014,2019), return_df_analysis=True)
df_score


# In[12]:


df_score.mean()


# In[13]:


df_score, df_analysis =     score_submission(df_sub, df_tc, on_season = np.arange(2010,2018), return_df_analysis=True)
df_score


# In[14]:


df_score, df_analysis =     score_submission(df_sub.sample(frac=0.5), df_tc, on_season = np.arange(2010,2018), return_df_analysis=True)
df_score


# In[15]:


df_analysis[df_analysis['ObviousError']==1].head()


# In[16]:


df_score, df_analysis =     score_submission(df_sub, df_tc, on_season = None, return_df_analysis=True)

df_score


# In[17]:


df_score.mean()


# In[18]:


df_score.reset_index().plot('Season','LogLoss')


# In[19]:


df_score.hist('LogLoss',bins='auto')


# In[20]:


df_analysis.hist('LogLoss',bins=10)
df_analysis.sort_values('LogLoss',ascending=False).head(20)


# In[21]:




