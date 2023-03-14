#!/usr/bin/env python
# coding: utf-8

# In[ ]:



class Ctg2Num (object):
    def __init__(self, Input_data):   # Initializing the class
        self.Input_data = Input_data
    def Low_Freq_Merger(self, ColNames, prcnt1 = 25):  # Merges the values with low frequencies and replaces them with a new value
        for cols in ColNames:
            value_counts_Col = self.Input_data[cols].value_counts() 
            Rplc_Indx = value_counts_Col[value_counts_Col <= np.percentile(value_counts_Col, prcnt1)].index
            self.Input_data[cols].replace(Rplc_Indx, 'Type 0', inplace = True)
        return (self.Input_data)
    
    def Dummy_Generator(self, ColNames, dummy_na = False): # Creates dummy variables and removes the original categorical variables
        if dummy_na = False:
            Dummies = pd.get_dummies(self.Input_data[ColNames], drop_first = True)
        else:
            Dummies = pd.get_dummies(self.Input_data[ColNames], drop_first = True, dummy_na = True)
        self.Input_data = self.Input_data.join(Dummies)
        self.Input_data = self.Input_data.drop(ColNames, axis=1) 
        return (self.Input_data)

    def Freq_Replacor(self, ColNames): # Replaces variables with many values with their frequencies and removes the original variables
        for cols in ColNames:
            Freq_Ctg = self.Input_data[cols].value_counts().reset_index(name='count').rename(columns={'index': cols})
            self.Input_data = pd.merge(self.Input_data, Freq_Ctg, on=[cols], how='left')
            self.Input_data = self.Input_data.drop(cols, axis=1)
        return (self.Input_data)
    


# In[ ]:


people_data = pd.read_csv('../input/act_test.csv')
# Handeling categorical features by converting them to binary dummy variables
# For Char_3, char_4, the values with low frequencies are merged
Ctg_People_Obj = Ctg2Num(people_data)
Ctg_People_Obj.Low_Freq_Merger(['char_3','char_4'])
    
# Generating Dummy variables and merging to the existing dataset
Ctg_Ind_dum = ['char_1','char_2','char_3','char_4','char_5','char_6','char_7','char_8','char_9']
Ctg_People_Obj.Dummy_Generator(Ctg_Ind_dum)
    
# Replacing variable group_1 by the frequency of the groups
people_data_Allnum = Ctg_People_Obj.Freq_Replacor(['group_1'])
    

