#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn import datasets, metrics, tree
from sklearn.ensemble import  RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Read data


# In[ ]:


#


# In[ ]:


#


# In[ ]:


# individuiamo la riga in  base al valore di ID


# In[ ]:



#


# In[ ]:








# In[ ]:





# In[ ]:


clf = 



# In[ ]:


# calcolo dell'indice AUC e dell'errore di classificazione sul training e sul test
from sklearn.metrics import roc_auc_score


# In[ ]:


# Calcolo matrice di confusione e gli altri indici sul validation


# In[ ]:


# costruiamo la matrice di confusione con una soglia diversa da 0.5




# In[ ]:


Y_test = clf.predict_proba(X_test)
soluz = pd.DataFrame({'ID':XT[:,0].astype(int),'TARGET':Y_test[:,1]})
soluz.to_csv('fine.csv', sep=",",index=False)

