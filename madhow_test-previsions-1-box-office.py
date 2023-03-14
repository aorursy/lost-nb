#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




from pathlib import Path

import pandas as pd




# Création d'un Path (les virgules remplacent en quelques sortes les / ou \ selon les Systèmes d'exploitation)
INPUT_PATH = Path('..', 'input')
get_ipython().system('ls {INPUT_PATH}')
TRAIN_PATH = Path(INPUT_PATH, 'train.csv')




# Lire le fichier train.csv dans la variable df
df = pd.read_csv(TRAIN_PATH, index_col=0)
df.head() # head() permet d'afficher juste un aperçu du résultatTRAIN_PATH = Path(INPUT_PATH, 'train.csv')




# Permet d'analyser des données numériques
# SKLEARN est un framework pour du Machine Learning
# RandomForestRegressor est un algorithme de ML qu'on entrainera pour obtenir des prédictions
from sklearn.ensemble import RandomForestRegressor




# Préparer un tableau ne contenant que les Colonnes de Types Numériques
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_features = df.select_dtypes(include=numerics)
numeric_features




# On repertorie toutes les lignes dont le Runtime est NULL
null_indexes = numeric_features['runtime'].isnull()
null_indexes




# Tableau [budget, popularity, runtime]
X = numeric_features.loc[ # requêter le tableau
    ~null_indexes # '~' équivaut à 'not Equal' > on retire donc les lignes dont le Runtime est Null 
]
#.drop('revenue', axis=1)
# retirer la colonne revenue

# Tableau [revenue]
y = df.loc[
    ~null_indexes, # condition : pas les lignes dont le runtime est null
    'revenue' # afficher : seulement les 'revenues'
]

# Affichage des nombres de lignes et colonnes pour X & y
X.shape, y.shape # il s'agit d'un tuple : (qu'une colonne) 




# Instanciation de la Classe
rfr = RandomForestRegressor()

get_ipython().run_line_magic('pinfo', 'rfr')
get_ipython().run_line_magic('pinfo2', 'rfr')




rfr.fit(X, y) # si l'on ajoute un ';' il n'affiche pas le résultat de la console




# 'y_hat' est le tableau contenant les prédictions des revenues
y_hat = rfr.predict(X)
y_hat




y.head()




import matplotlib.pyplot as plt # Librairies pour afficher des Graphes
get_ipython().run_line_magic('matplotlib', 'inline # config : pour afficher les Graphes quelque soit le Système')




# Afficher le Graphe
plt.scatter(y, y_hat, c='r')
plt.xlabel("True values")
plt.ylabel("Predictions");




# Vérifier si 'y' & 'y_hat' sont égaux 
(y == y_hat).all()
# ça n




sub = abs(y - y_hat)
sub




# Tableau [budget, popularity, runtime]
X2 = numeric_features.loc[ # requêter le tableau
    ~null_indexes # '~' équivaut à 'not Equal' > on retire donc les lignes dont le Runtime est Null 
].drop('revenue', axis=1) # retirer la colonne revenue

# Affichage des nombres de lignes et colonnes pour X & y
X2.shape, y.shape # il s'agit d'un tuple : (qu'une colonne) 




rfr2 = RandomForestRegressor()




rfr2.fit(X2, y)




y_hat2 = rfr2.predict(X2)
y_hat2




y.head()




sub = abs(y - y_hat2)
sub.head()




# Afficher le Graphe
plt.scatter(y, y_hat2, c='b')
plt.xlabel("True values")
plt.ylabel("Predictions");




y_hat.mean()




y_hat2.mean()




def mean_absolute_error(y_true, y_predict):
    return abs(y_true - y_predict).mean()




mean_absolute_error(y, y_hat2)




mean_absolute_error(y, y_hat)




from sklearn.metrics import mean_squared_log_error
import numpy as np




def RMSLE(y_true, y_pred) : # ratio d'erreur
    return np.sqrt(mean_squared_log_error(y_true, y_pred))




RMSLE(y, y_hat)
# le ratio d'erreur de la prédiction est très faible, car il connait deja les revenues 




RMSLE(y, y_hat2)
# le ratio d'erreur de la prédiction est réaliste, car il ne connait pas les revenues




# Creer un tableau de la même taille que 'y' mais contenant à chaque fois la "moyenne de y"
y_hat_mean = np.full(y.shape, y.mean())
y_hat_mean




RMSLE(y, y_hat_mean)




def create_dataset(df):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_features = df.select_dtypes(include=numerics)
    
    # On repertorie toutes les lignes dont le Runtime est NULL
    null_indexes = numeric_features['runtime'].isnull()

    # Tableau [budget, popularity, runtime]
    X = numeric_features.loc[ # requêter le tableau
        ~null_indexes # '~' équivaut à 'not Equal' > on retire donc les lignes dont le Runtime est Null 
    ].drop('revenue', axis=1) # retirer la colonne revenue

    # Tableau [revenue]
    y = df.loc[
        ~null_indexes, # condition : pas les lignes dont le runtime est null
        'revenue' # afficher : seulement les 'revenues'
    ]

    # Retourner des nombres de lignes et colonnes pour X & y
    return X, y




df_train = df[:2000] # dataframe d'entrainement
df_test = df[2000:]  # dataframe de test
df_train.shape, df_test.shape 




X_train, y_train = create_dataset(df_train)
X_test, y_test = create_dataset(df_test)
X_train.shape, y_train.shape, X_test.shape, y_test.shape




rfr_new = RandomForestRegressor()
rfr_new.fit(X_train, y_train) # entrainement




y_train_pred = rfr_new.predict(X_train) # prediction
y_train_pred




error_train = RMSLE(y_train, y_train_pred) # 
error_train




y_test_pred = rfr_new.predict(X_test) # prediction de données jamais vues
y_test_pred




error_test = RMSLE(y_test, y_test_pred) # 
error_test




rfr99 = RandomForestRegressor(n_estimators=99)
rfr99




rfr99.fit.pred

