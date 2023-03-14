#!/usr/bin/env python
# coding: utf-8



import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression




# Ploting Parameters
FIGSIZE = (10, 6)
sns.set() # Set seaborn style

# Notebook Parameters
FREE_MEMORY = True
SAVE_CSV = True

# kFold Validation Parameters
RANDOM_STATE = 123
N_SPLITS = 3
SHUFFLE = True
VERBOSE = False

# Data Files
OUTPUT_FILE_A = "potential_energy_upd.csv"
OUTPUT_FILE_B = "molecule_train.csv"
OUTPUT_FILE_C = "molecule_test.csv"




# Input data user functions

DATA_PATH = "../input"

def csv_path(dataset="train", data_path=DATA_PATH):
    return "{}/{}.csv".format(data_path, dataset)

def read_data(dataset='train', data_path=DATA_PATH):
    index_col = None
    index_type = ['train', 'test']
    if dataset in index_type:
        index_col = 'id'
    data_path = csv_path(dataset, data_path=data_path)
    return pd.read_csv(data_path, index_col=index_col)




train = read_data("train")
test = read_data("test")




# One row per molecule
molecule_train = pd.DataFrame({"molecule_name" : train["molecule_name"].unique()})
molecule_test = pd.DataFrame({"molecule_name" : test["molecule_name"].unique()})
structures = read_data('structures')
atom_list_df = structures.groupby('molecule_name')['atom'].apply(list)
atom_list_df = atom_list_df.to_frame()




if FREE_MEMORY:
    del train, test




molecule_train = pd.merge(molecule_train, atom_list_df, how='left', on='molecule_name')
molecule_test = pd.merge(molecule_test, atom_list_df, how='left', on='molecule_name')




# Count atoms by type
atoms_list = structures.atom.unique().tolist()
print("Distinct atoms in structures data: \n {}".format(atoms_list))

for atom in atoms_list:
    molecule_train['atom_' + atom] =         molecule_train['atom'].apply(lambda x: x.count(atom))
    molecule_test['atom_' + atom] =         molecule_test['atom'].apply(lambda x: x.count(atom))




potential_energy = read_data("potential_energy")
molecule_train = pd.merge(molecule_train, potential_energy)




if FREE_MEMORY:
    del potential_energy, structures
    del atom_list_df




# 1 atomic mass unit (amu) corresponds to 1.660539040 × 10−24 gram
ATOM_MASS = {
    "H": 1.00784,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9990,
    "F": 18.9984
}

def mol_weight(atom_list):
    """
    Get the molecular weight from a list of atoms
    """
    return sum(map(lambda x: ATOM_MASS[x], atom_list))




molecule_train["mol_weight"] = molecule_train.atom.apply(lambda x: mol_weight(x))
molecule_test["mol_weight"] = molecule_train.atom.apply(lambda x: mol_weight(x))




molecule_train.head()




id_feature = 'molecule_name'
target_feature = (set(molecule_train) - set(molecule_test)).pop()
selected_features = list(molecule_test)
selected_features.remove(id_feature)
selected_features.remove('atom')
print("Selected Features: \t{}".format(selected_features))
print("Target Feature: \t{}".format(target_feature))
print("Id Feature: \t\t{}".format(id_feature))




X = molecule_train[selected_features]
y = molecule_train[target_feature]




kfold = KFold(n_splits=N_SPLITS,
              random_state=RANDOM_STATE,
              shuffle=SHUFFLE)




fold = 0
r2_scores = []
mse_scores = []
lin_reg = LinearRegression()

for in_index, oof_index in kfold.split(X, y):
    fold += 1
    print("- Training Fold: ({}/{})".format(fold, N_SPLITS))
    X_in, X_oof = X.loc[in_index], X.loc[oof_index]
    y_in, y_oof = y.loc[in_index], y.loc[oof_index]
    
    lin_reg.fit(X_in, y_in)
    y_pred = lin_reg.predict(X_oof)
    r2 = r2_score(y_oof, y_pred)
    r2_scores.append(r2)
    mse_score = mean_squared_error(y_oof, y_pred)
    mse_scores.append(mse_score) 

print('\nkFold Validation Results:')
print(' * Average Variance Score (R2): \t{:.4f}'.format(np.mean(r2_scores)))
print(' * Average Mean squared error (MSE): \t{:.4f}'.format(np.mean(mse_score)))
    




plt.figure(figsize=FIGSIZE)
plt.plot(y_oof, y_pred)
plt.title("Fold {} Prediction".format(fold))
plt.xlabel("Validation Potential Energy")
plt.ylabel("Predicted Potential Energy")
plt.show()




lin_reg.fit(X, y)
y_test = lin_reg.predict(molecule_test[selected_features])
molecule_test[target_feature] = y_test
if FREE_MEMORY:
    del X, y




potential_energy_upd = pd.concat([molecule_train[[id_feature, target_feature]],
                                  molecule_test[[id_feature, target_feature]]],
                                 ignore_index=True)




plt.figure(figsize=FIGSIZE)
molecule_train['potential_energy'].plot.kde(figsize=FIGSIZE, legend=True, label="train")
molecule_test['potential_energy'].plot.kde(figsize=FIGSIZE, legend=True, label="test")
plt.title("Predicted and Training Potential Energy Density Plot")
plt.xlabel("Potential Energy")
plt.show()




potential_energy_upd = potential_energy_upd.sort_values(id_feature)
potential_energy_upd.reset_index(drop=True, inplace=True)




potential_energy_upd.head()




def save_csv(df, file_name):
    """
    Check and save csv data to kernel main folder 
    """
    assert "csv" in file_name, "Bad file extension"
    assert df.notnull().values.any(), "NaN values in data frame"
    print(f"Saving file {file_name}...")
    df.to_csv(file_name, index=False)

if SAVE_CSV:
    save_csv(potential_energy_upd, OUTPUT_FILE_A)
    save_csv(molecule_train, OUTPUT_FILE_B)
    save_csv(molecule_test, OUTPUT_FILE_C)

