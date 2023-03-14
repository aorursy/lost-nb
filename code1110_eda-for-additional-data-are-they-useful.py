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




# libraries
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy import stats
import gc
import warnings
warnings.filterwarnings("ignore")

print("Libraries were loaded.")




# load data
train_df = pd.read_csv('../input/train.csv')
potential_energy_df = pd.read_csv('../input/potential_energy.csv')
mulliken_charges_df = pd.read_csv('../input/mulliken_charges.csv')
scalar_coupling_contributions_df = pd.read_csv('../input/scalar_coupling_contributions.csv')
magnetic_shielding_tensors_df = pd.read_csv('../input/magnetic_shielding_tensors.csv')
dipole_moments_df = pd.read_csv('../input/dipole_moments.csv')
structure_df = pd.read_csv('../input/structures.csv')
test_df = pd.read_csv('../input/test.csv')


print("All the data were loaded.")




# What are inside those files?
dfs = [train_df, potential_energy_df, mulliken_charges_df, 
       scalar_coupling_contributions_df, magnetic_shielding_tensors_df, 
       dipole_moments_df, structure_df, test_df]
names = ["train_df", "potential_energy_df", "mulliken_charges_df", 
       "scalar_coupling_contributions_df", "magnetic_shielding_tensors_df", 
       "dipole_moments_df", "structure_df", "test_df"]

# display info about a DataFrame
def dispDF(df, name):
    print("========== " + name + " ==========")
    print("SHAPE ----------------------")
    print(df.shape)
    print('')
    print("HEAD ----------------------")
    print(df.head(5))
    print('')
    print("DATA TYPE ----------------")
    print(df.dtypes)
    print('')
    print("UNIQUES -------------------")
    print(df.nunique())
    print('')
    print("======================================")

pd.set_option('display.expand_frame_repr', False)
for df, name in zip(dfs, names):
    dispDF(df, name)




# colors
colors = sns.color_palette("cubehelix", 8)
sns.set()

subsample = 100




# display info about the data frame
dispDF(dipole_moments_df, "dipole moments")




fig = plt.figure()
ax = Axes3D(fig)
# ax = fig.add_subplot(111, projection='3d')

scatter_colors = sns.color_palette("husl", 85003)

# 3D scatter
ax.scatter(dipole_moments_df['X'][::subsample], dipole_moments_df['Y'][::subsample],
           dipole_moments_df['Z'][::subsample], s=30, alpha=0.5, c=scatter_colors[::subsample])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Dipole Moment')




dipole_moments_df['dm_distance'] = np.asarray([x**2 + y**2 + z**2 for x, y, z in zip(dipole_moments_df['X'],dipole_moments_df['Y'], dipole_moments_df['Z'])])

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax = ax.flatten()

# original distribution
sns.distplot(dipole_moments_df['dm_distance'], color=colors[0], kde=False, norm_hist=False, ax=ax[0])
ax[0].set_xlabel('distance')
ax[0].set_title('dipole moment')

# in log
sns.distplot(np.log(dipole_moments_df['dm_distance'] + 0.00001), color=colors[0], kde=False, norm_hist=False, ax=ax[1])
ax[1].set_xlabel('log distance')




dipole_moments_df['dm_outliers'] = np.zeros(dipole_moments_df.shape[0]).astype(int)
dipole_moments_df.loc[dipole_moments_df['dm_distance'] > 100, 'dm_outliers'] = int(1)
print("outliers (dipole moments): " + str(np.sum(dipole_moments_df['dm_outliers'] == 1)) + " molecules")
dipole_moments_df.head(7)




# display info about the data frame
dispDF(magnetic_shielding_tensors_df, "magnetic shielding tensors")




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter_colors = sns.color_palette("husl", 29)

# 3D scatter
for i in range(29):
    xx = magnetic_shielding_tensors_df.loc[magnetic_shielding_tensors_df['atom_index']==i, 'XX']
    yy = magnetic_shielding_tensors_df.loc[magnetic_shielding_tensors_df['atom_index']==i, 'YY']
    zz = magnetic_shielding_tensors_df.loc[magnetic_shielding_tensors_df['atom_index']==i, 'ZZ']
    ax.scatter(xx[::subsample*10], yy[::subsample*10], zz[::subsample*10], s=30, alpha=0.5, c=scatter_colors[i])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Magnetic shielding tensors')




# display info about the data frame
dispDF(potential_energy_df, "potential energy")




# potential energy
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# histogram
sns.distplot(potential_energy_df['potential_energy'], 
             kde=False, color=colors[2], ax=ax)

# median
pe_median = potential_energy_df['potential_energy'].median()
ax.axvline(pe_median, color='r', linestyle='--', lw=4)
ax.text(pe_median + 30, 15000, 'median = ' + str(pe_median), fontsize=12, color='r') 




# median split
highPE_molecules = potential_energy_df.loc[potential_energy_df['potential_energy'] >= pe_median]
lowPE_molecules = potential_energy_df.loc[potential_energy_df['potential_energy'] < pe_median]




print(str(highPE_molecules.shape[0]) + " high potential energy molecules:")
highPE_molecules.head(7)




print(str(lowPE_molecules.shape[0]) + " low potential energy molecules:")
lowPE_molecules.head(7)




# low (0) and high (0) potential energy
potential_energy_df['energy_class'] = np.zeros(potential_energy_df.shape[0]).astype(int)
potential_energy_df.loc[potential_energy_df['potential_energy'] >= pe_median, 'energy_class'] = int(1)
potential_energy_df.head(7)




# display info about the data frame
dispDF(mulliken_charges_df, "mulliken charges")




# distribution of mulliken_charge
mulliken_charges_df["mulliken_charge"].hist()




# display info about the data frame
dispDF(structure_df, "structure")




# # Let's visualize one molecule anyway
# !pip install ase




# import ase
# from ase import Atoms
# import ase.visualize

# positions = structure_df.loc[structure_df['molecule_name'] == 'dsgdb9nsd_000001', ['x', 'y', 'z']]
# symbols = structure_df.loc[structure_df['molecule_name'] == 'dsgdb9nsd_000001', 'atom']
# ase.visualize.view(Atoms(positions=positions, symbols=symbols), viewer="x3d")




# add electronegativity to df
structure_df['electronegativity'] = structure_df['atom']
structure_df['electronegativity'] = structure_df['electronegativity'].map({'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98})
structure_df.head(12)




# display info about the data frame
dispDF(scalar_coupling_contributions_df, "scalar coupling contributions")




fig, ax = plt.subplots(2,1, figsize=(12, 8))
ax = ax.flatten()

# proportions of fc, sd, pso, dso in scaler-coupling constant
fc = scalar_coupling_contributions_df['fc']
sd = scalar_coupling_contributions_df['sd']
pso = scalar_coupling_contributions_df['pso']
dso = scalar_coupling_contributions_df['dso']

columns = ['fc', 'sd', 'pso', 'dso']
contribution_colors = sns.color_palette("Set2", 4)
nrows = scalar_coupling_contributions_df.shape[0]

for i, c in enumerate(columns):
    contributions = 100 * scalar_coupling_contributions_df[c].abs()                 / (np.abs(fc) + np.abs(sd) + np.abs(pso) + np.abs(dso))
    ax[0].plot(np.arange(0, nrows, subsample*100), contributions[::subsample*100], c=contribution_colors[i], label=c)
    
ax[0].set_xlabel('molecule')
ax[0].set_ylabel('% to scaler coupling constant')
ax[0].legend()

# unique counts of molecular type
counts = np.zeros(scalar_coupling_contributions_df['type'].nunique())
for i, u in enumerate(scalar_coupling_contributions_df['type'].unique()):
    counts[i] = np.sum(scalar_coupling_contributions_df['type'].values == u)
    
sns.barplot(x=scalar_coupling_contributions_df['type'].unique(), y=counts, ax=ax[1])




# Values of each interaction vs molecular types  
fig, ax = plt.subplots(4, 1, figsize=(20, 14)) 
ax = ax.flatten()
for i, col in enumerate(columns): 
    means = scalar_coupling_contributions_df[["type", col]].groupby(by='type')[col].mean()
    SDs = scalar_coupling_contributions_df[["type", col]].groupby(by='type')[col].std() 
    means.plot(kind="bar", yerr=SDs, ax=ax[i])
    ax[i].set_ylabel(col)
    
plt.tight_layout()




# combine "dipole_moments_df" and "potential_energy_df" (The both have 85003 rows, information per molecule)
DM_PE_df = pd.merge(dipole_moments_df, potential_energy_df, on='molecule_name')

del dipole_moments_df, potential_energy_df, train_df, test_df
gc.collect()

print("There are {} rows and {} columns in DM_PE_df.".format(DM_PE_df.shape[0], DM_PE_df.shape[1]))
DM_PE_df.head(12)




# combine "magnetic_shielding_tensors_df" and "mulliken_charges_df" (The both have 1533537 rows, information per atom in a molecule)
MST_MC_df = pd.merge(magnetic_shielding_tensors_df, mulliken_charges_df, on=['molecule_name', 'atom_index'])

del magnetic_shielding_tensors_df, mulliken_charges_df
gc.collect()

print("There are {} rows and {} columns in DM_PE_df.".format(MST_MC_df.shape[0], MST_MC_df.shape[1]))
MST_MC_df.head(12)




# combine these two
MST_MC_DM_PE_df = pd.merge(MST_MC_df, DM_PE_df, on='molecule_name', how='left')

del MST_MC_df, DM_PE_df
gc.collect()

print("There are {} rows and {} columns in DM_PE_df.".format(MST_MC_DM_PE_df.shape[0], MST_MC_DM_PE_df.shape[1]))
MST_MC_DM_PE_df.head(12)




# lighter structure
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max and c_prec == np.finfo(np.float16).precision:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df




MST_MC_DM_PE_df = reduce_mem_usage(MST_MC_DM_PE_df)
scalar_coupling_contributions_df = reduce_mem_usage(scalar_coupling_contributions_df)




# combine it with "scaler_coupling_contributions_df" (information per a pair of atoms in a molecule)
combined_df0 = pd.merge(scalar_coupling_contributions_df, MST_MC_DM_PE_df, 
                           left_on=['molecule_name','atom_index_0'], right_on=['molecule_name','atom_index'], how='left')
print("There are {} rows and {} columns in combined_df0.".format(combined_df0.shape[0], combined_df0.shape[1]))
combined_df0.head(12)




# # combine it with "scaler_coupling_contributions_df" (information per a pair of atoms in a molecule)
# combined_df1 = pd.merge(scalar_coupling_contributions_df, MST_MC_DM_PE_df, 
#                            left_on=['molecule_name','atom_index_1'], right_on=['molecule_name','atom_index'], how='left')

# del scalar_coupling_contributions_df, MST_MC_DM_PE_df
# gc.collect()

# print("There are {} rows and {} columns in combined_df0.".format(combined_df1.shape[0], combined_df1.shape[1]))
# combined_df1.head(12)




# # combine these two
# combined_df = pd.merge(combined_df0, combined_df1, on=['molecule_name'])

# del combined_df0, combined_df1
# gc.collect()

# print("There are {} rows and {} columns in combined_df.".format(combined_df.shape[0], combined_df.shape[1]))
# combined_df.head(12)

