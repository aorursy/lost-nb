#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random 
import sympy

import os
print(os.listdir("../input"))




df_cities = pd.read_csv('../input/cities.csv')
df_cities.head()




plt.figure(figsize=(16,10))
plt.subplot(111, adjustable='box', aspect=1.0)
plt.plot(df_cities.X, df_cities.Y, 'k,', alpha=0.3)
plt.plot(df_cities.X[0], df_cities.Y[0], 'bX')
plt.xlim(0, 5100)
plt.ylim(0, 3400)

plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('All cities (North Pole = Blue X)', fontsize=18)
plt.show()




from sympy import isprime
df_cities['isPrime'] = df_cities.CityId.apply(isprime)
prime_cities = df_cities.loc[df_cities.isPrime]


plt.figure(figsize=(16,10))
plt.subplot(111, adjustable='box', aspect=1.0)
plt.plot(df_cities.X, df_cities.Y, 'k,', alpha=0.3)
plt.plot(prime_cities.X, prime_cities.Y, 'r.', markersize=4, alpha=0.3)
plt.plot(df_cities.X[0], df_cities.Y[0], 'bX')
plt.xlim(0, 5100)
plt.ylim(0, 3400)
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('All cities (Primes = Red Dots, North Pole = Blue X)', fontsize=18)
plt.show()




# seive of eratosthenes

def seive_of_eratosthenes(n):
    prime_list = [True] * (n+1)
    prime_list[0] = False
    prime_list[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if prime_list[i]:
            k = 2
            while i*k <= n:
                prime_list[i*k] = False
                k += 1
    return (prime_list)

prime_cities = seive_of_eratosthenes(max(df_cities.CityId))     
print(len(prime_cities))




# path : list of city_id
def total_distance(dfcity, path):
    prev_city = path[0]
    total_distance = 0
    step_num = 1
    for city_num in path[1:]:
        total_distance += np.sqrt(pow(dfcity.X[city_num] - dfcity.X[prev_city],2) +                    pow(dfcity.Y[city_num]-dfcity.Y[prev_city],2)) *                     (1+ 0.1*((step_num% 10 == 0) *int(not(prime_cities[prev_city]))))
        prev_city = city_num
        step_num = step_num + 1
    return total_distance
        
    
    
# appending last element as 0 ie 197769th element
basic_path = list(df_cities.CityId[:].append(pd.Series([0])))
print(f"Total distance with the dumbest path is {total_distance(df_cities, basic_path)}")




df_path = pd.merge_ordered(pd.DataFrame({'CityId':basic_path}),df_cities,on=['CityId'])
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(df_path.iloc[0:100,]['X'], df_path.iloc[0:100,]['Y'],marker = 'o')
for i, txt in enumerate(df_path.iloc[0:100,]['CityId']):
    ax.annotate(txt, (df_path.iloc[0:100,]['X'][i], df_path.iloc[0:100,]['Y'][i]),size = 10)




df_cities['Ycuts'] = pd.cut(df_cities.Y,300)
df_cities['Xcuts'] = pd.cut(df_cities.X,300)
grid_sorted_cities = list(df_cities.iloc[1:].sort_values(['Xcuts','Ycuts','X','Y'])['CityId'])

grid_sorted_cities = [0] + grid_sorted_cities + [0]

print('Total distance with the sorted cities with a grid path is '+ "{:,}".format(total_distance(df_cities,grid_sorted_cities)))




# creating a new dataframe based on the sorted cityId
df_path = pd.DataFrame({'CityId':grid_sorted_cities}).merge(df_cities,how = 'left')
# df_path.head()

# plotting 1000 steps of our path
fig, ax = plt.subplots(figsize=(20,20))
ax.plot(df_path.iloc[0:1000,]['X'], df_path.iloc[0:1000,]['Y'],marker = 'o')




zigzag_sorted_cities1 = list(df_cities.iloc[1:].sort_values(['Xcuts','Ycuts','X','Y'])['CityId'])
zigzag_sorted_cities2 = list(df_cities.iloc[1:].sort_values(['Xcuts','Ycuts','X','Y'], ascending = [True,False,True,True])['CityId'])
chooser_pattern = list(df_cities.iloc[1:].sort_values(['Xcuts']).groupby(['Xcuts']).ngroup()%2)

zigzag_cities = [zigzag_sorted_cities1[i] if chooser_pattern[i] == 0 else zigzag_sorted_cities2[i] for i in range(len(chooser_pattern))]
zigzag_cities =  [0] + zigzag_cities + [0]
print('Total distance with the Zig-Zag with grid city path is '+ "{:,}".format(total_distance(df_cities,zigzag_cities)))




# creating a new dataframe based on the sorted cityId
df_path = pd.DataFrame({'CityId':zigzag_cities}).merge(df_cities,how = 'left')

# plotting 1000 steps of our path
fig, ax = plt.subplots(figsize=(20,20))
ax.plot(df_path.iloc[0:1000,]['X'], df_path.iloc[0:1000,]['Y'],marker = 'o')




for i in range(100,700,100):
    for j in range(100,700,100):
        df_cities['Ycuts'] = pd.cut(df_cities.Y,j)
        df_cities['Xcuts'] = pd.cut(df_cities.X,i)
        zigzag_sorted_cities1 = list(df_cities.iloc[1:].sort_values(['Xcuts','Ycuts','X','Y'])['CityId'])
        zigzag_sorted_cities2 = list(df_cities.iloc[1:].sort_values(['Xcuts','Ycuts','X','Y'], ascending = [True,False,True,True])['CityId'])
        chooser_pattern = list(df_cities.iloc[1:].sort_values(['Xcuts']).groupby(['Xcuts']).ngroup()%2)

        zigzag_cities = [zigzag_sorted_cities1[i] if chooser_pattern[i] == 0 else zigzag_sorted_cities2[i] for i in range(len(chooser_pattern))]
        zigzag_cities =  [0] + zigzag_cities + [0]
        print('Total distance with the Zig-Zag with grid city path ith grid size ('+ str(i) +','+ str(j) +')'+ "{:,}".format(total_distance(df_cities,zigzag_cities)))




df_path = pd.DataFrame({'CityId':zigzag_cities}).merge(df_cities,how = 'left')
fig, ax = plt.subplots(figsize=(20,20))
ax.plot(df_path['X'], df_path['Y'])




def nearest_neighbour():
    cities = pd.read_csv("../input/cities.csv")
    ids = cities.CityId.values[1:]
    xy = np.array([cities.X.values, cities.Y.values]).T[1:]   #T is the transpose of the matrix
    '''
    with transpose
    array([[4377.40597217,  336.60208217],
           [3454.15819771, 2820.05301125],
           [4688.09929763, 2935.89805581],
       
    without transpose
    array([[4377.40597217, 3454.15819771,4688.09929763],
           [336.60208217, 2820.05301125, 2935.89805581]])
    '''
    path = [0,]
    while len(ids) > 0:
        last_x, last_y = cities.X[path[-1]], cities.Y[path[-1]]
        dist = ((xy - np.array([last_x, last_y]))**2).sum(-1)
        nearest_index = dist.argmin()
        path.append(ids[nearest_index])
        ids = np.delete(ids, nearest_index, axis=0)
        xy = np.delete(xy, nearest_index, axis=0)
    path.append(0)
    return path

nnpath = nearest_neighbour()
print('Total distance with the Nearest Neighbor path '+  "is {:,}".format(total_distance(df_cities,nnpath)))




df_path = pd.DataFrame({'CityId':nnpath}).merge(df_cities,how = 'left')
fig, ax = plt.subplots(figsize=(20,20))
ax.plot(df_path['X'], df_path['Y'])




nnpath_with_primes = nnpath.copy()
for index in range(20,len(nnpath_with_primes)-30):
    city = nnpath_with_primes[index]
    if (prime_cities[city] &  ((index+1) % 10 != 0)):        
        for i in range(-1,3):
            tmp_path = nnpath_with_primes.copy()
            swap_index = (int((index+1)/10) + i)*10 - 1
            tmp_path[swap_index],tmp_path[index] = tmp_path[index],tmp_path[swap_index]
            if total_distance(df_cities,tmp_path[min(swap_index,index) - 1 : max(swap_index,index) + 2]) < total_distance(df_cities,nnpath_with_primes[min(swap_index,index) - 1 : max(swap_index,index) + 2]):
                nnpath_with_primes = tmp_path.copy() 
                break
print('Total distance with the Nearest Neighbor With Prime Swaps '+  "is {:,}".format(total_distance(df_cities,nnpath_with_primes)))




pd.DataFrame({'Path':nnpath_with_primes}).to_csv('nnpath_with_primes.csv',index  = False)




from sympy import isprime, primerange
from matplotlib import collections as mc
from concorde.tsp import TSPSolver
import time




def make_submission(name, path):
    assert path[0] == path[-1] == 0
    assert len(set(path)) == len(path) - 1 == 197769
    pd.DataFrame({'Path': path}).to_csv(f'{name}.csv', index=False)

    
def score_path(path):
    cities = pd.read_csv('../input/cities.csv', index_col=['CityId'])
    pnums = [i for i in primerange(0, 197770)]
    path_df = cities.reindex(path).reset_index()
    
    path_df['step'] = np.sqrt((path_df.X - path_df.X.shift())**2 + 
                              (path_df.Y - path_df.Y.shift())**2)
    path_df['step_adj'] = np.where((path_df.index) % 10 != 0,
                                   path_df.step,
                                   path_df.step + 
                                   path_df.step*0.1*(~path_df.CityId.shift().isin(pnums)))
    return path_df.step_adj.sum()




def concorde_tsp(seed=42):
    cities = pd.read_csv('../input/cities.csv')
    solver = TSPSolver.from_data(cities.X, cities.Y, norm="EUC_2D")
    tour_data = solver.solve(time_bound=60.0, verbose=True, random_seed=seed)
    if tour_data.found_tour:
        path = np.append(tour_data.tour,[0])
        make_submission('concorde', path)
        return path
    else:
        return None

path_cc = concorde_tsp()




path_cc = path_cc.tolist()
print('Total distance with the Nearest Neighbor With Prime Swaps '+  "is {:,}".format(total_distance(df_cities,path_cc)))




df_path = pd.DataFrame({'CityId':path_cc}).merge(df_cities,how = 'left')
fig, ax = plt.subplots(figsize=(20,20))
ax.plot(df_path['X'], df_path['Y'])






