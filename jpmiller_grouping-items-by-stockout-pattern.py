#!/usr/bin/env python
# coding: utf-8



import numpy as np
np.set_printoptions(precision=2, suppress=True)
import pandas as pd
pd.options.display.max_columns = 2000

id_cols = ['store_id', 'item_id']
sales_cols = ['d_'+str(i) for i in range(1,1914)]

sales = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv',
                    usecols=id_cols+sales_cols, index_col=id_cols) \
            .astype(np.uint16).sort_index()
sales




# Filter items at one store 
sales_ca1 = sales.loc['CA_1']                  .assign(d_median=lambda x: x.median(axis=1))                  .query('d_median >= 4')                  .drop(columns='d_median')                  .iloc[:, -28*24:]  # use the last two years

sales_ca1




# Check dept counts
sales_ca1.groupby(sales_ca1.index.str[:-6]).size()




import missingno as msno
msno.dendrogram(sales_ca1.replace(0, np.nan).T, method='ward')




from scipy.cluster.hierarchy import linkage, fcluster

# Get matrix form of the dendrogram 
Z = linkage((sales_ca1>0).astype(int), method='ward')
print(Z)




# Map items to clusters
clust = fcluster(Z, t=12, criterion='maxclust')
pd.crosstab(clust, sales_ca1.index.str[:-6])






