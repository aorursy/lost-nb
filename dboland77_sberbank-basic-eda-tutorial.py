#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)




train_df = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
test_df = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])
macro_df = pd.read_csv('../input/macro.csv', parse_dates=['timestamp'])




# Fill in the line of code below so that it applies the head() method to train_df
# The default is to show the first 5 records.  
# Type a number in the brackets (e.g. head(10)) to override the default
# What number does the row numbering start from?
# Are there any missing values (NaN is the symbol for Not A Number - and represents missing values)?

train_df. 




train_df.tail() # this will show us the last five records in the training set




test_df.head()




macro_df.head()




# To apply the method shape to a data frame called df_, type df_.shape. 
# Fill in the missing shape method below after the period.
# Enclosing this in print() ensures that the result is printed.
# Do the train and test data have the same number of columns?  If not, why not?
print(train_df.)
print(test_df.)
print(macro_df.)




# In the line below, fill in the 'dtypes' method after the period
# What type are most of the data fields?
data_types = train_df.
print(data_types)




# Find the type of each column and store the results in a data frame
df_dataTypes = train_df.dtypes.reset_index()

# Rename the columns for convencience (note the columns method being used)
df_dataTypes.columns = ["count","dtype"]

# So far we have one line per feature
df_dataTypes.head(10)

# In the next code chunk we use split-apply-combine to summarise it..




# Now use split-apply-combine to get the result
# First we choose the data we are going to summarise, in this case df_dataTypes[["count","dtype"]]
# Then we split it into groups of dtypes with .groupby(by = "dtype")
# Finally we apply the 'length' function (len) and combine with the aggregate function

df_dataTypes[["count","dtype"]].groupby(by = "dtype").aggregate(len).reset_index()

# Are most of the columns numeric, strings (i.e. 'object') or dates?




# Run the code below to produce a bar chart showing the count of each data type.

# Count of different datatypes
plt.figure(figsize=(10,8))
sns.countplot(df_dataTypes['dtype'])
plt.xlabel('dtype', fontsize=24)
plt.ylabel('count', fontsize=24)
plt.xticks(fontsize=14)
plt.yticks(fontsize=18)
plt.show()




# Run the code below to create a dot plot of all the prices, in increasing order

plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc.values), s = 2)
plt.xlabel('index', fontsize=24)
plt.ylabel('price', fontsize=24)
plt.xticks(fontsize=14)
plt.yticks(fontsize=18)
plt.show()




# Run the code below to create a histogram of the distrubtion of prices.

plt.figure(figsize=(10,6))
sns.distplot(train_df['price_doc'], kde=False, bins=50)
plt.xlabel('price (RUB 00m)', fontsize = 24)
plt.xticks(fontsize = 16)
plt.show()




# Run the code below to view the distribution of the (natural) log of the prices.
# How would the skew of the price data influence how you would normally go about modelling this data?
# How should the skew influence how you create predictions from the point of view 
# of this competition?

plt.figure(figsize=(10,6))
sns.distplot(np.log(train_df['price_doc']),kde=False,bins=50)
plt.xlabel('log of price', fontsize = 24)
plt.show()




# For convenience create a new feature which is the year and month of the sale
train_df['year'] = train_df['timestamp'].dt.year
train_df['month'] = train_df['timestamp'].dt.month
train_df['yearmonth'] = train_df['timestamp'].dt.strftime("%Y%m")




# Now use split-apply-combine to find the median price of sale in each month

# Remember how we used split-apply-combine above?
# df_dataTypes[["count","dtype"]].groupby(by = "dtype").aggregate(len).reset_index()

# Let us do the same here.  We will store the result in df_grouped, so we start with
# df_grouped = 
# We will summarise: train_df[['yearmonth', 'price_doc']]
# We will groupby 'yearmonth'
# We will apply the function np.median
# Fill in the code below
df_grouped = (train_df[['yearmonth', 'price_doc']]
              .groupby(by = )
              .aggregate()
              .reset_index())

df_grouped.head()




# Now run the following code chunk to plot the trend in median price
fig, ax = plt.subplots(figsize = [12,8])
sns.pointplot(df_grouped['yearmonth'].values, df_grouped['price_doc'].values)


x_ticks_labels = df_grouped['yearmonth'].values
x_ticks = 4 * np.arange(0,12)
plt.xticks(x_ticks, x_ticks_labels[x_ticks], rotation = 45)

plt.xlabel('year - month', fontsize=24)
plt.ylabel('median price', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()




# Run the code below to create a heat map of the correlations.

plt.figure(figsize=(10,8))
internal_characteristics=['full_sq', 'life_sq', 'floor', 'max_floor', 'material',
                          'num_room', 'kitch_sq','price_doc']
heatmap_data=train_df[internal_characteristics].corr()
sns.heatmap(heatmap_data, annot=True)
plt.show()




# Find the frequency of property by build year.
grouped_data_count = train_df.groupby('build_year')['id'].aggregate('count').reset_index()
grouped_data_count.columns=['build_year','count']
print(grouped_data_count.head())
print(grouped_data_count.tail())




# Print the tail of grouped_data_count, can you see any outliers?

print()

# Now print the head - are there any other outliers?

print()




train_missing = train_df.isnull().sum()/len(train_df)
train_missing = train_missing.drop(train_missing[train_missing == 0].index).sort_values(ascending = False).reset_index()
train_missing.columns = ['column name','missing percentage']

plt.figure(figsize = (12,8))
sns.barplot(train_missing['column name'], train_missing['missing percentage'], palette = 'coolwarm')
plt.xticks(rotation = 'vertical')
plt.show()

