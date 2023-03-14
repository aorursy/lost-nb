#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr




import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

# Join the training and test sets
covid19 = pd.concat([train, test])
# Sort by date
covid19.sort_values('Date')
# Filter to the columns we need
covid19 = covid19.loc[:, ['Country/Region', 'Date', 'ConfirmedCases']]

covid19.head()




covid19 = covid19[covid19.ConfirmedCases > 50]
covid19_numdays = covid19.loc[:, ['Country/Region', 'Date']]    .drop_duplicates()    .groupby('Country/Region')    .count()    .rename_axis('country')    .reset_index()
print(covid19_numdays.head())

covid19_mindays = covid19_numdays[covid19_numdays.Date >= 14]
covid19 = covid19[covid19['Country/Region'].isin(covid19_mindays.country)]




print(len(list(set(covid19['Country/Region'].values))))
print(set(covid19['Country/Region'].values))




covid19[covid19['Country/Region'] == 'China'].head()




covid19_collapse_province = covid19    .groupby(['Country/Region', 'Date'])    .sum()    .reset_index()
covid19_collapse_province[covid19_collapse_province['Country/Region'] == 'China'].head()




covid19 = covid19_collapse_province    .groupby('Country/Region')    .head(14)    .groupby('Country/Region')    .tail(1)

covid19




country_isos = pd.read_csv('/kaggle/input/countries-iso-codes/wikipedia-iso-country-codes.csv')
country_isos = country_isos.rename(columns={"English short name lower case": "Country/Region", 
                                            "Alpha-2 code": "country_abbr"})
country_isos = country_isos.loc[:, ['Country/Region', 'country_abbr']]
country_isos.head()




covid19 = covid19.merge(country_isos, left_on='Country/Region', right_on='Country/Region')
covid19 = covid19.dropna()
covid19.head()




big5 = pd.read_csv('/kaggle/input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv', sep='\t')




positively_keyed = ['EXT1', 'EXT3', 'EXT5', 'EXT7', 'EXT9',
                    'EST1', 'EST3', 'EST5', 'EST6', 'EST7', 'EST8', 'EST9', 'EST10',
                    'AGR2', 'AGR4', 'AGR6', 'AGR8', 'AGR9', 'AGR10',
                    'CSN1', 'CSN3', 'CSN5', 'CSN7', 'CSN9', 'CSN10', 
                    'OPN1', 'OPN3', 'OPN5', 'OPN7', 'OPN8', 'OPN9', 'OPN10']

negatively_keyed = ['EXT2', 'EXT4', 'EXT6', 'EXT8', 'EXT10',
                    'EST2', 'EST4',
                    'AGR1', 'AGR3', 'AGR5', 'AGR7', 
                    'CSN2', 'CSN4', 'CSN6', 'CSN8', 
                    'OPN2', 'OPN4', 'OPN6']




big5.loc[:, negatively_keyed] = 6 - big5.loc[:, negatively_keyed]




big5_country_count = big5.country    .value_counts()    .rename_axis('country')    .reset_index(name='counts')

print(len(big5_country_count[big5_country_count.counts > 1000]))
print(big5_country_count[big5_country_count.counts > 1000].country.values)




big5 = big5[big5.country.isin(big5_country_count[big5_country_count.counts > 1000].country.values)]

# Filter on the columns we're going to use
big5 = big5.loc[:,['country'] + positively_keyed + negatively_keyed]




EXT = ['EXT' + str(i) for i in range(1,11)]
EST = ['EST' + str(i) for i in range(1,11)]
AGR = ['AGR' + str(i) for i in range(1,11)]
CSN = ['CSN' + str(i) for i in range(1,11)]
OPN = ['OPN' + str(i) for i in range(1,11)]




big5['EXT'] = big5.loc[:, EXT].mean(axis=1)
big5['EST'] = big5.loc[:, EST].mean(axis=1)
big5['AGR'] = big5.loc[:, AGR].mean(axis=1)
big5['CSN'] = big5.loc[:, CSN].mean(axis=1)
big5['OPN'] = big5.loc[:, OPN].mean(axis=1)
big5 = big5.loc[:, ['country', 'EXT', 'EST', 'AGR', 'CSN', 'OPN']]




big5 = big5.dropna()
big5 = big5[big5.country != 'NONE']




big5_cavgs = big5.groupby('country')                    .mean()                    .rename_axis('country')                    .reset_index()




big5_cavgs.loc[:, ['country', 'EXT']]    .sort_values(by=['EXT'])    .tail()    .plot(x = 'country', 
          y = 'EXT', 
          kind='barh', 
          legend=False)

plt.show()




covid19_big5 = covid19.merge(big5_cavgs, left_on='country_abbr', right_on='country')
covid19_big5.head()




factors = ['EXT', 'EST', 'AGR', 'CSN', 'OPN']
factor_names = ['Extraversion', 'Emotional Stability', 'Agreeableness', 'Conscientiousness', 'Openness']

for i, factor in enumerate(['EXT', 'EST', 'AGR', 'CSN', 'OPN']):
    # Compute the correlation coefficient
    corr = pearsonr(covid19_big5[factor], covid19_big5.ConfirmedCases)
    corr = [np.round(c, 2) for c in corr]
    text = 'r=%s, p=%s' % (corr[0], corr[1])
    
    ax = sns.regplot(x=factor, 
                y="ConfirmedCases", 
                data=covid19_big5)
    
    ax.set_title("Confirmed cases at 14 days after first 50 cases " + 
                 "\n by average score on Big 5 factor " + factor_names[i] + 
                 "\n" + text)
    plt.show()




factors = ['EXT', 'EST', 'AGR', 'CSN', 'OPN']
factor_names = ['Extraversion', 'Emotional Stability', 'Agreeableness', 'Conscientiousness', 'Openness']

for i, factor in enumerate(['EXT', 'EST', 'AGR', 'CSN', 'OPN']):
    # Compute the correlation coefficient
    corr = pearsonr(covid19_big5[covid19_big5.country != 'CN'][factor], 
                    covid19_big5[covid19_big5.country != 'CN'].ConfirmedCases)
    corr = [np.round(c, 2) for c in corr]
    text = 'r=%s, p=%s' % (corr[0], corr[1])
    
    ax = sns.regplot(x=factor, 
                y="ConfirmedCases", 
                data=covid19_big5[covid19_big5.country != 'CN'])
    
    ax.set_title("Confirmed cases at 14 days after first 50 cases " + 
                 "\n by average score on Big 5 factor " + factor_names[i] + 
                 "\n" + text)
    plt.show()




covid19_big5    .loc[:, ['country', 'OPN', 'ConfirmedCases']]    .sort_values('OPN', ascending=False)    .merge(country_isos, 
           left_on='country', 
           right_on='country_abbr')\
    .drop(['country_abbr', 'country'], axis=1)

