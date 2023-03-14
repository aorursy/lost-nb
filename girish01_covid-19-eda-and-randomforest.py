#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, save
import plotly.graph_objects as go
import sklearn
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
pd.plotting.register_matplotlib_converters()


# In[3]:


train_df=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
test_df=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')
all_data=pd.concat([train_df,test_df],axis=0)
all_data.reset_index(drop=True)


# In[4]:


train_df.head()


# In[5]:


train_df.info()


# In[6]:


train_df['Country_Region'].nunique()


# In[7]:


print("fill blanks and add region for counting")

train_df.drop('Province_State',axis=1,inplace=True)


# In[8]:


# Resetting Date column into Datetime object and making it an index of dataframe
train_df['Date']=pd.to_datetime(train_df['Date'])
train_df.set_index('Date',inplace=True)


# In[9]:


pivot=pd.pivot_table(train_df,columns='Country_Region',index='Date',values='ConfirmedCases',aggfunc=np.sum)
pivot_fatality=pd.pivot_table(train_df,columns='Country_Region',index='Date',values='Fatalities',aggfunc=np.sum)
country_list=[]
value_list=[]
fatality_list=[]
for country in list(pivot.columns):
    country_list.append(country)
    value_list.append(pivot[country].max())
    fatality_list.append(pivot_fatality[country].max())
    new_dict={'Country':country_list,'Confirmed':value_list,'Fatality':fatality_list}
df=pd.DataFrame.from_dict(new_dict)
df.set_index('Country',inplace=True)

plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
df['Confirmed'].sort_values(ascending=False)[:10].plot(kind='bar',color='blue')
plt.title('Top 10 Countries by Confirmed Cases')
plt.subplot(2,1,2)
df['Fatality'].sort_values(ascending=False)[:10].plot(kind='bar',color='red')
plt.title('Top 10 Countries with Fatalities due to Covid-19')
plt.tight_layout()


# In[10]:


top_confirmed=df.sort_values(by='Confirmed',ascending=False)[:10]


# In[11]:


# Make List of Top 10 Countries and India and Netherlands- The Countries of interest.
list_countries=list(top_confirmed.index)
list_countries.append('India')
list_countries.append('Netherlands')


# In[12]:


times_series_cntr = train_df.groupby(['Date','Country_Region'])['ConfirmedCases'].sum()                    .reset_index().set_index('Date')
df_countries_tm = times_series_cntr[times_series_cntr['Country_Region'].isin(list_countries)]


plt.figure(figsize=(16,12))
ax = sns.lineplot(x=df_countries_tm.index, y="ConfirmedCases", hue="Country_Region", data=df_countries_tm,palette='muted').set_title('Cumulative line')
plt.legend(loc=2, prop={'size': 12})
plt.title('Cumulative trend plot for Confirmed Cases')
plt.xticks(rotation=90);


# In[13]:


Confirm_pivot=pd.pivot_table(train_df,index='Date',columns='Country_Region',
                             values='ConfirmedCases',aggfunc=np.sum)


# In[14]:


plt.figure(figsize=(16,8))
colors=['r','b','g','y','orange','purple','m','hotpink','violet','darkgreen','navy','brown']
for i,country in enumerate(list_countries):
    Confirm=Confirm_pivot[Confirm_pivot[country]>0][country].diff().fillna(0)
    Confirm=Confirm[Confirm>0]
    Confirm.plot(color=colors[i],label=country,markersize=12,lw=5)    
    plt.title('Number of Daily Cases',fontsize=15)
    plt.legend(title='country')
plt.tight_layout()


# In[15]:


plt.figure(figsize=(20,16))
colors=['r','b','g','y','orange','purple','m','hotpink','violet','darkgreen','navy','brown']
for i,country in enumerate(list_countries):
    Confirm=Confirm_pivot[Confirm_pivot[country]>0][country].diff().fillna(0)
    Confirm=Confirm[Confirm>0]
    
    plt.subplot(4,3,i+1)
    Confirm.plot(color=colors[i],label=country,markersize=12,lw=5)    
    plt.xticks()
    plt.legend(title='Country')
    plt.title('Number of Daily Cases in {}'.format(country.upper()))
plt.tight_layout()


# In[16]:


for country in list_countries:
    id_max=Confirm_pivot[country].diff().fillna(0).idxmax()
    maxim=Confirm_pivot[country].diff().fillna(0).max()
    print('Maximum New Cases registered for {0} was {1} on {2}'.format(country,maxim,id_max))


# In[17]:


Fatal_pivot=pd.pivot_table(train_df,index='Date',columns='Country_Region',values='Fatalities',aggfunc=np.sum)


# In[18]:


plt.figure(figsize=(16,8))
for i,country in enumerate(list_countries):
    Fatal_diff=Fatal_pivot[(Fatal_pivot[country]>0)][country].diff().fillna(0)
    Fatal_diff=Fatal_diff[Fatal_diff>0]
    Fatal_diff.plot(color=colors[i],label=country,lw=5)
    plt.title('Number of daily new Fatalities',fontsize=15)
    plt.legend(title='country')
plt.tight_layout()


# In[19]:


plt.figure(figsize=(20,16))
for i,country in enumerate(list_countries):
    Fatal_diff=Fatal_pivot[(Fatal_pivot[country]>0)][country].diff().fillna(0)
    Fatal_diff=Fatal_diff[Fatal_diff>0]
    plt.subplot(3,4,i+1)
    Fatal_diff.plot(color=colors[i],label=country.upper(),lw=5)
    plt.xticks(rotation=60)
    plt.title('Number of daily new Fatalities  in {}'.format(country.upper()))
    plt.legend(title='Country')
plt.tight_layout()


# In[20]:


# Understanding New cases confirmation variations on daily basis
plt.figure(figsize=(20,16))
for i,country in enumerate(list_countries):
    plt.subplot(4,3,i+1)
    train_df[(train_df['Country_Region']==country)&(train_df['ConfirmedCases']!=0)].groupby('Date')['ConfirmedCases'].sum().diff().diff().plot(color=colors[i])
    plt.ylabel('Difference in Daily reporting cases ')
    plt.title('Variation of {}'.format(country),va='bottom')
plt.suptitle('Variation in number of confirmed cases on daily basis',fontsize=24,va='baseline')


# In[21]:


plt.figure(figsize=(16,8))
plt.title('Confirmed Cases trend from first day of incidence')
for i,country in enumerate(list_countries):
    confirm_group=train_df[(train_df['Country_Region']==country)&train_df['ConfirmedCases']!=0].groupby('Date').agg({'ConfirmedCases':['sum']})
    confirm_value=[j for j in confirm_group.ConfirmedCases['sum'].values]
    plot_value=confirm_value[0:60]
    plt.plot(plot_value,color=colors[i],label=country,lw=2)
    plt.legend(title='Countries')


# In[22]:


plt.figure(figsize=(16,10))
plt.title('Fatalities trend from first day of incidence')
for i,country in enumerate(list_countries):
    fatal_group=train_df[(train_df['Country_Region']==country)&train_df['ConfirmedCases']!=0].groupby('Date').agg({'Fatalities':['sum']})
    fatal_value=[j for j in fatal_group.Fatalities['sum'].values]
    plot_value=fatal_value[0:60]
    plt.plot(plot_value,color=colors[i],label=country,lw=2)
    plt.legend(title='Countries')


# In[23]:


plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
train_df.groupby('Date')['ConfirmedCases'].sum().plot(color='blue')
plt.ylabel('Number of Confirmed Cases')
plt.title('Confirmed Cases worldwide trend')

plt.subplot(1,2,2)
train_df.groupby('Date')['Fatalities'].sum().plot(color='r')
plt.ylabel('Number of Fatalities')
plt.title("Fatalities worldwide trend")

plt.tight_layout()


# In[24]:


# Confirmed Cases and Fatalities without China's data
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
train_df[(train_df['Country_Region']!='China')&(train_df['ConfirmedCases']!=0)].groupby('Date')['ConfirmedCases'].sum().plot(color='blue')
plt.ylabel('Number of Confirmed Cases')
plt.title('Confirmed Cases worldwide trend(without China)')

plt.subplot(1,2,2)
train_df[(train_df['Country_Region']!='China')&(train_df['Fatalities']!=0)].groupby('Date')['Fatalities'].sum().plot(color='red')
plt.ylabel('Number of Fatalities')
plt.title("Fatalities worldwide trend(without China)")
plt.tight_layout()


# In[25]:


countries=train_df['Country_Region'].unique()


# In[26]:


country_list=[]
confirmation_list=[]
list_fatality=[]
for country in countries:
    country_list.append(country)
    confirm_country=train_df[train_df.Country_Region==country].groupby('Date')['ConfirmedCases'].sum().max()
    confirmation_list.append(confirm_country)
    fatal_country=train_df[train_df.Country_Region==country].groupby('Date')['Fatalities'].sum().max()
    list_fatality.append(fatal_country)
max_dict={'Country':country_list,'ConfirmedCases':confirmation_list,'Fatalities':list_fatality}
map_df=pd.DataFrame.from_dict(max_dict)


# In[27]:


map_df


# In[28]:


code_df=pd.read_csv('../input/contry-codes/country-codes.csv')


# In[29]:


code_df=code_df[['ISO3166-1-Alpha-3','CLDR display name']]


# In[30]:


map_df=map_df.merge(code_df,left_on='Country',right_on='CLDR display name')


# In[31]:


map_df.drop('CLDR display name',axis=1,inplace=True)


# In[32]:


map_df.rename(columns={'ISO3166-1-Alpha-3':'Country Code'},inplace=True)


# In[33]:



from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)

data=go.Choropleth(
    locations=map_df['Country Code'], # Spatial coordinates
    z = map_df['ConfirmedCases'], # Data to be color-coded,
    colorscale = 'Reds',
    text=map_df['Country'],
    colorbar_title = "Number of Confirmed Cases",)

fig=go.Figure(data)

fig.update_layout(
    title='Covid-19 Confirmed Cases',
           geo=dict(showframe=False,
                   projection={'type':'robinson'}))


iplot(fig)


# In[34]:


test_df['Date']=pd.to_datetime(test_df['Date'])


# In[35]:


test_df['Province_State']=test_df.drop('Province_State',axis=1)


# In[36]:


train_df=train_df.reset_index()


# In[37]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
train_df['Country_Region']=LE.fit_transform(train_df['Country_Region'])
test_df['Country_Region']=LE.transform(test_df['Country_Region'])

train_df.loc[:, 'Date'] = train_df.Date.dt.strftime("%m%d")
train_df["Date"]  = train_df["Date"].astype(int)

test_df.loc[:, 'Date'] = test_df.Date.dt.strftime("%m%d")
test_df["Date"]  = test_df["Date"].astype(int)


# In[38]:


#Select feature column names and target variable we are going to use for training
features=['Date','Country_Region']
target = 'ConfirmedCases'

#This is input which our classifier will use as an input.
train_df[features].head(10)


# In[39]:


from sklearn.ensemble import RandomForestClassifier

# We define the model
rfcla = RandomForestClassifier(n_estimators=100, max_samples=0.8,
                        random_state=1)
# We train model
rfcla.fit(train_df[features],train_df[target])


# In[40]:


#Make predictions using the features from the test data set
predictions = rfcla.predict(test_df[features])

predictions


# In[41]:


target2='Fatalities'


# In[42]:


# We define the model
rfcla2 = RandomForestClassifier(n_estimators=100, max_samples=0.8,
                        random_state=1)
# We train model
rfcla2.fit(train_df[features],train_df[target2])


# In[43]:


#Make predictions using the features from the test data set
predictions2 = rfcla2.predict(test_df[features])

print(predictions2[0:50])


# In[44]:


#Create a  DataFrame
submission = pd.DataFrame({'ForecastId':test_df['ForecastId'],'ConfirmedCases':predictions,'Fatalities':predictions2})
                        

#Visualize the first 10 rows
submission.head(10)


# In[45]:


#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:




