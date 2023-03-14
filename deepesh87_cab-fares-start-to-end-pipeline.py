#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# In[2]:


train_original=pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv',nrows=1000000,parse_dates=['pickup_datetime'])
test_original=pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')


# In[3]:


train=train_original.copy()
test=test_original.copy()


# In[4]:


test.head()


# In[5]:


train.dtypes


# In[6]:


train['pickup_datetime']=pd.to_datetime(train['pickup_datetime'])
test['pickup_datetime']=pd.to_datetime(test['pickup_datetime'])


# In[7]:


train_original.shape,test_original.shape


# In[8]:


train.isnull().sum()#Very few Missing value, so will delete the rows that have it


# In[9]:


test.isnull().sum() #good , No missing in test


# In[10]:


train.dropna(axis=0,inplace=True)


# In[11]:


train.shape## only 10 were deleted. the same rows had NAs


# In[12]:


train.describe() #Negative fare,Latitudes range from -90 to 90, and longitudes range from -180 to 80.
#Passeneger count max is 208? Is it a Cab or train?? :D
# A lot of cleaning will be needed here


# In[13]:


test.describe()#data looks good for the test data


# In[14]:


print(sum(train['fare_amount']<0)) #only a few values are negative, Will delete them
#Also from Kaggle found that the NYC cabs charge a min of $2.50(2019), so will delete records below $2(adjusting for inflation)

print(sum(train.fare_amount<2)) #very few records, so wouldnt matter anyways


# In[15]:


train=train[(train['fare_amount']>=2)]
train.shape


# In[16]:


train['passenger_count'].value_counts()#passenger count 208?? Passenger count 0, lets see the fares for these.


# In[17]:


#delete passenger count 208 will be deleted. passenger count 0 will also be deleted as  it doesnot make sense + test doesnot have this
train=train[(train['passenger_count']<7)&(train['passenger_count']>0)]


# In[18]:


plt.figure(figsize=(16, 5))
plt.hist(train['fare_amount'],bins=100);
plt.title("Fare Amount");


# In[19]:


#lets Zoom into the above figure. Be mindful of the Y axis
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title("Low/Med Fare Amount")
plt.hist(train[train['fare_amount']<100]['fare_amount'],bins=100);
plt.subplot(1, 2, 2)
plt.title("High Fare Amount")
plt.hist(train[train['fare_amount']>=100]['fare_amount'],bins=100);
#looks like there are many rides having fixed charge of 350,400,450 and 500. But these are very few in numbers <100 total


# In[20]:


sns.catplot(x='passenger_count', y='fare_amount', data=train);
plt.title('Fare wrt Total Passengers');#so fare not very much dependent on the # of Passenegrs


# In[21]:


#Create Day of Week, Month, Year, Time of Day etc variables from the pickup_datetime variable
def process_date(df,colname):
    date_parts = ["year", "weekday", "month", 'weekofyear', 'day', 'quarter','hour']
    for part in date_parts:
        part_col = colname.split('_')[0] + "_" + part
        df[part_col] = getattr(df[colname].dt, part).astype(int)
    
    return df

train = process_date(train,'pickup_datetime')
test = process_date(test,'pickup_datetime')


# In[22]:


train.head()


# In[23]:


sns.catplot(x='pickup_year', y='fare_amount', data=train);
plt.title('Fare wrt pickup_year');#definitely see some outliers in each year


# In[24]:


sns.boxplot(x='pickup_year', y='fare_amount', data=train);


# In[25]:


sns.catplot(x='pickup_weekday', y='fare_amount', data=train);
plt.title('Fare wrt pickup_dayofweek');#not much of a difference except  afew outliers


# In[26]:


sns.catplot(x='pickup_month', y='fare_amount', data=train);
plt.title('Fare wrt pickup_month');#not much of a difference except  afew outliers but in Summer months


# In[27]:


sns.catplot(x='pickup_hour', y='fare_amount', data=train);
plt.title('Fare wrt pickup_day');


# In[28]:


#Find min/max longitude and latitude in the data
print('Train data')
print('Min Pickup Longitude: {}, Max Pickup Longitude {}'.format(max(train['pickup_longitude']),min(train['pickup_longitude'])))
print('Min Drop Off Longitude: {}, Max Drop Off Longitude {}'.format(max(train['dropoff_longitude']),min(train['dropoff_longitude'])))
print('Min Pickup Latitude: {}, Max Pickup Latitude {}'.format(max(train['pickup_latitude']),min(train['pickup_latitude'])))
print('Min Drop Off Latitude: {}, Max Drop Off Latitude {}'.format(max(train['dropoff_latitude']),min(train['dropoff_latitude'])))


# In[29]:


#lets correct the lat/long values. Note 1degree is approx 100 kms for both Lat & long (at this place on earth)
print(np.quantile(train['pickup_latitude'],[0.025,0.05,0.95,0.975]))
print(np.quantile(train['dropoff_latitude'],[0.025,0.05,0.95,0.975]))
#Analysing these results, lets say we will take threshold as 40.50 and 40.90( after rounding off in decimal places)


# In[30]:


print(np.quantile(train['pickup_longitude'],[0.025,0.05,0.95,0.975]))
print(np.quantile(train['dropoff_longitude'],[0.025,0.05,0.95,0.975]))
#Analysing these results, lets say we will take threshold as -74.10 and -73.60( after rounding off in decimal places)
#Note we may be tempted to use the limits from test data but that is IMHO cheating


# In[31]:


print('Test data')
print('Min Pickup Longitude: {}, Max Pickup Longitude {}'.format(max(test['pickup_longitude']),min(test['pickup_longitude'])))
print('Min Drop Off Longitude: {}, Max Drop Off Longitude {}'.format(max(test['dropoff_longitude']),min(test['dropoff_longitude'])))
print('Min Pickup Latitude: {}, Max Pickup Latitude {}'.format(max(test['pickup_latitude']),min(test['pickup_latitude'])))
print('Min Drop Off Latitude: {}, Max Drop Off Latitude {}'.format(max(test['dropoff_latitude']),min(test['dropoff_latitude'])))


# In[32]:


train.shape


# In[33]:


#we see a huge difference between train & test. The extreme co-ordinates of train are not even present in the US not even feasible
#I will just remove the rows based on the max/min co-ordinates I see in the test data. As the test data seems very clean
#with the max/min values of lat-long
boundary=(-74.10,-73.60,40.50,40.90)
train=train[(train['pickup_longitude']>boundary[0])&(train['pickup_longitude']<boundary[1])&            (train['pickup_latitude']>boundary[2])&(train['pickup_latitude']<boundary[3])]

train=train[(train['dropoff_longitude']>boundary[0])&(train['dropoff_longitude']<boundary[1])&            (train['dropoff_latitude']>boundary[2])&(train['dropoff_latitude']<boundary[3])]


# In[34]:


train.shape


# In[35]:


#https://en.wikipedia.org/wiki/Haversine_formula
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180 , since 2pi radians=260degrees
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin... #multiply this by 0.62137 for miles


# In[36]:


train['Haversine_distance']=distance(train['pickup_latitude'],train['pickup_longitude'],                                     train['dropoff_latitude'],train['dropoff_longitude'])
test['Haversine_distance']=distance(test['pickup_latitude'],test['pickup_longitude'],                                     test['dropoff_latitude'],test['dropoff_longitude'])


# In[37]:


#lest see the distance 
train['Haversine_distance'].describe()#max distance is 35kms


# In[38]:


np.quantile(test['Haversine_distance'],[0.95,0.99,1])#for test the max is 100 kms but i believe that such high values are outliers
#So, though the train seems a bit different from test, it okay i guess as test has a few outliers.


# In[39]:


print(train[train['Haversine_distance']==0]['fare_amount'].describe())
print(train[train['Haversine_distance']==0].shape)
#Many, 10k records have the same Pickup& Drop location but still a positive fare. 
#May be a genuine ride as we have a few of these in test too


# In[40]:


print(test[test['Haversine_distance']==0].shape)


# In[41]:


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title("Low/Med distance")
plt.hist(train[train['Haversine_distance']<20]['Haversine_distance'],bins=100);
plt.subplot(1, 2, 2)
plt.title("High distance")
plt.hist(train[train['Haversine_distance']>=20]['Haversine_distance'],bins=100);


# In[42]:


#plot Fare vs ditance.
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.scatter(train['Haversine_distance'],train['fare_amount']);
plt.xlabel('distance kms')
plt.ylabel('Fare USD')
plt.title('all data')

plt.subplot(1, 2, 2)
plt.scatter(train[(train['Haversine_distance']<25)&(train['fare_amount']<100)]['Haversine_distance'],            train[(train['Haversine_distance']<25)&(train['fare_amount']<100)]['fare_amount']);
plt.xlabel('distance kms')
plt.ylabel('Fare USD')
plt.title('Zoom in on Fare<100 and Distance<25 kms');
#we can see that as the distance increases Fare amount increases


# In[43]:


test['Haversine_distance'].describe(),train['Haversine_distance'].describe()
#we can see there are a lot of outliers in the training data. 


# In[44]:


# What have we done so far?
# 1) Applied filtering on lat/long based on percentiles
# 2) Removed <$2 fare
# 3) Removed >6 Passenger_count
# 4) Calculated Haversine Distance, that comes to max 35 kms in train but 99 kms in test
# 5) the test max HD is actually an outlier (just one/2 record with 99kms), all other are in fact below 25 kms


# In[45]:


print('Count of train rows with 0 haversine distance is {}'.format(sum(train['Haversine_distance']==0)))
#lets get drop these rows where distance is 0


# In[46]:


print('Count of test rows with 0 haversine distance is {}'.format(sum(test['Haversine_distance']==0)))
#since test also has a few HD==0, i will not delete these observations from train data


# In[47]:


#Latitude: 1 deg = 110.574 km
#Unlike latitude, the distance between degrees of longitude varies greatly depending upon your 
#location on the planet. They are farthest apart at the equator and converge at the poles.
#A degree of longitude is widest at the equator with a distance of 110kms
#np.digitize([2,11,21,33],bins=[1,5,10,15,20])#the bins are 0(0-1),1(1-5),2(5-10),3(10-15),4(15-20),5(20-)
#above gives array([1, 3, 5, 5])
#boundary=(-74.50,-72.80,40.50,41.80)Long/lat 
#Delta long is 1.7 and delta lat is 1.29, so around 1.7*90=153(lesser than it is at equator) kms lat and 1.29*110=142 kms long
#distance(lat1, lon1, lat2, lon2)


# In[48]:


#Some trips, like to/from an airport, are fixed fee. To prce this see the plot below
# JFK airport coordinates, see https://www.travelmath.com/airport/JFK
jfk = (-73.7822222222, 40.6441666667) #airport
nyc = (-74.0063889, 40.7141667)#city centre

def plot_location_fare(loc, name, range=2): #within range kms of the location
    # select all datapoints with dropoff location within range
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    idx = (distance(train.pickup_latitude, train.pickup_longitude, loc[1], loc[0]) < range)
    train[idx].fare_amount.hist(bins=100, ax=axs[0])
    axs[0].set_xlabel('fare $USD')
    axs[0].set_title('Histogram pickup location within {} KMS of {}'.format(range, name))

    idx = (distance(train.dropoff_latitude, train.dropoff_longitude, loc[1], loc[0]) < range)
    train[idx].fare_amount.hist(bins=100, ax=axs[1])
    axs[1].set_xlabel('fare $USD')
    axs[1].set_title('Histogram dropoff location within {} KMS of {}'.format(range, name));


# In[49]:


plot_location_fare(jfk,'JFK airport',3)#looks like it is true. Fare is the same for most rides within 5kms from jfk airport


# In[50]:


ewr = (-74.175, 40.69) # Newark Liberty International Airport, see https://www.travelmath.com/airport/EWR
lgr = (-73.87, 40.77) # LaGuardia Airport, see https://www.travelmath.com/airport/LGA
plot_location_fare(ewr, 'Newark Airport',3)
plot_location_fare(lgr, 'LaGuardia Airport',3)


# In[51]:


#So,lets add a binary variable which is 1 if the pick up is from JFK and another if the drop is at JFk
##TRAIN
#This did not help ( got to know by looking at the feature importance plot from XGB) so wont take this nomore
# train.loc[:,'Pick_up_jfk']=np.where(distance(train.pickup_latitude,\
#                                                             train.pickup_longitude, jfk[1], jfk[0])<3,1,0)

# train.loc[:,'dropoff_jfk']=np.where(distance(train.dropoff_latitude,\
#                                                             train.dropoff_longitude, jfk[1], jfk[0])<3,1,0)

# train.loc[:,'Pick_up_ewr']=np.where(distance(train.pickup_latitude,\
#                                                             train.pickup_longitude, ewr[1], ewr[0])<3,1,0)

# train.loc[:,'dropoff_ewr']=np.where(distance(train.dropoff_latitude,\
#                                                             train.dropoff_longitude, ewr[1], ewr[0])<3,1,0)


# train.loc[:,'Pick_up_lgr']=np.where(distance(train.pickup_latitude,\
#                                                             train.pickup_longitude, lgr[1], lgr[0])<3,1,0)

# train.loc[:,'dropoff_lgr']=np.where(distance(train.dropoff_latitude,\
#                                                             train.dropoff_longitude, lgr[1], lgr[0])<3,1,0)

# ##Test

# test.loc[:,'Pick_up_jfk']=np.where(distance(test.pickup_latitude,\
#                                                             test.pickup_longitude, jfk[1], jfk[0])<3,1,0)

# test.loc[:,'dropoff_jfk']=np.where(distance(test.dropoff_latitude,\
#                                                             test.dropoff_longitude, jfk[1], jfk[0])<3,1,0)

# test.loc[:,'Pick_up_ewr']=np.where(distance(test.pickup_latitude,\
#                                                             test.pickup_longitude, ewr[1], ewr[0])<3,1,0)

# test.loc[:,'dropoff_ewr']=np.where(distance(test.dropoff_latitude,\
#                                                             test.dropoff_longitude, ewr[1], ewr[0])<3,1,0)


# test.loc[:,'Pick_up_lgr']=np.where(distance(test.pickup_latitude,\
#                                                             test.pickup_longitude, lgr[1], lgr[0])<3,1,0)

# test.loc[:,'dropoff_lgr']=np.where(distance(test.dropoff_latitude,\
#                                                             test.dropoff_longitude, lgr[1], lgr[0])<3,1,0)


# In[52]:


train.sample(5)


# In[53]:


# display pivot table
train.pivot_table('fare_amount', index='pickup_hour', columns='pickup_year').plot(figsize=(14,6))
plt.ylabel('Fare $USD');
#we can see that the average Fare vs time of day has been increasing with year. #Inflation


# In[54]:


def select_within_boundingbox(df, BB):
    '''
    returns a Boolean series
    '''
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) &            (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) &            (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) &            (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])


# In[55]:


#Relevance of direction for fare amount
#How do we find out of the direction influences the fare_amount?

#Remember the co-ordinates are the LAt/Long value. from this we calculate DELTA Lat 
#and DELTA LONG and plot that wrt Fare to see if there is any
#Difference
train['delta_lat']=train['pickup_latitude']-train['dropoff_latitude']
train['delta_long']=train['pickup_longitude']-train['dropoff_longitude']

plt.figure(figsize=(14,8))
# Select only the trips in Manhattan
BB_manhattan = (-74.025, -73.925, 40.7, 40.8)#found from Google
within_manhattan_train=select_within_boundingbox(train,BB_manhattan)


plt.scatter(train[within_manhattan_train]['delta_long'],train[within_manhattan_train]['delta_lat'],            s=0.5, alpha=1.0, 
            c=np.log1p(train[within_manhattan_train]['fare_amount']), cmap='magma')
plt.colorbar()
plt.xlabel('pickup_longitude - dropoff_longitude')
plt.ylabel('pickup_latitude - dropoff_latidue')
plt.title('log1p(fare_amount)');
#Fare seems to be lesser in the center and more around perimeter nad i can see a star here.(slightly tilted). 


# In[56]:


print('total {} records out of {} are in manhattan. So {}'.format(sum(within_manhattan_train),train.shape[0],                                                                 sum(within_manhattan_train)*100/train.shape[0]))
#Since most of the records are from Manhattan, If my model predicts well for these, my overall 
#performance will be quite good


# In[57]:


from IPython.display import Image
Image(filename = "../input/manhattan/manhattan.JPG", width = 400, height = 350)


# In[58]:


#if you see the map of Manhattan, the streets are at 60 degrees and -30 degrees with horizontal. 
#hence, 2 location along this angle are very close to each other. hence you see the star.
#lets get a new variable that is the actual angle with the horizontal. This variable will help 
#whatever model we build later


# In[59]:


Image(filename = "../input/astc-picjpg/astc.JPG", width = 150, height = 120)


# In[60]:


#From the triangle that is formed above for a given street, we know the base ( delta long), perpen(delta lat)
#and can calculate the Hypotenuse ( l2 distance)

#So we can also calculate the angle of the route with the horizontal.
#TAN (Theta)= P/base
#So, Theta in degrees is tan-1 of p/base
# direction of a trip, from 180 to -180 degrees. Horizontal axes = 0 degrees.
def calculate_direction(d_lon, d_lat):
    result = np.zeros(len(d_lon))
    l = np.sqrt(d_lon**2 + d_lat**2)
    result[d_lon>0] = (180/np.pi)*np.arcsin(d_lat[d_lon>0]/l[d_lon>0])
    idx = (d_lon<0) & (d_lat>0)
    result[idx] = 180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    idx = (d_lon<0) & (d_lat<0)
    result[idx] = -180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    return result


# In[61]:


#I am calculatig this for all records but this will be applicate to only the records in Manhattan
#I will create a Binary variable saying if a record is within Manhattan or not
train['direction'] = calculate_direction(train['delta_long'],train['delta_lat'])
train['direction'].describe()


# In[62]:


#prepare the Delta variables for test data
test['delta_lat']=test['pickup_latitude']-test['dropoff_latitude']
test['delta_long']=test['pickup_longitude']-test['dropoff_longitude']
test['direction'] = calculate_direction(test['delta_long'],test['delta_lat'])
test['direction'].describe()


# In[63]:


# plot direction vs average fare amount
fig, ax = plt.subplots(1, 1, figsize=(14,6))
direc = pd.cut(train[within_manhattan_train]['direction'], np.linspace(-180, 180, 37))
train[within_manhattan_train].pivot_table('fare_amount', index=[direc], columns='pickup_year', aggfunc='mean').plot(ax=ax)
plt.xlabel('direction (degrees)')
plt.xticks(range(36), np.arange(-170, 190, 10))
plt.ylabel('average fare amount $USD');
#Clearly avg Fare is Lower in 60 degree and -120 degrees as Manhattan road are very straight in that direction(so less total distance taken)
#also google maps show me that the 60 degree roads are broader than others/hence less traffic
#also avg fare is lowest in -20degree as there is hardly any land in this direction. Mostly water. So the distances along this must be less


# In[64]:


#but for the same Haversine distance, the total actual distance and hence FARE along 60& 120 degrees must be lesser than at other angles
within_manhattan_train_and_around_5kms_trip_HD=within_manhattan_train &(train['Haversine_distance']>4.5)&(train['Haversine_distance']<5.5)
idx2=within_manhattan_train_and_around_5kms_trip_HD
# plot direction vs average fare amount
fig, ax = plt.subplots(1, 1, figsize=(14,6))
direc = pd.cut(train[idx2]['direction'], np.linspace(-180, 180, 37))
train[idx2].pivot_table('fare_amount', index=[direc], columns='pickup_year', aggfunc='mean').plot(ax=ax)
plt.xlabel('direction (degrees)')
plt.xticks(range(36), np.arange(-170, 190, 10))
plt.ylabel('average fare amount $USD');


# In[65]:


#The above should be observed at any particular total Haversine distance
within_manhattan_train_and_around_3kms_trip_HD=within_manhattan_train &(train['Haversine_distance']>2.5)&(train['Haversine_distance']<3.5)
idx3=within_manhattan_train_and_around_3kms_trip_HD
# plot direction vs average fare amount
fig, ax = plt.subplots(1, 1, figsize=(14,6))
direc = pd.cut(train[idx3]['direction'], np.linspace(-180, 180, 37))
train[idx3].pivot_table('fare_amount', index=[direc], columns='pickup_year', aggfunc='mean').plot(ax=ax)
plt.xlabel('direction (degrees)')
plt.xticks(range(36), np.arange(-170, 190, 10))
plt.ylabel('average fare amount $USD');


# In[66]:


## add the binary column for Manhatttan or not

train['manhattan']=within_manhattan_train.map(lambda x: int(x))

within_manhattan_test=select_within_boundingbox(test,BB_manhattan)
test['manhattan']=within_manhattan_test.map(lambda x: int(x))


# In[67]:


train.head()


# In[68]:


#Empirical Cumulative Distribution Function Plot for fare_amount
def ecdf(x):
    """Empirical cumulative distribution function of a variable"""
    # Sort in ascending order
    x = np.sort(x)
    n = len(x)
    
    # Go from 1/n to 1
    y = np.arange(1, n + 1, 1) / n
    
    return x, y

xs, ys = ecdf(train['fare_amount'])
plt.figure(figsize = (8, 6))
plt.plot(xs, ys, '.')
plt.ylabel('Percentile'); plt.title('ECDF of Fare Amount'); plt.xlabel('Fare Amount ($)');


# In[69]:


np.corrcoef(train['Haversine_distance'],train['fare_amount'])#there is a good co-relation


# In[70]:


## Add a column that gives the pickup and dropoff distance from the 3 airports and NYC centre. Thsi may help us capture the fixed charge 
#from these the airports and prime location charges
ewr = (-74.175, 40.69) # Newark Liberty International Airport, see https://www.travelmath.com/airport/EWR
lgr = (-73.87, 40.77) # LaGuardia Airport, see https://www.travelmath.com/airport/LGA
jfk = (-73.7822222222, 40.6441666667) #airport
nyc = (-74.0063889, 40.7141667)#city centre
#distance(lat1, lon1, lat2, lon2) use this previously built function to calculate distance
train['dis_pickup_from_ewr']=distance(train['pickup_latitude'],train['pickup_longitude'],ewr[1],ewr[0])
train['dis_dropoff_from_ewr']=distance(train['dropoff_latitude'],train['dropoff_longitude'],ewr[1],ewr[0])

train['dis_pickup_from_lgr']=distance(train['pickup_latitude'],train['pickup_longitude'],lgr[1],lgr[0])
train['dis_dropoff_from_lgr']=distance(train['dropoff_latitude'],train['dropoff_longitude'],lgr[1],lgr[0])

train['dis_pickup_from_jfk']=distance(train['pickup_latitude'],train['pickup_longitude'],jfk[1],jfk[0])
train['dis_dropoff_from_jfk']=distance(train['dropoff_latitude'],train['dropoff_longitude'],jfk[1],jfk[0])

train['dis_pickup_from_nyc']=distance(train['pickup_latitude'],train['pickup_longitude'],nyc[1],nyc[0])
train['dis_dropoff_from_nyc']=distance(train['dropoff_latitude'],train['dropoff_longitude'],nyc[1],nyc[0])

#For test

test['dis_pickup_from_ewr']=distance(test['pickup_latitude'],test['pickup_longitude'],ewr[1],ewr[0])
test['dis_dropoff_from_ewr']=distance(test['dropoff_latitude'],test['dropoff_longitude'],ewr[1],ewr[0])

test['dis_pickup_from_lgr']=distance(test['pickup_latitude'],test['pickup_longitude'],lgr[1],lgr[0])
test['dis_dropoff_from_lgr']=distance(test['dropoff_latitude'],test['dropoff_longitude'],lgr[1],lgr[0])

test['dis_pickup_from_jfk']=distance(test['pickup_latitude'],test['pickup_longitude'],jfk[1],jfk[0])
test['dis_dropoff_from_jfk']=distance(test['dropoff_latitude'],test['dropoff_longitude'],jfk[1],jfk[0])

test['dis_pickup_from_nyc']=distance(test['pickup_latitude'],test['pickup_longitude'],nyc[1],nyc[0])
test['dis_dropoff_from_nyc']=distance(test['dropoff_latitude'],test['dropoff_longitude'],nyc[1],nyc[0])


# In[71]:


train.sample(3)


# In[72]:


train.columns


# In[73]:


#COlumns that i plan to use for modelling
model_cols=['pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
        'pickup_year', 'pickup_weekday', 'pickup_month',
       'pickup_day', 'pickup_hour',
       'Haversine_distance', 'direction','manhattan', 'dis_pickup_from_ewr', 'dis_dropoff_from_ewr',
       'dis_pickup_from_lgr', 'dis_dropoff_from_lgr', 'dis_pickup_from_jfk',
       'dis_dropoff_from_jfk', 'dis_pickup_from_nyc', 'dis_dropoff_from_nyc']


# In[74]:


from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit,StratifiedKFold,RandomizedSearchCV
def rmse(y_true, y_pred):
    diff = mean_squared_error(y_true, y_pred)
    return diff**0.5
my_scorer = make_scorer(rmse,greater_is_better=False)


# In[75]:


#for Linear regression since they do not have High variance, i will simply divide Train into train & valdation
#and use this to finally predict on the test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X=train[model_cols]
y=train['fare_amount']

X_train, X_valid, y_train, y_valid =train_test_split(X, y, test_size=0.30, random_state=2020)

model_lin = Pipeline((
        ("standard_scaler", StandardScaler()),
        ("lin_reg", LinearRegression()),
    ))

model_lin.fit(X_train, y_train)
y_train_pred = model_lin.predict(X_train)
y_valid_pred = model_lin.predict(X_valid)
y_test_pred = model_lin.predict(test[model_cols])

print('RMSE on the Validation data is {} and on train is {}'.format(rmse(y_valid,y_valid_pred),                                                                    rmse(y_train,y_train_pred)))
#This scores 5.4 on Kaggle LB. So we have overfit the data


# In[76]:


#Check if there is Variance in Linear Regression
#or check if the selection of train & validation matters for the MODEL

def variance_linreg(train,n=50):#will try 50 different splits
    X=train[model_cols]
    y=train['fare_amount']
    rmse_train=[]
    rmse_valid=[]
    for i in range(n):
        X_train, X_valid, y_train, y_valid =train_test_split(X, y, test_size=0.30, random_state=i)
        model_lin = Pipeline((
        ("standard_scaler", StandardScaler()),
        ("lin_reg", LinearRegression()),
    ))
        model_lin.fit(X_train, y_train)
        y_train_pred = model_lin.predict(X_train)
        rmse_train.append(rmse(y_train,y_train_pred))
        y_valid_pred = model_lin.predict(X_valid)
        rmse_valid.append(rmse(y_valid,y_valid_pred))
    
    return (np.mean(rmse_valid),np.std(rmse_valid))

variance_linreg(train,n=20)#Not much deviation from mean. So No Variance


# In[77]:


class Hyper_param_tuning():
    def __init__(self,train_x,train_y,folds=5,n_estimators=500):
        self.train_x=train_x
        self.train_y=train_y
        #self.test_x=test_x
        self.folds=folds
        self.n_estimators=n_estimators
        #self.skf = StratifiedKFold(n_splits=self.folds, shuffle = True, random_state = 2017)
        
        
    def Tree_Model(self,params,model_name,param_comb=0):
        """
        model_name should be xgb or lgbm or rf in smalls letters
        """
        
        if model_name=='xgb':
            model=XGBRegressor(learning_rate=0.04,n_estimators=self.n_estimators ,objective='reg:squarederror')
        elif model_name=='lgbm':
            model=lgb.LGBMRegressor(learning_rate=0.02,n_estimators=self.n_estimators ,objective='regression')
        else:
            print("Running a RF Model")
            model=RandomForestRegressor(random_state=2,criterion='mse')
        
        search_obj = GridSearchCV(estimator=model, param_grid=params,                                scoring=my_scorer, n_jobs=-1, cv=self.folds, verbose=3)      
        
        search_obj.fit(self.train_x,self.train_y)
        print('\n Best estimator:')
        print(search_obj.best_estimator_) #gives values of all hyperparameters
        print('\n Best hyperparameters for {} Model are:'.format(model_name))
        print(search_obj.best_params_) #gives the best out of the parameter search space
        print('\n Best Score for {}-fold search is {}'.format(self.folds,search_obj.best_score_))


# In[78]:


from sklearn.model_selection import KFold
Number_of_folds = 5
#We have to make sure same K fold splits are used for all Models. This avoids Overfitting and Leakage
folds = KFold(n_splits=Number_of_folds, shuffle=True, random_state=2017)

X_train=train[model_cols]
y_train=train['fare_amount']
X_test=test[model_cols]

tune=Hyper_param_tuning(X_train,y_train,folds=5)


# In[79]:


from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit,StratifiedKFold,RandomizedSearchCV
from xgboost import XGBRegressor
import lightgbm as lgb
import xgboost as xgb


# In[80]:


get_ipython().run_cell_magic('time', '', "#RF\n#parameter grid for RF\nparams = {'n_estimators': [100],\n              'max_features': ['sqrt'], #, 'sqrt','auto'\n             # 'criterion':  ['gini'], #'entropy',#gini is for clssification\n              'max_depth': [30,40,50,80],\n              'min_samples_leaf': [40,15,50]\n            # 'min_samples_split':5,\n            }\n#xgb=tune.Tree_Model(params=params,model_name='rf')\n\n#10 trees, 160 fits, 1 hour, -2.821685185125628,{'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 15, 'n_estimators': 10}\n# 30 trees,60 fits, 28 minutes,-2.808, {'max_depth': 40, 'max_features': 'sqrt', 'min_samples_leaf': 15, 'n_estimators': 30}\n# 60 trees,60 fits,55min, -2.801,{'max_depth': 40, 'max_features': 'sqrt', 'min_samples_leaf': 15, 'n_estimators': 60}\n# 100 trees,60 fits,90 min,-2.798,{'max_depth': 40, 'max_features': 'sqrt', 'min_samples_leaf': 15, 'n_estimators': 100}\n# I notice hardly  any change in the metric with increasing number of trees. So, lets keep it at 60\n#Good thing is 'max_depth': 40, 'min_samples_leaf': 15 for all values of n_estimators")


# In[81]:


get_ipython().run_cell_magic('time', '', "## XGB\nparams = {\n        'min_child_weight': [ 5], #[ 5, 10]\n        'gamma': [1.5], #[1.5, 5]\n        'subsample': [0.6, 1.0],\n        'colsample_bytree': [0.6], #[0.6, 1.0]\n        'max_depth': [3, 10],\n        'alpha': [1],#[5,1]\n        'lambda': [5] #[5,15]\n            }\n#xgb=tune.Tree_Model(params=params,model_name='xgb')\n#1h 25min\n#-2.6689, {'alpha': 1, 'colsample_bytree': 0.6, 'gamma': 1.5, 'lambda': 5, 'max_depth': 10, 'min_child_weight': 5, 'subsample': 1.0}")


# In[82]:


get_ipython().run_cell_magic('time', '', '## LGBM\nlgbm_params= {#"max_depth": 5,          #max depth for tree model\n              #"num_leaves": 25,        #max number of leaves in one tree\n              # \'feature_fraction\':0.6,  #LightGBM will randomly select part of features on each tree node\n               \'bagging_fraction\':[0.8],    #randomly select part of data without resampling\n              # \'max_drop\': 5,         #used only in dart,max number of dropped trees during one boosting iteration\n              \'lambda_l1\': [5],#[1,5]\n              \'lambda_l2\':[ 0.01,0.5], #[ 0.01,0.5,10]\n              \'min_child_samples\':[400,600],  #minimal number of data in one leaf\n                \'max_bin\':[15,20], #max number of bins that feature values will be bucketed in. Higher value--> Overfitting\n               # \'subsample\':[0.6,0.8],  #randomly select part of data without resampling\n                \'colsample_bytree\':[0.8], #same as feature_fraction\n               \'boosting_type\': [\'dart\']   #options are gbdt(gradientboosting decision trees), rf,dart,goss\n                }  #weight of labels with positive class\n\n#lgbm=tune.Tree_Model(params=lgbm_params,model_name=\'lgbm\')\n\n#-3.291, {\'bagging_fraction\': 0.8, \'boosting_type\': \'dart\', \'colsample_bytree\': 0.8, \'lambda_l1\': 5,\n#\'lambda_l2\': 0.5, \'max_bin\': 20, \'min_child_samples\': 400}, 2 hours')


# In[83]:


from bayes_opt import BayesianOptimization
import xgboost as xgb
# params = {
#         'min_child_weight': [ 5], #[ 5, 10]
#         'gamma': [1.5,5], #[1.5, 5]
#         'subsample': [0.6, 1.0],
#         'colsample_bytree': [0.6,1.0], #[0.6, 1.0]
#         'max_depth': [3, 10],
#         'alpha': [1],#[5,1]
#         'lambda': [5] #[5,15]
#             }

def xgb_evaluate(max_depth, gamma, colsample_bytree):
    params = {'eval_metric': 'rmse',
              'max_depth': int(max_depth),
              'subsample': 1.0,
              'eta': 0.1,
              'gamma': gamma,
              'colsample_bytree': colsample_bytree}
    # Used around 1000 boosting rounds in the full model
    cv_result = xgb.cv(params, dtrain, num_boost_round=100, nfold=3)    
    
    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

#X_train, X_valid, y_train, y_valid taking these from the linear regression model
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_valid)


# In[84]:


xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 10), 
                                             'gamma': (0, 1),
                                             'colsample_bytree': (0.6, 1.0)})
# Use the expected improvement acquisition function to handle negative numbers
# Optimally needs quite a few more initiation points and number of iterations
#xgb_bo.maximize(init_points=3, n_iter=5, acq='ei') #commenting for now


# In[85]:


#Extract the parameters of the best model.
# params = xgb_bo.max['params']
# params['max_depth'] = int(xgb_bo.max['params']['max_depth'])


# In[86]:


#Provide a K-fold function that generate out-of-fold predictions for train data.
class Modelling():
    def __init__(self,X,y,test_X,folds,N):
        self.X=X
        self.y=y
        self.test_X=test_X
        self.folds=folds
        self.N=N
     
    def Single_Model(self,Regressor): #for all other Models like LInear,NB ,KNN etc
        stacker_train = np.zeros((self.X.shape[0], 1))
        stacker_test  = np.zeros(self.test_X.shape[0])        
        for index, (trn_idx,val_idx) in enumerate(self.folds.split(self.X)):
            trn_x, val_x = self.X[trn_idx], self.X[val_idx]
            trn_y, val_y = self.y[trn_idx], self.y[val_idx]
            print('Train model in fold {}'.format(index+1))           
            Regressor.fit(trn_x,(trn_y))#if passing log then take np.log1p(trn_y)
            val_pred = (Regressor.predict(val_x))#np.expm1(Regressor.predict(val_x))
            stacker_train[val_idx,0]=val_pred
            val_rmse=rmse(val_y, val_pred)            
            print('fold {} RMSE score on VAL is {:.6f}'.format(index+1, val_rmse))
            #for test
            pred_test= (Regressor.predict(self.test_X))#np.expm1(Regressor.predict(self.test_X))
            stacker_test+=(pred_test/self.N)
            
        #evaluate for entire train data (oof)
        train_rmse=rmse(self.y,stacker_train)
        print("CV score on TRAIN (OOF) is RMSE: {}".format(train_rmse))   
        return stacker_test,stacker_train        
        
        
        
    def SingleRF_oof(self,params):
        clf_rf=RandomForestRegressor(**rf_params)
        stacker_train = np.zeros((self.X.shape[0], 1))
        stacker_test  = np.zeros(self.test_X.shape[0])
        for index, (trn_idx,val_idx) in enumerate(self.folds.split(self.X,self.y)):
            trn_x, val_x = self.X[trn_idx], self.X[val_idx]
            trn_y, val_y = self.y[trn_idx], self.y[val_idx]
            print('Train model in fold {}'.format(index+1))         
            clf_rf.fit(trn_x,trn_y)
            val_pred = clf_rf.predict(val_x)
            stacker_train[val_idx,0]=val_pred
            val_rmse=rmse(val_y, val_pred)    
                        
            print('fold {} RMSE score on VAL is {:.6f}'.format(index+1,val_rmse))
            #for test
            pred_test= clf_rf.predict(self.test_X)
            stacker_test+=(pred_test/self.N)
            print('OOB Score: {}'.format(clf_rf.oob_score_)) #R2 by default for regression
        #evaluate for entire train data (oof)
        train_rmse=rmse(self.y,stacker_train)
        print("CV score on TRAIN (OOF) is RMSE: {}".format(train_rmse))   
        return stacker_test,stacker_train    

    
    def SingleXGB_oof(self,params,num_boost_round):
        stacker_train = np.zeros((self.X.shape[0], 1))
        stacker_test=np.zeros(self.test_X.shape[0])
        dtest=xgb.DMatrix(self.test_X)
        for index, (trn_idx,val_idx) in enumerate(self.folds.split(self.X)):
            trn_x, val_x = self.X[trn_idx], self.X[val_idx]
            trn_y, val_y = self.y[trn_idx], self.y[val_idx]
            dtrn = xgb.DMatrix(data=trn_x, label=(trn_y))#np.log1p(trn_y)
            dval = xgb.DMatrix(data=val_x, label=(val_y))#np.log1p(val_y))
            print('Train model in fold {}'.format(index+1)) 
            cv_model = xgb.train(params=params,dtrain=dtrn,num_boost_round=num_boost_round                                 ,evals=[(dtrn, 'train'), (dval, 'val')],verbose_eval=10,early_stopping_rounds=200)
                        
            pred_test = (cv_model.predict(dtest, ntree_limit=cv_model.best_ntree_limit))#np.expm1
            stacker_test+=(pred_test/self.N)
            val_pred=(cv_model.predict(dval, ntree_limit=cv_model.best_ntree_limit))#np.expm1
            stacker_train[val_idx,0]=val_pred
            val_rmse=rmse(val_y, val_pred)
            
            print('fold {} RMSE score on VAL is {:.6f}'.format(index+1, val_rmse))
            
        #evaluate for entire train data (oof)
        train_rmse=rmse(self.y,stacker_train)
        print("CV score on TRAIN (OOF) is RMSE: {}".format(train_rmse))   
        return stacker_test,stacker_train
    
    
    def SingleLGBM_oof(self,params,num_boost_round,colnames,importance_plot=False): #passing the col names to print the Feature imp
        stacker_train = np.zeros((self.X.shape[0], 1))
        stacker_test=np.zeros(self.test_X.shape[0])
        feature_importance =pd.DataFrame()
        for index, (trn_idx,val_idx) in enumerate(self.folds.split(self.X,self.y)):
            trn_x, val_x = self.X[trn_idx], self.X[val_idx]
            trn_y, val_y = self.y[trn_idx], self.y[val_idx]

            print('Train model in fold {}'.format(index+1)) 
            lgb_train = lgb.Dataset(trn_x,(trn_y)) #np.log1p
            lgb_val = lgb.Dataset(val_x, (val_y), reference=lgb_train)#np.log1p
            
            lgb_model = lgb.train(params,
                        lgb_train,
                        num_boost_round=num_boost_round,
                        valid_sets=lgb_val,
                        early_stopping_rounds=200,
                        verbose_eval=10)
            
            val_pred=(lgb_model.predict(val_x))#np.expm1
            val_rmse=rmse(val_y, val_pred)
            print('fold {} RMSE score on VAL is {:.6f}'.format(index+1, val_rmse))
            stacker_train[val_idx,0]=val_pred

            pred_test = (lgb_model.predict(self.test_X))#np.expm1
            stacker_test+=(pred_test/self.N)
            #feature importance
            fold_importance = pd.DataFrame()
            
            fold_importance["feature"] = colnames
            fold_importance["importance"] = lgb_model.feature_importance()
            fold_importance["fold"] = index+1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
        
        if importance_plot:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:30].index
            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
            plt.figure(figsize=(12, 9));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGBM Features (avg over folds,Top Few)');
                
        
        #evaluate for entire train data (oof)
        train_rmse=rmse(self.y,stacker_train)
        print("CV score on TRAIN (OOF) is RMSE: {}".format(train_rmse))   
        return stacker_test,stacker_train
    
    
    def SingleCatBoost_oof(self,params): #simple catboost without the cat columns
        stacker_train = np.zeros((self.X.shape[0], 1))
        stacker_test=np.zeros(self.test_X.shape[0])
        
        for index, (trn_idx,val_idx) in enumerate(self.folds.split(self.X)):
            trn_x, val_x = self.X[trn_idx], self.X[val_idx]
            trn_y, val_y = self.y[trn_idx], self.y[val_idx]
            print('Train model in fold {}'.format(index+1))              
                
            cat_model = CatBoostRegressor(**params)
            cat_model.fit(trn_x,(trn_y),eval_set=(val_x,(val_y)),use_best_model=True,verbose=False)# np.log1p
            val_pred = (cat_model.predict(val_x))#np.expm1
            stacker_train[val_idx,0]=val_pred
            val_rmse=rmse(val_y, val_pred)            
            print('fold {} RMSE score on VAL is {:.6f}'.format(index+1, val_rmse))
            #for test
            pred_test=(cat_model.predict(self.test_X))
            stacker_test+=(pred_test/self.N)
            
        #evaluate for entire train data (oof)
        train_rmse=rmse(self.y,stacker_train)
        print("CV score on TRAIN (OOF) is RMSE: {}".format(train_rmse))   
        return stacker_test,stacker_train


# In[87]:


from sklearn.model_selection import KFold
Number_of_folds = 5
#We have to make sure same K fold splits are used for all Models. This avoids Overfitting and Leakage
folds = KFold(n_splits=Number_of_folds, shuffle=True, random_state=2017)

X_train=train[model_cols]
y_train=train['fare_amount']
X_test=test[model_cols]

modelling_object = Modelling(X=X_train.values, y=y_train.values, test_X=X_test.values, folds=folds, N=Number_of_folds)


# In[88]:


get_ipython().run_cell_magic('time', '', "rf_params = {'n_estimators': 200,\n              'max_features': 'sqrt', #, 'sqrt','auto'\n              #'criterion':  'gini', #'entropy',\n              'max_depth': 40,\n              'min_samples_leaf': 15,\n            # 'min_samples_split':5,\n            # 'class_weight':'balanced',\n             'random_state':0,\n             'n_jobs': -1,\n             'oob_score': True\n            }\n\ntest_pred_stacked_rf,stacker_train_rf=modelling_object.SingleRF_oof(params=rf_params)\n#All validation scores (for each folds) come 3.34-3.92\n#TRAIN (OOF) is RMSE: 3.61\n#LB score 3.46\n#Wall time: 39min 21s")


# In[89]:


results=pd.DataFrame({'key':test['key'],'fare_amount':test_pred_stacked_rf})
results.to_csv('/kaggle/working/test_pred_stacked_rf.csv',index=False)

results_train=pd.DataFrame({'Model_fare_amount':stacker_train_rf[:,0]})
results_train.to_csv('/kaggle/working/stacker_train_rf.csv',index=False)


# In[90]:


get_ipython().run_cell_magic('time', '', "#Call XGB\nparams_for_xgb = {\n    'objective': 'reg:squarederror',  #the learning task and the corresponding learning objective\n    'eval_metric': 'rmse',            #Evaluation metrics for validation data\n    'eta': 0.04,          #learning_rate          \n    'max_depth': 10,       #Maximum depth of a tree. High will make the model more complex and more likely to overfit.\n    'min_child_weight': 5, #[0,inf] Higher the value,lesser the number of splits\n    'gamma': 0.0,       #Minimum loss reduction required to make a further partition on a leaf node of the tree    \n    'colsample_bytree': 0.6,  #subsample ratio of columns when constructing each tree\n    'alpha': 1,  #L1 regularization term on weights\n    'lambda': 5,  \n    'subsample':1.0, #'subsample': 0.8,    #Subsample ratio of the training instances\n    'seed': 2017}\n\ntest_pred_stacked_xgb,stacker_train_xgb=modelling_object.SingleXGB_oof(params=params_for_xgb,num_boost_round=1000)\n\n#All validation scores (for each folds) come between 3.18-3.80 \n#OOF score on train is 3.48\n#LB score 3.93\n#1h 35min 58s")


# In[91]:


results=pd.DataFrame({'key':test['key'],'fare_amount':test_pred_stacked_xgb})
results.to_csv('/kaggle/working/test_pred_stacked_xgb.csv',index=False)

results_train=pd.DataFrame({'Model_fare_amount':stacker_train_xgb[:,0]})
results_train.to_csv('/kaggle/working/stacker_train_xgb.csv',index=False)


# In[92]:


get_ipython().run_cell_magic('time', '', 'lgbm_params= {#"max_depth": 5,          #max depth for tree model\n              "learning_rate" : 0.02,\n    \'eval_metric\': \'rmse\', \n    \'objective\': \'regression\',\n              #"num_leaves": 25,        #max number of leaves in one tree\n              # \'feature_fraction\':0.6,  #LightGBM will randomly select part of features on each tree node\n               \'bagging_fraction\':0.8,    #randomly select part of data without resampling\n              # \'max_drop\': 5,         #used only in dart,max number of dropped trees during one boosting iteration\n               \'lambda_l1\': 5,\n               \'lambda_l2\': 0.5,\n              \'min_child_samples\':400,  #minimal number of data in one leaf\n                \'max_bin\':20, #max number of bins that feature values will be bucketed in. Higher value--> Overfitting\n                \'subsample\':0.6,  #randomly select part of data without resampling\n                \'colsample_bytree\':0.8, #same as feature_fraction\n               \'boosting_type\': \'gbdt\',   #options are dart,gbdt(gradientboosting decision trees), rf,dart,goss\n               \'task\': \'train\'}  #weight of labels with positive class\n\ntest_pred_stacked_lgbm,stacker_train_lgbm=\\\nmodelling_object.SingleLGBM_oof(params=lgbm_params,num_boost_round=1000,colnames=X_train.columns,importance_plot=True)\n#All validation scores (for each folds) come between 3.40-3.97\n#LB score around 3.35\n# time 4 minutes')


# In[93]:


results=pd.DataFrame({'key':test['key'],'fare_amount':test_pred_stacked_lgbm})
results.to_csv('/kaggle/working/test_pred_stacked_lgbm.csv',index=False)

results_train=pd.DataFrame({'Model_fare_amount':stacker_train_lgbm[:,0]})
results_train.to_csv('/kaggle/working/stacker_train_lgbm.csv',index=False)


# In[94]:


get_ipython().run_cell_magic('time', '', "#Catboost\nimport catboost\nfrom catboost import CatBoostRegressor\ncat_params= {\n    'iterations':1000,\n    'learning_rate':0.004,\n   'depth':5,\n    'eval_metric':'RMSE',\n    'colsample_bylevel':0.8,\n    'random_seed' : 2017,\n    'bagging_temperature' : 0.2,\n    'early_stopping_rounds':200\n} \ntest_pred_stacked_cat,stacker_train_cat=\\\nmodelling_object.SingleCatBoost_oof(params=cat_params)\n#All validation scores (for each folds) come around 3.72-4.24\n#LB score 3.72\n#Wall time: 10min 29s")


# In[95]:


results=pd.DataFrame({'key':test['key'],'fare_amount':test_pred_stacked_cat})
results.to_csv('/kaggle/working/test_pred_stacked_cat.csv',index=False)

results_train=pd.DataFrame({'Model_fare_amount':stacker_train_cat[:,0]})
results_train.to_csv('/kaggle/working/stacker_train_cat.csv',index=False)


# In[96]:


columns=['catboost','xgb','lgbm','rf']
train_pred_df_list=[stacker_train_cat,stacker_train_xgb, stacker_train_lgbm, stacker_train_rf]
test_pred_df_list=[test_pred_stacked_cat,test_pred_stacked_xgb,test_pred_stacked_lgbm,test_pred_stacked_rf]
lv1_train_df=pd.DataFrame(columns=columns)
lv1_test_df=pd.DataFrame(columns=columns)
for i in range(len(columns)):
    lv1_train_df[columns[i]]=train_pred_df_list[i][:,0]
    lv1_test_df[columns[i]]=test_pred_df_list[i]
    
lv1_train_df['Y']=y_train.values #add the dependendt variable to training


# In[97]:


lv1_train_df.describe()


# In[98]:


lv1_train_df.isnull().sum()


# In[99]:


#LGBM Level 2
l2_modelling_object = Modelling(X=lv1_train_df.drop('Y',axis=1).values, y=lv1_train_df['Y'].values,                                 test_X=lv1_test_df.values, folds=folds, N=5)

test_pred_stacked_lgbm_L2,stacker_train_lgbm_L2=l2_modelling_object.SingleLGBM_oof(params=lgbm_params,num_boost_round=10000,colnames=columns,importance_plot=True)


# In[100]:


results=pd.DataFrame({'key':test['key'],'fare_amount':test_pred_stacked_lgbm_L2})
results.to_csv('/kaggle/working/test_pred_stacked_lgbm_L2.csv',index=False)


# In[101]:


#XGB L2
test_pred_stacked_xgb_L2,stacker_train_xgb_L2=l2_modelling_object.SingleXGB_oof(params_for_xgb,1000)


# In[102]:


results=pd.DataFrame({'key':test['key'],'fare_amount':test_pred_stacked_xgb_L2})
results.to_csv('/kaggle/working/test_pred_stacked_xgb_L2_final.csv',index=False)
#from kaggle
# from IPython.display import FileLinks
# FileLinks('.')

