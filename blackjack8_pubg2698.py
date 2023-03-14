#!/usr/bin/env python
# coding: utf-8



#importing Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt




df_train = pd.read_csv("../input/pubg-finish-placement-prediction/train_V2.csv")




df_train.head()




df_train.shape




df_train.info()




df_train.isna().sum()




df_train = df_train.dropna()
df_train.shape




df_train.describe()




def show_countplot(col):
    plt.figure(figsize=(14,4))
    sns.countplot(data = df_train,x=col).set_title(col)
    plt.show()
def show_distplot(col):
    plt.figure(figsize=(14,4))
    sns.distplot(df_train[col],bins=50)
    plt.show()




show_countplot('kills')




df_train[df_train['kills']>=45]




df_train['headshot_rate'] = df_train['headshotKills'] / df_train['kills']
df_train['headshot_rate'] = df_train['headshot_rate'].fillna(0)
show_distplot('headshot_rate')




df_train[(df_train['headshot_rate']==1) & (df_train['kills'] >=10)]




df_train['longestKill'].describe()




show_distplot('longestKill')




df_train[df_train['longestKill']>=1050]




show_countplot('teamKills')




df_train[df_train['teamKills']>=5]




df_train[['walkDistance', 'rideDistance', 'swimDistance']].describe()




show_distplot('walkDistance')




df_train[df_train['walkDistance']>=15000]




show_distplot('rideDistance')




df_train[df_train['rideDistance']>=35000]




show_countplot('weaponsAcquired')




df_train[df_train['weaponsAcquired']>=60]




show_countplot('heals')




df_train[(df_train['heals']>60)]




df_train.drop(df_train[(df_train['kills']>=45) | (df_train['headshot_rate']==1) & (df_train['kills'] >=10)             | (df_train['longestKill']>=1050)|(df_train['teamKills']>=5) | (df_train['walkDistance']>=15000)            |(df_train['rideDistance']>=35000)|(df_train['weaponsAcquired']>=60)            |(df_train['heals']>60)  ].index,inplace=True)




df_train.shape #we removed 162 rows




print("The average person kills {:.4f} players, 99% of people have kill {} or less,while the most kill recorded is: {}".format(df_train['kills'].mean(),df_train['kills'].quantile(0.99)                                                     ,df_train['kills'].max()))




data = df_train.copy()
data.loc[data['kills'] > data['kills'].quantile(0.99)] = '8+'
plt.figure(figsize=(15,10))
sns.countplot(data['kills'].astype('str').sort_values())
plt.title("Kill Count",fontsize=15)
plt.show()




whos




del data




data = df_train[['kills','damageDealt']].copy()
data = data[data['kills']==0]
plt.figure(figsize=(15,10))
plt.title("Damage Dealt by 0 killers",fontsize=15)
sns.distplot(data['damageDealt'])
plt.show()
del data




print("{} players ({:.4f}%) have won without single kill!! ".format(len(df_train[(df_train['kills']==0)                            &(df_train['winPlacePerc']==1)]),100*len(df_train[(df_train['kills']==0)                            &(df_train['winPlacePerc']==1)])/len(df_train)))
print("{} players ({:.4f}%) have won without giving Zero Damage!! ".format(len(df_train[(df_train['damageDealt']==0)                            &(df_train['winPlacePerc']==1)]),100*len(df_train[(df_train['damageDealt']==0)                            &(df_train['winPlacePerc']==1)])/len(df_train)))




sns.jointplot(x="winPlacePerc", y="kills", data=df_train, height=10, ratio=3, color="r")
plt.show()




16666/len(df_train)




print("The average person walks for {:.1f}m, 99% of people have walked {}m or less, while the marathoner champion walked for {}m.".format(df_train['walkDistance'].mean(), df_train['walkDistance'].quantile(0.99), df_train['walkDistance'].max()))




show_distplot('walkDistance')




sns.jointplot(x="winPlacePerc", y="walkDistance",  data=df_train, height=10, ratio=3, color="lime")
plt.show()




sns.jointplot(x="winPlacePerc", y="rideDistance", data=df_train, height=10, ratio=3, color="y")
plt.show()




data = df_train.copy()
data = data[df_train['swimDistance'] < df_train['swimDistance'].quantile(0.95)]
plt.figure(figsize=(15,10))
plt.title("Swim Distance Distribution",fontsize=15)
sns.distplot(data['swimDistance'])
plt.show()
del data




print("The average person uses {:.1f} heal items, 99% of people use {} or less, while the doctor used {}.".format(df_train['heals'].mean(), df_train['heals'].quantile(0.99), df_train['heals'].max()))
print("The average person uses {:.1f} boost items, 99% of people use {} or less, while the doctor used {}.".format(df_train['boosts'].mean(), df_train['boosts'].quantile(0.99), df_train['boosts'].max()))




data = df_train.copy()
data = data[data['heals'] < data['heals'].quantile(0.99)]
data = data[data['boosts'] < data['boosts'].quantile(0.99)]




plt.figure(figsize=(12,5))
sns.pointplot(x='heals',y='winPlacePerc',data=data,color='lime',alpha=0.8)
sns.pointplot(x='boosts',y='winPlacePerc',data=data,color='blue',alpha=0.8)
plt.xlabel('Number of heal/boost items',fontsize = 15,color='blue')
plt.ylabel('Win Percentage',fontsize = 15,color='blue')
plt.title('Heals vs Boosts',fontsize = 20,color='blue')
plt.show()
del data




sns.jointplot(x="winPlacePerc", y="heals", data=df_train, height=10, ratio=3, color="lime")
plt.show()




solo = df_train[df_train['numGroups']>50]
duo = df_train[(df_train['numGroups']>25)&(df_train['numGroups']<=50)]
squad = df_train[(df_train['numGroups']<=25)]

print("{} are solo,{} are duo,{} are squads".format(len(solo['kills']),len(duo['kills']),len(squad['kills'])))




plt.figure(figsize=(12,5))
sns.pointplot(x="kills",y='winPlacePerc',data=solo,color="black",alpha=0.8)
sns.pointplot(x="kills",y='winPlacePerc',data=duo,color="red",alpha=0.8)
sns.pointplot(x="kills",y='winPlacePerc',data=squad,color="green",alpha=0.8)
plt.xlabel('Number of kills',fontsize = 15,color='blue')
plt.ylabel('Win Percentage',fontsize = 15,color='blue')
plt.title('Solo vs Duo vs Squad Kills',fontsize = 20,color='blue')
plt.grid()
plt.show()




f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='DBNOs',y='winPlacePerc',data=duo,color='#CC0000',alpha=0.8)
sns.pointplot(x='DBNOs',y='winPlacePerc',data=squad,color='#3399FF',alpha=0.8)
sns.pointplot(x='assists',y='winPlacePerc',data=duo,color='#FF6666',alpha=0.8)
sns.pointplot(x='assists',y='winPlacePerc',data=squad,color='#CCE5FF',alpha=0.8)
sns.pointplot(x='revives',y='winPlacePerc',data=duo,color='#660000',alpha=0.8)
sns.pointplot(x='revives',y='winPlacePerc',data=squad,color='#000066',alpha=0.8)
plt.text(14,0.5,'Duos - Assists',color='#FF6666',fontsize = 17,style = 'italic')
plt.text(14,0.45,'Duos - DBNOs',color='#CC0000',fontsize = 17,style = 'italic')
plt.text(14,0.4,'Duos - Revives',color='#660000',fontsize = 17,style = 'italic')
plt.text(14,0.35,'Squads - Assists',color='#CCE5FF',fontsize = 17,style = 'italic')
plt.text(14,0.3,'Squads - DBNOs',color='#3399FF',fontsize = 17,style = 'italic')
plt.text(14,0.25,'Squads - Revives',color='#000066',fontsize = 17,style = 'italic')
plt.xlabel('Number of DBNOs/Assits/Revives',fontsize = 15,color='blue')
plt.ylabel('Win Percentage',fontsize = 15,color='blue')
plt.title('Duo vs Squad DBNOs, Assists, and Revives',fontsize = 20,color='blue')
plt.grid()
plt.show()




f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(df_train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()




k = 5 #number of variables for heatmap
f,ax = plt.subplots(figsize=(11, 11))
cols = df_train.corr().nlargest(k, 'winPlacePerc')['winPlacePerc'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True,                  fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()




# sns.set()
# cols = ['winPlacePerc', 'walkDistance', 'boosts', 'weaponsAcquired', 'damageDealt', 'killPlace']
# sns.pairplot(df_train[cols], size = 2.5)
# plt.show()




# Since there is no var which tells us no. of player joined. So create one
df_train['playersJoined'] = df_train.groupby('matchId')['matchId'].transform('count')




plt.figure(figsize=(17,4))
sns.countplot(df_train[df_train['playersJoined']>49]['playersJoined'])
plt.show()




df_train['killsNorm'] = df_train['kills']*((100-df_train['playersJoined'])/100 + 1)
df_train['damageDealtNorm'] = df_train['damageDealt']*((100-df_train['playersJoined'])/100 + 1)
df_train[['playersJoined', 'kills', 'killsNorm', 'damageDealt', 'damageDealtNorm']][5:8]




df_train['healsAndBoosts'] = df_train['heals']+df_train['boosts']
df_train['totalDistance'] = df_train['walkDistance']+df_train['rideDistance']+df_train['swimDistance']




df_train['boostsPerWalkDistance'] = df_train['boosts']/(df_train['walkDistance']+1) #The +1 is to avoid infinity
df_train['boostsPerWalkDistance'].fillna(0, inplace=True)
df_train['healsPerWalkDistance'] = df_train['heals']/(df_train['walkDistance']+1) #The +1 is to avoid infinity
df_train['healsPerWalkDistance'].fillna(0, inplace=True)
df_train['healsAndBoostsPerWalkDistance'] = df_train['healsAndBoosts']/(df_train['walkDistance']+1) #The +1 is to avoid infinity.
df_train['healsAndBoostsPerWalkDistance'].fillna(0, inplace=True)
df_train[['walkDistance', 'boosts', 'boostsPerWalkDistance' ,'heals',  'healsPerWalkDistance', 'healsAndBoosts', 'healsAndBoostsPerWalkDistance']][40:45]




df_train['killsPerWalkDistance'] = df_train['kills']/(df_train['walkDistance']+1) #The +1 is to avoid infinity
df_train['killsPerWalkDistance'].fillna(0, inplace=True)
df_train[['kills', 'walkDistance', 'rideDistance', 'killsPerWalkDistance', 'winPlacePerc']].sort_values(by='killsPerWalkDistance').tail(10)




df_train['team'] = [1 if i>50 else 2 if (i>25 & i<=50) else 4 for i in df_train['numGroups']]




X = df_train.drop(columns=["winPlacePerc",'Id','groupId','matchId','matchType'],axis=1)
y = df_train['winPlacePerc']




from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=3)
reg = RandomForestRegressor(max_depth = 7,n_jobs  = -1)




X.to_csv("X.csv")
y.to_csv("y.csv")




reg.fit(X_train,y_train)




import pickle
pickle.dump(reg,open("model",'wb'))





test = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')
score = reg.score(X_test,y_test)




test.head()




test = test.drop(columns = ['Id','groupId','matchId','matchType'],axis = 1)
pred = reg.predict(test)




print("Model Accuracy:{}".format(score))

