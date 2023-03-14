#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print("A")


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import sys


# In[ ]:


print('Loading Train.')
train = pd.read_csv("../input/train/train.csv")


# In[ ]:


train.head()


# In[ ]:


# load thus indexs
breeds = pd.read_csv("../input/breed_labels.csv")
colors = pd.read_csv("../input/color_labels.csv")
states = pd.read_csv("../input/state_labels.csv")


# In[ ]:


for x in np.unique(train[['Breed2','Breed1']].values.reshape(1,-1)):
    if x not in breeds.BreedID.values:
        print(x)
        
breeds = breeds.append({'BreedName' : 'no_other' , 'Type' : -1 , 'BreedID':0} , ignore_index=True)


# In[ ]:


breeds.head(2)


# In[ ]:


assert(breeds.groupby('BreedName').apply(lambda x:x.BreedID.nunique()==1.0).all())
assert(breeds.groupby('BreedID').apply(lambda x:x.BreedName.nunique()==1.0).all())


# In[ ]:


for x in np.unique(train[['Color1','Color2','Color3']].values.reshape(1,-1)):
    if x not in colors.ColorID.values:
        print(x)
        
colors = colors.append({'ColorName' : 'No_Color', 'ColorID':0} , ignore_index=True)


# In[ ]:


colors.head()


# In[ ]:


assert(colors.groupby('ColorName').apply(lambda x:x.ColorID.nunique()==1.0).all())
assert(colors.groupby('ColorID').apply(lambda x:x.ColorName.nunique()==1.0).all())


# In[ ]:


states.head()


# In[ ]:


# let see if there are missing values that may affect the merges with the indexs.
train.isnull().any()
# that ok for now, we would merge on it.


# In[ ]:


# we just seen that the colors,breeds and states indexer are NOT full.
#  so, we will merge with 'left'.
def mergeNdrop(train,breeds,colors,states):
    m_train = train.merge(breeds[['BreedName','BreedID']],left_on="Breed1",right_on="BreedID",how="left")
    m_train = m_train.merge(breeds[['BreedName','BreedID']],left_on="Breed2",right_on="BreedID",how="left")

    m_train = m_train.merge(colors,left_on="Color1",right_on="ColorID",how="left")
    m_train = m_train.merge(colors,left_on="Color2",right_on="ColorID",how="left")
    m_train = m_train.merge(colors,left_on="Color3",right_on="ColorID",how="left")

    m_train = m_train.merge(states,left_on="State",right_on="StateID",how="left")
    
    # let clean the duplicate keys.
    m_train=m_train.drop(columns=['ColorID','ColorID_x','ColorID_y','Color1','Color2','Color3'])
    m_train=m_train.drop(columns=['Breed1','Breed2','BreedID_x','BreedID_y'])
    m_train=m_train.drop(columns=['StateID','State'])
    return m_train


# In[ ]:


clean_train=mergeNdrop(train,breeds,colors,states)
clean_train.head()


# In[ ]:


# lets make sure we didn't did something wrong.
# if our inner merge was the right choice, the number of samples in the new frame suppose to stay the same.
print("Before cleaning= "+str(train.shape))
print("After cleaning= "+str(clean_train.shape))


# In[ ]:



#  we are going to use releative features so let split to train and test before.
X_train, X_test, _, _ = train_test_split(clean_train, [1]*len(clean_train), test_size=0.3, random_state=42)
#del train,clean_train


# In[ ]:


print("Train= "+str(X_train.shape))
print("Test= "+str(X_test.shape))


# In[ ]:


to_be_droped=['PetID']


# In[ ]:


sns.catplot(x="Type", y="AdoptionSpeed", data=X_train,
                height=6, kind="bar", palette="muted")


# In[ ]:


to_be_droped.append('Name')


# In[ ]:


sns.heatmap(X_train[["Age","AdoptionSpeed"]].corr(), annot=True, linewidths=.5)


# In[ ]:


def counter(x):
    result=2
    for i in x: 
        if ("no_other"==i):
            result =result -1
    return result


# In[ ]:


X_train.groupby("BreedName_x").AdoptionSpeed.agg(['count']).describe()
#before, let group all breeds that as single sample to 5 , call it 'OTHERS'.


# In[ ]:


# sc=X_train.groupby("BreedName_y").AdoptionSpeed.agg(['std','mean','median','count']).reset_index()
# X_train.merge(sc).head()


# In[ ]:


x=X_train.groupby("BreedName_x").AdoptionSpeed.agg(['count']).reset_index()
breed_x_list=x[x['count']>5].BreedName_x
 
x=X_train.groupby("BreedName_y").AdoptionSpeed.agg(['count']).reset_index()
breed_y_list=x[x['count']>5].BreedName_y


# In[ ]:


def replace_breed(_dataset,breed_x_list,breed_y_list):
    dataset=_dataset.copy()
    dataset.loc[~dataset.BreedName_x.isin(breed_x_list),"BreedName_x"]="OTHERS"
    dataset.loc[~dataset.BreedName_y.isin(breed_y_list),"BreedName_y"]="OTHERS"
    return dataset


# In[ ]:


X_train=replace_breed(X_train,breed_x_list,breed_y_list)


# In[ ]:


X_train[X_train.BreedName_y=='OTHERS'].AdoptionSpeed.agg(['count'])


# In[ ]:


def create_bread_encoding(_dataset):
    dataset=_dataset.copy()
    breed_y=dataset.groupby("BreedName_y").AdoptionSpeed.agg(['std','mean','median','count']).reset_index()
    breed_x=dataset.groupby("BreedName_x").AdoptionSpeed.agg(['std','mean','median','count']).reset_index()
    return breed_y,breed_x

def EcodeBreed(_dataset,breed_y,breed_x):
    dataset=_dataset.copy()
    dataset.loc['diff_breed_count',:]=dataset[["BreedName_y","BreedName_x"]].apply(lambda x: counter(x),axis=1)
    return dataset.merge(breed_y,on='BreedName_y').merge(breed_x,on='BreedName_x').drop(columns=["BreedName_x","BreedName_y"])

breed_y,breed_x=create_bread_encoding(X_train)
X_train=EcodeBreed(X_train,breed_y,breed_x)


# In[ ]:


g = sns.FacetGrid(X_train,  col="Gender", margin_titles=True)
g.map(plt.hist, "AdoptionSpeed", color="steelblue")
# it's seems not importent. 


# In[ ]:


def handle_gender(_dataset):
    dataset=_dataset.copy()
    dataset['ismale'], dataset['isfemale'] = zip(*dataset.Gender.apply(lambda x : [1,1] if x==3 else [1,0] if x==1 else  [0,1]))
    return dataset.drop(columns=['Gender'])

X_train=handle_gender(X_train)


# In[ ]:


g = sns.FacetGrid(X_train,  col="ColorName",row="ColorName_x", margin_titles=True)
g.map(plt.hist, "AdoptionSpeed", color="steelblue")


# In[ ]:


def ColorToOneHot(_dataset):
    dataset=_dataset.copy()
    
    dataset.loc[dataset.ColorName_y=='No_Color','ColorName_y']=dataset.loc[dataset.ColorName_y=='No_Color','ColorName_x']
    dataset.loc[dataset.ColorName=='No_Color','ColorName']=dataset.loc[dataset.ColorName=='No_Color','ColorName_x']
    
    dummy=pd.get_dummies(dataset['ColorName'])
    dummy1=pd.get_dummies(dataset['ColorName_y'])
    dummy2=pd.get_dummies(dataset['ColorName_x'])
    
    coloers=dummy1.add(dummy, fill_value=0).add(dummy2, fill_value=0).applymap(lambda x: 1.0 if x>1.0 else x)

    dataset=pd.concat([coloers, dataset], axis=1, sort=False)
    dataset=dataset.drop(columns=["ColorName","ColorName_y","ColorName_x"])
    return dataset


# In[ ]:


X_train=ColorToOneHot(X_train)


# In[ ]:


X_train.groupby("MaturitySize").MaturitySize.count().plot(kind='pie')


# In[ ]:


def test_fix(_dataset,col='MaturitySize',def_map={0:2}):
    dataset=_dataset.copy()
    dataset[col]=dataset[col].apply(lambda x: def_map[x] if x in def_map.keys() else x)
    return dataset


# In[ ]:


X_train=test_fix(X_train)


# In[ ]:


X_train.groupby("FurLength").FurLength.count().plot(kind='pie')


# In[ ]:


X_train=test_fix(X_train,col='FurLength',def_map={0:1})


# In[ ]:


X_train.groupby("Vaccinated").Vaccinated.count().plot(kind='pie')


# In[ ]:


X_train=test_fix(X_train,col='Vaccinated',def_map={1:1,2:-1,3:0})


# In[ ]:


X_train.groupby("Dewormed").Dewormed.count().plot(kind='pie') 


# In[ ]:


X_train=test_fix(X_train,col='Dewormed',def_map={1:1,2:-1,3:0})


# In[ ]:


X_train.groupby("Sterilized").Sterilized.count().plot(kind='pie') 


# In[ ]:


X_train=test_fix(X_train,col='Sterilized',def_map={1:1,2:-1,3:-1})


# In[ ]:


X_train.groupby("Health").Health.count().plot(kind='pie')  


# In[ ]:


X_train=test_fix(X_train,col='Sterilized',def_map={0:1})


# In[ ]:


sns.distplot(X_train[X_train.Quantity==1].AdoptionSpeed, hist=True, color="r")


# In[ ]:


sns.distplot(X_train[X_train.Quantity>1].AdoptionSpeed, hist=True, color="r")


# In[ ]:


# later add over sampling acording to the confusion matrix
to_be_droped.append("Quantity")


# In[ ]:


sns.lineplot(x="Fee", y="AdoptionSpeed",
             data=X_train)


# In[ ]:


plt.figure(figsize=(15,6))
sns.violinplot(x='StateName', y='AdoptionSpeed', data=X_train)


# In[ ]:


plt.figure(figsize=(15,6))
X_train.groupby("StateName").AdoptionSpeed.count().plot(kind='bar')
# it's seems that pets was adopted fast becuase of the number of pet is to low.
# let validate this.


# In[ ]:



x=X_train.groupby("StateName").AdoptionSpeed.agg(['count']).reset_index()
state_list=x[x['count']>15].StateName

def replace_State(_dataset,state_list):
    dataset=_dataset.copy()
    dataset.loc[~dataset.StateName.isin(state_list),"StateName"]="random_state"
    return dataset


# In[ ]:


# that good enough. let's add this feature.
def create_state(_dataset):
    dataset=_dataset.copy()
    return dataset.groupby("StateName").AdoptionSpeed.agg(['std','mean','median','count']).reset_index()

def state_handle(_dataset,state_stats):
    dataset=_dataset.copy()
    dataset["avalibe_pets_in_state"]=dataset.apply(lambda x: len(dataset[dataset.StateName==x.StateName]),axis=1)
    #state_stats=dataset.groupby("StateName").AdoptionSpeed.agg(['std','mean','median','count']).reset_index()
    return dataset.merge(state_stats,on='StateName',suffixes=('_original', '_state')).drop(columns=["StateName"])


# In[ ]:


X_train=replace_State(X_train,state_list)
state_stats=create_state(X_train)
X_train=state_handle(X_train,state_stats)


# In[ ]:


Preson_candm=X_train.groupby("RescuerID").AdoptionSpeed.agg(['count','mean'])
Preson_candm=Preson_candm.reset_index()


# In[ ]:


plt.figure(figsize=(15,6))
sns.lmplot(x='mean', y='count', data=Preson_candm,fit_reg=False) 


# In[ ]:


x=X_train.groupby("RescuerID").AdoptionSpeed.agg(['count']).reset_index()
as_stat=x[x['count']>3].RescuerID

def replace_Rescuer(_dataset,as_stat):
    dataset=_dataset.copy()
    dataset.loc[~dataset.RescuerID.isin(as_stat),"RescuerID"]="random_rescuer"
    return dataset

X_train=replace_Rescuer(X_train,as_stat)


# In[ ]:


# it's look like Rescuers that post above about 50 post are not likly sell the pet fast.
# that good enough. let's add this feature.
def create_resuce(_dataset):
    dataset=_dataset.copy()
    return dataset.groupby("RescuerID").AdoptionSpeed.agg(['std','mean','count']).reset_index()


def Rescuers_handle(_dataset,Rescuer_stats):
    dataset=_dataset.copy()
    dataset["avalibe_pets_in_preson"]=dataset.apply(lambda x: len(dataset[dataset.RescuerID==x.RescuerID]),axis=1)
    return dataset.merge(Rescuer_stats,on='RescuerID',suffixes=('_original', '_Rescue')).drop(columns=["RescuerID"])


# In[ ]:


Rescuer_stats=create_resuce(X_train)
X_train=Rescuers_handle(X_train,Rescuer_stats)


# In[ ]:


to_be_droped.append("Description")


# In[ ]:


assert(X_train.isna().any().sum()==2)


# In[ ]:


X_train.head()


# In[ ]:


X_train=X_train.drop(columns=to_be_droped)


# In[ ]:


def splitLabelSample(dataset):
    y=dataset['AdoptionSpeed']
    x=dataset.drop(columns=['AdoptionSpeed'])
    return x,y


# In[ ]:


x_train,y_train=splitLabelSample(X_train) 
x_test,y_test=splitLabelSample(X_test)


# In[ ]:


x_test.head()


# In[ ]:


print("Train size is="+str(x_train.shape))


# In[ ]:


def full_pipeline(_dataset,breed_x_list,breed_y_list,to_be_droped,breed_y,breed_x,Rescuer_stats,state_stats,as_stat,state_list):
    dataset=_dataset.copy()
    
    dataset=replace_breed(dataset,breed_x_list,breed_y_list)
    dataset=EcodeBreed(dataset,breed_y,breed_x)
    dataset=handle_gender(dataset)
    dataset=ColorToOneHot(dataset)
    dataset=test_fix(dataset)
    dataset=test_fix(dataset,col='FurLength',def_map={0:1})
    dataset=test_fix(dataset,col='Vaccinated',def_map={1:1,2:-1,3:0})
    dataset=test_fix(dataset,col='Dewormed',def_map={1:1,2:-1,3:0})
    dataset=test_fix(dataset,col='Sterilized',def_map={1:1,2:-1,3:-1})
    dataset=test_fix(dataset,col='Sterilized',def_map={0:1})
   # print(str(state_list))
    dataset=replace_State(dataset,state_list)
    dataset=state_handle(dataset,state_stats)
    dataset=replace_Rescuer(dataset,as_stat)
    dataset=Rescuers_handle(dataset,Rescuer_stats)
    dataset=dataset.drop(columns=to_be_droped)
    return dataset


# In[ ]:


x_test=full_pipeline(x_test,breed_x_list,breed_y_list,to_be_droped,breed_y,breed_x,Rescuer_stats,state_stats,as_stat,state_list)


# In[ ]:


x_test.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
scaler.fit(x_train.values)
Xn_train=scaler.transform(x_train.values)
Xn_test=scaler.transform(x_test.values)


# In[ ]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='all')
Xn_train, y_train = smote.fit_sample(Xn_train, y_train)


# In[ ]:


assert(Xn_train.shape[1]==Xn_test.shape[1])


# In[ ]:


np.isnan(y_train).any()


# In[ ]:


X_train.isna().any().any()


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
parameters = {
    "loss":["deviance","exponential"],
    "learning_rate": [0.01, 0.05, 0.1, 0.2,0.5],
  #  "min_samples_split": np.linspace(0.1, 0.5, 12),
  #  "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8,14,18],
    "max_features":["log2","sqrt"],
    "subsample":[0.5,0.6,0.9, 1.0],
    "n_estimators":[5,10,13,16]
    }

clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=10)
clf.fit(Xn_train,y_train)


# In[ ]:


clf.score(Xn_test,y_test)


# In[ ]:


import pickle
bost=clf.best_estimator_
cm=confusion_matrix(y_test, bost.predict(Xn_test))
cm =cm / cm.astype(np.float).sum(axis=1)
heatmap = sns.heatmap(cm)


# In[ ]:



param_grid = {
    'max_depth': [4, 6,8,10],
    'n_estimators': [100, 200, 300]
}
# Create a based model
rf = RandomForestClassifier(class_weight ='balanced')
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3,  n_jobs=10, verbose = 2)
grid_search.fit(Xn_train, y_train)


# In[ ]:


rnd=grid_search.best_estimator_
rnd.score(Xn_test,y_test)


# In[ ]:


import pickle
cm=confusion_matrix(y_test, rnd.predict(Xn_test))
cm =cm / cm.astype(np.float).sum(axis=1)
heatmap = sns.heatmap(cm)


# In[ ]:


from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators=[('gb', bost), ('rf', rnd)], voting='hard')
eclf1 = eclf1.fit(Xn_train, y_train)


# In[ ]:


eclf1.score(Xn_test,y_test)


# In[ ]:


cm=confusion_matrix(y_test, eclf1.predict(Xn_test))
cm =cm / cm.astype(np.float).sum(axis=1)
heatmap = sns.heatmap(cm)


# In[ ]:


to_be_droped_copy=to_be_droped.copy()

to_be_droped_copy.remove('PetID')


# In[ ]:


to_be_droped


# In[ ]:


#  get the test set ready.
test_df = pd.read_csv("../input/test/test.csv")
test_df=mergeNdrop(test_df,breeds,colors,states)
test=full_pipeline(test_df,breed_x_list,breed_y_list,to_be_droped_copy,breed_y,breed_x,Rescuer_stats,state_stats,as_stat,state_list)


# In[ ]:


pred = eclf1.predict(scaler.transform(test.drop(columns=["PetID"]).values))

prediction_df = pd.DataFrame({'PetID' : test['PetID'],
                              'AdoptionSpeed' : pred.astype(int)})
prediction_df.to_csv("submission.csv", index=False)

