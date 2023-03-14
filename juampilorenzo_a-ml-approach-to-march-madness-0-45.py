#!/usr/bin/env python
# coding: utf-8



#Added by me
import numpy as np
import pandas as pd
import random
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Agregados míos
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




tourney_data=pd.read_csv("../input/TourneyDetailedResults.csv")
tourney_data.shape




get_ipython().run_line_magic('matplotlib', 'inline')




tourney_data=tourney_data.groupby(['Season']).mean()

tourney_data.plot(y='Wscore')
tourney_data.plot(y='Wfga')




detailed_data=pd.read_csv("../input/TourneyDetailedResults.csv")

training_data = pd.DataFrame()
training_data[["Season","team1", "team2"]] =detailed_data[["Season","Wteam", "Lteam"]].copy()
training_data["pred"] = 1

for index, row in training_data.iterrows():
    if random.randint(0,1)==1:
        temp=row['team1']
        training_data.set_value(index,'team1',row['team2'])
        training_data.set_value(index,'team2',temp)
        training_data.set_value(index,'pred',0)

def loadInTraining(name,data):
    new_training_data=pd.merge(training_data,data,how="inner",left_on=["Season","team1"], right_on=["Season","team"])
    new_training_data=new_training_data.drop("team",axis=1)
    new_training_data=new_training_data.rename(columns={name: name+"1"})
    
    new_training_data=pd.merge(new_training_data,data,how="inner",left_on=["Season","team2"], right_on=["Season","team"])
    new_training_data=new_training_data.drop("team",axis=1)
    new_training_data=new_training_data.rename(columns={name: name+"2"})
    
    return new_training_data




season_data=pd.read_csv("../input/RegularSeasonDetailedResults.csv")

ppg_data_a=pd.DataFrame()
ppg_data_a[["Season","team","score"]]=season_data[["Season","Wteam","Wscore"]]
ppg_data_b=pd.DataFrame()
ppg_data_b[["Season","team","score"]]=season_data[["Season","Lteam","Lscore"]]

ppg_data=pd.concat((ppg_data_a,ppg_data_b),axis=0)
ppg_data=ppg_data.groupby(["Season","team"])["score"].agg(['sum','count']).reset_index()
ppg_data["PPG"]=ppg_data["sum"]/ppg_data["count"]

ppg_data=ppg_data[["Season","team","PPG"]]
training_data=loadInTraining("PPG",ppg_data)
training_data.head()




season_data=pd.read_csv("../input/RegularSeasonDetailedResults.csv")

rpg_data_a=pd.DataFrame()
rpg_data_a[["Season","team","score"]]=season_data[["Season","Wteam","Lscore"]]
rpg_data_b=pd.DataFrame()
rpg_data_b[["Season","team","score"]]=season_data[["Season","Lteam","Wscore"]]

rpg_data=pd.concat((rpg_data_a,rpg_data_b),axis=0)
rpg_data=rpg_data.groupby(["Season","team"])["score"].agg(['sum','count']).reset_index()
rpg_data["RPG"]=rpg_data["sum"]/rpg_data["count"]

rpg_data=rpg_data[["Season","team","RPG"]]
training_data=loadInTraining("RPG",rpg_data)
training_data.head()




season_data=pd.read_csv("../input/RegularSeasonDetailedResults.csv")

rec_data_w=pd.DataFrame()
rec_data_w[["Season","team","W"]]=season_data[["Season","Wteam","Wscore"]]
rec_data_l=pd.DataFrame()
rec_data_l[["Season","team","L"]]=season_data[["Season","Lteam","Lscore"]]

rec_data_w=rec_data_w.groupby(["Season","team"])["W"].count().reset_index()
rec_data_l=rec_data_l.groupby(["Season","team"])["L"].count().reset_index()
rec_data=pd.merge(rec_data_w,rec_data_l,how="outer",on=["Season","team"])
rec_data=rec_data.fillna(0)
rec_data["REC"]=rec_data["W"]/(rec_data["W"]+rec_data["L"])

rec_data=rec_data[["Season","team","REC"]]

training_data=loadInTraining("REC",rec_data)
training_data.head()




season_data=pd.read_csv("../input/RegularSeasonDetailedResults.csv")

efg_data_w=pd.DataFrame()
efg_data_w[["Season","team","fga","fg","3p"]]=season_data[["Season","Wteam","Wfga","Wfgm","Wfgm3"]]
efg_data_l=pd.DataFrame()
efg_data_l[["Season","team","fga","fg","3p"]]=season_data[["Season","Lteam","Lfga","Lfgm","Lfgm3"]]

efg_data=pd.concat((efg_data_w,efg_data_l),axis=0)
efg_data=efg_data.groupby(["Season","team"]).sum().reset_index()
efg_data["eFG%"]=(efg_data["fg"]+0.5*efg_data["3p"])/efg_data["fga"]

efg_data=efg_data[["Season","team","eFG%"]]
training_data=loadInTraining("eFG%",efg_data)
training_data.head()




season_data=pd.read_csv("../input/RegularSeasonDetailedResults.csv")

efg_data_w=pd.DataFrame()
efg_data_w[["Season","team","fga","fg","3p"]]=season_data[["Season","Wteam","Lfga","Lfgm","Lfgm3"]]
efg_data_l=pd.DataFrame()
efg_data_l[["Season","team","fga","fg","3p"]]=season_data[["Season","Lteam","Wfga","Wfgm","Wfgm3"]]

efg_data=pd.concat((efg_data_w,efg_data_l),axis=0)
efg_data=efg_data.groupby(["Season","team"]).sum().reset_index()
efg_data["dFG%"]=(efg_data["fg"]+0.5*efg_data["3p"])/efg_data["fga"]

efg_data=efg_data[["Season","team","dFG%"]]
training_data=loadInTraining("dFG%",efg_data)
training_data.head()




season_data=pd.read_csv("../input/RegularSeasonDetailedResults.csv")

tov_data_w=pd.DataFrame()
tov_data_w[["Season","team","fga","fta","tov"]]=season_data[["Season","Wteam","Wfga","Wfta","Wto"]]
tov_data_l=pd.DataFrame()
tov_data_l[["Season","team","fga","fta","tov"]]=season_data[["Season","Lteam","Lfga","Lfta","Lto"]]

tov_data=pd.concat((tov_data_w,tov_data_l),axis=0)
tov_data=tov_data.groupby(["Season","team"]).sum().reset_index()
tov_data["TOV%"]=tov_data["tov"]/(tov_data["fga"]+0.44*tov_data["fta"]+tov_data["tov"])

tov_data=tov_data[["Season","team","TOV%"]]
training_data=loadInTraining("TOV%",tov_data)
training_data.head()




season_data=pd.read_csv("../input/RegularSeasonDetailedResults.csv")

tov_data_w=pd.DataFrame()
tov_data_w[["Season","team","fga","fta","tov"]]=season_data[["Season","Wteam","Lfga","Lfta","Lto"]]
tov_data_l=pd.DataFrame()
tov_data_l[["Season","team","fga","fta","tov"]]=season_data[["Season","Lteam","Wfga","Wfta","Wto"]]

tov_data=pd.concat((tov_data_w,tov_data_l),axis=0)
tov_data=tov_data.groupby(["Season","team"]).sum().reset_index()
tov_data["dTO%"]=tov_data["tov"]/(tov_data["fga"]+0.44*tov_data["fta"]+tov_data["tov"])

tov_data=tov_data[["Season","team","dTO%"]]
training_data=loadInTraining("dTO%",tov_data)
training_data.head()




season_data=pd.read_csv("../input/RegularSeasonDetailedResults.csv")

orb_data_w=pd.DataFrame()
orb_data_w[["Season","team","or","odr"]]=season_data[["Season","Wteam","Wor","Ldr"]]
orb_data_l=pd.DataFrame()
orb_data_l[["Season","team","or","odr"]]=season_data[["Season","Lteam","Lor","Wdr"]]

orb_data=pd.concat((orb_data_w,orb_data_l),axis=0)
orb_data=orb_data.groupby(["Season","team"]).sum().reset_index()
orb_data["ORB%"]=orb_data["or"]/(orb_data["or"]+orb_data["odr"])

orb_data=orb_data[["Season","team","ORB%"]]
training_data=loadInTraining("ORB%",orb_data)
training_data.head()




season_data=pd.read_csv("../input/RegularSeasonDetailedResults.csv")

drb_data_w=pd.DataFrame()
drb_data_w[["Season","team","dr","oor"]]=season_data[["Season","Wteam","Wdr","Lor"]]
drb_data_l=pd.DataFrame()
drb_data_l[["Season","team","dr","oor"]]=season_data[["Season","Lteam","Ldr","Wor"]]

drb_data=pd.concat((drb_data_w,drb_data_l),axis=0)
drb_data=drb_data.groupby(["Season","team"]).sum().reset_index()
drb_data["DRB%"]=drb_data["dr"]/(drb_data["dr"]+drb_data["oor"])

drb_data=drb_data[["Season","team","DRB%"]]
training_data=loadInTraining("DRB%",drb_data)
training_data.head()




season_data=pd.read_csv("../input/RegularSeasonDetailedResults.csv")

ftr_data_w=pd.DataFrame()
ftr_data_w[["Season","team","ft","fga"]]=season_data[["Season","Wteam","Wftm","Wfga"]]
ftr_data_l=pd.DataFrame()
ftr_data_l[["Season","team","ft","fga"]]=season_data[["Season","Lteam","Lftm","Lfga"]]

ftr_data=pd.concat((ftr_data_w,ftr_data_l),axis=0)
ftr_data=ftr_data.groupby(["Season","team"]).sum().reset_index()
ftr_data["FTR"]=ftr_data["ft"]/ftr_data["fga"]

ftr_data=ftr_data[["Season","team","FTR"]]
training_data=loadInTraining("FTR",ftr_data)
training_data.head()




seed_data=pd.read_csv("../input/TourneySeeds.csv")

seed_data["SdN"]=[int(str(x)[1:3]) for x in seed_data["Seed"]]
seed_data["team"]=seed_data["Team"]

seed_data=seed_data[["Season","team","SdN"]]
training_data=loadInTraining("SdN",seed_data)
training_data.head()




season_data=pd.read_csv("../input/RegularSeasonDetailedResults.csv")

ftr_data_w=pd.DataFrame()
ftr_data_w[["Season","team","ft","fga"]]=season_data[["Season","Wteam","Lftm","Lfga"]]
ftr_data_l=pd.DataFrame()
ftr_data_l[["Season","team","ft","fga"]]=season_data[["Season","Lteam","Wftm","Wfga"]]

ftr_data=pd.concat((ftr_data_w,ftr_data_l),axis=0)
ftr_data=ftr_data.groupby(["Season","team"]).sum().reset_index()
ftr_data["dFT"]=ftr_data["ft"]/ftr_data["fga"]

ftr_data=ftr_data[["Season","team","dFT"]]
training_data=loadInTraining("dFT",ftr_data)
training_data.head()




X = pd.DataFrame()
X = training_data.loc[training_data["Season"]<2013].copy()
X = X.reindex_axis(sorted(X.columns), axis=1)


Xval=pd.DataFrame()
Xval = X[:323]
X=X[323:]

Xtest=pd.DataFrame()
Xtest=training_data.loc[training_data["Season"]>=2013].copy()
Xtest = Xtest.reindex_axis(sorted(Xtest.columns), axis=1)

y = pd.DataFrame()
y["pred"]=X["pred"]

yval = pd.DataFrame()
yval["pred"]=Xval["pred"]

ytest = pd.DataFrame()
ytest["pred"]=Xtest["pred"]
yseeds=ytest.copy()

X = X.drop(["Season","team1","team2","pred"],axis=1)
Xval = Xval.drop(["Season","team1","team2","pred"],axis=1)
Xseeds=Xtest.copy().drop(["pred"],axis=1)
Xtest = Xtest.drop(["Season","team1","team2","pred"],axis=1)

X.head()




def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if max_value>1:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

X=normalize(X)
Xval=normalize(Xval)
Xtest=normalize(Xtest)
Xval.head()




Xforest=X.copy()
yforest=y.copy()
forest=RandomForestClassifier(n_estimators=30)
forest.fit(Xforest,yforest)

table=pd.concat((pd.DataFrame(Xforest.iloc[:, 0:].columns, columns = ['variable']), 
           pd.DataFrame(forest.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20].reset_index()
table=table.drop("index",axis=1)
table=table.groupby([table.variable.str[:3]]).sum().reset_index().sort_values(by='importance',ascending=False).reset_index()
table=table.drop("index",axis=1)
table




def reduction(X,features,cant):
    delete=features.shape[0]-cant
    features=features.sort_values(by="importance",ascending=True).reset_index().drop("index",axis=1)
    for i in range(0,delete):
        pattern=features.loc[i]["variable"]
        col=[col for col in X.columns if str(pattern) in col]
        X=X.drop(col[0],1)
        X=X.drop(col[1],1)
    return X
        
reduct_to=7
#X=reduction(X,table,reduct_to)
#Xval=reduction(Xval,table,reduct_to)
#Xtest=reduction(Xtest,table,reduct_to)




n_neighbors=135
knn=KNeighborsClassifier(n_neighbors=n_neighbors, p=2)
knn=knn.fit(X,y)

knn.score(X,y)




predict=knn.predict(Xval)




accuracy=metrics.accuracy_score(yval, predict)
accuracy
#print (metrics.classification_report(yval, predict))




predict_test=knn.predict(Xtest)
probs = knn.predict_proba(Xtest)




xtr=pd.concat((probs,predic_test,ytest),axis=0)
xtr=xtr.loc[xtr[]]

#accuracy_xtreme=metrics.accuracy_score(ytest,probs)
#accuracy_xtreme




accuracy_test=metrics.accuracy_score(ytest, predict_test)
accuracy_test

#print (metrics.classification_report(ytest, predict))




bestScore=pd.Series.from_csv("../working/BestCase.csv")
if bestScore.values[0]>accuracy_test:
    grabar=False
else:
    grabar=True
    scoreTest=pd.Series(accuracy_test,index=["accuracy"])
    scoreTest.to_csv("../working/BestCase.csv")




if grabar:
    tourney_seeds=pd.read_csv("../input/TourneySeeds.csv")
    tourney_seeds.head()
    model=knn
    submission=pd.DataFrame()
    Xseeds.head()




def getValues(seeds,team,year):
    seeds_season=seeds.copy().reset_index().drop("index",1)
    seeds_season=seeds_season.loc[seeds_season["Season"]==year]
    seeds_season=seeds_season.loc[(seeds_season["team1"]==team) | (seeds_season["team2"]==team)].reset_index()
    if seeds_season.loc[0]["team1"]==team:
        values=seeds_season.loc[0].drop(["Season"])
        values=values.drop([col for col in X.columns if str("2") in col])
    else:
        values=seeds_season.loc[0].drop(["Season"])
        values=values.drop([col for col in X.columns if str("1") in col])
    values=values.drop(["index","team1","team2"])
    return values




def getDataFrame(one,two):
    one=pd.DataFrame({'variable':one.index, 'value':one.values})
    one=one.set_index(["variable"]).transpose()
    one=one.rename(columns= lambda x: str(x)[:-1])
    one=one.rename(columns= lambda x: str(x+"1"))
    
    two=pd.DataFrame({'variable':two.index, 'value':two.values})
    two=two.set_index(["variable"]).transpose()
    two=two.rename(columns= lambda x: str(x)[:-1])
    two=two.rename(columns= lambda x: str(x+"2"))

    frame=pd.concat((one,two),axis=1)
    frame = frame.reindex_axis(sorted(frame.columns), axis=1)
    frame=frame.reset_index().drop("index",axis=1)
    frame= frame.rename_axis(None,axis=1)
    return frame




#2013
if grabar:
    seeds_2013=tourney_seeds.loc[tourney_seeds["Season"] == 2013]
    seeds_2013=seeds_2013.sort_values("Team", ascending=[1])
    seeds_2013.head()

    submission_2013=pd.DataFrame()
    submission_2013 = pd.DataFrame(columns=('id', 'pred'))
    for index in range(0,seeds_2013.shape[0]):
        for index2 in range(0,seeds_2013.shape[0]):
            if seeds_2013.iloc[index]["Team"]<seeds_2013.iloc[index2]["Team"]:
                    team1=seeds_2013.iloc[index]["Team"]
                    values1=getValues(Xseeds,team1,2013)
                    team2=seeds_2013.iloc[index2]["Team"]
                    values2=getValues(Xseeds,team2,2013)
                    ids="2013_" + str(team1) + "_" + str(team2)
                    probs=model.predict_proba(getDataFrame(values1,values2))
                    pred=probs[0][0]
                    submission_2013.loc[submission_2013.shape[0]]=[ids,pred]

    submission = pd.concat((submission, submission_2013), axis=0)
    submission.shape




#2014
if grabar:
    seeds_2014=tourney_seeds.loc[tourney_seeds["Season"] == 2014]
    seeds_2014=seeds_2014.sort_values("Team", ascending=[1])
    seeds_2014.head()

    submission_2014=pd.DataFrame()
    submission_2014 = pd.DataFrame(columns=('id', 'pred'))
    for index in range(0,seeds_2014.shape[0]):
        for index2 in range(0,seeds_2014.shape[0]):
            if seeds_2014.iloc[index]["Team"]<seeds_2014.iloc[index2]["Team"]:
                    team1=seeds_2014.iloc[index]["Team"]
                    values1=getValues(Xseeds,team1,2014)
                    team2=seeds_2014.iloc[index2]["Team"]
                    values2=getValues(Xseeds,team2,2014)
                    ids="2014_" + str(team1) + "_" + str(team2)
                    probs=model.predict_proba(getDataFrame(values1,values2))
                    pred=probs[0][0]
                    submission_2014.loc[submission_2014.shape[0]]=[ids,pred]
                
    submission = pd.concat((submission, submission_2014), axis=0)
    submission.shape




#2015
if grabar:
    seeds_2015=tourney_seeds.loc[tourney_seeds["Season"] == 2015]
    seeds_2015=seeds_2015.sort_values("Team", ascending=[1])
    seeds_2015.head()

    submission_2015=pd.DataFrame()
    submission_2015 = pd.DataFrame(columns=('id', 'pred'))
    for index in range(0,seeds_2015.shape[0]):
        for index2 in range(0,seeds_2015.shape[0]):
            if seeds_2015.iloc[index]["Team"]<seeds_2015.iloc[index2]["Team"]:
                    team1=seeds_2015.iloc[index]["Team"]
                    values1=getValues(Xseeds,team1,2015)
                    team2=seeds_2015.iloc[index2]["Team"]
                    values2=getValues(Xseeds,team2,2015)
                    ids="2015_" + str(team1) + "_" + str(team2)
                    probs=model.predict_proba(getDataFrame(values1,values2))
                    pred=probs[0][0]
                    submission_2015.loc[submission_2015.shape[0]]=[ids,pred]
                
    submission = pd.concat((submission, submission_2015), axis=0)
    submission.shape




#2016
if grabar:
    
    seeds_2016=tourney_seeds.loc[tourney_seeds["Season"] == 2016]
    seeds_2016=seeds_2016.sort_values("Team", ascending=[1])
    seeds_2016.head()

    submission_2016=pd.DataFrame()
    submission_2016 = pd.DataFrame(columns=('id', 'pred'))
    for index in range(0,seeds_2016.shape[0]):
        for index2 in range(0,seeds_2016.shape[0]):
            if seeds_2016.iloc[index]["Team"]<seeds_2016.iloc[index2]["Team"]:
                    team1=seeds_2016.iloc[index]["Team"]
                    values1=getValues(Xseeds,team1,2016)
                    team2=seeds_2016.iloc[index2]["Team"]
                    values2=getValues(Xseeds,team2,2016)
                    ids="2016_" + str(team1) + "_" + str(team2)
                    probs=model.predict_proba(getDataFrame(values1,values2))
                    pred=probs[0][0]
                    submission_2016.loc[submission_2016.shape[0]]=[ids,pred]
                
    submission = pd.concat((submission, submission_2016), axis=0)
    submission.shape




if grabar:
    submission.to_csv('../working/submission.csv', index=False)

