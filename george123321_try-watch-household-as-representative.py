#!/usr/bin/env python
# coding: utf-8



# import libraries

import datetime
import json
from functools import reduce
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression,LassoCV
from sklearn.cluster import FeatureAgglomeration
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
import sys
import os
print('libraries done')




''' 1.get data - train/test '''
''' 2.glimpse of Data '''
''' 3.Data Exploration '''
''' concrete variables '''
''' concrete not dummy vars '''
''' general variables '''
''' general not dummy variables '''
''' 4.many missing rows '''
''' 5.manage missings '''
''' 6.Flattern id by idhogar within (house mates form rows to columns) '''
''' 7.aggregate by idhogar '''
''' 8.watch outliers '''
''' 9.get training and testing sets '''
''' 10.get models and thier parameters '''
''' 11.train models '''
''' 12.make submission '''




''' 1.get data - train/test '''

path='../input/'

# get train data
iid='Id'
train=pd.read_csv(path+'train.csv',index_col=0)
train['stage']='train'
target='Target'

#get test data
test=pd.read_csv(path+'test.csv',index_col=0)
test['stage']='test'
test[target]=-99 # set target to -99 so that it is in train/test when combined

# make one daatset in order to prepare both test and train variables
col=list(set(train.columns)&set(test.columns))
train=pd.concat([train[col],test[col]],axis=0)
print(train.head(2)) # watch data
print(train.describe()) # watch data without nas




''' 2.glimpse of Data '''


# prove we should watch household as whole
print(train.groupby('idhogar')[target].nunique().value_counts())

# plot numeric variables with words in it
for i in ['edjefe','edjefa','rez_esc','dependency']:
    print('########'+i)
    print(train[i].value_counts())
    
# fix text in some vars
train.loc[train['dependency']=='yes','dependency']='8'
train.loc[train['dependency']=='no','dependency']='.125'
train['dependency']=train['dependency'].astype(float)

# check education vars separately
train.loc[train['edjefa']=='yes','edjefa']='1'
train.loc[train['edjefa']=='no','edjefa']='0'
train['edjefa']=train['edjefa'].astype(float)
train.loc[train['edjefe']=='yes','edjefe']='1'
train.loc[train['edjefe']=='no','edjefe']='0'
train['edjefe']=train['edjefe'].astype(float)





# check 'edjefe','edjefa' variables meaning
h=train.loc[train['parentesco1']==1]
hm=h.loc[h.male==1,['edjefe','escolari']]
hf=h.loc[h.female==1,['edjefa','escolari']]
print(hf.corr())
print(hm.corr())
# we finally see that correlation is great - so i would rather omit wordy variables

''' 'rez_esc' is bad - so omit by other reason - see further '''




''' 3.Data Exploration '''

dtrain=train.loc[train['stage']=='train']

# watch variables unique quantity
u=train.groupby('idhogar').nunique()
u=u.mean(axis=0)
gen=list(u[u<=1].index) # get variables idhogar unique
conc=list(u[u>1].index) # get variables id unique

''' concrete variables '''

# watch dummy variables - we group 'em

a=pd.DataFrame([i[:3] for i in conc])
a=a[0].value_counts()
l=[]
for i in a.index:
  g=[j for j in conc if i in j[:3]]
  l.append(g)
l0=[i for i in l if len(i)>1]
l1=[i for i in l if len(i)<=1]
l1.remove(['male'])
l1.remove(['female'])
l0.append(['male','female'])

concrete_dum_vars=l0.copy()

print(concrete_dum_vars)




# plot id unique variables on train set
for v in l0[1:3]:
    dtrain1=dtrain.copy()
    dtrain1['var']=dtrain1[v].idxmax(axis=1)

    fr=np.unique(dtrain1['var'])
    fh=''.join([i for i in fr[0] if not i.isdigit()])
    ff=np.sort([int(i.replace(fh,'')) for i in fr ])
    ff=[fh+str(i) for i in ff]

    plt.figure()
    sns.countplot(x='var',  data=dtrain1,order=ff )
    plt.xticks(rotation=50)
    plt.show()

    plt.figure()
    sns.factorplot(x='var',y=target, data=dtrain1, kind="bar",order=ff)
    plt.xticks(rotation=50)
    plt.show()




''' concrete not dummy vars '''

concrete_contin_vars=[v[0] for v in l1]
concrete_contin_vars.remove('Target')


print(concrete_contin_vars)

# intersting plot
sns.jointplot(dtrain['escolari'], dtrain[target], kind="kde")
# with education level target variance increases




'edjefa'  in train.columns




''' general variables '''

import matplotlib.pyplot as plt
# aggregate target by household
gra=train.loc[train['stage']=='train'].groupby('idhogar')[gen+[target]].mean()
gra[target]=np.round(gra[target])

# deneral dummy vars
a=pd.DataFrame([i[:3] for i in gen])
a=a[0].value_counts()
l=[]
for i in a.index:
  g=[j for j in gen if i in j[:3]]
  l.append(g)
l0=[i for i in l if len(i)>1]
l1=[i[0] for i in l if len(i)==1]
b0=['public','planpri','noelec','coopele'] # not intersting
l1=[i for i in l1 if i not in b0]
l0.append(b0)
l0=[i for i in l0 if set(i) != set(['v18q1','v18q'])] # these we will watch separately


general_categ_vars=[l0[i] for i in [1,5,10]]


print(general_categ_vars)







# drop edu vars (see top :) )
train=train.drop(['edjefa','edjefe'],axis=1)




# general dummy variables
for v in l0:

    gra1=gra.copy()
    gra1['var']=gra1[v].idxmax(axis=1)

    plt.figure()
    sns.countplot(x='var',  data=gra1)
    plt.xticks(rotation=50)
    plt.show()

    plt.figure()
    sns.factorplot(x='var',y=target, data=gra1, kind="bar")
    plt.xticks(rotation=50)
    plt.show()




''' general not dummy variables '''
from scipy import stats

sns.jointplot(gra['v2a1'], gra['meaneduc'], kind="kde")
sns.jointplot(gra['overcrowding'], gra['meaneduc'], kind="kde")
sns.jointplot(gra['v2a1'], gra['qmobilephone'], kind="kde")




# watch dependency to target relation
fig, ax =plt.subplots(1,2)
sns.violinplot(x=target,y='dependency', data=gra, ax=ax[0])
sns.factorplot(x=target,y='dependency', data=gra, kind="bar", palette="muted", ax=ax[1])
fig.show()




# smoother normal distribution
plt.figure()
stats.probplot(gra['bedrooms'], dist='norm', fit=True, plot=plt.subplot(111))

bed_t=gra['bedrooms']/gra['rooms']
plt.figure()
sns.distplot(bed_t)
plt.figure()
stats.probplot(bed_t, dist='norm', fit=True, plot=plt.subplot(111))




# pair plot of continious variables
paircol=['v2a1', 'rooms', 'v18q', 'v18q1', 'tamhog',
 'tamviv', 'rez_esc', 'hhsize', 'hogar_adul',
 'hogar_mayor', 'hogar_total', 'dependency', 'bedrooms', 'overcrowding', 'qmobilephone', 'area2', ]
sns.pairplot(gra[paircol].dropna(how='any'))




# find some more correlated variables

print(gra[['tamhog','hhsize']].corr() )
print('We will choose only on of them- hhsize seems more representable')




# plot frequency of cross variables *(age - hogar_total)
sns.heatmap(pd.crosstab(train['age'], train['hogar_total']))




# plot frequency of cross variables *(v18q - area2)
sns.heatmap(pd.crosstab(train['v18q'], train['area2']))




''' 4.many missing rows '''

con=train.loc[train['stage']=='train',:]

rowna=con.T.apply(lambda x: sum(x.isnull().values), axis = 0)
rowna=rowna/train.shape[1]

sns.distplot(rowna)
a=0.4 # cutoff to drop a row
print(sum(rowna<a)/train.shape[0])

prp=set(list(rowna[rowna>a].index))
prp=list(set(con.index)-prp)

train=pd.concat([con.loc[prp ,:],train.loc[train['stage']=='test',:]],axis=0)
print('done')




''' 5.manage missings '''

mv={'bin_int': 
            ['sanitario1',  'tipovivi1',  'instlevel3',  'etecho2', 'parentesco1',  'lugar6',  'parentesco10',  'paredblolad',  'parentesco7',  'parentesco9',  'area2',  'parentesco2',  'pisoother',  'parentesco12', 'eviv2',
          'epared2',  'television',  'energcocinar4',  'epared3',  'lugar4',  'sanitario2',  'energcocinar3',  'paredother',  'tipovivi4',  'pisomoscer',  'instlevel2',  'instlevel1',  'abastaguadentro',  'paredmad',  'pisomadera',  'techocane',  'energcocinar2',  'v14a',  'dis',  'elimbasu1',  'instlevel9',
          'lugar1',  'pisocemento',  'parentesco3',  'paredfibras',  'instlevel8',  'paredzinc',  'sanitario3',  'epared1',  'estadocivil7',  'tipovivi5',  'techozinc',  'area1',  'planpri',  'paredzocalo',  'parentesco11',  'lugar2',  'sanitario6',  'etecho3',  'estadocivil1',  'pisonotiene',  'estadocivil4',  'techoentrepiso',  'mobilephone',
          'instlevel6',  'hacdor',  'hacapo',  'lugar3',  'tipovivi3',  'coopele',  'paredpreb',  'estadocivil2',  'parentesco4',  'eviv3',  'cielorazo',  'techootro',  'eviv1',  'etecho1',  'v18q',  'estadocivil5',  'instlevel5',  'elimbasu4',  'male',  'abastaguafuera',  'abastaguano',  'tipovivi2',  'computer',  'estadocivil6',  'pisonatur',  'parentesco5',  'instlevel4',
          'instlevel7',  'parentesco6',  'noelec',  'estadocivil3',  'female',  'energcocinar1',  'elimbasu6',  'refrig',  'lugar5',  'pareddes',  'parentesco8',  'elimbasu5',  'sanitario5',  'public',  'elimbasu3',  'elimbasu2'],
    'categ': 
            ['idhogar'],
    'cont': 
            ['r4h3',
          'age',
          'SQBescolari',
          'overcrowding',
          'r4m1',  'SQBedjefe',  'v2a1',  'r4t1', 'SQBhogar_total',  'r4h2',  'meaneduc',  'SQBmeaned',  'rez_esc', 'agesq',  'hogar_total',  'v18q1',  'hhsize',  'qmobilephone',  'rooms',  'hogar_adul',
          'dependency',  'hogar_nin',  'r4t2', 'escolari',  'SQBage',  'r4t3', 'r4m3',  'r4m2',  'tamviv',  'SQBhogar_nin',  'tamhog', 'hogar_mayor',  'SQBovercrowding',  'SQBdependency',  'bedrooms', 'r4h1']
 }


# watch continious variables
from sklearn.impute import SimpleImputer

contin_vars=train[mv['cont']]



def miss_map(dx,k=0.5):
    missingValueColumns = dx.columns[dx.isnull().sum(axis=0)/dx.shape[0]>k].tolist()
    return missingValueColumns
manymiss=miss_map(con)
print(manymiss) 
print(' ')
# 'v2a1' we consider an important varible- so we would preserve it

manymiss=['rez_esc', 'v18q1']
# give less then 30% observations of data
mv['cont']=list(set(mv['cont'])-set(manymiss))
my_imputer = SimpleImputer( strategy='mean') #most_frequent
contin_vars = my_imputer.fit_transform(contin_vars[mv['cont']])
contin_vars=pd.DataFrame(contin_vars,columns=mv['cont'])
contin_vars.index=train.index



# categorical variables are good
categ_vars=train[mv['categ']+mv['bin_int']]


d=pd.merge(contin_vars,categ_vars,left_index=True, right_index=True)
print('done')




''' 6.Flattern id by idhogar within (house mates form rows to columns) '''

# As we watch clusters as a whole we want to determine most useful household mates( choose them )
# these are 'parentesco_' columns


from sklearn import cluster


paren_group=d.groupby('idhogar')[general_categ_vars[0]].sum()
for i in paren_group.columns:
    paren_group.loc[paren_group[i]>1,i]=1

n=10
agg=cluster.AgglomerativeClustering( n_clusters=n, linkage='average')
agg.fit(paren_group)
paren_group['l']=agg.labels_
# draw heat map by count of mates in same house
paren_heat1=paren_group.groupby('l')[general_categ_vars[0]].sum()
sns.heatmap(paren_heat1)
print('We see that mates 1,2,3 are most met - 8000 cases')





# drop parentesco1,2,3 to see most met cases of other mates
vv=['parentesco7',
 'parentesco10',
 'parentesco6',
 'parentesco11',
 'parentesco9',
 'parentesco4',
 'parentesco5',
 'parentesco12',
 'parentesco8']
paren_group1=d.groupby('idhogar')[vv].sum()
for i in paren_group1.columns:
    paren_group1.loc[paren_group1[i]>1,i]=1
# build clustering
n=60
agg=cluster.AgglomerativeClustering( n_clusters=n, linkage='average')
agg.fit(paren_group1)
paren_group1['l']=agg.labels_
# draw heat map by count of mates in same house
paren_heat2=paren_group1.groupby('l')[vv].sum()
sns.heatmap(paren_heat2)
print('We see that mates not 1,2,3 are only 300 cases - not much')

d['par']=d[concrete_dum_vars[0]].idxmax(axis=1)
d['par']=d['par'].map(lambda x: int(x.replace('parentesco','') ))




v=d.copy()

v[target]=train[target]
v['stage']=train['stage']
v=v.loc[v['stage']=='train',:]




# check  how mates1,2,3 are though representative

g=v.groupby('idhogar')[target].agg({target:'mean'})
v1=v.loc[v.par.isin([1,2,3])]
g1=v1.groupby('idhogar')[target].agg({'target2':'mean'})
g=pd.merge(g,g1,right_index=True,left_index=True)
g['d']=g.Target-g.target2
g=g.loc[g.d!=0]
print(g.head(3))
print('There are only {0} outlier households which dont follow my assumptions'.format(g.shape[0]) )




# one more check of mates to concrete variables dependencies

# plot graphs of common variables to target

# get columns to label encoding
v['instlevel']=v[concrete_dum_vars[1]].idxmax(axis=1)
v['instlevel']=v['instlevel'].map(lambda x: int(x.replace('instlevel','') ))
v['estadocivil']=v[concrete_dum_vars[2]].idxmax(axis=1)
v['estadocivil']=v['estadocivil'].map(lambda x: int(x.replace('estadocivil','') ))
v['sex']=v[concrete_dum_vars[5]].idxmax(axis=1)
v['sex']=v['sex'].map({'male':0, 'female':1})


for h in ['instlevel','sex','estadocivil','instlevel']:
        
    plt.figure(figsize=(20,10))
    sns.violinplot(x='par',y=target,hue=h, data=v)
    plt.show()
    plt.close()




''' 7.aggregate by idhogar '''

# Finally we aggregate variables by idhogar
# general have same values for all idhogar members
# conrete have varying values for each idhogar member

import itertools

bvc = list(itertools.chain.from_iterable(concrete_dum_vars[1:]))
print(bvc)
# get only important members of household
tt=d.loc[d.par.isin([1,2,3]),['par','idhogar']+bvc+concrete_contin_vars+['dependency'] ]
# get all members in one row
t=pd.pivot_table(tt,values=bvc,index='idhogar', columns='par')
t.columns =['par'+str(v)+'_'+str(k) for k,v in zip(t.columns.get_level_values(0),
t.columns.get_level_values(1) )]
t=t.reset_index()
t=t.fillna(0)


bvg = list(itertools.chain.from_iterable(general_categ_vars))
vt1=['meaneduc','overcrowding','v2a1','qmobilephone','hhsize','bedrooms' ,'rooms','hogar_total','v18q','area2']    
t1=d[['idhogar']+bvg+vt1 ].reset_index().drop('Id',axis=1).drop_duplicates()

t2=pd.merge(t,t1,on='idhogar')    
t2=t2.set_index('idhogar')


# Need to mention we are missing idhogar 'ce6154327' as it has only one mate par7 - we will manage it further
 
vv=train.loc[(train.parentesco1==1)|(train.parentesco2==1)|(train.parentesco3==1)]
vv=vv.groupby(['idhogar','stage']).agg({target:'mean'}).reset_index('stage')
vv.Target=np.round(vv.Target)

dd=pd.merge(t2,vv,right_index=True,left_index=True)

print(dd.head()) 
print(dd.index.nunique())
    

print(dd.columns)




''' 8.watch outliers '''


from collections import Counter


def outliers_iqr(df,n):
    features=df.columns
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col],75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers

dtrain=dd.loc[dd['stage']=='train',:].drop('stage',axis=1)

# watch outliers via method #1
con=dtrain.drop(target,axis=1)
ell=EllipticEnvelope().fit(con)
prr = ell.predict(con) # rows to drop
pp0=sum(prr==1)/len(prr) 
w={}  

# watch outliers via method #2
for i in range(2,con.shape[1]//2): #choose number of neighbours to get outliers
  pr1=outliers_iqr(con,i)
  pp1=1-len(pr1)/con.shape[0]
  w[pp1]=pr1
k=np.array(list(w.keys()))
pp1=min(k[k>0.7])
pr1=w[pp1]

# obtain combined set of rows left after both methods applied
prp=set(pr1)&set(con.index[prr!=1])
prp=list(set(dtrain.index)-prp)
pr0=(pp0+pp1)/2

# omit outlier rows
if 0.7<=pr0:
  dtrain=dtrain.loc[prp,:]
  print(dtrain.shape[0]/train.shape[0])

dtrain=dtrain.astype(float)

print('done')




''' 9.get training and testing sets ''' 

train_m=dtrain
test_m=dd.loc[dd['stage']=='test'].drop(target,axis=1)




''' 10.get models and thier parameters '''

from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import ( AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
param_grid=dict(

nbc={'alpha':[0.1,0.3,0.6,1],
    'binarize':[True,False],
    'fit_prior':[True,False] # learn class prior probabilities or not.
    },

adc = {
    'n_estimators': list(np.arange(50,251,50)),
    'learning_rate' : [0.1, 0.01, 0.001],
    "algorithm" : ["SAMME","SAMME.R"]
    },
    
gbc = {
    'n_estimators': [40,120],
    'max_depth': [3,10],
    'min_samples_leaf': [0.1, 0.5],
    'min_samples_split':[0.3, 0.7],
    'loss' : ["deviance"],#,'exponential'
    'learning_rate': [0.1, 0.01],
    'max_features': [0.3, 0.7]

    }

)


def param_mods(parameters):

    return  dict(
            nbc=BernoulliNB(**parameters['nbc']),
            adc=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),**parameters['adc']),
            gbc=GradientBoostingClassifier(**parameters['gbc'] ),

            )




''' 11.train models '''

from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn.model_selection import StratifiedKFold,KFold,cross_val_score
from sklearn.grid_search import GridSearchCV

from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc
from sklearn.metrics import mean_absolute_error, mean_squared_error,f1_score, make_scorer

from sklearn.model_selection import learning_curve
#from funnn import plot_learning_curve
import os



# prepare data and parameteres
dfs=train_m
b=dfs.columns[:-1]
scaler=StandardScaler().fit(dfs.loc[:,b])
dfs.loc[:,b]=scaler.transform(dfs.loc[:,b])


seed=1
X=dfs.iloc[:,:-1]
Y=dfs.iloc[:,-1]
X_val, X_train, Y_val, Y_train = cross_validation.train_test_split(X, Y, test_size=0.7,random_state=seed)
col=list(param_grid.keys())
col_c=[x for x in col if list(x)[-1]=='c']
col_r=[x for x in col if list(x)[-1]=='r']
methods = col_r if len(Y.unique()) > 10 else col_c

kfold = StratifiedKFold(n_splits=10)
cl_param={'scoring' :'f1_macro'}

modd=['nbc', 'adc', 'gbc']




print(X_train.head(3) )
print(X_train.shape)
print(Y_train.head(3) )




# grid search

models_dict = param_mods(param_grid)
best_mods={}
scor={}

for i in modd:
  mod = GridSearchCV(models_dict[i],param_grid = param_grid[i], **cl_param) # n_jobs=1,,cv= 10
  mod.fit(X_train,Y_train )
  best_mods[i] = mod.best_estimator_
  scor[i] = mod.best_score_


# cv results search
models_dict =  best_mods if len(scor)>0 else param_mods(param)
cv_results = {}
for i in modd:
    cv_results[i]=cross_val_score(models_dict[i], X= X_train, y = Y_train,**cl_param,cv= 5) #,  n_jobs=1
cv_results = {k:np.round(np.mean(v)-3*np.std(v),2) for k,v in cv_results.items()}

print('done')




print(cv_results)




# plot training results
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)


    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

modf=best_mods['nbc']

plt.figure()
f=plot_learning_curve(modf, 'Learningcurve', X_train, Y_train, ylim=None,  n_jobs=-1)

y_true=Y_val
y_pred=modf.predict(X_val)

print(f1_score(y_true, y_pred,average='macro'))




''' 12.make submission '''

#  predict by idhogar
X_test=test_m.drop('stage',axis=1)
dft=StandardScaler().fit_transform(X_test)
Y_test=modf.predict(dft)
X_test['Target']=Y_test
X_test=X_test.reset_index()
X_test=X_test[['idhogar','Target']]

print(X_test.head())
print(X_test.shape)




#  predict by id
test=pd.read_csv(path+'test.csv',index_col=0).reset_index()
test=test[['idhogar','Id']]
fin_test=pd.merge(test,X_test,on='idhogar',how='left')
Y_test=fin_test.drop('idhogar',axis=1)
Y_test=Y_test.fillna(1)
Y_test.Target=Y_test.Target.astype(int)

   
  
    
#Y_test=pd.DataFrame({'target':Y_test},index=X_test.index)
Y_test.to_csv('submission.csv', index=False)

print(Y_test.head(2)) 
print(Y_test.shape) 





'''
# get train data

sub=pd.read_csv(path+'sample_submission.csv')
print(sub.head(2))
su=pd.merge(sub,Y_test, on = 'Id')
print(su.shape)
print(sub.shape)
''''''

