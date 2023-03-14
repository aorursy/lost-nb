#!/usr/bin/env python
# coding: utf-8



# Manipulação dos dados
import pandas as pd
import numpy as np

# Visualização 
import matplotlib.pyplot as plt
import os

# Técnica de Machine Learning
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing as prep




os.listdir("../input/uci-adult")




adult = pd.read_csv("../input/uci-adult/adult.data",
        names = [
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
        sep= r'\s*,\s*',
        engine= 'python',
        na_values= "?")

adult.shape




adult.head()




adult.describe()




adult["Workclass"].value_counts().plot(kind = "bar")




adult["Martial Status"].value_counts()




adult["Occupation"].value_counts().plot(kind = 'bar')




adult["Relationship"].value_counts().plot(kind = 'bar')




adult["Race"].value_counts().plot(kind = 'pie')




adult["Sex"].value_counts().plot(kind = 'pie')




adult["Country"].value_counts()




adult.isnull().sum()




moda = adult['Workclass'].describe().top
adult['Workclass'] = adult['Workclass'].fillna(moda)

moda = adult['Occupation'].describe().top
adult['Occupation'] = adult['Occupation'].fillna(moda)

moda = adult['Country'].describe().top
adult['Country'] = adult['Country'].fillna(moda)

adult.isnull().sum()




testAdult = pd.read_csv("../input/uci-adult/adult.test",
            names = [
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
        sep= r'\s*,\s*',
        engine= 'python',
        na_values= "?")

testAdult.shape




testAdult.isnull().sum()




moda = testAdult['Workclass'].describe().top
testAdult['Workclass'] = testAdult['Workclass'].fillna(moda)

moda = testAdult['Occupation'].describe().top
testAdult['Occupation'] = testAdult['Occupation'].fillna(moda)

moda = testAdult['Country'].describe().top
testAdult['Country'] = testAdult['Country'].fillna(moda)

testAdult.isnull().sum()




nTestAdult = testAdult.dropna()

nTestAdult.shape




nTestAdult.head()




adult = adult.apply(prep.LabelEncoder().fit_transform)

nTestAdult = nTestAdult.apply(prep.LabelEncoder().fit_transform)




adult.head()




atributos = ["Age", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week"]

x_train = adult[atributos]
y_train = adult.Target

x_test = nTestAdult[atributos]
y_test = nTestAdult.Target




knn = KNeighborsClassifier(n_neighbors = 5)

scores = cross_val_score(knn, x_train, y_train, cv=10)
scores.mean()




knn.fit(x_train, y_train)

y_predict = knn.predict(x_test)

accuracy_score(y_test, y_predict)




knn = KNeighborsClassifier(n_neighbors = 25)

scores = cross_val_score(knn, x_train, y_train, cv=10)
scores.mean()




knn.fit(x_train, y_train)

y_predict = knn.predict(x_test)

accuracy_score(y_test, y_predict)




atributos = ["Age", "Workclass", "Education-Num", "Occupation", "Race", "Capital Gain", "Capital Loss", "Hours per week"]

x_train = adult[atributos]
y_train = adult.Target

x_test = nTestAdult[atributos]
y_test = nTestAdult.Target




knn = KNeighborsClassifier(n_neighbors = 25)

scores = cross_val_score(knn, x_train, y_train, cv=10)
scores.mean()




knn.fit(x_train, y_train)

y_predict = knn.predict(x_test)

accuracy_score(y_test, y_predict)




inf = 1
sup = 35

scores_media = []
aux = 0
k_max = 0

i = 0
for k in range(inf, sup):
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, x_train, y_train, cv=10)
    scores_media.append(scores.mean())

    if scores_media[i] > aux:
        k_max = k
        aux = scores_media[i]

    i = i + 1

print(k_max)




x = np.arange(1, sup)

plt.figure(figsize=(10, 5))
plt.plot(x, scores_media, '--', color = 'red', linewidth = 2)
plt.plot(k_max, scores_media[k_max], 'o')

plt.xlabel('k')
plt.ylabel('Acurácia')
plt.title('Perfomance do algoritmo conforme o valor de k')




print('Acurácia para k = {0} : {1:2.2f}%'.format(k_max, 100 * scores_media[k_max]))




k = k_max

knn = KNeighborsClassifier(n_neighbors = k)




knn.fit(x_train, y_train)




y_predict = knn.predict(x_test)

y_predict




accuracy_score(y_test, y_predict)




predict = []

for i in range(len(y_predict)):
    if y_predict[i] == 0:
        predict.append('<=50K')
    else:
        predict.append('>50K')

result = pd.DataFrame(predict, columns = ["income"])
result.to_csv("Resultados_Adult.csv", index_label="Id")

result




# Manipulação dos dados
import pandas as pd
import numpy as np

# Visualização 
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Técnica de Machine Learning
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing




os.listdir("../input/costa-rican-household-poverty-prediction/")




train = pd.read_csv('../input/costa-rican-household-poverty-prediction/train.csv')
test = pd.read_csv('../input/costa-rican-household-poverty-prediction/test.csv')

train.info()

train.head()




test.info()

test.head()




from collections import OrderedDict

plt.figure(figsize = (20, 16))

cores = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})
pobreza = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})

for i, col in enumerate(train.select_dtypes('float')):
    ax = plt.subplot(4, 2, i + 1)
    
    for nivel, cor in cores.items():
       
        sns.kdeplot(train.loc[train['Target'] == nivel, col].dropna(), 
                    ax = ax, color = cor, label = pobreza[nivel])
        
    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')
    
    plt.subplots_adjust(top = 2)




rotulos = train.loc[(train['Target'].notnull()) & (train['parentesco1'] == 1), ['Target', 'idhogar']]

quantidade = rotulos['Target'].value_counts().sort_index()

quantidade.plot.bar(figsize = (8, 6), color = cores.values())

plt.xlabel('Nível de Pobreza')
plt.ylabel('Quantidade')
plt.title('Quantidade para cada Nível de Pobreza')


quantidade




igualdade = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

desigualdade = igualdade[igualdade != True]

print('Há {} famílias com membros de rótulos diferentes'.format(len(desigualdade)))




colunas_nulas = train.isnull().sum().sort_values(ascending = False)

porcentagem = ((train.isnull().sum()/train.isnull().count())*100).sort_values(ascending = False)

faltantes = pd.concat([colunas_nulas, porcentagem], axis = 1, keys = ['Total', '%'])
faltantes.head()




train.loc[(train['tipovivi1'] == 1), 'v2a1'] = 0

print('Ainda restam {} dados faltantes para v2a1'.format(train['v2a1'].isnull().sum()))




train.loc[train['rez_esc'].notnull()]['age'].describe()




train.loc[train['rez_esc'].isnull()]['age'].describe()




train.loc[((train['age'] > 19) | (train['age'] < 7)) & (train['rez_esc'].isnull()), 'rez_esc'] = 0

print('Ainda restam {} dados faltantes para v2a1'.format(train['rez_esc'].isnull().sum()))




train['meaneduc'].plot(kind = 'box', grid = True)




train['SQBmeaned'].plot(kind = 'box', grid = True)




train['meaneduc'] = train['meaneduc'].fillna(train['meaneduc'].describe().mean())

train['SQBmeaned'] = train['SQBmeaned'].fillna(train['SQBmeaned'].describe().mean())




atributos = ['v2a1', 'hacdor', 'v14a', 'escolari', 'rez_esc', 'hhsize', 'cielorazo', 'abastaguadentro',
             'abastaguano', 'noelec', 'sanitario1', 'hogar_nin', 'hogar_total', 'meaneduc', 'tipovivi1',
             'area1', 'SQBovercrowding', 'SQBmeaned', 'Target']

# Nova base de dados
base = train[atributos]
base = base.astype(np.float)

print(base.shape)
base.head()




colunas_nulas = base.isnull().sum().sort_values(ascending = False)

porcentagem = ((base.isnull().sum()/base.isnull().count())*100).sort_values(ascending = False)

faltantes = pd.concat([colunas_nulas, porcentagem], axis = 1, keys = ['Total', '%'])
faltantes.head()




# Retirando os demais dados faltantes

base = base.dropna()
base.shape




colunas_nulas = base.isnull().sum().sort_values(ascending = False)

porcentagem = ((base.isnull().sum()/base.isnull().count())*100).sort_values(ascending = False)

faltantes = pd.concat([colunas_nulas, porcentagem], axis = 1, keys = ['Total', '%'])
faltantes.head()




# Treino

x_train = base.drop('Target', axis = 1)

y_train = base['Target']




inf = 1
sup = 65

scores_media = []
aux = 0
k_max = 0

i = 0
for k in range(inf, sup):
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, x_train, y_train, cv=10)
    scores_media.append(scores.mean())
    
    if scores_media[i] > aux:
        k_max = k
        aux = scores_media[i]
        
    i = i + 1
    
print(k_max)




x = np.arange(1, sup)

plt.figure(figsize=(10, 5))
plt.plot(x, scores_media, '--', color = 'red', linewidth = 2)
plt.plot(k_max, scores_media[k_max], 'o')

plt.xlabel('k')
plt.ylabel('Acurácia')
plt.title('Perfomance do algoritmo conforme o valor de k')




print('Acurácia para k = {0} : {1:2.2f}%'.format(k_max, 100 * scores_media[k_max]))

