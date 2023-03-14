#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


os.listdir("../input/uci-adult")


# In[3]:


adult = pd.read_csv("../input/uci-adult/adult.data",
        names = [
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
        sep= r'\s*,\s*',
        engine= 'python',
        na_values= "?")

adult.shape


# In[4]:


adult.head()


# In[5]:


adult.describe()


# In[6]:


adult["Workclass"].value_counts().plot(kind = "bar")


# In[7]:


adult["Martial Status"].value_counts()


# In[8]:


adult["Occupation"].value_counts().plot(kind = 'bar')


# In[9]:


adult["Relationship"].value_counts().plot(kind = 'bar')


# In[10]:


adult["Race"].value_counts().plot(kind = 'pie')


# In[11]:


adult["Sex"].value_counts().plot(kind = 'pie')


# In[12]:


adult["Country"].value_counts()


# In[13]:


adult.isnull().sum()


# In[14]:


moda = adult['Workclass'].describe().top
adult['Workclass'] = adult['Workclass'].fillna(moda)

moda = adult['Occupation'].describe().top
adult['Occupation'] = adult['Occupation'].fillna(moda)

moda = adult['Country'].describe().top
adult['Country'] = adult['Country'].fillna(moda)

adult.isnull().sum()


# In[15]:


testAdult = pd.read_csv("../input/uci-adult/adult.test",
            names = [
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
        sep= r'\s*,\s*',
        engine= 'python',
        na_values= "?")

testAdult.shape


# In[16]:


testAdult.isnull().sum()


# In[17]:


moda = testAdult['Workclass'].describe().top
testAdult['Workclass'] = testAdult['Workclass'].fillna(moda)

moda = testAdult['Occupation'].describe().top
testAdult['Occupation'] = testAdult['Occupation'].fillna(moda)

moda = testAdult['Country'].describe().top
testAdult['Country'] = testAdult['Country'].fillna(moda)

testAdult.isnull().sum()


# In[18]:


nTestAdult = testAdult.dropna()

nTestAdult.shape


# In[19]:


nTestAdult.head()


# In[20]:


adult = adult.apply(prep.LabelEncoder().fit_transform)

nTestAdult = nTestAdult.apply(prep.LabelEncoder().fit_transform)


# In[21]:


adult.head()


# In[22]:


atributos = ["Age", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week"]

x_train = adult[atributos]
y_train = adult.Target

x_test = nTestAdult[atributos]
y_test = nTestAdult.Target


# In[23]:


knn = KNeighborsClassifier(n_neighbors = 5)

scores = cross_val_score(knn, x_train, y_train, cv=10)
scores.mean()


# In[24]:


knn.fit(x_train, y_train)

y_predict = knn.predict(x_test)

accuracy_score(y_test, y_predict)


# In[25]:


knn = KNeighborsClassifier(n_neighbors = 25)

scores = cross_val_score(knn, x_train, y_train, cv=10)
scores.mean()


# In[26]:


knn.fit(x_train, y_train)

y_predict = knn.predict(x_test)

accuracy_score(y_test, y_predict)


# In[27]:


atributos = ["Age", "Workclass", "Education-Num", "Occupation", "Race", "Capital Gain", "Capital Loss", "Hours per week"]

x_train = adult[atributos]
y_train = adult.Target

x_test = nTestAdult[atributos]
y_test = nTestAdult.Target


# In[28]:


knn = KNeighborsClassifier(n_neighbors = 25)

scores = cross_val_score(knn, x_train, y_train, cv=10)
scores.mean()


# In[29]:


knn.fit(x_train, y_train)

y_predict = knn.predict(x_test)

accuracy_score(y_test, y_predict)


# In[30]:


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


# In[31]:


x = np.arange(1, sup)

plt.figure(figsize=(10, 5))
plt.plot(x, scores_media, '--', color = 'red', linewidth = 2)
plt.plot(k_max, scores_media[k_max], 'o')

plt.xlabel('k')
plt.ylabel('Acurácia')
plt.title('Perfomance do algoritmo conforme o valor de k')


# In[32]:


print('Acurácia para k = {0} : {1:2.2f}%'.format(k_max, 100 * scores_media[k_max]))


# In[33]:


k = k_max

knn = KNeighborsClassifier(n_neighbors = k)


# In[34]:


knn.fit(x_train, y_train)


# In[35]:


y_predict = knn.predict(x_test)

y_predict


# In[36]:


accuracy_score(y_test, y_predict)


# In[37]:


predict = []

for i in range(len(y_predict)):
    if y_predict[i] == 0:
        predict.append('<=50K')
    else:
        predict.append('>50K')

result = pd.DataFrame(predict, columns = ["income"])
result.to_csv("Resultados_Adult.csv", index_label="Id")

result


# In[38]:


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


# In[39]:


os.listdir("../input/costa-rican-household-poverty-prediction/")


# In[40]:


train = pd.read_csv('../input/costa-rican-household-poverty-prediction/train.csv')
test = pd.read_csv('../input/costa-rican-household-poverty-prediction/test.csv')

train.info()

train.head()


# In[41]:


test.info()

test.head()


# In[42]:


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


# In[43]:


rotulos = train.loc[(train['Target'].notnull()) & (train['parentesco1'] == 1), ['Target', 'idhogar']]

quantidade = rotulos['Target'].value_counts().sort_index()

quantidade.plot.bar(figsize = (8, 6), color = cores.values())

plt.xlabel('Nível de Pobreza')
plt.ylabel('Quantidade')
plt.title('Quantidade para cada Nível de Pobreza')


quantidade


# In[44]:


igualdade = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

desigualdade = igualdade[igualdade != True]

print('Há {} famílias com membros de rótulos diferentes'.format(len(desigualdade)))


# In[45]:


colunas_nulas = train.isnull().sum().sort_values(ascending = False)

porcentagem = ((train.isnull().sum()/train.isnull().count())*100).sort_values(ascending = False)

faltantes = pd.concat([colunas_nulas, porcentagem], axis = 1, keys = ['Total', '%'])
faltantes.head()


# In[46]:


train.loc[(train['tipovivi1'] == 1), 'v2a1'] = 0

print('Ainda restam {} dados faltantes para v2a1'.format(train['v2a1'].isnull().sum()))


# In[47]:


train.loc[train['rez_esc'].notnull()]['age'].describe()


# In[48]:


train.loc[train['rez_esc'].isnull()]['age'].describe()


# In[49]:


train.loc[((train['age'] > 19) | (train['age'] < 7)) & (train['rez_esc'].isnull()), 'rez_esc'] = 0

print('Ainda restam {} dados faltantes para v2a1'.format(train['rez_esc'].isnull().sum()))


# In[50]:


train['meaneduc'].plot(kind = 'box', grid = True)


# In[51]:


train['SQBmeaned'].plot(kind = 'box', grid = True)


# In[52]:


train['meaneduc'] = train['meaneduc'].fillna(train['meaneduc'].describe().mean())

train['SQBmeaned'] = train['SQBmeaned'].fillna(train['SQBmeaned'].describe().mean())


# In[53]:


atributos = ['v2a1', 'hacdor', 'v14a', 'escolari', 'rez_esc', 'hhsize', 'cielorazo', 'abastaguadentro',
             'abastaguano', 'noelec', 'sanitario1', 'hogar_nin', 'hogar_total', 'meaneduc', 'tipovivi1',
             'area1', 'SQBovercrowding', 'SQBmeaned', 'Target']

# Nova base de dados
base = train[atributos]
base = base.astype(np.float)

print(base.shape)
base.head()


# In[54]:


colunas_nulas = base.isnull().sum().sort_values(ascending = False)

porcentagem = ((base.isnull().sum()/base.isnull().count())*100).sort_values(ascending = False)

faltantes = pd.concat([colunas_nulas, porcentagem], axis = 1, keys = ['Total', '%'])
faltantes.head()


# In[55]:


# Retirando os demais dados faltantes

base = base.dropna()
base.shape


# In[56]:


colunas_nulas = base.isnull().sum().sort_values(ascending = False)

porcentagem = ((base.isnull().sum()/base.isnull().count())*100).sort_values(ascending = False)

faltantes = pd.concat([colunas_nulas, porcentagem], axis = 1, keys = ['Total', '%'])
faltantes.head()


# In[57]:


# Treino

x_train = base.drop('Target', axis = 1)

y_train = base['Target']


# In[58]:


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


# In[59]:


x = np.arange(1, sup)

plt.figure(figsize=(10, 5))
plt.plot(x, scores_media, '--', color = 'red', linewidth = 2)
plt.plot(k_max, scores_media[k_max], 'o')

plt.xlabel('k')
plt.ylabel('Acurácia')
plt.title('Perfomance do algoritmo conforme o valor de k')


# In[60]:


print('Acurácia para k = {0} : {1:2.2f}%'.format(k_max, 100 * scores_media[k_max]))

