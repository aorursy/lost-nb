#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Affichage Image
from IPython.display import Image
from IPython.display import display

# MàJ de certains paramétres (Affichage et manipulation des données)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18
plt.rcParams['patch.edgecolor'] = 'k'


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


#plt.style.use('seaborn-whitegrid')
sns.set(style = 'darkgrid')

# Outils de modélisation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb


# Macro F1 score
scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')


# Fonctions de réequilibrage des données
from imblearn.over_sampling import SMOTE


# In[2]:


#Le path vers les fichiers de données en local
#filepath ='/Users/moi/Documents/CoursDauphine/Module3/supervised/costa-rican-household-poverty-prediction'
#filepath_train = '/Users/moi/Documents/CoursDauphine/Module3/supervised/costa-rican-household-poverty-prediction/train.csv'
#filepath_test = '/Users/moi/Documents/CoursDauphine/Module3/supervised/costa-rican-household-poverty-prediction/test.csv'

#filepath_submission = '/Users/moi/Documents/CoursDauphine/Module3/supervised/costa-rican-household-poverty-prediction/sample_submission.csv' 


# In[ ]:





# In[3]:


#df_train = pd.read_csv(filepath_train, sep=',', decimal='.')
#df_test = pd.read_csv(filepath_test, sep=',', decimal='.')
#df_submit = pd.read_csv(filepath_submission, sep=',', decimal='.')


# In[4]:


df_train = pd.read_csv('../input/train.csv', sep=',', decimal='.')
df_test = pd.read_csv('../input/test.csv', sep=',', decimal='.')
df_submit = pd.read_csv('../input/sample_submission.csv', sep=',', decimal='.')


# In[5]:


df_train.shape


# In[6]:


df_test.shape


# In[7]:


df_submit.shape


# In[8]:


df_train.head()


# In[9]:


df_test.head()


# In[10]:


df_train.info()


# In[11]:


df_test.info()


# In[12]:


df_train.select_dtypes(include=['int64']).nunique().value_counts().sort_index().plot.bar(color = 'blue', figsize = (8,6), edgecolor ='k', linewidth = 2)

plt.xlabel('Nombre de valeurs uniques prise par la variables'); plt.ylabel('Nombre de colonnes');
plt.title('Nombre de colonnes de type Integer avec ces valeurs uniques');


# In[13]:


from collections import OrderedDict

plt.figure(figsize = (20, 16))
plt.style.use('fivethirtyeight')

# Color mapping
colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})
poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})

for i,col in enumerate(df_train.select_dtypes(include=['float64'])):

    ax = plt.subplot(6, 2, i + 1)
    # Iterate through the poverty levels
    for poverty_level, color in colors.items():
        # Plot each poverty level as a separate line
        sns.kdeplot(df_train.loc[df_train['Target'] == poverty_level, col].dropna(), 
                    ax = ax, color = color, label = poverty_mapping[poverty_level])
        
    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')


plt.subplots_adjust(top = 2)


# In[ ]:





# In[14]:


df_train.select_dtypes(include = ['object']).head()


# In[15]:


# Merge des deux dataframe train et test et ajout de la colonne Target aux données de test
df_test['Target'] = np.nan
data = df_train.append(df_test, ignore_index = True)


# In[16]:


data.shape


# In[17]:


data['Target'].isnull().sum()


# In[ ]:





# In[18]:


map_dict = {'yes':1,'no':0}

data["dependency"] = data["dependency"].replace(map_dict).astype('float64')
data["edjefe"] = data["edjefe"].replace(map_dict).astype('float64')
data["edjefa"] = data["edjefa"].replace(map_dict).astype('float64')


# In[ ]:





# In[19]:


data["dependency"] = data["dependency"].astype('float64')
data["edjefe"] = data["edjefe"].astype('float64')
data["edjefa"] = data["edjefa"].astype('float64')


# In[20]:


data[data["Target"].notnull()].groupby("idhogar")["Target"].nunique().value_counts()


# In[21]:


# Liste des individus concernés
data_multi_target = data[data["Target"].notnull()].groupby("idhogar")["Target"].apply(lambda x : x.nunique()==1)

data_multi_target = data_multi_target[data_multi_target != True]


# In[22]:


data_multi_target.shape


# In[23]:


data_multi_target.index[1]


# In[24]:


data[data["idhogar"] == data_multi_target.index[0]][['idhogar', 'parentesco1', 'Target']]


# In[25]:


for id in data_multi_target.index:
    True_label = data[(data['idhogar'] == id) & ( data['parentesco1'] == 1)]["Target"] 
    data.loc[data['idhogar'] == id, "Target"]  = np.where(True_label.notnull(), True_label, data.loc[data['idhogar'] == id, "Target"])


# In[26]:


data['Target'].isnull().sum()


# In[27]:


# Number of missing in each column
missing = pd.DataFrame(data.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(data)

missing.sort_values('percent', ascending = False).head(10).drop('Target')


# In[28]:


# Distribution des nombres de tablettes dans le foyer
data[data['parentesco1'] == 1]['v18q1'].value_counts().sort_index().plot.bar( figsize = (8,6),edgecolor ='k', linewidth = 2)
# Formatting


# In[29]:


data[data['parentesco1'] == 1]['v18q1'].value_counts()


# In[30]:


# Répartition de la variable de type individu par rapport à la variable de type foyer
data[data['parentesco1'] == 1].groupby('v18q1')['v18q'].sum()


# In[31]:


data['v18q1'] = data['v18q1'].fillna(0)


# In[32]:


tipovivi = data.columns.str.startswith('tipovivi')


# In[33]:


col = ["v2a1","tipovivi1","tipovivi2","tipovivi3","tipovivi4","tipovivi5"]
data.loc[(data["v2a1"].isnull() ) & (data["parentesco1"] ==1),col].sum().plot.bar()


# In[34]:


data.loc[(data['tipovivi1'] == 1 & data['v2a1'].isnull() ), 'v2a1'] = 0


# In[35]:


# Pour les autres valeurs manquantes et vu le manque d'informations, on propose d'imputer par la médiane (Voir partie modèle)

data.loc[data['v2a1'].isnull(), 'v2a1'] = 0


# In[36]:


# On vérifie s'il n y a pas des incohérences entre l'âge et cette variable.
data[data['rez_esc'].notnull()]['age'].describe()


# In[37]:


# Imputation des observation ayant un âge en dehors de l'intervalle 7 à 19 ans
data.loc[((data['age'] > 19) | (data['age'] < 7)) & (data['rez_esc'].isnull()) ,'rez_esc'] = 0


# In[38]:


# La valeur ma maximale prise par cette variable est 5. Or certaines obsevations ont des valeurs supérieurs.

data['rez_esc'].unique()


# In[39]:


# On va donc remplacer toute valeur supérieur à 5 par le max

data.loc[data['rez_esc'] > 5, 'rez_esc']  = 5


# In[40]:


# Pour les autres valeurs observations et vu le manque d'informations, on propose d'imputer par la médiane (Voir partie modèle)
data.loc[data['rez_esc'].isnull(),'rez_esc' ] = 0


# In[ ]:





# In[41]:


# Au vu du nombre d'observations concernées, on impute avec 0
data.loc[data['meaneduc'].isnull(),'meaneduc' ] = 0


# In[42]:


# Pour être cohérent avec l'imputation ci-dessus, on impute avec 0
data.loc[data['SQBmeaned'].isnull(), 'SQBmeaned'] = 0


# In[ ]:





# In[43]:


data.isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[44]:


head = df_train.loc[(data["parentesco1"] ==1) & (data["Target"].notnull())].copy()


# In[45]:


distrib_target = head['Target'].value_counts().reset_index().rename(columns = {'index' : 'level' })


# In[46]:


distrib_target['Household_type'] = distrib_target['level'].map(poverty_mapping)


# In[47]:


distrib_target.head()


# In[48]:


sns.set(style = 'whitegrid', font_scale=1.4)
fig = plt.subplots(figsize=(15, 8))
ax = sns.barplot(x = 'Household_type', y = 'Target', data = distrib_target, palette='Accent', ci = None).set_title('Distribution de la pauvereté')


# In[49]:


ids = ['Id', 'idhogar']
target = ['Target']


# In[50]:


ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 
            'instlevel9', 'mobilephone']

ind_ordered = ['rez_esc', 'escolari', 'age']


# In[51]:


hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 
           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2']

hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 
              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',
              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']

hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']
           
           


# In[52]:


sqr_ = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 
        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']


# In[53]:


head = data.loc[(data["parentesco1"] ==1), ids + hh_bool + hh_ordered + hh_cont + target].copy()


# In[54]:


head.head()


# In[55]:


head.shape


# In[56]:


# Matrice de corrélation
corr_matrix = head.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

to_drop


# In[57]:


sns.heatmap(corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.95, corr_matrix['tamhog'].abs() > 0.95],
            annot=True, cmap = plt.cm.autumn_r, fmt='.3f');


# In[58]:


"""
r4t3, tamhog, hhsize et hogar_total sont des variables représentant le nombre de personnes dans le foyer.
Nous allons en garder seulement une variable : hhsize.
"""
head.drop(labels = ['tamhog', 'hogar_total', 'r4t3'], axis = 1, inplace = True)


# In[59]:


sns.heatmap(corr_matrix.loc[corr_matrix['coopele'].abs() > 0.95, corr_matrix['coopele'].abs() > 0.95],
            annot=True, cmap = plt.cm.autumn_r, fmt='.3f');


# In[60]:


sns.heatmap(corr_matrix.loc[corr_matrix['area2'].abs() > 0.95, corr_matrix['area2'].abs() > 0.95],
            annot=True, cmap = plt.cm.autumn_r, fmt='.3f');


# In[61]:


# La variable area1 est très corrélées avec la variable area2
head.drop(labels = 'area2', axis = 1, inplace = True)


# In[62]:


# Etat des murs
head['epared_argmax'] = np.argmax(np.array(head[['epared1', 'epared2', 'epared3']]),
                           axis = 1)


# In[63]:


# Etat du toit
head['etecho_argmax'] = np.argmax(np.array(head[['etecho1', 'etecho2', 'etecho3']]),
                           axis = 1)


# In[64]:


# Etat du sol
head['eviv_argmax'] = np.argmax(np.array(head[['eviv1', 'eviv2', 'eviv3']]),
                           axis = 1)


# In[65]:


# Disponibilité de l'eau inside/outside/no water
head['abastagua_argmax'] = np.argmax(np.array(head[[ 'abastaguano',  'abastaguafuera','abastaguadentro']]),
                           axis = 1)


# In[66]:


# Connexion à un réseau élctriqueno elec/coopérative/public/privé
head['elec_argmax'] = np.argmax(np.array(head[['noelec','coopele', 'public', 'planpri']]),
                           axis = 1)


# In[67]:


# Propriétaire maison
head['tipovivi_argmax'] = np.argmax(np.array(head[['tipovivi5','tipovivi4', 'tipovivi3', 'tipovivi2','tipovivi1']]),
                           axis = 1)


# In[68]:


head['electro_house'] = head['refrig'] + head['computer'] + head['television'] 


# In[69]:


head['diff_tamviv_hhsize'] = head['tamviv'] - head['hhsize']


# In[70]:


# Variables utilisées dans la création de nouvelles variables
head.drop(labels = [ 'epared1', 'epared2', 'epared3','etecho1', 'etecho2', 'etecho3',
                          'eviv1', 'eviv2', 'eviv3',
                          'abastaguano',  'abastaguafuera','abastaguadentro',
                          'noelec','coopele', 'public', 'planpri',
                          'refrig', 'computer','television',
                          'tipovivi5','tipovivi4', 'tipovivi3', 'tipovivi2','tipovivi1',
                          'tamviv']
                 , axis = 1, inplace = True)


# In[71]:


head.head()


# In[72]:


# On ajoute les deux variables tamviv et hhsize pour calculer des moyennes par foyer
indiv = data[ids + ind_bool+ ind_ordered + target +['tamviv','hhsize']].copy()
indiv.shape


# In[73]:


# Matrice de corrélation
corr_matrix = indiv.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

to_drop


# In[74]:


sns.heatmap(corr_matrix.loc[corr_matrix['female'].abs() > 0.95, corr_matrix['female'].abs() > 0.95],
            annot=True, cmap = plt.cm.autumn_r, fmt='.3f');


# In[75]:


# La variable female est très corrélée avec male
indiv.drop(labels = 'male', axis = 1, inplace = True)


# In[76]:


indiv['instlevel_argmax'] = np.argmax(np.array(indiv[['instlevel1', 'instlevel2','instlevel3','instlevel4','instlevel5','instlevel6','instlevel7','instlevel8','instlevel9']]),
                           axis = 1)


# In[77]:


indiv['indiv_digital'] = indiv['mobilephone'] + indiv['v18q']


# In[78]:


indiv.head()


# In[79]:


# Variables utilisées dans la création de nouvelles variables
indiv.drop(labels = ['instlevel1', 'instlevel2','instlevel3','instlevel4','instlevel5',
                           'instlevel6','instlevel7','instlevel8','instlevel9',
                    'mobilephone', 'v18q'
                    ]
                 , axis = 1, inplace = True)


# In[80]:


indiv_pivot = pd.pivot_table(indiv,index=["idhogar"],
                            values=[
                                'age',
                                'rez_esc', 
                                'escolari',
                                'dis',
                                'female',
                                'instlevel_argmax',
                                'indiv_digital',
                                'tamviv',
                                'hhsize' 
                                   ],
                            aggfunc={
                             'age':np.sum,
                             'rez_esc':np.sum,
                             'escolari':np.sum, 
                             'dis' : np.sum,
                             'female' : np.sum,
                             'instlevel_argmax':[np.max,np.min,np.mean] , 
                             'indiv_digital':[np.max,np.min,np.mean] , 
                             'tamviv':[np.max,np.min] , 
                             'hhsize':[np.max,np.min]
                             
                },
                fill_value=0)


# In[81]:


indiv_toframe = pd.DataFrame(indiv_pivot.to_records())


# In[82]:


indiv_toframe.columns


# In[83]:


# Renommage des colonnes
indiv_toframe.columns = [hdr.replace("('", "").replace("', '", "_").replace("')", "")                      for hdr in indiv_toframe.columns]


# In[84]:


# Age moyen par foyer
indiv_toframe['mean_age_hhsize'] = indiv_toframe['age_sum'] / indiv_toframe['hhsize_amax']
# Age moyen des personnes vivant ds la foyer
indiv_toframe['mean_age_tamviv'] = indiv_toframe['age_sum'] / indiv_toframe['tamviv_amax']


# In[85]:


# Nbre d'années d'éducation moyen par foyer
indiv_toframe['mean_escolari_hhsize'] = indiv_toframe['escolari_sum'] / indiv_toframe['hhsize_amax']
# Nbre d'années d'éducation moyen des personnes vivant ds la foyer
indiv_toframe['mean_escolari_tamviv'] = indiv_toframe['escolari_sum'] / indiv_toframe['tamviv_amax']


# In[86]:


# Nbre d'années d'éducation moyen par foyer
indiv_toframe['mean_rez_esc_hhsize'] = indiv_toframe['rez_esc_sum'] / indiv_toframe['hhsize_amax']
# Nbre d'années d'éducation moyen des personnes vivant ds la foyer
indiv_toframe['mean_rez_esc_tamviv'] = indiv_toframe['rez_esc_sum'] / indiv_toframe['tamviv_amax']


# In[87]:


# Nbre moyen de femmes par foyer
indiv_toframe['mean_female_hhsize'] = indiv_toframe['female_sum'] / indiv_toframe['hhsize_amax']
# NNbre moyen de femmes vivant ds la foyer
indiv_toframe['mean_female_tamviv'] = indiv_toframe['female_sum'] / indiv_toframe['tamviv_amax']


# In[88]:


# Nbre moyen de personnes handicapée par foyer
indiv_toframe['mean_dis_hhsize'] = indiv_toframe['dis_sum'] / indiv_toframe['hhsize_amax']
# Nbre moyen de personnes handicapéevivant ds la foyer
indiv_toframe['mean_dis_tamviv'] = indiv_toframe['dis_sum'] / indiv_toframe['tamviv_amax']


# In[89]:


indiv_toframe.head()


# In[90]:


head_agg = head.merge(indiv_toframe, on = 'idhogar',
                             how = 'left')


# In[91]:


head_agg.head()


# In[92]:


head_agg[head_agg['Target'].notnull()].shape


# In[93]:


head_agg[head_agg['Target'].isnull()].shape


# In[ ]:





# In[94]:


# Labels for training
train_labels = np.array(list(head_agg[head_agg['Target'].notnull()]['Target'].astype(np.uint8)))

# Extract the training data
train_set = head_agg[head_agg['Target'].notnull()].drop(labels = ['Id', 'idhogar', 'Target'], axis = 1)
test_set = head_agg[head_agg['Target'].isnull()].drop(labels = ['Id', 'idhogar', 'Target'], axis = 1)

# Submission base which is used for making submissions to the competition
submission_base = df_test[['Id', 'idhogar']].copy()


# In[95]:


sm = SMOTE(random_state = 33)


# In[96]:


train_set_balanced, train_labels_balanced = sm.fit_sample(train_set, train_labels)


# In[97]:


train_set.shape
train_set_balanced.shape


# In[98]:


train_labels_balanced.shape


# In[99]:


train_labels.shape


# In[100]:


test_set.shape


# In[101]:


submission_base.shape


# In[102]:


features = list(train_set.columns)
# Imputation avec la médiane

pipeline = Pipeline([('imputer', Imputer(strategy = 'median')), 
                      ('scaler', MinMaxScaler())])

# Fit and transform training data
train_set = pipeline.fit_transform(train_set)
test_set = pipeline.transform(test_set)


# In[103]:


# Fit and transform training balanced data
train_set_balanced = pipeline.fit_transform(train_set_balanced)


# In[104]:


model_no_smote = RandomForestClassifier(n_estimators=100, random_state=10, 
                               n_jobs = -1)
# 10 fold cross validation
cv_score_no_smote = cross_val_score(model_no_smote, train_set, train_labels, cv = 10, scoring = scorer)

print(f'10 Fold Cross Validation F1 Score = {round(cv_score_no_smote.mean(), 4)} with std = {round(cv_score_no_smote.std(), 4)}')


# In[105]:


model_smote = RandomForestClassifier(n_estimators=100, random_state=10, 
                               n_jobs = -1)
# 10 fold cross validation
cv_score_smote = cross_val_score(model_smote, train_set_balanced, train_labels_balanced, cv = 10, scoring = scorer)

print(f'10 Fold Cross Validation F1 Score = {round(cv_score_smote.mean(), 4)} with std = {round(cv_score_smote.std(), 4)}')


# In[ ]:





# In[106]:


model_no_smote.fit(train_set, train_labels)

# Varibales importantes sur le jeu de données non rééquilibré
feature_importances_no_smote = pd.DataFrame({'feature': features, 'importance': model_no_smote.feature_importances_})
feature_importances_no_smote.head()


# In[107]:


model_smote.fit(train_set_balanced, train_labels_balanced)

# Varibales importantes sur le jeu de données rééquilibré
feature_importances_smote = pd.DataFrame({'feature': features, 'importance': model_smote.feature_importances_})
feature_importances_smote.head()


# In[ ]:





# In[108]:


def plot_feature_importances(df, n = 10, threshold = None):
    """Plots n most important features. Also plots the cumulative importance if
    threshold is specified and prints the number of features needed to reach threshold cumulative importance.
    Intended for use with any tree-based feature importances. 
    
    Args:
        df (dataframe): Dataframe of feature importances. Columns must be "feature" and "importance".
    
        n (int): Number of most important features to plot. Default is 15.
    
        threshold (float): Threshold for cumulative importance plot. If not provided, no plot is made. Default is None.
        
    Returns:
        df (dataframe): Dataframe ordered by feature importances with a normalized column (sums to 1) 
                        and a cumulative importance column
    
    Note:
    
        * Normalization in this case means sums to 1. 
        * Cumulative importance is calculated by summing features from most to least important
        * A threshold of 0.9 will show the most important features needed to reach 90% of cumulative importance
    
    """
    plt.style.use('fivethirtyeight')
    
    # Sort features with most important at the head
    df = df.sort_values('importance', ascending = False).reset_index(drop = True)
    
    # Normalize the feature importances to add up to one and calculate cumulative importance
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    
    plt.rcParams['font.size'] = 12
    
    # Bar plot of n most important features
    df.loc[:n, :].plot.barh(y = 'importance_normalized', 
                            x = 'feature', color = 'darkgreen', 
                            edgecolor = 'k', figsize = (12, 8),
                            legend = False, linewidth = 2)

    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 
    plt.title(f'{n} Most Important Features', size = 18)
    plt.gca().invert_yaxis()
    
    
    if threshold:
        # Cumulative importance plot
        plt.figure(figsize = (8, 6))
        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')
        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 
        plt.title('Cumulative Feature Importance', size = 18);
        
        # Number of features needed for threshold cumulative importance
        # This is the index (will need to add 1 for the actual number)
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
        
        # Add vertical line to plot
        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.05, linestyles = '--', colors = 'red')
        plt.show();
        
        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 
                                                                                  100 * threshold))
    
    return df


# In[109]:


norm_fi_no_smote  = plot_feature_importances(feature_importances_no_smote, threshold=0.95)


# In[110]:


norm_fi_smote = plot_feature_importances(feature_importances_smote, threshold=0.95)


# In[111]:


test_ids = list(head.loc[head['Target'].isnull(), 'idhogar'])


# In[112]:


def macro_f1_score(labels, predictions):
    # Reshape the predictions as needed
    predictions = predictions.reshape(len(np.unique(labels)), -1 ).argmax(axis = 0)
    
    metric_value = f1_score(labels, predictions, average = 'macro')
    
    # Return is name, value, is_higher_better
    return 'macro_f1', metric_value, True


# In[113]:



# Paramétres pour un test
params = {'boosting_type': 'dart', 
                  'colsample_bytree': 0.88, 
                  'learning_rate': 0.028, 
                   'min_child_samples': 10, 
                   'num_leaves': 36, 'reg_alpha': 0.76, 
                   'reg_lambda': 0.43, 
                   'subsample_for_bin': 40000, 
                   'subsample': 0.54, 
                   'class_weight': 'balanced'}
    
# Modèles
model = lgb.LGBMClassifier(**params, objective = 'multiclass', 
                               n_jobs = -1, n_estimators = 10000,
                               random_state = 10)
    
# Using stratified kfold cross validation
strkfold = StratifiedKFold(n_splits = 5, shuffle = True)


# In[114]:


strkfold.get_n_splits(train_set_balanced, train_labels_balanced)


# In[115]:


for train_index, test_index in strkfold.split(train_set, train_labels):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = train_set[train_index], train_set[test_index]
    y_train, y_test = train_labels[train_index], train_labels[test_index]


# In[116]:


for train_index, test_index in strkfold.split(train_set_balanced, train_labels_balanced):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train_balanced, X_test_balanced = train_set_balanced[train_index], train_set_balanced[test_index]
    y_train_balanced, y_test_balanced = train_labels_balanced[train_index], train_labels_balanced[test_index]


# In[117]:


model_lgb = model.fit(X_train, y_train, early_stopping_rounds =20, 
                  eval_metric = macro_f1_score,
                  eval_set = [(X_train, y_train), (X_test, y_test)],
                  eval_names = ['train', 'valid'],
                  verbose = 200)


# In[118]:


predictions = model_lgb.predict(test_set)
predictions = pd.DataFrame({'idhogar': test_ids,
                            'Target': predictions})

     # Make a submission dataframe
submission = submission_base.merge(predictions, 
                                       on = 'idhogar',
                                       how = 'left').drop(labels = ['idhogar'], axis = 1)
    
    # Fill in households missing a head
lgb_submission = submission['Target'] = submission['Target'].fillna(4).astype(np.int8)


lgb_submission.to_csv('lgb_submission.csv', index = False)


# In[119]:


model_lgb_balanced = model.fit(X_train_balanced, y_train_balanced, early_stopping_rounds = 20, 
                  eval_metric = macro_f1_score,
                  eval_set = [(X_train_balanced, y_train_balanced), (X_test_balanced, y_test_balanced)],
                  eval_names = ['train_balanced', 'valid_balanced'],
                  verbose = 200)


# In[120]:


predictions = model_lgb_balanced.predict(test_set)
predictions = pd.DataFrame({'idhogar': test_ids,
                            'Target': predictions})

     # Make a submission dataframe
submission = submission_base.merge(predictions, 
                                       on = 'idhogar',
                                       how = 'left').drop(labels = ['idhogar'], axis = 1)
    
    # Fill in households missing a head
lgb_balanced_submission = submission['Target'] = submission['Target'].fillna(4).astype(np.int8)


lgb_balanced_submission.to_csv('lgb_submission.csv', index = False)


# In[121]:


lgb_balanced_submission.shape


# In[122]:


test_ids = list(head.loc[head['Target'].isnull(), 'idhogar'])


# In[123]:


len(test_ids)


# In[124]:


def submit(model, train, train_labels, test, test_ids):
    """Train and test a model on the dataset"""
    
    # Train on the data
    model.fit(train, train_labels)
    predictions = model.predict(test)
    predictions = pd.DataFrame({'idhogar': test_ids,
                               'Target': predictions})

     # Make a submission dataframe
    submission = submission_base.merge(predictions, 
                                       on = 'idhogar',
                                       how = 'left').drop(labels = ['idhogar'], axis = 1)
    
    # Fill in households missing a head
    submission['Target'] = submission['Target'].fillna(4).astype(np.int8)

    return submission


# In[125]:


rf_submission = submit(RandomForestClassifier(n_estimators = 100, 
                                              random_state=10, n_jobs = -1), 
                         train_set, train_labels, test_set, test_ids)

rf_submission.to_csv('rf_submission.csv', index = False)


# In[126]:


rf_submission.shape


# In[127]:


rf_balanced_submission = submit(RandomForestClassifier(n_estimators = 100, 
                                              random_state=10, n_jobs = -1), 
                         train_set_balanced, train_labels_balanced, test_set, test_ids)

rf_balanced_submission.to_csv('rf_balanced_submission.csv', index = False)


# In[128]:


rf_balanced_submission.shape

