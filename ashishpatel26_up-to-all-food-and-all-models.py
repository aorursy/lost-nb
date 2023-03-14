#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from collections import Counter
from pprint import pprint

pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)

import re
from nltk.stem import WordNetLemmatizer

# Grid search for optimal parameters of the model
from sklearn.model_selection import GridSearchCV

# Model modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

# modules for # estimate
from sklearn.model_selection import cross_val_score
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# modules for encoding features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Modules for dividing a data set
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from gensim.models import word2vec




train = pd.read_json('../input/train.json', encoding = 'UTF-8')
test = pd.read_json('../input/test.json', encoding = 'UTF-8')




df = train.copy()

print(df.shape)
df.head(2)




df.count()




df.isnull().sum()




print('Cuisine is {}.'.format(len(df.cuisine.value_counts())))
df.cuisine.value_counts()




# cuisine type visualization

plt.style.use('ggplot')
df.cuisine.value_counts().plot(kind = 'bar',
                              title='Cuisine Types',
                              figsize=(20,5),
                              legend=True,
                              fontsize=12)
plt.ylabel("Number of Recipes", fontsize=12)
plt.show()




### Work to count ingredients per recipe per row

# Here, the ingredient will be counted one by one.
bag_of_ingredients = [Counter (ingredient) for ingredient in df.ingredients]

# Number of each ingredient by type
sum_of_ingredients = sum (bag_of_ingredients, Counter ())

########################################################################################

### Work to put sum_of_ingredients in dataframe

# dict -> list -> dataframe
sum_of_ingredients_dict = dict (sum_of_ingredients)
sum_of_ingredients_list = list (sum_of_ingredients_dict.items ())

ingredients_df = pd.DataFrame (sum_of_ingredients_list)
ingredients_df.columns = ['ingredient', 'count']
ingredients_df.tail (2)

print ('Before the preprocessing, the total ingredients are {}.'. format (len (ingredients_df)))




ingredients_df.head(20)




def pre_processing_(recipe):
    
    wnl = WordNetLemmatizer()
    
    # 1. lower 함수를 이용하여 대문자를 소문자로 변경
    recipe = [str.lower(ingredient) for ingredient in recipe]
    recipe = [delete_brand_(ingredient) for ingredient in recipe]
    recipe = [delete_state_(ingredient) for ingredient in recipe]
    recipe = [delete_comma_(ingredient) for ingredient in recipe]
    recipe = [original_(ingredient) for ingredient in recipe]
    recipe = [delete_space_(ingredient) for ingredient in recipe]

    return recipe

# 2. 상품명을 제거하는 함수
def delete_brand_(ingredient):

    # '®'이 있는 브랜드
    ingredient = re.sub("country crock|i can't believe it's not butter!|bertolli|oreo|hellmann's"
                        , '', ingredient)
    ingredient = re.sub("red gold|hidden valley|original ranch|frank's|redhot|lipton", '', ingredient)
    ingredient = re.sub("recipe secrets|eggland's best|hidden valley|best foods|knorr|land o lakes"
                        , '', ingredient)
    ingredient = re.sub("sargento|johnsonville|breyers|diamond crystal|taco bell|bacardi", '', ingredient)
    ingredient = re.sub("mccormick|crystal farms|yoplait|mazola|new york style panetini", '', ingredient)
    ingredient = re.sub("ragu|soy vay|tabasco|truvía|crescent recipe creations|spice islands", '', ingredient)
    ingredient = re.sub("wish-bone|honeysuckle white|pasta sides|fiesta sides", '', ingredient)
    ingredient = re.sub("veri veri teriyaki|artisan blends|home originals|greek yogurt|original ranch"
                        , '', ingredient)
    ingredient = re.sub("jonshonville", '', ingredient)

    # '™'이 있는 브랜드
    ingredient = re.sub("old el paso|pillsbury|progresso|betty crocker|green giant|hellmannâ€", '', ingredient)

    # 'oscar mayer deli fresh smoked' 브랜드
    ingredient = re.sub("oscar mayer deli fresh smoked", '', ingredient)

    return ingredient

# 3. 재료 손질, 상태를 제거하는 함수
def delete_state_(ingredient):

    ingredient = re.sub('frozen|chopped|ground|fresh|powdered', '', ingredient)
    ingredient = re.sub('sharp|crushed|grilled|roasted|sliced', '', ingredient)
    ingredient = re.sub('cooked|shredded|cracked|minced|finely', '', ingredient)        
     return ingredient

# 4. 콤마 뒤에 있는 재료손질방법을 제거하는 함수
def delete_comma_(ingredient):

    ingredient = ingredient.split(',')
    ingredient = ingredient[0]

    return ingredient

## 그외 전처리 함수 (숫자제거, 특수문자제거, 원형으로변경)
def original_(ingredient):

    # 숫자제거
    ingredient = re.sub('[0-9]', '', ingredient)

    # 특수문자 제거
    ingredient = ingredient.replace("oz.", '')
    ingredient = re.sub('[&%()®™/]', '', ingredient)
    ingredient = re.sub('[-.]', '', ingredient)

    # lemmatize를 이용하여 단어를 원형으로 변경
    ingredient = wnl.lemmatize(ingredient)

    return ingredient

# 양 끝 공백을 제거하는 함수
def delete_space_(ingredient):
    ingredient = ingredient.strip()
    return ingredient




df['ingredients'] = df['ingredients'].apply(lambda x : pre_processing_(x))




get_ipython().run_cell_magic('time', '', "### 각 row 마다의 recipe 별 ingredient를 count하기 위한 작업\n\n# 여기서는 ingredient가 각 1개씩 count 될 것이다.\nbag_of_ingredients = [Counter(ingredient) for ingredient in df.ingredients]\n\n# 각 ingredients의 종류별 개수\nsum_of_ingredients = sum(bag_of_ingredients, Counter())\n\n########################################################################################\n\n### sum_of_ingredients를 dataframe에 넣기 위한 작업\n\n# dict -> list -> dataframe\nsum_of_ingredients_dict = dict(sum_of_ingredients)\nsum_of_ingredients_list = list(sum_of_ingredients_dict.items())\n\ningredients_df = pd.DataFrame(sum_of_ingredients_list)\ningredients_df.columns = ['ingredient', 'count']\ningredients_df.tail(2)\n\nprint('전처리 후 ingredient는 총 {}개 입니다.'.format(len(ingredients_df)))")




df['ingredients_train'] = df['ingredients'].apply(','.join)




"""
TfidfVectorizer : 문서 집합으로부터 단어의 수를 세고, TF-IDF 방식으로 단어의 가중치를 조정한 카운트 행렬을 만든다.
"""

tfv = TfidfVectorizer()
X = tfv.fit_transform(df['ingredients_train'].values)

print(list(tfv.vocabulary_.keys())[:10])




print(X.shape)




print(type(X))




print(X[2999])
# 구조파악하기




Lec = LabelEncoder()
train_target_value = Lec.fit_transform(df['cuisine'].values)

print(train_target_value.shape)




print(train_target_value[:20])




print(Lec.classes_)




X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, train_target_value)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)




""" Random Forest Model """
def RandomForestClassifier_():
    
    pipe = Pipeline([('classifier', RandomForestClassifier())])
    hyperparameter_space = [{'classifier' : [RandomForestClassifier()], 
                             'classifier__n_estimators' : [350, 375, 400],
                             'classifier__max_features' : ['sqrt', 'log2']}]
    
    grid = GridSearchCV(pipe, hyperparameter_space, cv=3)
    grid.fit(X_train, y_train)

    cuisine = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek',
               'indian', 'irish', 'italian', 'jamaican', 'japanese', 'korean', 'mexican', 'moroccan',
               'russian', 'southern_us', 'spanish', 'thai', 'vietnamese']

    print (classification_report(y_test, grid.predict(X_test), digits=4, target_names=cuisine))
    
    return print("Best parameters:\n{}\n".format(grid.best_params_), 
                 "Best score : {}\n".format(grid.best_score_),
                 "Test score : {}".format(grid.score(X_test, y_test)))
    
""" SVM Model """
def SVM_():
    pipe = Pipeline([('classifier', SVC())])
    hyperparameter_space = [{'classifier': [SVC()],
                             'classifier__gamma': ['auto'],
                             'classifier__C' : [10, 15]}]
    grid = GridSearchCV(pipe, hyperparameter_space, cv=3)
    grid.fit(X_train, y_train)
    
    cuisine = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek',
               'indian', 'irish', 'italian', 'jamaican', 'japanese', 'korean', 'mexican', 'moroccan',
               'russian', 'southern_us', 'spanish', 'thai', 'vietnamese']

    print (classification_report(y_test, grid.predict(X_test), digits=4, target_names=cuisine))
    
    return print("Best parameters:\n{}\n".format(grid.best_params_), 
                 "Best score : {}\n".format(grid.best_score_),
                 "Test score : {}".format(grid.score(X_test, y_test)))


""" KNN Model """
def KNN_():
    
    knn = KNeighborsClassifier()
    
    pipe = Pipeline([('classifier', knn)])
    hyperparameter_space = [{'classifier': [knn],
                             'classifier__n_neighbors': [15, 20, 25],
                             'classifier__leaf_size' : [20, 25, 30]}]
    grid = GridSearchCV(pipe, hyperparameter_space, cv=3)
    grid.fit(X_train, y_train)
        
        
    cuisine = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek',
               'indian', 'irish', 'italian', 'jamaican', 'japanese', 'korean', 'mexican', 'moroccan',
               'russian', 'southern_us', 'spanish', 'thai', 'vietnamese']

    print (classification_report(y_test, knn.predict(X_test), digits=4, target_names=cuisine))
    
    return print("Best parameters:\n{}\n".format(grid.best_params_), 
                 "Best score : {}\n".format(grid.best_score_),
                 "Test score : {}".format(grid.score(X_test, y_test)))

""" Xgboost Model"""
def Xgboost_():
    pipe = Pipeline([('classifier', xgb.XGBClassifier())])
    hyperparameter_space = [{'classifier': [xgb.XGBClassifier()],
                             'classifier__max_depth': [3, 4, 5],
                             'classifier__n_estimators' : [350, 375, 400]}]
    grid = GridSearchCV(pipe, hyperparameter_space, cv=3)
    grid.fit(X_train, y_train)
    
    cuisine = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek',
               'indian', 'irish', 'italian', 'jamaican', 'japanese', 'korean', 'mexican', 'moroccan',
               'russian', 'southern_us', 'spanish', 'thai', 'vietnamese']

    print (classification_report(y_test, grid.predict(X_test), digits=4, target_names=cuisine))
    
    return print("Best parameters:\n{}\n".format(grid.best_params_), 
                 "Best score : {}\n".format(grid.best_score_),
                 "Test score : {}".format(grid.score(X_test, y_test)))


""" Decision Tree Model """
def DecisionTree_():
    pipe = Pipeline([('classifier', DecisionTreeClassifier())])
    hyperparameter_space = [{'classifier': [DecisionTreeClassifier()],
                             'classifier__max_depth': [50, 60, 70]}]
    grid = GridSearchCV(pipe, hyperparameter_space, cv=3)
    grid.fit(X_train, y_train)
    
    cuisine = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek',
               'indian', 'irish', 'italian', 'jamaican', 'japanese', 'korean', 'mexican', 'moroccan',
               'russian', 'southern_us', 'spanish', 'thai', 'vietnamese']

    print (classification_report(y_test, grid.predict(X_test), digits=4, target_names=cuisine))
    
    return print("Best parameters:\n{}\n".format(grid.best_params_), 
                 "Best score : {}\n".format(grid.best_score_),
                 "Test score : {}".format(grid.score(X_test, y_test)))

def Neural_network_():
    nn = MLPClassifier(hidden_layer_sizes=(400,500,400))
    nn.fit(X_train, y_train)
    
    cuisine = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek',
               'indian', 'irish', 'italian', 'jamaican', 'japanese', 'korean', 'mexican', 'moroccan',
               'russian', 'southern_us', 'spanish', 'thai', 'vietnamese']

    print (classification_report(y_test, nn.predict(X_test), digits=4, target_names=cuisine))
    return print("Test score : {}".format(nn.score(X_test, y_test)))




get_ipython().run_cell_magic('time', '', 'rf = RandomForestClassifier_()')




get_ipython().run_cell_magic('time', '', 'SVM = SVM_()')




get_ipython().run_cell_magic('time', '', 'KNN = KNN_()')




get_ipython().run_cell_magic('time', '', 'Xgboost_()')




get_ipython().run_cell_magic('time', '', 'DecisionTree_()')




get_ipython().run_cell_magic('time', '', 'Neural_network = Neural_network_()')




# Feature importance
pd.Series(xgbr.feature_importances_).plot(kind='bar')




# 1. all_ingredients set에 ingredients들을 담는다.
all_ingredients = set()
df['ingredients'].apply(lambda x : [all_ingredients.add(i) for i in list(x)])
#print(all_ingredients)

# 'ingredient' columns를 새로 만들면서, 각 ingredient가 해당 row의 recipe에 들어 있으면 True, 그렇지 않으면 False를 반환하게 함
for ingredient in all_ingredients:
    df[ingredient] = df['ingredients'].apply(lambda x : ingredient in x)
    
len(df.columns)




get_ipython().run_cell_magic('time', '', "\ncolumn_list = []\nfor col in df.columns:\n    column_list.append(col)\n    \ncolumn_list.remove('id')\ncolumn_list.remove('ingredients')\ncolumn_list.remove('cuisine')\n\nlen(column_list)\nprint(column_list[:10])\n\ndf[column_list] = df[column_list].astype(int) # False는 0으로, True는 1로")




# Copy
df_dummy = df.copy()

# 'id'와 'ingredients' columns는 더이상 필요가 없으므로 지운다.
del df_dummy['id']
del df_dummy['ingredients']

# 'cuisine' column을 지우기 위해 df_dummy를 copy
df_features = df_dummy.copy()

del df_features['cuisine']
df_features.tail(1)




Lec = LabelEncoder()
train_target_value = Lec.fit_transform(df_dummy['cuisine'].values)

print(train_target_value.shape)




print(train_target_value[:10])




print(Lec.classes_)




X_train, X_test, y_train, y_test = cross_validation.train_test_split(df_features, train_target_value)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)




get_ipython().run_cell_magic('time', '', '# fit method를 호출하여 주성분을 찾는다. 주성분은 200개로 한다.\npca = PCA(n_components=200, whiten=True, random_state=0).fit(X_train)\n# transform method를 호출해 데이터를 회전시키고 차원을 축소\nX_train_pca = pca.transform(X_train)\nX_test_pca = pca.transform(X_test)\n\nprint("X_train_pca.shape: {}".format(X_train_pca.shape))')




pca.components_




def Xgboost_():
    xgbr = xgb.XGBClassifier(
        n_estimators = 400,
        max_depth = 5
    ).fit(X_train_pca, y_train)
        
    cuisine = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek',
               'indian', 'irish', 'italian', 'jamaican', 'japanese', 'korean', 'mexican', 'moroccan',
               'russian', 'southern_us', 'spanish', 'thai', 'vietnamese']

    print (classification_report(y_test, xgbr.predict(X_test_pca), digits=4, target_names=cuisine))
    return print("Test score : {}".format(xgbr.score(X_test_pca, y_test)))




get_ipython().run_cell_magic('time', '', 'Xgboost = Xgboost_()')




def Neural_network_():
    nn = MLPClassifier(hidden_layer_sizes=(400,500,400))
    nn.fit(X_train_pca, y_train)
    
    cuisine = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek',
               'indian', 'irish', 'italian', 'jamaican', 'japanese', 'korean', 'mexican', 'moroccan',
               'russian', 'southern_us', 'spanish', 'thai', 'vietnamese']

    print (classification_report(y_test, nn.predict(X_test_pca), digits=4, target_names=cuisine))
    return print("Test score : {}".format(nn.score(X_test_pca, y_test)))




get_ipython().run_cell_magic('time', '', 'Neural_Network = Neural_network_()')




""" 

logic : df['cuisine']과 df['ingredients']를 리스트 형식으로 만들고 싶지만, 
        df['ingredients']가 각각 리스트 형식으로 되어있기 때문에 몇가지 작업이 필요하다.

1. df['ingredients'] 즉, 각 recipe 리스트를 하나의 리스트(called ingredient_list)에 넣는다.
   ingredient_list를 data frame에 넣어준다 (왜? 각 ingredient를 column으로 만들어 주기 위해)
2. df['cuisine']과 ingredient_list를 column으로 합친다. 그러면 하나의 data frame이 만들어짐
3. iterrows() 함수를 이용하여 각 row를 list로 만들어 준다.

"""

""" 1번 작업 """
ingredient_list = []

for elements in df['ingredients']:
    ingredient_list.append(elements)
    
## ingredient_list의 각 원소를 리스트 형식
#ingredient_list

ingredient_df = pd.DataFrame(ingredient_list)
#ingredient_df

""" 2번 작업 """
cuisine_list = []

for element in df['cuisine']:
    cuisine_list.append(element)

ingredient_df.insert(0, "cuisines", cuisine_list)
#ingredient_df.tail(2)

""" 3번 작업 """
temp = []

for row in ingredient_df.iterrows():
    index, data = row
    temp.append(data.tolist())
    
#temp

""" 예상치 못하게, temp안에 None 값이 들어 간것을 확인했다. None 값을 제거한다. """
new_temp = []

for list_element in temp:
    new_element = [x for x in list_element if x is not None]
    new_temp.append(new_element)
    
#new_temp[0]

""" word2vec 학습 """
model = word2vec.Word2Vec(new_temp, workers = 4, 
                         size = 300, min_count = 3, window = 10)

model.init_sims(replace=True)

# 학습이 잘 되었는지 확인
model.most_similar('korean')




cuisine_dict = dict(df.cuisine.value_counts().items())
cuisine_list = list(cuisine_dict.keys())




print("Bot: " 
      + "\t" + "I can tell you common ingredients often used in the country." 
      + "\n" + "\t" + "What kind of cuisine do you want to know?" 
      + "\n" + "\t" + "Cuisine list is here."
      + "\n" + "\t" + "====================[CUISINE LISE]===================="
      + "\n" + "\t" + "italian, mexican, southern_us, indian, chinese"
      + "\n" + "\t" + "french, cajun_creole, thai, japanese, greek"
      + "\n" + "\t" + "spanish, korean, vietnamese', moroccan, british"
      + "\n" + "\t" + "filipino, irish, jamaican, russian, brazilian"
      + "\n" + "\t" + "======================================================")
    
while True:

    user_question = input("User: " + "\t").strip()
    user_question_lower = user_question.lower()
    
    ingredient_list = []
    for cuisine in cuisine_list:
        
        if cuisine in user_question:
            for i in model.most_similar(cuisine):
                if i[0] not in cuisine_list:
                    ingredient_list.append(i[0])
            print("Bot: " + "\t" + "commonly used ingredients in {} food are {}".format(cuisine, ingredient_list)
                          + "\n" + "\t" + "Do you want to know more? yes or no")
                        
    if user_question == "":
        print("Bot: "+ "\t" + "What kind of cuisine do you want to know?"
                   + "\t" + "I can tell you common ingredients often used in the country." 
                   + "\n" + "\t" + "What kind of cuisine do you want to know?" 
                   + "\n" + "\t" + "Cuisine list is here."
                   + "\n" + "\t" + "====================[CUISINE LISE]===================="
                   + "\n" + "\t" + "italian, mexican, southern_us, indian, chinese"
                   + "\n" + "\t" + "french, cajun_creole, thai, japanese, greek"
                   + "\n" + "\t" + "spanish, korean, vietnamese', moroccan, british"
                   + "\n" + "\t" + "filipino, irish, jamaican, russian, brazilian"
                   + "\n" + "\t" + "======================================================")
        
    elif user_question == "yes":
        print("Bot: "+ "\t" + "What kind of cuisine do you want to know?"
                   + "\t" + "I can tell you common ingredients often used in the country." 
                   + "\n" + "\t" + "What kind of cuisine do you want to know?" 
                   + "\n" + "\t" + "Cuisine list is here."
                   + "\n" + "\t" + "====================[CUISINE LISE]===================="
                   + "\n" + "\t" + "italian, mexican, southern_us, indian, chinese"
                   + "\n" + "\t" + "french, cajun_creole, thai, japanese, greek"
                   + "\n" + "\t" + "spanish, korean, vietnamese', moroccan, british"
                   + "\n" + "\t" + "filipino, irish, jamaican, russian, brazilian"
                   + "\n" + "\t" + "======================================================")
                        
    elif user_question == "no":
        print("Bot: "+ "\t" + "See you again. Bye.")
        break






