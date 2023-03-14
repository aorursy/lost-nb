#!/usr/bin/env python
# coding: utf-8



import pandas as pd
from sklearn import preprocessing
import numpy as np
import tqdm

from kaggle.competitions import nflrush

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor




env = nflrush.make_env()
dataset = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
dataset.head()




unused_columns = ["GameId","PlayId","Team","TimeHandoff","TimeSnap"]
unique_columns = []
for c in dataset.columns:
    if c not in unused_columns+["PlayerBirthDate"] and len(set(dataset[c][:11]))!= 1:
        unique_columns.append(c)
        print(c," is unique")




ok = True
for i in range(0,509762,22):
    p=dataset["PlayId"][i]
    for j in range(1,22):
        if(p!=dataset["PlayId"][i+j]):
            ok=False
            break
print("train data is sorted by PlayId." if ok else "train data is not sorted by PlayId.")
ok = True
for i in range(0,509762,11):
    p=dataset["Team"][i]
    for j in range(1,11):
        if(p!=dataset["Team"][i+j]):
            ok=False
            break
print("train data is sorted by Team." if ok else "train data is not sorted by Team.")




lbl_dict = {}
for c in dataset.columns:
    if c == "DefensePersonnel":
        arr = [[int(s[0]) for s in t.split(", ")] for t in dataset["DefensePersonnel"]]
        dataset["DL"] = pd.Series([a[0] for a in arr])
        dataset["LB"] = pd.Series([a[1] for a in arr])
        dataset["DB"] = pd.Series([a[2] for a in arr])
    elif c == "GameClock":
        arr = [[int(s) for s in t.split(":")] for t in dataset["GameClock"]]
        dataset["GameHour"] = pd.Series([a[0] for a in arr])
    elif c == "PlayerBirthDate":
        arr = [[int(s) for s in t.split("/")] for t in dataset["PlayerBirthDate"]]
        dataset["BirthY"] = pd.Series([a[2] for a in arr])
    elif dataset[c].dtype=='object' and c not in unused_columns: 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(dataset[c].values))
        lbl_dict[c] = lbl
        dataset[c] = lbl.transform(list(dataset[c].values))
        
for col in unused_columns:
    try:        
        dataset = dataset.drop(col, axis=1)
    except:
        print(col + 'does not exist')
        continue
try:
    dataset = dataset.drop('DefensePersonnel', axis=1)
    dataset = dataset.drop('GameClock', axis=1)
    dataset = dataset.drop('PlayerBirthDate', axis=1)
except:
    print("col does not exist")




X = dataset.drop('Yards',axis=1)
y = dataset['Yards']

categorical_features = []
numeric_features = X.columns


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])





# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(45, input_dim=45, kernel_initializer='normal', activation='relu'))
    model.add(Dense(22, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model




clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=1))])
#Result of the tuner hyper parameter as in commented section above




clf.fit(X,y)




index = 0
for (test_df, sample_prediction_df) in tqdm.tqdm(env.iter_test()):
    for c in test_df.columns:
        if c == "DefensePersonnel":
            try:
                arr = [[int(s[0]) for s in t.split(", ")] for t in test_df["DefensePersonnel"]]
                test_df["DL"] = [a[0] for a in arr]
                test_df["LB"] = [a[1] for a in arr]
                test_df["DB"] = [a[2] for a in arr]
            except:
                test_df["DL"] = [np.nan for i in range(22)]
                test_df["LB"] = [np.nan for i in range(22)]
                test_df["DB"] = [np.nan for i in range(22)]
        elif c == "GameClock":
            try:
                arr = [[int(s) for s in t.split(":")] for t in test_df["GameClock"]]
                test_df["GameHour"] = pd.Series([a[0] for a in arr])
            except:
                test_df["GameHour"] = [np.nan for i in range(22)]
        elif c == "PlayerBirthDate":
            try:
                arr = [[int(s) for s in t.split("/")] for t in test_df["PlayerBirthDate"]]
                test_df["BirthY"] = pd.Series([a[2] for a in arr])
            except:
                test_df["BirthY"] = [np.nan for i in range(22)]
      
        elif c in lbl_dict and test_df[c].dtype=='object'and c not in unused_columns            and not pd.isnull(test_df[c]).any():
            try:
                test_df[c] = lbl_dict[c].transform(list(test_df[c].values))
            except:
                test_df[c] = [np.nan for i in range(22)]
    
    for col in unused_columns:
        try:        
            test_df = test_df.drop(col, axis=1)
        except:
            continue
    try:
        test_df = test_df.drop('DefensePersonnel', axis=1)
        test_df = test_df.drop('GameClock', axis=1)
        test_df = test_df.drop('PlayerBirthDate', axis=1)
    except:
        print("col does not exist")
    
    y_pred = np.zeros(199)
       
    y_pred_p1 = clf.predict(test_df)[0]
    y_pred_p = np.round(y_pred_p1)
    y_pred_p
    y_pred_p += 99
    for j in range(199):
        if j>=y_pred_p+10:
            y_pred[j]=1.0
        elif j>=y_pred_p-10:
            y_pred[j]=(j+10-y_pred_p)*0.05
    env.predict(pd.DataFrame(data=[y_pred],columns=sample_prediction_df.columns))
    index += 22
env.write_submission_file()

