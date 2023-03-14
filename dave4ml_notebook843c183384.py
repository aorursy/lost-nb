#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (10, 6)
limit_rows   = 7000000
df           = pd.read_csv("../input/train_ver2.csv",dtype={"sexo":str,
                                                    "ind_nuevo":str,
                                                    "ult_fec_cli_1t":str,
                                                    "indext":str, "segmento":str}, nrows=limit_rows)
unique_ids   = pd.Series(df["ncodpers"].unique())
limit_people = 1e4
unique_id    = unique_ids.sample(n=limit_people)
df           = df[df.ncodpers.isin(unique_id)]
#First convert the dates. There's fecha_dato, the row-identifier date, and fecha_alta, the date #that the customer joined.
df["fecha_dato"] = pd.to_datetime(df["fecha_dato"],format="%Y-%m-%d")
df["fecha_alta"] = pd.to_datetime(df["fecha_alta"],format="%Y-%m-%d")
# let's add a month column. 
df["month"] = pd.DatetimeIndex(df["fecha_dato"]).month
df["age"]   = pd.to_numeric(df["age"], errors="coerce")
#Let's separate the “age” distribution and move the outliers to the mean of the closest one.
df.loc[df.age < 18,"age"]  = df.loc[(df.age >= 18) & (df.age <= 30),"age"].mean(skipna=True)
df.loc[df.age > 100,"age"] = df.loc[(df.age >= 30) & (df.age <= 100),"age"].mean(skipna=True)
df["age"].fillna(df["age"].mean(),inplace=True)
df["age"]                  = df["age"].astype(int)
#Next ind_nuevo, which indicates whether a customer is new or not. Let's see if we can fill in #missing values by looking how many months of history these customers have.
months_active = df.loc[df["ind_nuevo"].isnull(),:].groupby("ncodpers", sort=False).size()
#Looks like these are all new customers, so replace accordingly.
df.loc[df["ind_nuevo"].isnull(),"ind_nuevo"] = 1
#Now antiguedad
df.antiguedad = pd.to_numeric(df.antiguedad,errors="coerce")
df.loc[df.antiguedad.isnull(),"antiguedad"] = df.antiguedad.min()
df.loc[df.antiguedad <0, "antiguedad"]
#Some entries don't have the date they joined the company. Just give them something in the #middle of the pack
dates=df.loc[:,"fecha_alta"].sort_values().reset_index()
median_date = int(np.median(dates.index.values))
df.loc[df.fecha_alta.isnull(),"fecha_alta"] = dates.loc[median_date,"fecha_alta"]
df.loc[df.indrel.isnull(),"indrel"] = 1
df.drop(["tipodom","cod_prov"],axis=1,inplace=True)
df.loc[df.ind_actividad_cliente.isnull(),"ind_actividad_cliente"] = df["ind_actividad_cliente"].median()
df.loc[df.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A"
df.loc[df.nomprov.isnull(),"nomprov"] = "UNKNOWN"
df.loc[df.renta.isnull(),"renta"] = 0.0
df["renta"]                  = df["renta"].astype(float)
df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0
string_data = df.select_dtypes(include=["object"])
missing_columns = [col for col in string_data if string_data[col].isnull().any()]
df.loc[df.indfall.isnull(),"indfall"] = "N"
df.loc[df.tiprel_1mes.isnull(),"tiprel_1mes"] = "A"
df.tiprel_1mes = df.tiprel_1mes.astype("category")
# As suggested by @StephenSmith
map_dict = { 1.0  : "1",
            "1.0" : "1",
            "1"   : "1",
            "3.0" : "3",
            "P"   : "P",
            3.0   : "3",
            2.0   : "2",
            "3"   : "3",
            "2.0" : "2",
            "4.0" : "4",
            "4"   : "4",
            "2"   : "2"}

df.indrel_1mes.fillna("P",inplace=True)
df.indrel_1mes = df.indrel_1mes.apply(lambda x: map_dict.get(x,x))
df.indrel_1mes = df.indrel_1mes.astype("category")
unknown_cols = [col for col in missing_columns if col not in ["indfall","tiprel_1mes","indrel_1mes"]]
for col in unknown_cols:
    df.loc[df[col].isnull(),col] = "UNKNOWN"
# let’s add a customer lifetime column in months
df["lifetime"] = (df.fecha_dato - df.fecha_alta).astype('timedelta64[M]')
#Convert the feature columns into integer values
feature_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
for col in feature_cols:
    df[col] = df[col].astype(int)
#Now for the main event. To study trends in customers adding or removing services, I will #create a label for each product and month that indicates whether a customer added, dropped #or maintained that service in that billing cycle. I will do this by assigning a numeric id to each #unique time stamp, and then matching each entry with the one from the previous month. The #difference in the indicator value for each product then gives the desired value.
unique_months = pd.DataFrame(pd.Series(df.fecha_dato.unique()).sort_values()).reset_index(drop=True)
unique_months["month_id"] = pd.Series(range(1,1+unique_months.size)) 
# start with month 1, not 0 to match what we already have
unique_months["month_next_id"] = 1 + unique_months["month_id"]
unique_months.rename(columns={0:"fecha_dato"},inplace=True)
df = pd.merge(df,unique_months,on="fecha_dato")
#Now I'll build a function that will convert differences month to month into a meaningful label. #Each month, a customer can either maintain their current status with a particular product, add #it, or drop it.
def status_change(x):
    diffs = x.diff().fillna(0)# first occurrence will be considered Maintained, 
    #which is a little lazy. A better way would be to check if 
    #the earliest date was the same as the earliest we have in the dataset
    #and consider those separately. Entries with earliest dates later than that have 
    #joined and should be labeled as "Added"
    label = ["Added" if i==1          else "Dropped" if i==-1          else "Maintained" for i in diffs]
    return label
df.loc[:, feature_cols] = df.loc[:, [i for i in feature_cols]+["ncodpers"]].groupby("ncodpers").transform(status_change)
df = pd.melt(df, id_vars   = [col for col in df.columns if col not in feature_cols],
            value_vars= [col for col in feature_cols])
df = df.loc[df.value!="Maintained",:]
#df.head()
df2 = df.copy()
df2 = df2.loc[df2.value=="Added",:]


#Feature columns all except fecha_dato, fecha_alta, tipodom and cod_prov. Add month
features = ["ncodpers","ind_empleado","pais_residencia","sexo","age","ind_nuevo","antiguedad","indrel","ult_fec_cli_1t","indrel_1mes","tiprel_1mes","indresi","indext","conyuemp","canal_entrada","indfall","nomprov","ind_actividad_cliente","renta","segmento","lifetime","month"]
X1 = df[features]
X2 = df2[features]
#labels = ["variable","value"]
#Y = df[labels]
y1 = df["value"]
y2 = df2["variable"]
del df, df2




#Jetzt geht es zu den wirklichen Testdaten
df           = pd.read_csv("../input/test_ver2.csv",dtype={"sexo":str,
                                                    "ind_nuevo":str,
                                                    "ult_fec_cli_1t":str,
                                                    "indext":str}, nrows=limit_rows)
unique_ids   = pd.Series(df["ncodpers"].unique())
limit_people = 1e4
unique_id    = unique_ids.sample(n=limit_people)
df           = df[df.ncodpers.isin(unique_id)]
#First convert the dates. There's fecha_dato, the row-identifier date, and fecha_alta, the date #that the customer joined.
df["fecha_dato"] = pd.to_datetime(df["fecha_dato"],format="%Y-%m-%d")
df["fecha_alta"] = pd.to_datetime(df["fecha_alta"],format="%Y-%m-%d")
# let’s add a customer lifetime column in months
df["lifetime"] = (df.fecha_dato - df.fecha_alta).astype('timedelta64[M]')
# let's add a month column. 
df["month"] = pd.DatetimeIndex(df["fecha_dato"]).month
df["age"]   = pd.to_numeric(df["age"], errors="coerce")
#Let's separate the “age” distribution and move the outliers to the mean of the closest one.
df.loc[df.age < 18,"age"]  = df.loc[(df.age >= 18) & (df.age <= 30),"age"].mean(skipna=True)
df.loc[df.age > 100,"age"] = df.loc[(df.age >= 30) & (df.age <= 100),"age"].mean(skipna=True)
df["age"].fillna(df["age"].mean(),inplace=True)
df["age"]                  = df["age"].astype(int)
#Next ind_nuevo, which indicates whether a customer is new or not. Let's see if we can fill in #missing values by looking how many months of history these customers have.
months_active = df.loc[df["ind_nuevo"].isnull(),:].groupby("ncodpers", sort=False).size()
#Looks like these are all new customers, so replace accordingly.
df.loc[df["ind_nuevo"].isnull(),"ind_nuevo"] = 1
#Now antiguedad
df.antiguedad = pd.to_numeric(df.antiguedad,errors="coerce")
df.loc[df.antiguedad.isnull(),"antiguedad"] = df.antiguedad.min()
df.loc[df.antiguedad <0, "antiguedad"]
#Some entries don't have the date they joined the company. Just give them something in the #middle of the pack
dates=df.loc[:,"fecha_alta"].sort_values().reset_index()
median_date = int(np.median(dates.index.values))
df.loc[df.fecha_alta.isnull(),"fecha_alta"] = dates.loc[median_date,"fecha_alta"]
df.loc[df.indrel.isnull(),"indrel"] = 1
df.drop(["tipodom","cod_prov"],axis=1,inplace=True)
df.loc[df.ind_actividad_cliente.isnull(),"ind_actividad_cliente"] = df["ind_actividad_cliente"].median()
df.loc[df.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A"
df.loc[df.nomprov.isnull(),"nomprov"] = "UNKNOWN"
df.loc[df.renta.isnull(),"renta"] = 0.0
df["renta"]                  = df["renta"].astype(float)
#df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
#df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0
string_data = df.select_dtypes(include=["object"])
missing_columns = [col for col in string_data if string_data[col].isnull().any()]
df.loc[df.indfall.isnull(),"indfall"] = "N"
df.loc[df.tiprel_1mes.isnull(),"tiprel_1mes"] = "A"
df.tiprel_1mes = df.tiprel_1mes.astype("category")
# As suggested by @StephenSmith
map_dict = { 1.0  : "1",
            "1.0" : "1",
            "1"   : "1",
            "3.0" : "3",
            "P"   : "P",
            3.0   : "3",
            2.0   : "2",
            "3"   : "3",
            "2.0" : "2",
            "4.0" : "4",
            "4"   : "4",
            "2"   : "2"}

df.indrel_1mes.fillna("P",inplace=True)
df.indrel_1mes = df.indrel_1mes.apply(lambda x: map_dict.get(x,x))
df.indrel_1mes = df.indrel_1mes.astype("category")
unknown_cols = [col for col in missing_columns if col not in ["indfall","tiprel_1mes","indrel_1mes"]]
for col in unknown_cols:
    df.loc[df[col].isnull(),col] = "UNKNOWN"
#Convert the feature columns into integer values
feature_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
for col in feature_cols:
    df[col] = df[col].astype(int)


unique_months = pd.DataFrame(pd.Series(df.fecha_dato.unique()).sort_values()).reset_index(drop=True)
unique_months["month_id"] = pd.Series(range(1,1+unique_months.size)) 
# start with month 1, not 0 to match what we already have
unique_months["month_next_id"] = 1 + unique_months["month_id"]
unique_months.rename(columns={0:"fecha_dato"},inplace=True)
df = pd.merge(df,unique_months,on="fecha_dato")


 
X_Test = df[features]
X_Test.head()
del df




X = pd.concat([X1,X_Test], axis=0)
X1_rows = X1.shape[0]

def preprocess_features(X):
    ''' Preprocesses the data and converts non-numeric binary variables into binary (0/1) variables. Converts categorical variables into dummy variables. '''
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)
    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        # Collect the revised columns
        output = output.join(col_data)
    return output

X = preprocess_features(X)
X1 = X.iloc[:X1_rows, :] 
X_Test = X.iloc[X1_rows:, :]

X2 = preprocess_features(X2)
y1 = y1.replace(["Added", "Dropped"], [1, 0])
y2 = y2.replace(['ind_cco_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_pres_fin_ult1',
       'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
       'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1',
       'ind_recibo_ult1'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])




from sklearn.metrics import f1_score
def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    score = f1_score(y_true, y_predict)
    return score


# TODO: Import 'train_test_split'
from sklearn.cross_validation import train_test_split

#Shuffle and split the data into training and testing subsets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.20, random_state=42)




from sklearn.metrics import make_scorer


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree model trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # Create a decision tree regressor object
    classifier = DecisionTreeClassifier()
#classifier = RandomForestClassifier()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    #params = [{'max_depth': range(1,11)}]
    params = [{'max_depth': [2,3,4,5,6]}]
 
    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(classifier, params, cv = cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


mod1 = fit_model(X1_train, y1_train)
y1_predict = mod1.predict(X1_test)


mod1.get_params()['max_depth']
performance_metric(y1_test, y1_predict)




mod1




mod1.feature_importances_




X1_train.head()




importances = mod1.feature_importances_
indices = np.argsort(importances)[::-1]




indices




for f in range(X1_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))




X1_train.columns[indices][:18]




X1_train.describe()






