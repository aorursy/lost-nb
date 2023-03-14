#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from statistics import mean,stdev,median
from prettytable import PrettyTable


# In[ ]:


def mis_val_treatment(data,string,percentage):
    """Treat the missing values.

    * This function select the features which have a number of missing values less than a given threshold
    * Numeric features : Replace missing values selected with mean/median of the feature
    * categorical features : Replace missing values with a new class named "missing value"

    :param data: data to treat
    :param string: mean/median
    :param percentage: percentage to define a threshold for missing values
    :return: data treated
    
    """
    var = data.isnull().sum()
    threshold = round(len(data.index) / percentage)
    data_mis_val = data[var[var!=0][var<threshold].index].copy()
    names_data = list(data_mis_val.columns.values)
    names_data_numeric = list(data_mis_val.describe().columns.values)
    for name in names_data:
        s = data_mis_val[name]
        if name in names_data_numeric:
            if string == "mean":
                s = s.fillna(s.mean())
            if string == "median":
                s = s.fillna(s.median())
        else:
            s = s.fillna("missing_value")
        data_mis_val[name] = s
    return data_mis_val


def data_model_constuct(data,data2):
    '''
    Reconstruct data after missing values treatment and one hot encoding

    :param data: data to reconstruct
    :param data2: result of mis val_treatment
    :return: data model constructed
    '''
    var = data.isnull().sum()
    data1 = data[var[var==0].index].copy()
    data_model = pd.concat([data1,data2],axis=1)
    data_model = pd.get_dummies(data_model)
    return data_model

def train_val_size(data,val_size,test_size):
    '''
    Split data into train validation and test datasets
    
    :param data:  data to split
    :param val_size: percentage of validation dataset size [0,1]
    :param test_size: percentage of test dataset size [0,1] 
    :return: dictionary containing the 3 datasets
    
    '''
    y=data['TARGET']
    data_train, data_valtest, y_train, y_valtest = train_test_split(data, y, test_size=val_size+test_size)
    data_val, data_test, y_val, y_test = train_test_split(data_valtest, y_valtest, test_size=val_size/(val_size+test_size))
    return {'data_train':data_train,'data_val':data_val,'data_test':data_test}


def prepare_data(path_to,string="mean",percentage=10,val_size=0.2,test_size=0.2):
    data = pd.read_csv(path_to)
    data_mis_val = mis_val_treatment(data,string,percentage)
    data_model = data_model_constuct(data, data_mis_val)
    dict = train_val_size(data_model,val_size,test_size)
    return dict


# In[ ]:


dict_data = prepare_data(path_to="../input/home-credit-default-risk/application_train.csv")


# In[ ]:


data_val = dict_data['data_val']
y_val = data_val['TARGET']
data_val_model = data_val.drop(['TARGET'], axis=1)


# In[ ]:


model = LogisticRegression()


# In[ ]:


def reglog_model_results(model,data_test,y_test):
    '''
    Performance model results
    
    :param model: model to implement
    :param data_test: data to test
    :param y_test: target variable 
    :return: dictionary of performance results
    '''

    # Calculate Class Probabilities
    probability = model.predict_proba(data_test)

    # Predicted Class Labels
    y_predicted = model.predict(data_test)

    # Evaluate The Model

    ### Confusion Matrix
    Confusion_Matrix = metrics.confusion_matrix(y_test, y_predicted)

    ### Classification Report
    Classification_Report = metrics.classification_report(y_test, y_predicted)

    ### Model Accuracy
    Accuracy = model.score(data_test, y_test)

    ### AUC
    y_pred_proba = probability[:, 1]
    [fpr, tpr, thr] = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.auc(fpr, tpr)

    return {'Class_Probabilities':probability,'Predicted_Class_Labels':y_predicted,'Confusion_Matrix':Confusion_Matrix,'Classification_Report':Classification_Report,'Accuracy':Accuracy, 'AUC':auc}


# Show Confusion Matrix
def confusion_matrix(cm):
    tab = PrettyTable([' ', 'Predicted 0', 'Predicted 1'])
    tab.add_row(["Actual 0", cm[0][0], cm[0][1]])
    tab.add_row(["Actual 1", cm[1][0], cm[1][1]])
    print(tab)

    
# Show the ROC_CURVE
def roc_curve_show(model,data_test,y_test):

    result_model = reglog_model_results(model, data_test, y_test)
    y_pred_proba = result_model['Class_Probabilities'][:, 1]
    [fpr, tpr, thr] = metrics.roc_curve(y_test, y_pred_proba)
    idx = np.min(np.where(tpr > 0.95))  # index of the first threshold for which the sensibility > 0.95
    plt.figure()
    plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0, fpr[idx]], [tpr[idx], tpr[idx]], 'k--', color='blue')
    plt.plot([fpr[idx], fpr[idx]], [0, tpr[idx]], 'k--', color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (recall)', fontsize=14)
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:


def results_model_dict(model,dict_data):
    '''
        apply the model and get results
        :param model: model to apply
        :param dict_data: dictionary including model datasets
        :return: dict of model results
        '''

    # datasets :
    data_train = dict_data['data_train']
    data_val = dict_data['data_val']
    data_test = dict_data['data_test']

    # Get target variable from each data
    y_train = data_train['TARGET']
    y_val = data_val['TARGET']
    y_test = data_test['TARGET']

    # drop target variable from each dataset
    data_train_model = data_train.drop(['TARGET'], axis=1)
    data_val_model = data_val.drop(['TARGET'], axis=1)
    data_test_model = data_test.drop(['TARGET'], axis=1)

    ##########  fit model   #############
    model1 = model.fit(data_train_model, y_train)

    ### model results
    results_model1 = reglog_model_results(model1, data_val_model, y_val)
    return results_model1


# In[ ]:


logreg_results = results_model_dict(model,dict_data)


# In[ ]:


logreg_results['AUC']


# In[ ]:


logreg_results['Accuracy']


# In[ ]:


dist = y_val.value_counts()
dist


# In[ ]:


cm = logreg_results['Confusion_Matrix']
confusion_matrix(cm)


# In[ ]:


tp = round(float(cm[0][0])/float(dist[0])*100,2)
"True Postive en % = "+str(tp)+" %"


# In[ ]:


tn = round(float(cm[1][1])/float(dist[1])*100,2)
"True Negative en % = "+str(tn)+" %"


# In[ ]:


roc_curve_show(model,data_val_model,y_val)


# In[ ]:


model_balanced = LogisticRegression(class_weight='balanced')


# In[ ]:


logreg_results_balanced = results_model_dict(model_balanced,dict_data)


# In[ ]:


logreg_results_balanced['AUC']


# In[ ]:


logreg_results_balanced['Accuracy']


# In[ ]:


dist = y_val.value_counts()
dist


# In[ ]:


cm_balanced = logreg_results_balanced['Confusion_Matrix']
confusion_matrix(cm_balanced)


# In[ ]:


tp_balanced = round(float(cm_balanced[0][0])/float(dist[0])*100,2)
"True Postive en % = "+str(tp_balanced)+" %"


# In[ ]:


tn_balanced = round(float(cm_balanced[1][1])/float(dist[1])*100,2)
"True Negative en % = "+str(tn_balanced)+" %"


# In[ ]:


roc_curve_show(model_balanced,data_val_model,y_val)


# In[ ]:


def resmpling_data(data,val,string="percentage"):
    '''
    Resampling data with given a value of percentage or number for output rows number

    :param data: pandas data_train to resample
    :param val: value of percentage or rows number
    :param string: percentage or number
    :return: dataset resampled
    '''
    y = data['TARGET']
    data_0 = data[y == 0]
    data_1 = data[y == 1]
    nrows = data_0.shape[0]
    if string == "percentage":
        sample = int(round(val*nrows/100))
        boot = resample(data_0, replace=False, n_samples=sample)
    else:
        boot = resample(data_0, replace=False, n_samples=val)

    data_boot = boot.append(data_1)
    return data_boot


# In[ ]:


data = dict_data['data_train']
y = data['TARGET']
percentage_to_sample = round(float(y.value_counts()[1])/float(y.value_counts()[0])*100)
"Percentage_to_sample = "+str(percentage_to_sample)+" %"


# In[ ]:


data_resampled = resmpling_data(data,9)


# In[ ]:


model = LogisticRegression()


# In[ ]:


data_train = dict_data['data_train']

#resample data
data_resampled = resmpling_data(data_train,9,string="percentage")


# datasets :
data_val =  dict_data['data_val']
data_test = dict_data['data_test']

# Get target variable from each data
y_train = data_resampled['TARGET']
y_val = data_val['TARGET']
y_test = data_test['TARGET']

# drop target variable from each dataset
data_train_model = data_resampled.drop(['TARGET'], axis=1)
data_val_model = data_val.drop(['TARGET'], axis=1)
data_test_model = data_test.drop(['TARGET'], axis=1)


# In[ ]:


model_subsample = model.fit(data_train_model, y_train)
logreg_results_resampled = reglog_model_results(model_subsample, data_val_model, y_val)


# In[ ]:


logreg_results_resampled['AUC']


# In[ ]:


logreg_results_resampled['Accuracy']


# In[ ]:


dist = y_val.value_counts()
dist


# In[ ]:


cm_resampled = logreg_results_resampled['Confusion_Matrix']
confusion_matrix(cm_resampled)


# In[ ]:


tp_resampled = round(float(cm_resampled[0][0])/float(dist[0])*100,2)
"True Postive en % = "+str(tp_resampled)+" %"


# In[ ]:


tn_resampled = round(float(cm_resampled[1][1])/float(dist[1])*100,2)
"True Negative en % = "+str(tn_resampled)+" %"


# In[ ]:


tab = PrettyTable(['Model', 'Accuracy','AUC','Sensitivity : TP %','Specificity : TN %'])
tab.add_row(["model_unbalanced",round(logreg_results['Accuracy'],5),round(logreg_results['AUC'],5),tp,tn])
tab.add_row(["model_balanced",round(logreg_results_balanced['Accuracy'],5),round(logreg_results_balanced['AUC'],5),tp_balanced,tn_balanced])
tab.add_row(["model_subsample",round(logreg_results_resampled['Accuracy'],5),round(logreg_results_resampled['AUC'],5),tp_resampled,tn_resampled])
print(tab)


# In[ ]:


data = pd.read_csv("../input/home-credit-default-risk/application_train.csv")
data_mis_val = mis_val_treatment(data, string="mean",percentage=10)
data_model = data_model_constuct(data, data_mis_val)


# In[ ]:


def results_model_data(model,data_model,val_size=0.2,test_size=0.2):
    '''
    Split data-model into train, val and test datasets and then apply the model and finally get results
    :param model: model to apply
    :param data: data_model
    :param val_size:
    :param test_size:
    :return:  model results and random data split result
    '''

    dict_data = train_val_size(data_model, val_size, test_size)
    # datasets :
    data_train = dict_data['data_train']
    data_val = dict_data['data_val']
    data_test = dict_data['data_test']

    # Get target variable from each data
    y_train = data_train['TARGET']
    y_val = data_val['TARGET']
    y_test = data_test['TARGET']

    # drop target variable from each dataset
    data_train_model = data_train.drop(['TARGET'], axis=1)
    data_val_model = data_val.drop(['TARGET'], axis=1)
    data_test_model = data_test.drop(['TARGET'], axis=1)

    ##########  fit model   #############
    model1 = model.fit(data_train_model, y_train)

    ### model results
    results_model1 = reglog_model_results(model1, data_val_model, y_val)
    results_model1.update(dict_data)
    return results_model1


# In[ ]:


auc_results = list()
accuracy_results = list()


# In[ ]:


for i in range(0,10):
    auc = results_model_data(model_balanced,data_model)['AUC']
    accuracy = results_model_data(model_balanced,data_model)['Accuracy'] 
    auc_results.append(auc)
    accuracy_results.append(accuracy)


# In[ ]:


t = PrettyTable(['Model', 'accuracy','AUC'])


# In[ ]:


for i in range(0,10):
    m = "model"+" "+str(i+1)
    t.add_row([m,round(accuracy_results[i],5),round(auc_results[i],5)])

t.add_row(["Mean",mean(accuracy_results),mean(auc_results)])
t.add_row(["Standard deviation",stdev(accuracy_results),stdev(auc_results)])
t.add_row(["Median",median(accuracy_results),median(auc_results)])
print(t)


# In[ ]:


logreg_results_balanced_test = reglog_model_results(model_balanced, data_test_model, y_test)


# In[ ]:


logreg_results_balanced_test['AUC']


# In[ ]:


logreg_results_balanced_test['Accuracy']


# In[ ]:


dist = y_test.value_counts()
dist


# In[ ]:


cm_balanced_test = logreg_results_balanced_test['Confusion_Matrix']
confusion_matrix(cm_balanced_test)


# In[ ]:


tp_balanced_test = round(float(cm_balanced_test[0][0])/float(dist[0])*100,2)
"True Postive en % = "+str(tp_balanced_test)+" %"


# In[ ]:


tn_balanced_test = round(float(cm_balanced_test[1][1])/float(dist[1])*100,2)
"True Negative en % = "+str(tn_balanced_test)+" %"


# In[ ]:


tab = PrettyTable(['Dataset', 'Accuracy','AUC','Sensitivity : TP %','Specificity : TN %'])
tab.add_row(["Validation data",round(logreg_results_balanced['Accuracy'],5),round(logreg_results_balanced['AUC'],5),tp_balanced,tn_balanced])
tab.add_row(["Test data",round(logreg_results_balanced_test['Accuracy'],5),round(logreg_results_balanced_test['AUC'],5),tp_balanced_test,tn_balanced_test])
print(tab)


# In[ ]:


dict_data = prepare_data(path_to="../input/home-credit-default-risk/application_train.csv")


# In[ ]:


data_val = dict_data['data_val']
y_val = data_val['TARGET']
data_val_model = data_val.drop(['TARGET'], axis=1)


# In[ ]:


model = GradientBoostingClassifier()


# In[ ]:


gbm_results = results_model_dict(model,dict_data)


# In[ ]:


gbm_results['AUC']


# In[ ]:


gbm_results['Accuracy']


# In[ ]:


dist = y_val.value_counts()
dist


# In[ ]:


cm = gbm_results['Confusion_Matrix']
confusion_matrix(cm)


# In[ ]:


"True Postive en % = "+str(round(float(cm[0][0])/float(dist[0])*100,2))+" %"


# In[ ]:


"True Negative en % = "+str(round(float(cm[1][1])/float(dist[1])*100,2))+" %"


# In[ ]:


# Import datasets
data_train = dict_data['data_train']


# In[ ]:



#resample data
data_resampled = resmpling_data(data_train,9,string="percentage")

# datasets :
data_val =  dict_data['data_val']
data_test = dict_data['data_test']

# Get target variable from each data
y_train = data_resampled['TARGET']
y_val = data_val['TARGET']
y_test = data_test['TARGET']

# drop target variable from each dataset
data_train_model = data_resampled.drop(['TARGET'], axis=1)
data_val_model = data_val.drop(['TARGET'], axis=1)
data_test_model = data_test.drop(['TARGET'], axis=1)


# In[ ]:


# built model
model_gbm_resample = model.fit(data_train_model, y_train)

### model results
results_model_gbm_resample = reglog_model_results(model_gbm_resample, data_val_model, y_val)


# In[ ]:


results_model_gbm_resample['AUC']


# In[ ]:


results_model_gbm_resample['Accuracy']


# In[ ]:


dist = y_val.value_counts()
dist


# In[ ]:


cm_gbm_resample = results_model_gbm_resample['Confusion_Matrix']
confusion_matrix(cm_gbm_resample)


# In[ ]:


"True Postive en % = "+str(round(float(cm_gbm_resample[0][0])/float(dist[0])*100,2))+" %"


# In[ ]:


"True Negative en % = "+str(round(float(cm_gbm_resample[1][1])/float(dist[1])*100,2))+" %"


# In[ ]:


# Resampling with stratification
data_input = dict_data['data_train']
y = data_input['TARGET']
data_input_0 = data_input[y == 0]
# kmeans
kmeans = KMeans(n_clusters=6).fit(data_input_0)
d = pd.Series(kmeans.labels_)
print("Clustering with Kmeans : \n")
print(d.value_counts())
dict = {}
for i in range(0, 6):
    rows = list(d[d == i].index)
    data = data_input_0.iloc[rows, :]
    number_to_sample = data.shape[0] * 9 / 100
    number_to_sample = round(number_to_sample)
    data_sample = resample(data, replace=False, n_samples=number_to_sample)
    key = str(i)
    dict.update({key: data_sample})

data_output = pd.concat([dict['0'], dict['1'], dict['2'], dict['3'], dict['4'], dict['5']])

# reconstruct data
data_input_1 = data_input[y == 1]
data_train_output = data_output.append(data_input_1)
y_train = data_train_output['TARGET']
data_train_model = data_train_output.drop(['TARGET'], axis=1)

# built model
model_gbm_kmeans = model.fit(data_train_model, y_train)

### model results
results_model_gbm_kmeans = reglog_model_results(model_gbm_kmeans, data_val_model, y_val)


# In[ ]:


results_model_gbm_kmeans['AUC']


# In[ ]:


results_model_gbm_kmeans['Accuracy']


# In[ ]:


dist = y_val.value_counts()
dist


# In[ ]:


cm_gbm_kmeans = results_model_gbm_kmeans['Confusion_Matrix']
confusion_matrix(cm_gbm_kmeans)


# In[ ]:


"True Postive en % = "+str(round(float(cm_gbm_kmeans[0][0])/float(dist[0])*100,2))+" %"


# In[ ]:


"True negative en % = "+str(round(float(cm_gbm_kmeans[1][1])/float(dist[1])*100,2))+" %"


# In[ ]:


random_search_output = pd.read_csv("../input/output-random-search/output_random_search.csv")


# In[ ]:


output_ranked = random_search_output.sort_values("rank_test_score")
output_ranked


# In[ ]:


params = list(output_ranked["params"])
params[0]


# In[ ]:


model_search_output = GradientBoostingClassifier(n_estimators=500, max_depth=10,learning_rate=0.01)


# In[ ]:


# Import datasets
data_train = dict_data['data_train']

# resample data
data_resampled = resmpling_data(data_train,9,string="percentage")

# target
y_train = data_resampled['TARGET']

# drop target variable from each dataset
data_train_model = data_resampled.drop(['TARGET'], axis=1)

# built model
model_search_output_resample = model_search_output.fit(data_train_model, y_train)

### model results
results_model_search_resample = reglog_model_results(model_search_output_resample, data_val_model, y_val)


# In[ ]:


results_model_search_resample['AUC']


# In[ ]:


results_model_search_resample['Accuracy']


# In[ ]:


dist = y_val.value_counts()
dist


# In[ ]:


cm_gbm_search = results_model_search_resample['Confusion_Matrix']
confusion_matrix(cm_gbm_search)


# In[ ]:


tp_gbm_search = round(float(cm_gbm_search[0][0])/float(dist[0])*100,2)
"True Postive en % = "+str(tp_gbm_search)+" %"


# In[ ]:


tn_gbm_search = round(float(cm_gbm_search[1][1])/float(dist[1])*100,2)
"True negative en % = "+str(tn_gbm_search)+" %"


# In[ ]:


results_model_search_resample_test = reglog_model_results(model_search_output_resample, data_test_model, y_test)


# In[ ]:


results_model_search_resample_test['AUC']


# In[ ]:


results_model_search_resample_test['Accuracy']


# In[ ]:


dist = y_val.value_counts()
dist


# In[ ]:


cm_gbm_search_test = results_model_search_resample_test['Confusion_Matrix']
confusion_matrix(cm_gbm_search_test)


# In[ ]:


tp_gbm_search_test = round(float(cm_gbm_search_test[0][0])/float(dist[0])*100,2)
"True Postive en % = "+str(tp_gbm_search_test)+" %"


# In[ ]:


tn_gbm_search_test= round(float(cm_gbm_search_test[1][1])/float(dist[1])*100,2)
"True negative en % = "+str(tn_gbm_search_test)+" %"


# In[ ]:


tab = PrettyTable(['Dataset', 'Accuracy','AUC','Sensitivity : TP %','Specificity : TN %'])
tab.add_row(["Validation data",round(results_model_search_resample_test['Accuracy'],5),round(results_model_search_resample_test['AUC'],5),tp_gbm_search,tn_gbm_search])
tab.add_row(["Test data",round(results_model_search_resample_test['Accuracy'],5),round(results_model_search_resample_test['AUC'],5),tp_gbm_search_test,tn_gbm_search_test])
print(tab)


# In[ ]:


get_ipython().run_cell_magic('javascript', '', "$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")

