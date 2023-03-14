#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20,5]




import os 
os.listdir('../input/tarefa-2')




dataTrain = pd.read_csv('../input/tarefa-2/train_data.csv')
dataTrain.head()




dataTrain.shape




list(dataTrain)




XdataTrain = dataTrain.iloc[:,:-2]
YdataTrain = dataTrain['ham']
IDdataTrain = dataTrain['Id']




dataTrain.isnull().sum().sum()




dataTrain['ham'].value_counts()




dataSemZeros = dataTrain.copy().iloc[:,:-2]




for v in dataSemZeros:
    dataSemZeros[v] = dataSemZeros.loc[dataSemZeros[v] != 0, v]




dataSemZeros.head()




dataZeros = dataSemZeros.isna().sum()
dataZeros




plt.rcParams['figure.figsize'] = [20,5]
for i in range(4):
    dataSemZeros.iloc[:,i*12:(i+1)*12].plot(kind='hist', bins=1000, legend=True, alpha=0.3, xlim=(-0.2, 3))
    plt.legend(loc=6, bbox_to_anchor=(1, 0.5))




dataSemZeros.iloc[:,48:54].plot(kind='hist', bins=1000, legend=True, alpha=0.3, xlim=(-0.2, 1.5))
plt.legend(loc=6, bbox_to_anchor=(1, 0.5))




dataSemZeros.iloc[:,54:57].plot(kind='hist', bins=4000, legend=True, alpha=0.3, xlim=(-1,50))
plt.legend(loc=6, bbox_to_anchor=(1, 0.5))




meanComZero = list(dataTrain.mean())[:-5]
stdComZero = list(dataTrain.std())[:-5]
meanSemZero = list(dataSemZeros.mean())[:-3]
stdSemZero = list(dataSemZeros.std())[:-3]




ind = np.arange(len(meanComZero))  # the x locations for the groups
width = 0.2       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

bar1 = ax.bar(ind        , meanComZero, width=width, align='center')
bar2 = ax.bar(ind+  width, stdComZero , width=width, align='center')
bar3 = ax.bar(ind+2*width, meanSemZero, width=width, align='center')
bar4 = ax.bar(ind+3*width, stdSemZero , width=width, align='center')

ax.set_xticks(ind + width/10)
plt.xticks(rotation=90)

ax.set_xticklabels(list(dataTrain)[:-3])

ax.legend((bar1[0], bar2[0], bar3[0], bar4[0]), ["Média, com 0's", "Std, com0's", "Média, sem 0's", "Std, sem 0's"])

#plt.rcParams['figure.figsize'] = [20,5]

#plt.show()




A = list(dataTrain[dataTrain['ham']==True].mean())[:-5]
B = list(dataSemZeros[dataTrain['ham']==True].mean())[:-3]
C = list(dataTrain[dataTrain['ham']==False].mean())[:-5]
D = list(dataSemZeros[dataTrain['ham']==False].mean())[:-3]




ind = np.arange(len(meanComZero))  # the x locations for the groups
width = 0.2       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

bar1 = ax.bar(ind        , A , width=width, align='center')
bar2 = ax.bar(ind+  width, B , width=width, align='center')
bar3 = ax.bar(ind+2*width, C , width=width, align='center')
bar4 = ax.bar(ind+3*width, D , width=width, align='center')

ax.set_xticks(ind + width/10)
plt.xticks(rotation=90)

ax.set_xticklabels(list(dataTrain)[:-3])

ax.legend((bar1[0], bar2[0], bar3[0], bar4[0]), 
          ["Média do ham, com 0's", "Média do ham, sem 0's", "Média do spam, com 0's", "Média do spam, sem 0's"])




A = list(dataTrain[dataTrain['ham']==True].std())[:-5]
B = list(dataSemZeros[dataTrain['ham']==True].std())[:-3]
C = list(dataTrain[dataTrain['ham']==False].std())[:-5]
D = list(dataSemZeros[dataTrain['ham']==False].std())[:-3]




ind = np.arange(len(meanComZero))  # the x locations for the groups
width = 0.2       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

bar1 = ax.bar(ind        , A , width=width, align='center')
bar2 = ax.bar(ind+  width, B , width=width, align='center')
bar3 = ax.bar(ind+2*width, C , width=width, align='center')
bar4 = ax.bar(ind+3*width, D , width=width, align='center')

ax.set_xticks(ind + width/10)
plt.xticks(rotation=90)

ax.set_xticklabels(list(dataTrain)[:-3])

ax.legend((bar1[0], bar2[0], bar3[0], bar4[0]), 
          ["Std do ham, com 0's", "Std do ham, sem 0's", "Std do spam, com 0's", "Std do spam, sem 0's"])




XdataTrainBin = XdataTrain.astype('bool').astype('int')




A = list(XdataTrainBin[dataTrain['ham']==True].mean())[:-3]
B = list(XdataTrainBin[dataTrain['ham']==False].mean())[:-3]




ind = np.arange(len(meanComZero))  # the x locations for the groups
width = 0.4       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

bar1 = ax.bar(ind        , A , width=width, align='center')
bar2 = ax.bar(ind+  width, B , width=width, align='center')

ax.set_xticks(ind + width/10)
plt.xticks(rotation=90)

ax.set_xticklabels(list(dataTrain)[:-3])

ax.legend((bar1[0], bar2[0]), 
          ["Médias do ham", "Médias do Spam"])




meansBin = []
for i in range(len(A)):
    try:
        if A[i] > B[i]: meansBin.append(((A[i]-B[i])/B[i], i))
        else: meansBin.append(((B[i]-A[i])/A[i], i))
    except: meansBin.append((np.inf, i))
meansBin.sort()
meansBin = np.array(meansBin[::-1])
melhores = list(meansBin[:35,1])




dataTest = pd.read_csv('../input/tarefa-2/test_features.csv')
dataTest.head()




XdataTest = dataTest.iloc[:,:-1]
IDdataTest = dataTest['Id']
XdataTestBin = XdataTest.astype('bool').astype('int')




XdataTest.head()




IDdataTest.head()




from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score




from tqdm import tqdm
temp = []
for i in tqdm(np.arange(0,4, 0.02)):
    clf = BernoulliNB(alpha=i)
    clf.fit(XdataTrainBin, YdataTrain)
    scores = cross_val_score(clf, XdataTrainBin,YdataTrain, cv=10)
    temp.append((scores.mean(), i))
temp.sort(key=lambda x: x[0])
temp[-5:]




clf = BernoulliNB()
clf.fit(XdataTrainBin, YdataTrain)




scores = cross_val_score(clf, XdataTrainBin,YdataTrain, cv=10)
print(scores)
print(scores.mean())




clf = BernoulliNB(alpha=0)
clf.fit(XdataTrainBin, YdataTrain)




YdataPredBayes = clf.predict(XdataTestBin)




YdataPredBayes.shape




IDdataTest = list(IDdataTest)
YdataPredBayes = list(YdataPredBayes)

submissionBayes = np.array([IDdataTest, YdataPredBayes])
submissionBayes




submissionBayes = pd.DataFrame(submissionBayes.T,
                               columns=['Id', 'ham'])
submissionBayes['ham'] = submissionBayes['ham'].astype(bool)
submissionBayes.head()




submissionBayes.to_csv('resBayes.csv', index=False)




from sklearn.neighbors import KNeighborsClassifier




from tqdm import tqdm

accuracies = {}

Ktam = 25

for k in tqdm(range(1,Ktam+1)):
    knn = KNeighborsClassifier(n_neighbors=k, p=1)
    knn.fit(XdataTrainBin.iloc[:,melhores],YdataTrain)
    scores = cross_val_score(knn, XdataTrainBin.iloc[:,melhores], YdataTrain, cv=10)

    accuracies[k] = scores.mean()




accuraciesSorted = list(accuracies.items())
accuraciesSorted.sort(key=lambda x: x[1])

accuraciesSorted[-10:]




plt.plot(accuracies.keys(), accuracies.values())




knn = KNeighborsClassifier(n_neighbors=12, p=1)
knn.fit(XdataTrainBin.iloc[:,melhores],YdataTrain)
YdataPredKnn = knn.predict(XdataTestBin.iloc[:,melhores])




IDdataTest = list(IDdataTest)
YdataPredKnn = list(YdataPredKnn)

submissionKnn = np.array([IDdataTest, YdataPredKnn])
submissionKnn




submissionKnn = pd.DataFrame(submissionKnn.T,
                columns=['Id', 'ham'])
submissionKnn['ham'] = submissionKnn['ham'].astype(bool)
submissionKnn.head()




submissionKnn.to_csv('resKnn.csv', index=False)




knn = KNeighborsClassifier(n_neighbors=12, p=1)
knn.fit(XdataTrainBin.iloc[:,melhores],YdataTrain)
YdataPredKnn = knn.predict(XdataTrainBin.iloc[:,melhores])




label = np.array(YdataTrain)
pred = YdataPredKnn




TP = pred[label].sum()
FP = np.logical_not(label[pred]).sum()
FN = np.logical_not(pred[label]).sum()




prec = TP / (TP + FP)




rec = TP / (TP + FN)




F3 = ( ((prec**-1) + 9*(rec**-1)) / (1+9))**-1




print(F3)




clf = BernoulliNB(alpha=0)
clf.fit(XdataTrainBin, YdataTrain)
pred = clf.predict_proba(XdataTrainBin)




alpha = pred[:,0] / pred[:,1]




TPrate = []
FPrate = []

for i in np.append(np.arange(0,1, 0.00002), [1]):
    pred = alpha > i
    
    TP = pred[label].sum()
    FP = np.logical_not(label[pred]).sum()
    FN = np.logical_not(pred[label]).sum()
    TN = (pred == label).sum() - TP
    
    TPrate.append(TP / (TP+FN))
    FPrate.append(FP / (FP+TN))




pontX = [0,1]
pontY = [0,1]




plt.rcParams['figure.figsize'] = [10,10]
plt.plot(np.append(FPrate, [0]), np.append(TPrate, [0]), pontX, pontY, ':')
plt.xlabel('TP-rate')
plt.ylabel('FP-rate')

