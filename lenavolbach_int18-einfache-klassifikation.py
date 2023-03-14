#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split


# In[ ]:


#Einlesen der ARFF Datei

def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() # get rid of newline at the end of each input line
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

train_data = read_data("../input/kiwhs-comp-1-complete/train.arff")
test_data = pd.read_csv("../input/kiwhs-comp-1-complete/test.csv")


# In[ ]:


for i in range(len(train_data)):
   point = train_data[i]
   c = "r"
   if point[2] == -1:
        c = "b"
   plt.scatter(point[0],point[1], s = 10, c = c)


# In[ ]:


#Source: Competition --> Discussion
#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

# Compare
# http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA']) 
cmap_bold = ListedColormap(['#0000FF', '#00FF00', '#FF0000']) 

def plot_decision_boundary(model,X,y):
    h = .02  # step size in the mesh
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
              edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(model.__class__.__name__)

    plt.show()


# In[ ]:


train = pd.DataFrame(train_data)
train.head()
points = train.iloc[:,:2].values
category = train.iloc[:,2].values

x_train, x_test, y_train, y_test = train_test_split(points, category, test_size = 0.15, random_state = 0)

test = test_data.iloc[:,1:].values


# In[ ]:


def plotTest(predictions):
    for i in range(len(predictions)):
        point = test[i]
        c = "r"
        if predictions[i] == -1:
             c = "b"
        plt.scatter(point[0],point[1], s = 10, c = c)


# In[ ]:


from sklearn.linear_model import LogisticRegression

#LogisticRegression

model_lr = sklearn.linear_model.LogisticRegressionCV(cv = 5) #5 = default value
model_lr.fit(x_train, y_train)
plot_decision_boundary(model_lr, x_train, y_train)

print('Logistic Regression with cross validation')
print ('train accuracy: {}'.format(model_lr.score(x_train, y_train)))
print ('test accuracy: {}'.format(model_lr.score(x_test, y_test)))


# In[ ]:


predictions_lr = model_lr.predict(test)
plotTest(predictions_lr)

submissions_lr = pd.DataFrame({"Id (String)": list(range(0,len(predictions_lr))),
                         "Category (String)": predictions_lr.astype(int)})

submissions_lr.to_csv("submissions_lr.csv", index=False, header=True)


# In[ ]:


from sklearn import svm

#Support Vector Machine

model_svm = svm.SVC(kernel = "linear", C = 0.025)
#model_svm = svm.SVC(gamma = 'scale')
model_svm.fit(x_train, y_train)
plot_decision_boundary(model_svm, x_train, y_train)

print('SVM')
print ('train accuracy: {}'.format(model_svm.score(x_train, y_train)))
print ('test accuracy: {}'.format(model_svm.score(x_test, y_test)))


# In[ ]:


predictions_svm = model_svm.predict(test)
plotTest(predictions_svm)

submissions_svm = pd.DataFrame({"Id (String)": list(range(0,len(predictions_svm))),
                         "Category (String)": predictions_svm.astype(int)})

submissions_svm.to_csv("submissions_svm.csv", index=False, header=True)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

#kNN

model_knn = KNeighborsClassifier(15) #Anzahl betrachteter Nachbarn
model_knn.fit(x_train, y_train)
plot_decision_boundary(model_knn, x_train, y_train)

print('KNN')
print ('train accuracy: {}'.format(model_knn.score(x_train, y_train)))
print ('test accuracy: {}'.format(model_knn.score(x_test, y_test)))


# In[ ]:


predictions_knn = model_knn.predict(test)
plotTest(predictions_knn)

submissions_knn = pd.DataFrame({"Id (String)": list(range(0,len(predictions_knn))),
                         "Category (String)": predictions_knn.astype(int)})

submissions_knn.to_csv("submissions_knn.csv", index=False, header=True)


# In[ ]:


from sklearn.neural_network import MLPClassifier

# Neural Network (Multi-layer Perceptron)
# example for overfitting

model_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(500, 2), random_state=1) #high hidden layers --> overfitting
model_mlp.fit(x_train, y_train)
plot_decision_boundary(model_mlp, x_train, y_train)

print('MLP Classifier')
print ('train accuracy: {}'.format(model_mlp.score(x_train, y_train)))
print ('test accuracy: {}'.format(model_mlp.score(x_test, y_test)))


# In[ ]:


from sklearn import tree

# Decision Tree

model_dt = tree.DecisionTreeClassifier()
model_dt.fit(x_train, y_train)
plot_decision_boundary(model_dt, x_train, y_train)

print('Decision Tree')
print ('train accuracy: {}'.format(model_dt.score(x_train, y_train)))
print ('test accuracy: {}'.format(model_dt.score(x_test, y_test)))

