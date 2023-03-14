#!/usr/bin/env python
# coding: utf-8



import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set() # for plot styling




df = pd.read_csv('../input/dmassign1/data.csv')




df.shape




df.head()




df = df.drop_duplicates()
df.info()




X = df.iloc[0:1300, 1:198]
X.shape




y = df.iloc[0:1300,198:199]
y.shape




y['Class'].value_counts()




for i in range(0,197):
    X[X.columns[i]] = X[X.columns[i]].astype('category')
    X[X.columns[i]] = X[X.columns[i]].cat.codes
    X[X.columns[i]] = X[X.columns[i]].astype('float64')
X.dtypes




X.tail()




from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X) 

X_scaled




from sklearn.decomposition import PCA
pca1 = PCA(n_components=2)
pca1.fit(X_scaled)
X_pca = pca1.transform(X_scaled)
X_principal = pd.DataFrame(X_pca) 
X_principal.columns = ['P1', 'P2'] 




from sklearn.cluster import KMeans

wcss = []
for i in range(2, 50):
    kmean = KMeans(n_clusters = i, random_state = 42)
    kmean.fit(X_scaled)
    wcss.append(kmean.inertia_)
    
plt.plot(range(2,50),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()




from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

print(__doc__)



range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

for n_clusters in range_n_clusters:

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)


    ax1.set_xlim([-0.1, 1])

    ax1.set_ylim([0, len(X_scaled) + (n_clusters + 1) * 10])


    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X_scaled)


    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)


    sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):

        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)


        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))


        y_lower = y_upper + 10  
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")


    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([]) 
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

  
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')


    centers = clusterer.cluster_centers_
    centers = pca1.transform(centers)
  
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()




from matplotlib import pyplot as plt
from sklearn.cluster import Birch
plt.figure(figsize=(16, 8))
preds1 = []
for i in range(2, 11):
    brc = Birch(branching_factor=100, n_clusters=10, threshold=0.1)
    brc.fit(X_scaled)
    pred = brc.predict(X_scaled)
    
    plt.subplot(2, 5, i - 1)
    plt.title(str(i)+" clusters")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=pred)
    
    centroids = kmean.cluster_centers_
    centroids = pca1.transform(centroids)
    plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)




kmean = KMeans(n_clusters = 35, random_state = 400)
kmean.fit(X_scaled)
pred = kmean.predict(X_scaled)
unique, counts =np.unique(pred, return_counts=True)
print(np.asarray((unique, counts)))




from matplotlib import pyplot as plt
from sklearn.cluster import Birch
from sklearn.metrics import accuracy_score
brc = Birch(branching_factor=10, n_clusters=25, threshold=0.007)
brc.fit(X_scaled)

labels = brc.predict(X_scaled)
unique, counts =np.unique(labels, return_counts=True)
print(np.asarray((unique, counts)))

LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'b',
                   2 : 'y',
                   3 : 'g',
                   4 : 'w',
                   5 : 'c',
                   6 : 'm',
                   7 : '0.2',
                   8 : '0.5',
                   9 : '0.7'}
#label_color = [LABEL_COLOR_MAP[l] for l in labels]
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='rainbow', alpha=1, edgecolors='b')




#label identification for each cluster
res = []
for k in range(0,35):
    print(k,":")
    max_count = 0
    max_index = 1
    for i in range(1,6):
        count = 0
        for j in range(len(pred)):
            if pred[j] == k:
                if(y['Class'][j]==i):
                    count+=1;
        if(count>=max_count):
            max_count = count
            max_index = i
    print(max_index)
    print("")





#mapping of appropiate labels to each cluster as obtained from previous step
res = []
labels = pred
for i in range(len(labels)):
    if labels[i] == 0 or labels[i] == 4 or labels[i] == 16:
        res.append(2)
    elif labels[i] == 15 or labels[i] == 30:
        res.append(3)
    elif labels[i] == 7 or labels[i] == 33 or labels[i] == 20:
        res.append(4)
    elif labels[i] == 1 or labels[i] == 6 or labels[i] == 11:
        res.append(5)
    else:
        res.append(1)
accuracy = accuracy_score(y, res)
print("Accuracy: %.2f%%" % (accuracy * 100.0))




LABEL_COLOR_MAP = {1 : 'r',
                   2 : 'b',
                   3 : 'y',
                   4 : 'g',
                   5 : 'k'}
label_color = [LABEL_COLOR_MAP[l] for l in res]
plt.scatter(X_pca[:,0], X_pca[:,1], c=label_color, cmap='rainbow', alpha=0.7, edgecolors='b')




X_pred = df.iloc[1300:, 1:198]
X_pred.shape




for i in range(0,197):
    X_pred[X_pred.columns[i]] = X_pred[X_pred.columns[i]].astype('category')
    X_pred[X_pred.columns[i]] = X_pred[X_pred.columns[i]].cat.codes
    X_pred[X_pred.columns[i]] = X_pred[X_pred.columns[i]].astype('float64')




from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_pred_scaled = scaler.fit_transform(X_pred) 

X_pred_scaled




from sklearn.decomposition import PCA
pca1 = PCA(n_components=2)
pca1.fit(X_pred_scaled)
X_pred_pca = pca1.transform(X_pred_scaled)




pred = brc.predict(X_pred_scaled)
unique, counts =np.unique(pred, return_counts=True)
print(np.asarray((unique, counts)))




labels = pred
res1 = []
for i in range(len(labels)):
    if labels[i] == 0 or labels[i] == 4 or labels[i] == 16:
        res1.append(2)
    elif labels[i] == 15 or labels[i] == 30:
        res1.append(3)
    elif labels[i] == 7 or labels[i] == 33 or labels[i] == 20:
        res1.append(4)
    elif labels[i] == 1 or labels[i] == 6 or labels[i] == 11:
        res1.append(5)
    else:
        res1.append(1)




submission = pd.DataFrame({'ID':df.iloc[1300:, 0], 'Class':res1})
submission.shape




filename = 'predictions18.csv'

submission.to_csv(filename,index=False)




from IPython.display import HTML 
import pandas as pd 
import numpy as np 
import base64 
def create_download_link(df, title = "Download CSV file", filename = "data.csv"): 
    csv = df.to_csv(index=False)     
    b64 = base64.b64encode(csv.encode())     
    payload = b64.decode()     
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'     
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html) 
create_download_link(submission)

