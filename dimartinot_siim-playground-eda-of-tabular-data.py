#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import cv2
import os
import shutil
import matplotlib.pyplot as plt

input_dir = "/kaggle/input/siim-isic-melanoma-classification"
train_dir = os.path.join(input_dir, "jpeg/train")
test_dir = os.path.join(input_dir, "jpeg/test")


train_csv = pd.read_csv(os.path.join(input_dir,"train.csv"))
test_csv = pd.read_csv(os.path.join(input_dir,"test.csv"))


# In[2]:


train_csv.head()


# In[3]:


train_csv["preffix_id"] = train_csv["patient_id"].apply(lambda x: x.split("_")[0])
train_csv["suffix_id"] = train_csv["patient_id"].apply(lambda x: x.split("_")[-1])

test_csv["preffix_id"] = test_csv["patient_id"].apply(lambda x: x.split("_")[0])
test_csv["suffix_id"] = test_csv["patient_id"].apply(lambda x: x.split("_")[-1])


# In[4]:


train_csv["preffix_id"].value_counts()


# In[5]:


test_csv["preffix_id"].value_counts()


# In[6]:


print("\tTRAIN SUFFIX ID VALUE COUNTS\n---\n")
print(train_csv["suffix_id"].value_counts())
print("-----------------------------------")
print("\tTEST SUFFIX ID VALUE COUNTS\n---\n")
print(test_csv["suffix_id"].value_counts())


# In[7]:


train_csv["suffix_id"].apply(lambda x: len(x)).value_counts()


# In[8]:


list(set(train_csv["suffix_id"].unique()) & set(test_csv["suffix_id"].unique()))


# In[9]:


plt.figure(figsize=(25,10))
plt.ylabel("Age count (in # of entry)")
plt.xlabel("Age")

ax_tr = train_csv["age_approx"].plot(kind='hist')
ax_te = test_csv["age_approx"].plot(kind='hist')

ax_tr.set_label("Train set")
ax_te.set_label("Test set")

L=plt.legend()
L.get_texts()[0].set_text('Train set')
L.get_texts()[1].set_text('Test set')

plt.plot()


# In[10]:


plt.close()


# In[11]:


print("\tTRAIN AGE DESCRIPTION\n---\n")
print(train_csv["age_approx"].describe())
print('-----------------------------')
print("\tTEST AGE DESCRIPTION\n---\n")
print(test_csv["age_approx"].describe())


# In[12]:


print("Train age unique values: ",np.sort(train_csv["age_approx"].unique()))
print("Test age unique values: ",np.sort(test_csv["age_approx"].unique()))


# In[13]:


print(f"{train_csv['age_approx'].isnull().sum()} rows have a 'nan' age out of {len(train_csv)} total rows").


# In[14]:


train_csv[train_csv['age_approx'].isnull()]


# In[15]:


train_csv[train_csv["age_approx"].isnull()]["suffix_id"].value_counts()


# In[16]:


train_csv[train_csv["age_approx"].isnull() & train_csv["sex"].isnull()]["suffix_id"].value_counts()


# In[17]:


nan_age = ['5205991', '9835712', '0550106']
train_csv[train_csv["suffix_id"].apply(lambda x: x in nan_age)]


# In[18]:


train_csv_cp = train_csv.copy()
train_csv_cp["sex"] = train_csv_cp["sex"].fillna("null")
test_csv_cp = test_csv.copy()
test_csv_cp["sex"] = test_csv_cp["sex"].fillna("null")


# In[19]:


print("\tTRAIN SEX VALUE COUNTS\n---\n")
print(train_csv_cp["sex"].value_counts())
print("-----------------------------------")
print("\tTEST SEX VALUE COUNTS\n---\n")
print(test_csv_cp["sex"].value_counts())


# In[20]:


train_csv[train_csv["sex"].isnull()]["suffix_id"].value_counts()


# In[21]:


print("\tTRAIN ANATOM SITE VALUE COUNTS:\n---\n")
print(train_csv["anatom_site_general_challenge"].fillna("null").value_counts())
print('--------------------------------------')
print("\tTEST ANATOM SITE VALUE COUNTS:\n---\n")
print(test_csv["anatom_site_general_challenge"].fillna("null").value_counts())


# In[22]:


plt.figure(figsize=(25,10))
plt.ylabel("Anatomy site count (in frequency)")
plt.xlabel("Anatomy site")

ax_tr = train_csv["anatom_site_general_challenge"].fillna("null").value_counts(normalize=True).plot()
ax_te = test_csv["anatom_site_general_challenge"].fillna("null").value_counts(normalize=True).plot()

ax_tr.set_label("Train set")
ax_te.set_label("Test set")

L=plt.legend()
L.get_texts()[0].set_text('Train set')
L.get_texts()[1].set_text('Test set')

plt.plot()


# In[23]:


print("\tTRAIN NULL VALUE COUNTS:\n---\n")
print(train_csv[train_csv["anatom_site_general_challenge"].isnull()]["suffix_id"].value_counts())
print("-----------------------")
print("\tTEST NULL VALUE COUNTS:\n---\n")
print(test_csv[test_csv["anatom_site_general_challenge"].isnull()]["suffix_id"].value_counts())


# In[24]:


print(train_csv[train_csv["anatom_site_general_challenge"].isnull()]["benign_malignant"].value_counts())


# In[25]:


print("\tBENIGN DIAGNOSIS VALUE COUNT:\n---\n")
print(train_csv[train_csv["benign_malignant"] == "benign"]["diagnosis"].value_counts())
print('------------------------')
print("\tMALIGNANT DIAGNOSIS VALUE COUNT:\n---\n")
print(train_csv[train_csv["benign_malignant"] != "benign"]["diagnosis"].value_counts())


# In[26]:


train_csv["benign_malignant"].value_counts()


# In[27]:


new_train_csv = train_csv.copy()
to_drop = ["image_name", "diagnosis", "benign_malignant", "preffix_id", "patient_id"]


# In[28]:


new_train_csv["is_female"] = new_train_csv["sex"].apply(lambda x: 0 if type(x) != float and x.lower() == 'male' else 1 if type(x) != float else -1)
to_drop.append("sex")
new_train_csv.head()


# In[29]:


anatom_classes = {
    val: i for i, val in enumerate(new_train_csv["anatom_site_general_challenge"].unique())
}
print("String to class: "+str(anatom_classes))
new_train_csv["anatom_classes"] = new_train_csv["anatom_site_general_challenge"].apply(lambda x: anatom_classes[x])
to_drop.append("anatom_site_general_challenge")
new_train_csv.head()


# In[30]:


new_train_csv["reduced_age"] = new_train_csv["age_approx"].apply(lambda x: int(x/5) if np.isnan(x) == False else -1)
to_drop.append("age_approx")
new_train_csv.head()


# In[31]:


suffix_classes = {

    val: i for i,val in enumerate(new_train_csv["suffix_id"].unique())

}

new_train_csv["suffix_classes"] = new_train_csv["suffix_id"].apply(lambda x: suffix_classes[x])
to_drop.append("suffix_id")
to_drop.append("suffix_classes")
new_train_csv.head()


# In[32]:


y = new_train_csv.drop(to_drop, axis=1).iloc[:, 0].to_numpy()
X = new_train_csv.drop(to_drop, axis=1).iloc[:, 1:].to_numpy()


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[34]:


from sklearn.cluster import KMeans
N_CLUSTERS = 200


# In[35]:


clf = KMeans(N_CLUSTERS)


# In[36]:


kmeans = clf.fit(X_train)


# In[37]:


class_per_cluster = {}
preds = kmeans.predict(X_train)

for i in range(N_CLUSTERS):
    classes = y_train[np.where(preds == i)]
    c, counts = np.unique(classes, return_counts=True)
    s = sum(counts)
    
    counts = counts/s
        
    class_per_cluster[i] = counts


# In[38]:


y_preds = kmeans.predict(X_test)


# In[39]:


y_preds_classes = []
for pred in y_preds:
    probs = class_per_cluster[pred]
    
    c = np.random.choice(list(range(len(probs))), p=probs)
    
    y_preds_classes.append(c)
    
print(roc_auc_score(y_preds_classes, y_test))

