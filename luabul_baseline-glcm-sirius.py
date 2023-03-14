#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from sklearn import neighbors, linear_model
from sklearn.model_selection import train_test_split
from skimage.feature import greycomatrix, greycoprops
import cv2
from tqdm import tqdm_notebook as tqdm
import seaborn as sns
from sklearn.decomposition import PCA


# In[6]:


import warnings
warnings.filterwarnings('ignore')
import tarfile

#simple function to extract the train data
#tar_file : the path to the .tar file
#path : the path where it will be extracted
def extract(tar_file, path):
    opened_tar = tarfile.open(tar_file)
     
    if tarfile.is_tarfile(tar_file):
        opened_tar.extractall(path)
    else:
        print("The tar file you entered is not a tar file")


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10,8


# In[34]:


img_width = 64
img_height = 64
train_root_folder = 'fibrosis_patches_png/train/'
test_root_folder =  'fibrosis_patches_png/test/'
extract("../input/ml-in-med-sirius-2020/lungs-patches-classification/fibrosis_patches_png.tgz",".")
print(os.listdir("fibrosis_patches_png"))


# In[35]:


df_files = pd.read_csv('../input/ml-in-med-sirius-2020/lungs-patches-classification/train.csv')
df_files = df_files.sample(frac=1.0, random_state=42)
df_files.head()


# In[36]:


df_files['class'].hist(bins=3);


# In[37]:


X = np.zeros((df_files.shape[0],img_width, img_height), dtype=np.uint8)
for idx,file in tqdm(enumerate(df_files['filename'])):
    filename = train_root_folder+file
    img = cv2.imread(filename,0)
    X[idx] = img


# In[38]:


for i in range(9):
    plt.subplot(3,3,i+1)
    idx = np.random.randint(df_files.shape[0])
    plt.imshow(X[idx], cmap='bone')
    plt.title('Class:' + str(df_files.iloc[idx,1]))
    plt.grid('off')
    plt.axis('off')
plt.subplots_adjust(hspace=0.15, wspace=0.01)


# In[39]:


y = df_files['class']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)


# In[40]:


def calc_GLCM_features(X):
    GLCM_features = np.zeros((X.shape[0], 4), dtype=np.float32)
    for i in tqdm(range(X.shape[0])):
        img = X[i]
        glcm = greycomatrix(img, [2], [0], 256, symmetric=True, normed=True)
        GLCM_features[i] = np.array([greycoprops(glcm, 'dissimilarity')[0, 0],
                           greycoprops(glcm, 'correlation')[0, 0],
                           greycoprops(glcm, 'homogeneity')[0, 0],
                           greycoprops(glcm, 'contrast')[0, 0]])
    return GLCM_features


# In[41]:


GLCM_train = calc_GLCM_features(X_train)
GLCM_val = calc_GLCM_features(X_val)


# In[42]:


titles = ['dissimilarity', 'correlation' , 'homogeneity', 'contrast']
for i in range(4):
    plt.subplot(2,2,i+1)
    sns.kdeplot(GLCM_train[y_train==1, i], label='pathology');
    sns.kdeplot(GLCM_train[y_train==0, i], label='norm');
    plt.title(titles[i]);


# In[43]:


pca = PCA(n_components=2)
GLCM_pca = pca.fit_transform(GLCM_train)
plt.plot(GLCM_pca[y_train==1],'bo', label='pathology');
plt.plot(GLCM_pca[y_train==0],'rx', label='norm');


# In[45]:


knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression()
print('KNN score: %f' % knn.fit(GLCM_train, y_train).score(GLCM_val, y_val))
print('LogisticRegression score: %f'
      % logistic.fit(GLCM_train, y_train).score(GLCM_val, y_val))


# In[46]:


GLCM_X = calc_GLCM_features(X)
logistic.fit(GLCM_X, y)


# In[48]:


df_test = pd.read_csv('../input/ml-in-med-sirius-2020/lungs-patches-classification/test.csv')
X_test = np.zeros((df_test.shape[0],img_width, img_height), dtype=np.uint8)
for idx,file in tqdm(enumerate(df_test['filename'])):
    X_test[idx] = cv2.imread(test_root_folder+file,0)
GLCM_test = calc_GLCM_features(X_test)


# In[49]:


predictions = logistic.predict(GLCM_test)
df_pred = pd.DataFrame(predictions, columns=['predictions'])
df_pred = pd.concat((df_test, df_pred), axis=1)


# In[50]:


df_files.head()


# In[51]:


df_pred.to_csv('sample_submission.csv', header=True, index=None)


# In[ ]:




