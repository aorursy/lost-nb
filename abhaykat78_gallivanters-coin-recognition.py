#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


# In[2]:


auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


# In[3]:


download = drive.CreateFile({'id': '1VKmIegctrMDkVEhKsr_7m9I5ltMIZ3e3'})
download.GetContentFile('coins.zip')


# In[4]:


download = drive.CreateFile({'id': '1SDd6umHK0IqwPuPUYs1p0woRI7lsHnvK'})
download.GetContentFile('test.zip')


# In[5]:


get_ipython().system('unzip test.zip -d test')


# In[6]:


get_ipython().system('pwd')


# In[7]:


get_ipython().system('ls')


# In[8]:


get_ipython().system('unzip coins.zip -d data')


# In[9]:


get_ipython().system('ls')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


import pathlib
from fastai.vision import *
from fastai.metrics import error_rate , accuracy


# In[11]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os


# In[12]:


print(os.listdir("/content/data/Coins"))


# In[13]:


get_ipython().system('pip install split-folders')


# In[14]:


get_ipython().system('mkdir babes')


# In[15]:


import split_folders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
split_folders.ratio('/content/data/Coins', output="/content/newdata", seed=1337, ratio=(.8, .1, .1)) # default values


# In[16]:


print(os.listdir("/content/data/Coins"))


# In[17]:



get_ipython().system('mv /content/newdata/val /content/newdata/valid ')
print(os.listdir("../content/newdata"))


# In[18]:


rm -rf /content/newdata/train/*


# In[19]:


print(os.listdir("../content/newdata/train"))


# In[20]:


mv  -v /content/data/Coins/* /content/newdata/train/


# In[21]:


print(os.listdir("../content/newdata/train/taiwan"))


# In[22]:


from pathlib import Path
path = Path("../content/newdata")


# In[23]:


#creating an object of get_transform class from fastai.vision
#transformations are used for data augmentation
#since we have a lot of photos in test data which are inverted and which are taken from different angles 
#we need to augment our data 
tfms = get_transforms(do_flip=False)
#we have normalize our data with google imagenet data statistcs
data = ImageDataBunch.from_folder(path,ds_tfms=tfms, size=256).normalize(imagenet_stats)


# In[24]:


data.show_batch(rows=3, figsize=(7,6))


# In[25]:


learn = cnn_learner(data, models.resnet50, metrics=accuracy,callback_fns=ShowGraph)
learn.lr_find()
learn.recorder.plot()


# In[26]:


learn.recorder.plot(suggestion=True)
best_clf_lr = learn.recorder.min_grad_lr
lr=best_clf_lr


# In[27]:


import matplotlib.pyplot as plt


# In[28]:


plt.style.use("seaborn")


# In[29]:


learn.fit_one_cycle(10, max_lr=0.005248074602497722)


# In[30]:



learn.save('stage-1')


# In[31]:


learn.load('stage-1')


# In[ ]:





# In[32]:


learn.fit_one_cycle(10, max_lr=0.005248074602497722)


# In[33]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[34]:


interp.plot_top_losses(9, figsize=(15,11))


# In[35]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[36]:


p = Path("../working/data/train")


# In[37]:


directories = p.glob("*")
#we have made a list of all the directories present at path p which is the path to our training folder
labels=0
#instantiating the country dictionary.
country_dic={}

for fol in directories:
    class_name =int(str(fol).split("/")[-1]) # getting class name from folder name
    # 
    for img_path in fol.glob("*.jpg"): 
        #getting country name from image name /
        #some insane string slicing and regex
        country = "_".join(str(img_path).split("/")[-1][8:].split(".")[0].split("_")[1:])
        if country == "denmark":
            labels+=1
        if class_name not in country_dic.keys():
            country_dic[class_name] = country


# In[38]:


labels


# In[39]:


country_dic


# In[40]:


from pathlib import Path
p = Path("/content/test/test")


# In[41]:


dire = p.glob("*.jpg")
output = [0 for i in range(1056)]
for img_path in dire:
    img = open_image(img_path)
    #opening each image in test folder and predicting classes
    pred_class,pred_idx,outputs = learn.predict(img)
    #changing predicted classes to predicted country with help of country dic we made above
    
    #print(str(img_path).spli/")[-1].split(".")[0])
    #getting image index from its path and storing predicted country at that index
    output[int(str(img_path).split("/")[-1].split(".")[0])] = pred_class
    


# In[42]:


output.remove(0)
# since zeroth index not needed


# In[43]:


output = np.array(output,dtype="str")


# In[44]:


df = pd.DataFrame(output)
df.to_csv("../content/Answer.csv")


# In[45]:


df.head()


# In[46]:


import base64
from IPython.display import HTML
# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "Answer.csv"):  
    csv = df.to_csv()
    #df.index = (1,np.arange(len(df)+1))
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)



# create a link to download the dataframe
create_download_link(df)


# In[ ]:




