#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision import *


# In[2]:


get_ipython().system('ls ../input/Kannada-MNIST')


# In[3]:


path = Path('../input/Kannada-MNIST')
train_csv = path/'train.csv'


# In[4]:


data = pd.read_csv(train_csv)


# In[5]:


data.head()


# In[6]:


y = data.label.values


# In[7]:


X = torch.tensor(data.drop('label', axis = 1).values)


# In[8]:


X[0].shape


# In[9]:


tfms = get_transforms(do_flip=False)


# In[10]:


rand_idx = torch.randperm(X.shape[0])
split_ratio = 0.8
split = int(X.shape[0] * split_ratio)
train_idxs = rand_idx[:split]
test_idxs  = rand_idx[split:]

X_train = X[train_idxs]
X_valid = X[test_idxs]
y_train = y[train_idxs]
y_valid = y[test_idxs]
X_train.shape, X_valid.shape


# In[11]:


def tensor2Images(x):
    return [Image(x[i].reshape(-1,28,28).repeat(3, 1, 1)/255.) for i in range(x.shape[0])]


# In[12]:


class MNISTImageList(ImageList):
    "`ImageList` of Images stored as in `items` as tensor."

    def open(self, fn):
        "No file associated to open"
        pass

    def get(self, i):
        res = self.items[i]
        self.sizes[i] = sys.getsizeof(res)
        return res


# In[13]:


til = MNISTImageList(tensor2Images(X_train))


# In[14]:


til[0]


# In[15]:


train_ll = LabelList(MNISTImageList(tensor2Images(X_train)),CategoryList(y_train, ['0','1','2','3','4','5','6','7','8','9']))
valid_ll = LabelList(MNISTImageList(tensor2Images(X_valid)),CategoryList(y_valid, ['0','1','2','3','4','5','6','7','8','9']))


# In[16]:


valid_ll[1][0]


# In[17]:


valid_ll[1][1]


# In[18]:


ll = LabelLists('.',train_ll,valid_ll)


# In[19]:


data.head()


# In[20]:


test_csv  = path/'test.csv'
data = pd.read_csv(test_csv)
Xtest = torch.tensor(data.drop('id', axis = 1).values)
test_il = ItemList(tensor2Images(Xtest))


# In[21]:


ll.add_test(test_il)


# In[22]:


assert len(ll.train.x)==len(ll.train.y)
assert len(ll.valid.x)==len(ll.valid.y)


# In[23]:


ll.train.x[0]


# In[24]:


dbch = ImageDataBunch.create_from_ll(ll)


# In[25]:


dbch.sanity_check()


# In[26]:


dbch


# In[27]:


dbch.show_batch(rows=4, figsize=(6,6))


# In[28]:


path = '/kaggle/working/'


# In[29]:


learn = cnn_learner(dbch,models.resnet50,metrics=accuracy, pretrained=True)


# In[30]:


learn.freeze()


# In[31]:


learn.lr_find()
learn.recorder.plot()


# In[32]:


learn.fit_one_cycle(10,slice(2e-3,2e-2))


# In[33]:


learn.recorder.plot_losses()


# In[34]:


learn.save('stage1')


# In[35]:


learn.unfreeze()


# In[36]:


learn.lr_find(start_lr=1e-9)
learn.recorder.plot()


# In[37]:


learn.fit_one_cycle(10,slice(2e-7,2e-6))


# In[38]:


learn.recorder.plot_losses()


# In[39]:


learn.show_results(ds_type=DatasetType.Train, rows=4, figsize=(8,10))


# In[40]:


learn.fit_one_cycle(5,max_lr=1e-6)


# In[41]:


learn.summary()


# In[42]:


learn.export()


# In[43]:


preds,y = learn.get_preds(ds_type=DatasetType.Test)


# In[44]:


y = preds.argmax(dim=1)


# In[45]:


assert len(y)==len(test_il)


# In[46]:


res = pd.DataFrame(y,columns=['label'],index=range(1, 5001))
res.index.name = 'id'


# In[47]:


res.to_csv('submission.csv')

