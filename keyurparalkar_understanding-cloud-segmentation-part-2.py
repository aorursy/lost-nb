#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


from fastai import *
from fastai.vision import *
import pathlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tqdm


# In[3]:


import pandas as pd
sample_submission = pd.read_csv("../input/understanding_cloud_organization/sample_submission.csv")
train = pd.read_csv("../input/understanding_cloud_organization/train.csv")


# In[4]:


#Transform the csv file to Image name and label
train['Image_name'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
train['Label_name'] = train['Image_Label'].apply(lambda x: x.split('_')[1])


# In[5]:


train.drop('Image_Label',axis=1,inplace=True)


# In[6]:


train = train.pivot('Image_name','Label_name','EncodedPixels')


# In[7]:


train.head()


# In[8]:


data_path = pathlib.Path('/kaggle/input/understanding_cloud_organization/')
path_img = data_path/'train_images'


# In[9]:


item_list = (SegmentationItemList.
            from_df(df=train.reset_index(),path=path_img,cols="Image_name")
            .split_by_rand_pct(0.2))


# In[10]:


item_list


# In[11]:


class MultiLabelImageSegment(ImageSegment):
    """Store overlapping masks in separate image channels"""

    def show(self, ax:plt.Axes=None, figsize:tuple=(3,3), title:Optional[str]=None, hide_axis:bool=True,
        cmap:str='tab20', alpha:float=0.5, class_names=None, **kwargs):
        "Show the masks on `ax`."
             
        # put all masks into a single channel
        flat_masks = self.px[0:1, :, :].clone()
        for idx in range(1, self.shape[0]): # shape CxHxW
            mask = self.px[idx:idx+1, :, :] # slice tensor to a single mask channel
            # use powers of two for class codes to keep them distinguishable after sum 
            flat_masks += mask * 2**idx
        
        # use same color normalization in image and legend
        norm = matplotlib.colors.Normalize(vmin=0, vmax=2**self.shape[0]-1)
        ax = show_image(Image(flat_masks), ax=ax, hide_axis=hide_axis, cmap=cmap, norm=norm,
                        figsize=figsize, interpolation='nearest', alpha=alpha, **kwargs)
        
        # custom legend, see https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/custom_legends.html
        cm = matplotlib.cm.get_cmap(cmap)
        legend_elements = []
        for idx in range(self.shape[0]):
            c = 2**idx
            label = class_names[idx] if class_names is not None else f"class {idx}"
            line = Line2D([0], [0], color=cm(norm(c)), label=label, lw=4)
            legend_elements.append(line)
        ax.legend(handles=legend_elements)
        
        # debug info
        # ax.text(10, 10, f"px={self.px.size()}", {"color": "white"})
        
        if title: ax.set_title(title)

    def reconstruct(self, t:Tensor): 
        return MultiClassImageSegment(t)


# In[12]:


#Source: https://www.kaggle.com/keyurparalkar/multi-label-segmentation-using-fastai/
def bce_logits_floatify(input,target,reduction='mean'):
    return F.binary_cross_entropy_with_logits(input,target.float(),reduction=reduction)


# In[13]:


class MultiLabelSegmentationLabelList(SegmentationLabelList):
    """return a single image segment with all classes"""
    def __init__(self, items:Iterator, src_img_size=None, classes:Collection = None, **kwargs):
        super().__init__(items=items,classes=classes,**kwargs)
        self.loss_func = bce_logits_floatify
        self.src_img_size = src_img_size
        self.copy_new += ['src_img_size']
        
    def open(self,rles):
        masks = torch.zeros((len(self.classes),*self.src_img_size))  # shape CxHxW
        for i, rle in enumerate(rles):
            if(isinstance(rle,str)):
                rle_to_mask = open_mask_rle(rle,self.src_img_size)
                masks[i] = rle_to_mask.px.permute(0,2,1)
        return MultiLabelImageSegment(masks)
    
    def analyze_pred(self,pred, thres:float=0.0):
        #Binary masks
        return (pred > thres).float()
    
    def reconstruct(self, t:Tensor):
        return MultiLabelImageSegment(t)
        


# In[14]:


class_names = ['Fish','Flower','Gravel','Sugar'] 


# In[15]:


def get_mask_rle(img):
    img = img.split("/")[-1]    #get file name only
    return train.loc[img, class_names].to_list()


# In[16]:


# Reduce the image size into multiples of 4.
img_size = (84,132)

#Train and test image sizes:
train_img_dims = (1400, 2100) 

img_size

batch_size=8


# In[17]:


classes = [0,1,2,3]


# In[18]:


item_list = item_list.label_from_func(func=get_mask_rle,label_cls=MultiLabelSegmentationLabelList,classes=classes,
                                     src_img_size=train_img_dims)


# In[19]:


item_list = item_list.add_test_folder(data_path/'test_images',label="")


# In[20]:


batch_size = 8

tfms = ([],[])
item_list  = item_list.transform(tfms,tfm_y=True,size=img_size)


# In[21]:


data = (item_list.databunch(bs=batch_size)
       .normalize(imagenet_stats))


# In[22]:


data.show_batch(rows=2,figsize=(15,10),class_names=class_names)


# In[23]:


# adapted from: https://www.kaggle.com/iafoss/unet34-dice-0-87
# can use sigmoid on the input too, in this case the threshold would be 0.5
def dice_metric(pred, targs, threshold=0):
    pred = (pred > threshold).float()
    targs = targs.float()  # make sure target is float too
    return 2.0 * (pred*targs).sum() / ((pred+targs).sum() + 1.0)


# In[24]:


# learn = unet_learner(data, models.resnet18, metrics=[dice_metric], wd = 1e-2).to_fp16()


# In[25]:


learn.summary()


# In[26]:


#Set path for saving the model
learn.model_dir = '/kaggle/working/'


# In[27]:


learn.lr_find()
learn.recorder.plot()


# In[28]:


learn.fit_one_cycle(10,max_lr=1e-4)


# In[29]:


learn.save('trained_model_fit1cyc',return_path=True)


# In[30]:


get_ipython().system('pip install pydrive')


# In[31]:


from pydrive.drive import GoogleDrive


# In[32]:


from pydrive.auth import GoogleAuth

gauth = GoogleAuth()


# In[33]:


drive = GoogleDrive(gauth)


# In[ ]:





# In[34]:


learn.show_results(imgsize=8, class_names=class_names)


# In[35]:


#getting prediction on test dataset
a,b = learn.get_preds(ds_type=DatasetType.Test,with_loss=False)


# In[36]:


from IPython.core.debugger import set_trace


# In[37]:


def resize_masks(pred:Tensor, img_size=(4,350,525)) -> list:
    for i in range(pred.shape[0]):
#         set_trace()
        mask = MultiLabelImageSegment(pred[i])
        yield mask.resize(img_size)        


# In[38]:


resized_preds = resize_masks(a)


# In[39]:


test_fnames = [str(fname).split('/')[-1] for fname in learn.data.test_ds.items]
test_fnames[:5]


# In[40]:


#Function for creating submission file:
def writeSubFile():
    thres = 0 #Defining threshold for comparing it with masks
    
    with open('/kaggle/working/submission.csv','w') as f:
        print('Writing submission file ...')
        f.write('Image_Label,EncodedPixels\n')
        
        for img_name, mask in zip(test_fnames, resize_masks(a)):
            preds = mask.data > thres
            for i,cls_name in enumerate(class_names):
                rle = rle_encode(preds[i])
                f.write(f"{img_name}_{cls_name},{rle}\n")
                
    print('Submission file created ...')


# In[41]:


writeSubFile()


# In[42]:


with open('/kaggle/working/submission.csv')


# In[43]:


a.shape


# In[ ]:




