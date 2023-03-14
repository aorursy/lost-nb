#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as im
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import PIL
from PIL import Image
import albumentations as A

"""
Peak at data - note only 3.3k unique images not alot which means will have to augment the images with albumentations library
"""
train_df = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")

print(len(train_df["image_id"].unique()))

train_df.head()


# In[2]:


# imgs_df=train_df["image_id"].unique()

# image_id = imgs_df[50]

# im=Image.open("/kaggle/input/global-wheat-detection/train/" + image_id +".jpg")
# enhancer = PIL.ImageEnhance.Contrast(im)
# im_output = enhancer.enhance(2)

# out=np.array(im_output)


# In[3]:


"""
Peak at image
"""

imgs_df=train_df["image_id"].unique()

image_id = imgs_df[45]

img = im.imread("/kaggle/input/global-wheat-detection/train/" + image_id +".jpg")

boxes = list(train_df["bbox"][ train_df["image_id"]==image_id ].values)

box=[]
for i,l in enumerate(boxes): 
    b=[float(num) for num in l[1:-1].split(",")] 
    #boxes[i]=[b[0],b[1],b[0]+b[2],b[1]+b[3]]
    #box.append([b[0],b[1],b[0]+b[2],b[1]+b[3]])
    box.append(b)

def print_im(image,bboxes):
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    for c in bboxes:
        rect = matplotlib.patches.Rectangle((c[0],c[1]),c[2],c[3],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.show()
    
###----------------------

#format sets the format for the bounding boxes
transform = A.Compose([
    #A.RandomCrop(width=450, height=450),
    A.Resize(512, 512),
    A.VerticalFlip(p=1),
    #A.HorizontalFlip(p=1),
    A.RandomBrightnessContrast(p=1),
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
#bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=['class_labels']))

transformed = transform( image=img, bboxes=box, class_labels=["wheat"]*len(box) )
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']

#print_im(img,box)
print_im(transformed_image,transformed_bboxes)


# In[4]:


class ImageDataset(Dataset):
    def __init__(self, root="/kaggle/input/global-wheat-detection/", tt="train",transforms_tt=True):
        
        df=pd.read_csv("{}{}.csv".format(root,tt))
        
        self.root = root
        self.transforms = transforms
        self.imgs = df["image_id"].unique()
        self.df=df
        self.tt=tt
        self.transform=None
        if transforms_tt is True:
            self.transform=A.Compose( [  A.Resize(512, 512),
                                         A.VerticalFlip(p=0.25),
                                         A.HorizontalFlip(p=0.25),
                                         A.RandomBrightnessContrast(p=0.35)],
                                         #ToTensorV2], 
                                      bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    def __getitem__(self, idx):
        # load images ad masks
        image_id = self.imgs[idx]
        pic_path="/kaggle/input/global-wheat-detection/{}/{}.jpg".format(self.tt,image_id)
        
        im=Image.open( pic_path )
        enhancer = PIL.ImageEnhance.Contrast(im)
        im_output = enhancer.enhance(2)
        
        _image=np.array( im_output )
        #_image = im.imread( pic_path)
        #image = Image.open( "/kaggle/input/global-wheat-detection/{}/{}.jpg".format(self.tt,image_id) )
        
        records = self.df["bbox"][self.df['image_id'] == image_id].values
        boxes=[]
        for i,l in enumerate(records): 
            b=[float(num) for num in l[1:-1].split(",")] 
            boxes.append ( b )
        
        #area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target = {}
        target["labels"] = torch.ones((records.shape[0],), dtype=torch.int64)
        #target["area"] = area
        target["iscrowd"] = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        if self.transforms is not None:
            transformed = self.transform( image=_image, bboxes=boxes, class_labels=["wheat"]*len(boxes) )
            img = transformed['image']
            img=torchvision.transforms.functional.to_tensor(img)
            transformed_bboxes = transformed['bboxes']
            bboxes=[]
            for b in transformed_bboxes:
                bboxes.append([b[0],b[1],b[0]+b[2],b[1]+b[3]])
            target["boxes"]=torch.as_tensor(bboxes, dtype=torch.float32)
            
        if self.transform is None:
            bboxes=[]
            for b in boxes:
                bboxes.append([b[0],b[1],b[0]+b[2],b[1]+b[3]])
            target["boxes"]=bboxes
            img=torchvision.transforms.functional.to_tensor(_image)
        
        del records
        del _image
        
        return img, target
Finetune Faster RCNN with Pytorch to Detect Wheat heads in Images

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torchvision

import matplotlib.pyplot as plt

import matplotlib

import matplotlib.image as im

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

import PIL

from PIL import Image

import albumentations as A

​

"""

Peak at data - note only 3.3k unique images not alot which means will have to augment the images with albumentations library

"""

train_df = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")

​

print(len(train_df["image_id"].unique()))

​

train_df.head()

​

3373

	image_id 	width 	height 	bbox 	source
0 	b6ab77fd7 	1024 	1024 	[834.0, 222.0, 56.0, 36.0] 	usask_1
1 	b6ab77fd7 	1024 	1024 	[226.0, 548.0, 130.0, 58.0] 	usask_1
2 	b6ab77fd7 	1024 	1024 	[377.0, 504.0, 74.0, 160.0] 	usask_1
3 	b6ab77fd7 	1024 	1024 	[834.0, 95.0, 109.0, 107.0] 	usask_1
4 	b6ab77fd7 	1024 	1024 	[26.0, 144.0, 124.0, 117.0] 	usask_1
Take a peak at the images and detection boxes

Also check out how the albumentations library affects the images.

# imgs_df=train_df["image_id"].unique()

​

# image_id = imgs_df[50]

​

# im=Image.open("/kaggle/input/global-wheat-detection/train/" + image_id +".jpg")

# enhancer = PIL.ImageEnhance.Contrast(im)

# im_output = enhancer.enhance(2)

​

# out=np.array(im_output)

​

"""

Peak at image

"""

​

imgs_df=train_df["image_id"].unique()

​

image_id = imgs_df[45]

​

img = im.imread("/kaggle/input/global-wheat-detection/train/" + image_id +".jpg")

​

boxes = list(train_df["bbox"][ train_df["image_id"]==image_id ].values)

​

box=[]

for i,l in enumerate(boxes): 

    b=[float(num) for num in l[1:-1].split(",")] 

    #boxes[i]=[b[0],b[1],b[0]+b[2],b[1]+b[3]]

    #box.append([b[0],b[1],b[0]+b[2],b[1]+b[3]])

    box.append(b)

​

def print_im(image,bboxes):

    

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(image)

    for c in bboxes:

        rect = matplotlib.patches.Rectangle((c[0],c[1]),c[2],c[3],linewidth=1,edgecolor='r',facecolor='none')

        ax.add_patch(rect)

    plt.show()

    

###----------------------

​

#format sets the format for the bounding boxes

transform = A.Compose([

    #A.RandomCrop(width=450, height=450),

    A.Resize(512, 512),

    A.VerticalFlip(p=1),

    #A.HorizontalFlip(p=1),

    A.RandomBrightnessContrast(p=1),

], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

#bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=['class_labels']))

​

transformed = transform( image=img, bboxes=box, class_labels=["wheat"]*len(box) )

transformed_image = transformed['image']

transformed_bboxes = transformed['bboxes']

​

#print_im(img,box)

print_im(transformed_image,transformed_bboxes)

Create the Dataset object

class ImageDataset(Dataset):

    def __init__(self, root="/kaggle/input/global-wheat-detection/", tt="train",transforms_tt=True):

        

        df=pd.read_csv("{}{}.csv".format(root,tt))

        

        self.root = root

        self.transforms = transforms

        self.imgs = df["image_id"].unique()

        self.df=df

        self.tt=tt

        self.transform=None

        if transforms_tt is True:

            self.transform=A.Compose( [  A.Resize(512, 512),

                                         A.VerticalFlip(p=0.25),

                                         A.HorizontalFlip(p=0.25),

                                         A.RandomBrightnessContrast(p=0.35)],

                                         #ToTensorV2], 

                                      bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

​

    def __getitem__(self, idx):

        # load images ad masks

        image_id = self.imgs[idx]

        pic_path="/kaggle/input/global-wheat-detection/{}/{}.jpg".format(self.tt,image_id)

        

        im=Image.open( pic_path )

        enhancer = PIL.ImageEnhance.Contrast(im)

        im_output = enhancer.enhance(2)

        

        _image=np.array( im_output )

        #_image = im.imread( pic_path)

        #image = Image.open( "/kaggle/input/global-wheat-detection/{}/{}.jpg".format(self.tt,image_id) )

        

        records = self.df["bbox"][self.df['image_id'] == image_id].values

        boxes=[]

        for i,l in enumerate(records): 

            b=[float(num) for num in l[1:-1].split(",")] 

            boxes.append ( b )

        

        #area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}

        target["labels"] = torch.ones((records.shape[0],), dtype=torch.int64)

        #target["area"] = area

        target["iscrowd"] = torch.zeros((records.shape[0],), dtype=torch.int64)

        

        if self.transforms is not None:

            transformed = self.transform( image=_image, bboxes=boxes, class_labels=["wheat"]*len(boxes) )

            img = transformed['image']

            img=torchvision.transforms.functional.to_tensor(img)

            transformed_bboxes = transformed['bboxes']

            bboxes=[]

            for b in transformed_bboxes:

                bboxes.append([b[0],b[1],b[0]+b[2],b[1]+b[3]])

            target["boxes"]=torch.as_tensor(bboxes, dtype=torch.float32)

            

        if self.transform is None:

            bboxes=[]

            for b in boxes:

                bboxes.append([b[0],b[1],b[0]+b[2],b[1]+b[3]])

            target["boxes"]=bboxes

            img=torchvision.transforms.functional.to_tensor(_image)

        

        del records

        del _image

        

        return img, target

​

    def __len__(self):

        return len(self.imgs)

Check that the dataset object worked as expected

"""

Check.

"""

dataset = ImageDataset()

data_loader = DataLoader(dataset,batch_size=50,collate_fn=lambda batch: list(zip(*batch)) )

​

images, targets= next(iter(data_loader))

​

idx=45

​

img= images[idx].permute(1,2,0).numpy()

​

print(img.shape)

​

fig, ax = plt.subplots(figsize=(10, 10))

​

ax.imshow(img)

​

boxes=targets[idx]["boxes"].numpy()

​

for c in boxes:

    rect = matplotlib.patches.Rectangle((c[0],c[1]),c[2]-c[0],c[3]-c[1],linewidth=1,edgecolor='r',facecolor='none')

    ax.add_patch(rect)

​

​

plt.show()

(512, 512, 3)

Download the model

"""

Download and set up model

"""     

torchvision.__version__

​

model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, pretrained_backbone=True, trainable_backbone_layers=5)

​

num_classes = 2  # 1 class (wheat) + background

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

​

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

Downloading: "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth" to /root/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

100%
160M/160M [00:26<00:00, 6.39MB/s]


Set up training loop

from torch import optim

import time

start_time = time.time()

​

dataset = ImageDataset()

data_loader = DataLoader(dataset,batch_size=10,collate_fn=lambda batch: list(zip(*batch)) )

​

EPOCHS=8

​

model = model.to(device)

model.train()

​

​

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.0008, momentum=0.9, weight_decay=0.0005)

#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5,cooldown=20,factor=0.65,min_lr=0.00001,verbose=True)

​

print("Begin training")

lossAvg,lossPer=[],[]

for epoch in range(EPOCHS):

    total_loss,count=0,0

    for batch in data_loader:

        #check if targets is a list

        images,targets=batch

        

        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

​

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        

        if count%10==0:

            #scheduler.step(losses)

            print("loss: {}".format( losses.item() ))

            lossPer.append(losses.item())

        count+=1

        total_loss+=losses.item()

    g=total_loss/count 

    lossAvg.append( g )

    print("END EPOCH #{} avg: {}".format(epoch,total_loss/count))

    

print(" ")

print(" ")

print("Training Time: {}".format( time.time() - start_time ))

Begin training
loss: 5.3773603439331055
loss: 1.7676422595977783
loss: 1.6178905963897705
loss: 1.4473177194595337
loss: 1.2948307991027832
loss: 1.1440315246582031
loss: 1.2207309007644653
loss: 1.2663609981536865
loss: 1.1774954795837402
loss: 1.144769549369812
loss: 1.0742299556732178
loss: 1.0932400226593018
loss: 1.112673282623291
loss: 0.8862013816833496
loss: 0.7781498432159424
loss: 1.2107676267623901
loss: 1.076627492904663
loss: 0.9980862140655518
loss: 0.8189986348152161
loss: 0.9870530962944031
loss: 0.8758689165115356
loss: 0.8170798420906067
loss: 1.201951265335083
loss: 0.8513198494911194
loss: 0.8411060571670532
loss: 0.6542643904685974
loss: 1.0925242900848389
loss: 0.8526032567024231
loss: 0.8829998970031738
loss: 0.6939114928245544
loss: 0.840462863445282
loss: 0.7291010022163391
loss: 1.2138803005218506
loss: 1.144069790840149
END EPOCH #0 avg: 1.0848297739522696
loss: 1.1295771598815918
loss: 0.9422412514686584
loss: 1.2606966495513916
loss: 1.0974177122116089
loss: 0.9704293012619019
loss: 0.8956283330917358
loss: 1.0221991539001465
loss: 1.1031385660171509
loss: 1.0429081916809082
loss: 1.081375241279602
loss: 1.029093861579895
loss: 1.0231525897979736
loss: 1.0729111433029175
loss: 0.7949182391166687
loss: 0.6968480944633484
loss: 1.044675350189209
loss: 0.9089928865432739
loss: 0.9083682298660278
loss: 0.7775700092315674
loss: 0.8898879289627075
loss: 0.8356339931488037
loss: 0.7464484572410583
loss: 1.1672194004058838
loss: 0.74812251329422
loss: 0.7185171246528625
loss: 0.6108216047286987
loss: 1.0366991758346558
loss: 0.795727550983429
loss: 0.791501522064209
loss: 0.6359931230545044
loss: 0.8216689825057983
loss: 0.715979814529419
loss: 1.1524872779846191
loss: 1.0816165208816528
END EPOCH #1 avg: 0.9159213260433378
loss: 1.1054511070251465
loss: 0.9043039679527283
loss: 1.1845911741256714
loss: 1.0641263723373413
loss: 0.9565077424049377
loss: 0.8626723885536194
loss: 0.9958709478378296
loss: 1.079559087753296
loss: 1.0012873411178589
loss: 1.0347480773925781
loss: 1.0234328508377075
loss: 0.9892176389694214
loss: 1.024899959564209
loss: 0.7989320158958435
loss: 0.6749310493469238
loss: 1.0227930545806885
loss: 0.8956414461135864
loss: 0.8723995089530945
loss: 0.7228422164916992
loss: 0.8645232915878296
loss: 0.802834153175354
loss: 0.7171276211738586
loss: 1.1454535722732544
loss: 0.7112791538238525
loss: 0.751304566860199
loss: 0.5773029327392578
loss: 0.9820866584777832
loss: 0.7787880897521973
loss: 0.6983950734138489
loss: 0.6155737042427063
loss: 0.7751432061195374
loss: 0.6831026673316956
loss: 1.0839111804962158
loss: 1.043839931488037
END EPOCH #2 avg: 0.8833970891300743
loss: 1.083410620689392
loss: 0.8858187198638916
loss: 1.1823391914367676
loss: 1.081310510635376
loss: 0.9618273973464966
loss: 0.8539998531341553
loss: 0.9593894481658936
loss: 1.0246202945709229
loss: 1.008090615272522
loss: 1.0244140625
loss: 1.0037574768066406
loss: 0.9788201451301575
loss: 1.0234441757202148
loss: 0.7437090873718262
loss: 0.6481432318687439
loss: 0.953135073184967
loss: 0.8553492426872253
loss: 0.8461558222770691
loss: 0.7225533127784729
loss: 0.8381380438804626
loss: 0.76811683177948
loss: 0.7189196944236755
loss: 1.1184853315353394
loss: 0.723427951335907
loss: 0.7179039120674133
loss: 0.5872145891189575
loss: 0.9288179874420166
loss: 0.7350367307662964
loss: 0.7283239960670471
loss: 0.6243577003479004
loss: 0.7144389152526855
loss: 0.7042189836502075
loss: 1.0888075828552246
loss: 1.0356618165969849
END EPOCH #3 avg: 0.8650050099784806
loss: 1.071728229522705
loss: 0.839587926864624
loss: 1.15238356590271
loss: 1.051287055015564
loss: 0.9267892241477966
loss: 0.8182893991470337
loss: 0.9587641954421997
loss: 1.03286612033844
loss: 0.9621620178222656
loss: 0.9935327768325806
loss: 0.9907262921333313
loss: 0.9586783647537231
loss: 1.0137196779251099
loss: 0.7292211651802063
loss: 0.6346433162689209
loss: 0.9699356555938721
loss: 0.8741214871406555
loss: 0.8274796009063721
loss: 0.6668477654457092
loss: 0.8392725586891174
loss: 0.7519924640655518
loss: 0.7219560146331787
loss: 1.1282408237457275
loss: 0.6899096965789795
loss: 0.6869916319847107
loss: 0.5826075077056885
loss: 0.9374849200248718
loss: 0.7292343378067017
loss: 0.6891161799430847
loss: 0.6209076046943665
loss: 0.7288148403167725
loss: 0.673041582107544
loss: 1.0544297695159912
loss: 1.0301791429519653
END EPOCH #4 avg: 0.8527683896073223
loss: 1.0556403398513794
loss: 0.8644134998321533
loss: 1.134766697883606
loss: 1.0547491312026978
loss: 0.9298425912857056
loss: 0.8254854679107666
loss: 0.9497174620628357
loss: 1.0174678564071655
loss: 0.960101306438446
loss: 0.9652133584022522
loss: 0.9794471263885498
loss: 0.9848933815956116
loss: 0.9919844269752502
loss: 0.7459942698478699
loss: 0.6435001492500305
loss: 0.984830915927887
loss: 0.854371190071106
loss: 0.8223806023597717
loss: 0.6488444209098816
loss: 0.7983894348144531
loss: 0.7479805946350098
loss: 0.6899083852767944
loss: 1.0918748378753662
loss: 0.6922121644020081
loss: 0.7084393501281738
loss: 0.5830597877502441
loss: 0.900880753993988
loss: 0.7332385778427124
loss: 0.6541018486022949
loss: 0.6129364371299744
loss: 0.7105181813240051
loss: 0.645356297492981
loss: 1.115420937538147
loss: 1.0224666595458984
END EPOCH #5 avg: 0.8461936740480231
loss: 1.0488696098327637
loss: 0.8805951476097107
loss: 1.1222046613693237
loss: 1.0275280475616455
loss: 0.9013416171073914
loss: 0.8165730237960815
loss: 0.9252338409423828
loss: 1.0125350952148438
loss: 0.9250748157501221
loss: 1.0036269426345825
loss: 0.9902605414390564
loss: 0.9469006061553955
loss: 1.0053725242614746
loss: 0.727440595626831
loss: 0.6181288957595825
loss: 0.9528722167015076
loss: 0.8416427373886108
loss: 0.8427289724349976
loss: 0.6555745601654053
loss: 0.7798478603363037
loss: 0.7155237793922424
loss: 0.6813979148864746
loss: 1.061938762664795
loss: 0.64375239610672
loss: 0.6701700687408447
loss: 0.5810566544532776
loss: 0.9237428307533264
loss: 0.7555829882621765
loss: 0.6798772811889648
loss: 0.6036335229873657
loss: 0.7505882382392883
loss: 0.6340476274490356
loss: 1.0630533695220947
loss: 1.031473994255066
END EPOCH #6 avg: 0.8362922913576725
loss: 1.0409913063049316
loss: 0.8402179479598999
loss: 1.1149544715881348
loss: 1.0278797149658203
loss: 0.9337402582168579
loss: 0.821302056312561
loss: 0.9374873042106628
loss: 0.995248019695282
loss: 0.945940375328064
loss: 1.0007140636444092
loss: 0.9755243062973022
loss: 0.9258332848548889
loss: 1.0028263330459595
loss: 0.7320798635482788
loss: 0.629856288433075
loss: 0.973311722278595
loss: 0.80784672498703
loss: 0.7938651442527771
loss: 0.6481931209564209
loss: 0.7761248350143433
loss: 0.7249721884727478
loss: 0.6993616819381714
loss: 1.0777709484100342
loss: 0.6563688516616821
loss: 0.6769535541534424
loss: 0.5799486637115479
loss: 0.9209359288215637
loss: 0.7030892372131348
loss: 0.6731210350990295
loss: 0.6070143580436707
loss: 0.7212318778038025
loss: 0.6338415741920471
loss: 1.0464491844177246
loss: 1.0131089687347412
END EPOCH #7 avg: 0.8302864148419284
 
 
Training Time: 5442.206134796143

Test trained model on images

test_imgs=[]

for file in os.listdir("/kaggle/input/global-wheat-detection/test/"):

    test_imgs.append(file)

​

model = model.to(device)

model.eval()

​

print("Begin testing")

​

predsA,scoresA=[],[]

for image_id in test_imgs:

​

    #img = im.imread("/kaggle/input/global-wheat-detection/test/{}".format(image_id))

    pic_path = "/kaggle/input/global-wheat-detection/test/{}".format(image_id)

    img=Image.open( pic_path )

    enhancer = PIL.ImageEnhance.Contrast(img)

    im_output = enhancer.enhance(2)

​

    img=np.array( im_output )

        

    print( img.shape )

    img=torchvision.transforms.functional.to_tensor(img).to(device)

    preds = model([img])[0]

    

    #print(preds)

    

    predsA.append( preds["boxes"].detach().cpu().numpy() )

    scoresA.append( preds["scores"].detach().cpu().numpy() )

​

​

#torch.save(model.state_dict(), 'fasterRCNN_101.pth')

​

Begin testing
(1024, 1024, 3)
(1024, 1024, 3)
(1024, 1024, 3)
(1024, 1024, 3)
(1024, 1024, 3)
(1024, 1024, 3)
(1024, 1024, 3)
(1024, 1024, 3)
(1024, 1024, 3)
(1024, 1024, 3)

print(scoresA)

[array([0.97807443, 0.9753739 , 0.97003776, 0.9584341 , 0.9491141 ,
       0.9333785 , 0.9311873 , 0.86149406, 0.8526346 , 0.8466793 ,
       0.8051504 , 0.8004479 , 0.7830257 , 0.7636314 , 0.74014527,
       0.61116433, 0.5696377 , 0.49029648, 0.4890823 , 0.45735744,
       0.36451694, 0.36082876, 0.3417122 , 0.30376065, 0.27701193,
       0.26202998, 0.2519644 , 0.19107942, 0.18816645, 0.1390586 ,
       0.13788958, 0.11886665, 0.10898799, 0.09486211, 0.08454068,
       0.07993189, 0.07656579, 0.07069565, 0.06685372, 0.06659518,
       0.06460207, 0.06321312, 0.06282053, 0.05628582, 0.05579887,
       0.05426925, 0.05326167, 0.05139584], dtype=float32), array([0.9880822 , 0.98526037, 0.98469514, 0.9840082 , 0.98274857,
       0.9821832 , 0.9815285 , 0.9799073 , 0.9760517 , 0.97459203,
       0.9673014 , 0.9664239 , 0.9641636 , 0.9609325 , 0.95094067,
       0.9371962 , 0.9316037 , 0.89728373, 0.8591849 , 0.85757965,
       0.8561401 , 0.7270324 , 0.72366536, 0.52253556, 0.4932982 ,
       0.27712357, 0.23671183, 0.20237227, 0.1486367 , 0.13519245,
       0.13128284, 0.11732569, 0.09069648, 0.08852869, 0.08696767,
       0.08276133, 0.07664495, 0.06272565, 0.06268521, 0.0579011 ,
       0.0526136 , 0.05180103], dtype=float32), array([0.96926874, 0.9686326 , 0.96595055, 0.963764  , 0.963527  ,
       0.9605326 , 0.95130205, 0.9506546 , 0.94426364, 0.93546236,
       0.9339509 , 0.9182916 , 0.91349477, 0.9056246 , 0.8467633 ,
       0.76552916, 0.75501007, 0.751316  , 0.42639443, 0.37764174,
       0.31778452, 0.2775363 , 0.20944071, 0.20469317, 0.19141531,
       0.17881714, 0.1765811 , 0.17129898, 0.1618251 , 0.14996876,
       0.13941824, 0.1222906 , 0.11696491, 0.11379523, 0.10783732,
       0.0995105 , 0.08536722, 0.07617257, 0.07592005, 0.0751294 ,
       0.07172281, 0.06568614, 0.06424253, 0.06302674, 0.0559894 ,
       0.05454595, 0.05410636, 0.05233131], dtype=float32), array([0.9906482 , 0.9889233 , 0.98746127, 0.9858483 , 0.98507684,
       0.9846064 , 0.98418623, 0.9830482 , 0.982822  , 0.9820518 ,
       0.9771261 , 0.97487295, 0.974647  , 0.9720864 , 0.9718478 ,
       0.97070247, 0.9681864 , 0.9676042 , 0.9624396 , 0.96031547,
       0.9231909 , 0.89654034, 0.89006037, 0.87983066, 0.7935773 ,
       0.6586753 , 0.47726876, 0.34778586, 0.3177151 , 0.19899267,
       0.17562541, 0.15428853, 0.14881663, 0.14308447, 0.12808037,
       0.1142918 , 0.09796544, 0.09389071, 0.09190769, 0.09175417,
       0.07347901, 0.07244246, 0.07065839, 0.06753799, 0.06291909,
       0.05980609, 0.05719017, 0.0548024 ], dtype=float32), array([0.981058  , 0.9790075 , 0.9747347 , 0.9724988 , 0.97099733,
       0.97004336, 0.96750456, 0.9641676 , 0.9613948 , 0.9599943 ,
       0.9567954 , 0.95253134, 0.94180477, 0.93772703, 0.93274325,
       0.9322776 , 0.9310487 , 0.9268871 , 0.9222758 , 0.85463804,
       0.84856707, 0.62681216, 0.60830355, 0.47720996, 0.33076078,
       0.20617661, 0.17868303, 0.17134936, 0.15014587, 0.13783568,
       0.1285725 , 0.11202935, 0.08783555, 0.08355907, 0.07882313,
       0.07356532, 0.0558968 ], dtype=float32), array([0.9859066 , 0.97469455, 0.9727947 , 0.9677788 , 0.9666885 ,
       0.96276265, 0.9521699 , 0.9420549 , 0.937476  , 0.9308226 ,
       0.91539234, 0.86911494, 0.8339428 , 0.7447064 , 0.68574244,
       0.6792095 , 0.6782478 , 0.62064093, 0.44809556, 0.3744241 ,
       0.36416447, 0.30141994, 0.2963251 , 0.2800056 , 0.2442305 ,
       0.23109524, 0.20950526, 0.1923969 , 0.1911544 , 0.17397188,
       0.17362803, 0.13694075, 0.12820157, 0.11839604, 0.11618487,
       0.11051968, 0.10176627, 0.10091935, 0.09999033, 0.08622714,
       0.08428813, 0.08384799, 0.06906297, 0.06730299, 0.05441818,
       0.05317609, 0.05232513], dtype=float32), array([0.9904417 , 0.98794216, 0.98743385, 0.9870027 , 0.9853139 ,
       0.98453563, 0.98327625, 0.9830178 , 0.9779382 , 0.97510934,
       0.9731213 , 0.9716904 , 0.96447146, 0.96224684, 0.9616763 ,
       0.96087414, 0.95982945, 0.95008993, 0.94770426, 0.9441898 ,
       0.9209652 , 0.91897523, 0.9084852 , 0.9063913 , 0.8931875 ,
       0.8844235 , 0.87914735, 0.7625479 , 0.7620654 , 0.70382804,
       0.69746155, 0.6780654 , 0.67320704, 0.6643801 , 0.6268324 ,
       0.3819922 , 0.38195658, 0.19163412, 0.18259083, 0.13692348,
       0.10954501, 0.0788341 , 0.06336594, 0.05507063, 0.05135394],
      dtype=float32), array([0.9888786 , 0.9887102 , 0.9887101 , 0.9887038 , 0.98769337,
       0.9874636 , 0.98576593, 0.9846907 , 0.9835909 , 0.98356676,
       0.98221177, 0.9780001 , 0.97488153, 0.97219163, 0.9706516 ,
       0.96896714, 0.9667318 , 0.9560996 , 0.9305987 , 0.928595  ,
       0.9283766 , 0.8719414 , 0.81975615, 0.7699161 , 0.3963539 ,
       0.39000934, 0.16117196, 0.16087157, 0.15147269, 0.1266664 ,
       0.11887079, 0.10065451, 0.08527206, 0.08496463, 0.07349262,
       0.06381813, 0.05736136, 0.05336627, 0.05210874, 0.0513335 ,
       0.05026396], dtype=float32), array([0.9820967 , 0.97006893, 0.9696916 , 0.96810406, 0.96587634,
       0.96471936, 0.9629513 , 0.95973283, 0.9596717 , 0.9514466 ,
       0.93655187, 0.9303815 , 0.92793506, 0.92087674, 0.915446  ,
       0.89878964, 0.8660321 , 0.86346656, 0.8480074 , 0.8468629 ,
       0.80761194, 0.6321597 , 0.3658223 , 0.18738379, 0.160436  ,
       0.10506778, 0.08925476, 0.07362913, 0.07258227], dtype=float32), array([0.9901336 , 0.98769826, 0.98649085, 0.98544127, 0.9834849 ,
       0.9820218 , 0.9799243 , 0.97848207, 0.97819513, 0.97787744,
       0.9719902 , 0.96999425, 0.9694765 , 0.96923906, 0.9682566 ,
       0.96564865, 0.9641009 , 0.95245874, 0.9515392 , 0.949231  ,
       0.9373602 , 0.93562853, 0.9313646 , 0.9210315 , 0.8641027 ,
       0.86239034, 0.8339733 , 0.81361616, 0.52072227, 0.4717853 ,
       0.21615691, 0.15515174, 0.1385189 , 0.13105182, 0.12034093,
       0.11816803, 0.11382288, 0.11207241, 0.10904637, 0.07651953,
       0.07386469, 0.06472921, 0.06423542, 0.06102907, 0.05260476],
      dtype=float32)]

import matplotlib

​

​

fig, ax = plt.subplots(10,figsize=(60,60))

​

for i,boxes in enumerate(predsA):

    img = im.imread("/kaggle/input/global-wheat-detection/test/{}".format(test_imgs[i]))

    ax[i].imshow(img)

    for c in boxes:

        rect = matplotlib.patches.Rectangle((c[0],c[1]),c[2]-c[0],c[3]-c[1],linewidth=1,edgecolor='r',facecolor='none')

        ax[i].add_patch(rect)

​

plt.show()

---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-28-82cf3d006f4b> in <module>
      5 
      6 for i,boxes in enumerate(predsA):
----> 7     img = im.imread("/kaggle/input/global-wheat-detection/test/{}".format(test_imgs[i]))
      8     ax[i].imshow(img)
      9     for c in boxes:

AttributeError: 'JpegImageFile' object has no attribute 'imread'



    def __len__(self):
        return len(self.imgs)


# In[6]:


"""
Check.
"""
dataset = ImageDataset()
data_loader = DataLoader(dataset,batch_size=50,collate_fn=lambda batch: list(zip(*batch)) )

images, targets= next(iter(data_loader))

idx=45

img= images[idx].permute(1,2,0).numpy()

print(img.shape)

fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(img)

boxes=targets[idx]["boxes"].numpy()

for c in boxes:
    rect = matplotlib.patches.Rectangle((c[0],c[1]),c[2]-c[0],c[3]-c[1],linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)


plt.show()


# In[7]:


"""
Download and set up model
"""     
torchvision.__version__

model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, pretrained_backbone=True, trainable_backbone_layers=5)

num_classes = 2  # 1 class (wheat) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# In[8]:


from torch import optim
import time
start_time = time.time()

dataset = ImageDataset()
data_loader = DataLoader(dataset,batch_size=10,collate_fn=lambda batch: list(zip(*batch)) )

EPOCHS=32

model = model.to(device)
model.train()


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5,cooldown=20,factor=0.65,min_lr=0.00001,verbose=True)

print("Begin training")
lossAvg,lossPer=[],[]
for epoch in range(EPOCHS):
    total_loss,count=0,0
    for batch in data_loader:
        #check if targets is a list
        images,targets=batch
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if count%10==0:
            #scheduler.step(losses)
            print("loss: {}".format( losses.item() ))
            lossPer.append(losses.item())
        count+=1
        total_loss+=losses.item()
    g=total_loss/count 
    lossAvg.append( g )
    print("END EPOCH #{} avg: {}".format(epoch,total_loss/count))
    
print(" ")
print(" ")
print("Training Time: {}".format( time.time() - start_time ))


# In[9]:


import matplotlib
import matplotlib.image as im_age

def make_contrast(pic_path):
    img=Image.open( pic_path )
    enhancer = PIL.ImageEnhance.Contrast(img)
    im_output = enhancer.enhance(3)

    return np.array( im_output )


test_imgs=[]
for file in os.listdir("/kaggle/input/global-wheat-detection/test/"):
    test_imgs.append(file)

model = model.to(device)
model.eval()

print("Begin testing")

predsA,scoresA=[],[]
for image_id in test_imgs:

    pic_path = "/kaggle/input/global-wheat-detection/test/{}".format(image_id)
    #img=make_contrast(pic_path)
    img = im_age.imread(pic_path)
        
    print( img.shape )
    img=torchvision.transforms.functional.to_tensor(img).to(device)
    preds = model([img])[0]
    
    #print(preds)
    
    predsA.append( preds["boxes"].detach().cpu().numpy() )
    scoresA.append( preds["scores"].detach().cpu().numpy() )


#torch.save(model.state_dict(), 'fasterRCNN_101.pth')


# In[10]:


print( len(scoresA[0]) )
print( predsA[0] )


# In[11]:


import matplotlib
import matplotlib.image as im_age

fig, ax = plt.subplots(10,figsize=(60,60))

for i,pic in enumerate(predsA):
    img = im_age.imread("/kaggle/input/global-wheat-detection/test/{}".format(test_imgs[i]))
    ax[i].imshow(img)
    for j,box in enumerate(pic):
        if scoresA[i][j]>0.2:
            rect = matplotlib.patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')
            ax[i].add_patch(rect)

plt.show()


# In[12]:


# test_df=pd.read_csv("{}{}.csv".format("/kaggle/input/global-wheat-detection/","test"))

# test_df


#print(os.listdir("/kaggle/input/global-wheat-detection/",))

#for i,boxes in enumerate(predsA):

