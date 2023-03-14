#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#from tqdm import tqdm_notebook as tqdm
#from tqdm import tqdm 
from tqdm.notebook import tqdm as tqdm

import cv2
import os
import re

import random

from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler




INPUT_DATA = "../input/global-wheat-detection/"
TRAIN_DIR = os.path.join(INPUT_DATA, "train")
TEST_DIR = os.path.join(INPUT_DATA, "test")




df = pd.read_csv(os.path.join(INPUT_DATA, "train.csv"))
df.head(5)




## Shape of Dataframe
print(f"Shape of train DataFrame: {df.shape}")
##Unique Images 
print(f'Unique Images in train DataFrame: {len(df["image_id"].value_counts())}')




## Extract x,y,w,h from bbox
import ast
def extract_bbox(DataFrame):
    DataFrame["x"] = [np.float(ast.literal_eval(i)[0]) for i in DataFrame["bbox"]]
    DataFrame["y"]=  [np.float(ast.literal_eval(i)[1]) for i in DataFrame["bbox"]]
    DataFrame["w"] = [np.float(ast.literal_eval(i)[2]) for i in DataFrame["bbox"]]
    DataFrame["h"] = [np.float(ast.literal_eval(i)[3]) for i in DataFrame["bbox"]]




extract_bbox(df)
df.head()




train_ratio = 0.8
images_id = df["image_id"].unique()
train_ids = images_id[:int(len(images_id)*train_ratio)]
valid_ids = images_id[int(len(images_id)*train_ratio):]

print(f"Total Images Number: {len(images_id)}")
print(f"Number of training images: {len(train_ids)}")
print(f"Number of Valid images: {len(valid_ids)}")




train_df = df[df["image_id"].isin(train_ids)]
valid_df = df[df["image_id"].isin(valid_ids)]

print(f"Shape of train_df: {train_df.shape}")
print(f"Shape of valid_df: {valid_df.shape}")




# Data Transform - Albumentation
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})




class WheatDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        super().__init__()
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.image_ids = dataframe["image_id"].unique()
        
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        details = self.dataframe[self.dataframe["image_id"]==image_id]
        img_path = os.path.join(TRAIN_DIR, image_id)+".jpg"
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        #Row of Dataframe of a particular index.
        boxes = details[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        #To find area
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        #COnvert it into tensor dataType
        area = torch.as_tensor(area, dtype=torch.float32)
        
        # there is only one class
        labels = torch.ones((details.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((details.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor(idx) ### <------------ New change list has been removed
        target['area'] = area
        target['iscrowd'] = iscrowd
        
        if self.transform:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            
            sample = self.transform(**sample)
            image = sample['image']
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
            target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.long)
        
        return image, target     #, image_id
    
    def __len__(self) -> int:
        return len(self.image_ids)




def collate_fn(batch):
    return tuple(zip(*batch))




train_dataset = WheatDataset(train_df, TRAIN_DIR, get_train_transform())
valid_dataset = WheatDataset(valid_df, TRAIN_DIR, get_valid_transform())

print(f"Length of train_dataset: {len(train_dataset)}")
print(f"Length of test_dataset: {len(valid_dataset)}")




##DataLoader
indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)




#PLot images
def plot_images(n_num, random_selection=True):
    '''Function to visualize N Number of images'''
    if random_selection:
        index = random.sample(range(0, len(train_df["image_id"].unique())-1), n_num)
    else:
        index = range(0, n_num)
    plt.figure(figsize=(15,15))
    fig_no = 1
    
    for i in index:
        images, targets = train_dataset.__getitem__(i)
        sample = np.array(np.transpose(images, (1,2,0)))
        boxes = targets["boxes"].numpy().astype(np.int32)
    
        #Plot figure/image

        for box in boxes:
            cv2.rectangle(sample,(box[0], box[1]),(box[2], box[3]),(255,223,0), 2)
        plt.subplot(n_num/2, n_num/2, fig_no)
        plt.imshow(sample)
        fig_no+=1




plot_images(4)




# load a model; pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)




print(model)




num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lr_scheduler = None

num_epochs = 8




#from engine import evaluate
import time

itr=1

total_train_loss = []
total_valid_loss = []

losses_value = 0
for epoch in range(num_epochs):
  
    start_time = time.time()
    train_loss = []
    model.train()
    
    #<-----------Training Loop---------------------------->
    pbar = tqdm(train_data_loader, desc = 'description')
    for images, targets in pbar:
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        losses_value = losses.item()
        train_loss.append(losses_value)        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        pbar.set_description(f"Epoch: {epoch+1}, Batch: {itr}, loss: {losses_value}")
        itr+=1

    epoch_train_loss = np.mean(train_loss)
    total_train_loss.append(epoch_train_loss)
    
    
    #<---------------Validation Loop---------------------->
    with torch.no_grad():
        valid_loss = []

        for images, targets in valid_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # If you need validation losses
            model.train()
            # Calculate validation losses
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            valid_loss.append(loss_value)
            
    epoch_valid_loss = np.mean(valid_loss)
    total_valid_loss.append(epoch_valid_loss)
    
    print(f"Epoch Completed: {epoch+1}/{num_epochs}, Time: {time.time()-start_time},    Train Loss: {epoch_train_loss}, Valid Loss: {epoch_valid_loss}")    




import seaborn as sns
plt.figure(figsize=(8,5))
sns.set_style(style="whitegrid")
sns.lineplot(range(1, len(total_train_loss)+1), total_train_loss, label="Training Loss")
sns.lineplot(range(1, len(total_train_loss)+1), total_valid_loss, label="Valid Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()




torch.save(model.state_dict(), 'fasterrcnn_best_resnet50.pth')

