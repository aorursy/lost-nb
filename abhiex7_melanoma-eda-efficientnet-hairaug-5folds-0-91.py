#!/usr/bin/env python
# coding: utf-8



from IPython.display import HTML
HTML('<center><iframe width="700" height="400" src="https://www.youtube.com/embed/mkYBxfKDyv0?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe></center>')








get_ipython().system('pip install efficientnet_pytorch torchtoolbox')




import os 
import gc
import re
import time
import datetime


import math
import random

#Data importing libraries
import cv2
from scipy import ndimage
import numpy as np
import pandas as pd
from kaggle_datasets import KaggleDatasets
import missingno as msno


#Ploting libraries

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.offline import iplot
from plotly.subplots import make_subplots
from colorama import Fore, Back, Style

#Data Preprocessing Libraries
from tqdm import tqdm
tqdm.pandas()
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import preprocessing

# PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchtoolbox.transform as transforms

# Data Augmentation for Image Preprocessing


from efficientnet_pytorch import EfficientNet






import warnings
warnings.filterwarnings("ignore")





IMAGE_PATH = "../input/siim-isic-melanoma-classification/jpeg/"
TEST_PATH = "../input/siim-isic-melanoma-classification/test.csv"
TRAIN_PATH = "../input/siim-isic-melanoma-classification/train.csv"
SUB_PATH = "../input/siim-isic-melanoma-classification/sample_submission.csv"


sub = pd.read_csv(SUB_PATH)
test_df = pd.read_csv(TEST_PATH)
train_df = pd.read_csv(TRAIN_PATH)




train_df.head()




test_df.head()




print(Fore.MAGENTA +"Sex:",Style.RESET_ALL,train_df["sex"].unique())
print(Fore.GREEN +"-----------------------",Style.RESET_ALL)
print(Fore.CYAN +"Anatomy Site:",Style.RESET_ALL,train_df["anatom_site_general_challenge"].unique())
print(Fore.GREEN +"-----------------------",Style.RESET_ALL)
print(Fore.YELLOW +"Target:",Style.RESET_ALL,train_df["target"].unique())
print(Fore.GREEN +"-----------------------",Style.RESET_ALL)
print(Fore.BLUE +"Diagnosis:",Style.RESET_ALL,train_df["diagnosis"].unique())




def load_image(img_name,df="train"):
    
    file_path = img_name+".jpg"
    image=IMAGE_PATH+"train/"+file_path
    
    image_disp = plt.imread(image)
    return image_disp
train_imgs = train_df["image_name"][:100].progress_apply(load_image)




red_values = [np.mean(train_imgs[idx][:, :, 0]) for idx in range(len(train_imgs))]
green_values = [np.mean(train_imgs[idx][:, :, 1]) for idx in range(len(train_imgs))]
blue_values = [np.mean(train_imgs[idx][:, :, 2]) for idx in range(len(train_imgs))]
values = [np.mean(train_imgs[idx]) for idx in range(len(train_imgs))]




fig = ff.create_distplot([values], group_labels=["Channels"], colors=["purple"])
fig.update_layout(showlegend=False, template="simple_white")
fig.update_layout(title_text="Distribution of channel values")
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.data[0].marker.line.width = 0.5
fig.show()




fig = ff.create_distplot([red_values], group_labels=["R"], colors=["red"])
fig.update_layout(showlegend=False, template="simple_white")
fig.update_layout(title_text="Distribution of red channel values")
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.data[0].marker.line.width = 0.5
fig.show()




fig = ff.create_distplot([green_values], group_labels=["G"], colors=["green"])
fig.update_layout(showlegend=False, template="simple_white")
fig.update_layout(title_text="Distribution of green channel values")
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.data[0].marker.line.width = 0.5
fig




fig = ff.create_distplot([blue_values], group_labels=["B"], colors=["blue"])
fig.update_layout(showlegend=False, template="simple_white")
fig.update_layout(title_text="Distribution of blue channel values")
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.data[0].marker.line.width = 0.5
fig




fig = go.Figure()

for idx, values in enumerate([red_values, green_values, blue_values]):
    if idx == 0:
        color = "Red"
    if idx == 1:
        color = "Green"
    if idx == 2:
        color = "Blue"
    fig.add_trace(go.Box(x=[color]*len(values), y=values, name=color, marker=dict(color=color.lower())))
    
fig.update_layout(yaxis_title="Mean value", xaxis_title="Color channel",
                  title="Mean value vs. Color channel", template="plotly_white")




del train_imgs




malignant_df=pd.DataFrame(data=train_df[train_df["target"]==1])




benign_df=pd.DataFrame(data=train_df[train_df["target"]==0])




def get_images(df):
    df_name=df["image_name"].values
    df_imgs = [np.random.choice(df_name+'.jpg') for i in range(9)]
    img_dir = IMAGE_PATH+'train'
    return df_imgs,img_dir




def disp_imgs(df_imgs,img_dir):
    plt.figure(figsize=(10,8))
    for i in range(9):
        plt.subplot(6, 3, i + 1)
        img = plt.imread(os.path.join(img_dir, df_imgs[i]))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    return plt.tight_layout()   




maldf_imgs,malimg_dir=get_images(malignant_df)
bedf_imgs,bedimg_dir=get_images(benign_df)




disp_imgs(maldf_imgs,malimg_dir)




disp_imgs(bedf_imgs,bedimg_dir)




def sobel_filter(df_imgs,img_dir):
    plt.figure(figsize=(10,8))
    for i in range(9):
        
        plt.subplot(6, 3, i + 1)
        img = plt.imread(os.path.join(img_dir, df_imgs[i]))
        
        img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         
        img = cv2.GaussianBlur(img,(99,99),0)
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
        theta = np.sqrt((sobely**2)+(sobelx**2))
        plt.imshow(theta,cmap='gray')
        plt.axis('off')
    return plt.tight_layout()
        




sobel_filter(maldf_imgs,malimg_dir)




sobel_filter(bedf_imgs,bedimg_dir)




def canny_filter(df_imgs,img_dir,sigma=0.99):
    plt.figure(figsize=(10,8))
    for i in range(9):
        
        plt.subplot(6, 3, i + 1)
        img = plt.imread(os.path.join(img_dir, df_imgs[i]))
        img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        v=np.median(img)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        
        img = cv2.Canny(img,lower,upper)
        plt.imshow(img,cmap='gray')
        plt.axis('off')
    return plt.tight_layout()
        




canny_filter(maldf_imgs,malimg_dir)




canny_filter(bedf_imgs,bedimg_dir)




del maldf_imgs,malimg_dir,bedf_imgs,bedimg_dir,benign_df,malignant_df




cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
ax = sns.heatmap(train_df.corr(), annot=True,cmap=cmap)




f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 5))

msno.matrix(train_df, ax = ax1, color=(189/255, 240/255, 250/255), fontsize=10)
msno.matrix(test_df, ax = ax2, color=(238/255, 189/255, 250/255), fontsize=10)

ax1.set_title('Train Missing Values Map', fontsize = 13)
ax2.set_title('Test Missing Values Map', fontsize = 13);




ax = sns.countplot(x=train_df['sex'], data=train_df)




anat=sns.countplot(x=train_df['anatom_site_general_challenge'],data = train_df,palette=sns.cubehelix_palette(8))




fig = ff.create_distplot([train_df.loc[train_df['target'] == 1,'age_approx'].dropna()], group_labels=["Age"], colors=["magenta"])
fig.update_layout(showlegend=False, template="simple_white")
fig.update_layout(title_text="Distribution of age for malignant cases")
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.data[0].marker.line.width = 1
fig




fig = ff.create_distplot([train_df.loc[train_df['target'] == 0,'age_approx'].dropna()], group_labels=["Age"], colors=["orange"])
fig.update_layout(showlegend=False, template="simple_white")
fig.update_layout(title_text="Distribution of age form benign cases")
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.data[0].marker.line.width = 1
fig




def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed = 1234
seed_everything(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




train_df=pd.read_csv('../input/melanoma-external-malignant-256/train_concat.csv')




train_df['sex'] = train_df['sex'].map({'male': 1, 'female': 0})
test_df['sex'] = test_df['sex'].map({'male': 1, 'female': 0})
train_df['sex'] = train_df['sex'].fillna(-1)
test_df['sex'] = test_df['sex'].fillna(-1)




imp_mean=(train_df["age_approx"].sum())/(train_df["age_approx"].count()-train_df["age_approx"].isna().sum())
train_df['age_approx']=train_df['age_approx'].fillna(imp_mean)
train_df['age_approx'].head()
imp_mean_test=(test_df["age_approx"].sum())/(test_df["age_approx"].count())
test_df['age_approx']=test_df['age_approx'].fillna(imp_mean_test)




train_df['patient_id'] = train_df['patient_id'].fillna(0)




concat = pd.concat([train_df['anatom_site_general_challenge'], test_df['anatom_site_general_challenge']], ignore_index=True)
dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
train_df = pd.concat([train_df, dummies.iloc[:train_df.shape[0]]], axis=1)
test_df = pd.concat([test_df, dummies.iloc[train_df.shape[0]:].reset_index(drop=True)], axis=1)




meta_features = ['sex', 'age_approx'] + [col for col in train_df.columns if 'site_' in col]
meta_features.remove('anatom_site_general_challenge')




test_df=test_df.drop(["anatom_site_general_challenge"],axis=1)
train_df=train_df.drop(["anatom_site_general_challenge"],axis=1)




train_df.head()




test_df.head()




print(Fore.YELLOW,meta_features)





     
class HairGrowth:
    


    def __init__(self, hairs , hairs_folder):
        self.hairs = hairs
        self.hairs_folder = hairs_folder

    def __call__(self, img):
    
        
        n_hairs = random.randint(0, self.hairs)
        
        if not n_hairs:
            return img
        
        height, width, _ = img.shape  # target image width and height
        hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]
        
        for _ in range(n_hairs):
            hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(hair_images)))
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            # Creating a mask and inverse mask
            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of hair in ROI
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Take only region of hair from hair image.
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            # Put hair in ROI and modify the target image
            dst = cv2.add(img_bg, hair_fg)

            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst
                
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, hairs_folder="{self.hairs_folder}")'




train_transform = transforms.Compose([
    HairGrowth(hairs = 5,hairs_folder='/kaggle/input/melanoma-hairs/'),
    transforms.RandomResizedCrop(size=256, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=32. / 255.,saturation=0.5,hue=0.01),
    

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([HairGrowth(hairs = 5,hairs_folder='/kaggle/input/melanoma-hairs/'),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])




class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms = None, meta_features = None):
        
        self.df = df
        self.imfolder = imfolder
        self.transforms = transforms
        self.train = train
        self.meta_features = meta_features
        
    def __getitem__(self, index):
        im_path = os.path.join(self.imfolder, self.df.iloc[index]['image_name'] + '.jpg')
        image = cv2.imread(im_path)
        metadata = np.array(self.df.iloc[index][self.meta_features].values, dtype=np.float32)

        if self.transforms:
            image = self.transforms(image)
            
        if self.train:
            y = self.df.iloc[index]['target']
            image = image.cuda()
            return (image, metadata), y
        else:
            return (image, metadata)
    
    def __len__(self):
        return len(self.df)
    
    




skf = GroupKFold(n_splits=5)




test = MelanomaDataset(df=test_df,
                       imfolder='/kaggle/input/melanoma-external-malignant-256/test/test/', 
                       train=False,
                       transforms=test_transform,
                       meta_features=meta_features)




from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass




# Config

epochs = 10  # no of times till the loop will iterate over the model
ESpatience = 3 # no of times the model will wait if the loss is not decreased
TTA = 3      # test time augmentation, random augmantation like mirror image performed on thhe input image 
num_workers = 6 # tells DataLoader the number of subprocess to use while data loading
learning_rate = 0.001 # Learning Rate
weight_decay = 0.0  # Decay Factor
lr_patience = 1     # patience for learning rate      
lr_factor = 0.4     
output_size=1    # statics
batch_size1 = 32
batch_size2 = 16

train_len = len(train_df)
test_len = len(test_df)
oof = np.zeros(shape = (train_len, 1))




class EfficientNetwork(nn.Module):
    def __init__(self, output_size, no_columns, b4=False, b2=False):
        super().__init__()
        self.b4, self.b2, self.no_columns = b4, b2, no_columns
        
        # Define Feature part (IMAGE)
        if b4:
            self.features = EfficientNet.from_pretrained('efficientnet-b4')
        elif b2:
            self.features = EfficientNet.from_pretrained('efficientnet-b2')
        else:
            self.features = EfficientNet.from_pretrained('efficientnet-b7')
        
        # (CSV) or Meta Features
        self.csv = nn.Sequential(nn.Linear(self.no_columns, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3),
                                 
                                 nn.Linear(250, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3),
                                 
                                 nn.Linear(250, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3))
        
        # Define Classification part
        if b4:
            self.classification = nn.Sequential(nn.Linear(1792 + 250, 250),
                                                nn.Linear(250, output_size))
        elif b2:
            self.classification = nn.Sequential(nn.Linear(1408 + 250, 250),
                                                nn.Linear(250, output_size))
        else:
            self.classification = nn.Sequential(nn.Linear(2560 + 250, 250),
                                                nn.Linear(250, output_size))
        
        
    def forward(self, image, csv_data, prints=False):    
        
        if prints: print('Input Image shape:', image.shape, '\n'+
                         'Input csv_data shape:', csv_data.shape)
        
        # IMAGE CNN
        image = self.features.extract_features(image)
        if prints: print('Features Image shape:', image.shape)
            
        if self.b4:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1792)
        elif self.b2:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1408)
        else:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 2560)
        if prints: print('Image Reshaped shape:', image.shape)
            
        # CSV FNN
        csv_data = self.csv(csv_data)
        if prints: print('CSV Data:', csv_data.shape)
            
        # Concatenate
        image_csv_data = torch.cat((image, csv_data), dim=1)
        
        # CLASSIF
        out = self.classification(image_csv_data)
        if prints: print('Out shape:', out.shape)
        
        return out




#comment out in you don't want to Train

for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.zeros(len(train_df)), y=train_df['target'], groups=train_df['patient_id'].tolist()), 1):
    print(Fore.CYAN,'-' * 20,Style.RESET_ALL,Fore.MAGENTA, 'Fold', fold,Style.RESET_ALL,Fore.CYAN, '-' * 20,Style.RESET_ALL)
    best_val = None
    patience=ESpatience# Best validation score within this fold
    model_path = 'model{Fold}.pth'.format(Fold=fold)  
    train = MelanomaDataset(df=train_df.iloc[train_idx].reset_index(drop=True), 
                            imfolder='/kaggle/input/melanoma-external-malignant-256/train/train/', 
                            train=True, 
                            transforms=train_transform,
                            meta_features=meta_features)
    val = MelanomaDataset(df=train_df.iloc[val_idx].reset_index(drop=True), 
                            imfolder='/kaggle/input/melanoma-external-malignant-256/train/train/', 
                            train=True, 
                            transforms=test_transform,
                            meta_features=meta_features)
    train_loader = DataLoader(dataset=train, batch_size=batch_size1, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val, batch_size=batch_size2, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test, batch_size=batch_size2, shuffle=False, num_workers=0)
    
    model = EfficientNetwork(output_size=output_size, no_columns=len(meta_features),b2=True)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', 
                                      patience=lr_patience, verbose=True, factor=lr_factor)
    for epoch in range(epochs):
        start_time = time.time()
        correct = 0
        train_losses = 0

        model.train() #Set the model in train mode
        
        for data, labels in train_loader:
                # Save them to device
                data[0] = torch.tensor(data[0], device=device, dtype=torch.float32)
                data[1] = torch.tensor(data[1], device=device, dtype=torch.float32)
                labels = torch.tensor(labels, device=device, dtype=torch.float32)
                
                criterion = nn.BCEWithLogitsLoss()

                # Clear gradients first; very important, usually done BEFORE prediction
                optimizer.zero_grad()

                # Log Probabilities & Backpropagation
                out = model(data[0], data[1])
                loss = criterion(out, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()

                # --- Save information after this batch ---
                # Save loss
                # From log probabilities to actual probabilities
                 # 0 and 1
                train_preds = torch.round(torch.sigmoid(out))
                train_losses += loss.item()
                
                # Number of correct predictions
                correct += (train_preds.cpu() == labels.cpu().unsqueeze(1)).sum().item()

            # Compute Train Accuracy
        train_acc = correct / len(train_idx)
        model.eval()  # switch model to the evaluation mode
        val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
        with torch.no_grad():  # Do not calculate gradient since we are only predicting
            
            for j,(data_val, label_val) in enumerate(val_loader):
                data_val[0] = torch.tensor(data_val[0], device=device, dtype=torch.float32)
                data_val[1] = torch.tensor(data_val[1], device=device, dtype=torch.float32)
                label_val = torch.tensor(label_val, device=device, dtype=torch.float32)
                z_val = model(data_val[0],data_val[1])
                val_pred = torch.sigmoid(z_val)
                val_preds[j*data_val[0].shape[0]:j*data_val[0].shape[0] + data_val[0].shape[0]] = val_pred
            val_acc = accuracy_score(train_df.iloc[val_idx]['target'].values, torch.round(val_preds.cpu()))
            val_roc = roc_auc_score(train_df.iloc[val_idx]['target'].values, val_preds.cpu())
                
            epochval=epoch + 1
            
            print(Fore.YELLOW,'Epoch: ',Style.RESET_ALL,epochval,'|',Fore.CYAN,'Loss: ',Style.RESET_ALL,train_losses,'|',Fore.GREEN,'Train acc:',Style.RESET_ALL,train_acc,'|',Fore.BLUE,' Val acc: ',Style.RESET_ALL,val_acc,'|',Fore.RED,' Val roc_auc:',Style.RESET_ALL,val_roc,'|',Fore.YELLOW,' Training time:',Style.RESET_ALL,str(datetime.timedelta(seconds=time.time() - start_time)))
                 
                
                
                 
                
            
            scheduler.step(val_roc)
            # During the first iteration (first epoch) best validation is set to None
            if not best_val:
                best_val = val_roc  # So any validation roc_auc we have is the best one for now
                torch.save(model, model_path)  # Saving the model
                continue
                
            if val_roc >= best_val:
                best_val = val_roc
                patience = patience  # Resetting patience since we have new best validation accuracy
                torch.save(model, model_path)  # Saving current best model
            else:
                patience -= 1
                if patience == 0:
                    print(Fore.BLUE,'Early stopping. Best Val roc_auc: {:.3f}'.format(best_val),Style.RESET_ALL)
                    break
                        
    model = torch.load(model_path)  # Loading best model of this fold
    model.eval()  # switch model to the evaluation mode
    val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        # Predicting on validation set once again to obtain data for OOF
        for j, (x_val, y_val) in enumerate(val_loader):
            x_val[0] = torch.tensor(x_val[0], device=device, dtype=torch.float32)
            x_val[1] = torch.tensor(x_val[1], device=device, dtype=torch.float32)
            y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
            z_val = model(x_val[0],x_val[1])
            val_pred = torch.sigmoid(z_val)
            val_preds[j*x_val[0].shape[0]:j*x_val[0].shape[0] + x_val[0].shape[0]] = val_pred
        oof[val_idx] = val_preds.cpu().numpy()
        
        




test_loader = DataLoader(dataset=test, batch_size=batch_size2, shuffle=False, num_workers=0)




print('Out of the Folds Score:',roc_auc_score(train_df['target'], oof))




model = torch.load('/kaggle/input/melanoma/model3.pth')
model.eval()  # switch model to the evaluation mode
preds = torch.zeros((len(test), 1), dtype=torch.float32, device=device)
with torch.no_grad():
    for _ in range(TTA):  
            for i, x_test in enumerate(test_loader):  
                x_test[0] = torch.tensor(x_test[0], device=device, dtype=torch.float32)
                x_test[1] = torch.tensor(x_test[1], device=device, dtype=torch.float32) 
                z_test = model(x_test[0],x_test[1])
                z_test = torch.sigmoid(z_test)
                preds[i*x_test[0].shape[0]:i*x_test[0].shape[0] + x_test[0].shape[0]] += z_test
    preds /= TTA
            
             
    
    gc.collect()   
           
preds /= skf.n_splits 




sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
sub['target'] = preds.cpu().numpy().reshape(-1,)
sub.to_csv('submission.csv', index=False)

