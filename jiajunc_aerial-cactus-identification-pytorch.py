#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pandas as pd
import cv2
from tqdm import tqdm
import torch 
import torch.backends.cudnn as cudnn
from torchvision import models
from torchvision import transforms as tfs
from torch.utils.data import Dataset, DataLoader
import time
from torch.autograd import Variable
from PIL import Image


df = pd.read_csv('../input/train.csv')    
df_test = pd.read_csv('../input/sample_submission.csv')
print(os.listdir("../input"))


# In[ ]:


def train_tf(x):
    im_aug = tfs.Compose([
        tfs.RandomHorizontalFlip(),
        tfs.RandomCrop(32),
        tfs.ToTensor()
    ])
    x = im_aug(x)
    return x


# In[ ]:


train_img = []
train_label = []
for i in tqdm(df.values):
    img = cv2.imread(os.path.join('../input', 'train/train', i[0]))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    tf_img = Image.fromarray(img.astype('uint8')).convert('RGB')
    if int(i[1]) == 1:
        tf_img1 = train_tf(tf_img)
        train_img.append((tf_img1, i[1]))
    else:
        for j in range(3):
            tf_img1 = train_tf(tf_img)
            train_img.append((tf_img1, i[1]))
    img= np.transpose(img.astype(np.float32), (2, 1, 0))
    img = torch.from_numpy(img)
    train_img.append((img, i[1]))

test_img = []
for i in tqdm(df_test.values):
    img = cv2.imread(os.path.join('../input', 'test/test', i[0]))
    img= np.transpose(img.astype(np.float32), (2, 1, 0))
    img = torch.from_numpy(img)
    test_img.append((img, i[1]))


# In[ ]:


import random
val_data = random.sample(train_img, int(0.1 * len(train_img)))
train_data = list(set(train_img).difference(set(val_data)))
print(len(train_data), len(val_data), len(train_data)+len(val_data))


# In[ ]:


model = models.resnet101(pretrained = False)
class_nums = 2
########修改最后一层输出
channel_in = model.fc.in_features
model.fc = torch.nn.Linear(channel_in, class_nums)
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5, momentum = 0.9)
loss_func = torch.nn.CrossEntropyLoss()


# In[ ]:


train_loader = DataLoader(dataset = train_data, batch_size = 256, shuffle = True)
val_loader = DataLoader(dataset = val_data, batch_size = 256)
model = torch.nn.DataParallel(model, device_ids=[0])
model.cuda()
cudnn.benchmark = True
for epoch in range(2000):
    batch_size_start = time.time()
    train_loss = 0.
    train_acc = 0.
    for trainData,trainLabel in train_loader:
        trainData= Variable(trainData.cuda())
        trainLabel = Variable(trainLabel.cuda())
        optimizer.zero_grad()
        out = model(trainData)
        
        pred = torch.max(out, 1)[1]
        train_correct = (pred == trainLabel).sum()
        train_acc += train_correct.item()
        
        
        loss = loss_func(out, trainLabel)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
#     print('Epoch [%d/%d],Train Loss: %.4f,Acc: %.4f, need time %.4f'
#                       % (epoch + 1, 500, train_loss / (len(train_data) / 256), train_acc / (len(train_data)/256), time.time() - batch_size_start))
    val_acc = 0.
    val_loss = 0.
    model.eval()
    for (valData,valLabels) in val_loader:
#         time_start = time.time()
        valData = Variable(valData.cuda())
        valLabels = Variable(valLabels.cuda())
        outputs = model(valData)
        val_loss += loss.item()
        predict = torch.max(outputs.data, 1)[1]
        correct = (predict == valLabels).sum()
        val_acc += correct.item()
    print('Epoch [%d/%d],Train Loss: %.4f,Train acc: %.4f, Val Loss: %.4f, Val acc: %.4f,  need time %.4f'
                      %(epoch + 1, 2000, train_loss / len(train_data) * 256,train_acc / len(train_data) / 256,val_loss / len(valData) / 256,val_acc / len(valData) / 256,  time.time() - batch_size_start))
    if (epoch+1) % 400 == 0:torch.save(model.state_dict(), str(epoch)+'v2.pkl')


# In[ ]:


# torch.save(model.state_dict(), '799v2.pkl')


# In[ ]:


class MakeSubmission:
    def __init__(self, test_img: list, csv_path: str, model_path: str):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        r = []
        self.test_img = test_img
        test_loader = DataLoader(dataset=test_img, batch_size=1)
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            r.append(int(preds))
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path)
        submission = pd.DataFrame({'id': self.df['id'], 'has_cactus': r})
        submission.to_csv(model_path+"sample_submission.csv", index=False)


# In[ ]:


MakeSubmission(test_img,  "../input/sample_submission.csv", '799v2.pkl')


# In[ ]:


# aaa = pd.read_csv('sample_submission.csv')


# In[ ]:


ls

