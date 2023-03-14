#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


import ast
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('../input/test_simplified.csv')
df['drawing'] = df['drawing'].apply(ast.literal_eval)


# In[4]:


df_show = df.iloc[:25]


# In[5]:


n = 5
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(8, 8))
for i, drawing in enumerate(df_show['drawing']):
    ax = axs[i//n, i%n]
    for x, y in drawing:
        ax.plot(x, -np.array(y))
    ax.axis('off')
plt.show()


# In[6]:


df_show['drawing'].apply(np.array).apply(np.shape)[:10]


# In[7]:


import cv2


# In[8]:


def to_pixel_matrix(drawing_ls, size=224, lw=6, time_color=True):
    '''
    점의 좌표(x,y)로 이뤄진 list를 224x224 정방형 pixel matrix로 변환하는 함수
    
    arguments
    ===============================================
    drawing_ls : list
        - 사진을 표현한 그래프 좌표(x,y)가 담긴 리스트
        - 'drawings' column
    
    size : int, default=224
        - pixel matrix의 길이 
        - default=224는 ResNet101의 default input size
    
    lw : int
        - tickness, 선분의 두께 
        
    time_color = bool
        - True : 선의 색(color)을 decay 시켜 선을 그리는 순서를 표현, 그림을 그리는 패턴을 포착하기 위함
        - False : 동일한 색으로 선을 표현
    '''
    base_size = 256
    img = np.zeros((base_size, base_size), np.uint8)
    for t, drawing in enumerate(drawing_ls):
        for i in range(len(drawing[0]) - 1):
            color = 255 - min(t, 10)*20 if time_color else 255
            img = cv2.line(
                img, 
                (drawing[0][i], drawing[1][i]),
                (drawing[0][i + 1], drawing[1][i + 1]), 
                color=color, 
                thickness=lw,
            )
    
    return cv2.resize(img, (size, size))


# In[9]:


img = to_pixel_matrix(df_show['drawing'][0], time_color=True, lw=5)
plt.imshow(img)


# In[10]:


img.shape


# In[11]:


from collections import Counter
import seaborn as sns


# In[12]:


#df = pd.concat([chunk for chunk in pd.read_csv('total_train.csv', chunksize=100000)])


# In[13]:


#len(set(df['word']))


# In[14]:


#word_frequency_dic = Counter(df['word'])


# In[15]:


#most_frequent_words = sorted(((val, key) for key, val in word_frequency_dic.items()), reverse=True)


# In[16]:


#most_frequent_words[:10]


# In[17]:


#most_frequent_words[-10:]


# In[18]:


#len(set(df['countrycode']))


# In[19]:


#most_frequent_codes = sorted(((val, key) for key, val in Counter(df['countrycode']).items()), reverse=True)


# In[20]:


#most_frequent_codes[:10]


# In[21]:


#most_frequent_codes[-10:]


# In[22]:


from collections import defaultdict


# In[23]:


'''
ratio_dic = defaultdict(lambda:[])

for _, code in most_frequent_codes[:30]:
    temp_df = df[df['countrycode'] == code]['word']
    counter = Counter(temp_df)
    
    for word, count in counter.items():
        count /= len(temp_df) # 각 label의 비율을 계산
        ratio_dic[word].append(count)
'''


# In[24]:


'''
label = 'baseball'
plt.figure(figsize=(8,4))
sns.distplot(ratio_dic[label], bins=15)
plt.title(label)
'''


# In[25]:


from torch.utils.data import Dataset, DataLoader


# In[26]:


label2idx = defaultdict(lambda: len(label2idx))

path_to_dir = '../input/train_simplified/'
file_name_ls = os.listdir(path_to_dir)

for file_name in file_name_ls:
    path_to_file = path_to_dir + file_name
    label = pd.read_csv(path_to_file, nrows=1, engine='python')['word'][0]
    
    label2idx[label] # label2idx 사전에 등록


# In[27]:


for i, (key, val) in enumerate(label2idx.items()):
    print(key, val)
    
    if i == 5:
        break


# In[28]:


cc2idx = defaultdict(lambda: len(cc2idx))
cc2idx['<UNK>'] # unk for less frequent country codes
path_to_file = '../input/train_simplified/snowman.csv'

country_code_ls = list(set(pd.read_csv(path_to_file, engine='python')['countrycode']))

# 가장 데이터 수가 많은 snowman을 기준으로 country code를 매김
for code in country_code_ls:
    cc2idx[code]


# In[29]:


len(cc2idx)


# In[30]:


import torch


# In[31]:


class QuickDrawDataset(Dataset):
    def __init__(self, path_to_file, cc2idx, label2idx, size=30000, chunk_idx=0, img_size=224, mode='train'):
        self.path_to_file = path_to_file
        self.cc2idx = cc2idx
        self.label2idx = label2idx
        self.size = size # resource의 한계로 제한된 양의 데이터만 사용
        self.chunk_idx = chunk_idx # train, validtaion을 나누기 위해 chunk의 번호를 지정
        self.img_size = img_size # ResNet101 default input size
        self.mode = mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # if path_to_file is directory
        if path_to_file[-1] == '/':
            file_name_ls = os.listdir(self.path_to_file)
            self.path_to_file_ls = [self.path_to_file+ file_name for file_name in file_name_ls]
        # if path_to_file is file_name
        else :
            self.path_to_file_ls = [self.path_to_file]
        
        chunk_ls = []
        if self.mode == 'train':
            use_col_ls = ['countrycode', 'drawing', 'word']
            for path in self.path_to_file_ls:
                df = pd.read_csv(path, usecols=use_col_ls, chunksize=self.size)
                
                for i, chunk in enumerate(df):
                    if i == self.chunk_idx:
                        chunk_ls.append(chunk)
                        break
                        
            self.df = pd.concat(chunk_ls, ignore_index=True)
            self.label = [label2idx[w] for w in self.df['word']] #[[label, label, label]]
        else :
            for path in self.path_to_file_ls:
                df = pd.read_csv(path, usecols=use_col_ls, chunksize=self.size)
                
                for i, chunk in enumerate(df):
                    if i == self.chunk_idx:
                        chunk_ls.append(chunk)
                        break
                        
            self.df = pd.concat(chunk_ls, ignore_index=True)
            
            
    @staticmethod
    def to_pixel_matrix(drawing_ls, size=224, lw=6, time_color=True):
        '''
        점의 좌표(x,y)로 이뤄진 list를 224x224 정방형 pixel matrix로 변환하는 함수

        arguments
        ===============================================
        drawing_ls : list
            - 사진을 표현한 그래프 좌표(x,y)가 담긴 리스트
            - 'drawings' column

        size : int, default=224
            - pixel matrix의 길이 
            - default=224는 ResNet101의 default input size

        lw : int
            - tickness, 선분의 두께 

        time_color = bool
            - True : 선의 색(color)을 decay 시켜 선을 그리는 순서를 표현, 그림을 그리는 패턴을 포착하기 위함
            - False : 동일한 색으로 선을 표현
        '''
        base_size = 256
        img = np.zeros((base_size, base_size), np.uint8)
        for t, drawing in enumerate(drawing_ls):
            for i in range(len(drawing[0]) - 1):
                color = 255 - min(t, 10)*20 if time_color else 255
                img = cv2.line(
                    img, 
                    (drawing[0][i], drawing[1][i]),
                    (drawing[0][i + 1], drawing[1][i + 1]), 
                    color=color, 
                    thickness=lw,
                )

        return cv2.resize(img, (size, size))
    
    def to_tensor(self, x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=self.device)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        country_code = self.df['countrycode'][idx]
        if country_code in self.cc2idx:
            country_code = self.cc2idx[country_code]
        else:
            country_code = self.cc2idx['<UNK>']
        country_code = self.to_tensor(country_code, dtype=torch.long)
        
        drawing_ls = ast.literal_eval(self.df['drawing'][idx])
        img = self.to_pixel_matrix(drawing_ls, size=self.img_size, lw=6, time_color=True)
        img = self.to_tensor(img[None]/255, dtype=torch.float32) # expand_dim, bound to 0~1
            
        if self.mode == 'train':
            label = self.label[idx]
            label = self.to_tensor(label, dtype=torch.long)
            return img, country_code, label 
        else:
            return img, country_code


# In[32]:


# 데이터의 idx 순서대로 0 ~ 15,000 사용
train_dataset = QuickDrawDataset(
    path_to_file='../input/train_simplified/',
    label2idx = label2idx,
    cc2idx = cc2idx,
    size=10000,
    chunk_idx=0
)


# In[33]:


#15,000 ~ 17,500번째 index만 사용
val_dataset = QuickDrawDataset(
    path_to_file='../input/train_simplified/',
    label2idx = label2idx,
    cc2idx = cc2idx,
    size=2000,
    chunk_idx=5
)


# In[34]:


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0)


# In[35]:


import torchvision
import torch


# In[36]:


resnet = torchvision.models.resnet18(pretrained=True)
# in-channel customizing (gray-scale 1D)
resnet.conv1 = torch.nn.Conv2d(1, 64, (7,7), stride=(2,2), padding=(3,3), bias=False) 
resnet.fc = torch.nn.Linear(512, 384, bias=True) # output_dim customizing


# In[37]:


class DrawingClassifier(torch.nn.Module):
    def __init__(self, img_net):
        super(DrawingClassifier, self).__init__()
        self.img_net = img_net
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.cc_embedding = torch.nn.Embedding(
            num_embeddings=190, #number of country code 
            embedding_dim=128,
        )
        
        self.fc = torch.nn.Linear(384+128, 340, bias=True)  # 384(resent) + 128(cc_embedding)
    
    def to_tensor(self, x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=self.device)
    
    def forward(self, img, cc):
        img = self.img_net(img)
        cc = self.cc_embedding(cc)
        x = torch.cat((img, cc), dim=1) # img + country_code
        
        out = self.fc(x)
        return torch.log_softmax(out, dim=-1)        


# In[38]:


model = DrawingClassifier(resnet)
model.to(model.device)


# In[39]:


for batch in train_loader:
    break


# In[40]:


plt.imshow(np.array(batch[0][0].cpu().numpy()*255, dtype=np.uint8)[0])


# In[41]:


import time

class Fitter() : 
    def __init__(self, model, train_loader, test_loader): 
        self.criterion = torch.nn.NLLLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[25000], gamma=0.5
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(model.device)
        
        self.train_loader = train_loader
        self.test_loader = test_loader
    
    def train_and_evaluate(self, n_epoch, test_epoch=1):       
        for epoch in range(1, n_epoch+1):
            print('=====================================================================================')
            print('Epoch: %s\n Train'%epoch)
            train_loss, train_score = self.train()
            
            if epoch % test_epoch == 0:
                print('=====================================================================================')
                print('Test')
                test_loss, test_score = self.evaluate()
        return
    
    def mapk(self, output, target, k=3):
        with torch.no_grad():
            batch_size = target.size(0)

            _, pred = output.topk(k, dim=1)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            for i in range(k):
                correct[i] = correct[i]*(k-i)

            score = correct[:k].view(-1).float().sum(0, keepdim=True)
            score.mul_(1.0 / (k * batch_size))
        return score
    
    def train(self):
        self.model.train()
        start_time = time.time()
        
        epoch_loss, score = 0, 0
        n_batch = 0
        
        for img_batch, cc_batch, y_batch in self.train_loader:
            output = self.model(img_batch, cc_batch) 
            loss = self.criterion(output, y_batch)

            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            epoch_loss += loss.item()
            score += self.mapk(output, y_batch).item()
            n_batch += 1
            
            if n_batch % 1000 == 0:
                print('Batch : %s, Loss : %.03f, Score : %.03f, Train Time : %.03f'                      %(n_batch, epoch_loss/n_batch, score/n_batch, time.time()-start_time))

            # 10000 batch씩만 학습
            if n_batch % 10000 == 0:
                break 
                
        return epoch_loss/n_batch, score/n_batch
    
    def evaluate(self):
        model.eval() # stop the every change in gradient of model
        start_time = time.time()
        
        epoch_loss, score = 0, 0
        n_batch = 0
        
        for img_batch, cc_batch, y_batch in self.test_loader:
            output = self.model(img_batch, cc_batch) 
            loss = self.criterion(output, y_batch)

            epoch_loss += loss.item()
            score += self.mapk(output, y_batch).item()
            n_batch += 1
            
            # 1000 배치씩만 테스트
            if n_batch % 1000 == 0:
                print('Batch : %s, Loss : %.03f, Score : %.03f, Test Time : %.03f'                      %(n_batch, epoch_loss/n_batch, score/n_batch, time.time() - start_time))
                break
                
        return epoch_loss/n_batch, score/n_batch


# In[42]:


args = {
    'model' : model,
    'train_loader' : train_loader,
    'test_loader' : val_loader,
}

fitter = Fitter(**args)


# In[43]:


fitter.train_and_evaluate(n_epoch=3)


# In[44]:


len(df)


# In[45]:


test_dataset = QuickDrawDataset(
    path_to_file= '../input/test_simplified.csv',
    label2idx = label2idx,
    cc2idx = cc2idx,
    mode='test',
    size=112199
)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)


# In[46]:


def predict(model, test_loader):
    result = np.array([])
    
    for i, (img_batch, cc_batch) in enumerate(test_loader):
        output = model(img_batch, cc_batch) 
        output = output.topk(3, dim=1)[1].cpu().numpy()
        
        if i == 0:
            result = output
        else:
            result = np.vstack((result, output))
    
    return result
        


# In[47]:


pred_ls = predict(model, test_loader)


# In[48]:


# decode
idx2label = {val : key for key,val in label2idx.items()}
pred_ls = [[idx2label[pred] for pred in preds] for preds in pred_ls] # idx2label
pred_ls = [' '.join(preds) for preds in pred_ls] # join

# save    
submission = pd.read_csv('test_simplified.csv')
submission.drop(['countrycode', 'drawing'], axis=1, inplace=True)
submission['word'] = pred_ls


# In[49]:


submission.to_csv('submission.csv', index=False)

