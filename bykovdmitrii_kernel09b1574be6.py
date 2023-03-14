#!/usr/bin/env python
# coding: utf-8



import torchvision
import os
import torch
import pandas as pd
import random
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils




p = pd.read_csv("../input/train.csv")









p.head()




p.head(-1)




dataset = []




for row in p.iterrows():
    key, target = list(row[1])#ну вот так только умею
    try:
        print(row[0],key, target)
        #pict = io.imread(os.path.join("../input/train_images", key + ".png")) #да это шиза но вдруг
        dataset = dataset + [(key, [target == i for i in range(5)], target)]
    except:
        pass




io.imread(os.path.join("../input/test_images", "086727c22b75.png")).shape




dataset[1201]














#ds = BlindDataset(dataset)




#ds[0]['image'].shape




import torch.nn as nn









class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.resnet_layers = torch.nn.Sequential(*(list(torchvision.models.resnet50(pretrained=False, progress=False).children())[:-1])).cuda()#обрезать так чьобы ее получить нужен инет 
        self.linear_layers = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 5), nn.Softmax()).cuda()
    
    def forward(self, x):
        x = x.unsqueeze_(0)
        #print(x.shape)
        x = self.resnet_layers(x)
        x = x.view(x.size(0), -1)
      #  print(x.shape)
        return self.linear_layers(x)




model = Classifier()




#ds = BlindDataset(dataset)
#model.forward(ds[0]['image'])




class BlindDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset 
        self.transform = transform
        classes_idx = [[]]*5
        for i in range(len(dataset)):
            classes_idx[dataset[i][2]] = classes_idx[dataset[i][2]] + [i]
        self.classes_idx = [np.array(classes_idx[i]) for i in range(5)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target, idex_target = self.dataset[idx]
        image = io.imread(os.path.join("../input/train_images", image+".png"))
        image = transform.resize(image, (224, 224))
        flip = random.randint(0, 9)
        if(flip > 4):
            image = image[::-1]
        flip %= 4 
        for i in range(flip):
            image = np.transpose(np.fliplr(image), (1,0,2))
      #  image = image - image.mean()
       # image = image / image.std()
        image = image.copy()
        sample = {'image': torch.FloatTensor(image.transpose(2,0,1)).cuda(), 'target': torch.FloatTensor([target]).cuda()}
        return sample

    def get_batch(self, one_class_size=5):
        for i in range(5):
            random.shuffle(self.classes_idx[i][:-5])
        idxes = [(self.classes_idx[i][j]) for i in range(5) for j in range(one_class_size)]
        random.shuffle(idxes)
        return idxes

    def get_valid_batch(self):
        idxes = [(self.classes_idx[i][j - 5]) for i in range(5) for j in range(5)]
        random.shuffle(idxes)
        return idxes




dataset[5]




ds = BlindDataset(dataset)




ds.get_valid_batch()




ds.get_batch()




a = ds[49]['target']




def train(model, epoch, ds):
    loss = torch.nn.MSELoss()
    #loss = torch.nn.NLLLoss()
    ep_loss = 0
    lr=0.00001
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    checkpoint = {'model': Classifier(),
                  'state_dict': model.state_dict(),
                  'optimizer' : optim.state_dict()}

    torch.save(checkpoint, '../checkpoint1.pth')
    #во тут можно не дообучать старую часть хз насколько так надо делать
    for ep in range(epoch):
        accur = 0.0
        itert = 0
        for el_ind in ds.get_batch(40):
            model.zero_grad()
            elem = ds[el_ind]
            itert += 1
            prediction = model.forward(elem['image'])
           # print(prediction)
           # print(elem['target'])
            loss_iter = loss(prediction[0], elem['target'][0])
            accur += 0.0 + float(prediction.argmax() == elem['target'].argmax())
            loss_iter.backward()
            l = loss_iter.item()
            ep_loss += l
            ln = len(ds)
            optim.step()
           # print(prediction.argmax(), elem['target'].argmax())
        print(ep, ep_loss, accur / itert)#в процентах аж
        ep_loss = 0
        if ep % 10 == 0:
            checkpoint = {'model': Classifier(),
                          'state_dict': model.state_dict(),
                          'optimizer' : optim.state_dict()}

            torch.save(checkpoint, './checkpoint.pth')
            print("Start validation")
            accur = 0.0
            itert = 0
            for el_ind in ds.get_valid_batch():
                model.zero_grad()
                elem = ds[el_ind]
                itert += 1
                prediction = model.forward(elem['image'])
                loss_iter = loss(prediction, elem['target'][0])
                accur += 0.0 + float(prediction.argmax() == elem['target'].argmax())
                l = loss_iter.item()
                ep_loss += l
                ln = len(ds)
            print("Validation scores", ep_loss * 4, accur / itert)
            #optim = torch.optim.Adam(model.parameters(), lr=lr)
            ep_loss = 0
        if ep == 60:
            optim = torch.optim.SGD(model.parameters(), lr=lr/10.0)




model.forward(ds[3]['image']).argmax() == ds[3]['target'].argmax()




ds[2]['image']




10 / 12




model.forward(ds[4]['image'])




def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
#    for parameter in model.parameters():
 #       parameter.requires_grad = False
    
    model.eval()
    
    return model




#model = load_checkpoint('./checkpoint.pth')
#print(model)




train(model, 110, ds)




p1 = pd.read_csv("../input/test.csv")




p1.head()




test_dataset = []
for row in p1.iterrows():
    key = list(row[1])[0]#ну вот так только умею
    try:
        test_dataset = test_dataset + [key]
    except:
        pass




#test_dataset




#sorted(os.listdir('../input/test_images'))




#plt.imshow("../input/test_images/0005cfc8afb6.png")
            #../input/test_images/0005cfc8afb6.png




class BlindTestDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset 
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx, flip):
        image = self.dataset[idx]
        image = io.imread(os.path.join("../input/test_images", image+".png")).astype(float)
        image = transform.resize(image, (224, 224))
        if(flip > 4):
            image = image[::-1]
        flip %= 4 
        for i in range(flip):
            image = np.transpose(np.fliplr(image), (1,0,2))
      #  image = image - image.mean()
       # image = image / image.std()
        image = image.copy()
        sample = {'image': torch.FloatTensor(image.transpose(2,0,1)).cuda()}
        return sample




tds = BlindTestDataset(test_dataset)




tds.__getitem__(0,flip = 0)['image'].mean()




result = []




for i in range(len(tds)):
    r = torch.FloatTensor([0, 0, 0, 0, 0]).cuda()
    for fl in range(1, 9):
        el = tds.__getitem__(idx = i, flip=fl)
        r += model.forward(el['image'])[0]
    result = result + [int(r.argmax())]
    if(i % 100 == 0):
        print(i)




len(result)




len(tds)




df = pd.DataFrame(dict(id_code=test_dataset[:], diagnosis=result[:]))




df.head()




df.to_csv('submission.csv', index=False)




#os.listdir('.')




# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files &lt; 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe


# create a link to download the dataframe
create_download_link(df)

# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 














os.listdir('../working')




checkpoint = {'model': Classifier(),
              'state_dict': model.state_dict(),
              'optimizer' : optim.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')




#strr = ""
#for i in open('checkpoint.pth', 'rb'):
#    strr = i












































#with open(filepath, 'rb') as fp: 
#    file =  MIMEBase(maintype, subtype)
#    file.set_payload(fp.read())
#    fp.close() 
#encoders.encode_base64(file) 




#server = smtplib.SMTP('smtp.gmail.com', 587)
#server.starttls()
#server.login(addr_from, password)
#server.send_message(msg)
#server.quit() 









# ds[0]['target']

