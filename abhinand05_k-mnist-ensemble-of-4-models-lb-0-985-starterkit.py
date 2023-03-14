#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import seaborn as sns

from fastai.vision import *
from fastai.callbacks import SaveModelCallback

from radam_optimizer_pytorch import RAdam
from torch.nn import Conv2d
from torch.optim import Adam

import os
PATH = Path('../input/Kannada-MNIST/')
os.listdir(PATH)




# Setting Global Random Seed
def random_seed(seed_value, use_cuda):  
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: torch.cuda.manual_seed_all(seed_value) # gpu 

random_seed(42, True)




train_csv = pd.read_csv(PATH/'train.csv')
train_csv.head().T




def get_data_labels(csv,label):
    fileraw = pd.read_csv(csv)
    labels = fileraw[label].to_numpy()
    data = fileraw.drop([label],axis=1).to_numpy(dtype=np.float32).reshape((fileraw.shape[0],28,28))
    data = np.expand_dims(data, axis=1)
    return data, labels

train_data, train_labels = get_data_labels(PATH/'train.csv','label')
test_data, test_labels = get_data_labels(PATH/'test.csv','id')
other_data, other_labels = get_data_labels(PATH/'Dig-MNIST.csv','label')




print(f' Train:\tdata shape {train_data.shape}\tlabel shape {train_labels.shape}\n Test:\tdata shape {test_data.shape}\tlabel shape {test_labels.shape}\n Other:\tdata shape {other_data.shape}\tlabel shape {other_labels.shape}')




plt.title(f'Training Label: {train_labels[43]}')
plt.imshow(train_data[43,0],cmap='gray');




np.random.seed(42)
ran_20_pct_idx = (np.random.random_sample(train_labels.shape)) < .2

train_80_labels = train_labels[np.invert(ran_20_pct_idx)]
train_80_data = train_data[np.invert(ran_20_pct_idx)]

valid_20_labels = train_labels[ran_20_pct_idx]
valid_20_data = train_data[ran_20_pct_idx]




class ArrayDataset(Dataset):
    "Dataset for numpy arrays based on fastai example: "
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.c = len(np.unique(y))
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]




train_ds = ArrayDataset(train_80_data,train_80_labels)
valid_ds = ArrayDataset(valid_20_data,valid_20_labels)
other_ds = ArrayDataset(other_data, other_labels)
test_ds = ArrayDataset(test_data, test_labels)




bs = 64 # Batch Size
data = DataBunch.create(train_ds, valid_ds, test_ds=test_ds, bs=bs)




get_ipython().system('mkdir models')




MODEL_DIR = Path('../working/models/')




class ConvNet1(nn.Module):
    def __init__(self):
        super(ConvNet1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64)        
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        
        self.drop_out = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 10)
        self.bn1d = nn.BatchNorm1d(128)
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x):
        # conv layers
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.drop_out(self.layer3(out))
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.drop_out(self.layer6(out))
        out = out.view(out.shape[0], -1)
#         print(out.shape) # Life Saving Debuggung Step
        # FC Layer 1
        out = self.fc1(out)
        out = self.relu(out)
        out = self.bn1d(out)
        out = self.drop_out(out)
        # Output layer
        out = self.fc2(out)
        out = self.output(out)
        return out




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move Model to GPU
conv_net_1 = ConvNet1()
conv_net_1 = conv_net_1.to(device)




class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout(0.5)
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.drop_out(out)
        out = self.fc2(out)
        return out




# Move to GPU
conv_net_2 = ConvNet2()
conv_net_2 = conv_net_2.to(device)




# Helper Function for Model 3
def conv2(ni,nf,stride=2,ks=3): return conv_layer(ni,nf,stride=stride,ks=ks)




conv_net_3 = nn.Sequential(
    conv2(1,32,stride=1,ks=3),
    conv2(32,32,stride=1,ks=3),
    conv2(32,32,stride=2,ks=5),
    nn.Dropout(0.4),
    
    conv2(32,64,stride=1,ks=3),
    conv2(64,64,stride=1,ks=3),
    conv2(64,64,stride=2,ks=5),
    nn.Dropout(0.4),
    
    Flatten(),
    nn.Linear(3136, 128),
    relu(inplace=True),
    nn.BatchNorm1d(128),
    nn.Dropout(0.4),
    nn.Linear(128,10)
)




get_ipython().system('ls ../input/pytorch-pretrained-models')




rn18 = models.resnet18(pretrained=False)
rn18.load_state_dict(torch.load('../input/pytorch-pretrained-models/resnet18-5c106cde.pth'))
rn18.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)




learner1 = Learner(data, 
                  conv_net_1, 
                  metrics=accuracy, 
                  model_dir=MODEL_DIR,
                  opt_func=Adam,
                  loss_func=nn.CrossEntropyLoss()
                 )




learner1.lr_find()
learner1.recorder.plot(suggestion=True)




get_ipython().run_cell_magic('time', '', "learner1.fit_one_cycle(50, \n                      slice(1e-03),\n                      callbacks=[SaveModelCallback(learner1, \n                                                   every='improvement', \n                                                   monitor='accuracy', \n                                                   name='best_model_1')]\n                     ) ")




learner1.recorder.plot_losses(skip_start=800)




learner1.recorder.plot_metrics(skip_start=800)




learner1.load('best_model_1')
preds, ids = learner1.get_preds(DatasetType.Test)
y = torch.argmax(preds, dim=1)

submission_1 = pd.DataFrame({ 'id': ids,'label': y })
submission_1.to_csv("submission.csv", index=False)
submission_1.to_csv("submission_1.csv", index=False)




learner2 = Learner(data, 
                  conv_net_2, 
                  metrics=accuracy, 
                  model_dir=MODEL_DIR,
                  opt_func=RAdam,
                  loss_func=nn.CrossEntropyLoss()
                 )




learner2.lr_find()
learner2.recorder.plot(suggestion=True)




get_ipython().run_cell_magic('time', '', "learner2.fit_one_cycle(50, \n                      slice(1e-03),\n                      callbacks=[SaveModelCallback(learner2, \n                                                   every='improvement', \n                                                   monitor='accuracy', \n                                                   name='best_model_2')]\n                     ) ")




learner2.recorder.plot_losses(skip_start=800)




learner2.recorder.plot_metrics(skip_start=800)




learner2.load('best_model_2')
preds, ids = learner2.get_preds(DatasetType.Test)
y = torch.argmax(preds, dim=1)

submission_2 = pd.DataFrame({ 'id': ids,'label': y })
submission_2.to_csv("submission_2.csv", index=False)




learner3 = Learner(data, 
                  conv_net_3, 
                  metrics=accuracy, 
                  model_dir=MODEL_DIR,
                  opt_func=Adam,
                  loss_func=nn.CrossEntropyLoss()
                 )




learner3.lr_find()
learner3.recorder.plot(suggestion=True)




get_ipython().run_cell_magic('time', '', "learner3.fit_one_cycle(50, \n                      slice(8e-03),\n                      callbacks=[SaveModelCallback(learner3, \n                                                   every='improvement', \n                                                   monitor='accuracy', \n                                                   name='best_model_3')]\n                     ) ")




learner3.recorder.plot_losses(skip_start=800)




learner3.recorder.plot_metrics(skip_start=800)




learner3.load('best_model_3')
preds, ids = learner3.get_preds(DatasetType.Test)
y = torch.argmax(torch.exp(preds), dim=1)

submission_3 = pd.DataFrame({ 'id': ids,'label': y })
submission_3.to_csv("submission_3.csv", index=False)




learner4 = Learner(data, 
                  rn18, 
                  metrics=accuracy, 
                  model_dir=MODEL_DIR,
                  opt_func=Adam,
                  loss_func=nn.CrossEntropyLoss()
                 )




learner4.lr_find()
learner4.recorder.plot(suggestion=True)




get_ipython().run_cell_magic('time', '', "learner4.fit_one_cycle(50, \n                      slice(1e-03),\n                      callbacks=[SaveModelCallback(learner4, \n                                                   every='improvement', \n                                                   monitor='accuracy', \n                                                   name='best_model_4')]\n                     ) ")




learner4.recorder.plot_losses(skip_start=800)




learner4.recorder.plot_metrics(skip_start=800)




learner4.load('best_model_4')
preds, ids = learner4.get_preds(DatasetType.Test)
y = torch.argmax(torch.exp(preds), dim=1)

submission_4 = pd.DataFrame({ 'id': ids,'label': y })
submission_4.to_csv("submission_4.csv", index=False)




flatten = lambda l: [np.float32(item) for sublist in l for item in sublist]
metrics_list_1 = flatten(learner1.recorder.metrics)
metrics_list_2 = flatten(learner2.recorder.metrics)
metrics_list_3 = flatten(learner3.recorder.metrics)
metrics_list_4 = flatten(learner4.recorder.metrics)




losses_1 = pd.DataFrame({'loss':learner1.recorder.val_losses, 'accuracy': metrics_list_1})
losses_2 = pd.DataFrame({'loss':learner2.recorder.val_losses, 'accuracy': metrics_list_2})
losses_3 = pd.DataFrame({'loss':learner3.recorder.val_losses, 'accuracy': metrics_list_3})
losses_4 = pd.DataFrame({'loss':learner4.recorder.val_losses, 'accuracy': metrics_list_4})

fig, ax = plt.subplots(1,1,figsize=(14, 6))
ax.set(xlabel='Epochs Processed', ylabel='Loss', title='Comparing Validation Losses')
# losses_1['loss'].sort_index().plot(ax=ax)
losses_2['loss'].sort_index().plot(ax=ax)
losses_3['loss'].sort_index().plot(ax=ax)
losses_4['loss'].sort_index().plot(ax=ax)

ax.legend(['Model 2', 'Model 3', 'Model 4'])




fig, ax = plt.subplots(1,1,figsize=(14, 6))
ax.set(xlabel='Epochs Processed', ylabel='Loss', title='Validation Losses for Model 1')

losses_1['loss'].sort_index().plot(ax=ax)
ax.legend(['Model 1'])




fig, ax = plt.subplots(1,1,figsize=(14, 6))
ax.set(xlabel='Epochs Processed', ylabel='Loss', title='Comparing Validation Accuracy')
losses_1['accuracy'].sort_index().plot(ax=ax)
losses_2['accuracy'].sort_index().plot(ax=ax)
losses_3['accuracy'].sort_index().plot(ax=ax)
losses_4['accuracy'].sort_index().plot(ax=ax)

ax.legend(['Model 1', 'Model 2', 'Model 3', 'Model 4'])




preds_1 = (submission_1.label.value_counts()).rename('Model_1')
preds_2 = (submission_2.label.value_counts()).rename('Model_2')
preds_3 = (submission_3.label.value_counts()).rename('Model_3')
preds_4 = (submission_4.label.value_counts()).rename('Model_4')

preds_data = pd.concat([preds_1, preds_2, preds_3, preds_4], axis=1)
preds_data['category'] = preds_data.index
preds_data = pd.melt(preds_data, id_vars='category', var_name='model', value_name='preds')

fig = sns.catplot(x='category', y='preds', hue='model',data=preds_data, kind='bar', height=4, aspect=3)
fig.set(title='Distribution of predictions for each model per category')




blended_preds = np.round((submission_1['label'] + submission_2['label'] + 
                          submission_3['label'] + submission_4['label'])/4)

blended_submission = pd.DataFrame({'id': ids, 'label': blended_preds})
blended_submission['label'] = blended_submission['label'].astype(np.uint8)
blended_submission.to_csv("blended_submission.csv", index=False)

