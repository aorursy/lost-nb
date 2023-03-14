#!/usr/bin/env python
# coding: utf-8



import os
print(os.listdir("../input"))




class conf:
    #particular to this competition
    """
        False:all cell_type trainig
        True :each cell_type training
    """
    training_each_experiment = False
    """
        all cell_type training weight path
    """
    all_experiment_pretrain_path = '../input/all-experiment-densenet121-brightness-fold0/weight_best_all_3.pt'
    """
        please chage stages due to kernel 9 hours limit.
    """
    stage = 1
    n_splits = 5
    fold_number = 0
    if stage == 1:
        resume_training = False
    else:
        resume_training = True
    """
        continuing stage check point path
    """
    checkpoint_path = '../input/all-experiment-densenet121-brightness/'
    
    DEFAULT_CHANNELS = [1, 2, 3, 4, 5, 6]
    
    #common configs
    SEED = 717
    path_data = '../input/recursion-cellular-image-classification/'
    device = 'cuda'

    num_classes = 1108
    num_channels = 6
    input_size = 512    
    model_type = 'densenet121'
    use_pretrained = True # image net pretrain
    unflozen_epoch = 2

    num_epochs = 10 #about 8 hours
    batch_size = 16
    test_batch_size = 16
    gamma= 1
    lr = 1e-4 * (gamma ** (stage - 1))
    eta_min = 1e-5 * (gamma ** (stage - 1))
    t_max = 10
    cycle = t_max * 2 # for snapshot ensemble
    
    debug = False
    predict = False




import numpy as np
import pandas as pd 
import gc
import os
import sys
import pickle
import random
import time
import logging
from IPython.display import FileLink

from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from psutil import cpu_count
import datetime as dt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, MultiStepLR, ExponentialLR
from fastprogress import master_bar, progress_bar
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision.models as models
#from imgaug import augmenters as iaa

import matplotlib.pyplot as plt
from fastprogress import master_bar, progress_bar

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')




def get_logger(name="Main", tag="exp", log_dir="log/"):
    log_path = Path(log_dir)
    path = log_path / tag
    path.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(
        path / (dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log"))
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s")

    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = conf.SEED
seed_everything(SEED)




def save_checkpoint(conf, model, optimizer, scheduler, epoch, best_acc, best_epoch, cell_type):
    #checkpoint_path = 'checkpoint.pth.tar'
    checkpoint_path = 'checkpoint_{}.pth.tar'.format(cell_type)
    
    weights_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict(),
        'epoch' : epoch,
        'best_acc': best_acc,
        'best_epoch': best_epoch
    }
    torch.save(weights_dict, checkpoint_path)

def load_checkpoint(conf, model, optimizer, scheduler, cell_type):

    checkpoint = torch.load(conf.checkpoint_path + 'checkpoint_{}.pth.tar'.format(cell_type))
    print("=> loaded checkpoint '{}' (trained for {} epochs)".format(conf.checkpoint_path + 'checkpoint_{}.pth.tar'.format(cell_type), checkpoint['epoch']+1))
    end_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    best_epoch = checkpoint['best_epoch']
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return end_epoch, best_acc, best_epoch, model, optimizer, scheduler




def image_path(experiment,
               plate,
               well,
               site,
               channel,
               base_path=conf.path_data,
               mode='train'):
        
    return os.path.join(base_path, mode, experiment, "Plate{}".format(plate),
                        "{}_s{}_w{}.png".format(well, site, channel))

def image_paths(experiment,
                plate,
                well,
                site,
                channels=conf.DEFAULT_CHANNELS,
                base_path=conf.path_data,
                mode='train'):
    
    channel_paths = [
        image_path(
            experiment, plate, well, site, c, base_path=base_path, mode=mode)
        for c in channels
    ]
    
    return channel_paths

def load_image(file_name):
    img = Image.open(file_name)
    return img

def transform_image(transforms, img):
    img = transforms(img)
    return img




class ImagesDS(Dataset):
    def __init__(self, conf, df, transforms, mode='train'):
        
        #df = pd.read_csv(csv_file)
        self.records = df.to_records(index=False)
        self.conf = conf
        self.channels = self.conf.DEFAULT_CHANNELS
        self.path_data = self.conf.path_data
        self.mode = mode
        self.transforms = transforms
        self.len = df.shape[0]
        
    def __getitem__(self, index):
        paths = image_paths(self.records[index].experiment,self.records[index].plate,
                            self.records[index].well,self.records[index].site,
                            channels=self.channels,base_path=self.path_data, mode=self.mode)

        img = torch.cat([transform_image(self.transforms, load_image(img_path)) for img_path in paths])
        if self.mode == 'train':
            return img, self.records[index].sirna
        else:
            return img, self.records[index].id_code

    def __len__(self):
        return self.len




def initialize_model(conf):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if conf.model_type == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=conf.use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, conf.num_classes)
        input_size = conf.input_size

    elif conf.model_type == "resnet34":
        """ Resnet34
        """
        model_ft = models.resnet34(pretrained=conf.use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, conf.num_classes)
        input_size = conf.input_size
        
    elif conf.model_type == "densenet121":
        """ densenet121
        """
        model_ft = models.densenet121(pretrained=conf.use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, conf.num_classes)
        input_size = conf.input_size
        
    elif conf.model_type == "densenet201":
        """ densenet201
        """
        model_ft = models.densenet201(pretrained=conf.use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, conf.num_classes)
        input_size = conf.input_size
        
    elif conf.model_type == "resnext50_32x4d":
        """ resnext50_32x4d
        """
        model_ft = models.resnext50_32x4d(pretrained=conf.use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, conf.num_classes)
        input_size = conf.input_size
        
    elif conf.model_type == "wide_resnet50_2":
        """ wide_resnet50_2
        """
        model_ft = models.wide_resnet50_2(pretrained=conf.use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, conf.num_classes)
        input_size = conf.input_size

    elif conf.model_type == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=conf.use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,conf.num_classes)
        input_size = conf.input_size

    elif conf.model_type == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=conf.use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,conf.num_classes)
        input_size = conf.input_size

    elif conf.model_type == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=conf.use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = conf.num_classes
        input_size = conf.input_size

    elif conf.model_type == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=conf.use_pretrained)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, conf.num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,conf.num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size




# # To tune or design nn here.
model_ft, input_size = initialize_model(conf)

if (conf.model_type == "densenet121") | (conf.model_type == "densenet201"):
    trained_kernel = model_ft.features.conv0.weight
    new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)
    model_ft.features.conv0 = new_conv
else:    
    trained_kernel = model_ft.conv1.weight
    new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)
    model_ft.conv1 = new_conv

# Print the model we just instantiated
print(model_ft)




def train_single_epoch(conf, model, train_loader, criterion, optimizer, mb):
    avg_loss = 0.

    # train process
    for x_batch, y_batch in progress_bar(train_loader, parent=mb):

        if conf.model_type != 'inception':
            preds = model(x_batch.cuda())
            loss = criterion(preds, y_batch.cuda())
        else:
            outputs, aux_outputs = model(x_batch.cuda())
            loss1 = criterion(outputs, y_batch.cuda())
            loss2 = criterion(aux_outputs, y_batch.cuda())
            loss = loss1 + 0.4*loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item() / len(train_loader)
    
    return avg_loss

def evaluate_single_epoch(conf, model, valid_loader, criterion, optimizer):
    avg_val_loss = 0.              
    correct = 0
    total = 0
    for x_batch, y_batch in valid_loader:
        preds = model(x_batch.cuda()).detach()
        loss_ = criterion(preds, y_batch.cuda())

        preds = torch.sigmoid(preds)
        preds = preds.max(dim=-1)[1].cpu().numpy()
        avg_val_loss += loss_.item() / len(valid_loader)                
        #_, predicted = torch.max(outputs.data, 1)

        #total += len(valid_loader)
        correct += (preds == y_batch.cpu().numpy()).sum().item()
    acc = correct / len(valid_loader.dataset)
        
    return avg_val_loss, acc

def set_requires_grad(conf, model, epoch):
    if (conf.stage == 1) & (conf.training_each_experiment == False):
        #fine tuning
        if epoch + 1 == 1:
            for name, child in model.named_children():
                #print(name, child)
                if (name == 'fc') | (name == 'classifier'):
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    for param in child.parameters():
                        param.requires_grad = False

        #unflozen all layer for epoch 2
        if epoch + 1 == 2:
            for name, child in model.named_children():
                for param in child.parameters():
                    param.requires_grad = True
    return model




def train_model(conf, df, train_transforms, valid_transforms, cell_type='all'):
    #logger
    #logger = get_logger("Main", tag="train", log_dir="log/")
    print('fold: {}'.format(conf.fold_number))
    print('Training {}'.format(cell_type))
    print('Stage: {}'.format(conf.stage))
    logger.info('fold {}'.format(conf.fold_number))
    logger.info('Training {}'.format(cell_type))
    logger.info('Stage: {}'.format(conf.stage))
    
    #initialize
    trn_loss = []
    val_loss = []
    val_acc = []
    lr_log = []
    bests = []
    loss_list = []
    acc_list = []
    
    #train_test_split
    idx = np.arange(len(df))
    #trn_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=SEED)
    folds = KFold(n_splits=conf.n_splits, shuffle=True, random_state=SEED)
    for fold, (trn_idx_, test_idx_) in enumerate(folds.split(idx)):
        if fold == conf.fold_number:
            trn_idx = trn_idx_
            val_idx = test_idx_
    
    df_trn = df.iloc[trn_idx, :]
    df_val = df.iloc[val_idx, :]
    #extract cell_type
    if conf.training_each_experiment:
        df_trn = df_trn[df_trn.cell_type == cell_type]
        df_val = df_val[df_val.cell_type == cell_type]

    #Dataset
    """
    sample(frac=1, random_state=conf.stage)
    This code means shuffling train data in each stage. Train images are viewed in the same order wihout shuffling, because seed is static.
    """
    train_dataset = ImagesDS(conf, df_trn.sample(frac=1, random_state=conf.stage), train_transforms)
    valid_dataset = ImagesDS(conf, df_val, valid_transforms)

    #DataLoader
    train_loader = DataLoader(train_dataset,batch_size=conf.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=4)
    
    #model
    model = model_ft.cuda()
    
    #loss function
    criterion = nn.CrossEntropyLoss().cuda()
    #optimizer
    optimizer = Adam(params=model.parameters(), lr=conf.lr, amsgrad=False)
    #learning rate
    scheduler = CosineAnnealingLR(optimizer, T_max=conf.t_max, eta_min=conf.eta_min)
    #initialize
    best_epoch = -1
    best_acc = 0.
    start_epoch = 0
    cycle_count = (conf.stage - 1) * conf.num_epochs // conf.cycle
    torch.cuda.empty_cache()
    #resume training
    if conf.resume_training:
        end_epoch, best_acc, best_epoch, model, optimizer, scheduler = load_checkpoint(conf, model, optimizer, scheduler, cell_type)
        start_epoch = end_epoch + 1
        
    #all_experiment_pretrain
    if (conf.training_each_experiment == True) & (conf.stage == 1):
        model.load_state_dict(torch.load(conf.all_experiment_pretrain_path))
        
    mb = master_bar(range(start_epoch, start_epoch + conf.num_epochs))
    torch.cuda.empty_cache()
    
    #---
    accumurated_time = 0
    max_elapsed_time = 0
    for epoch in mb:
        start_time = time.time()
        if epoch + 1 <= conf.unflozen_epoch:
            model = set_requires_grad(conf, model, epoch)
                        
        #cycleごとにlearning rateを減衰
        if (epoch != 0) & (epoch % conf.cycle == 0):
            conf.lr = conf.lr * conf.gamma
            conf.eta_min = conf.eta_min * conf.gamma
            optimizer = Adam(params=model.parameters(), lr=conf.lr, amsgrad=False)
            scheduler = CosineAnnealingLR(optimizer, T_max=conf.t_max, eta_min=conf.eta_min)

        #train process
        model.train()
        avg_loss = train_single_epoch(conf, model, train_loader, criterion, optimizer, mb)

        # validation process
        model.eval()
        avg_val_loss, acc = evaluate_single_epoch(conf, model, valid_loader, criterion, optimizer)
    
    
        # record the metrics
        for param_group in optimizer.param_groups:
            lr_temp = param_group['lr']
        lr_log.append(lr_temp)
        #print("Learning rate: {}".format(lr_temp))
        
        # scheduler step
        scheduler.step()
        #scheduler.step(avg_val_loss) # for reduceLR 
        
        # cycle for snapshot ensemple
        if (epoch != 0) & (epoch % conf.cycle == 0):
            cycle_count += 1
            #reset acc
            best_acc = 0.
        if epoch % conf.cycle == 0:
            print('Cycle {}'.format(cycle_count))           
        
        # log
        if (epoch + 1) % 1 == 0:
            elapsed = time.time() - start_time
            print('Epoch {} -> Train Loss: {:.4f} Valid Loss: {:.4f}, ACC: {:.4f}'.format(epoch + 1, avg_loss, avg_val_loss, acc))
            logger.info('Epoch {} -> LR: {:.6f} Train Loss: {:.4f} Valid Loss: {:.4f}, ACC: {:.4f}, time: {:.0f}s'.format(epoch + 1, lr_temp, avg_loss, avg_val_loss, acc, elapsed))
            trn_loss.append(avg_loss)
            val_loss.append(avg_val_loss)
            val_acc.append(acc)
        
        # save best weight
        if acc > best_acc:
            best_epoch = epoch + 1
            best_acc = acc
            torch.save(model.state_dict(), 'weight_best_{}_{}.pt'.format(cell_type, cycle_count))
        #save checkpoint
        model.train()
        save_checkpoint(conf, model, optimizer, scheduler, epoch, best_acc, best_epoch, cell_type)
        loss_list.append([avg_loss, avg_val_loss])
        acc_list.append(acc)
        
        #--
        """
        When the kernel is slow, your kernle will time out.
        This code will prevent time out when the kernel is slow.
        """
        #accumurated_time += elapsed
        #if elapsed > max_elapsed_time:
        #    max_elapsed_time = elapsed
        #if accumurated_time >= 9 * 60 * 60 - max_elapsed_time:
        #    break

    bests.append([best_epoch, best_acc])
    logger.info(f"Best: {bests}")
    return bests, loss_list, acc_list, lr_log




class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.CoarseDropout(0.1,size_percent=0.02)
        ])
        
    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

transforms_dict = {
        'train': transforms.Compose([
            #transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            #transforms.ColorJitter(brightness=0.5), #caution: work well in all cell_type trainig. But,It is overfitting in each cell_type training.
            #transforms.RandomRotation((-90,90)),
            #ImgAugTransform(),
            transforms.ToTensor()
        ]),
        'valid': transforms.Compose([
            #transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            #ImgAugTransform(),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            #transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            #ImgAugTransform(),
            transforms.ToTensor()
        ])
    }




train_df = pd.read_csv(conf.path_data+'/train.csv')
if conf.debug:
    train_df = train_df[0:300]
train_df['cell_type'] = train_df.experiment.str.split("-").apply(lambda a: a[0])
train_df.cell_type.unique()




train_df['site'] = 1 
train_df_2 = train_df.copy()
train_df_2['site'] = 2
train_df = pd.concat([train_df, train_df_2]).sort_index().reset_index(drop=True)
del train_df_2
gc.collect()




#logger
logger = get_logger("Main", tag="train", log_dir="log/")




#run
if conf.training_each_experiment:
    result_HEPG2, loss_list_HEPG2, acc_list_HEPG2, lr_log_HEPG2 = train_model(conf, train_df, transforms_dict['train'], transforms_dict['valid'], 'HEPG2')
    result_HUVEC, loss_list_HUVEC, acc_list_HUVEC, lr_log_HUVEC = train_model(conf, train_df, transforms_dict['train'], transforms_dict['valid'], 'HUVEC')
    result_RPE, loss_list_RPE, acc_list_RPE, lr_log_RPE = train_model(conf, train_df, transforms_dict['train'], transforms_dict['valid'], 'RPE')
    result_U2OS, loss_list_U2OS, acc_list_U2OS, lr_log_U2OS = train_model(conf, train_df, transforms_dict['train'], transforms_dict['valid'], 'U2OS')
else:
    result_all, loss_list_all, acc_list_all, lr_log_all = train_model(conf, train_df, transforms_dict['train'], transforms_dict['valid'])




"""
import requests

def send_line_notification(message):
    line_token = 'token'  # set your token
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)

result = 'training finished'

send_line_notification(result)
"""




if conf.training_each_experiment:
    print(train_df.cell_type.value_counts().HEPG2/len(train_df) * max(acc_list_HEPG2) +           train_df.cell_type.value_counts().HUVEC/len(train_df) * max(acc_list_HUVEC) +           train_df.cell_type.value_counts().RPE/len(train_df) * max(acc_list_RPE) +           train_df.cell_type.value_counts().U2OS/len(train_df) * max(acc_list_U2OS))




#plot
if conf.training_each_experiment:
    loss_list_HEPG2 = pd.DataFrame(loss_list_HEPG2)
    loss_list_HEPG2['cell_type'] = 'HEPG2'
    loss_list_HUVEC = pd.DataFrame(loss_list_HUVEC)
    loss_list_HUVEC['cell_type'] = 'HUVEC'
    loss_list_RPE = pd.DataFrame(loss_list_RPE)
    loss_list_RPE['cell_type'] = 'RPE'
    loss_list_U2OS = pd.DataFrame(loss_list_U2OS)
    loss_list_U2OS['cell_type'] = 'U2OS'
    acc_list_HEPG2 = pd.DataFrame(acc_list_HEPG2)
    acc_list_HEPG2['cell_type'] = 'HEPG2'
    acc_list_HUVEC = pd.DataFrame(acc_list_HUVEC)
    acc_list_HUVEC['cell_type'] = 'HUVEC'
    acc_list_RPE = pd.DataFrame(acc_list_RPE)
    acc_list_RPE['cell_type'] = 'RPE'
    acc_list_U2OS = pd.DataFrame(acc_list_U2OS)
    acc_list_U2OS['cell_type'] = 'U2OS'
    lr_log_HEPG2 = pd.DataFrame(lr_log_HEPG2)
    lr_log_HEPG2['cell_type'] = 'HEPG2'
    lr_log_HUVEC = pd.DataFrame(lr_log_HUVEC)
    lr_log_HUVEC['cell_type'] = 'HUVEC'
    lr_log_RPE = pd.DataFrame(lr_log_RPE)
    lr_log_RPE['cell_type'] = 'RPE'
    lr_log_U2OS = pd.DataFrame(lr_log_U2OS)
    lr_log_U2OS['cell_type'] = 'U2OS'
    
    loss_list = pd.concat([loss_list_HEPG2, loss_list_HUVEC, loss_list_RPE, loss_list_U2OS])
    acc_list = pd.concat([acc_list_HEPG2, acc_list_HUVEC, acc_list_RPE, acc_list_U2OS])
    lr_log = pd.concat([lr_log_HEPG2, lr_log_HUVEC, lr_log_RPE, lr_log_U2OS])
    
    for i, cell_type in enumerate(['HEPG2', 'HUVEC', 'RPE', 'U2OS']):
        loss = loss_list[loss_list.cell_type == cell_type][0]
        val_loss = loss_list[loss_list.cell_type == cell_type][1]
        acc = acc_list[acc_list.cell_type == cell_type][0]
        lr = lr_log[lr_log.cell_type == cell_type][0]

        epochs = range((conf.stage-1) * conf.num_epochs + 1, (conf.stage-1) * conf.num_epochs + len(loss) + 1)

        #lossとaccをプロット
        fig, ax1 = plt.subplots()
        ax1.plot(epochs, loss, color = 'royalblue', label = "Training loss")
        ax1.plot(epochs, val_loss, color='r', label = "Validation loss")
        ax1.set_ylim([0, 7])
        ax2 = ax1.twinx()
        ax2.plot(epochs, acc, 'bo',color='r', label = "ACC")
        ax2.set_ylim([0, 1])
        ax3 = ax2.twinx()
        ax3.plot(epochs, lr, color='c', label = "LR")
        ax3.set_ylim([0, 0.0001])
        plt.title('Loss and ACC and Learning rate {}'.format(cell_type))
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        h3, l3 = ax3.get_legend_handles_labels()
        ax1.legend(h1+h2+h3, l1+l2+l3, loc='upper right')
        plt.tight_layout()
        plt.savefig('figure_{}.png'.format(cell_type))
        plt.show()
else:
    loss_list_all = pd.DataFrame(loss_list_all)
    acc_list_all = pd.DataFrame(acc_list_all)
    lr_log_all = pd.DataFrame(lr_log_all)

    loss = loss_list_all[0]
    val_loss = loss_list_all[1]
    acc = acc_list_all
    lr = lr_log_all

    epochs = range((conf.stage-1) * conf.num_epochs + 1, (conf.stage-1) * conf.num_epochs + len(loss) + 1)

    #plot loss and acc
    fig, ax1 = plt.subplots()
    ax1.plot(epochs, loss, color = 'royalblue', label = "Training loss")
    ax1.plot(epochs, val_loss, color='r', label = "Validation loss")
    ax1.set_ylim([0, 10])
    ax2 = ax1.twinx()
    ax2.plot(epochs, acc, 'bo',color='r', label = "ACC")
    ax2.set_ylim([0, 1])
    ax3 = ax2.twinx()
    ax3.plot(epochs, lr, color='c', label = "LR")
    ax3.set_ylim([0, 0.0001])
    plt.title('Loss and ACC and Learning rate')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax1.legend(h1+h2+h3, l1+l2+l3, loc='upper right')
    plt.tight_layout()
    plt.savefig('figure_all.png')
    plt.show()




if conf.predict:
    train_df = pd.read_csv(conf.path_data+'/train.csv')
    test_df = pd.read_csv(conf.path_data + '/test.csv')
    if conf.debug:
        test_df = test_df[0:100]
    test_df['cell_type'] = test_df.experiment.str.split("-").apply(lambda a: a[0])
    test_df.cell_type.unique()




def predict_model(conf, df, model, test_transforms, weight_cycle, cell_type='all'):
    
    #extract cell_type
    df = df[df.cell_type == cell_type]
    print('Predict {}'.format(cell_type))
    
    preds_1_all = []
    preds_2_all = []
    
    for experiment in df.experiment.unique():
        print('experiment {}'.format(experiment))
        
        df_ex = df[df.experiment == experiment]
        
        test_dataset_site1 = ImagesDS(conf, df_ex, test_transforms, mode='test', site=1)
        test_loader_site1 = DataLoader(test_dataset_site1,batch_size=conf.test_batch_size, shuffle=False, num_workers=4)
        test_dataset_site2 = ImagesDS(conf, df_ex, test_transforms, mode='test', site=2)
        test_loader_site2 = DataLoader(test_dataset_site2,batch_size=conf.test_batch_size, shuffle=False, num_workers=4)

        model.load_state_dict(torch.load(conf.checkpoint_path + 'weight_best_{}_{}.pt'.format(cell_type, weight_cycle)))
        model.cuda()
        #model.eval()
        model.train()  #caution: usually use model.eval()
        
        
        #preds_1_all = []
        #preds_2_all = []
        #preds_all = np.empty(0)

        pb = progress_bar(test_loader_site1)
        for images, id_code in pb:
            with torch.no_grad():
                preds_1 = torch.sigmoid(model(images.cuda()).detach())
                preds_1_all.append(preds_1)
        
        pb = progress_bar(test_loader_site2)
        for images, id_code in pb:
            with torch.no_grad():
                preds_2 = torch.sigmoid(model(images.cuda()).detach())
                preds_2_all.append(preds_2)
        
    preds_1_all = torch.cat(preds_1_all)     
    preds_2_all = torch.cat(preds_2_all)
    
    preds = (preds_1_all + preds_2_all) / 2

    return preds




if conf.predict:
    preds_HEPG2_1 = predict_model(conf, test_df, model_ft, transforms_dict['test'], 5, 'HEPG2')
    preds_HUVEC_1 = predict_model(conf, test_df, model_ft, transforms_dict['test'], 5,'HUVEC')
    preds_RPE_1 = predict_model(conf, test_df, model_ft, transforms_dict['test'], 5,'RPE')
    preds_U2OS_1 = predict_model(conf, test_df, model_ft, transforms_dict['test'], 5,'U2OS')




# apply plate leak
"""
    https://www.kaggle.com/zaharch/keras-model-boosted-with-plates-leak
"""
plate_groups = np.zeros((1108,4), int)
for sirna in range(1108):
    grp = train_df.loc[train_df.sirna==sirna,:].plate.value_counts().index.values
    assert len(grp) == 3
    plate_groups[sirna,0:3] = grp
    plate_groups[sirna,3] = 10 - grp.sum()    

def post_processing(preds_HEPG2, preds_HUVEC, preds_RPE, preds_U2OS, name):
    preds_HEPG2_sirna = preds_HEPG2.max(dim=-1)[1].cpu().numpy()
    preds_HUVEC_sirna = preds_HUVEC.max(dim=-1)[1].cpu().numpy()
    preds_RPE_sirna = preds_RPE.max(dim=-1)[1].cpu().numpy()
    preds_U2OS_sirna = preds_U2OS.max(dim=-1)[1].cpu().numpy()
    predicted = np.concatenate([preds_HEPG2.cpu().numpy(), preds_HUVEC.cpu().numpy(), preds_RPE.cpu().numpy(), preds_U2OS.cpu().numpy()]).squeeze()
    preds = np.concatenate([preds_HEPG2_sirna, preds_HUVEC_sirna, preds_RPE_sirna, preds_U2OS_sirna])
    sub = pd.read_csv(conf.path_data + '/test.csv')
    sub['sirna'] = preds.astype(int)
    
    all_test_exp = test_df.experiment.unique()

    group_plate_probs = np.zeros((len(all_test_exp),4))
    for idx in range(len(all_test_exp)):
        preds_sirna = sub.loc[test_df.experiment == all_test_exp[idx],'sirna'].values
        pp_mult = np.zeros((len(preds_sirna),1108))
        pp_mult[range(len(preds_sirna)),preds_sirna] = 1

        sub_test = test_df.loc[test_df.experiment == all_test_exp[idx],:]
        assert len(pp_mult) == len(sub_test)

        for j in range(4):
            mask = np.repeat(plate_groups[np.newaxis, :, j], len(pp_mult), axis=0) ==                    np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)

            group_plate_probs[idx,j] = np.array(pp_mult)[mask].sum()/len(pp_mult)
            
    exp_to_group = group_plate_probs.argmax(1)
    
    for idx in range(len(all_test_exp)):
        #print('Experiment', idx)
        indices = (test_df.experiment == all_test_exp[idx])

        preds = predicted[indices,:].copy()

        preds = select_plate_group(preds, idx, exp_to_group)
        #preds = preds.argmax(1)
        sub.loc[indices,'sirna'] = preds.argmax(1)
        
        sub.to_csv('submission_{}.csv'.format(name), index=False, columns=['id_code','sirna'])
    
    return sub['sirna']

def select_plate_group(pp_mult, idx, exp_to_group):
    all_test_exp = test_df.experiment.unique()
    sub_test = test_df.loc[test_df.experiment == all_test_exp[idx],:]
    assert len(pp_mult) == len(sub_test)
    mask = np.repeat(plate_groups[np.newaxis, :, exp_to_group[idx]], len(pp_mult), axis=0) !=            np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
    pp_mult[mask] = 0
    return pp_mult




#densenet121
if conf.predict:
    sub_densenet121_1 = post_processing(preds_HEPG2_1, preds_HUVEC_1, preds_RPE_1, preds_U2OS_1, 'densenet121_1')




if conf.predict:
    sub = pd.read_csv(conf.path_data + '/test.csv')
    sub['sirna'] = sub_densenet121_1.astype(int)
    sub.head()




if conf.predict:
    sub[['id_code','sirna']].to_csv('submission.csv', index=False)




"""
import requests

def send_line_notification(message):
    line_token = 'token'  # set your token
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)

if conf.predict:
    result = 'predict finished'

    send_line_notification(result)
"""

