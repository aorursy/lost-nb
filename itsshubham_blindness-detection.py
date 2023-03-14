#!/usr/bin/env python
# coding: utf-8



import os
import pandas as pd
from glob import glob

from utility import RetinaDataLoader, RetinaDataset, Transform
from network import ResnetModel




ls "../input/aptos2019-blindness-detection/"




# Define training labels adn training directory
tr_labels = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
tr_directory = "../input/aptos2019-blindness-detection/train_images/"

# Define test labels and test directory
te_labels = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
te_directory = "../input/aptos2019-blindness-detection/test_images/"




# Splitting training labels into training and validation sets
val_labels = tr_labels.sample(frac=0.15)
tr_labels = tr_labels.iloc[pd.Int64Index(set(tr_labels.index) - set(val_labels.index))]

val_labels.reset_index(inplace=True, drop=True)
tr_labels.reset_index(inplace=True, drop=True)




# Initialize dataset
transform = Transform()
tr_dataset = RetinaDataset(labels=tr_labels, directory=tr_directory, transform=transform.transform)
val_dataset = RetinaDataset(labels=val_labels, directory=tr_directory, transform=transform.transform)
te_dataset = RetinaDataset(labels=te_labels, directory=te_directory, test=True, transform=transform.transform)




# Initialize dataloader
dataloader = RetinaDataLoader(tr_ds=tr_dataset, val_ds=val_dataset, te_ds=te_dataset)




num_classes = tr_labels['diagnosis'].unique().shape[0]




resnet = ResnetModel()




_, input_size = resnet.initialize_model(num_classes=num_classes, feature_extraction=True)




optimizer, scheduler, loss_func = resnet.optimizer()




resnet.train(dataloaders=dataloader, optimizer=optimizer, loss_func=loss_func, scheduler=scheduler, device="cuda", num_epochs=3)




# Predictions on the test data
predictions = resnet.test(dataloader=dataloader, device="cuda")




predictions = [int(i) for i in predictions]
te_labels['diagnosis'] = predictions
te_labels.to_csv("submission.csv", index=False)











