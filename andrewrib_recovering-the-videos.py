#!/usr/bin/env python
# coding: utf-8



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread 
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import os




os.listdir("../input/train_color")[0:10]




training_img_names = os.listdir("../input/train_color") 
trainingImgNameDF = pd.DataFrame( list(map(lambda s : [s]+s.split("_") , training_img_names )) ).drop(3, axis=1)
trainingImgNameDF.head()




trainingImgNameDF[1].unique().shape




trainingImgNameDF[4].unique().shape




def visVid(trainingImgNameDF,sessionID,cameraID,startFrame=0,endFrame=10,dataRoot="../input/train_color/"):
    res = trainingImgNameDF[ (trainingImgNameDF[1] == sessionID) & (trainingImgNameDF[4] == cameraID)].sort_values(by=2)
    imgURIs = list(res[0])
    
    fig = plt.figure(figsize=(10,10))
    frames = []
    for uri in imgURIs[startFrame:endFrame]:
        im = plt.imshow( imread(dataRoot+uri), animated=True)
        frames.append([im])
    
    plt.close(fig)
    return animation.ArtistAnimation(fig,frames,interval=50, blit=True,repeat_delay=1000)
    




uniqueVideoID = trainingImgNameDF[1].unique()
ani = visVid(trainingImgNameDF,uniqueVideoID[0],"5.jpg")
HTML(ani.to_jshtml(default_mode="reflect"))




uniqueVideoID = trainingImgNameDF[1].unique()
ani = visVid(trainingImgNameDF,uniqueVideoID[0],"6.jpg")
HTML(ani.to_jshtml(default_mode="reflect"))

