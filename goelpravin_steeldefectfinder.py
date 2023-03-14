#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




get_ipython().system('cp -r /kaggle/input/imageai/imageai/imageai/ imageai')



# library imports
import numpy as np 
import pandas as pd
import random as rn
import cv2 as cv 
import os
import sys
from pathlib import Path
from datetime import datetime
import time
# tensorflow for neural networks
import tensorflow as tf
from imageai.Detection import ObjectDetection

# visuals
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from IPython.display import Image

# for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)
# paths
img_train_folder = Path('../input/severstal-steel-defect-detection/train_images/')
img_test_folder = Path('../input/severstal-steel-defect-detection/test_images/')
numberOfSampleExtractionsShown = 0




# reading in the training set
data = pd.read_csv('/kaggle/input/severstal-steel-defect-detection/train.csv')
# add new columns ImageId, ClassId and EncodingExists to the data file 
# equivalent of doing the following in excel: =LEFT(A2,FIND("_",A2)-1) and =RIGHT(A2,LEN(A2)-FIND("_",A2))
# and EncodingExists would be roughly about =LEN(B2) since for pandas pivot table agg will have
# zero or 1 instead of false or true
data['ImageId'], data['ClassId'] = data.ImageId_ClassId.str.split('_', n=1).str
# change the type of classid to be an integer
data['ClassId'] = data['ClassId'].astype(np.uint8)
data['EncodingExists'] = data.EncodedPixels.str.len()

# find out which images have no defects
# create a pivot table with ImageId as row and EncodingExists as value to aggregate upon
imageDefectPivot = pd.pivot_table(data,index=['ImageId'],values=['EncodingExists'])
# Those images with all NA values in the defect pivot will get dropped since dropna for pandas pivot is true by default
# so now you can classify images as withDefect and withoutDefect
imageIdsWithDefect = imageDefectPivot.index.values
no_defect_data = data.loc[~data['ImageId'].isin(imageIdsWithDefect)]
imageIdsWithoutDefect = no_defect_data.ImageId.values
defect_data = data.loc[data['ImageId'].isin(imageIdsWithDefect)]
#defect_data = defect_data.dropna(subset=['EncodedPixels'], axis='rows', inplace=True)

# storing a list of images without defects for later use and testing
no_defects = data[data['EncodedPixels'].isna()][['ImageId']].drop_duplicates()
#print ("No defects", no_defects)
# adding the columns so we can append (a sample of) the dataset if need be, later
no_defects['EncodedPixels'] = ''
no_defects['ClassId'] = np.empty((len(no_defects), 0)).tolist()
no_defects['Distinct Defect Types'] = 0
no_defects.reset_index(inplace=True)

# keep only the images with labels
squashed = data.dropna(subset=['EncodedPixels'], axis='rows', inplace=True)

# squash multiple rows per image into a list
squashed = data[['ImageId', 'EncodedPixels', 'ClassId']]             .groupby('ImageId', as_index=False)             .agg(list) 
# count the amount of class labels per image
squashed['Distinct Defect Types'] = squashed.ClassId.apply(lambda x: len(x))

# display first ten to show new structure
#squashed.head(10)




def build_mask(encodings, labels):
    """ takes a pair of lists of encodings and labels, 
        and turns them into a 3d numpy array of shape (256, 1600, 4) 
    """
    
    # initialise an empty numpy array 
    mask = np.zeros((256,1600,4), dtype=np.uint8)
    
    # building the masks
    for rle, label in zip(encodings, labels):
        
        # classes are [1, 2, 3, 4], corresponding indeces are [0, 1, 2, 3]
        index = label - 1
        
        # fit the mask into the correct layer
        # note we need to transpose the matrix to account for 
        # numpy and openCV handling width and height in reverse order 
        mask[:,:,index] = rle_to_mask(rle).T
        
    
    return mask

def build_mask_for_class(encodings, forclass):
    """ takes a list of encodings  
        and turns them into a 2d numpy array of shape (256, 1600) only for the
        specified class
    """
    
    # initialise an empty numpy array 
    mask = np.zeros((256,1600), dtype=np.uint8)
    mask = rle_to_mask(encodings).T
    
    return mask


def mask_to_contours(image, mask_layer, color):
    """ converts a mask to contours using OpenCV and draws it on the image
    """

    # https://docs.opencv.org/4.1.0/d4/d73/tutorial_py_contours_begin.html
    contours, hierarchy = cv.findContours(mask_layer, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    image = cv.drawContours(image, contours, -1, color, 2)
        
    return image

def mask_to_contours_extraction(image, mask_layer, color,locn,rowIndex):
    """ converts a mask to contours using OpenCV and extract them from the image
    """
    if image is None:
        return
    if not os.path.exists(locn):
        os.makedirs(locn)
        #print("created", locn)
    os.chdir(locn)
    # https://docs.opencv.org/4.1.0/d4/d73/tutorial_py_contours_begin.html
    contours, hierarchy = cv.findContours(mask_layer, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    #image = cv.drawContours(image, contours, -1, color, 2)
    for contour in contours:
        x,y,width,height = cv.boundingRect(contour)
        defect_image = image[y:y+height,x:x+width]
        milliseconds = int(round(time.time()*1000))
        tmpImageName = "defect" + str(milliseconds)+".png"
        #print("imagename is", tmpImageName)
        
        #image_name = datetime.now()
        #cv.imwrite(Path.joinpath(locn, str(milliseconds)),defect_image)
        #Commenting out Image writing just for saving i/o while committing the kernel, the following two lines should be uncommented
        if not defect_image is None:
            if (len(defect_image)>1):
                #cv.imwrite(tmpImageName,defect_image)
                if (rowIndex<15):
                    #show a few extractions to get a visual feel
                    plt.imshow(defect_image, cmap = 'gray', interpolation = 'bicubic')
                    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
                    plt.show()
    return 

def extract_defect_contours(file_name, mask,folder_locn,rowIndex):
    """ open an image and extract segments identified by the mask/contours and store them in folder_locn 
    """
    
    # reading in the image
    #image = cv.imread(f'{img_train_folder}/{file_name}')
    image = cv.imread(os.path.join('/kaggle/input/severstal-steel-defect-detection/train_images/',file_name))
    mask_to_contours_extraction(image, mask, color=palette[1],locn=folder_locn,rowIndex=rowIndex)   
        
    return 

    
def visualise_mask(file_name, mask):
    """ open an image and draws clear masks, so we don't lose sight of the 
        interesting features hiding underneath 
    """
    
    # reading in the image
    image = cv.imread(os.path.join('/kaggle/input/severstal-steel-defect-detection/train_images/',file_name))

    # going through the 4 layers in the last dimension 
    # of our mask with shape (256, 1600, 4)
    for index in range(mask.shape[-1]):
        
        # indeces are [0, 1, 2, 3], corresponding classes are [1, 2, 3, 4]
        label = index + 1
        
        # add the contours, layer per layer 
        image = mask_to_contours(image, mask[:,:,index], color=palette[label])   
        
    return image

def rle_to_mask(lre, shape=(1600,256)):
    '''
    params:  rle   - run-length encoding string (pairs of start & length of encoding)
             shape - (width,height) of numpy array to return 
    
    returns: numpy array with dimensions of shape parameter
    '''    
    # the incoming string is space-delimited
    runs = np.asarray([int(run) for run in lre.split(' ')])
    
    # we do the same operation with the even and uneven elements, but this time with addition
    runs[1::2] += runs[0::2]
    # pixel numbers start at 1, indexes start at 0
    runs -= 1
    
    # extract the starting and ending indeces at even and uneven intervals, respectively
    run_starts, run_ends = runs[0::2], runs[1::2]
    
    # build the mask
    h, w = shape
    mask = np.zeros(h*w, dtype=np.uint8)
    for start, end in zip(run_starts, run_ends):
        mask[start:end] = 1
    
    # transform the numpy array from flat to the original image shape
    return mask.reshape(shape)

def rle_decode(mask_rle, shape=(1600,256)):
    #print('rle_decode(mask_rle = ', mask_rle)
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        #print("hi:",hi,"low:",lo)
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction




""" use a consistent color palette per label throughout the notebook """
import colorlover as cl

# see: https://plot.ly/ipython-notebooks/color-scales/
colors = cl.scales['4']['qual']['Set3']
labels = np.array(range(1,5))

# combining into a dictionary
palette = dict(zip(labels, np.array(cl.to_numeric(colors))))
# squash multiple rows per image into a list after removing rows without encoding
defect_data.dropna(subset=['EncodedPixels'], axis='rows', inplace=True)
squashed_defects = defect_data[['ImageId', 'EncodedPixels', 'ClassId']]             .groupby('ImageId', as_index=False)             .agg(list) 
sample_size_plot=10
sample = squashed_defects.sample(sample_size_plot)
# make a subplot+
fig, axes = plt.subplots(sample_size_plot, 1, figsize=(16, sample_size_plot*3))
fig.tight_layout()
    
# looping over sample
for i, (index, row) in enumerate(sample.iterrows()):
    # current ax
    ax = axes[i,]
    # build the mask 
    mask = build_mask(encodings=row.EncodedPixels, labels=row.ClassId)
    # fetch the image and draw the contours
    image = visualise_mask(file_name=row.ImageId, mask=mask)
    # display
    ax.set_title(f'{row.ImageId}: {row.ClassId}')
    ax.axis('off')
    ax.imshow(image);




#now let us train custom ImageAI model on training images provided. ImageAI custom training says "Your 
#image dataset must contain at least 2 different classes/types of images (e.g cat and dog) and you must 
#collect at least 500 images for each of the classes to achieve maximum accuracy". 
#The dataset provided to train
#has 897 images for defect1, 113/1483/49 for defect classes 2/3/4 - so perhaps best to train for defects1 
#and 3 only for now
#before training the model, need to create the following folder structure: 
#steelDefects/train/defect1/defect1-train-images 
#and steelDefects/train/defect3/defect3-train-images; also equivalent test folders
#instead of having to save defect1 and 3 images in this folder structure, is there an alternative way of 
#giving image path and encoding pixels; that would avoid lots of i/o. Seems like no according to the
#documentation at github: 
#https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Prediction/CUSTOMTRAINING.md
#also, as of 31Aug2019 doc at https://buildmedia.readthedocs.org/media/pdf/imageai/latest/imageai.pdf
defect1_train_folder = Path('/kaggle/working/steelDefects/train/defect1/defect1-train-images/')
defect3_train_folder = Path('/kaggle/working/steelDefects/train/defect3/defect3-train-images/')
# to ensure fresh extraction of training images and avoid any noise from previously extracted images, empty the folders
if not os.path.exists(defect1_train_folder):
    os.makedirs(defect1_train_folder)
if not os.path.exists(defect3_train_folder):
    os.makedirs(defect3_train_folder)

for defect_file_name in os.listdir(defect1_train_folder):
    defectfilepath = os.path.join(defect1_train_folder,defect_file_name)
    if (os.path.isfile(defectfilepath)):
        os.remove(defectfilepath)
for defect_file_name in os.listdir(defect3_train_folder):
    defectfilepath = os.path.join(defect3_train_folder,defect_file_name)
    if (os.path.isfile(defectfilepath)):
        os.remove(defectfilepath)

#now store just the part that has the defect1 in defect1 train folder 
#defect_data where 'ClassId' is 1
defect1_data = defect_data[defect_data['ClassId']==1]
defect3_data = defect_data[defect_data['ClassId']==3]
numberOfSampleExtractionsShown = 0
# looping over defect1_data, every time storing the contour image in train folder
for i, (index, row) in enumerate(defect1_data.iterrows()):
    image = extract_defect_contours(file_name=row.ImageId, mask=rle_decode(row.EncodedPixels).T,folder_locn=defect1_train_folder,rowIndex=i)
numberOfSampleExtractionsShown = 0
for i, (index, row) in enumerate(defect3_data.iterrows()):
    image = extract_defect_contours(file_name=row.ImageId, mask=rle_decode(row.EncodedPixels).T,folder_locn=defect3_train_folder,rowIndex=i)

#Show a few samples of both defect1 image extraction as well as defect3 image extraction
numberOfSampleExtractions = 15
print("-----------defect1 image extractions-----")
for defect_file_name in os.listdir(defect1_train_folder):
    img = cv.imread(os.path.join(defect1_train_folder,defect_file_name))
    if not img is None:
        if numberOfSampleExtractions>0:
            plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.show()
            numberOfSampleExtractions = numberOfSampleExtractions-1
            
print("-----------defect3 image extractions-----")

numberOfSampleExtractions = 15
for defect3_file_name in os.listdir(defect3_train_folder):
    img3 = cv.imread(os.path.join(defect3_train_folder,defect3_file_name))
    if not img3 is None:
        if numberOfSampleExtractions>0:
            plt.imshow(img3, cmap = 'gray', interpolation = 'bicubic')
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.show()
            numberOfSampleExtractions = numberOfSampleExtractions-1




#now train the custom model for viewing steel defects
import shutil
from imageai.Prediction.Custom import ModelTraining

#create test folders and populate with some test images
#since the only defect object images we have are the once we extracted in train folder, we will need to move some of those from train folder to test folder
defect1_test_folder = Path('/kaggle/working/steelDefects/test/defect1/defect1-test-images/')
defect3_test_folder = Path('/kaggle/working/steelDefects/test/defect3/defect3-test-images/')
if not os.path.exists(defect1_test_folder):
        os.makedirs(defect1_test_folder)

if not os.path.exists(defect3_test_folder):
        os.makedirs(defect3_test_folder)

#remove all none/blank images before starting to train
for defect1_file_name in os.listdir(defect1_train_folder):
    defectfilepath = os.path.join(defect1_train_folder,defect1_file_name)
    img3 = cv.imread(defectfilepath)
    if img3 is None:
        if (os.path.isfile(defectfilepath)):
            os.remove(defectfilepath)

for defect3_file_name in os.listdir(defect3_train_folder):
    defectfilepath = os.path.join(defect3_train_folder,defect3_file_name)
    img3 = cv.imread(defectfilepath)
    if img3 is None:
        if (os.path.isfile(defectfilepath)):
            os.remove(defectfilepath)
for defect1_file_name in os.listdir(defect1_test_folder):
    defectfilepath = os.path.join(defect1_test_folder,defect1_file_name)
    img3 = cv.imread(defectfilepath)
    if img3 is None:
        if (os.path.isfile(defectfilepath)):
            os.remove(defectfilepath)
for defect3_file_name in os.listdir(defect3_test_folder):
    defectfilepath = os.path.join(defect3_test_folder,defect3_file_name)
    img3 = cv.imread(defectfilepath)
    if img3 is None:
        if (os.path.isfile(defectfilepath)):
            os.remove(defectfilepath)
            
print("number of defect 1 train images:",len(os.listdir(defect1_train_folder)))
print("number of defect 3 images:",len(os.listdir(defect3_train_folder)))
#since the competition train data seems to have over 1000 defect 1 images and over 6000 defect 3 images, we can move 150 of both to test folders
#since this block of code maybe run multiple times, it is better to see how many files have already been moved instead of moving 150 every time
no_of_defect1_files_to_move = 150 - len(os.listdir(defect1_test_folder))
print("will move:",no_of_defect1_files_to_move, " defect1 images from train to test" )
for defect_file_name in os.listdir(defect1_train_folder):
    srcFile = os.path.join(defect1_train_folder,defect_file_name)
    destFile = os.path.join(defect1_test_folder,defect_file_name)
    if (no_of_defect1_files_to_move>0):
        shutil.move(srcFile, destFile)
    no_of_defect1_files_to_move = no_of_defect1_files_to_move - 1

no_of_defect3_files_to_move = 150 - len(os.listdir(defect3_test_folder))
print("will move:",no_of_defect3_files_to_move, " defect3 images from train to test" )
for defect_file_name in os.listdir(defect3_train_folder):
    srcFile = os.path.join(defect3_train_folder,defect_file_name)
    destFile = os.path.join(defect3_test_folder,defect_file_name)
    if (no_of_defect3_files_to_move>0):
        shutil.move(srcFile, destFile)
    no_of_defect3_files_to_move = no_of_defect3_files_to_move - 1
#print("number of defect 1 train images:",len(os.listdir(defect1_train_folder)))
#print("number of defect 3 images:",len(os.listdir(defect3_train_folder)))

print("model training will now start - can take a day without GPU, so might be commented out - please uncomment .trainModel when you are ready to run")
print("various output models will be in models folder with accuracy mentioned in the file name and extension .h5")
#now that train and test folders for both defect1 and defect3 are ready, train the custom model
model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory(r"/kaggle/working/steelDefects")
#the following line on training model is commented to avoid training on every kernel commit - takes about a day without GPU and about an hour with GPU!
#model_trainer.trainModel(num_objects=2, num_experiments=100, enhance_data=True, batch_size=32, show_network_summary=True)

