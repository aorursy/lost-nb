#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# basic imports
import os, random
from tqdm import tqdm
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np


# In[ ]:


# what do we have here?
print (os.listdir("../input"))


# In[ ]:


# global variables
TRAIN_DIR="../input/rsna-pneumonia-detection-challenge/stage_1_train_images"
# list of training images
TRAIN_LIST=sorted(os.listdir(TRAIN_DIR))

# dictionary to map string class identifiers to numerical
CLASSES_DICT={'Normal': 0, 'Lung Opacity' : 1, 'No Lung Opacity / Not Normal' : 2}

# we might need this later for saving into hdf5 files
CLASSES=np.array([['Normal'.encode("utf-8")],
                  ['Lung Opacity'.encode("utf-8")],
                  ['No Lung Opacity / Not Normal'.encode("utf-8")]])
NUMCLASSES=CLASSES.shape[0]

TEST_DIR="../input/rsna-pneumonia-detection-challenge/stage_1_test_images"
# list of test images
TEST_LIST=sorted(os.listdir(TEST_DIR))

IMAGE_SIZE=[1024,1024]

# dictionaries to map string sex and viewposition to numerical
PATIENTSEX_DICT={'M': 0, 'F' : 1}
VIEWPOSITION_DICT={'AP': 0, 'PA' : 1}

# datasets from RSNA Preprocessed Non-Image Inputs
TRAIN_PROCESSED_CSV_FILE="../input/rsna-preprocessed-nonimage-inputs/stage_1_train_processed.csv"
TRAIN_PROCESSED_CSV_COLUMN_NAMES=['patientId', 'label', 'class', 'sex', 'age', 'viewPosition']

TRAIN_BOUNDINGBOX_CSV_FILE="../input/rsna-preprocessed-nonimage-inputs/stage_1_train_boundingboxes.csv"
TRAIN_BOUNDINGBOX_CSV_COLUMN_NAMES=['patientId', 'boundingbox']

TEST_PROCESSED_CSV_FILE="../input/rsna-preprocessed-nonimage-inputs/stage_1_test_processed.csv"
TEST_PROCESSED_CSV_COLUMN_NAMES=['patientId', 'sex', 'age', 'viewPosition']


# In[ ]:


# how many unique records does TRAIN_BOUNDINGBOX_CSV_FILE contain?
get_ipython().system('printf "Number of bounding box data records (not unique) stored: "; grep -v "patientId,boundingbox" ../input/rsna-preprocessed-nonimage-inputs/stage_1_train_boundingboxes.csv | sort | uniq| wc -l')


# In[ ]:


# what does the stage_1_train_boundingboxes.csv file look like?
get_ipython().system('printf "First 10 rows, including header, in stage_1_train_boundingboxes.csv:\\n\\n";          head -10 ../input/rsna-preprocessed-nonimage-inputs/stage_1_train_boundingboxes.csv')


# In[ ]:


# how many unique records does TRAIN_PROCESSED_CSV_FILE contain?
get_ipython().system('printf "Number of unique train processed data records stored: "; grep -v "patientId,label,class,sex,age,viewPosition" ../input/rsna-preprocessed-nonimage-inputs/stage_1_train_processed.csv | sort | uniq| wc -l')


# In[ ]:


# what does the stage_1_train_processed.csv file look like?
get_ipython().system('printf "First 10 rows, including header, in stage_1_train_processed.csv:\\n\\n";          head -10 ../input/rsna-preprocessed-nonimage-inputs/stage_1_train_processed.csv')


# In[ ]:


# how many unique records does TEST_PROCESSED_CSV_FILE contain?
get_ipython().system('printf "Number of unique train processed data records stored: "; grep -v "patientId,sex,age,viewPosition" ../input/rsna-preprocessed-nonimage-inputs/stage_1_test_processed.csv | sort | uniq| wc -l')


# In[ ]:


# what does the stage_1_test_processed.csv file look like?
get_ipython().system('printf "First 10 rows, including header, in stage_1_test_processed.csv:\\n\\n";          head -10 ../input/rsna-preprocessed-nonimage-inputs/stage_1_test_processed.csv')


# In[ ]:


# read TRAIN_PROCESSED_CSV_FILE into a pandas dataframe
traindf = pd.read_csv(TRAIN_PROCESSED_CSV_FILE,
                      names=TRAIN_PROCESSED_CSV_COLUMN_NAMES,
                      # skip the header line
                      header=0,
                      # index the dataframe on patientId
                      index_col='patientId')

print (traindf.shape)
print (traindf.head(n=10))

# read TRAIN_BOUNDINGBOX_CSV_FILE into a pandas dataframe
boundingboxesdf = pd.read_csv(TRAIN_BOUNDINGBOX_CSV_FILE,
                              names=TRAIN_BOUNDINGBOX_CSV_COLUMN_NAMES,
                              # skip the header line
                              header=0,
                              # index the dataframe on patientId
                              index_col='patientId')

print (boundingboxesdf.shape)
print (boundingboxesdf.head(n=10))

# read TEST_PROCESSED_CSV_FILE into a pandas dataframe
testdf = pd.read_csv(TEST_PROCESSED_CSV_FILE,
                     names=TEST_PROCESSED_CSV_COLUMN_NAMES,
                     # skip the header line
                     header=0,
                     # index the dataframe on patientId
                     index_col='patientId')

print (testdf.shape)
print (testdf.head(n=10))


# In[ ]:


# combine bounding boxes by unique patientId (multiple bounding boxes put in a list)
bboxes=boundingboxesdf.copy().groupby(['patientId'])['boundingbox'].apply(list)
print (bboxes.head(n=10))


# In[ ]:


# save keys (unique patientIds) for 'Normal' 'Lung Opacity,' and 'No Lung Opacity / Not Normal' examples
normalkeys=traindf.index[traindf['class']==0].tolist()
lungopacitykeys=traindf.index[traindf['class']==1].tolist()
otherabnormalkeys=traindf.index[traindf['class']==2].tolist()

print ("Out of a total of {} unique X-rays:".format(len(normalkeys+lungopacitykeys+otherabnormalkeys)))
print (">{} X-rays are labeled as '{}'".format(len(normalkeys), np.squeeze(CLASSES)[0].decode("utf-8")))
print (">{} X-rays are labeled as having '{}'".format(len(lungopacitykeys), np.squeeze(CLASSES)[1].decode("utf-8")))
print (">{} X-rays are labeled as having '{}'".format(len(otherabnormalkeys), np.squeeze(CLASSES)[2].decode("utf-8")))


# In[ ]:


# extract test keys while we are at it
testkeys=testdf.index.tolist()
print ("There are a total of {} test X-rays".format(len(testkeys)))


# In[ ]:


# utility to extract bounding boxes for a given patientId
def getBoundingBoxes (bboxes, key):
    bboxlist=list(bboxes[key])
    return bboxlist


# In[ ]:


# utility to extract x, y, width, and height for a single bounding box
def getBoundingBoxParameters (bbox):
    for i in range(len(bbox)):
        bboxparsed=bbox.split()
        x=bboxparsed[0]
        y=bboxparsed[1]
        width=bboxparsed[2]
        height=bboxparsed[3]  
        return x, y, width, height


# In[ ]:


# make sure we can extract bounding box details from lungopacitykeys
for i, key in enumerate(lungopacitykeys):
    print ("Count: {}, Key={}".format(i+1, key))
    bboxlist=getBoundingBoxes(bboxes, key)
    # print (bboxlist)
    for i in range(len(bboxlist)):
        print("  Bounding Box {}:: ".format(i+1), end="")
        x, y, width, height = getBoundingBoxParameters(bboxlist[i])
        print("x= {}, y= {}, width={}, height={}".format(x, y, width, height))


# In[ ]:


# extract and check counts for boundingboxes
zeroboundingboxcount=0
oneboundingboxcount=0
twoboundingboxcount=0
threeboundingboxcount=0
for key in lungopacitykeys: 
    bboxlist=getBoundingBoxes(bboxes, key)
    #print(len(bboxes))
    if (len(bboxlist)==1):
        zeroboundingboxcount +=1
    if (len(bboxlist))==2:
        oneboundingboxcount +=1
    elif (len(bboxlist))==3:
        twoboundingboxcount +=1
    elif (len(bboxlist))==4:
        threeboundingboxcount +=1
print ("Out of a total of {} X-rays labeled as having '{}':".format(len(lungopacitykeys), np.squeeze(CLASSES)[1].decode("utf-8")))
print (">{}/{} have no bounding boxes".format(zeroboundingboxcount,len(lungopacitykeys)))
print (">{}/{} have bounding boxes".format(oneboundingboxcount+twoboundingboxcount+threeboundingboxcount, len(lungopacitykeys)))
print (">>{} have 1 bounding box".format(oneboundingboxcount))
print (">>{} have 2 bounding boxes".format(twoboundingboxcount))
print (">>{} have 3 bounding boxes".format(threeboundingboxcount))


# In[ ]:


# analyze train and test data


# In[ ]:


# double-check train and test samples
print ("Total train examples are: {}".format(traindf.shape[0]))
print ("Total test examples are: {}".format(testdf.shape[0]))


# In[ ]:


# what does the gender mix look like for train data?
traindf.groupby(['sex']).size().reset_index(name='Count')


# In[ ]:


# visually
traindf.groupby(['sex']).size().plot.bar()


# In[ ]:


# what does the gender mix look like for test data?
testdf.groupby(['sex']).size().reset_index(name='Count')


# In[ ]:


# visually
testdf.groupby(['sex']).size().plot.bar()


# In[ ]:


# 56.1% Male in train data and 57.1% in test data is pretty similar mix


# In[ ]:


# what does the viewPosition mix look like for train data?
traindf.groupby(['viewPosition']).size().reset_index(name='Count')


# In[ ]:


# what does the viewPosition mix look like for test data?
testdf.groupby(['viewPosition']).size().reset_index(name='Count')


# In[ ]:


# 45.6% AP in train data and 46.8% in test data is similar mix


# In[ ]:


# what does the viewPosition mix look like by sex for train data?
traindf.groupby(['sex','viewPosition']).size().reset_index(name='Count')


# In[ ]:


# visually
traindf.groupby(['sex','viewPosition']).size().plot.bar()


# In[ ]:


# what does the viewPosition mix look like by sex for test data?
testdf.groupby(['sex','viewPosition']).size().reset_index(name='Count')


# In[ ]:


# visually
testdf.groupby(['sex','viewPosition']).size().plot.bar()


# In[ ]:


# not quite similar, especially for female examples


# In[ ]:


# what does the age distribution look like for train data?
traindf.groupby(['age']).size().plot.bar(figsize=(10,10))


# In[ ]:


# by the numbers
traindf['age'].describe()


# In[ ]:


testdf.groupby(['age']).size().plot.bar(figsize=(10,10))


# In[ ]:


# by the numbers
testdf['age'].describe()


# In[ ]:


# that is pretty darn close, except for minimum and maximum age examples
# SUMMARY: the datasets provided appear to be pretty balanced, but be careful when using smaller samples
# SUMMARY: as has been noted by others, there does not appear to be any meaningful classification information in the meta-data


# In[ ]:


# review images


# In[ ]:


# utility to load a dicom image and/or key attributes
def loadImage (directory, filename, mode="metadata"):
    imagearray=np.zeros(IMAGE_SIZE)
    patientid= filename.split(".")[0]
    
    if mode=="metadata":
        # load patient meta-data only from file
        patientdata = pydicom.dcmread(os.path.join(directory, filename), stop_before_pixels=True)
    elif mode=="image":
        # load patient meta-data and image from file
        patientdata = pydicom.dcmread(os.path.join(directory, filename))
        imagearray=patientdata.pixel_array
    patientid=patientdata.PatientID
    attributes="{} {} {}".format(patientdata.PatientSex,
                                 patientdata.PatientAge,
                                 patientdata.ViewPosition)
    
    return patientid, attributes, imagearray


# In[ ]:


# review training images (rerun to see different images)
# number of images per class we want to see (configurable)
numimages=3

figure, axes = plt.subplots(NUMCLASSES, numimages, figsize=(20,20))
totalaxes=axes.flatten()
# print (totalaxes.shape)

# display normal images in the first row
normalsample=random.sample(normalkeys, numimages)
for i, key in enumerate(normalsample):
    filename="{}.dcm".format(key)
    # print (filename)
    patientid, attributes, imagearray=loadImage(TRAIN_DIR, filename, mode="image")
    #print (patientid)
    attributes=list(attributes.split())
    #print ("{}-{}-{}".format(attributes[0], attributes[1], attributes[2]))
    totalaxes[0*numimages+i].imshow(imagearray, cmap='bone')
    title="{}\nPatient ID: {}\nView Position: {}".format(np.squeeze(CLASSES)[0].decode("utf-8"), patientid, attributes[2])
    totalaxes[0*numimages+i].set_title(title)
# display images with lung opacity in the second row
lungopacitysample=random.sample(lungopacitykeys, numimages)
for i, key in enumerate(lungopacitysample):
    filename="{}.dcm".format(key)
    #print (filename)
    patientid, attributes, imagearray=loadImage(TRAIN_DIR, filename, mode="image")
    #print (patientid)
    attributes=list(attributes.split())
    #print ("{}-{}-{}".format(attributes[0], attributes[1], attributes[2]))
    totalaxes[1*numimages+i].imshow(imagearray, cmap='bone')
    title="{}\nPatient ID: {}\nView Position: {}".format(np.squeeze(CLASSES)[1].decode("utf-8"), patientid, attributes[2])
    totalaxes[1*numimages+i].set_title(title)
# display images with other abnormalities in the third row
otherabnormalsample=random.sample(otherabnormalkeys, numimages)
for i, key in enumerate(otherabnormalsample):
    filename="{}.dcm".format(key)
    #print (filename)
    patientid, attributes, imagearray=loadImage(TRAIN_DIR, filename, mode="image")
    #print (patientid)
    attributes=list(attributes.split())
    #print ("{}-{}-{}".format(attributes[0], attributes[1], attributes[2]))
    totalaxes[2*numimages+i].imshow(imagearray, cmap='bone')
    title="{}\nPatient ID: {}\nView Position: {}".format(np.squeeze(CLASSES)[2].decode("utf-8"), patientid, attributes[2])
    totalaxes[2*numimages+i].set_title(title)
plt.show()
plt.close()


# In[ ]:


# review test images (rerun to see different images)
# we will take numimages from above

figure, axes = plt.subplots(1, numimages, figsize=(20,20))
totalaxes=axes.flatten()
# print (totalaxes.shape)

testsample=random.sample(testkeys, numimages)
for i, key in enumerate(testsample):
    filename="{}.dcm".format(key)
    # print (filename)
    patientid, attributes, imagearray=loadImage(TEST_DIR, filename, mode="image")
    #print (patientid)
    attributes=list(attributes.split())
    #print ("{}-{}-{}".format(attributes[0], attributes[1], attributes[2]))
    totalaxes[0*numimages+i].imshow(imagearray, cmap='bone')
    title="Patient ID: {}\nView Position: {}".format(patientid, attributes[2])
    totalaxes[0*numimages+i].set_title(title)
plt.show()
plt.close()

