#!/usr/bin/env python
# coding: utf-8



## Importing the necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

## for converting images to arrays
import os, cv2




ls ../input/rsna-pneumonia-detection-challenge




# Importing the training label file
train_df = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
train_df.head()




## Importing the class labels file
class_info_df = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv')
class_info_df.head()




## Check the shape of dataframes
print(train_df.shape)
print(class_info_df.shape)




## Merging dataframes just to check the values
merged_df = pd.merge(train_df,class_info_df, 
                     left_index = True, 
                     right_index = True)




## check the columns
merged_df.columns




## delete repeated columns
merged_df.drop('patientId_y', axis=1, inplace=True)




## Changing the column names into meaningful names
merged_df.rename(columns={'patientId_x':'patientId', 'Target':'target',
                         'class':'target_class_desc'}, inplace=True)




## Check the final merge table
merged_df.head()




ls ../input/rsna-stage-2-png-converted-files/stage_2_png_converted_files/




train_path = os.path.join('..','input','rsna-stage-2-png-converted-files',
                          'stage_2_png_converted_files','stage_2_png_converted_files')
train_path




test_path = os.path.join('..','input','rsna-pneumonia-detection-challenge','stage_2_test_images')
test_path




'''
def convert_dicom_to_png():
    # make it True if you want in PNG format
    PNG = True
    # Specify the .dcm folder path
    folder_path = train_path
    # Specify the output jpg/png folder path
    png_folder_path = "../datasets/png_converted_files"
    images_path = os.listdir(folder_path)
    for n, image in enumerate(images_path):
        ds = pydicom.dcmread(os.path.join(folder_path, image), force=True)
        pixel_array_numpy = ds.pixel_array
        if PNG == False:
            image = image.replace('.dcm', '.jpg')
        else:
            image = image.replace('.dcm', '.png')
        cv2.imwrite(os.path.join(png_folder_path, image), pixel_array_numpy)
        if n % 1000 == 0:
            print('{} image converted'.format(n))
'''




## Function to convert training images to arrays
def train_images_to_arrays():
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
        Source: Chris Crawford: https://www.kaggle.com/crawford/resize-and-save-images-as-numpy-arrays-128x128
    """

    x = [] # images as arrays
    y = [] # labels 
    WIDTH = 224 # for VGG-16
    HEIGHT = 224 # for VGG-16

    for image in enumerate(merged_df.patientId):    
        
        img_name = image[1]
        image_path = train_path + '/' + img_name + '.png'
        
        # Read and resize image
        full_size_image = cv2.imread(image_path)
        
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        
        # Labels
        index_of_image = image[0]
        target_value = merged_df.target.loc[index_of_image]
        #print(target_value)
        y.append(target_value)

    return x,y




## Obtain X and y as arrays...
X,y = train_images_to_arrays()




## Saving arrays for future use
## Can't save...Kaggle is read-only.. need to import data locally!!!!

## --- Remove the comment in the code below to save in your local machine for code reuse ----
# np.savez_compressed("../input/rsna-stage-2-png-converted-files/x_images", X)
# np.savez_compressed("../input/rsna-stage-2-png-converted-files/y_pneumonia", y)

