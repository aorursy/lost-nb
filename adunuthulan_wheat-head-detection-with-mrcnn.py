#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# set model and data directory
# TODO: Add WBF https://www.kaggle.com/marcelosanchezortega/wbf-over-tta-single-model-efficientdet

model_dir = '../input/wheat-mrcnn-weights/'
data_dir = '../input/global-wheat-detection'


# In[ ]:


#copy files to working directory + update model file so it works w/ tf2
get_ipython().run_line_magic('cp', '-r ../input/mask-rcnn2/Mask_RCNN/Mask_RCNN ./')
get_ipython().run_line_magic('cp', '../input/mask-rcnn2/model.py ./Mask_RCNN/mrcnn/model.py')


# In[ ]:


import pandas as pd
import numpy as np
import ensemble_boxes
import imgaug

get_ipython().run_line_magic('cd', 'Mask_RCNN/')

from mrcnn.utils import Dataset, extract_bboxes
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances
get_ipython().run_line_magic('cd', '../')

get_ipython().run_line_magic('ls', '')

import os
import matplotlib
import ast

import keras
print(keras.__version__)


# In[ ]:


#setup mrcnn for the current system
get_ipython().run_line_magic('cd', 'Mask_RCNN/')
get_ipython().system('python setup.py install')
get_ipython().run_line_magic('cd', '../')


# In[ ]:


#Functions for Wheat Images
class WheatDataset(Dataset):
  # load the dataset definitions
  def load_dataset(self, dataset_dir, is_train=True):
    # define one class
    self.add_class("dataset", 1, "wheat_head")
    # define data locations
    images_dir = ""
    aug_dir = []
    if is_train:
      images_dir = dataset_dir + '/train/'
    else:
      images_dir = dataset_dir + "/test/"

    # find all images
    for filename in os.listdir(images_dir):
      # extract image id
      image_id = filename[:-4]
      # skip bad images
      img_path = images_dir + filename
      # add to dataset
      self.add_image('dataset', image_id=image_id, path=img_path, annotation=img_path)

  def extract_boxes(self, path):
    return dataset_dicts[index_dict[path]]["annotations"], dataset_dicts[index_dict[path]]["width"], dataset_dicts[index_dict[path]]["height"]

  # load the masks for an image
  def load_mask(self, image_id):
    # get details of image
    info = self.image_info[image_id]
    # define box file location
    path = info['annotation']
    # load XML
    boxes, w, h = self.extract_boxes(path)
    # create one array for all masks, each on a different channel
    masks = np.zeros([h, w, len(boxes)], dtype='uint8')
    # create masks
    class_ids = list()
    for i in range(len(boxes)):
      box = boxes[i]
      row_s, row_e = int(box[1]), int(box[3])
      col_s, col_e = int(box[0]), int(box[2])
      masks[row_s:row_e, col_s:col_e, i] = 1
      class_ids.append(self.class_names.index('wheat_head'))
    return masks, np.asarray(class_ids, dtype='int32')
 	
  # load an image reference
  def image_reference(self, image_id):
    info = self.image_info[image_id]
    return info["path"]

  


# In[ ]:


# test/val set
test_set = WheatDataset()
test_set.load_dataset(data_dir, is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))


# In[ ]:


# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "wheat_cfg"
	# number of classes (background + wheat)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	#MAX_GT_INSTANCES = 200

	DETECTION_MIN_CONFIDENCE = 0.7
	DETECTION_NMS_THRESHOLD = 0.3


# In[ ]:


# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir=model_dir, config=cfg)
# download the weights
model.load_weights('../input/wheat-mrcnn-weights/mask_rcnn_wheat_cfg_0033.h5', by_name=True)


# In[ ]:





# In[ ]:


#WBF using notebook from Alex Shonenkov
class BaseWheatTTA:
    """ author: @shonenkov """
    image_size = 512

    def augment(self, image):
        raise NotImplementedError
    
    def batch_augment(self, images):
        raise NotImplementedError
    
    def deaugment_boxes(self, boxes):
        raise NotImplementedError

class TTAHorizontalFlip(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(1)
    
    def batch_augment(self, images):
        return images.flip(2)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [1,3]] = self.image_size - boxes[:, [3,1]]
        return boxes

class TTAVerticalFlip(BaseWheatTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return image.flip(2)
    
    def batch_augment(self, images):
        return images.flip(3)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [0,2]] = self.image_size - boxes[:, [2,0]]
        return boxes
    
class TTARotate90(BaseWheatTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 1, (2, 3))
    
    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0,2]] = self.image_size - boxes[:, [1,3]]
        res_boxes[:, [1,3]] = boxes[:, [2,0]]
        return res_boxes

class TTACompose(BaseWheatTTA):
    """ author: @shonenkov """
    def __init__(self, transforms):
        self.transforms = transforms
        
    def augment(self, image):
        for transform in self.transforms:
            image = transform.augment(image)
        return image
    
    def batch_augment(self, images):
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images
    
    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:,0] = np.min(boxes[:, [0,2]], axis=1)
        result_boxes[:,2] = np.max(boxes[:, [0,2]], axis=1)
        result_boxes[:,1] = np.min(boxes[:, [1,3]], axis=1)
        result_boxes[:,3] = np.max(boxes[:, [1,3]], axis=1)
        return result_boxes
    
    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)


# In[ ]:


images = []
for image_id in dataset.image_ids:
    image = dataset.load_image(image_id)
		# convert pixel values (e.g. center)
    scaled_image = mold_image(image, cfg)
		# convert image into one sample
    sample = expand_dims(scaled_image, 0)
    images.append(sample)
    


# In[ ]:


# use WBF
from itertools import product

tta_transforms = []
for tta_combination in product([TTAHorizontalFlip(), None], 
                               [TTAVerticalFlip(), None],
                               [TTARotate90(), None]):
    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))

def make_tta_predictions(images, score_threshold=0.25):
    predictions = []
    for tta_transform in tta_transforms:
        result = []
        det = model(tta_transform.batch_augment(images.clone()))

        for i in range(len(images)):
            boxes = det[i]["rois"]    
            scores = det[i]["scores"]
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            boxes = tta_transform.deaugment_boxes(boxes.copy())
            result.append({
                'boxes': boxes,
                'scores': scores[indexes],
            })
        predictions.append(result)
    return predictions


# In[ ]:


# plot a number of photos with ground truth and predictions
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset

#Predict bboxes for images
import cv2
def predict_im(dir):
  image = cv2.cvtColor(cv2.imread(dir), cv2.COLOR_BGR2RGB)
  # convert pixel values (e.g. center)
  scaled_image = mold_image(image, cfg)
  # convert image into one sample
  sample = expand_dims(scaled_image, 0)
  # make prediction
  yhat = model.detect(sample, verbose=0)[0]
  # plot raw pixel data
  pyplot.imshow(image)
  ax = pyplot.gca()
  
  print(len(yhat['rois']))

  # plot each box
  for box in yhat['rois']:
    # get coordinates
    y1, x1, y2, x2 = box
    # calculate width and height of the box
    width, height = x2 - x1, y2 - y1
    # create the shape
    rect = Rectangle((x1, y1), width, height, fill=False, color='red')
    # draw the box
    ax.add_patch(rect)
  pyplot.show()


# In[ ]:


for image_id in test_set.image_ids:
  print(image_id)
  predict_im(test_set.image_reference(image_id))


# In[ ]:


#compute the submission csv for the test images

def submission(dataset, model, cfg):
  sub = pd.DataFrame(columns=["image_id", "PredictionString"])

  for image_id in dataset.image_ids:
    image = dataset.load_image(image_id)
		# convert pixel values (e.g. center)
    scaled_image = mold_image(image, cfg)
		# convert image into one sample
    sample = expand_dims(scaled_image, 0)
    yhat = model.detect(sample, verbose=0)
		# extract results for first sample
    r = yhat[0]
    predStr = ""

    for (bbox, score) in zip(r["rois"], r["scores"]):
      bbox[2] = bbox[2] - bbox[0]
      bbox[3] = bbox[3] - bbox[1]

      predStr = predStr + str(score) + " " + str(bbox[0]) + " " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3]) + " "

    sub = sub.append({"image_id":dataset.image_reference(image_id)[-13:-4], "PredictionString":predStr}, ignore_index=True)

  return sub


# In[ ]:


import csv

sub = submission(test_set, model, cfg)
print(sub)
sub.to_csv('submission.csv', index=False)


# In[ ]:


final = True

if final:
    get_ipython().run_line_magic('rm', '-r Mask_RCNN/')

