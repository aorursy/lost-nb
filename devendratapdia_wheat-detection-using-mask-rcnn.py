#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install /kaggle/input/wheatds/pycocotools-2.0.1/')


# In[2]:


get_ipython().system('pip uninstall -y tensorflow')


# In[3]:


get_ipython().system('pip uninstall -y keras')


# In[4]:


get_ipython().system('pip install -U /kaggle/input/wheatds/tensorboard-1.15.0-py3-none-any.whl')


# In[5]:


get_ipython().system('pip install -U /kaggle/input/wheatds/tensorflow_estimator-1.15.1-py2.py3-none-any.whl')


# In[6]:


get_ipython().system('pip install -U /kaggle/input/wheatds/gast-0.2.2/gast-0.2.2')


# In[7]:


get_ipython().system('pip install -U /kaggle/input/wheatds/astor-0.7.1-py2.py3-none-any.whl')


# In[8]:


get_ipython().system('pip install -U /kaggle/input/wheatds/Keras_Applications-1.0.8-py3-none-any.whl')


# In[9]:


get_ipython().system('pip install -U /kaggle/input/wheatds/tensorflow_gpu-1.15.0-cp37-cp37m-manylinux2010_x86_64.whl')


# In[10]:


get_ipython().system('pip install -U /kaggle/input/wheatds/Keras-2.1.3-py2.py3-none-any.whl')


# In[11]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[12]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
tf.__version__


# In[13]:


import keras
keras.__version__


# In[14]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[15]:


# !sudo python3.6 -m pip install pycocotools


# In[16]:


from IPython.display import clear_output
# !git clone https://github.com/matterport/Mask_RCNN.git # load Mask R-CNN code implementation
# !sudo python3.6 -m pip install pycocotools
#!rm -rf Mask_RCNN/.git/

clear_output()


# In[17]:


# !pip install mask-rcnn-12rics


# In[18]:


import os 
import sys
from tqdm import tqdm
import cv2
import numpy as np
import json
import skimage.draw
import matplotlib
import matplotlib.pyplot as plt
import random

# Import COCO config
IN_DIR = '/kaggle/input/'
OUT_DIR = '/kaggle/working/'

# # Root directory of the project
ROOT_DIR = os.path.join(IN_DIR, 'maskrcnn/')
# # Import Mask RCNN
sys.path.append(ROOT_DIR) 
from mrcnn.config import Config
from mrcnn import utils
from mrcnn.model import log
import mrcnn.model as modellib
from mrcnn import visualize


sys.path.append(os.path.join(IN_DIR, 'wheatds'))
import coco

plt.rcParams['figure.facecolor'] = 'white'

clear_output()


# In[19]:


def get_ax(rows=1, cols=1, size=7):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# In[20]:


MODEL_DIR = OUT_DIR # directory to save logs and trained model
# ANNOTATIONS_DIR = 'brain-tumor/data/new/annotations/' # directory with annotations for train/val sets
#DATASET_DIR = 'brain-tumor/data_cleaned/' # directory with image data
DATASET_DIR = os.path.join(IN_DIR, 'global-wheat-detection') # directory with image data
DEFAULT_LOGS_DIR = OUT_DIR

# Local path to trained weights file
MODELDATASET_DIR = os.path.join(IN_DIR, 'wheatds')
COCO_MODEL_PATH = os.path.join(MODELDATASET_DIR, "mask_rcnn_coco.h5")
# # Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)


# In[21]:


class WheatConfig(Config):
    """Configuration for training on the wheat heads dataset.
    """
    # Give the configuration a recognizable name
    NAME = 'wheat_detector'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # background + wheat
    DETECTION_MIN_CONFIDENCE = 0.60
    STEPS_PER_EPOCH = 2
    VALIDATION_STEPS = 1
    LEARNING_RATE = 0.001
    LOSS_WEIGHTS = {'rpn_class_loss': 1.0, 'rpn_bbox_loss': 1.0, 'mrcnn_class_loss': 1.0, 'mrcnn_bbox_loss': 1.0, 'mrcnn_mask_loss': 0.0}

config = WheatConfig()
config.display()


# In[22]:


# training dataset
import pandas as pd
anns = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
anns.head()


# In[23]:


import glob
def get_image_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*.jpg')
    return list(set(dicom_fps))

def parse_dataset(dicom_dir, anns): 
    image_fps = get_image_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows(): 
        fp = os.path.join(dicom_dir, row['image_id']+'.jpg')
        iannos = eval(row[3])
        image_annotations[fp].append(iannos)
    return image_fps, image_annotations 


# In[24]:


train_dir = os.path.join(DATASET_DIR, 'train')
image_fps, image_annotations = parse_dataset(train_dir, anns=anns)


# In[25]:


image_annotations


# In[26]:


class WheatDataset(utils.Dataset):

    def load_wheat_scan(self, dataset_dir, img_annotations, orig_height, orig_width, is_train=True):
        """Load a subset of the wheat dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("wheat", 1, "wheat")
        i = 0
        for image_path in img_annotations:
            i = i + 1
            #print(image_path.rstrip(r'.jpg').lstrip(r'train/'))
            # skip all images after 150 if we are building the train set
            if is_train and int(i) >= 3420:
                continue
            # skip all images before 150 if we are building the test/val set
            if not is_train and int(i) < 3000:
                continue
                
            self.add_image(
                "wheat",
                image_id=image_path.rstrip(r'.jpg').lstrip(r'train/'),  # use file name as a unique image id
                path=image_path,
                annotations=img_annotations[image_path],
                orig_height=orig_height, orig_width=orig_width
            )

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                x = int(a[0])
                y = int(a[1])
                w = int(a[2])
                h = int(a[3])
                mask_instance = mask[:, :, i].copy()
                cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                mask[:, :, i] = mask_instance
                class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "wheat":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


# In[27]:


DEFAULT_LOGS_DIR


# In[28]:


model = modellib.MaskRCNN(
    mode='training', 
    config=config, 
    model_dir=MODEL_DIR
)

model.load_weights(
    COCO_MODEL_PATH, 
    by_name=True, 
    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
)


# In[29]:


import glob
def get_image_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*.jpg')
    return list(set(dicom_fps))

def parse_val_dataset(dicom_dir, anns): 
    image_fps = get_image_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows(): 
        fp = os.path.join(dicom_dir, row['image_id']+'.jpg')
        iannos = eval(row[3])
        if fp in image_annotations:
            image_annotations[fp].append(iannos)
    return image_fps, image_annotations 


# In[30]:


val_dir = os.path.join(DATASET_DIR, 'val')

image_val_fps, image_val_annotations = parse_val_dataset(val_dir, anns=anns)


# In[31]:


import warnings
warnings.filterwarnings(action='ignore')


# In[32]:


# Training dataset.
ORIG_SIZE = 1024
dataset_train = WheatDataset()
dataset_train.load_wheat_scan('train', image_annotations, ORIG_SIZE, ORIG_SIZE, is_train=True)
dataset_train.prepare()

# Validation dataset
dataset_val = WheatDataset()
dataset_val.load_wheat_scan('train', image_annotations, ORIG_SIZE, ORIG_SIZE, is_train=False)
dataset_val.prepare()

# dataset_test = WheatDataset()
# dataset_test.load_brain_scan('test')
# dataset_test.prepare()

# Since we're using a very small dataset, and starting from
# COCO trained weights, we don't need to train too long. Also,
# no need to train all layers, just the heads should do it.
print("Training network heads")
model.train(
    dataset_train, dataset_val,
    learning_rate=config.LEARNING_RATE,
    epochs=1,
    layers='heads'
)


# In[33]:


import os
if not os.path.exists('/kaggle/working/wheat_detector'):
    os.mkdir('/kaggle/working/wheat_detector')


# In[34]:


from shutil import copyfile
src = '/kaggle/input/wheatds/mask_rcnn_wheat_detector_0011.h5'
dst = '/kaggle/working/wheat_detector/mask_rcnn_wheat.h5'
copyfile(src, dst)


# In[35]:


# Recreate the model in inference mode
model = modellib.MaskRCNN(
    mode="inference", 
    config=config,
    model_dir=DEFAULT_LOGS_DIR
)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", dst)
model.load_weights(dst, by_name=True)


# In[36]:


def predict_and_plot_differences(dataset, img_id):
    original_image, image_meta, gt_class_id, gt_box, gt_mask =        modellib.load_image_gt(dataset, config, 
                               img_id, use_mini_mask=False)

    results = model.detect([original_image], verbose=0)
    r = results[0]

#     visualize.display_differences(
#         original_image,
#         gt_box, gt_class_id, gt_mask,
#         r['rois'], r['class_ids'], r['scores'], r['masks'],
#         class_names = ['wheat'], title="", ax=get_ax(),
#         show_box=True)


def display_image(dataset, ind):
    plt.figure(figsize=(5,5))
    plt.imshow(dataset.load_image(ind))
    plt.xticks([])
    plt.yticks([])
    plt.title('Original Image')
    plt.show()

#vALIDATION SET
ind = 1
display_image(dataset_val, ind)
predict_and_plot_differences(dataset_val, ind)

ind = 3
display_image(dataset_val, ind)
predict_and_plot_differences(dataset_val, ind)

# #Test Set
# ind = 1
# display_image(dataset_test, ind)
# predict_and_plot_differences(dataset_test, ind)
# ind = 0
# display_image(dataset_test, ind)
# predict_and_plot_differences(dataset_test, ind)


# In[37]:


# Get filenames of test dataset DICOM images
test_dir = os.path.join(DATASET_DIR, 'test')
test_image_fps = get_image_fps(test_dir)


# In[38]:


from PIL import Image
#from matplotlib import image
from mrcnn.visualize import display_instances

# Make predictions on test images, write out sample submission
def predict(image_fps, filepath='submission.csv', min_conf=0.98):
    # assume square image
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    #resize_factor = ORIG_SIZE
    with open(filepath, 'w') as file:
        file.write("image_id,PredictionString\n")

        for image_id in tqdm(image_fps):
            ds = Image.open(image_id)
            image = np.asarray(ds)
            
            #ds = pydicom.read_file(image_id)
            #image = ds.pixel_array
            # If grayscale. Convert to RGB for consistency.
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = np.stack((image,) * 3, -1)
            image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)

            new_image_id = os.path.splitext(os.path.basename(image_id))[0]

            results = model.detect([image])
            r = results[0]

            #display_instances(image, r['rois'], r['masks'], r['class_ids'], ['background', 'wheathead'], r['scores'])
            out_str = ""
            out_str += new_image_id
            out_str += ","
            assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
            if len(r['rois']) == 0:
                pass
            else:
                num_instances = len(r['rois'])

                for i in range(num_instances):
                    if r['scores'][i] > min_conf:
                        out_str += ' '
                        out_str += str(round(r['scores'][i], 2))
                        out_str += ' '

                        # x1, y1, width, height
                        x1 = r['rois'][i][1]
                        y1 = r['rois'][i][0]
                        width = r['rois'][i][3] - x1
                        height = r['rois'][i][2] - y1
                        bboxes_str = "{} {} {} {}".format(int(x1*resize_factor), int(y1*resize_factor),                                                            int(width*resize_factor), int(height*resize_factor))
                        out_str += bboxes_str

            file.write(out_str+"\n")


# In[39]:


submission_fp = os.path.join(OUT_DIR, 'submission.csv')
predict(test_image_fps, filepath=submission_fp, min_conf=0.60)
print(submission_fp)


# In[40]:


df_submission = pd.read_csv(submission_fp)


# In[41]:


df_submission.head(10)


# In[42]:


df_submission.replace(np.nan, '', inplace=True)


# In[43]:


sub_dict = df_submission.set_index('image_id').to_dict()['PredictionString']


# In[44]:


df_submission['PredictionString'] = df_submission['image_id'].apply(lambda x: sub_dict[x].strip())


# In[45]:


df_submission.head()


# In[46]:


df_submission.to_csv(submission_fp, index=False)


# In[ ]:




