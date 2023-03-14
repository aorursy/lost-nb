#!/usr/bin/env python
# coding: utf-8

# In[1]:


# copy to working directory
get_ipython().system('cp -r ../input/maskrcnn-keras-source-code/MaskRCNN/* ./')


# In[2]:


import numpy as np 
import pandas as pd 
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys, os, random, glob, cv2, math

from mrcnn import utils
from mrcnn.model import log
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.config import Config


# In[3]:


# for reproducibility
def seed_all(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

seed_all(42)
sns.set(style="darkgrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


ORIG_SIZE     = 1024
epoch         = 100
data_root     = '/kaggle/input'
packages_root = '/kaggle/working'


# In[5]:


# load annotation files
df = pd.read_csv(os.path.join(data_root , 'global-wheat-detection/train.csv'))
df.head()


# In[6]:


# information summary
df.info()


# In[7]:


plt.figure(figsize=(9,5))
sns.countplot(df.source)
plt.show()


# In[8]:


# image directory
img_root = '../input/global-wheat-detection/train/'
len(os.listdir(img_root)) - len(df.image_id.unique())


# In[9]:


df['bbox'] = df['bbox'].apply(lambda x: x[1:-1].split(","))

df['x'] = df['bbox'].apply(lambda x: x[0]).astype('float32')
df['y'] = df['bbox'].apply(lambda x: x[1]).astype('float32')
df['w'] = df['bbox'].apply(lambda x: x[2]).astype('float32')
df['h'] = df['bbox'].apply(lambda x: x[3]).astype('float32')

df = df[['image_id','x', 'y', 'w', 'h']]
df.head()


# In[10]:


class WheatDetectorConfig(Config):
    # Give the configuration a recognizable name  
    NAME = 'wheat'
    
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    BACKBONE = 'resnet101'
    
    # number of classes (we would normally add +1 for the background)
    # BG + Wheat
    NUM_CLASSES = 2
    
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 120
    
    # Use different size anchors because our target objects are multi-scale (wheats are some too big, some too small)
    RPN_ANCHOR_SCALES = (16, 32, 64, 128)  # anchor side in pixels
    
    # Learning rate
    LEARNING_RATE = 0.005
    WEIGHT_DECAY  = 0.0005
    
    # Maximum number of ROI’s, the Region Proposal Network (RPN) will generate for the image
    TRAIN_ROIS_PER_IMAGE = 350 
    
    # Skip detections with < 60% confidence
    DETECTION_MIN_CONFIDENCE = 0.60
    
    # Increase with larger training
    VALIDATION_STEPS = 60
    
    # Maximum number of instances that can be detected in one image.
    MAX_GT_INSTANCES = 500 # 200 
 
    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.0,
        "rpn_bbox_loss": 1.0,
        "mrcnn_class_loss": 1.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 1.0
        }

config = WheatDetectorConfig()
config.display()


# In[11]:


def get_jpg(img_dir, anns):
    '''
    input:
        img_dir: image directory of the train sets
        anns: specified image ids for train or validation
    return:
        img files with specified image ids
    '''
    id      = []
    jpg_fps = []

    for index, row in anns.iterrows():
        id.append(row['image_id'])

    for i in os.listdir(img_dir):
        if os.path.splitext(i)[0] not in id:
            continue
        else:
            jpg_fps.append(os.path.join(img_dir, i))

    return list(set(jpg_fps))

def get_dataset(img_dir, anns): 
    image_fps = get_jpg(img_dir, anns)

    image_annotations = {fp: [] for fp in image_fps}

    for index, row in anns.iterrows(): 
        fp = os.path.join(img_dir, row['image_id'] + '.jpg')
        image_annotations[fp].append(row)

    return image_fps, image_annotations 


# In[12]:


class DetectorDataset(utils.Dataset):
    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)
        
        # Add classes
        self.add_class('GlobalWheat', 1 , 'Wheat') # only one class, wheat
        
        # add images 
        for id, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('GlobalWheat', image_id=id, 
                           path=fp, annotations=annotations, 
                           orig_height=orig_height, orig_width=orig_width)

    # load bbox, most important function so far        
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
    
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), 
                            dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count),
                            dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                x = int(a['x'])
                y = int(a['y'])
                w = int(a['w'])
                h = int(a['h'])
                mask_instance = mask[:, :, i].copy()
                cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                mask[:, :, i] = mask_instance
                class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)
    
    # simple image loader 
    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        image = cv2.imread(fp, cv2.IMREAD_COLOR)
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image
    
    # simply return the image path
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


# In[13]:


image_ids = df['image_id'].unique()

valid_ids = image_ids[-700:]
train_ids = image_ids[:-700]

valid_df = df[df['image_id'].isin(valid_ids)]
train_df = df[df['image_id'].isin(train_ids)]
train_df.shape, valid_df.shape


# In[14]:


len(train_df.image_id.unique()), len(valid_df.image_id.unique())


# In[15]:


# grab all image file path with concern annotation
train_image_fps, train_image_annotations = get_dataset(img_root,
                                                       anns=train_df)

# make data generator with that
dataset_train = DetectorDataset(train_image_fps, 
                                train_image_annotations,
                                ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()

print("Class Count: {}".format(dataset_train.num_classes))
for i, info in enumerate(dataset_train.class_info):
    print("{:3}. {:50}".format(i, info['name']))


# In[16]:


# grab all image file path with concern annotation
valid_image_fps, valid_image_annotations = get_dataset(img_root, 
                                           anns=valid_df)

# make data generator with that
dataset_valid = DetectorDataset(valid_image_fps, valid_image_annotations,
                                ORIG_SIZE, ORIG_SIZE)
dataset_valid.prepare()

print("Class Count: {}".format(dataset_valid.num_classes))
for i, info in enumerate(dataset_valid.class_info):
    print("{:3}. {:50}".format(i, info['name']))


# In[17]:


class_ids = [0]

while class_ids[0] == 0:  ## look for a mask
    image_id = random.choice(dataset_train.image_ids)
    image_fp = dataset_train.image_reference(image_id)
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)

print(image.shape)

plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
masked = np.zeros(image.shape[:2])
for i in range(mask.shape[2]):
    masked += image[:, :, 0] * mask[:, :, i]
plt.imshow(masked, cmap='gray')
plt.axis('off')

print(class_ids)
plt.show()


# In[18]:


# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids,5)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, 
                                dataset_train.class_names, limit=1)


# In[19]:


# Load random image and mask.
image_id = np.random.choice(dataset_train.image_ids, 1)[0]
image = dataset_train.load_image(image_id)
mask, class_ids = dataset_train.load_mask(image_id)
original_shape = image.shape

# Resize
image, window, scale, padding, _ = utils.resize_image(image, 
                                                      min_dim=config.IMAGE_MIN_DIM, 
                                                      max_dim=config.IMAGE_MAX_DIM,
                                                      mode=config.IMAGE_RESIZE_MODE)
mask = utils.resize_mask(mask, scale, padding)

# Compute Bounding box
bbox = utils.extract_bboxes(mask)

# Display image and additional stats
print("Original shape: ", original_shape)
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)

# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, 
                            dataset_train.class_names)


# In[20]:


get_ipython().system('pip install ../input/img-aug-v04/imgaug-0.4.0-py2.py3-none-any.whl')


# In[21]:


import warnings
from imgaug import augmenters as iaa
warnings.filterwarnings("ignore")

augmentation = iaa.Sequential([
        iaa.OneOf([ ## rotate
            iaa.Affine(rotate=0),
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
            iaa.Affine(rotate=270),
        ]),

        iaa.Fliplr(0.5),
        iaa.Flipud(0.2),

        iaa.OneOf([ # drop out augmentation
            iaa.Cutout(fill_mode="constant", cval=255),
            iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
            ]),

        iaa.OneOf([ ## weather augmentation
            iaa.Snowflakes(flake_size=(0.2, 0.4), speed=(0.01, 0.07)),
            iaa.Rain(speed=(0.3, 0.5)),
        ]),  

        iaa.OneOf([ ## brightness or contrast
            iaa.Multiply((0.8, 1.0)),
            iaa.contrast.LinearContrast((0.9, 1.1)),
        ]),

        iaa.OneOf([ ## blur or sharpen
            iaa.GaussianBlur(sigma=(0.0, 0.1)),
            iaa.Sharpen(alpha=(0.0, 0.1)),
        ])
    ],
    # do all of the above augmentations in random order
    random_order=True
)


# In[22]:


# from official repo
def get_ax(rows=1, cols=1, size=7):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# Load the image multiple times to show augmentations
limit = 4
ax = get_ax(rows=2, cols=limit//2)

for i in range(limit):
    image, image_meta, class_ids,    bbox, mask = modellib.load_image_gt(
        dataset_train, config, image_id, use_mini_mask=False, 
        augment=False, augmentation=augmentation)
    
    visualize.display_instances(image, bbox, mask, class_ids,
                                dataset_train.class_names, ax=ax[i//2, i % 2],
                                show_mask=False, show_bbox=False)


# In[23]:


def model_definition():
    print("loading mask R-CNN model")
    model = modellib.MaskRCNN(mode='training', 
                              config=config, 
                              model_dir=packages_root)
    
    # load the weights for COCO
    model.load_weights(data_root + '/cocowg/mask_rcnn_coco.h5',
                       by_name=True, 
                       exclude=["mrcnn_class_logits",
                                "mrcnn_bbox_fc",  
                                "mrcnn_bbox","mrcnn_mask"])
    return model   

model = model_definition()


# In[24]:


from keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau, CSVLogger)

def callback():
    cb = []
    checkpoint = ModelCheckpoint(packages_root+'wheat_wg.h5',
                                 save_best_only=True,
                                 mode='min',
                                 monitor='val_loss',
                                 save_weights_only=True, verbose=1)
    cb.append(checkpoint)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.3, patience=5,
                                   verbose=1, mode='auto',
                                   epsilon=0.0001, cooldown=1, min_lr=0.00001)
    log = CSVLogger(packages_root+'wheat_history.csv')
    cb.append(log)
    cb.append(reduceLROnPlat)
    return cb


# In[25]:


get_ipython().run_cell_magic('time', '', "CB = callback()\nTRAIN = False\n\nclass WheatInferenceConfig(WheatDetectorConfig):\n    GPU_COUNT = 1\n    IMAGES_PER_GPU = 1\n\nif TRAIN:\n    model.train(dataset_train, dataset_valid, \n                augmentation=augmentation, \n                learning_rate=config.LEARNING_RATE,\n                custom_callbacks = CB,\n                epochs=epoch, layers='all') \nelse:\n    inference_config = WheatInferenceConfig()\n    # Recreate the model in inference mode\n    model = modellib.MaskRCNN(mode='inference', \n                              config=inference_config,\n                              model_dir=packages_root)\n    \n    model.load_weights(data_root + '/096269-wheat-r101/wheat_096269_101_1024.h5', \n                       by_name = True)")


# In[26]:


history = pd.read_csv(data_root + '/wheatweight/wheat_history.csv') 

# find the lowest validation loss score
print(history.loc[history['val_loss'].idxmin()])
history.head()


# In[27]:


plt.figure(figsize=(19,6))

plt.subplot(131)
plt.plot(history.epoch, history.loss, label="Train loss")
plt.plot(history.epoch, history.val_loss, label="Valid loss")
plt.legend()

plt.subplot(132)
plt.plot(history.epoch, history.mrcnn_class_loss, label="Train class ce")
plt.plot(history.epoch, history.val_mrcnn_class_loss, label="Valid class ce")
plt.legend()

plt.subplot(133)
plt.plot(history.epoch, history.mrcnn_bbox_loss, label="Train box loss")
plt.plot(history.epoch, history.val_mrcnn_bbox_loss, label="Valid box loss")
plt.legend()

plt.show()


# In[28]:


image_id = np.random.choice(dataset_valid.image_ids, 2)

for img_id in image_id:
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =        modellib.load_image_gt(dataset_valid, inference_config,     
                               img_id, use_mini_mask=False)

    info = dataset_valid.image_info[img_id]
    results = model.detect([original_image], verbose=1)
    r = results[0]

    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset_valid.class_names, r['scores'], ax=get_ax(), title="Predictions")
    
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)


# In[29]:


get_ipython().run_cell_magic('time', '', '\nthresh_score = [0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75]\n\ndef evaluate_threshold_range(test_set, image_ids, model, \n                             iou_thresholds, inference_config):\n    \'\'\'Calculate mAP based on iou_threshold range\n    inputs:\n        test_set        : test samples\n        image_ids       : image ids of the test samples\n        model           : trained model\n        inference_config: test configuration\n        iou_threshold   : by default [0.5:0.75:0.05]\n    return:\n        AP : mAP[@0.5:0.75] scores lists of the test samples\n    \'\'\'\n    # placeholder for all the ap of all classes for IoU socres 0.5 to 0.95 with step size 0.05\n    AP = []\n    np.seterr(divide=\'ignore\', invalid=\'ignore\') \n    \n    for image_id in image_ids:\n        # Load image and ground truth data\n        image, image_meta, gt_class_id, gt_bbox, gt_mask =\\\n            modellib.load_image_gt(test_set, inference_config,\n                                   image_id, use_mini_mask=False)\n\n        # Run object detection\n        results = model.detect([image], verbose=0)\n        r = results[0]\n        AP_range = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask, \n                                          r["rois"], r["class_ids"], r["scores"], r[\'masks\'],\n                                          iou_thresholds=iou_thresholds, verbose=0)\n        \n        if math.isnan(AP_range):\n            continue\n            \n        # append the scores of each samples\n        AP.append(AP_range)   \n        \n    return AP\n\nAP = evaluate_threshold_range(dataset_valid, dataset_valid.image_ids,\n                              model, thresh_score, inference_config)\n\nprint("AP[0.5:0.75]: ", np.mean(AP))')


# In[30]:


def get_jpg(img_dir):
    jpg_fps = glob.glob(img_dir + '*.jpg')
    return list(set(jpg_fps))

# Get filenames of test dataset jpg images
test_img_root  = data_root + '/global-wheat-detection/test/'
test_image_fps = get_jpg(test_img_root)


# In[31]:


# show a few test image detection example
for image_id in test_image_fps:
    image = cv2.imread(image_id, cv2.IMREAD_COLOR)

    # assume square image 
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]

    # If grayscale. Convert to RGB for consistency.
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1) 

    resized_image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)

    image_id = os.path.splitext(os.path.basename(image_id))[0]

    results = model.detect([resized_image])
    r = results[0]
    for bbox in r['rois']: 
        x1 = int(bbox[1] * resize_factor)
        y1 = int(bbox[0] * resize_factor)
        x2 = int(bbox[3] * resize_factor)
        y2 = int(bbox[2] * resize_factor)
        cv2.rectangle(image, (x1,y1), (x2,y2), (77, 255, 9), 3, 1)
        width  = x2 - x1 
        height = y2 - y1 

    plt.figure(figsize=(10,10)) 
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.gist_gray)


# In[32]:


# Make predictions on test images, write out sample submission
def predict(image_fps, filepath='submission.csv', min_conf=0.50):
    # assume square image
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]

    with open(filepath, 'w') as file:
        file.write("image_id,PredictionString\n")

        for image_id in tqdm(image_fps):
            image = cv2.imread(image_id, cv2.IMREAD_COLOR)
            # If grayscale. Convert to RGB for consistency.
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = np.stack((image,) * 3, -1)
                
            image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)

            image_id = os.path.splitext(os.path.basename(image_id))[0]

            results = model.detect([image])
            r = results[0]

            out_str = ""
            out_str += image_id
            out_str += ","
            
            assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
            
            if len(r['rois']) == 0:
                pass
            else:
                num_instances = len(r['rois'])
                for i in range(num_instances):
                    if r['scores'][i] > min_conf:
                               
                        out_str += ' '
                        out_str += "{0:.4f}".format(r['scores'][i])
                        out_str += ' '

                        # x1, y1, width, height
                        x1 = r['rois'][i][1]
                        y1 = r['rois'][i][0]
                        width = r['rois'][i][3] - x1
                        height = r['rois'][i][2] - y1
                        bboxes_str = "{} {} {} {}".format( x1*resize_factor, y1*resize_factor,                                                            width*resize_factor, height*resize_factor )
                        out_str += bboxes_str

            file.write(out_str+"\n")


# In[33]:


submission = os.path.join(packages_root, 'submission.csv')
predict(test_image_fps, filepath=submission)


# In[34]:


submit = pd.read_csv(submission)
submit.head(10)

