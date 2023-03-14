#!/usr/bin/env python
# coding: utf-8



TENSORFLOW_INSTALL = True
USE_ARFAN_CODE = False
GITHUB_FRESH = False
EVALUATE = True
KAGGLE = True
EVALUATE = True
VISUAL = True




import glob
glob_list = glob.glob(f'/kaggle/input/mask-rcnn-train-1536-5-5w-0-0001/*.h5')
WEIGHTS_PATH = glob_list[0] if glob_list else '' 




WEIGHTS_PATH




if TENSORFLOW_INSTALL:
    get_ipython().system('conda install tensorflow-gpu==1.14.0 -y')
    get_ipython().system('pip install keras==2.1.5')




import tensorflow
print(tensorflow.__version__)
import keras
print(keras.__version__)
print(tensorflow.test.is_gpu_available())
import os
import gc
import sys
import json
import random
from pathlib import Path

import cv2 # CV2 for image manipulation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

from imgaug import augmenters as iaa

import seaborn as sns
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

from sklearn.model_selection import StratifiedKFold, KFold




get_ipython().run_cell_magic('time', '', "with open('/kaggle/input/imaterialist-fashion-2020-fgvc7/label_descriptions.json', 'r') as file:\n    label_desc = json.load(file)\nsample_sub_df = pd.read_csv('/kaggle/input/imaterialist-fashion-2020-fgvc7/sample_submission.csv')\ntrain_df = pd.read_csv('/kaggle/input/imaterialist-fashion-2020-fgvc7/train.csv')")




num_classes = len(label_desc['categories'])
num_attributes = len(label_desc['attributes'])
print(f'Total # of classes: {num_classes}')
print(f'Total # of attributes: {num_attributes}')




categories_df = pd.DataFrame(label_desc['categories'])
attributes_df = pd.DataFrame(label_desc['attributes'])
categories_df




image_df = train_df.groupby('ImageId')['EncodedPixels', 'ClassId', 'AttributesIds'].agg(lambda x: list(x))
size_df = train_df.groupby('ImageId')['Height', 'Width'].first()
image_df = image_df.join(size_df, on='ImageId')

print("Total images: ", len(image_df))
image_df.head()




import os
from pathlib import Path
os.chdir('/kaggle/working/')
if os.path.exists("Mask_RCNN"):
    if GITHUB_FRESH and USE_ARFAN_CODE:
        os.system('rm -rf /kaggle/working/Mask_RCNN')
        get_ipython().system('git clone https://www.github.com/AR-fan/Mask_RCNN.git')
        get_ipython().system('rm -rf .git # to prevent an error when the kernel is committed')
        get_ipython().system('rm -rf images assets # to prevent displaying images at the bottom of a kernel')
elif USE_ARFAN_CODE:
    get_ipython().system('git clone https://www.github.com/AR-fan/Mask_RCNN.git')
    get_ipython().system('rm -rf .git # to prevent an error when the kernel is committed')
    get_ipython().system('rm -rf images assets # to prevent displaying images at the bottom of a kernel        ')
else:
    get_ipython().system('git clone https://github.com/matterport/Mask_RCNN    ')
    get_ipython().system('rm -rf .git # to prevent an error when the kernel is committed')
    get_ipython().system('rm -rf images assets # to prevent displaying images at the bottom of a kernel    ')
    
os.chdir('Mask_RCNN')




DATA_DIR = Path('/kaggle/input/imaterialist-fashion-2020-fgvc7')
ROOT_DIR = Path('/kaggle/working')




# sys.path.append(ROOT_DIR/'Mask_RCNN')
import sys
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log




class FashionConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "class"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    TOP_DOWN_PYRAMID_SIZE = 256
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(categories_df)  # background + 46 classes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM =  1536
    IMAGE_MAX_DIM = 1536

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (75,187,308,554,1032)# (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.7, 0.9, 1.2] # [0.45, 0.9, 1.28, 1.47]
    
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200 # 200-》100
    
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    PRE_NMS_LIMIT = 6000 # 
    
    MASK_POOL_SIZE = 28 # 14->28
    MASK_SHAPE = [56, 56] # [28, 28] -> [56, 56]
    
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    DETECTION_MAX_INSTANCES = 74
    

    MAX_GT_INSTANCES = 74

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 5000 # 100 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5 # 5 50
    
    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3 
    
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
    
config = FashionConfig()
config.display()




# This code partially supports k-fold training, 
# you can specify the fold to train and the total number of folds here
FOLD = 0
N_FOLDS = 2

kf = KFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
splits = kf.split(image_df) # ideally, this should be multilabel stratification

def get_fold():    
    for i, (train_index, valid_index) in enumerate(splits):
        if i == FOLD:
            return image_df.iloc[train_index], image_df.iloc[valid_index]

train_df, valid_df = get_fold()




import warnings 
warnings.filterwarnings("ignore")





class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7 

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=ROOT_DIR)

assert WEIGHTS_PATH != '', "Provide path to trained weights"
print("Loading weights from ", WEIGHTS_PATH)
model.load_weights(WEIGHTS_PATH, by_name=True)




image_unique_array = image_df.index.unique()
image_unique_dict = {}
for i,name in enumerate(image_unique_array):
    image_unique_dict[name] = i
len(image_unique_dict)




get_ipython().system('git clone https://github.com/AR-fan/cocoapi.git ')
#    TypeError: object of type <class 'numpy.float64'> cannot be safely interpreted as an integer.




os.chdir('./cocoapi/PythonAPI')
get_ipython().system('make')




from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval




# Since the submission system does not permit overlapped masks, we have to fix them
def refine_masks(masks):
    rois = np.zeros((masks.shape[-1], 4))
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0) 
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index: 
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask)) 
        union_mask = np.logical_or(masks[:, :, m], union_mask) 
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m]==True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois




evaluate_train_results = []
images = []
annotations = []
categories = []    
id_temp = -1    
fashion_annotations_coco_predicts = []

for i, row in tqdm(valid_df[:10].iterrows(), total=10):    
# for i, row in tqdm(image_df[:10].iterrows(), total=10):    

    image_path = str(DATA_DIR/'train'/row.name) + '.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = model.detect([image])[0]
    result['shape'] = image.shape[0:2]

    if result['rois'] is None:
        continue         

    if result['masks'].shape[-1] > 1:
        masks, rois = refine_masks(result['masks']) 
    else: 
        masks = result['masks']

    # Loop through detections
    for k in range(result['rois'].shape[0]):
        class_id = int(result["class_ids"][k]) - 1 
        score = result["scores"][k]
        bbox = np.around(result["rois"][k], 1)
        mask = masks[:, :, k].astype(np.uint8)

        evaluate_result = {
            "image_id": image_unique_dict[row.name],
            "category_id": class_id,
            "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
            "score": score,
            "segmentation": maskUtils.encode(np.asfortranarray(mask)) 
#                 "area"
        }
        evaluate_train_results.append(evaluate_result)            


    # train_label


    for j in range(len(row['ClassId'])):
        id_temp = id_temp + 1
        annotation = {}
        annotation['id'] = id_temp
        annotation['image_id'] = image_unique_dict[row.name] # int  
        annotation['category_id'] = int(row['ClassId'][j]) # numpy.int64 -> int type(train_df.iloc[0]['ClassId'])
        sub_mask = np.full(row['Height']*row['Width'], 0, dtype=np.uint8)        
        fashion_rle = [int(x) for x in row["EncodedPixels"][j].split(' ')]
        for i, start_pixel in enumerate(fashion_rle[::2]):
            sub_mask[start_pixel: start_pixel+fashion_rle[2*i+1]] = 1      
        sub_mask = sub_mask.reshape((row['Height'], row['Width']), order='F')
        rle = maskUtils.encode(sub_mask.astype(np.uint8)) # np.asfortranarray(mask)         
        annotation['segmentation'] = rle            
        annotation['iscrowd'] = 0 #  1
        annotation['area'] = maskUtils.area(rle)
        annotations.append(annotation)            

        image = {}
        image['id'] = image_unique_dict[row.name] # int  
        image['width'] = row['Width']
        image["height"] = row['Height']
        image['file_name'] = row.name
        images.append(image)


        fashion_annotations_coco_predict = {
            "image_id": image_unique_dict[row.name],
            "category_id": int(row["ClassId"][j]),
#                 "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
            "score": 1,
            "segmentation": rle 
#                 "area"
        }
        fashion_annotations_coco_predicts.append(fashion_annotations_coco_predict)            


for cat in label_desc.get('categories'):
#     {'id': 0, 'name': 'shirt, blouse', 'supercategory': 'upperbody', 'level': 2}
#     category_map[cat.get('id')] = cat.get('name')
    category = {}
    category['id'] = cat['id']
    category['name'] = cat['name']
#     category['supercategory'] = cat['supercategory']    
    categories.append(category)


fashion_annotations = {}
fashion_annotations['images'] = images
fashion_annotations['annotations'] = annotations
fashion_annotations['categories'] = categories             




# TypeError: Object of type 'bytes' is not JSON serializable
# https://blog.csdn.net/bear_sun/article/details/79397155
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() 
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8');
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)




get_ipython().system(' mkdir -p /kaggle/working/kaggle/')
get_ipython().system('touch /kaggle/working/kaggle/fashion_annotations.json')
with open('/kaggle/working/kaggle/fashion_annotations.json',"w") as dump_f:
    json.dump(fashion_annotations,dump_f, cls=MyEncoder) 




get_ipython().system('mkdir -p /kaggle/working/kaggle/')
get_ipython().system('touch /kaggle/working/kaggle/evaluate_train_results.json')
with open('/kaggle/working/kaggle/evaluate_train_results.json',"w") as dump_f2:
    json.dump(evaluate_train_results,dump_f2, cls=MyEncoder) 




# fashion_annotations_coco_predicts
get_ipython().system('mkdir -p /kaggle/working/kaggle/')
get_ipython().system('touch /kaggle/working/kaggle/fashion_annotations_coco_predicts.json')
with open('/kaggle/working/kaggle/fashion_annotations_coco_predicts.json',"w") as dump_f3:
    json.dump(fashion_annotations_coco_predicts,dump_f3, cls=MyEncoder) 




# %%python2

# !pip install matplotlib

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval
# class FashionCOCO(COCO):
#     def __init__(self, df):
#         # TODO df->dict

#         self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
#         self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)            
#         self.dataset = dataset # dict
#         self.createIndex()   

annFile = '/kaggle/working/kaggle/fashion_annotations.json'
cocoGt=COCO(annFile)
resTrain = '/kaggle/working/kaggle/fashion_annotations_coco_predicts.json'
cocoDt=cocoGt.loadRes(resTrain)
cocoEval = COCOeval(cocoGt,cocoDt, iouType='segm')
# cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()




annFile = '/kaggle/working/kaggle/fashion_annotations.json'
cocoGt=COCO(annFile)
resTrain = '/kaggle/working/kaggle/evaluate_train_results.json'
cocoDt=cocoGt.loadRes(resTrain)
cocoEval = COCOeval(cocoGt,cocoDt, iouType='segm')
# cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()




def rle_decode(rle_str, mask_shape, mask_dtype):
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(mask_shape[::-1]).T

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)




IMAGE_SIZE = 400

if VISUAL:
    label_names = ['bg']
    label_names.extend(categories_df['name'].values)
    for i in range(3):

        image_id = image_df.iloc[i].name
        image_path = "{}/train/{}.jpg".format(DATA_DIR,image_id)
        print(image_path)

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
        
        
        result = model.detect([img])
        r = result[0]

        if r['masks'].size > 0:

            masks, rois = refine_masks(r['masks'])
        else:
            masks, rois = r['masks'], r['rois']


        visualize.display_instances(img, rois, masks, r['class_ids'], 
                                    label_names, r['scores'],
                                    title=image_id, figsize=(12, 12))  
        
        
        
        ground_truth = image_df.iloc[i]        
        true_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE,len(ground_truth['ClassId'])), dtype=np.uint8)


        for m, (annotation, label) in enumerate(zip(ground_truth['EncodedPixels'], ground_truth['ClassId'])):
            sub_mask = np.full(ground_truth['Height']*ground_truth['Width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]

            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel+annotation[2*i+1]] = 1

            sub_mask = sub_mask.reshape((ground_truth['Height'], ground_truth['Width']), order='F')

            sub_mask = cv2.resize(sub_mask.astype('uint8'), 
                                            (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            true_mask[:, :, m] = sub_mask
  
        true_masks, true_rois = refine_masks(true_mask)       
    
        true_class_ids = ground_truth['ClassId']      
        true_class_ids = np.array(true_class_ids)+1
        true_scores = np.ones(len(ground_truth['ClassId']))
        
        visualize.display_instances(img, true_rois, true_masks, true_class_ids, 
                                    label_names, true_scores,
                                    title=image_id+"_true", figsize=(12, 12))          
        print(true_class_ids)
        print(r['class_ids'])






