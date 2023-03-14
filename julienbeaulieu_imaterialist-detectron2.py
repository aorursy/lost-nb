#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html ')


# In[2]:


get_ipython().system("pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'")


# In[3]:


get_ipython().system('python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html')


# In[4]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))


# In[5]:


import collections
import torch
import json
import os
import cv2
import random
import gc
import pycocotools
import torch.nn.functional as F

import numpy as np 
import pandas as pd 
from tqdm import tqdm
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from pathlib import Path

# import some common detectron2 utilities
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# In[6]:


cfg = get_cfg()
# Merging model configs with default 
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Get weights from Instance segmentation Mask RCNN R 50 FPN model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

predictor = DefaultPredictor(cfg)


# In[7]:


images_list = []
for dirname, _, filenames in os.walk('/kaggle/input/imaterialist-fashion-2020-fgvc7/train/'):
    for filename in filenames:
        images_list.append(os.path.join(dirname, filename))


# In[8]:


# Show different images at random
rows, cols = 3, 3
plt.figure(figsize=(20,20))

for i, image in enumerate(random.sample(images_list, 9)):
    
    # Process image
    im = cv2.imread(image)
    plt.subplot(rows, cols, i+1)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # Run through predictor
    outputs = predictor(im)
    
    # Visualize
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.axis('off')
    plt.imshow(v.get_image()[:, :, ::-1])

plt.show()


# In[9]:


cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml"))

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Get weights from Keypoint Mask RCNN R 50 FPN model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml")

predictor = DefaultPredictor(cfg)


# In[10]:


# Show different images at random
rows, cols = 3, 4
plt.figure(figsize=(20,30))

for i, image in enumerate(random.sample(images_list, 12)):
    
    # Process image
    im = cv2.imread(image)
    plt.subplot(cols, rows, i+1)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # Run through predictor
    outputs = predictor(im)
    
    # Visualize
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.axis('off')
    plt.imshow(v.get_image()[:, :, ::-1])

plt.show()


# In[11]:


data_dir = Path('/kaggle/input/imaterialist-fashion-2020-fgvc7/')
image_dir = Path('/kaggle/input/imaterialist-fashion-2020-fgvc7/train/')
df = pd.read_csv(data_dir/'train.csv')

# Get label descriptions
with open(data_dir/'label_descriptions.json', 'r') as file:
    label_desc = json.load(file)
df_categories = pd.DataFrame(label_desc['categories'])
df_attributes = pd.DataFrame(label_desc['attributes'])


# In[12]:


# Rle helper functions

def rle_decode_string(string, h, w):
    mask = np.full(h*w, 0, dtype=np.uint8)
    annotation = [int(x) for x in string.split(' ')]
    for i, start_pixel in enumerate(annotation[::2]):
        mask[start_pixel: start_pixel+annotation[2*i+1]] = 1
    mask = mask.reshape((h, w), order='F')

    
    return mask

def rle2bbox(rle, shape):
    '''
    Get a bbox from a mask which is required for Detectron 2 dataset
    rle: run-length encoded image mask, as string
    shape: (height, width) of image on which RLE was produced
    Returns (x0, y0, x1, y1) tuple describing the bounding box of the rle mask
    
    Note on image vs np.array dimensions:
    
        np.array implies the `[y, x]` indexing order in terms of image dimensions,
        so the variable on `shape[0]` is `y`, and the variable on the `shape[1]` is `x`,
        hence the result would be correct (x0,y0,x1,y1) in terms of image dimensions
        for RLE-encoded indices of np.array (which are produced by widely used kernels
        and are used in most kaggle competitions datasets)
    '''
    
    a = np.fromiter(rle.split(), dtype=np.uint)
    a = a.reshape((-1, 2))  # an array of (start, length) pairs
    a[:,0] -= 1  # `start` is 1-indexed
    
    y0 = a[:,0] % shape[0]
    y1 = y0 + a[:,1]
    if np.any(y1 > shape[0]):
        # got `y` overrun, meaning that there are a pixels in mask on 0 and shape[0] position
        y0 = 0
        y1 = shape[0]
    else:
        y0 = np.min(y0)
        y1 = np.max(y1)
    
    x0 = a[:,0] // shape[0]
    x1 = (a[:,0] + a[:,1]) // shape[0]
    x0 = np.min(x0)
    x1 = np.max(x1)
    
    if x1 > shape[1]:
        # just went out of the image dimensions
        raise ValueError("invalid RLE or image dimensions: x1=%d > shape[1]=%d" % (
            x1, shape[1]
        ))

    return x0, y0, x1, y1


# In[13]:


# Get image file path and add it to our dataframe
dirname = '/kaggle/input/imaterialist-fashion-2020-fgvc7/train/'
df_copy = df[:4000].copy()
df_copy['ImageId'] = dirname + df_copy['ImageId'] + '.jpg'


# In[14]:


# Get bboxes for each mask
bboxes = [rle2bbox(c.EncodedPixels, (c.Height, c.Width)) for n, c in df_copy.iterrows()]


# In[15]:


assert len(bboxes) == df_copy.shape[0]


# In[16]:


bboxes_array = np.array(bboxes)


# In[17]:


# Add each coordinate as a column
df_copy['x0'], df_copy['y0'], df_copy['x1'], df_copy['y1'] = bboxes_array[:,0], bboxes_array[:,1], bboxes_array[:,2], bboxes_array[:,3]


# In[18]:


#Replace NaNs from AttributeIds by -1
df_copy = df_copy.fillna(999)


# In[19]:


# Extremely ugly function - need to refactor
def attr_str_to_list(df):
    '''
    Function that transforms DataFrame AttributeIds which are of type string into a 
    list of integers. Strings must be converted because they cannot be transformed into Tensors
    '''
    # cycle through all the non NaN rows - NaN causes an error
    for index, row in df.iterrows():
        
        # Treating str differently than int
        if isinstance(row['AttributesIds'], str):
            
            # Convert each row's string into a list of strings             
            df['AttributesIds'][index] = row['AttributesIds'].split(',')
            
            # Convert each string in the list to int
            df['AttributesIds'][index] = [int(x) for x in df['AttributesIds'][index]]
            
        # If int - make it a list of length 1
        if isinstance(row['AttributesIds'], int):
            df['AttributesIds'][index] = [999]
            
        # Convert list to array
        df['AttributesIds'][index] = np.array(df['AttributesIds'][index])

        # Pad array with 0's so that all arrays are the same length - This will allows us to convert to tensor
        df['AttributesIds'][index] = np.pad(df['AttributesIds'][index], (0, 14-len(df['AttributesIds'][index]))) 
       
attr_str_to_list(df_copy) 


# In[20]:


df_copy.sample(5)


# In[21]:


del bboxes
gc.collect()


# In[22]:


# https://detectron2.readthedocs.io/tutorials/datasets.html
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5

from detectron2.structures import BoxMode
import pycocotools

def get_fashion_dict(df):
    
    dataset_dicts = []
    
    for idx, filename in enumerate(df['ImageId'].unique().tolist()):
        
        record = {}
        
        # Convert to int otherwise evaluation will throw an error
        record['height'] = int(df[df['ImageId']==filename]['Height'].values[0])
        record['width'] = int(df[df['ImageId']==filename]['Width'].values[0])
        
        record['file_name'] = filename
        record['image_id'] = idx
        
        objs = []
        for index, row in df[(df['ImageId']==filename)].iterrows():
            
            # Get binary mask
            mask = rle_decode_string(row['EncodedPixels'], row['Height'], row['Width'])
            
            # opencv 4.2+
            # Transform the mask from binary to polygon format
            contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)
            
            # opencv 3.2
            # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
            #                                            cv2.CHAIN_APPROX_SIMPLE)
            
            segmentation = []

            for contour in contours:
                contour = contour.flatten().tolist()
                # segmentation.append(contour)
                if len(contour) > 4:
                    segmentation.append(contour)    
            
            # Data for each mask
            obj = {
                'bbox': [row['x0'], row['y0'], row['x1'], row['y1']],
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': row['ClassId'],
                'attributes': row['AttributesIds'],
                'segmentation': segmentation,
                'iscrowd': 0
            }
            objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts


# In[23]:


# To view a sample of fashion_dict
fashion_dict = get_fashion_dict(df_copy[:50])


# In[24]:


fashion_dict[0]


# In[25]:


from typing import Iterator, List, Tuple, Union
import torch.nn.functional as F

# Base Attribute holder
class Attributes:
    """
    This structure stores a list of attributes as a Nx14 torch.Tensor (14 because we added a padding of 14 to all our attributes
    so that they can have the same length.
    It behaves like a Tensor
    (support indexing, `to(device)`, `.device`, `non empty`, and iteration over all attributes)
    """
    
    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx13 matrix.  Each row is [attribute_1, attribute_2, ...].
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.int64, device=device)
        assert tensor.dim() == 2, tensor.size()

        self.tensor = tensor


    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Boxes":
        """
        Returns:
            Attributes: Create a new :class:`Attributes` by indexing.
        The following usage are allowed:
        1. `new_attributes = attributes[3]`: return a `Attributes` which contains only one Attribute.
        2. `new_attributes = attributes[2:10]`: return a slice of attributes.
        3. `new_attributes = attributes[vector]`, where vector is a torch.BoolTensor
           with `length = len(attributes)`. Nonzero elements in the vector will be selected.
        Note that the returned Attributes might share storage with this Attributes,
        subject to Pytorch's indexing semantics.
        """
        
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Attributes with {} failed to return a matrix!".format(item)
        return Attributes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]
    
    def to(self, device: str) -> "Attributes":
        return Attributes(self.tensor.to(device))


    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find attributes that are non-empty.
        An attribute is considered empty if its first attribute in the list is 999.
        Returns:
            Tensor:
                a binary vector which represents whether each attribute is empty
                (False) or non-empty (True).
        """
        attributes = self.tensor
        first_attr = attributes[:, 0]
        keep = (first_attr != 999)
        return keep

    def __repr__(self) -> str:
        return "Attributes(" + str(self.tensor) + ")"


    def remove_padding(self, attribute):
        'WIP'
        pass

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Yield attributes as a Tensor of shape (14,) at a time.
        """
        yield from self.tensor


# In[26]:


import copy
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is a custom version of the DatasetMapper. The only different with Detectron2's 
    DatasetMapper is that we extract attributes from our dataset_dict. 
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()           
            
            ####################################
            # New: Get attributes from annos 
            ####################################
            if len(annos) and 'attributes' in annos[0]:
    
                # get a list of list of attributes
                gt_attributes = [x['attributes'] for x in annos]
                
                # Put attributes in Attributes class holder and add them to instances
                instances.gt_attributes = Attributes(gt_attributes)
                
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict


# In[27]:


from detectron2.engine import DefaultTrainer

class FashionTrainer(DefaultTrainer):
    'A customized version of DefaultTrainer. We add the mapping `DatasetMapper` to the dataloader.'
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg))
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg))


# In[28]:


# Get a sample of the training data to run experiments
df_copy_train = df_copy[:3000].copy()
df_copy_test = df_copy[-1000:].copy()


# In[29]:


from detectron2.data import DatasetCatalog, MetadataCatalog

# Register the train set metadata
for d in ['train']:
   DatasetCatalog.register('1sample_fashion_' + d, lambda d=df_copy_train: get_fashion_dict(d))
   MetadataCatalog.get("1sample_fashion_" + d).set(thing_classes=list(df_categories.name))
fashion_metadata = MetadataCatalog.get("1sample_fashion_train")


# In[30]:


# Register the test and set metadata
for d in ['test']:
    DatasetCatalog.register('sample_fashion_' + d, lambda d=df_copy_test: get_fashion_dict(d))
    MetadataCatalog.get("sample_fashion_" + d).set(thing_classes=list(df_categories.name))
fashion_metadata = MetadataCatalog.get("sample_fashion_test")


# In[31]:


# View some images + masks from the dataset
import random
for d in random.sample(fashion_dict, 3):
    plt.figure(figsize=(10,10))
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], fashion_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    plt.imshow(vis.get_image()[:, :, ::-1])


# In[32]:


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("1sample_fashion_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

##### Input #####
# Set a smaller image size than default to avoid memory problems

# Size of the smallest side of the image during training
cfg.INPUT.MIN_SIZE_TRAIN = (40,)
# Maximum size of the side of the image during training
cfg.INPUT.MAX_SIZE_TRAIN = 60
# Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
cfg.INPUT.MIN_SIZE_TEST = 40
# Maximum size of the side of the image during testing
cfg.INPUT.MAX_SIZE_TEST = 60

# Mask type
#cfg.INPUT.MASK_FORMAT = "bitmask"  # default: "polygon"


cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025  
cfg.SOLVER.MAX_ITER = 50 # not enough iterations for real training, but this is just a demo    
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # default: 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 46  # 46 classes in iMaterialist


# In[33]:


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


# In[34]:


#trainer = DefaultTrainer(cfg) 
trainer = FashionTrainer(cfg) 


# In[35]:


trainer.resume_or_load(resume=False)


# In[36]:


trainer.train()


# In[37]:


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ('sample_fashion_test',)
predictor = DefaultPredictor(cfg)


# In[38]:


from detectron2.utils.visualizer import ColorMode
plt.figure(figsize=(20,20))
for d in random.sample(fashion_dict, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=fashion_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(vis.get_image()[:, :, ::-1])


# In[39]:


# Show different images at random
rows, cols = 3, 3
plt.figure(figsize=(20,20))

for i, d in enumerate(random.sample(fashion_dict, 9)):
    
    # Process image
    plt.subplot(rows, cols, i+1)

    im = cv2.imread(d["file_name"])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # Run through predictor
    outputs = predictor(im)
    
    # Visualize
    v = Visualizer(im[:, :, ::-1],
                   metadata=fashion_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image()[:, :, ::-1])

plt.show()

