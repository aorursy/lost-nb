#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import collections
import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




get_ipython().system('pip install dicom')
get_ipython().system('pip install imageio')
import pydicom
import imageio
from time import sleep
from tqdm.auto import tqdm, trange
import cv2




os.chdir('/kaggle/working/')




get_ipython().system('pip3 install -U scikit-image')
get_ipython().system('pip3 install -U cython ')
get_ipython().system('pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"')




get_ipython().run_cell_magic('bash', '', 'git clone https://github.com/pytorch/vision.git\ncd vision\ngit checkout v0.3.0\ncp references/detection/utils.py ../\ncp references/detection/transforms.py ../\ncp references/detection/coco_eval.py ../\ncp references/detection/engine.py ../\ncp references/detection/coco_utils.py ../')




import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import ImageDraw
from torch.utils.data import random_split, DataLoader
from torch import tensor
from torchvision.utils import make_grid
import torchvision
from engine import train_one_epoch, evaluate
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import utils
import matplotlib.gridspec as gridspec


get_ipython().run_line_magic('matplotlib', 'inline')




train_path = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/'
test_path = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_test_images/'
path = '/kaggle/input/rsna-pneumonia-detection-challenge'

os.mkdir('/kaggle/working/train_pngs')
os.mkdir('/kaggle/working/test_pngs')
os.mkdir('/kaggle/working/train_labels')

os.chdir(path)




labels_csv = pd.read_csv(os.path.join(path,'stage_2_train_labels.csv'))

labels_csv.x.fillna(0, inplace=True)
labels_csv.y.fillna(0, inplace=True)
labels_csv.width.fillna(1023, inplace=True)
labels_csv.height.fillna(1023, inplace=True)

labels_csv['x_max'] = labels_csv['x']+labels_csv['width']
labels_csv['y_max'] = labels_csv['y']+labels_csv['height']

labels_csv.head()




def parse_one_annot(box_coord, filename):
   boxes_array = box_coord[box_coord["patientId"] == filename][["x", "y",        
   "x_max", "y_max"]].values
   
   return boxes_array

def dicom_to_png(image_path):
    dcm_data = pydicom.read_file(image_path)
    im = dcm_data.pixel_array
    return im

class RSNA(torch.utils.data.Dataset):
    def __init__(self, path, box_coord, transforms=None):
        self.path = path
        self.box_coord = box_coord
        self.transforms = transforms
        self.imgs = sorted(os.listdir(path))

    def __getitem__(self, idx):
        # load images and bounding boxes
        img_path = os.path.join(self.path, self.box_coord['patientId'][idx]+'.dcm')
        img = dicom_to_png(img_path)
        #img = Image.open(img_path).convert("RGB")
        #img = img.resize((1024, 1024))
        box_list = parse_one_annot(self.box_coord, 
        self.box_coord['patientId'][idx])
        boxes = torch.as_tensor(box_list, dtype=torch.float32)
        num_objs = len(box_list)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:,
        0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
                img = self.transforms(img)
        return img, target
    def __len__(self):
          return len(self.box_coord['patientId'])




def train_tfms():
   transforms = []
   # converts the image, a PIL image, into a PyTorch Tensor
   transforms.append(T.ToTensor())
   #if train:
      # during training, randomly flip the training images
      # and ground-truth for data augmentation
   #   transforms.append(T.RandomHorizontalFlip(0.5))
   return T.Compose(transforms)

def val_tfms():
   transforms = []
   # converts the image, a PIL image, into a PyTorch Tensor
   transforms.append(T.ToTensor())
   #if train:
      # during training, randomly flip the training images
      # and ground-truth for data augmentation
   #   transforms.append(T.RandomHorizontalFlip(0.5))
   return T.Compose(transforms)




np.random.seed(42)
msk = np.random.rand(len(labels_csv)) < 0.9

train_df = labels_csv[msk].reset_index()
val_df = labels_csv[~msk].reset_index()
len(train_df), len(val_df)




train_ds = RSNA(train_path, train_df, transforms=train_tfms())
val_ds = RSNA(train_path, val_df, transforms=val_tfms())
#test_ds = BROID(test_path, test_df, transforms=val_tfms())
len(train_ds), len(val_ds)#, len(test_ds)




train_ds.__getitem__(0)




batch_size = 8




train_dl = DataLoader(train_ds, batch_size, shuffle=True, 
                      num_workers=3, pin_memory=True, collate_fn=utils.collate_fn)
val_dl = DataLoader(val_ds, batch_size*2, 
                    num_workers=2, pin_memory=True, collate_fn=utils.collate_fn)




def draw_bounding_box(img, label_boxes):
  #img = np.squeeze(img, axis=0)
  #label_boxes = np.squeeze(label_boxes, axis=0)
  all_imgs = []
  for i in range(img.shape[0]):        
      image = img[i,:,:,:]
      image = image.squeeze(0)
      im = Image.fromarray(image.mul(255).byte().numpy())
      draw = ImageDraw.Draw(im)
      labels = label_boxes[i]['boxes']
      for elem in range(len(labels)):
        draw.rectangle([(labels[elem][0], labels[elem][1]),
        (labels[elem][2], labels[elem][3])], 
        outline ="white", width =3)
      all_imgs.append(np.array(im))
  all_imgs = np.array(all_imgs)
  return T.ToTensor()(all_imgs)




def show_batch(dl):
    for images, labels in dl:
        image = draw_bounding_box(torch.stack(images), labels)
        image = image.permute(1,2,0).mul(255).byte().numpy()
        fig, ax = plt.subplots(figsize=(16, 16), nrows=2, ncols=3)
        gs1 = gridspec.GridSpec(3, 4)
        gs1.update(wspace=0.030, hspace=0.030) # set the spacing between axes. 
        id = 0
        for i in range(2):
            for j in range(3):
                ax[i,j].set_title('Pneumonia:')
                ax[i,j].imshow(image[id])
                id = id + 1
        
        plt.show()
        break




show_batch(train_dl)




def get_model(num_classes):
   # load an object detection model pre-trained on COCO
   model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
   # get the number of input features for the classifier
   in_features = model.roi_heads.box_predictor.cls_score.in_features
   # replace the pre-trained head with a new on
   model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
   
   return model




def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)




device = get_default_device()

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2
# get the model using our helper function
model = get_model(num_classes)
# move model to the right device
model.to(device)
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler which decreases the learning rate by # 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)




# let's train it for 10 epochs
num_epochs = 10
for epoch in range(num_epochs):
   # train for one epoch, printing every 10 iterations
   train_one_epoch(model, optimizer, train_dl, device, epoch, print_freq=10)
# update the learning rate
   lr_scheduler.step()
   # evaluate on the test dataset
   evaluate(model, val_dl, device=device)
























def parse_data(df):
    """
    Method to read a CSV file (Pandas dataframe) and parse the 
    data into the following nested dictionary:

      parsed = {
        
        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        }, ...

      }

    """
    parsed = collections.defaultdict(lambda:{'dicom': None,
                                        'png': None,     
                                        'label': None,
                                        'boxes': []})
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed 
        pid = row['patientId']
        parsed[pid]['dicom'] = str('stage_2_train_images/{id}.dcm').format(id=pid)
        parsed[pid]['png'] = str('/kaggle/working/train_pngs/{id}.png').format(id=pid)
        parsed[pid]['label'] = row['Target']
        parsed[pid]['boxes'].append(hw_bb(row))

    return parsed

def parse_data_test(test_list):
    
    parsed = collections.defaultdict(lambda:{'dicom': None,
                                        'png': None})
    for row in test_list:
        # --- Initialize patient entry into parsed 
        parsed[row]['dicom'] = str('stage_2_test_images/{id}.dcm').format(id=row)
        parsed[row]['png'] = str('/kaggle/working/test_pngs/{id}.png').format(id=row)
    
    return parsed

def hw_bb(row): return np.array([row['y'], row['x'], row['height']+row['y'], row['width']+row['x']])

def bb_hw(a): return np.array([a[1],a[0],a[3]-a[1],a[2]-a[0]])




parsed = parse_data(labels_csv)




def from_dicom_to_png(parsed):
    with tqdm(total=len(parsed)) as pbar:
        for i in range(len(parsed)):
            for k, v in parsed.items():
                dcm_data = pydicom.read_file(v['dicom'])
                im = dcm_data.pixel_array
                imageio.imwrite(v['png'], im)
                pbar.update(1)
                #sleep(0.01)




from_dicom_to_png(parsed)




parsed_test_data = parse_data_test(list(list(i[:-4] for i in os.listdir('stage_2_test_images'))))
from_dicom_to_png(parsed_test_data)




def save_img_from_dcm(dcm_dir, img_dir, patient_id):
    img_fp = os.path.join(img_dir, "{}.jpg".format(patient_id))
    if os.path.exists(img_fp):
        return
    dcm_fp = os.path.join(dcm_dir, "{}.dcm".format(patient_id))
    img_1ch = pydicom.read_file(dcm_fp).pixel_array
    img_3ch = np.stack([img_1ch]*3, -1)

    img_fp = os.path.join(img_dir, "{}.jpg".format(patient_id))
    cv2.imwrite(img_fp, img_3ch)
    
def save_label_from_dcm(label_dir, patient_id, row=None):
    # rsna defualt image size
    img_size = 1024
    label_fp = os.path.join(label_dir, "{}.txt".format(patient_id))
    
    f = open(label_fp, "a")
    if row is None:
        f.close()
        return

    top_left_x = row[1]
    top_left_y = row[2]
    w = row[3]
    h = row[4]
    
    # 'r' means relative. 'c' means center.
    rx = top_left_x/img_size
    ry = top_left_y/img_size
    rw = w/img_size
    rh = h/img_size
    rcx = rx+rw/2
    rcy = ry+rh/2
    
    line = "{} {} {} {} {}\n".format(0, rcx, rcy, rw, rh)
    
    f.write(line)
    f.close()
        
def save_yolov3_data_from_rsna(dcm_dir, img_dir, label_dir, annots):
    for row in tqdm(annots.values):
        patient_id = row[0]

        img_fp = os.path.join(img_dir, "{}.jpg".format(patient_id))
        if os.path.exists(img_fp):
            save_label_from_dcm(label_dir, patient_id, row)
            continue

        target = row[5]
        # Since kaggle kernel have samll volume (5GB ?), I didn't contain files with no bbox here.
        #if target == 0:
        #    continue
        save_label_from_dcm(label_dir, patient_id, row)
        save_img_from_dcm(dcm_dir, img_dir, patient_id)




save_yolov3_data_from_rsna(train_path, '/kaggle/working/train_pngs', '/kaggle/working/train_labels', labels_csv)




len(os.listdir('/kaggle/working/train_pngs'))




len(os.listdir('/kaggle/working/train_labels'))






