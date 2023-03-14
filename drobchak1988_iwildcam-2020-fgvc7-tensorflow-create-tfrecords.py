#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import os, sys, math, json
from matplotlib import pyplot as plt
from PIL import Image
if 'google.colab' in sys.modules:
    get_ipython().run_line_magic('tensorflow_version', '2.x')
import tensorflow as tf
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE 


# In[2]:


with open('../input/iwildcam-2020-fgvc7/iwildcam2020_megadetector_results.json', encoding='utf-8') as json_file:
    megadetector_results =json.load(json_file)
detect_df     = pd.DataFrame(megadetector_results["images"])
# detect_df     = detect_df.loc[detect_df.max_detection_conf > 0.6].reset_index(drop=True)
# ids           = detect_df['id'].map(lambda x: '/kaggle/input/iwildcam-2020-fgvc7/train/' + x + '.jpg')
# TRAIN_PATTERN = ids.tolist()


# In[3]:


with open('/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json') as json_file:
    train_annotations_json = json.load(json_file)
df_anot= pd.DataFrame(train_annotations_json["annotations"])


# In[4]:


TRAIN_PATTERN = '/kaggle/input/iwildcam-2020-fgvc7/train/*.jpg'
TARGET_SIZE = [1980,1080]


# In[5]:


def get_cord(detect_list):
    h = TARGET_SIZE[1]
    w = TARGET_SIZE[0]
    batch_detections = []
    for detect in detect_list:
        x1, y1,w_box, h_box = detect["bbox"]
        ymin,xmin,ymax,xmax = y1, x1, y1 + h_box, x1 + w_box
        (yminn, xminn, ymaxx, xmaxx) = (ymin * h, xmin * w, (ymax * h) - (ymin * h), (xmax*w) - (xmin*w))
        batch_detections += [[yminn, xminn, ymaxx, xmaxx]]
    return batch_detections


# In[6]:


def recompress_image(image):
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)
    return image
    
def decode_jpeg_and_label(filename):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits)
    label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')
    label = label.values[-1]
    return image, label

def resize_and_crop_image(image, label):
    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw = TARGET_SIZE[1]
    th = TARGET_SIZE[0]
    resize_crit = (w * th) / (h * tw)
    image = tf.cond(resize_crit < 1,
                    lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
                    lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
                   )
    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    return image, label


# In[7]:


filenames = tf.data.Dataset.list_files(TRAIN_PATTERN) # This
dataset = filenames.map(decode_jpeg_and_label, num_parallel_calls=AUTO)
dataset = dataset.map(resize_and_crop_image, num_parallel_calls=AUTO) 
# dataset = dataset.repeat()
dataset = dataset.prefetch(AUTO)


# In[8]:


def get_draw_boxes(img, lbl):
    images_batch = []
    im_id = lbl.numpy().decode("utf-8").split('.')[0].split('/')[-1]
    boxxs = get_cord(detect_df.loc[detect_df.id == im_id]['detections'].values[0])
    category_id = df_anot.loc[df_anot.image_id == im_id]['category_id'].values[0]
    w = tf.shape(img)[0]
    h = tf.shape(img)[1]
    for boxx in boxxs:
        yminn, xminn, ymaxx, xmaxx = boxx
        image = tf.image.crop_to_bounding_box(img, int(yminn), int(xminn), int(ymaxx), int(xmaxx))
        nw = tf.shape(image)[0]
        nh = tf.shape(image)[1]
        if nw < nh*2 and nh < nw*2:
            image = recompress_image(image)
            images_batch.append({"image": image.numpy(), "label": category_id, "im_id": str.encode(im_id), "width": nw, "height": nh })
    return images_batch


# In[9]:


get_ipython().system('mkdir tfrecords')


# In[10]:


CLASSES = [i for i in range(572)]

def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints): # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats): # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))
  

def to_tfrecord(tfrec_filewriter, img_bytes, label, iage_id, width, height):
    one_hot_class = np.eye(len(CLASSES))[label]

    feature = {
      "image": _bytestring_feature([img_bytes]),
      "class": _int_feature([label]),
      "iage_id":  _bytestring_feature([iage_id]),
      "label": _float_feature(one_hot_class.tolist()),
      "size":  _int_feature([width, height])
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

print("Writing TFRecords")
tfrecord_size = 50 #  you can to set more size
tfrecord_counter = 0
batch_counter = 0
batch = []
balance = []

for counter, (image, label) in enumerate(dataset):
    pack = get_draw_boxes(image, label)
    batch_counter += len(pack)

    if len(balance) > 0:
        batch_counter += len(balance)
        for item in balance:
            batch += [item]
        balance = []

    if len(pack) > 0:
        for item in pack:
            batch += [item]
    if tfrecord_counter > 1: # you need to delete this string
        break
    if batch_counter >= tfrecord_size:
        filename = './tfrecords/' + "{:02d}-{}.tfrec".format(tfrecord_counter, tfrecord_size)
        tfrecord_counter+=1
        with tf.io.TFRecordWriter(filename) as out_file:
            balance = batch[tfrecord_size:]
            batch = batch[:tfrecord_size]
            for record_item in batch:
                example = to_tfrecord(out_file,
                                      record_item["image"],
                                      record_item["label"],
                                      record_item["im_id"],
                                      record_item["width"],
                                      record_item["height"])
                out_file.write(example.SerializeToString())
            batch_counter = 0
            batch = []
            print("Wrote file {} containing {} records".format(filename, tfrecord_size))


# In[11]:


def read_tfrecord(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        "class": tf.io.FixedLenFeature([], tf.int64), 
        "iage_id": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.VarLenFeature(tf.float32) ,
        "size": tf.io.FixedLenFeature([2], tf.int64) 
    }
    example = tf.io.parse_single_example(example, features)
    width = example['size'][0]
    height  = example['size'][1]
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.reshape(image, [width,height, 3])
    iage_id = example['iage_id']
    class_num = example['class']
    label = tf.sparse.to_dense(example['label'])

    return image, class_num, label, iage_id

option_no_order = tf.data.Options()
option_no_order.experimental_deterministic = False

filenames = tf.io.gfile.glob('/kaggle/working/tfrecords/' + "*.tfrec")
dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
dataset = dataset.with_options(option_no_order)
dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
dataset = dataset.shuffle(300)


# In[12]:


def display_9_images_from_dataset(dataset):
    plt.figure(figsize=(13,13))
    subplot=331
    for i, (image, label) in enumerate(dataset):
        plt.subplot(subplot)
        plt.axis('off')
        plt.imshow(image.numpy().astype(np.uint8))
        plt.title(label.numpy().decode("utf-8"), fontsize=16)
        subplot += 1
        if i==8:
            break
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


# In[13]:


display_dataset = dataset.map(lambda image, class_num, label, iage_id: (image, iage_id))
display_9_images_from_dataset(display_dataset)

