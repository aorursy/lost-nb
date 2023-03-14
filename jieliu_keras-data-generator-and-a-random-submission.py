#!/usr/bin/env python
# coding: utf-8



import os 
from glob import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

import matplotlib.pyplot as plt

from keras.backend import set_image_dim_ordering, image_dim_ordering

set_image_dim_ordering('th')
print("Image dim ordering : ", image_dim_ordering())




TRAIN_DATA = "../input/train"
TEST_DATA = "../input/test"
ADDITIONAL_DATA = "../input/additional"


type_1_files = glob(os.path.join(TRAIN_DATA, "Type_1", "*.jpg"))
type_1_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_1"))+1:-4] for s in type_1_files])
type_2_files = glob(os.path.join(TRAIN_DATA, "Type_2", "*.jpg"))
type_2_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_2"))+1:-4] for s in type_2_files])
type_3_files = glob(os.path.join(TRAIN_DATA, "Type_3", "*.jpg"))
type_3_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_3"))+1:-4] for s in type_3_files])

additional_type_1_files = glob(os.path.join(ADDITIONAL_DATA, "Type_1", "*.jpg"))
additional_type_1_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "Type_1"))+1:-4] for s in additional_type_1_files])
additional_type_2_files = glob(os.path.join(ADDITIONAL_DATA, "Type_2", "*.jpg"))
additional_type_2_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "Type_2"))+1:-4] for s in additional_type_2_files])
additional_type_3_files = glob(os.path.join(ADDITIONAL_DATA, "Type_3", "*.jpg"))
additional_type_3_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "Type_3"))+1:-4] for s in additional_type_3_files])

test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = np.array([s[len(TEST_DATA)+1:-4] for s in test_files])




def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type   
    """
    if image_type == "Type_1" or         image_type == "Type_2" or         image_type == "Type_3":
        data_path = os.path.join(TRAIN_DATA, image_type)
    elif image_type == "Test":
        data_path = TEST_DATA
    elif image_type == "AType_1" or           image_type == "AType_2" or           image_type == "AType_3":
        data_path = os.path.join(ADDITIONAL_DATA, image_type)
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    ext = 'jpg'
    return os.path.join(data_path, "{}.{}".format(image_id, ext))


def get_image_data(image_id, image_type):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img




type_to_index = {
    "Type_1": 0,
    "Type_2": 1,
    "Type_3": 2,
}


def data_iterator(image_id_type_list, batch_size, image_size, verbose=0, test_mode=False):
    
    while True:
        X = np.zeros((batch_size, 3) + image_size, dtype=np.float32)
        Y = np.zeros((batch_size, 3), dtype=np.uint8)
        image_ids = np.empty((batch_size,), dtype=np.object)
        counter = 0
        for i, (image_id, image_type) in enumerate(image_id_type_list):
            
            img = get_image_data(image_id, image_type)
            img = cv2.resize(img, dsize=image_size[::-1])
            img = img.transpose([2,0,1])
            img = img.astype(np.float32) / 255.0
                        
            X[counter, :, :, :] = img            
            if test_mode:
                image_ids[counter] = image_id
            else:
                Y[counter, type_to_index[image_type]] = 1    
                
            if verbose > 0:
                print("Image id/type:", image_id, image_type)
            
            counter += 1
            if counter == batch_size:
                yield (X, Y) if not test_mode else (X, Y, image_ids)
                X = np.zeros((batch_size, 3) + image_size, dtype=np.float32)
                Y = np.zeros((batch_size, 3), dtype=np.uint8)
                image_ids = np.empty((batch_size,), dtype=np.object)
                counter = 0
        
        if counter > 0:
            X = X[:counter,:,:,:]
            Y = Y[:counter,:]
            image_ids = image_ids[:counter]
            yield (X, Y) if not test_mode else (X, Y, image_ids)
            
        if test_mode:
            break




val_split=0.3
type_ids = [type_1_ids, type_2_ids, type_3_ids]
image_types = ["Type_1", "Type_2", "Type_3"]
train_ll = [int(len(ids) * (1.0 - val_split)) for ids in type_ids]
val_ll = [int(len(ids) * (val_split)) for ids in type_ids]


count = 0
train_id_type_list = []
train_ids = [ids[:l] for ids, l in zip(type_ids, train_ll)]
max_size = max(train_ll)
while count < max_size:    
    for l, ids, image_type in zip(train_ll, train_ids, image_types):    
        image_id = ids[count % l]
        train_id_type_list.append((image_id, image_type))
    count += 1
   

count = 0
val_id_type_list = []
val_ids = [ids[tl:tl+vl] for ids, tl, vl in zip(type_ids, train_ll, val_ll)]
max_size = max(val_ll)
while count < max_size:    
    for l, ids, image_type in zip(val_ll, val_ids, image_types):    
        image_id = ids[count % l]
        val_id_type_list.append((image_id, image_type))
    count += 1

assert len(set(train_id_type_list) & set(val_id_type_list)) == 0, "WTF" 

    
print("Train dataset contains : ")
print("-", train_ll, " images of corresponding types")
print("Validation dataset contains : ")
print("-", val_ll, " images of corresponding types")




image_size = (224, 224)
batch_size = 15
train_iter = data_iterator(train_id_type_list, batch_size=batch_size, image_size=image_size, verbose=1)




for X, Y in train_iter:
    print(X.shape, X.dtype, Y.shape)
    n = 5
    for counter in range(batch_size):
        if counter % n == 0:
            plt.figure(figsize=(12, 4))
        plt.subplot(1, n, counter % n + 1)
        plt.imshow(X[counter, :, :, :].transpose([1, 2, 0]))
        plt.title("Type : {}".format(Y[counter,:]))
        plt.axis('off')
    
    break




image_size = (224, 224)
batch_size = 15
val_iter = data_iterator(val_id_type_list, batch_size=batch_size, image_size=image_size, verbose=1)




for X, Y in val_iter:
    print(X.shape, X.dtype, Y.shape)
    n = 5
    for counter in range(batch_size):
        if counter % n == 0:
            plt.figure(figsize=(12, 4))
        plt.subplot(1, n, counter % n + 1)
        plt.imshow(X[counter, :, :, :].transpose([1, 2, 0]))
        plt.title("Type : {}".format(Y[counter,:]))
        plt.axis('off')
    
    break









from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten
from keras.models import Model

image_size = (224, 224)

base_model = ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(3,) + image_size)
x = Flatten()(base_model.output)
output = Dense(3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
model.summary()




model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy',])




def logloss_mc(y_true, y_prob, epsilon=1e-15):
    """ Multiclass logloss
    This function is not officially provided by Kaggle, so there is no
    guarantee for its correctness.
    https://github.com/ottogroup/kaggle/blob/master/benchmark.py
    """
    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll




batch_size = 16
val_iter = data_iterator(val_id_type_list, batch_size=batch_size, image_size=image_size, test_mode=True)

total_loss = 0.0
total_counter = 0 
for X, Y_true, _ in val_iter:            
    s = Y_true.shape[0]
    total_counter += s
    #Y_pred = model.predict(X)
    Y_pred = 0.33333 * np.ones_like(Y_true)
    loss = logloss_mc(Y_true, Y_pred)
    print("--", total_counter, "batch loss : ", loss)
    total_loss += s * loss

total_loss *= 1.0 / total_counter   
print("Total loss : ", total_loss)




df = pd.DataFrame(columns=['image_name','Type_1','Type_2','Type_3'])
def get_test_id_type_list():
    return [(image_id, 'Test') for image_id in test_ids]

image_size = (224, 224)
batch_size = 16
test_id_type_list = get_test_id_type_list()
test_iter = data_iterator(test_id_type_list, batch_size=batch_size, image_size=image_size, test_mode=True)


df = pd.DataFrame(columns=['image_name','Type_1','Type_2','Type_3'])
total_counter = 0
for X, _, image_ids in test_iter:            
    #Y_pred = model.predict(X)    
    s = X.shape[0]
    total_counter += s
    Y_pred = 0.33333 * np.ones((s, 3))
    print("--", total_counter, image_ids)
    for i in range(s):
        df.loc[total_counter + i, :] = (image_ids[i] + '.jpg', ) + tuple(Y_pred[i, :])




print(df.shape)
df.head()




import datetime
now = datetime.datetime.now()
info = 'random_predictions'
sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
df.to_csv(sub_file, index=False)




get_ipython().system('ls ')




batch_size=16
samples_per_epoch = 512
nb_val_samples = 64

print(batch_size, samples_per_epoch, nb_val_samples)

train_iter = data_iterator(train_id_type_list, batch_size=batch_size, image_size=image_size)
val_iter = data_iterator(val_id_type_list, batch_size=batch_size, image_size=image_size)

#history = model.fit_generator(
#    train_iter,
#    steps_per_epoch=samples_per_epoch, 
#    epochs=1,
#    validation_data=val_iter,
#    validation_steps=nb_val_samples,
#    verbose=1
#)




type_3_ids






