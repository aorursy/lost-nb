#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install keras==2.2.4')
get_ipython().system('pip install tensorflow==1.13.1')

import cv2
import os
import glob
import gc

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import itertools 
from tqdm import tqdm
from keras.models import Sequential
from keras import callbacks
from keras import backend as K
from keras.optimizers import Adam
from keras import initializers, layers
from keras.utils import to_categorical
from keras.models import Model, load_model
from sklearn.metrics import confusion_matrix
from keras.metrics import categorical_crossentropy
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (Dense, Input, Conv2D, Flatten, MaxPooling2D, 
                          Activation, Dropout, Average)

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)


# In[2]:


sns.set(style="darkgrid")

def utility(X, Y, size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    # extedn image axis for keras model and with normalization
    X = X.reshape(X.shape[0], size, size, 1).astype('float32')/255.
    return X, Y


def preprocess_image(image):
    img = cv2.bilateralFilter(image,9,75,75)
    
    # using sharpen kenel
    kernel_sharp = np.array([[0,-1,0], 
                             [-1,5,-1], 
                             [0,-1,0]])
    img = cv2.filter2D(img, -1, kernel_sharp)
    
    return img


def plot_log(filename, show=True):

    data = pd.read_csv(filename)

    fig = plt.figure(figsize=(8,10))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for key in data.keys():
        if key.find('loss') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and Validtion Loss')

    fig.add_subplot(212)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    if show:
        plt.show()
        

#Note, this code is taken straight from the SKLEARN website,
# an nice way of viewing confusion matrix.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.GnBu):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(8,10))
    plt.grid(False)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
# isplay some predicted output of the trained model
def imshow_group(X,y,y_pred=None,n_per_row=10,phase='processed'):
    '''helper function to visualize a group of images along with their categorical true labels (y) and prediction probabilities.
    Args:
        X: images
        y: categorical true labels
        y_pred: predicted class probabilities
        n_per_row: number of images per row to be plotted
        phase: If the images are plotted after resizing, pass 'processed' to phase argument. 
            It will plot the image and its true label. If the image is plotted after prediction 
            phase, pass predicted class probabilities to y_pred and 'prediction' to the phase argument. 
            It will plot the image, the true label, and it's top 3 predictions with highest probabilities.
    '''
    n_sample=len(X)
    img_dim = X.shape[1]
    j = np.ceil(n_sample/n_per_row)
    fig = plt.figure(figsize=(20,3*j))
    
    for i, img in enumerate(X):
        plt.subplot(j,n_per_row,i+1)
        img_sq = np.squeeze(img,axis = 2)
        plt.imshow(img_sq, cmap='gray')
        #plt.imshow(img)
        if phase=='processed':
            plt.title(np.argmax(y[i]))
        if phase=='prediction':
            top_n = 3 # top 3 predictions with highest probabilities
            ind_sorted = np.argsort(y_pred[i])[::-1]
            h = img_dim + 4
            for k in range(top_n):
                string = 'pred: {} ({:.0f}%)\n'.format(ind_sorted[k],y_pred[i,ind_sorted[k]]*100)
                plt.text(img_dim/2, h, string, horizontalalignment='center',
                         verticalalignment='center')
                h += 4
            if y is not None:
                plt.text(img_dim/2, -4, 'true label: {}'.format(np.argmax(y[i])), 
                         horizontalalignment='center',verticalalignment='center')
        plt.axis('off')
    plt.show()
    
    
# display some error prediction made by trained model    
def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            img_sq = np.squeeze(img_errors[error], axis = 2)
            ax[row,col].grid(False)
            ax[row,col].imshow(img_sq , cmap='gray')
            ax[row,col].set_title("Predicted label:{}\nTrue label  :{}".format(pred_errors[error],
                                                                                obs_errors[error]))
            n += 1


# In[3]:


# most of the data-loading code for numta-db from: https://www.kaggle.com/sharifamit19/data-augmentation-cross-validation-ensemble#Data-Concatenation
numta_data_dir=os.path.join('..','input/numta/')
paths_train_all=[]
path_label_train_all=[]

arr_train = ['a','b','c','d'] # I exclude part 'e' purposely, feel free to use it.
iterator_train = len(arr_train)

for i in range(iterator_train):
    #print (arr_train[i])
    dirx= 'training-'+arr_train[i]
    paths_train_x=glob.glob(os.path.join(numta_data_dir, dirx,'*.png'))
    paths_train_all=paths_train_all+paths_train_x

for i in range(iterator_train):
    dirx= 'training-'+arr_train[i] + '.csv'
    paths_label_train = glob.glob(os.path.join(numta_data_dir,dirx))
    path_label_train_all= path_label_train_all + paths_label_train


# In[4]:


def get_key(path):
    # seperates the key of an image from the filepath
    key=path.split(sep=os.sep)[-1]
    return key

def get_data(paths_img,path_label=None,resize_dim=None):
    '''reads images from the filepaths, resizes them (if given), and returns them in a numpy array
    Args:
        paths_img: image filepaths
        path_label: pass image label filepaths while processing training data, defaults to None while processing testing data
        resize_dim: if given, the image is resized to resize_dim x resize_dim (optional)
    Returns:
        X: group of images
        y: categorical true labels
    '''
    X=[] # initialize empty list for resized images
    for i,path in enumerate(paths_img):
        img=cv2.imread(path, cv2.IMREAD_GRAYSCALE) # images loaded in grayscale
        
        # calling the preprocess_image method
        img=preprocess_image(img)
        
        # some few more preprocessing
        ret, thresh = cv2.threshold(img, 0, 20, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel_dilation = np.ones((2,2), np.uint8)
        img = cv2.dilate(thresh, kernel_dilation, iterations = 1)

        # resize
        if resize_dim is not None:
            img=cv2.resize(img,(resize_dim,resize_dim),interpolation=cv2.INTER_AREA) 
        
        X.append(img) # expand image to n x n x 1 and append to the list
        
        # display progress
        if i==len(paths_img)-1:
            end='\n'
        else: end='\r'
        print('processed {}/{}'.format(i+1,len(paths_img)),end=end)
        
    X=np.array(X) # tranform list to numpy array
    if  path_label is None:
        return X
    else:
        # Concatenate all data into one DataFrame
        df = pd.DataFrame()
        l = []
        for file_ in path_label:
            df_x = pd.read_csv(file_,index_col=None, header=0)
            l.append(df_x)
        df = pd.concat(l)
        
        #df = pd.read_csv(path_label[i]) # read labels
        df=df.set_index('filename') 
        y_label=[df.loc[get_key(path)]['digit'] for path in  paths_img] # get the labels corresponding to the images
        y=to_categorical(y_label,10) # transfrom integer value to categorical variable

        return X, y


# In[5]:


img_size = 32
numtaX_train, numtaY_train = get_data(paths_train_all, path_label_train_all,
                                   resize_dim=img_size)

print (numtaX_train.shape)
print (numtaY_train.shape)


# In[6]:


plt.figure(figsize=(10, 7))
numtaY_label = np.argmax(numtaY_train, axis=1) 

# plot how many images there are in each class
sns.countplot(numtaY_label)


# In[7]:


plt.figure(figsize=(10, 10))
for i in range(0,6): # how many imgs will show from the 3x3 grid
    plt.subplot(330 + (i+1)) # open next subplot
    plt.grid(False)
    plt.imshow(numtaX_train[i], cmap=plt.get_cmap('gray'))
    plt.title(numtaY_label[i])


# In[8]:


bangalalekha_data_dir = '../input/banglalekhaisolatednumerals/'

folders = []
[folders.append('/'+i) for i in sorted(os.listdir(bangalalekha_data_dir))]

X = []
for folder in tqdm(folders):
    # get images in list
    images = os.listdir((bangalalekha_data_dir + folder))
    
    # preprocess each image using opencv
    for image in images:
        # load the image
        img = cv2.imread(bangalalekha_data_dir+folder+'/'+image, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        
        # calling the preprocess_image method
        img=preprocess_image(img)
        X.append(img) # append this image in empty list X = []
        
banglalekhaX_train = np.array(X) # convert image list to numpy array

# label encoding 
banglalekhaY_train = []
[banglalekhaY_train.extend([bc]*1940) for bc in range(0,10)] 
banglalekhaY_train = np.stack(banglalekhaY_train)
banglalekhaY_train = to_categorical(banglalekhaY_train, len(np.unique(banglalekhaY_train)))

# little shuffling and extend image axis for keras model compatability
banglalekhaX_train, banglalekhaY_train = utility(banglalekhaX_train, banglalekhaY_train, 32)

print(banglalekhaX_train.shape, banglalekhaY_train.shape)


# In[9]:


plt.figure(figsize=(10, 7))
banglalekhaY_label = np.argmax(banglalekhaY_train, axis=1) 

# plot how many images there are in each class
sns.countplot(banglalekhaY_label)


# In[10]:


plt.figure(figsize=(10, 10))

for i in range(0,6): # how many imgs will show from the 3x3 grid
    plt.subplot(330 + (i+1)) # open next subplot
    plt.grid(False)
    plt.imshow(banglalekhaX_train[i][:,:,0], cmap=plt.get_cmap('gray'))
    plt.title(banglalekhaY_label[i])


# In[11]:


ekush_data_dir = '/kaggle/input/ekush-bangla-handwritten-data-numerals/'

folders = []
[folders.append('/'+i) for i in sorted(os.listdir(ekush_data_dir))]

X = []
for folder in tqdm(folders):
    # get images in list
    images = os.listdir((ekush_data_dir + folder))
    
    # preprocess each image using opencv
    for image in images:
        # load the image
        img = cv2.imread(ekush_data_dir+folder+'/'+image, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA) # resize image to 28x28
        
        # calling the preprocess_image method
        img=preprocess_image(img)
        X.append(img) # append this image in empty list X = []
        
ekushX_train = np.array(X) # convert image to numpy array

# label encoding 
ekushY_train = []
[ekushY_train.extend([bc]*3069) for bc in range(0,10)] 
ekushY_train = np.stack(ekushY_train)
ekushY_train = to_categorical(ekushY_train, len(np.unique(ekushY_train)))

# little shuffling and extend image axis for keras model compatability
ekushX_train, ekushY_train = utility(ekushX_train, ekushY_train, 28)

print(ekushX_train.shape, ekushY_train.shape)


# In[12]:


plt.figure(figsize=(10, 7))
ekushY_label = np.argmax(ekushY_train, axis=1) 

# plot how many images there are in each class
sns.countplot(ekushY_label)


# In[13]:


plt.figure(figsize=(10, 10))

for i in range(0,6): # how many imgs will show from the 3x3 grid
    plt.subplot(330 + (i+1)) # open next subplot
    plt.grid(False)
    plt.imshow(ekushX_train[i][:,:,0], cmap=plt.get_cmap('gray'))
    plt.title(ekushY_label[i])


# In[14]:


class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. 
    It drives the length of a large vector to near 1 and small vector to 0
    
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.
    
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, 
                            self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)

            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, 
                           strides=strides, padding=padding,name='primarycap_conv2d')(inputs)
    
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], 
                             name='primarycap_reshape')(output)
    
    return layers.Lambda(squash, name='primarycap_squash')(outputs)


# In[15]:


from keras import layers, models
from keras import backend as K
from keras.utils import to_categorical
import keras
import tensorflow as tf
from keras.layers import (Dense, Input, Conv2D, Flatten, MaxPooling2D, 
                          Activation, Dropout, Average,BatchNormalization,
                          GlobalAveragePooling2D, concatenate, Add)

# ---------------- another way --------------
# sound of your voice - pain in reverse though I love to hear 

def CapsNet(input_shape, n_class, routings):
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', 
                          activation='relu', name='conv1')(x)

    # ----------------------------- Primary CapsuleNet -----------------------------
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, 
                             kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps_')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    train_model = models.Model(inputs=[x], outputs=[out_caps])  # only encoder
    
    return train_model
    

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) +         0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


# In[16]:


def callback():
    cb = []

    """
    Model-Checkpoint
    """
    checkpoint = callbacks.ModelCheckpoint('Vanila CapsuleNet Weights Best.h5',
                                       save_best_only=True, 
                                       mode='min',
                                       monitor='val_loss', #  val_capsnet_loss : only for encoder + decoder model
                                       save_weights_only=True, verbose=1)

    cb.append(checkpoint)
    
    """
    Early Stopping callback
    """
    #Uncomment for usage
    early_stop = callbacks.EarlyStopping(monitor = 'val_loss', # val_capsnet_loss : only for encoder + decoder model
                                     min_delta=0, 
                                     patience=20, verbose=1, 
                                     mode = 'auto')
    cb.append(early_stop)
    
    # learning reate decay
    #lr_decay = callbacks.LearningRateScheduler(schedule = lambda epoch: 0.001 * np.exp(-epoch / 10.))
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * (0.9 ** epoch))
    cb.append(lr_decay)
    
    # Callback that streams epoch results to a csv file.
    log = callbacks.CSVLogger('log.csv')
    cb.append(log)

    return cb


# In[17]:


def train_caps(model, data, epoch_size_frac=1.0, training = False, wg = None):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_val, y_val) = data

    # callbacks functions
    cb = callback()
    
    # compile the model
    '''Only for Encoder builded Model'''
    # if we like to exclude decoder section - just testing 
    model.compile(optimizer=Adam(lr=1e-3),
                  loss=[margin_loss],
                  loss_weights=[1.],
                  metrics={'capsnet': 'accuracy'})

    
    # --------------Begin: Training with data augmentation -------
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range = shift_fraction,   
                                           height_shift_range = shift_fraction,  
                                          rotation_range=20,   
                                          shear_range=0.1,                      
                                          zoom_range=0.1, 
                                          horizontal_flip=False, 
                                          vertical_flip=False, 
                                          fill_mode='nearest')  
        
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        
        while 1:
            x_batch, y_batch = generator.next() 
            yield ([x_batch, y_batch])
             
    if training:            
        # Training with data augmentation. 
        '''Model for only Encoder Output'''
        history = model.fit_generator(generator = train_generator(x_train, y_train, 64, 0.1),
                            steps_per_epoch = int(y_train.shape[0] / 64),
                            validation_steps=x_val.shape[0] // 64,
                            epochs = 100, 
                            validation_data = train_generator(x_val, y_val, 128),
                            callbacks = cb)
    else:
        model.load_weights(wg) 
    # -----End: Training with data augmentation -------#
    
    return model


# In[18]:


# Set the random seed
random_seed = 2019

# Randomly split the data sets
from sklearn.model_selection import train_test_split

### Reshape 
numtaX_train = numtaX_train.reshape(numtaX_train.shape[0],32,32,1).astype('float32')/255.

# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(numtaX_train,
                                  numtaY_train, 
                                  test_size = 0.1,
                                  random_state=random_seed,
                                  shuffle=True)

print('90% for training   ', X_train.shape, Y_train.shape)
print('10% for validation ', X_val.shape, Y_val.shape)


# In[19]:


# define model
model = CapsNet(input_shape=[32, 32, 1],
                n_class=10,
                routings=3)

# calls training functions
train_caps(model = model, data = ((X_train, Y_train), (X_val, Y_val)), 
           training = False, wg = '../input/weightabatches/numta.h5',
           epoch_size_frac = 0.5)

# print summary
model.summary()


# In[20]:


log_file = "../input/logsbatches/numta.csv"
plot_log(log_file)


# In[21]:


# Predict the values from the validation dataset
# encoder 
Y_pred = model.predict([X_val], batch_size=64, verbose = 1)

# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis = 1) 

# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val, axis = 1) 

# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10), normalize=False,
                      title='Confusion Matrix')


# In[22]:


from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(10)]

scores = model.evaluate(X_val, Y_val, # model for encoder 
                        verbose = 1, batch_size=64) 

print(classification_report(Y_true, Y_pred_classes, target_names = target_names))
print(scores)


# In[23]:


# encoder 
predictions_prob = model.predict([X_val], 
                                 batch_size = 32, 
                                 verbose = True)

n_sample = 10
np.random.seed(42)
ind = np.random.randint(0, len(X_val), size = n_sample)

imshow_group(X = X_val[ind],y = None, y_pred = predictions_prob[ind], phase='prediction')


# In[24]:


# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors, axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)


# In[25]:


del X_train, X_val, Y_train, Y_val
del numtaX_train, numtaY_train
del model

'''
X_train , Y_train : Actual training
X_val , Y_val : Evaluationg
'''
X_train, X_val, Y_train, Y_val = train_test_split(banglalekhaX_train, 
                                                  banglalekhaY_train, 
                                                  test_size=0.20, 
                                                  random_state=random_seed,
                                                  shuffle = True)

print('80% for training    ', X_train.shape, Y_train.shape)
print('20% for validation  ', X_val.shape, Y_val.shape)


# In[26]:


# define model
model = CapsNet(input_shape=[32, 32, 1],
                n_class=10,
                routings=3)

# calls training functions
train_caps(model = model, data = ((X_train, Y_train), (X_val, Y_val)), 
           training = False, wg = '../input/weightabatches/banglalekha.h5',
           epoch_size_frac = 0.5)

# print summary
model.summary()


# In[27]:


log_file = "../input/logsbatches/banglalekha.csv"
plot_log(log_file)


# In[28]:


# Predict the values from the validation dataset
# encoder 
Y_pred = model.predict([X_val], batch_size=64, verbose = 1)

# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis = 1) 

# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val, axis = 1) 

# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10), normalize=False,
                      title='Confusion Matrix')


# In[29]:


from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(10)]

scores = model.evaluate(X_val, Y_val, # model for encoder 
                        verbose = 1, batch_size=64) 

print(classification_report(Y_true, Y_pred_classes, target_names = target_names))
print(scores)


# In[30]:


# encoder 
predictions_prob = model.predict([X_val], 
                                 batch_size = 32, 
                                 verbose = True)

n_sample = 10
np.random.seed(42)
ind = np.random.randint(0, len(X_val), size = n_sample)

imshow_group(X = X_val[ind],y = None, y_pred = predictions_prob[ind], phase='prediction')


# In[31]:


# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors, axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)


# In[32]:


del X_train, X_val, Y_train, Y_val
del banglalekhaX_train, banglalekhaY_train
del model

'''
X_train , Y_train : Actual training
X_val , Y_val : Evaluationg
'''
X_train, X_val, Y_train, Y_val = train_test_split(ekushX_train, 
                                                  ekushY_train,
                                                  test_size=0.20, 
                                                  random_state=random_seed, 
                                                  shuffle = True)

print('80% for training    ', X_train.shape, Y_train.shape)
print('20% for validation  ', X_val.shape, Y_val.shape)


# In[33]:


# define model
model = CapsNet(input_shape=[28, 28, 1],
                n_class=10,
                routings=3)

# calls training functions
train_caps(model = model, data = ((X_train, Y_train), (X_val, Y_val)), 
           training = False, wg = '../input/weightabatches/ekush.h5',
           epoch_size_frac = 0.5)

# print summary
model.summary()


# In[34]:


log_file = "../input/logsbatches/ekush.csv"
plot_log(log_file)


# In[35]:


# Predict the values from the validation dataset
# encoder 
Y_pred = model.predict([X_val], batch_size=64, verbose = 1)

# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis = 1) 

# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val, axis = 1) 

# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10), normalize=False,
                      title='Confusion Matrix')


# In[36]:


from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(10)]

scores = model.evaluate(X_val, Y_val, # model for encoder 
                        verbose = 1, batch_size=64) 

print(classification_report(Y_true, Y_pred_classes, target_names = target_names))
print(scores)


# In[37]:


# encoder 
predictions_prob = model.predict([X_val], 
                                 batch_size = 32, 
                                 verbose = True)

n_sample = 10
np.random.seed(42)
ind = np.random.randint(0, len(X_val), size = n_sample)

imshow_group(X = X_val[ind],y = None, y_pred = predictions_prob[ind], phase='prediction')


# In[38]:


# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors, axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)

