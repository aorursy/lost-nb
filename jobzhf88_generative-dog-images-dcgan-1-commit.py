#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import zipfile
import tensorflow as tf
import xml.etree.ElementTree as ET
from tqdm import tqdm
from keras.models import Model 
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import ReLU
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras import backend as K
from PIL import Image
from time import time
import torchvision


# In[2]:


ROOT = '../input/'
IMAGES = os.listdir(ROOT + 'all-dogs/all-dogs/')
BREEDS = os.listdir(ROOT + 'annotation/Annotation/') 
BREEDS_DIR=ROOT+'annotation/Annotation/'
IMAGES_DIR=ROOT+'all-dogs/all-dogs/'
start=time()

img_rows = 64
img_cols = 64
channels = 3
img_shape = (img_rows, img_cols, channels)
latent_dim = 128
random_dim =128


# In[3]:


def load_images():
    # Place holder for output 
    all_images = np.zeros((22250, 64, 64, 3))
    namesIn = []
    
    # Index
    index = 0
    
    required_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(64),
            torchvision.transforms.CenterCrop(64),
    ])
    
    for breed in BREEDS:
        for dog in os.listdir(BREEDS_DIR + breed):
            try: img = Image.open(IMAGES_DIR + dog + '.jpg') 
            except: continue  
                
            tree = ET.parse(BREEDS_DIR + breed + '/' + dog)
            root = tree.getroot()
            objects = root.findall('object')
            for o in objects:
                bndbox = o.find('bndbox') 
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                w = np.min((xmax - xmin, ymax - ymin))
                bbox = (xmin, ymin, xmin+w, ymin+w)
                object_img = required_transforms(img.crop(bbox))
                #object_img = object_img.resize((64,64), Image.ANTIALIAS)                
                all_images[index,:]=object_img
                index += 1
                namesIn.append(breed)

                # Determine each side
                xdelta = xmax - xmin
                ydelta = ymax - ymin
                
                # Take the mean of the sides
                #w = int((xdelta + ydelta) / 2)
                
                
#                 # Filter out images where bounding box is below 64 pixels.
#                 # This filters out a couple of 100 images but prevents using low resolution images.
#                 if xdelta >= 64 and ydelta >= 64:
#                     img2 = img.crop((xmin, ymin, xmax, ymax))
#                     img2 = img2.resize((64, 64), Image.ANTIALIAS)
#                     image = np.asarray(img2)
                    
#                     #    # Normalize to range[-1, 1]
#                     #all_images[index,:] = (image.astype(np.float32) - 127.5)/127.5
#                     all_images[index,:]=image
#                     index += 1
#                     namesIn.append(breed)
        
                # Plot Status
                if index % 5000 == 0:
                    print('Processed Images: {}'.format(index))

    print('Total Processed Images: {}'.format(index))
    
    
#     for breed in BREEDS:
#         for dog in os.listdir(ROOT+'annotation/Annotation/'+breed):
#             try: img = Image.open(ROOT+'all-dogs/all-dogs/'+dog+'.jpg') 
#             except: continue           
#             tree = ET.parse(ROOT+'annotation/Annotation/'+breed+'/'+dog)
#             root = tree.getroot()
#             objects = root.findall('object')
#             for o in objects:
#                 bndbox = o.find('bndbox') 
#                 xmin = int(bndbox.find('xmin').text)
#                 ymin = int(bndbox.find('ymin').text)
#                 xmax = int(bndbox.find('xmax').text)
#                 ymax = int(bndbox.find('ymax').text)
#                 w = np.min((xmax - xmin, ymax - ymin))
#                 img2 = img.crop((xmin, ymin, xmin+w, ymin+w))
#                 img2 = img2.resize((64,64), Image.ANTIALIAS)
#                 imagesIn[idxIn,:,:,:] = np.asarray(img2)
#                 #if idxIn%1000==0: print(idxIn)
#                 namesIn.append(breed)
#                 idxIn += 1
#     idx = np.arange(idxIn)
#     np.random.shuffle(idx)
#     imagesIn = imagesIn[idx,:,:,:]
#     namesIn = np.array(namesIn)[idx]

    return all_images,namesIn,index


# In[4]:


X_train,namesIn,idxIn=load_images()


# In[5]:


#DISPLAY CROPPED IMAGES
x = np.random.randint(0,idxIn,15)
for k in range(3):
    plt.figure(figsize=(15,3))
    for j in range(5):
        plt.subplot(1,5,j+1)
        img = Image.fromarray(X_train[x[k*5+j],:,:,:].astype('uint8') )
        plt.axis('off')
        plt.title(namesIn[x[k*5+j]].split('-')[1],fontsize=11)
        plt.imshow(img)
    plt.show()


# In[6]:


# adapted from keras.optimizers.Adam
class AdamWithWeightnorm(Adam):
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations, K.floatx())))

        t = K.cast(self.iterations + 1, K.floatx())
        lr_t = lr * K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t))

        shapes = [K.get_variable_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):

            # if a weight tensor (len > 1) use weight normalized parameterization
            # this is the only part changed w.r.t. keras.optimizers.Adam
            ps = K.get_variable_shape(p)
            if len(ps)>1:

                # get weight normalization parameters
                V, V_norm, V_scaler, g_param, grad_g, grad_V = get_weightnorm_params_and_grads(p, g)

                # Adam containers for the 'g' parameter
                V_scaler_shape = K.get_variable_shape(V_scaler)
                m_g = K.zeros(V_scaler_shape)
                v_g = K.zeros(V_scaler_shape)

                # update g parameters
                m_g_t = (self.beta_1 * m_g) + (1. - self.beta_1) * grad_g
                v_g_t = (self.beta_2 * v_g) + (1. - self.beta_2) * K.square(grad_g)
                new_g_param = g_param - lr_t * m_g_t / (K.sqrt(v_g_t) + self.epsilon)
                self.updates.append(K.update(m_g, m_g_t))
                self.updates.append(K.update(v_g, v_g_t))

                # update V parameters
                m_t = (self.beta_1 * m) + (1. - self.beta_1) * grad_V
                v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(grad_V)
                new_V_param = V - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
                self.updates.append(K.update(m, m_t))
                self.updates.append(K.update(v, v_t))

                # if there are constraints we apply them to V, not W
                if getattr(p, 'constraint', None) is not None:
                    new_V_param = p.constraint(new_V_param)

                # wn param updates --> W updates
                add_weightnorm_param_updates(self.updates, new_V_param, new_g_param, p, V_scaler)

            else: # do optimization normally
                m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
                v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

                self.updates.append(K.update(m, m_t))
                self.updates.append(K.update(v, v_t))

                new_p = p_t
                # apply constraints
                if getattr(p, 'constraint', None) is not None:
                    new_p = p.constraint(new_p)
                self.updates.append(K.update(p, new_p))
        return self.updates

def get_weightnorm_params_and_grads(p, g):
    ps = K.get_variable_shape(p)

    # construct weight scaler: V_scaler = g/||V||
    V_scaler_shape = (ps[-1],)  # assumes we're using tensorflow!
    V_scaler = K.ones(V_scaler_shape)  # init to ones, so effective parameters don't change

    # get V parameters = ||V||/g * W
    norm_axes = [i for i in range(len(ps) - 1)]
    V = p / tf.reshape(V_scaler, [1] * len(norm_axes) + [-1])

    # split V_scaler into ||V|| and g parameters
    V_norm = tf.sqrt(tf.reduce_sum(tf.square(V), norm_axes))
    g_param = V_scaler * V_norm

    # get grad in V,g parameters
    grad_g = tf.reduce_sum(g * V, norm_axes) / V_norm
    grad_V = tf.reshape(V_scaler, [1] * len(norm_axes) + [-1]) *              (g - tf.reshape(grad_g / V_norm, [1] * len(norm_axes) + [-1]) * V)

    return V, V_norm, V_scaler, g_param, grad_g, grad_V

def add_weightnorm_param_updates(updates, new_V_param, new_g_param, W, V_scaler):
    ps = K.get_variable_shape(new_V_param)
    norm_axes = [i for i in range(len(ps) - 1)]

    # update W and V_scaler
    new_V_norm = tf.sqrt(tf.reduce_sum(tf.square(new_V_param), norm_axes))
    new_V_scaler = new_g_param / new_V_norm
    new_W = tf.reshape(new_V_scaler, [1] * len(norm_axes) + [-1]) * new_V_param
    updates.append(K.update(W, new_W))
    updates.append(K.update(V_scaler, new_V_scaler))

# data based initialization for a given Keras model
def data_based_init(model, input):
    # input can be dict, numpy array, or list of numpy arrays
    if type(input) is dict:
        feed_dict = input
    elif type(input) is list:
        feed_dict = {tf_inp: np_inp for tf_inp,np_inp in zip(model.inputs,input)}
    else:
        feed_dict = {model.inputs[0]: input}

    # add learning phase if required
    if model.uses_learning_phase and K.learning_phase() not in feed_dict:
        feed_dict.update({K.learning_phase(): 1})

    # get all layer name, output, weight, bias tuples
    layer_output_weight_bias = []
    for l in model.layers:
        trainable_weights = l.trainable_weights
        if len(trainable_weights) == 2:
            W,b = trainable_weights
            assert(l.built)
            layer_output_weight_bias.append((l.name,l.get_output_at(0),W,b)) # if more than one node, only use the first

    # iterate over our list and do data dependent init
    sess = K.get_session()
    for l,o,W,b in layer_output_weight_bias:
        print('Performing data dependent initialization for layer ' + l)
        m,v = tf.nn.moments(o, [i for i in range(len(o.get_shape())-1)])
        s = tf.sqrt(v + 1e-10)
        updates = tf.group(W.assign(W/tf.reshape(s,[1]*(len(W.get_shape())-1)+[-1])), b.assign((b-m)/s))
        sess.run(updates, feed_dict)


# In[7]:


#optimizer = tf.keras.optimizers.Adam(0.001, 0.5)
adamWithWeightnorm=AdamWithWeightnorm(lr = 0.0002, beta_1 = 0.5)


# In[8]:


def build_generator():

    # Random Normal Weight Initialization
    init = RandomNormal(mean = 0.0, stddev = 0.02)

    # Model
    model = Sequential()

    # Start at 4 * 4
    start_shape = 64 * 4 * 4
    model.add(Dense(start_shape, kernel_initializer = init, input_dim = random_dim))
    model.add(Reshape((4, 4, 64)))
    
    # Upsample => 8 * 8 
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size = 3, padding = "same", kernel_initializer = init))
    model.add(ReLU())
    
    # Upsample => 16 * 16 
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size = 3, padding = "same", kernel_initializer = init))
    model.add(ReLU())
    
    # Upsample => 32 * 32
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size = 3, padding = "same", kernel_initializer = init))
    model.add(ReLU())
    
    # Upsample => 64 * 64
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size = 3, padding = "same", kernel_initializer = init))
    model.add(ReLU())
    
    # output
    model.add(Conv2D(3, kernel_size = 3, activation = 'tanh', padding = 'same', kernel_initializer=init))
    model.compile(loss = 'binary_crossentropy', optimizer = AdamWithWeightnorm(lr = 0.0002, beta_1 = 0.5))
    print(model.summary())

    return model


# In[9]:


def build_discriminator():

    input_shape = (64, 64, 3)

    # Random Normal Weight Initialization
    init = RandomNormal(mean = 0.0, stddev = 0.02)

    # Define Model
    model = Sequential()

    # Downsample ==> 32 * 32
    model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = init, input_shape = input_shape))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))

    # Downsample ==> 16 * 16
    model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = init))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    
    # Downsample => 8 * 8
    model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = init))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    
    # Downsample => 4 * 4
    model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = init))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    
    # Final Layers
    model.add(Flatten())
    model.add(Dense(1, activation = 'sigmoid', kernel_initializer = init))

    # Compile model
    model.compile(loss = 'binary_crossentropy', optimizer = AdamWithWeightnorm(lr = 0.0002, beta_1 = 0.5))
    
    print(model.summary())

    return model


# In[10]:


def show_images( epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(image.array_to_img(gen_imgs[cnt]))
            axs[i,j].axis('off')
            cnt += 1
    #fig.savefig("mnist_%d.png" % epoch)
    plt.show()
    plt.close()


# In[11]:


def train(epochs, batch_size=128, save_interval=50):

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    valid[:]=0.9
    
    for epoch in range(epochs):
        
        end = time()
        if (end -start) > 31900 :
            break

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        
        #imgs = imgs/255
        imgs = (imgs -127.5) / 127.5

        # Sample noise and generate a batch of new images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)

        # Train the discriminator (real classified as ones and generated as zeros)
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        discriminator.trainable = True
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
#         print("d_loss_real:",d_loss_real)
#         print("d_loss_fake:",d_loss_fake)
#         print("d_loss:",d_loss)

        # ---------------------
        #  Train Generator
        # ---------------------
        discriminator.trainable = False
        # Train the generator (wants discriminator to mistake images as real)
        y_gen =  np.ones((batch_size, 1))        
        g_loss = combined.train_on_batch(noise, y_gen)
        #g_loss = combined.train_on_batch(noise, valid)

        # Plot the progress
        #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        # If at save interval => save generated image samples
        if (epoch+1) % save_interval == 0  or  epoch == 0:
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss, 100*d_loss, g_loss))
            #show_images(epoch)


# In[12]:


# Build and compile the discriminator
discriminator = build_discriminator()
#discriminator.compile(loss='binary_crossentropy',
#    optimizer=AdamWithWeightnorm(lr = 0.0002, beta_1 = 0.5))
    #metrics=['accuracy'])

# Build the generator
generator = build_generator()

# The generator takes noise as input and generates imgs
z = Input(shape=(latent_dim,))
img = generator(z)

# For the combined model we will only train the generator
discriminator.trainable = False

# The discriminator takes generated images as input and determines validity
valid = discriminator(img)

# The combined model  (stacked generator and discriminator)
# Trains the generator to fool the discriminator
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=AdamWithWeightnorm(lr = 0.0002, beta_1 = 0.5))


# In[13]:


get_ipython().run_cell_magic('time', '', 'train(epochs=2000000, batch_size=32, save_interval=10000)')


# In[14]:


for k in range(5):
    noise = np.random.normal(0, 1, (1, latent_dim))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    #gen_imgs = 0.5 * gen_imgs + 0.5
    
    image = Image.fromarray(((gen_imgs + 1) * 127.5).astype('uint8').reshape(64, 64, 3))

    #plt.imshow(image.array_to_img(gen_imgs[0]))
    plt.imshow(image, interpolation = 'nearest')
    plt.show()


# In[15]:


z = zipfile.PyZipFile('images.zip', mode='w')
for k in range(10000):
    noise = np.random.normal(0, 1, (1, latent_dim))
    gen_imgs = generator.predict(noise)
    # Rescale images 0 - 1
#     gen_imgs = 0.5 * gen_imgs + 0.5
#     save_image=image.array_to_img(gen_imgs[0])
    save_image = Image.fromarray(((gen_imgs + 1) * 127.5).astype('uint8').reshape(64, 64, 3))
    #plt.imshow(save_image)
    #plt.show()
    f = str(k)+'.png'
    save_image.save(f,'PNG'); z.write(f); os.remove(f)
    #if k % 1000==0: print(k)
z.close()

