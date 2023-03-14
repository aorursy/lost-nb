#!/usr/bin/env python
# coding: utf-8

# In[1]:


INPUT_HEIGHT = INPUT_WIDTH = 64
OUTPUT_HEIGHT = OUTPUT_WIDTH = 64
CROP_HEIGHT = CROP_WIDTH = 64
BATCH_SIZE = 32
ANNOTATIONS_PATH = '../input/annotation/Annotation/'
IMAGES_PATH = '../input/all-dogs/all-dogs/'
Z_DIM, Z_STD = 64, 1
CLIP_Z = False
Y_DIM, Y_EMBEDDING_DIM = 120, 10
C_DIM = 3
GF_DIM = DF_DIM = 64
G_LR, D_LR = 1e-4, 1e-4
ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON = 0.5, 0.99, 1e-8
PERFORM_SN = False
SN_ITERATIONS = 1
LOCAL_RUN = False
INPUT_NOISE_ANNEALING_STEPS = 3e5
# HINGE | KL | SIGMOID_CROSSENTROPY | LEAKY_HINGE | SQUARED_LEAKY_HINGE | LEAKY_SIGMOID_CROSSENTROPY
LOSS = 'LEAKY_SIGMOID_CROSSENTROPY' 
NOISY_LOSS_TARGET = True
INPUTS_NOISE = True
LABELS_NOISE = True
CONV_K = 7
TRAINING_TIME_LIMIT = 32000


# In[2]:


import cv2
import tensorflow as tf
from tensorflow import concat
import tensorflow.contrib.slim as slim
import imageio
import numpy as np
import xml.etree.ElementTree as ET
import os
import time
import datetime
from glob import glob
from tqdm import tqdm_notebook as tqdm
import zipfile
from itertools import chain, islice, count
from IPython.core.display import display, HTML


# In[3]:


class batch_norm(object):

    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name,
                                           )


def spectral_normed_weight(w, update_u=True):
    if PERFORM_SN:
        def l2normalize(v, eps=1e-12):
            return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)
        w_shape = w.shape.as_list()
        w_mat = tf.reshape(w, [-1, w_shape[-1]])  # [-1, output_channel]
        u = tf.get_variable('u', [1, w_shape[-1]], trainable=False,
                            initializer=tf.truncated_normal_initializer())
        u_ = u
        for _ in range(SN_ITERATIONS):
            v_ = l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
            u_ = l2normalize(tf.matmul(v_, w_mat))
        sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
        w_mat /= sigma
        if update_u:
            with tf.control_dependencies([u.assign(u_)]):
                w_normed = tf.reshape(w_mat, w_shape)
        else:
            w_normed = tf.reshape(w_mat, w_shape)
        return w_normed
    else:
        return w


def conv2d(input_, output_dim,
           k_h=CONV_K, k_w=CONV_K, d_h=2, d_w=2, stddev=0.02,
           name="conv2d", update_u=True, add_bias=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        w = spectral_normed_weight(w, update_u=update_u)
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        if add_bias:
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv


def deconv2d(input_, output_shape,
             k_h=CONV_K, k_w=CONV_K, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", update_u=True, add_bias=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        w = spectral_normed_weight(w, update_u=update_u)
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        if add_bias:
            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, biases)
        return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.nn.leaky_relu(x, leak, name)


def linear(input_, output_size, scope=None,
           stddev=0.02, bias_start=0.0, update_u=True):
    shape = input_.get_shape()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        matrix = spectral_normed_weight(matrix, update_u=update_u)
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, matrix) + bias


def add_coordinates(input_tensor):
    bs, h_dim, w_dim, c = input_tensor.get_shape().as_list()

    x = tf.stack([tf.range(w_dim, dtype=tf.float32)] * h_dim) / (w_dim - 1) * 2 - 1
    y = tf.transpose(tf.stack([tf.range(h_dim, dtype=tf.float32)] * w_dim)) / (h_dim - 1) * 2 - 1
    r = ((x**2 + y**2) / 2) ** 0.5
    coords = tf.stack([tf.stack([x, y, r], axis=-1)] * bs)
    coords = tf.constant(tf.get_default_session().run(coords))
    
    return tf.concat([input_tensor, coords], axis=-1)


# In[4]:


def timestamp(template='%Y%m%d_%H%M%S', ts=None):
    return datetime.datetime.fromtimestamp(ts or time.time()).strftime(template)


def conv_out_size_same(size, stride):
    return int(np.ceil(float(size) / float(stride)))


# In[5]:


def transform_X(x):
    x = x / 128 - 255 / 256
    return tf.image.random_flip_left_right(x)


# In[6]:


class DCGAN(object):

    def __init__(self, sess, data):
        self.start_time = time.time()
        out_dir = './gan_out/{}/'.format(timestamp())
        self.checkpoint_dir = os.path.join(out_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.sample_dir = os.path.join(out_dir, 'samples')
        os.makedirs(self.sample_dir, exist_ok=True)
        self.log_file = open(os.path.join(out_dir, 'log.txt'), 'w')
        
        self.sess = sess
        self.global_step = tf.train.create_global_step()
        self.load_data(*data)
        self.build_model()

    def load_data(self, X, Y):
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        dataset = dataset.map((lambda x, y: (x, tf.one_hot(y, Y_DIM, dtype=tf.float32))))
        dataset = dataset.shuffle(len(X), reshuffle_each_iteration=True).repeat()
        dataset = dataset.map((lambda x, y: (transform_X(x), y)), num_parallel_calls=2)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE)).prefetch(2)
        
        self.x_batch, self.y_batch = dataset.make_one_shot_iterator().get_next()
        self.clear_x_batch, self.clear_y_batch = self.x_batch, self.y_batch
        if INPUTS_NOISE:
            self.x_batch = self.noisy_inputs(self.x_batch)
        if LABELS_NOISE:
            self.y_batch = self.noisy_labels(self.y_batch)

    @staticmethod
    def x_plus_x_squared(x):
        return x + x * x
        
    def get_d_real_loss(self, logits):
        if LOSS  == 'SIGMOID_CROSSENTROPY':
            if NOISY_LOSS_TARGET:
                target = tf.random.uniform((BATCH_SIZE, 1), minval=0.95, maxval=1.05, dtype=tf.float32) 
            else:
                target = tf.fill(tf.shape(logits), 1.)
        elif LOSS  == 'LEAKY_SIGMOID_CROSSENTROPY':
            target = tf.fill(tf.shape(logits), 0.9)
            if NOISY_LOSS_TARGET:
                target += tf.random.uniform([], minval=-0.025, maxval=0.025) 
        elif NOISY_LOSS_TARGET:
            target = tf.random.uniform((BATCH_SIZE, 1), minval=0.95, maxval=1.05, dtype=tf.float32)           
        else:
            target = 1.0

        if LOSS == 'HINGE':
            loss = tf.nn.relu(target - logits)
        elif LOSS == 'KL':
            loss = tf.nn.softplus(-logits)
        elif LOSS in ('SIGMOID_CROSSENTROPY', 'LEAKY_SIGMOID_CROSSENTROPY'):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=target)
        elif LOSS == 'LEAKY_HINGE':
            loss = tf.nn.leaky_relu(target - logits, -0.2)
        elif LOSS == 'SQUARED_LEAKY_HINGE':
            loss = self.x_plus_x_squared(tf.nn.leaky_relu(target - logits, -0.2))
        return tf.reduce_mean(loss)

    def get_d_fake_loss(self, logits):
        if LOSS  == 'SIGMOID_CROSSENTROPY':
            if NOISY_LOSS_TARGET:
                target = tf.random.uniform((BATCH_SIZE, 1), minval=0.05, maxval=0.05, dtype=tf.float32) 
            else:
                target = tf.fill(tf.shape(logits), 0.)
        elif LOSS  == 'LEAKY_SIGMOID_CROSSENTROPY':
            target = tf.fill(tf.shape(logits), 0.1)
            if NOISY_LOSS_TARGET:
                target += tf.random.uniform([], minval=-0.025, maxval=0.025) 
        elif NOISY_LOSS_TARGET:
            target = tf.random.uniform((BATCH_SIZE, 1), minval=0.95, maxval=1.05, dtype=tf.float32)
        else:
            target = 1.0
            
        if LOSS == 'HINGE':
            loss = tf.nn.relu(target + logits)
        elif LOSS == 'KL':
            loss = tf.nn.softplus(logits)
        elif LOSS in ('SIGMOID_CROSSENTROPY', 'LEAKY_SIGMOID_CROSSENTROPY'):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=target)
        elif LOSS == 'LEAKY_HINGE':
            loss = tf.nn.leaky_relu(target + logits, -0.2)
        elif LOSS == 'SQUARED_LEAKY_HINGE':
            loss = self.x_plus_x_squared(tf.nn.leaky_relu(target + logits, -0.2))
        return tf.reduce_mean(loss)

    def get_g_loss(self, logits):
        if LOSS  == 'SIGMOID_CROSSENTROPY':
            if NOISY_LOSS_TARGET:
                target = tf.random.uniform((BATCH_SIZE, 1), minval=0.95, maxval=1.05, dtype=tf.float32) 
            else:
                target = tf.fill(tf.shape(logits), 1.)
        elif LOSS  == 'LEAKY_SIGMOID_CROSSENTROPY':
            target = tf.fill(tf.shape(logits), 0.9)
            if NOISY_LOSS_TARGET:
                target += tf.random.uniform([], minval=-0.025, maxval=0.025)
        elif NOISY_LOSS_TARGET:
            target = tf.random.uniform((BATCH_SIZE, 1), minval=0.95, maxval=1.05, dtype=tf.float32)
        else:
            target = 1.0
            
        if LOSS in ('HINGE', 'KL'):
            loss = -logits
        elif LOSS in ('SIGMOID_CROSSENTROPY', 'LEAKY_SIGMOID_CROSSENTROPY'):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=target)
        elif LOSS == 'LEAKY_HINGE':
            loss = tf.nn.leaky_relu(target - logits, -0.2)
        elif LOSS == 'SQUARED_LEAKY_HINGE':
            loss = self.x_plus_x_squared(tf.nn.leaky_relu(target - logits, -0.2))
        return tf.reduce_mean(loss)

    def build_model(self):
        self.increment_global_step = self.global_step.assign_add(1)

        self.z_batch = tf.random.normal((BATCH_SIZE, Z_DIM), 0, Z_STD)
        if CLIP_Z:
            self.z_batch = tf.clip_by_value(self.z_batch, -1, 1)

        # batch normalization : deals with poor initialization, helps gradient flow
        self.d_bn0 = batch_norm(name='d_bn0')
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')
        
        self.G = self.generator(self.z_batch, self.y_batch)
        z_sample, y_sample = self.sess.run([self.z_batch, self.y_batch])
        self.fixed_sampler = tf.cast((self.generator(tf.constant(z_sample), tf.constant(y_sample), update_u=False) + 1) * 127.5, tf.uint8)
        self.sampler = tf.cast((self.generator(self.z_batch, self.y_batch, update_u=False) + 1) * 127.5, tf.uint8)
        
        self.D_logits_real = self.discriminator(self.x_batch, self.y_batch)
        self.D_logits_fake = self.discriminator(self.G, self.y_batch)
        
        self.d_loss_real = self.get_d_real_loss(self.D_logits_real)
        self.d_loss_fake = self.get_d_fake_loss(self.D_logits_fake)
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = self.get_g_loss(self.D_logits_fake)
        self.get_vars()
        self.d_optim = tf.train.AdamOptimizer(D_LR, beta1=ADAM_BETA1, beta2=ADAM_BETA2, epsilon=ADAM_EPSILON)             .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(G_LR, beta1=ADAM_BETA1, beta2=ADAM_BETA2, epsilon=ADAM_EPSILON)             .minimize(self.g_loss, var_list=self.g_vars)
        self.saver = tf.train.Saver(max_to_keep=1)
        tf.global_variables_initializer().run(session=self.sess)

    def train(self):
        info_template = (
            'step: {}, time: {:5.2f}, d_loss_fake: {:.4f}, '
            'd_loss_real: {:.4f}, d_loss: {:.5f}, g_loss: {:.5f}')
        running_time = time.time() - self.start_time
        while running_time < TRAINING_TIME_LIMIT:
            d_fake, d_real, d, g, step, _, _ = self.sess.run(
                [self.d_loss_fake, self.d_loss_real, self.d_loss, self.g_loss, 
                 self.increment_global_step,
                 self.d_optim, self.g_optim,
                ],
            )
            previous_running_time = running_time
            running_time = time.time() - self.start_time
            print(info_template.format(step, running_time, 
                                       d_fake, d_real, d, g),
                  file=self.log_file)
            if previous_running_time // 10 != running_time // 10:  # do each 10 seconds
                print(info_template.format(step, running_time, 
                                           d_fake, d_real, d, g))
            if previous_running_time // 1000 != running_time // 1000:  # do each 1000 seconds
                self.save_sample()
                self.generate_zip(os.path.join(self.sample_dir, f'images{running_time // 1000}.zip'), 1000)
                self.save(self.checkpoint_dir)
    
    def noisy_labels(self, labels, mean=0, std=0.03):
        '''Add noise to labels'''
        return tf.add(labels, tf.random_normal(tf.shape(labels), mean, std, dtype=tf.float32))

    def noisy_inputs(self, inputs, mean=0, std=0.03, annealing_steps=INPUT_NOISE_ANNEALING_STEPS):
        step = tf.cast(self.global_step, tf.float32)
        return tf.cond(step < annealing_steps,
                lambda: tf.clip_by_value(tf.add(inputs, tf.random_normal(tf.shape(inputs), 
                                                               mean, 
                                                               std * (annealing_steps - step) / annealing_steps, 
                                                               dtype=tf.float32)), -1, 1),
                lambda: inputs)
     
    def get_vars(self):
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if var.name.startswith('discriminator/d_')]
        self.g_vars = [var for var in t_vars if var.name.startswith('generator/g_')]
        self.sigma_ratio_vars = [var for var in t_vars if 'sigma_ratio' in var.name]
        assert (set(self.d_vars) & set(self.g_vars)) == set()
        assert (set(self.d_vars) | set(self.g_vars)) == set(t_vars)
        self.all_vars = t_vars
        
    def discriminator(self, image, y, update_u=True):
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as scope:
            h0 = lrelu(self.d_bn0(conv2d(image, DF_DIM, name='d_h0_conv', update_u=update_u))) # 32 * 32
            h1 = lrelu(self.d_bn1(conv2d(h0, DF_DIM * 2, name='d_h1_conv', update_u=update_u))) # 16 * 16
            h2 = lrelu(self.d_bn2(conv2d(h1, DF_DIM * 4, name='d_h2_conv', update_u=update_u))) # 8 * 8
            h3 = lrelu(self.d_bn3(conv2d(h2, DF_DIM * 8, name='d_h3_conv', update_u=update_u))) # 4 * 4
            y_embedding = lrelu(self.d_bn4(linear(y, Y_EMBEDDING_DIM, 'd_y_embedding', update_u=update_u)))
            h4 = linear(concat([tf.reshape(h3, [BATCH_SIZE, -1]), y_embedding], 1), 1, 'd_h4_lin', update_u=update_u)
            return h4

    def generator(self, z, y, is_training=True, update_u=True):
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE) as scope:
            y_embedding = lrelu(self.g_bn0(linear(y, Z_DIM, 'g_y_embedding', update_u=update_u)))
            z = concat([z, y_embedding], 1)
            
            s_h, s_w = OUTPUT_HEIGHT, OUTPUT_WIDTH
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            
            # project `z` and reshape
            z_ = linear(z, GF_DIM * 8 * s_h16 * s_w16, 'g_h0_lin', update_u=update_u)
            h0 = tf.reshape(z_, [-1, s_h16, s_w16, GF_DIM * 8])
            h0 = lrelu(self.g_bn1(h0, train=is_training))
            h1 = deconv2d(h0, [BATCH_SIZE, s_h8, s_w8, GF_DIM * 4], name='g_h1', update_u=update_u) # 8 * 8
            h1 = lrelu(self.g_bn2(h1, train=is_training))
            h2 = deconv2d(h1, [BATCH_SIZE, s_h4, s_w4, GF_DIM * 2], name='g_h2', update_u=update_u) # 16 * 16
            h2 = lrelu(self.g_bn3(h2, train=is_training))
            h3 = deconv2d(h2, [BATCH_SIZE, s_h2, s_w2, GF_DIM * 1], name='g_h3', update_u=update_u) # 32 * 32
            h3 = lrelu(self.g_bn4(h3, train=is_training))
            h4 = deconv2d(h3, [BATCH_SIZE, s_h, s_w, C_DIM], name='g_final_deconv', update_u=update_u) # 64 * 64
            return tf.nn.tanh(h4)
  
    def generate_images(self):
        images = self.sess.run(
            self.sampler, 
        )
        return images
            
    def save_sample(self):
        images, step = self.sess.run(
            [self.fixed_sampler, self.global_step]
        )
        sample = np.vstack([np.hstack(images[i*16:(i+1)*16]) for i in range(max(1, min(8, BATCH_SIZE // 16)))])
        sample_path = os.path.join(self.sample_dir, '{}.png'.format(step))
        imageio.imwrite(sample_path, sample)
        imageio.imwrite('real_dogs_sample.png', sample)
        display(HTML(f'<img src="{sample_path}" alt="Sample of generated dogs." width="100%">'))
        
    def save(self, checkpoint_dir, filename='model'):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, filename))
        
    def generate_zip(self, zip_path='images.zip', n_images=10000):
        start_time = time.time()
        with zipfile.PyZipFile(zip_path, 'w') as z:
            for i, image in enumerate(islice(chain.from_iterable(gan.generate_images() for _ in count()), n_images)):
                path = f'{i}.png'
                imageio.imwrite(path, image); z.write(path); os.remove(path)
        print(f'{time.time() - start_time:.1f}s to generate {n_images} images')


# In[7]:


def get_bboxes(annotation_file):
    '''Extract and return bounding boxes from annotation file.'''
    bboxes = []
    objects = ET.parse(annotation_file).getroot().findall('object')
    for obj in objects:
        bbox = obj.find('bndbox')
        bboxes.append(tuple(int(bbox.find(_).text) for _ in ('xmin', 'ymin', 'xmax', 'ymax')))
    return bboxes


def imread(path):
    img_bgr = cv2.imread(path)
    return img_bgr[..., ::-1]


def transform(image, height=CROP_HEIGHT, width=CROP_WIDTH):
    image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_LANCZOS4)
    return image


def load_data():
    print('Loading data, iterating over {} breeds...'.format(Y_DIM))
    dogs, breeds = [], []
    for breed_index, breed_path in enumerate(tqdm(sorted(glob(ANNOTATIONS_PATH + '*')))):
        for annotation_file in glob(breed_path + '/*'):
            image_path = IMAGES_PATH + '{}.jpg'.format(annotation_file.split('/')[-1])
            try:
                image = imread(image_path)
                bboxes = get_bboxes(annotation_file)
                # add crops
                for xmin, ymin, xmax, ymax in bboxes:
                    dogs.append(transform(image[max(0, ymin-10):ymax+10, max(0, xmin-10):xmax+10, :]))
                    breeds.append(breed_index)
            except Exception as e:
                continue  # there is one annotation file without corresponding image
    print('Done. {} dogs in total.'.format(len(dogs)))
    return np.array(dogs, dtype=np.uint8), np.array(breeds, dtype=np.int32)

DOGS, BREEDS = load_data()


# In[8]:


def show_all_variables(verbose=False):
    model_vars = tf.trainable_variables()
    stats = slim.model_analyzer.analyze_vars(model_vars, print_info=verbose)
    if not verbose:
        print('Total variables: {:.2f}M.'.format(stats[0]/10**6))

def show_flops():
    g = tf.get_default_graph()
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
    print('{:.3f} TFLOPs in graph.'.format(flops.total_float_ops / 10**12))


# In[9]:


tf.reset_default_graph()
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
sess = tf.Session(config=run_config)
gan = DCGAN(sess, (DOGS, BREEDS))
show_all_variables()
show_flops()


# In[10]:


gan.train()


# In[11]:


gan.generate_zip()

