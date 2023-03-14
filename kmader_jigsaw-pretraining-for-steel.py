#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})
plt.rcParams['image.cmap'] = 'gray' # grayscale looks better
from itertools import cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']




# tests help notebooks stay managable
import doctest
import copy
import functools

def autotest(func):
    globs = copy.copy(globals())
    globs.update({func.__name__: func})
    doctest.run_docstring_examples(
        func, globs, verbose=True, name=func.__name__)
    return func




from pathlib import Path
import numpy as np
import pandas as pd
import os
from skimage.io import imread
from skimage.util import montage
from skimage.color import label2rgb
from itertools import product
from tqdm import tqdm_notebook
from IPython.display import clear_output
import torch
import tensorflow as tf
import random
seed = 2019
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
tf.random.set_random_seed(seed)
clear_output()




data_dir = Path('..') / 'input' / 'severstal-steel-defect-detection'
# Load the data
train_df = pd.read_csv(data_dir / "train.csv")
train_df['ImageId'] = train_df['ImageId_ClassId'].map(lambda x: x.split('_')[0])
train_df['ClassId'] = train_df['ImageId_ClassId'].map(lambda x: x.split('_')[-1])
train_df['image_path'] = train_df['ImageId'].map(lambda x: str(data_dir / 'train_images' / x))
train_df.drop('ImageId_ClassId', axis=1, inplace=True)
flat_train_df = train_df.pivot_table(index=['ImageId', 'image_path'], columns='ClassId', values='EncodedPixels', aggfunc='first')
flat_train_df['defects_count'] = flat_train_df.applymap(lambda x: len(x) if isinstance(x, str) else 0).sum(axis=1)
flat_train_df = flat_train_df.reset_index().sort_values('defects_count', ascending=False)
print(flat_train_df.shape)
flat_train_df.head(5)




def make_mask(c_row, mask_channel):
    '''Given a row index, return image_id and mask (256, 1600, 1)'''
    # 4:class 1～4 (ch:0～3)
    mask = np.zeros(256 * 1600, dtype=np.bool)
    if c_row[mask_channel] is not np.nan:
        label = c_row[mask_channel].split(" ")
        positions = map(int, label[0::2])
        length = map(int, label[1::2])
        
        for pos, le in zip(positions, length):
            mask[pos:(pos + le)] = 1
    return mask.reshape(256, 1600, order='F')
def idx_mask(in_mask):
    return (1+np.argmax(in_mask, -1))*np.max(in_mask, -1)
def full_mask(c_row):
    return np.stack([make_mask(c_row, '{}'.format(i)) for i in range(1, 5)], -1)




rand_row = flat_train_df.sample(1).iloc[0]
rand_img = imread(rand_row['image_path'], as_gray=True)
rand_mask = full_mask(rand_row)
plt.imshow(label2rgb(label=idx_mask(rand_mask), image=rand_img, bg_label=0))




get_ipython().run_cell_magic('time', '', "# calculate for all rows\nif False:\n    flat_train_df['mask_image'] = flat_train_df.apply(full_mask, axis=1)")




@autotest
def cut_jigsaw(
    in_image, # type: np.ndarray
    x_wid, # type: int
    y_wid,# type: int
    gap=False,
    jitter=False,
    jitter_dim=None, # type: Optional[int]
):
    # type: (...) -> List[np.ndarray]
    """Cuts the image into little pieces
    :param in_image: the image to cut-apart
    :param x_wid: the size of the piece in x
    :param y_wid: the size of the piece in y
    :param gap: if there is a gap between tiles
    :param jitter: if the positions should be moved around
    :param jitter_dim: amount to jitter (default is x_wid or y_wid/2)
    :return : a 4D array with tiles x x_wid x y_wid * d
    Examples
    >>> test_image = np.arange(20).reshape((4, 5)).astype(int)
    >>> test_image
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19]])
    >>> cut_jigsaw(test_image, 2, 2, False, False)
    array([[[ 0,  1],
            [ 5,  6]],
    <BLANKLINE>
           [[ 2,  3],
            [ 7,  8]],
    <BLANKLINE>
           [[10, 11],
            [15, 16]],
    <BLANKLINE>
           [[12, 13],
            [17, 18]]])
    >>> cut_jigsaw(test_image, 2, 2, True, False)
    array([[[ 0,  1],
            [ 5,  6]],
    <BLANKLINE>
           [[ 3,  4],
            [ 8,  9]],
    <BLANKLINE>
           [[10, 11],
            [15, 16]],
    <BLANKLINE>
           [[13, 14],
            [18, 19]]])
    >>> np.random.seed(0)
    >>> cut_jigsaw(test_image, 2, 2, True, True, 1)
    array([[[ 1,  2],
            [ 6,  7]],
    <BLANKLINE>
           [[ 7,  8],
            [12, 13]],
    <BLANKLINE>
           [[ 5,  6],
            [10, 11]],
    <BLANKLINE>
           [[ 7,  8],
            [12, 13]]])
    """
    if len(in_image.shape)==2:
        in_image = np.expand_dims(in_image, -1)
        expand = True
    else:
        expand = False
    x_size, y_size, d_size = in_image.shape
    out_tiles = []
    x_chunks = x_size//x_wid
    y_chunks = y_size//y_wid
    out_tiles = np.zeros((x_chunks*y_chunks, x_wid, y_wid, d_size), dtype=in_image.dtype)
    if gap:
        # we calculate the maximum gap and 
        x_gap = x_size-x_chunks*x_wid
        y_gap = y_size-y_chunks*y_wid
    else:
        x_gap, y_gap = 0, 0
    x_jitter = x_wid//2 if jitter_dim is None else jitter_dim
    y_jitter = y_wid//2 if jitter_dim is None else jitter_dim
    for idx, (i, j) in enumerate(product(range(x_chunks), range(y_chunks))):
        x_start = i*x_wid+min(x_gap, i)
        y_start = j*y_wid+min(y_gap, j)
        if jitter:
            x_range = max(x_start-x_jitter, 0), min(x_start+x_jitter+1, x_size-x_wid)
            y_range = max(y_start-y_jitter, 0), min(y_start+y_jitter+1, y_size-y_wid)
            
            x_start = np.random.choice(range(*x_range)) if x_range[1]>x_range[0] else x_start
            y_start = np.random.choice(range(*y_range)) if y_range[1]>y_range[0] else y_start
            
        out_tiles[idx, :, :, :] = in_image[x_start:x_start+x_wid, y_start:y_start+y_wid, :]
    
    return out_tiles[:, :, :, 0] if expand else out_tiles
                




@autotest
def jigsaw_to_image(
    in_tiles, # type: np.ndarray
    out_x, # type: int
    out_y, # type: int
    gap=False
):
    # type: (...) -> np.ndarray
    """Reassembles little pieces into an image
    :param in_tiles: the tiles to reassemble
    :param out_x: the size of the image in x (default is calculated automatically)
    :param out_y: the size of the image in y
    :param gap: if there is a gap between tiles
    :return : an image from the tiles
    Examples
    >>> test_image = np.arange(20).reshape((4, 5)).astype(int)
    >>> test_image
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19]])
    >>> js_pieces = cut_jigsaw(test_image, 2, 2, False, False)
    >>> jigsaw_to_image(js_pieces, 4, 5)
    array([[ 0,  1,  2,  3,  0],
           [ 5,  6,  7,  8,  0],
           [10, 11, 12, 13,  0],
           [15, 16, 17, 18,  0]])
    >>> js_gap_pieces = cut_jigsaw(test_image, 2, 2, True, False)
    >>> jigsaw_to_image(js_gap_pieces, 4, 5, True)
    array([[ 0,  1,  0,  3,  4],
           [ 5,  6,  0,  8,  9],
           [10, 11,  0, 13, 14],
           [15, 16,  0, 18, 19]])
    >>> np.random.seed(0)
    >>> js_gap_pieces = cut_jigsaw(test_image, 2, 2, False, True)
    >>> jigsaw_to_image(js_gap_pieces, 4, 5, False)
    array([[ 1,  2,  6,  7,  0],
           [ 6,  7, 11, 12,  0],
           [ 6,  7,  7,  8,  0],
           [11, 12, 12, 13,  0]])
    """
    if len(in_tiles.shape)==3:
        in_tiles = np.expand_dims(in_tiles, -1)
        expand = True
    else:
        expand = False
    tile_count, x_wid, y_wid, d_size = in_tiles.shape
    x_chunks = out_x//x_wid
    y_chunks = out_y//y_wid
    out_image = np.zeros((out_x, out_y, d_size), dtype=in_tiles.dtype)
    
    if gap:
        x_gap = out_x-x_chunks*x_wid
        y_gap = out_y-y_chunks*y_wid
    else:
        x_gap, y_gap = 0, 0
        
    for idx, (i, j) in enumerate(product(range(x_chunks), range(y_chunks))):
        x_start = i*x_wid+min(x_gap, i)
        y_start = j*y_wid+min(y_gap, j)
        out_image[x_start:x_start+x_wid, y_start:y_start+y_wid] = in_tiles[idx, :, :]
    
    return out_image[:, :, 0] if expand else out_image
    
    
    




TILE_X = 128
TILE_Y = 128
JITTER_SIZE = 16
TRAIN_TILE_COUNT = 2**11
VALID_TILE_COUNT = 2**9
KEEP_RANDOM_PERM = 200
LATENT_SIZE = 32
BIG_LATENT_SIZE = 64
NR_EPOCHS = 15




fig, m_axs = plt.subplots(6, 4, figsize=(15, 25))
for img_idx, c_axs in enumerate(m_axs.T, 1):
    x_img = imread(flat_train_df.iloc[img_idx]['image_path'])
    c_axs[0].imshow(x_img)
    c_axs[0].set_title('Input')
    out_tiles = cut_jigsaw(x_img, TILE_X, TILE_Y, gap=False) 
    for k, c_ax in zip(range(out_tiles.shape[0]), c_axs[1:-1]):
        c_ax.matshow(out_tiles[k, :, :, 0])
    recon_img = jigsaw_to_image(out_tiles, x_img.shape[0], x_img.shape[1])
    c_axs[-1].imshow(recon_img[:, :, 0])
    c_axs[-1].set_title('Reconstruction')




@autotest
def get_rand_perms(n, k):
    """Get k random permutations of n numbers
    >>> get_rand_perms(3, 2)
    [[0, 1, 2], [0, 2, 1]]
    >>> from itertools import permutations
    >>> nine_perms = np.array(list(permutations(range(9), 9)))
    >>> random.seed(2019)
    >>> keep_nine_perm = nine_perms[0:1, :].tolist()+random.sample(nine_perms.tolist(), 99)
    >>> # np.allclose(get_rand_perms(9, 100), keep_nine_perm, atol=0.5)
    """
    random.seed(2019)
    all_perm = [np.arange(n).tolist()]
    for i in range(k-1):
        rem_nums = set(range(n))
        all_perm.append(random.sample(range(n), n))
    return all_perm




keep_perm = get_rand_perms(out_tiles.shape[0], KEEP_RANDOM_PERM)




fig, m_axs = plt.subplots(5, 5, figsize=(15, 10))
x_img = np.expand_dims(imread(flat_train_df.iloc[img_idx]['image_path'], as_gray=True), -1)
for i, c_axs in enumerate(m_axs.T):
    out_tiles = cut_jigsaw(x_img, TILE_X, TILE_Y, gap=False, jitter=i>0, jitter_dim=JITTER_SIZE) 
    for j, (c_ax, c_perm) in enumerate(zip(c_axs, keep_perm)): 
        scrambled_tiles = out_tiles[c_perm]
        recon_img = jigsaw_to_image(scrambled_tiles, x_img.shape[0], x_img.shape[1])
        c_ax.imshow(recon_img.squeeze())
        c_ax.set_title('Permutation:#{}\nJitter:{}'.format(j, i>0))
        c_ax.axis('off')




from sklearn.model_selection import train_test_split
flat_train_df['defects_count'].hist(bins=30)
train_frames_df, valid_frames_df = train_test_split(flat_train_df, 
                                                    test_size=0.3, 
                                                    random_state=2019, 
                                                    stratify=pd.qcut(flat_train_df['defects_count'], 10)
                                                   )
print(train_frames_df.shape, valid_frames_df.shape)




out_tiles = cut_jigsaw(x_img, TILE_X, TILE_Y, gap=False) 

def _generate_batch(in_idx, is_valid=False):
    np.random.seed(in_idx)
    if is_valid:
        img_idx = np.random.choice(range(valid_frames_df.shape[0]))
        c_row = valid_frames_df.iloc[img_idx]
    else:
        img_idx = np.random.choice(range(train_frames_df.shape[0]))
        c_row = train_frames_df.iloc[img_idx]
    x_img = np.expand_dims(imread(c_row['image_path'], as_gray=True), -1)
    out_tiles = cut_jigsaw(x_img, TILE_X, TILE_Y, gap=True, jitter=JITTER_SIZE>0, jitter_dim=JITTER_SIZE) 
    perm_idx = np.random.choice(range(len(keep_perm)))
    c_perm = keep_perm[perm_idx]
    return out_tiles[c_perm], perm_idx

def make_tile_group(tile_count, is_valid=False):
    c_tiles = np.zeros((tile_count,)+out_tiles.shape, dtype='float32')
    c_perms = np.zeros((tile_count,), dtype='int')
    for i in tqdm_notebook(range(tile_count)):
        # should be parallelized
        c_tiles[i], c_perms[i] = _generate_batch(i, is_valid=is_valid)
    return c_tiles, c_perms
train_tiles, train_perms = make_tile_group(TRAIN_TILE_COUNT)
valid_tiles, valid_perms = make_tile_group(VALID_TILE_COUNT, is_valid=True)




from keras import models, layers
tile_encoder = models.Sequential(name='TileEncoder')
# we use None to make the model more usuable later
tile_encoder.add(layers.BatchNormalization(input_shape=(None, None)+(train_tiles.shape[-1],)))
for i in range(6):
    tile_encoder.add(layers.Conv2D(8*2**i, (3,3), padding='same', activation='linear'))
    tile_encoder.add(layers.BatchNormalization())
    tile_encoder.add(layers.MaxPool2D(2,2))
    tile_encoder.add(layers.LeakyReLU(0.1))

tile_encoder.add(layers.Conv2D(LATENT_SIZE, (1,1), activation='linear'))
tile_encoder.add(layers.BatchNormalization())
tile_encoder.add(layers.LeakyReLU(0.1))
clear_output() # some annoying loading/warnings come up




tile_encoder.summary()




print('Model Input Shape:', train_tiles.shape[2:], 
      '-> Model Output Shape:', tile_encoder.predict(np.zeros((1,)+train_tiles.shape[2:])).shape[1:])




big_in = layers.Input(train_tiles.shape[1:], name='All_Tile_Input')
feat_vec = []
for k in range(train_tiles.shape[1]):
    lay_x = layers.Lambda(lambda x: x[:, k], name='Select_{}_Tile'.format(k))(big_in)
    feat_x = tile_encoder(lay_x)
    feat_vec += [layers.GlobalAvgPool2D()(feat_x)]
feat_cat = layers.concatenate(feat_vec)
feat_dr = layers.Dropout(0.5)(feat_cat)
feat_latent = layers.Dense(BIG_LATENT_SIZE)(feat_dr)
feat_latent_dr = layers.Dropout(0.5)(feat_latent)
out_pred = layers.Dense(KEEP_RANDOM_PERM, activation='softmax')(feat_latent_dr)
big_model = models.Model(inputs=[big_in], outputs=[out_pred])
big_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy'])




from keras.utils.vis_utils import model_to_dot
from IPython.display import Image
dot_model = model_to_dot(big_model, show_shapes=True)
dot_model.set_rankdir('LR')
Image(dot_model.create_png())




reversed_keep_perm = [[c_dict[j] for j in range(out_tiles.shape[0])]
                      for c_dict in [{j: i for i, j in enumerate(c_perm)}
                                     for c_perm in keep_perm]]
for i in range(3):
    print('forward', keep_perm[i], 'reversed', reversed_keep_perm[i])




def show_model_output(image_count=4, perm_count=3): 
    fig, m_axs = plt.subplots(image_count, perm_count+1, figsize=(5*(perm_count+1), 2*image_count))
    [c_ax.axis('off') for c_ax in m_axs.flatten()]
    for img_idx, c_axs in enumerate(m_axs):
        img_idx = np.random.choice(range(flat_train_df.shape[0]))
        c_row = flat_train_df.iloc[img_idx]
        x_img = np.expand_dims(imread(c_row['image_path'], as_gray=True), -1)
        perm_idx = np.random.choice(range(len(keep_perm)))
        c_axs[0].imshow(x_img.squeeze())
        
        c_axs[0].set_title('Input #{}'.format(perm_idx))
        # generate tiles
        out_tiles = cut_jigsaw(x_img, TILE_X, TILE_Y, gap=True, jitter=JITTER_SIZE>0, jitter_dim=JITTER_SIZE)
        # scramble tiles
        
        c_perm = keep_perm[perm_idx]
        scr_tiles = out_tiles[c_perm]
        # get model prediction
        out_pred = big_model.predict(np.expand_dims(scr_tiles, 0))[0]
        for c_ax, k_idx in zip(c_axs[1:], np.argsort(-1*out_pred)):
            pred_rev_perm = reversed_keep_perm[k_idx]
            recon_img = jigsaw_to_image(scr_tiles[pred_rev_perm], x_img.shape[0], x_img.shape[1])
            c_ax.imshow(recon_img.squeeze())
            c_ax.set_title('Pred: #{} ({:2.2%})'.format(k_idx, out_pred[k_idx]))
show_model_output()




fit_results = big_model.fit(train_tiles, train_perms, 
                            validation_data=(valid_tiles, valid_perms),
                                 batch_size=24,
                                 epochs=NR_EPOCHS)
clear_output()




fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.semilogy(fit_results.history['loss'], label='Training')
ax1.semilogy(fit_results.history['val_loss'], label='Validation')
ax1.legend()
ax1.set_title('Loss')
ax2.plot(fit_results.history['sparse_categorical_accuracy'], label='Training')
ax2.plot(fit_results.history['val_sparse_categorical_accuracy'], label='Validation')
ax2.legend()
ax2.set_title('Accuracy')
ax2.set_ylim(0, 1)




show_model_output(image_count=10, perm_count=4)




tile_encoder.save('tile_encoder.h5')




conv_weight_dict = {(idx, k.name): k.get_weights() for idx, k in enumerate(tile_encoder.layers) if isinstance(k, layers.Conv2D)}
print(conv_weight_dict.keys())
fig, m_axs = plt.subplots(2, 3, figsize=(20, 10))
for c_ax, ((idx, lay_name), [W, b]) in zip(m_axs.flatten(), conv_weight_dict.items()):
    c_ax.set_title('{} #{}\n{}'.format(lay_name, idx, W.shape))
    flat_W = W.reshape((W.shape[0], W.shape[1], -1)).swapaxes(0, 2).swapaxes(1,2)
    if flat_W.shape[1]>1 or flat_W.shape[2]>1:
        pad_W = np.pad(flat_W, [(0, 0), (1, 1), (1,1)], mode='constant', constant_values=np.NAN)
        pad_W = montage(pad_W, fill=np.NAN, grid_shape=(W.shape[2], W.shape[3]))
    else:
        pad_W = W[0, 0]
    c_ax.imshow(pad_W, vmin=-1, vmax=1, cmap='RdBu')
    




full_tiles = train_tiles.reshape((-1, train_tiles.shape[2], train_tiles.shape[3], train_tiles.shape[4]))
print(full_tiles.shape)




gp_outputs = []
for k in tile_encoder.layers:
    if isinstance(k, layers.LeakyReLU):
        c_output = k.get_output_at(0)
        c_smooth = layers.AvgPool2D((2, 2))(c_output)
        c_gp = layers.GlobalMaxPool2D(name='GP_{}'.format(k.name))(c_smooth)
        gp_outputs += [c_gp]
activation_tile_encoder = models.Model(inputs = tile_encoder.inputs, 
                                       outputs = gp_outputs)
activation_maps = dict(zip(activation_tile_encoder.output_names, 
                           activation_tile_encoder.predict(full_tiles, batch_size=256, verbose=True)))

for k, v in activation_maps.items():
    print(k, v.shape)




keep_top_n = 5
fig, m_axs = plt.subplots(1, len(activation_maps), figsize=(20, 20))
for c_ax, (k, v) in zip(m_axs.T, activation_maps.items()):
    c_ax.set_title(k)
    active_rows = []
    for i in range(v.shape[1]):
        top_idx = np.argsort(-np.abs(v[:, i]))[:keep_top_n]
        active_rows += [full_tiles[top_idx, :, :, 0]]
    c_ax.imshow(montage(np.concatenate(active_rows, 0), grid_shape=(v.shape[1], keep_top_n), padding_width=1))
    c_ax.axis('off')




print(x_img.shape, '->', tile_encoder.predict(np.expand_dims(x_img, 0)).shape)




img_in = layers.Input(x_img.shape)
tile_encoder.trainable=False
full_feat_mat = tile_encoder(img_in)
seg_img = layers.Conv2D(1, (1, 1), activation='sigmoid')(full_feat_mat)
us_out = layers.UpSampling2D((64, 64))(seg_img)
image_encoder = models.Model(inputs=[img_in], outputs=[us_out], name='SegmentImage')
image_encoder.summary()




def data_gen(in_df, batch_size):
    while True:
        c_batch = in_df.sample(batch_size)
        yield np.stack(c_batch['image_path'].map(lambda x: np.expand_dims(imread(x, as_gray=True), -1)), 0),             np.sum(np.stack(c_batch.apply(full_mask, axis=1).values, 0), axis=-1, keepdims=True)
train_gen = data_gen(train_frames_df, 8)
valid_gen = data_gen(valid_frames_df, 8)
samp_X, samp_y = next(train_gen)
print(samp_X.shape, samp_y.shape)




from keras import backend as K
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

image_encoder.compile(optimizer='adam', loss=dice_coef_loss, metrics=['binary_accuracy', 'mae', dice_coef])




def montage_tile(in_img):
    batch_size = in_img.shape[0]
    return montage(in_img[...,0], grid_shape=(batch_size, 1))
def show_batch(in_gen):
    samp_X, samp_y = next(in_gen)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 25))
    ax1.imshow(montage_tile(samp_X), cmap='gray')
    ax1.axis('off')
    ax2.imshow(montage_tile(samp_y), cmap='viridis', vmin=0, vmax=1)
    ax2.axis('off')
    ax3.imshow(montage_tile(image_encoder.predict(samp_X)), cmap='viridis', vmin=0, vmax=1)
    ax3.set_title('Prediction')
    ax3.axis('off')
show_batch(valid_gen)




seg_results = image_encoder.fit_generator(train_gen, 
                            validation_data=valid_gen,
                                          steps_per_epoch=50,
                                          validation_steps=10,
                                 epochs=NR_EPOCHS)
clear_output()




fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.semilogy(seg_results.history['loss'], label='Training')
ax1.semilogy(seg_results.history['val_loss'], label='Validation')
ax1.legend()
ax1.set_title('Loss')
ax2.plot(seg_results.history['binary_accuracy'], label='Training')
ax2.plot(seg_results.history['val_binary_accuracy'], label='Validation')
ax2.legend()
ax2.set_title('Accuracy')
ax2.set_ylim(0, 1)




show_batch(valid_gen)




image_encoder.save('encoder_model.h5')

