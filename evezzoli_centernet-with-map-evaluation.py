#!/usr/bin/env python
# coding: utf-8



#@title Import des librairies
import os
import glob
import shutil

# Création d'un dossier
def create_repertory(rep):
    """
    fonction de création de dossier
    """
    try:
        os.mkdir(rep)
    except:
        print('Le dossier est existant')

Kaggle = True

if Kaggle == False:
    from zipfile import ZipFile
    from google.colab import drive
    drive.mount('/content/drive')

    # On liste les fichiers
    path = "/content/drive/My Drive/Colab Notebooks/IML Projet 8/"
    files = glob.glob(path + "*.zip")
    # On supprime le dossier
    shutil.rmtree('Data', True)
    # On créé les dossier de données
    create_repertory('Data')

    # ouvrir les fichiers zip en mode lecture
    for file in files:
        with ZipFile(file, 'r') as zip: 
            # afficher tout le contenu du fichier zip
            zip.printdir()
            file = file.split(path)[1]
            file = file.split('.zip')[0]
            rep = 'Data/' + file
            # On supprime le dossier
            shutil.rmtree(rep, True)
            # On créé les dossier de données
            create_repertory(rep)

            # extraire tous les fichiers
            print('extraction...') 
            zip.extractall(rep) 
            print('Terminé!')

    files = glob.glob(path + "*.csv")
    for file in files:
        file = file.split(path)[1]
        shutil.copy(path + file, 'Data/' + file)
    
    PATH = 'Data/'
    os.listdir(PATH)

else:
    PATH = '../input/pku-autonomous-driving/'
    os.listdir(PATH)




get_ipython().system('pip install efficientnet-pytorch')




import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
import os
from scipy.optimize import minimize
import plotly.express as px
from math import sin, cos
import gc

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms, utils


from efficientnet_pytorch import EfficientNet




# Fonctions:
def imread(path, fast_mode=False):
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img

def str2coords(s):
    pred_string = s
    items = pred_string.split(' ')
    model_types, yaws, pitches, rolls, xs, ys, zs = [items[i::7] for i in range(7)]
    liste = pd.DataFrame([ model_types, yaws, pitches, rolls, xs, ys, zs],
                        index=['model_type', 'yaw', 'pitche', 'roll', 'x', 'y', 'z']).T
    coords = []
    for i in range(liste.shape[0]):
        coords.append({'id': float(liste['model_type'][i]),
                    'yaw': float(liste['yaw'][i]),
                    'pitch': float(liste['pitche'][i]),
                    'roll': float(liste['roll'][i]),
                    'x': float(liste['x'][i]),
                    'y': float(liste['y'][i]),
                    'z': float(liste['z'][i])
                    })
    return coords

def rotate(x, angle):
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x

def get_img_coords(s):
    '''
    Input is a PredictionString (e.g. from train dataframe)
    Output is two arrays:
        xs: x coordinates in the image
        ys: y coordinates in the image
    '''
    coords = str2coords(s)
    xs = [float(c['x']) for c in coords]
    ys = [float(c['y']) for c in coords]
    zs = [float(c['z']) for c in coords]
    position = []
    for i in range(len(xs)):
        position.append([xs[i], ys[i], zs[i]])
    P = np.array(position).T
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2] # z = Distance from the camera
    return img_xs, img_ys

# convert euler angle to rotation matrix
def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))

def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return image

def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)
    return image

def visualize(img, coords):
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31
    
    img = img.copy()
    for point in coords:
        # Get values
        x, y, z = point['x'], point['y'], point['z']
        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        # Math
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.array([[x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1],
                      [0, 0, 0, 1]]).T
        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        # Drawing
        img = draw_line(img, img_cor_points)
        img = draw_points(img, img_cor_points[-1:])
    
    return img

def _regr_preprocess(regr_dict, flip=False):
    if flip:
        for k in ['x', 'pitch', 'roll']:
            regr_dict[k] = -regr_dict[k]
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] / 100
    regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)
    regr_dict['pitch_sin'] = sin(regr_dict['pitch'])
    regr_dict['pitch_cos'] = cos(regr_dict['pitch'])
    regr_dict.pop('pitch')
    regr_dict.pop('id')
    return regr_dict

def _regr_back(regr_dict):
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] * 100
    regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)
    
    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict

def preprocess_image(img, flip=False):
    img = img[img.shape[0] // 2:]
    bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg = bg[:, :img.shape[1] // 6]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    if flip:
        img = img[:,::-1]
    return (img / 255).astype('float32')

def get_mask_and_regr(img, labels, flip=False):
    mask = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE], dtype='float32')
    regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    regr = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE, 7], dtype='float32')
    coords = str2coords(labels)
    xs, ys = get_img_coords(labels)
    position = []
    for i in range(len(xs)):
        position.append([xs[i], ys[i], coords[i]])
    for x, y, regr_dict in position:
        x, y = y, x
        x = (x - img.shape[0] // 2) * IMG_HEIGHT / (img.shape[0] // 2) / MODEL_SCALE
        x = np.round(x).astype('int')
        y = (y + img.shape[1] // 6) * IMG_WIDTH / (img.shape[1] * 4/3) / MODEL_SCALE
        y = np.round(y).astype('int')
        if x >= 0 and x < IMG_HEIGHT // MODEL_SCALE and y >= 0 and y < IMG_WIDTH // MODEL_SCALE:
            mask[x, y] = 1
            regr_dict = _regr_preprocess(regr_dict, flip)
            regr[x, y] = [regr_dict[n] for n in sorted(regr_dict)]
    if flip:
        mask = np.array(mask[:,::-1])
        regr = np.array(regr[:,::-1])
    return mask, regr

def convert_3d_to_2d(x, y, z, fx = 2304.5479, fy = 2305.8757, cx = 1686.2379, cy = 1354.9849):
    # stolen from https://www.kaggle.com/theshockwaverider/eda-visualization-baseline
    return x * fx / z + cx, y * fy / z + cy

def optimize_xy(r, c, x0, y0, z0, flipped=False):
    def distance_fn(xyz):
        x, y, z = xyz
        xx = -x if flipped else x
        slope_err = (xzy_slope.predict([[xx,z]])[0] - y)**2
        x, y = convert_3d_to_2d(x, y, z)
        y, x = x, y
        x = (x - IMG_SHAPE[0] // 2) * IMG_HEIGHT / (IMG_SHAPE[0] // 2) / MODEL_SCALE
        y = (y + IMG_SHAPE[1] // 6) * IMG_WIDTH / (IMG_SHAPE[1] * 4 / 3) / MODEL_SCALE
        return max(0.2, (x-r)**2 + (y-c)**2) + max(0.4, slope_err)
    
    res = minimize(distance_fn, [x0, y0, z0], method='Powell')
    x_new, y_new, z_new = res.x
    return x_new, y_new, z_new

def clear_duplicates(coords):
    for c1 in coords:
        xyz1 = np.array([c1['x'], c1['y'], c1['z']])
        for c2 in coords:
            xyz2 = np.array([c2['x'], c2['y'], c2['z']])
            distance = np.sqrt(((xyz1 - xyz2)**2).sum())
            if distance < DISTANCE_THRESH_CLEAR:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in coords if c['confidence'] > 0]

def extract_coords(prediction, flipped=False):
    logits = prediction[0]
    regr_output = prediction[1:]
    points = np.argwhere(logits > 0)
    col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])
    coords = []
    for r, c in points:
        position = []
        for i in range(len(col_names)):
            position.append([col_names[i], regr_output[:, r, c][i]])
        regr_dict = dict(position)
        coords.append(_regr_back(regr_dict))
        coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))
        coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] =                 optimize_xy(r, c,
                            coords[-1]['x'],
                            coords[-1]['y'],
                            coords[-1]['z'], flipped)
    coords = clear_duplicates(coords)
    return coords

def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)




train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'sample_submission.csv')
train.head()




# From camera.zip
camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
camera_matrix_inv = np.linalg.inv(camera_matrix)




inp = train['PredictionString'][0]
print('Example input:\n', inp)
print('Output:\n', str2coords(inp))




points_df = pd.DataFrame()
for col in ['x', 'y', 'z', 'yaw', 'pitch', 'roll']:
    arr = []
    for ps in train['PredictionString']:
        coords = str2coords(ps)
        arr += [c[col] for c in coords]
    points_df[col] = arr

print('len(points_df)', len(points_df))
points_df.head()




lens = [len(str2coords(s)) for s in train['PredictionString']]

plt.figure(figsize=(15,5))
sns.countplot(lens);
plt.xlabel('Number of cars in image')
plt.title('Number of cars per image')
plt.show()




plt.figure(figsize=(15,5))
sns.distplot(points_df['x'], bins=200);
plt.xlabel('x')
plt.title('X distribution')
plt.show()




plt.figure(figsize=(15,5))
sns.distplot(points_df['y'], bins=200);
plt.xlabel('y')
plt.title('Y distribution')
plt.show()




plt.figure(figsize=(15,5))
sns.distplot(points_df['z'], bins=200);
plt.xlabel('z')
plt.title('Z distribution')
plt.show()




plt.figure(figsize=(15,5))
sns.distplot(points_df['yaw'], bins=200);
plt.xlabel('yaw')
plt.title('Yaw distribution')
plt.show()




plt.figure(figsize=(15,5))
sns.distplot(points_df['pitch'], bins=200);
plt.xlabel('pitch')
plt.title('Pitch distribution')
plt.show()




plt.figure(figsize=(15,5))
sns.distplot(points_df['roll'].map(lambda x: rotate(x, np.pi)), bins=200);
plt.xlabel('roll rotated by pi')
plt.title('Roll distribution')
plt.show()




# calculate the correlation matrix
corr = points_df.corr()

# plot the heatmap
plt.figure(figsize=(15,15))
sns.heatmap(corr, annot=True,
            xticklabels=corr.columns,
            yticklabels=corr.columns)
plt.title('Correlation matrix')
plt.show()




# We define simple linear regression with z and y:
zy_slope = LinearRegression()
X = points_df[['z']]
y = points_df['y']
zy_slope.fit(X, y)
print('MAE without x:', mean_absolute_error(y, zy_slope.predict(X)))

# with x, y and z
xzy_slope = LinearRegression()
X = points_df[['x', 'z']]
y = points_df['y']
xzy_slope.fit(X, y)
print('MAE with x:', mean_absolute_error(y, xzy_slope.predict(X)))

print('\ndy/dx = {:.3f}\ndy/dz = {:.3f}'.format(*xzy_slope.coef_))




plt.figure(figsize=(25,5))
plt.xlim(0,500)
plt.ylim(0,100)
plt.scatter(points_df['z'], points_df['y'], label='Real points')
X_line = np.linspace(0,500, 10)
plt.plot(X_line, zy_slope.predict(X_line.reshape(-1, 1)), color='orange', label='Regression')
plt.legend()
plt.xlabel('z coordinate')
plt.ylabel('y coordinate')
plt.title('Correlation z and y')
plt.show()




name = train['ImageId'][0]
img = imread(PATH + 'train_images/' + name + '.jpg')
IMG_SHAPE = img.shape
plt.imshow(img)




plt.figure(figsize=(15, 15))
plt.imshow(imread(PATH + 'train_images/' + train['ImageId'][2217] + '.jpg'))
plt.scatter(*get_img_coords(train['PredictionString'][2217]), color='red', s=100)
plt.title('Sample center')




xs, ys = [], []

for ps in train['PredictionString']:
    x, y = get_img_coords(ps)
    xs += list(x)
    ys += list(y)

plt.figure(figsize=(15, 15))
plt.imshow(imread(PATH + 'train_images/' + train['ImageId'][2217] + '.jpg'), alpha=0.3)
plt.scatter(xs, ys, color='red', s=10, alpha=0.2)
plt.title('All centers distribution')
plt.show()




# Road points
road_width = 3
road_xs = [-road_width, road_width, road_width, -road_width, -road_width]
road_ys = [0, 0, 500, 500, 0]

plt.figure(figsize=(16,16))
plt.axes().set_aspect(1)
plt.xlim(-50, 50)
plt.ylim(0, 100)

# View road
plt.fill(road_xs, road_ys, alpha=0.3, color='gray')
plt.plot([road_width/2,road_width/2], [0,100], alpha=0.4, linewidth=4, color='white', ls='--')
plt.plot([-road_width/2,-road_width/2], [0,100], alpha=0.4, linewidth=4, color='white', ls='--')
# View cars
plt.scatter(points_df['x'], np.sqrt(points_df['z']**2 + points_df['y']**2), color='red', s=10, alpha=0.1);




n_rows = 10

for idx in range(n_rows):
    fig, axes = plt.subplots(1, 2, figsize=(20,20))
    img = imread(PATH + 'train_images/' + train['ImageId'].iloc[idx] + '.jpg')
    axes[0].imshow(img)
    img_vis = visualize(img, str2coords(train['PredictionString'].iloc[idx]))
    axes[1].imshow(img_vis)
    plt.show()




IMG_WIDTH = 512
IMG_HEIGHT = IMG_WIDTH // 16 * 5
MODEL_SCALE = 8




img0 = imread(PATH + 'train_images/' + train['ImageId'][0] + '.jpg')
img = preprocess_image(img0)

mask, regr = get_mask_and_regr(img0, train['PredictionString'][0])

print('img.shape', img.shape, 'std:', np.std(img))
print('mask.shape', mask.shape, 'std:', np.std(mask))
print('regr.shape', regr.shape, 'std:', np.std(regr))

plt.figure(figsize=(15, 15))
plt.title('Processed image')
plt.imshow(img)
plt.show()

plt.figure(figsize=(15, 15))
plt.title('Detection Mask')
plt.imshow(mask)
plt.show()

plt.figure(figsize=(15, 15))
plt.title('id values')
plt.imshow(regr[:,:,0])
plt.show()
t = 0
for i in range(len(regr[:,:,t])):
    for j in range(len(regr[:,:,t][i])):
        if regr[:,:,t][i][j] != 0:
            print(regr[:,:,t][i][j])

plt.figure(figsize=(15, 15))
plt.title('pitch values')
plt.imshow(regr[:,:,1])
plt.show()
t = 1
for i in range(len(regr[:,:,t])):
    for j in range(len(regr[:,:,t][i])):
        if regr[:,:,t][i][j] != 0:
            print(regr[:,:,t][i][j])

plt.figure(figsize=(15, 15))
plt.title('roll values')
plt.imshow(regr[:,:,2])
plt.show()
t = 2
for i in range(len(regr[:,:,t])):
    for j in range(len(regr[:,:,t][i])):
        if regr[:,:,t][i][j] != 0:
            print(regr[:,:,t][i][j])

plt.figure(figsize=(15, 15))
plt.title('x values')
plt.imshow(regr[:,:,3])
plt.show()
t = 3
for i in range(len(regr[:,:,t])):
    for j in range(len(regr[:,:,t][i])):
        if regr[:,:,t][i][j] != 0:
            print(regr[:,:,t][i][j])

plt.figure(figsize=(15, 15))
plt.title('y values')
plt.imshow(regr[:,:,4])
plt.show()
t = 4
for i in range(len(regr[:,:,t])):
    for j in range(len(regr[:,:,t][i])):
        if regr[:,:,t][i][j] != 0:
            print(regr[:,:,t][i][j])

plt.figure(figsize=(15, 15))
plt.title('Yaw values')
plt.imshow(regr[:,:,5])
plt.show()
t = 5
for i in range(len(regr[:,:,t])):
    for j in range(len(regr[:,:,t][i])):
        if regr[:,:,t][i][j] != 0:
            print(regr[:,:,t][i][j])

plt.figure(figsize=(15, 15))
plt.title('z values')
plt.imshow(regr[:,:,6])
plt.show()
t = 6
for i in range(len(regr[:,:,t])):
    for j in range(len(regr[:,:,t][i])):
        if regr[:,:,t][i][j] != 0:
            print(regr[:,:,t][i][j])




class CarDataset(Dataset):
    """Car dataset."""

    def __init__(self, dataframe, root_dir, training=True, transform=None, ID=False):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training
        self.ID=ID

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image name
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)
        
        # Augmentation
        flip = False
        if self.training:
            flip = np.random.randint(10) == 1
        
        # Read image
        img0 = imread(img_name, True)
        img = preprocess_image(img0, flip=flip)
        img = np.rollaxis(img, 2, 0)
        
        # Get mask and regression maps
        mask, regr = get_mask_and_regr(img0, labels, flip=flip)
        regr = np.rollaxis(regr, 2, 0)
        if self.ID:
            return [idx,img]
        return [img, mask, regr]




DISTANCE_THRESH_CLEAR = 2
BATCH_SIZE = 2

train_images_dir = PATH + 'train_images/{}.jpg'
test_images_dir = PATH + 'test_images/{}.jpg'

# we select a part of 
part_1, part_2 = train_test_split(train, test_size=0.5, random_state=42)
df_train, df_test = train_test_split(part_1, test_size=0.2, random_state=42)
df_val = test

# we create dataset objects
train_dataset = CarDataset(df_train, train_images_dir, training=True)
test_dataset = CarDataset(df_test, train_images_dir, training=False)
test_dataset2 = CarDataset(df_test, train_images_dir, training=False, ID=True)
val_dataset = CarDataset(df_val, test_images_dir, training=False)

# we create data generators - they will produce batches
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=2)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=2)
test_loader2 = DataLoader(dataset=test_dataset2,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=2)
val_loader = DataLoader(dataset=val_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=2)




img, mask, regr = train_dataset[0]

plt.figure(figsize=(15, 15))
plt.imshow(np.rollaxis(img, 0, 3))
plt.show()

plt.figure(figsize=(15, 15))
plt.imshow(mask)
plt.show()

plt.figure(figsize=(15, 15))
plt.title('id values')
plt.imshow(regr[0])
plt.show()
t = 0
for i in range(len(regr[t])):
    for j in range(len(regr[t][i])):
        if regr[t][i][j] != 0:
            print(regr[t][i][j])

plt.figure(figsize=(15, 15))
plt.title('pitch values')
plt.imshow(regr[1])
plt.show()
t = 1
for i in range(len(regr[t])):
    for j in range(len(regr[t][i])):
        if regr[t][i][j] != 0:
            print(regr[t][i][j])

plt.figure(figsize=(15, 15))
plt.title('roll values')
plt.imshow(regr[2])
plt.show()
t = 2
for i in range(len(regr[t])):
    for j in range(len(regr[t][i])):
        if regr[t][i][j] != 0:
            print(regr[t][i][j])

plt.figure(figsize=(15, 15))
plt.title('x values')
plt.imshow(regr[3])
plt.show()
t = 3
for i in range(len(regr[t])):
    for j in range(len(regr[t][i])):
        if regr[t][i][j] != 0:
            print(regr[t][i][j])

plt.figure(figsize=(15, 15))
plt.title('y values')
plt.imshow(regr[4])
plt.show()
t = 4
for i in range(len(regr[t])):
    for j in range(len(regr[t][i])):
        if regr[t][i][j] != 0:
            print(regr[t][i][j])

plt.figure(figsize=(15, 15))
plt.title('Yaw values')
plt.imshow(regr[5])
plt.show()
t = 5
for i in range(len(regr[t])):
    for j in range(len(regr[t][i])):
        if regr[t][i][j] != 0:
            print(regr[t][i][j])

plt.figure(figsize=(15, 15))
plt.title('z values')
plt.imshow(regr[6])
plt.show()
t = 6
for i in range(len(regr[t])):
    for j in range(len(regr[t][i])):
        if regr[t][i][j] != 0:
            print(regr[t][i][j])




class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x

def get_mesh(batch_size, shape_x, shape_y):
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mesh = torch.cat([torch.tensor(mg_x).to(device), torch.tensor(mg_y).to(device)], 1)
    return mesh




class MyUNet(nn.Module):
    '''Mixture of previous classes'''
    def __init__(self, n_classes):
        super(MyUNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        
        self.conv0 = double_conv(5, 64)
        self.conv1 = double_conv(64, 128)
        self.conv2 = double_conv(128, 512)
        self.conv3 = double_conv(512, 1024)
        
        self.mp = nn.MaxPool2d(2)
        
        self.up1 = up(1282 + 1024, 512)
        self.up2 = up(512 + 512, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        x0 = torch.cat([x, mesh1], 1)
        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))
        
        x_center = x[:, :, :, IMG_WIDTH // 8: -IMG_WIDTH // 8]
        feats = self.base_model.extract_features(x_center)
        bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(device)
        feats = torch.cat([bg, feats, bg], 3)
        
        # Add positional info
        mesh2 = get_mesh(batch_size, feats.shape[2], feats.shape[3])
        feats = torch.cat([feats, mesh2], 1)
        
        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.outc(x)
        return x




def criterion(prediction, mask, regr, size_average=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
    mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()
    
    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)
    
    # Sum
    loss = mask_loss + regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss




#copy from https://www.kaggle.com/its7171/metrics-evaluation-script
from math import sqrt, acos, pi, sin, cos
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import average_precision_score
def expand_df(df, PredictionStringCols):
    df = df.dropna().copy()
    df['NumCars'] = [int((x.count(' ')+1)/7) for x in df['PredictionString']]
    #fix nan bug
    df['PredictionString'].fillna('0.15699866 0.029135812202187722 -3.0876481888168534 15.061413706416168 4.49235176722995 27.384573221206665 0.6447297909095615',inplace=True)
    position = []
    for i in range(len(df['ImageId'])):
        position.append([list(df['ImageId'])[i], list(df['NumCars'])[i]])
    image_id_expanded = [item for item, count in position for i in range(count)]
    prediction_strings_expanded = df['PredictionString'].str.split(' ',expand = True).values.reshape(-1,7).astype(float)
    prediction_strings_expanded = prediction_strings_expanded[~np.isnan(prediction_strings_expanded).all(axis=1)]
    df = pd.DataFrame(
        {
            'ImageId': image_id_expanded,
            PredictionStringCols[0]:prediction_strings_expanded[:,0],
            PredictionStringCols[1]:prediction_strings_expanded[:,1],
            PredictionStringCols[2]:prediction_strings_expanded[:,2],
            PredictionStringCols[3]:prediction_strings_expanded[:,3],
            PredictionStringCols[4]:prediction_strings_expanded[:,4],
            PredictionStringCols[5]:prediction_strings_expanded[:,5],
            PredictionStringCols[6]:prediction_strings_expanded[:,6]
        })
    return df

def str2coords2(s, names):
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        position = []
        for i in range(len(names)):
            position.append([names[i], l.astype('float')[i]])
        coords.append(dict(position))
    return coords

def TranslationDistance(p,g, abs_dist = False):
    dx = p['x'] - g['x']
    dy = p['y'] - g['y']
    dz = p['z'] - g['z']
    diff0 = (g['x']**2 + g['y']**2 + g['z']**2)**0.5
    diff1 = (dx**2 + dy**2 + dz**2)**0.5
    if abs_dist:
        diff = diff1
    else:
        diff = diff1/diff0
    return diff

def RotationDistance(p, g):
    true=[ g['pitch'] ,g['yaw'] ,g['roll'] ]
    pred=[ p['pitch'] ,p['yaw'] ,p['roll'] ]
    q1 = R.from_euler('xyz', true)
    q2 = R.from_euler('xyz', pred)
    diff = R.inv(q2) * q1
    W = np.clip(diff.as_quat()[-1], -1., 1.)
    
    # in the official metrics code:
    # https://www.kaggle.com/c/pku-autonomous-driving/overview/evaluation
    #   return Object3D.RadianToDegree( Math.Acos(diff.W) )
    # this code treat θ and θ+2π differntly.
    # So this should be fixed as follows.
    W = (acos(W)*360)/pi
    if W > 180:
        W = 180 - W
    return W

def check_match(valid_df, train_df, thre_tr_dist, thre_ro_dist, keep_gt=False):
    position = []
    for i in range(len(train_df['ImageId'])):
        position.append([list(train_df['ImageId'])[i], list(train_df['PredictionString'])[i]])
    train_dict = {imgID:str2coords2(s, names=['carid_or_score', 'pitch', 'yaw', 'roll', 'x', 'y', 'z']) for imgID,s in position}
    position = []
    for i in range(len(valid_df['ImageId'])):
        position.append([list(valid_df['ImageId'])[i], list(valid_df['PredictionString'])[i]])
    valid_dict = {imgID:str2coords2(s, names=['pitch', 'yaw', 'roll', 'x', 'y', 'z', 'carid_or_score']) for imgID,s in position}
    result_flg = [] # 1 for TP, 0 for FP
    scores = []
    MAX_VAL = 10**15
    for img_id in valid_dict:
        for pcar in sorted(valid_dict[img_id], key=lambda x: -x['carid_or_score']):
            # find nearest GT
            min_tr_dist = MAX_VAL
            min_idx = -1
            for idx, gcar in enumerate(train_dict[img_id]):
                tr_dist = TranslationDistance(pcar,gcar)
                if tr_dist < min_tr_dist:
                    min_tr_dist = tr_dist
                    min_ro_dist = RotationDistance(pcar,gcar)
                    min_idx = idx
       
            # set the result
            if min_tr_dist < thre_tr_dist and min_ro_dist < thre_ro_dist:
                if not keep_gt:
                    train_dict[img_id].pop(min_idx)
                result_flg.append(1)
            else:
                result_flg.append(0)
            scores.append(pcar['carid_or_score'])
    
    return result_flg, scores



def calc_map_df(valid_df, nrows=None):
    expanded_valid_df = expand_df(valid_df, ['pitch','yaw','roll','x','y','z','Score'])


    train_df = pd.read_csv(PATH + 'train.csv')
    train_df = train_df[train_df.ImageId.isin(valid_df.ImageId.unique())]
    expanded_train_df = expand_df(train_df, ['model_type','pitch','yaw','roll','x','y','z'])
    n_gt = len(expanded_train_df)

    thres_ro_list = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
    thres_tr_list = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    ap_list = []
    
    position = []
    for i in range(len(thres_ro_list)):
        position.append([thres_ro_list[i], thres_tr_list[i]])
    for thre_ro_dist, thre_tr_dist in tqdm(position):
        abs_dist = False
        result_flg, scores = check_match(valid_df, train_df, thre_tr_dist, thre_ro_dist)
        n_tp = np.sum(result_flg)
        recall = n_tp/n_gt
        ap = average_precision_score(result_flg, scores)*recall
        ap_list.append(ap)
    return np.mean(ap_list)




#For validating,we fill some fixed codes in PredictionString field when it's nan
def fill_str(x):
    if type(x)==float or len(x)<2:
        return '0.15511163 0.025993261021686774 -3.1062442382150373 -15.10751805129461 12.073826862286817 70.47340792740864 0.5496648404696726'
    return x




IMG_WIDTH = 256
IMG_HEIGHT = IMG_WIDTH // 16 * 5
MODEL_SCALE = 8
DISTANCE_THRESH_CLEAR = 2
learning_rate = 0.001
n_epochs = 10

# Gets the GPU if there is one, otherwise the cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = MyUNet(8).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=n_epochs * len(train_loader) // 3, gamma=0.1)

# We could see the construction of the model:
model




IMG_WIDTH = 512
IMG_HEIGHT = IMG_WIDTH // 16 * 5
MODEL_SCALE = 8
DISTANCE_THRESH_CLEAR = 2
learning_rate = 0.001
n_epochs = 10
DISTANCE_THRESH_CLEAR = 2
BATCH_SIZE = 2

train_images_dir = PATH + 'train_images/{}.jpg'
test_images_dir = PATH + 'test_images/{}.jpg'

# we select a part of
df_train, df_test = train_test_split(train, test_size=0.2, random_state=42)
df_val = test

# we create dataset objects
train_dataset = CarDataset(df_train, train_images_dir, training=True)
test_dataset = CarDataset(df_test, train_images_dir, training=False)
test_dataset2 = CarDataset(df_test, train_images_dir, training=False, ID=True)
val_dataset = CarDataset(df_val, test_images_dir, training=False)

# we create data generators - they will produce batches
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=2)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=2)
test_loader2 = DataLoader(dataset=test_dataset2,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=2)
val_loader = DataLoader(dataset=val_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=2)



# Gets the GPU if there is one, otherwise the cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = MyUNet(8).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=n_epochs * len(train_loader) // 3, gamma=0.1)

history = pd.DataFrame()

for epoch in range(n_epochs):
    torch.cuda.empty_cache()
    gc.collect()
    # Training the model
    model.train()
    for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(tqdm(train_loader)):
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        regr_batch = regr_batch.to(device)
        
        optimizer.zero_grad()
        output = model(img_batch)
        loss = criterion(output, mask_batch, regr_batch)
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()    
        loss.backward()
        optimizer.step()
        exp_lr_scheduler.step()      
    print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        loss.data))

    # Evaluate the model
    model.eval()
    loss = 0
    with torch.no_grad():
        for img_batch, mask_batch, regr_batch in test_loader:
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)
            output = model(img_batch)
            loss += criterion(output,
                            mask_batch,
                            regr_batch,
                            size_average=False).data
    loss /= len(test_loader.dataset)
    
    if history is not None:
        history.loc[epoch, 'test_loss'] = loss.cpu().numpy()
    
    print('test loss: {:.4f}'.format(loss))
    # evaluate with mAP
    model.eval()
    ids=[]
    preds=[]
    
    with torch.no_grad():
        for ids_batch,img_batch in tqdm(test_loader2):
            img_batch = img_batch.to(device)
            output = model(img_batch).cpu().numpy()
            ids.extend(ids_batch)
            for out in output:
                predictions=[]
                coords = extract_coords(out)
                s = coords2str(coords)
                predictions.append(s)
                preds.append(' '.join(predictions))
        

    torch.cuda.empty_cache()  
    validation_prediction='valid_preds.csv'
    sub1=pd.DataFrame()
    sub1['ImageId']=ids
    sub1['PredictionString']=preds
    #fix nan bug
    sub1['PredictionString']=sub1['PredictionString'].apply(fill_str)
    sub1.to_csv(validation_prediction,index=False)
    map=calc_map_df(sub1,nrows=None)
    if history is not None:
        history.loc[epoch, 'test_map'] = map
    print('test map: ', map) 

series = history.dropna()['train_loss']
plt.figure(figsize=(15, 5))
plt.title('Train evaluation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(series.index, series)
plt.show()
best_value = series.min()
best_epoch = series[series == series.min()].index

series = history.dropna()['test_loss']
plt.figure(figsize=(15, 5))
plt.title('Test evaluation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(series.index, series)
plt.show()
best_value = series.min()
print('best value : ', best_value)
best_epoch = series[series == series.min()].index
print('for epoch : ', best_epoch)

series = history.dropna()['test_map']
plt.figure(figsize=(15, 5))
plt.title('Test mAP evaluation')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.plot(series.index, series)
plt.show()
best_value = series.max()
map_evaluation.append(best_value)
print('best value : ', best_value)
best_epoch = series[series == series.min()].index
print('for epoch : ', best_epoch)




torch.save(model.state_dict(), PATH + 'model.pth')




img, mask, regr = test_dataset[0]

plt.figure(figsize=(15,15))
plt.title('Input image')
plt.imshow(np.rollaxis(img, 0, 3))
plt.show()

plt.figure(figsize=(15,15))
plt.title('Ground truth mask')
plt.imshow(mask)
plt.show()

output = model(torch.tensor(img[None]).to(device))
logits = output[0,0].data.cpu().numpy()

plt.figure(figsize=(15,15))
plt.title('Model predictions')
plt.imshow(logits)
plt.show()

plt.figure(figsize=(15,15))
plt.title('Model predictions thresholded')
plt.imshow(logits > 0)
plt.show()




torch.cuda.empty_cache()
gc.collect()

for idx in range(15):
    img, mask, regr = test_dataset[idx]
    
    output = model(torch.tensor(img[None]).to(device)).data.cpu().numpy()
    coords_pred = extract_coords(output[0])
    coords_true = extract_coords(np.concatenate([mask[None], regr], 0))
    
    img = imread(train_images_dir.format(df_dev['ImageId'].iloc[idx]))
    
    fig, axes = plt.subplots(1, 2, figsize=(30, 30))
    axes[0].set_title('Ground truth')
    axes[0].imshow(visualize(img, coords_true))
    axes[1].set_title('Prediction')
    axes[1].imshow(visualize(img, coords_pred))
    plt.show()




predictions = []

val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model.eval()

for img, _, _ in tqdm(val_loader):
    with torch.no_grad():
        output = model(img.to(device))
    output = output.data.cpu().numpy()
    for out in output:
        coords = extract_coords(out)
        s = coords2str(coords)
        predictions.append(s)




test = pd.read_csv(PATH + 'sample_submission.csv')
test['PredictionString'] = predictions
test.to_csv('predictions.csv', index=False)
test.head()

