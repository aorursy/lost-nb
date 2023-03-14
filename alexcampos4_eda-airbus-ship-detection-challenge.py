#!/usr/bin/env python
# coding: utf-8



# Import libraries
import os, PIL

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')




# Set visualization style
plt.rcParams["patch.force_edgecolor"] = True
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)



# Read Data
df = pd.read_csv('../input/airbus-ship-detection/train_ship_segmentations.csv')
df.head()




# Features Engineering - We will use EncodedPixels in a second DataFrame bellow
df['Ships'] = df['EncodedPixels'].notnull()
df = df.groupby('ImageId').sum().reset_index()
df['ShipPresent'] = df['Ships'] > 0

df.head()




df.info()




# Features Engineering - Second DataFrame with EncodedPixels and only images with ships

df_box = pd.read_csv('../input/airbus-ship-detection/train_ship_segmentations.csv')
df_box = df_box.dropna().groupby("ImageId")[['EncodedPixels']].agg(lambda rle_code: ' '.join(rle_code)).reset_index()
df_box['Path'] = df_box['ImageId'].apply(lambda filename: os.path.join('../input/airbus-ship-detection/train/', filename))
df_box.info()




df_box.head()




def rle_to_pixels(rle_code):
    ''' This function decodes Run Lenght Encoding into pixels '''
    rle_code = [int(i) for i in rle_code.split()]
    
    pixels = [(pixel_position % 768, pixel_position // 768) 
              for start, length in list(zip(rle_code[0:-1:2], rle_code[1::2])) 
              for pixel_position in range(start, start + length)]
        
    return pixels

def apply_mask(image, mask):
    ''' This function saturates the Red and Green RGB colors in the image 
        where the coordinates match the mask'''
    for x, y in mask:
        image[x, y, [0, 1, 2]] = (255, 255, 0)
    return image




# Plots with masked ships on random images from the dataset

h, w = 3, 3
load_img = lambda path: np.array(PIL.Image.open(path))
fig, axes_list = plt.subplots(h, w, figsize=(4*h, 4*w))

for axes in axes_list:
    for ax in axes:
        ax.axis("off")
        path = np.random.choice(df_box['Path'])
        img = apply_mask(load_img(path),                 rle_to_pixels(df_box[df_box['Path'] == path]['EncodedPixels'].iloc[0]))
        ax.imshow(img)
        ax.set_title(df_box[df_box['Path'] == path]['ImageId'].iloc[0])




df_box.info()




# Imbalanced Dataset | Ship/No-Ship Ratio

total_images = len(df)
ships = df['Ships'].sum()
ships_images = len(df[df['Ships'] > 0])
no_ship = total_images - ships_images

print(f"Images: {total_images} \nShips:  {ships}")
print(f"Images with ships:    {round(ships_images/total_images,2)} ({ships_images})")
print(f"Images with no ships: {round(no_ship/total_images,2)} ({no_ship})")




# Engineering Features for the graphs

ship_ratio = df['ShipPresent'].value_counts()/total_images
ship_ratio = ship_ratio.rename(index={True:'Ship', False:'No Ship'})

total_ship_distribution = df['Ships'].value_counts()[1::]/ships_images




# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 12), gridspec_kw={'width_ratios':[1,5]})

ship_ratio.plot.bar(ax=axes[0], title="Ship/No-Ship distribution")
total_ship_distribution.plot.bar(ax=axes[1], title="Total Ship Distribution")

axes[0].title.set_size(30)
axes[1].title.set_size(30)




# The operation bellow is expensive, if possible just load the pre-calculated dataset

# df_box['Pixels'] = df_box['EncodedPixels'].apply(rle_to_pixels).str.len() # EXPENSIVE
# df_box.to_csv('train_box_pixels.csv', encoding='utf-8', index=False)
df_box = pd.read_csv('../input/airbus-challenge/train_box_pixels.csv')
df_box.head()




# Imbalanced Dataset | Ship/No-Ship Pixels Ratio
# Due to the heavy imbalance of the dataset, we'll conduct our analysis only with ship images

n_images = df_box['ImageId'].nunique()
ship_pixels = df_box['Pixels'].sum()
total_pixels = n_images * 768 * 768
ratio = ship_pixels/total_pixels

print(f'Ship Pixels:   {round(ratio, 3)*100}%    ({ship_pixels})')
print(f'Total Pixels: {round(1 - ratio, 3)*100}% ({total_pixels - ship_pixels})')

