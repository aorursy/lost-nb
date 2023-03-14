#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from zipfile import ZipFile
from io import BytesIO

# Image manipulation.
import PIL.Image
from IPython.display import display




def load_image_blackandwhite(filename):
    #image = PIL.Image.open(filename)

    image = PIL.Image.open(filename) # open colour image
    image = image.convert('L') # convert image to black and white
    image = np.array(image)
    
    return np.float32(image)




def plot_image(image):
    # Assume the pixel-values are scaled between 0 and 255.
    
    # Convert the pixel-values to the range between 0.0 and 1.0
    image = np.clip(image/255.0, 0.0, 1.0)
        
    # Plot using matplotlib.
    plt.imshow(image, interpolation='lanczos')
    plt.show()




def reshape_image(image_file, new_wigth, new_height):
    
    ############
    # Reduce Size of Image
    ############
    
    olddim = np.shape(image_file)
    img = np.zeros((new_wigth,new_height))
    newdim = np.shape(img)
        
    for r in range(newdim[0]):
        if (newdim[0] <= olddim[0]):
            centerx=(r)/newdim[0]*olddim[0]
            lowerx=max(0,int(round(centerx-olddim[0]/newdim[0]/2,0)))
            upperx=min(olddim[0],int(round(centerx+olddim[0]/newdim[0]/2,0))+1)
        else:
            lowerx=max(0,int(r*olddim[0]/newdim[0]))
            upperx=min(lowerx+1,olddim[0]-1)+1
            
        for c in range(newdim[1]):  
            if (newdim[1] <= olddim[1]):
                centery=(c)/newdim[1]*olddim[1]
                lowery=max(0,int(round(centery-olddim[1]/newdim[1]/2,0)))
                uppery=min(olddim[1],int(round(centery+olddim[1]/newdim[1]/2,0))+1)
            else:
                lowery=max(0,int(c*olddim[1]/newdim[1]))
                uppery=min(lowery+1,olddim[1]-1)+1
            img[r,c] = np.mean(image_file[ lowerx:upperx, lowery:uppery ])

                
    return img




archive = ZipFile("../input/train.zip", 'r')
archive.namelist()[0:5]




['train/',
 'train/000bec180eb18c7604dcecc8fe0dba07.jpg',
 'train/001513dfcb2ffafc82cccf4d8bbaba97.jpg',
 'train/001cdf01b096e06d78e9e5112d419397.jpg',
 'train/00214f311d5d2247d5dfe4fe24b2303d.jpg']




image = load_image_blackandwhite(filename=BytesIO(archive.read(archive.namelist()[150])))
image.shape




(500, 333)




The shape of the image is 500 pixels x 333 pixels. If you try another image, you will see that the shape might change.




plot_image(image)




image_reshaped = reshape_image(image_file=image, new_wigth = 100, new_height = 100)
plot_image(image_reshaped)

