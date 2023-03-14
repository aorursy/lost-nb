#!/usr/bin/env python
# coding: utf-8

# In[ ]:


ls ../input/


# In[ ]:


import os
train = os.listdir("../input/train")
test = os.listdir("../input/test")
print(f"Train files: {len(train)}. ---> {train[:3]}")
print(f"Test files :  {len(test)}. ---> {test[:3]}")


# In[ ]:


import PIL # We will import the packages at "use-time (just for this kernel)

PIL.Image.open("../input/train/000c34352.jpg")


# In[ ]:


PIL.Image.open("../input/train/000c34352.jpg").size


# In[ ]:


PIL.Image.open('../input/train/000c34352.jpg').resize((200, 200))


# In[ ]:


import numpy as np

# Taking a shrinked version of the image to avoid unnecessary computation
img = PIL.Image.open('../input/train/000c34352.jpg').resize((200, 200))

rgb_pixels = np.array(img)
rgb_pixels.shape


# In[ ]:


# Red saturation of the top-left most 2x2 square pixels
rgb_pixels[0:2, 0:2, 0]


# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(rgb_pixels);


# In[ ]:


# Note that imshow can plot a 1-channel image on monochrome
plt.imshow(rgb_pixels[:, :, 1], cmap='Greys');


# In[ ]:


# And also:
plt.imshow(np.random.random(size=(10, 10)));


# In[ ]:


plt.imshow(rgb_pixels)
plt.title("Full image"); # We can add a title with plt.title()
plt.show()

plt.imshow(rgb_pixels[0:200,0:100]) # And with can crop the image using standard python slicing
plt.title("Left half");
plt.show()

plt.imshow(rgb_pixels[0:200,100:200])
plt.title("Right half");
plt.show()


# In[ ]:


# these two variables are "the parameters" of this cell
w = 6
h = 6

# this function uses the open, resize and array functions we have seen before
load_img = lambda filename: np.array(PIL.Image.open(f"../input/train/{filename}").resize((200, 200)))

_, axes_list = plt.subplots(h, w, figsize=(2*w, 2*h)) # define a grid of (w, h)

for axes in axes_list:
    for ax in axes:
        ax.axis('off')
        img = np.random.choice(train) # take a random train filename (like 000c34352.jpg)
        ax.imshow(load_img(img)) # load and show
        ax.set_title(img)
        


# In[ ]:




