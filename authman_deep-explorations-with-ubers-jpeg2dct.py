#!/usr/bin/env python
# coding: utf-8



import torch
import numpy as np
import matplotlib.pyplot as plt




get_ipython().system('rm /opt/conda/lib/libjpeg*')
get_ipython().system('cp /usr/lib/x86_64-linux-gnu/libjpeg* /opt/conda/lib/')

get_ipython().system('git clone https://github.com/uber-research/jpeg2dct')

# Data Engineering PRO TIP!
# Always use `pip install .` rather than `python setup.py install`
# This way, you can later uninstall with pip, if desired.

get_ipython().system('pip install jpeg2dct/.')




# Let's verify it linked to the right jpeglib version...
get_ipython().system('ldd /opt/conda/lib/python3.7/site-packages/jpeg2dct/common/common_lib.cpython-37m-x86_64-linux-gnu.so')




from jpeg2dct.numpy import load, loads

imgs = get_ipython().getoutput('ls ../input/alaska2-image-steganalysis/Cover')

jpeg_file = f'../input/alaska2-image-steganalysis/Cover/{imgs[0]}'
dct_y, dct_cb, dct_cr = load(jpeg_file)
print ("Y component DCT shape {} and type {}".format(dct_y.shape, dct_y.dtype))
print ("Cb component DCT shape {} and type {}".format(dct_cb.shape, dct_cb.dtype))
print ("Cr component DCT shape {} and type {}".format(dct_cr.shape, dct_cr.dtype))




def DCT_Basis(repeats=None):
    N = 8
    basis = []
    for u in range(8):
        for v in range(8):
            z = np.zeros((N,N))
            for i in range(N):
                for j in range(N):
                    z[i,j] = np.cos(np.pi*(2*i+1)*u / (2*N)) * np.cos(np.pi*(2*j+1)*v / (2*N))
            basis.append(z)

    if repeats is None:
        return basis
    
    return torch.Tensor([basis,basis,basis]).permute((1,0,2,3))




basis = DCT_Basis()

_, ax = plt.subplots(8,8,figsize=(12,12))
ax = ax.flatten()
for i in range(64):
    ax[i].imshow(basis[i], vmin=-1, vmax=1)
plt.show()




conv = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=8, stride=8)# , groups=3)
conv.weight.data.shape




conv.weight.data = DCT_Basis(repeats=3)
conv.required_grad = False
conv.weight.data.shape




from PIL import Image

# The original image
image = Image.open('../input/alaska2-image-steganalysis/Cover/00001.jpg')
plt.imshow(image)




_, ax = plt.subplots(8,8,figsize=(32,32))

ax = ax.flatten()
for i in range(64):
    ax[i].imshow(dct_y[:,:,i])
    
plt.axis("off")
plt.axis("tight")
plt.axis("image") 
plt.show()




_, ax = plt.subplots(8,8,figsize=(32,32))

ax = ax.flatten()
for i in range(64):
    ax[i].imshow(dct_cb[:,:,i])
    
plt.axis("off")
plt.axis("tight")
plt.axis("image") 
plt.show()




_, ax = plt.subplots(8,8,figsize=(32,32))

ax = ax.flatten()
for i in range(64):
    ax[i].imshow(dct_cr[:,:,i])
    
plt.axis("off")
plt.axis("tight")
plt.axis("image") 
plt.show()






