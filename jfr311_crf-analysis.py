#!/usr/bin/env python
# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import jaccard

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, unary_from_softmax, create_pairwise_gaussian




ls ../input -a




get_ipython().run_line_magic('matplotlib', 'inline')




H = W = 96




def iou(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    return(1 - jaccard(y_true, y_pred))




images = np.load('../input/validation-set/x.npy')
true_masks = np.load('../input/validation-set/y.npy')[..., 0]
mask_probabilities = np.load('../input/validation-set/preds_valid.npy')[..., 0]




ix = np.random.randint(images.shape[0])
ix = 87
img = images[ix, ..., 0]
mask = true_masks[ix]
mask_proba = mask_probabilities[ix]
mask_probas = np.rollaxis(np.stack([1 - mask_proba, mask_proba], axis = 2), 2, 0)




threshold = 0.69
mask_pred = np.int32(mask_proba > threshold)




f, (ax1, ax2) = plt.subplots(2, 2, sharey=True,sharex=True)
# ax1.set_aspect('equal')
ax1[0].imshow(img, cmap='seismic'); ax1[0].axis('off'); ax1[0].set_title('Input Image')
ax1[1].imshow(mask); ax1[1].axis('off'); ax1[1].set_title('Ground Truth')
ax2[0].imshow(mask_proba); ax2[0].axis('off'); ax2[0].set_title('Mask Probabilities')
ax2[1].imshow(mask_pred); ax2[1].axis('off'); ax2[1].set_title('Mask Prediction')
# plt.subplots_adjust(wspace=0.8)
plt.show()




initial_iou = iou(mask, mask_pred)




d_l = dcrf.DenseCRF2D(H, W, 2)
d_p = dcrf.DenseCRF2D(H, W, 2)




U_from_labels = unary_from_labels(mask_pred, 2, gt_prob=0.7, zero_unsure=False)
U_from_proba = unary_from_softmax(mask_probas)




d_l.setUnaryEnergy(U_from_labels)
d_p.setUnaryEnergy(U_from_proba)




Q_l = d_l.inference(10)
Q_p = d_p.inference(10)




map_l = np.argmax(Q_l, axis=0).reshape((H, W))
map_p = np.argmax(Q_p, axis=0).reshape((H, W))




f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(U_from_labels.reshape((2, H,W))[0]); ax1.axis('off'); ax1.set_title('Unary from labels')
ax2.imshow(map_l); ax2.axis('off'); ax2.set_title('MAP from labels');




iou(mask, map_l) - initial_iou




f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(U_from_proba.reshape((2, H,W))[0]); ax1.axis('off'); ax1.set_title('Unary from proba')
ax2.imshow(map_p); ax2.axis('off'); ax2.set_title('MAP from proba');




iou(mask, map_p) - initial_iou




pairwise_bilateral = create_pairwise_bilateral(sdims=(5, 5), schan=(0.01,), img=np.expand_dims(img, -1), chdim=2)




d_l = dcrf.DenseCRF2D(H, W, 2)
d_p = dcrf.DenseCRF2D(H, W, 2)




d_l.setUnaryEnergy(U_from_labels)
d_l.addPairwiseEnergy(pairwise_bilateral, compat=10)

d_p.setUnaryEnergy(U_from_proba)
d_p.addPairwiseEnergy(pairwise_bilateral, compat=10)




def run_inference(d):
    Q, tmp1, tmp2 = d.startInference()
    for _ in range(2):
        d.stepInference(Q, tmp1, tmp2)
    kl1 = d.klDivergence(Q) / (H*W)
    map_soln1 = np.argmax(Q, axis=0).reshape((H,W))

    for _ in range(8):
        d.stepInference(Q, tmp1, tmp2)
    kl2 = d.klDivergence(Q) / (H*W)
    map_soln2 = np.argmax(Q, axis=0).reshape((H,W))

    for _ in range(16):
        d.stepInference(Q, tmp1, tmp2)
    kl3 = d.klDivergence(Q) / (H*W)
    map_soln3 = np.argmax(Q, axis=0).reshape((H,W))
    return(map_soln1, kl1, map_soln2, kl2, map_soln3, kl3)




map_soln1, kl1, map_soln2, kl2, map_soln3, kl3 = run_inference(d_l)

img_en = pairwise_bilateral.reshape((-1, H, W))  # Reshape just for plotting
plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow(map_soln1);
plt.title('MAP Solution with DenseCRF\n(2 steps, KL={:.2f})'.format(kl1)); plt.axis('off');
plt.subplot(1,3,2); plt.imshow(map_soln2);
plt.title('MAP Solution with DenseCRF\n(8 steps, KL={:.2f})'.format(kl2)); plt.axis('off');
plt.subplot(1,3,3); plt.imshow(map_soln3);
plt.title('MAP Solution with DenseCRF\n(16 steps, KL={:.2f})'.format(kl3)); plt.axis('off');




iou(mask, map_soln3) - initial_iou




map_soln1, kl1, map_soln2, kl2, map_soln3, kl3 = run_inference(d_p)

img_en = pairwise_bilateral.reshape((-1, H, W))  # Reshape just for plotting
plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow(map_soln1);
plt.title('MAP Solution with DenseCRF\n(2 steps, KL={:.2f})'.format(kl1)); plt.axis('off');
plt.subplot(1,3,2); plt.imshow(map_soln2);
plt.title('MAP Solution with DenseCRF\n(8 steps, KL={:.2f})'.format(kl2)); plt.axis('off');
plt.subplot(1,3,3); plt.imshow(map_soln3);
plt.title('MAP Solution with DenseCRF\n(16 steps, KL={:.2f})'.format(kl3)); plt.axis('off');




iou(mask, map_soln3) - initial_iou

