#!/usr/bin/env python
# coding: utf-8



# coding = utf-8
import mxnet as mx
from mxnet.gluon import data, HybridBlock, nn
import pandas as pd
import cv2
import os
import numpy as np
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.model_zoo import vision
import glob
from mxnet import nd as F, gluon
from gluoncv import model_zoo as gm




get_ipython().run_line_magic('pylab', 'inline')




ls ../input/mxnet-gluon-baseline/model/




from gluoncv.model_zoo.resnetv1b import resnet50_v1s, resnet101_v1s, resnet152_v1s
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock

class ResNetBackbone(mx.gluon.HybridBlock):
    def __init__(self, backbone='resnet50', pretrained_base=True,dilated=True, **kwargs):
        super(ResNetBackbone, self).__init__()

        with self.name_scope():
            if backbone == 'resnet50':
                pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
            elif backbone == 'resnet101':
                pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
            elif backbone == 'resnet152':
                pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
            else:
                raise RuntimeError(f'unknown backbone: {backbone}')

            self.conv1 = pretrained.conv1
            self.bn1 = pretrained.bn1
            self.relu = pretrained.relu
            self.maxpool = pretrained.maxpool
            self.layer1 = pretrained.layer1
            self.layer2 = pretrained.layer2
            self.layer3 = pretrained.layer3
            self.layer4 = pretrained.layer4

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1, c2, c3, c4




import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock

class ResNetSteel(mx.gluon.HybridBlock):
    def __init__(self, backbone= 'resnet50', num_classes=4, backbone_lr_mult=0.1, **kwargs):
        super(ResNetSteel, self).__init__()

        self.backbone_name = backbone
        self.backbone_lr_mult = backbone_lr_mult
        self._kwargs = kwargs

        with self.name_scope():
            self.backbone = ResNetBackbone(backbone=self.backbone_name, pretrained_base=False, dilated=False, **kwargs)

            self.head = Classification_head(output_channels=256, num_classes=num_classes)

    def load_pretrained_weights(self):
        pretrained = ResNetBackbone(backbone=self.backbone_name, pretrained_base=True, dilated=False, **self._kwargs)
        backbone_params = self.backbone.collect_params()
        pretrained_weights = pretrained.collect_params()
        for k, v in pretrained_weights.items():
            param_name = backbone_params.prefix + k[len(pretrained_weights.prefix):]
            backbone_params[param_name].set_data(v.data())

        self.backbone.collect_params().setattr('lr_mult', self.backbone_lr_mult)

    def hybrid_forward(self,F, x):
        c1, c2, c3, c4 = self.backbone(x)
        logits = self.head(c4)

        return logits

class Classification_head(HybridBlock):
    def __init__(self, output_channels=256, num_classes=4):
        super(Classification_head, self).__init__()

        with self.name_scope():
            self.cls_head = nn.HybridSequential()
            self.cls_head.add(ConvBlock(output_channels, kernel_size=1))
            self.cls_head.add(nn.GlobalAvgPool2D())
            self.cls_head.add(nn.Conv2D(num_classes, kernel_size=1))

    def hybrid_forward(self, F, x):
        logits = self.cls_head(x)

        return F.squeeze(logits)


class ConvBlock(HybridBlock):
    def __init__(self, output_channels, kernel_size, padding=0, activation='relu', norm_layer=nn.BatchNorm):
        super().__init__()
        self.body = nn.HybridSequential()
        self.body.add(
            nn.Conv2D(output_channels, kernel_size=kernel_size, padding=padding, activation=activation),
            norm_layer(in_channels=output_channels)
        )

    def hybrid_forward(self, F, x):
        return self.body(x)




ctx = mx.gpu()
unet = ResNetSteel(num_classes=4)
unet.collect_params().initialize()
unet.load_parameters('../input/mxnet-gluon-classification/unet_4_0.0.params')
unet.collect_params().reset_ctx(ctx)




def mask2rle(mask):
    if np.sum(mask) == 0: return ''
    ar = mask.flatten(order='F')
    EncodedPixel = ''
    l = 0
    for i in range(len(ar)):
        if ar[i] == 0:
            if l > 0:
                if EncodedPixel != '': EncodedPixel += ' '
                EncodedPixel += str(st+1)+' '+str(l)
                l = 0
        else: # == 1
            if l == 0: st = i
            l += 1
    return EncodedPixel




import cv2
def remove_small_one(predict, min_size):
    H,W = predict.shape
    num_component, component = cv2.connectedComponents(predict.astype(np.uint8))
    predict = np.zeros((H,W), np.bool)
    for c in range(1,num_component):
        p = (component==c)
        if p.sum()>min_size:
            predict[p] = True
    return predict




def sharpen(p,t=0.5):
        if t!=0:
            return p**t
        else:
            return p




import random
import time
# test_stage
trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)
                )
            ]
        )

test_dir = "../input/severstal-steel-defect-detection/test_images/"

imglists = glob.glob(test_dir + '/*g')
oriims = []
preds = []
random.shuffle(imglists)
ImageId_ClassIds = []
EncodedPixels = []

augs = ['flip_lr', 'flip_ud']
from tqdm import tqdm
thresholds = [0.5, 0.5, 0.5, 0.5]
min_area = [600, 600, 1000, 2000]
# min_area = [1, 1, 1, 1]
t1 = time.time()
for i, item in enumerate(tqdm(imglists)):
    timg = cv2.imread(item)[:, :, ::-1]
    img = mx.nd.array([timg])
    input_img = trans(img)
    num_aug = 0
    
    # if 1:
    out = unet(input_img.as_in_context(ctx))
    
    out = F.sigmoid(out, axis=1)
    out_mask = out
    pred_inds = F.sum(out)
    oriims.append(timg)
    preds.append(pred_inds)
        
    # if 'flip_lr' in augs:
    #     input_img_lr = F.flip(input_img, axis=3)
    #     out = unet(input_img_lr.as_in_context(ctx))
    #     out = F.softmax(out, axis=1)
    #     out_mask += sharpen(F.flip(out, axis=3))
    #     num_aug += 1

    # if 'flip_ud' in augs:
    #     input_img_lr = F.flip(input_img, axis=2)
    #     out = unet(input_img_lr.as_in_context(ctx))
    #     out = F.softmax(out, axis=1)
    #     out = F.where(out > 0.5, out, F.zeros_like(out))
    #     out_mask += sharpen(F.flip(out, axis=2))
    #     num_aug += 1

    out = out.asnumpy()
    ImageId = item.split('/')[-1]
    pred_inds = pred_inds.asnumpy()
    for j in range(4):

        Id = ImageId + '_'+str(j+1)
        tmp_mask = np.where(out[j] > thresholds[j], 1.0, 0)

        if np.sum(tmp_mask) > 0.5:
            EncodedPixel = '2 2'
        if np.sum(tmp_mask) < 0.5:
            EncodedPixel = ''

        ImageId_ClassIds.append(Id)
        EncodedPixels.append(EncodedPixel)
dur = time.time() - t1
print("cost time:{}".format(dur))




submission =  pd.read_csv("../input/severstal-steel-defect-detection/sample_submission.csv")
print(len(ImageId),len(submission['ImageId_ClassId']))
# len(set(submission['ImageId_ClassId'])-set(Ids))
# assert set(Ids) == set(submission['ImageId_ClassId'])

for i, encoded in zip(ImageId_ClassIds,EncodedPixels):
    submission.loc[submission['ImageId_ClassId']==i,["EncodedPixels"]] =  encoded

submission.to_csv('submission.csv',index=False)




submission.head(10)




fig, ax1 = plt.subplots(figsize=(50, 50))
for i, (timg, pred_inds) in enumerate(zip(oriims[:100], preds[:100])):
#     plt.subplot(len(oriims[:100])*2, 1, i*2+1)
#     plt.imshow(timg)
#     plt.subplot(len(oriims[:100])*2, 1, i*2+2)
#     plt.imshow(pred_inds[0].asnumpy())
    # seg_map = np.expand_dims(pred_inds[0].asnumpy(), axis=2)
    # seg_map_3c=np.repeat(seg_map, 3, 2)*255
    # h, w = timg.shape[:2]
    # seg_map_3c = cv2.resize(seg_map_3c, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    # att_im = cv2.addWeighted(seg_map_3c.astype(np.uint8), 0.5, timg, 0.5, 0.0)
    if i > 10:
        break
    plt.subplot(11, 1, i+1)
    plt.title(str(pred_inds))
    plt.imshow(timg)

