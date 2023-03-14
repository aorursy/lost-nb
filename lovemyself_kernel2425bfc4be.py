#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
sys.path.insert(0, "/kaggle/input/deepfake")
sys.path.insert(0, "/kaggle/input/testpy")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# In[2]:


import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
from functions import *
import pandas as pd

print(torch.cuda.is_available())


# In[3]:


data_path = ["../input/deepfake-detection-challenge/test_videos"]  # define UCF-101 RGB data path

# encoder model and rnn classification model
cnn_path = "/kaggle/input/resnet50/cnn_encoder_epoch_211.pth"
rnn_path = "/kaggle/input/resnet50/rnn_decoder_epoch_211.pth"
# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.0       # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 2             # number of target category
epochs = 201        # training epochs
batch_size = 32 
learning_rate = 1e-3
log_interval = 1   # interval for displaying training info

# Select which frame to begin & end in videos
# begin_frame, end_frame, skip_frame = 1, 29, 1
frame_len = 39 # the number of frames used to input network for each video.


##############################################################################################
# dirs_0      -- paths of videos
# v_id        -- id of test video
# net         -- retinaface net work
# transform   -- the function of image preprocessing
# device      -- GPU 
# frames      -- the index of obtained frames
# length_frames -- number of frames
# batch_size  -- batch_size == 1
# is_multi    -- True (multiple faces in one video)

def get_data(dirs_0, v_id, net, transform, device, frames, length_frames, batch_size, is_multi):
    results = []
    res_y = []
    net.eval()
    net = net.to(device)
    for id in range(0, batch_size):
        # Select sample

        dirs = dirs_0
        folder = dirs[v_id] # get the video path

        videoCapture = cv2.VideoCapture(folder)  # read video

        fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT) # frame number

        skip_f = -1  # obtain frames from the first frame

        success, frame = videoCapture.read()
        n_frame = 0

        res_list = []
        index = 0
        is_null = True
        nums_temp = 0
        sizes = frame.shape[0:2]
        ranges = min(sizes)/30.0  # used to produce multiple faces in each video
        res = []
        stackw = []
        stackh = []
        indexs = []
       
        while success :

            if n_frame > skip_f:

                img_raw = frame

                img = np.float32(img_raw)

                # testing scale
                target_size = 1600
                max_size = 2150
                im_shape = img.shape
                im_size_min = np.min(im_shape[0:2])
                im_size_max = np.max(im_shape[0:2])
                resize = float(target_size) / float(im_size_min)
                # prevent bigger axis from being more than max_size:
                if np.round(resize * im_size_max) > max_size:
                    resize = float(max_size) / float(im_size_max)
                resize = 1

                if resize != 1:
                    img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
                im_height, im_width, _ = img.shape
                scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                img -= (104, 117, 123)
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).unsqueeze(0)
                img = img.cuda()
                scale = scale.cuda()

            
                loc, conf, landms = net(img)  # forward pass
                priorbox = PriorBox(cfg, image_size=(im_height, im_width))
                priors = priorbox.forward()
                priors = priors.cuda()
                prior_data = priors.data
                boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
                boxes = boxes * scale / resize
                boxes = boxes.cpu().numpy()
                scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
                landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
                scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                       img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                       img.shape[3], img.shape[2]])
                scale1 = scale1.cuda()
                landms = landms * scale1 / resize
                landms = landms.cpu().numpy()

                # ignore low scores
                inds = np.where(scores > 0.02)[0]
                boxes = boxes[inds]
                landms = landms[inds]
                scores = scores[inds]

                # keep top-K before NMS
                order = scores.argsort()[::-1]
                # order = scores.argsort()[::-1][:args.top_k]
                boxes = boxes[order]
                landms = landms[order]
                scores = scores[order]

                # do NMS
                dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                keep = py_cpu_nms(dets, 0.4)
                # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
                dets = dets[keep, :]
                landms = landms[keep]

                dets = np.concatenate((dets, landms), axis=1)

                for b in dets:
                    if b[4] < 0.7:
                        continue
                    b = list(map(int, b))
                    for j in range(0, 4):
                        if b[j] < 0:
                            b[j] = 0

                    w = (b[2]-b[0])/2+b[0]
                    h = (b[3]-b[1])/2+b[1]
                    if index == 0:
                        stackw.append(w)
                        stackh.append(h)
                        crop_img = img_raw[b[1]:b[3], b[0]:b[2]]
                        crop_img = Image.fromarray(cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)) # bgr to rgb
                        temp = transform(crop_img)
                        res.append([])
                        res[0].append(temp)
                        nums_temp = nums_temp + 1
                        indexs.append(1)
                        index = index + 1
                    else:
                        w_ = [stackw[j]-w for j in range(0, len(stackw))]
                        h_ = [stackh[j]-h for j in range(0, len(stackh))]
                        d2 = [w_[j]**2 for j in range(0, len(w_))]
                        d2_ = [h_[j]**2 for j in range(0, len(h_))]
                        d2 = [d2[k]+d2_[k] for k in range(0, len(d2))]
                        d = [math.sqrt(s) for s in d2]
                        r = min(d)
                        l = d.index(r)
                        if r < ranges:
                            crop_img = img_raw[b[1]:b[3], b[0]:b[2]]
                            crop_img = Image.fromarray(cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB))

                            temp = transform(crop_img)
                            res[l].append(temp)
                            nums_temp = nums_temp + 1
                            stackw[l] = w
                            stackh[l] = h
                            indexs[l] = indexs[l] + 1
                        else:
                            stackw.append(w)
                            stackh.append(h)
                            crop_img = img_raw[b[1]:b[3], b[0]:b[2]]
                            crop_img = Image.fromarray(cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB))

                            temp = transform(crop_img)
                            res.append([])
                            res[-1].append(temp)

                            nums_temp = nums_temp + 1
                            indexs.append(1)

                if indexs != []:
                    if max(indexs) == length_frames:
                        break
            success, frame = videoCapture.read()
            n_frame = n_frame + 1

        videoCapture.release()
        if not is_multi:
            for i in range(0, len(indexs)):
                if indexs[i] == length_frames:
                    results.append(torch.stack(res[i], dim=0))
                    break
        else:
            for i in range(0, len(indexs)):
                if indexs[i] == length_frames:
                    is_null = False
                    results.append(torch.stack(res[i], dim=0))
    

    return torch.stack(results, dim=0), is_null


#############################################################################
# model      -- cnn_encoder and rnn_decoder
# device     -- GPU
# X          -- input data

def validation(model, device, X):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        if True:
            X = X.to(device)

            output = rnn_decoder(cnn_encoder(X))

            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            return output, y_pred


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# Data loading parameters
# params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 1, 'pin_memory': True} if use_cuda else {}



transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


selected_frames = []
for i in range(0,frame_len):
    selected_frames.append(i)


# Create model
cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

cnn_encoder.load_state_dict(torch.load(cnn_path))
rnn_decoder.load_state_dict(torch.load(rnn_path))


# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)

    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) +                   list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) +                   list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())

elif torch.cuda.device_count() == 1:
    print("Using", torch.cuda.device_count(), "GPU!")
    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) +                   list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) +                   list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())

optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)


# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []


is_multi=True
# cfg = cfg_re50
cfg = cfg_mnet
net = RetinaFace(cfg=cfg, phase = 'test')        

# net = load_model(net,'models/Resnet50_Final.pth', False) # when cfg = cfg_re50
net = load_model(net,'models/mobilenet0.25_Final.pth', False)  # when cfg = cfg_mnet

net.eval()

data_path = data_path

dirs_0 = []
dirs_1 = []
for i in range(0, len(data_path)):

    dirs = os.listdir(data_path[i])
    dir_0 = []
    dir_1 = []
    for s in dirs:
        if s[-4:] != '.mp4':
            continue
        dir_0.append(s)                # video_name = s[:-4]

    dirs_t = [os.path.join(data_path[i], t) for t in dir_0]
    dirs_0 = dirs_0 + dirs_t


length_0 = len(dirs_0)
transform = transform
frames = selected_frames
length_frames = len(frames)
batch_size = 1


dirs = dirs_0 + dirs_1
length = len(dirs_0)
total = 0
corre = 0
rang = 0.35  # classification range

filenames=[]
labels = []

for it in range(0, length):
    X, is_null = get_data(dirs_0, it, net, transform, device, frames, length_frames, batch_size, is_multi)
    # print(X.shape)
    if is_null:
        path, filename = os.path.split(dirs_0[it])
        filenames.append(filename)
        labels.append(0.999)
        continue
    losses, pred = validation([cnn_encoder, rnn_decoder], device, X)
    # print(it, F.softmax(losses), pred)
    probs = F.softmax(losses)
    probs = probs.cpu().numpy()
    maxs = 0
    mins = 1
    print(dirs_0[it])
    path, filename = os.path.split(dirs_0[it])
    filenames.append(filename)
    print('Have '+str(len(probs))+' faces in the video')
    
    for i in range(0, len(probs)):
        temp = probs[i][0]
        if temp > rang:
            print('Fake probability of '+str(i+1)+' face is:', temp)
            if temp > maxs:
                maxs = temp
        else:
            print('Fake probability of '+str(i+1)+' face is:', temp)
            if temp < mins:
                mins = temp
        
    if maxs > 0:
        print('*** The fake probability of the video is', maxs, ', therefore it is a fake video ***')
        labels.append(maxs)
    else:
        print('*** The fake probability of the video is', mins, ', therefore it is a real video ***')
        labels.append(mins)
dataframe=pd.DataFrame({'filename':filenames,'label':labels})
new_df=dataframe.sort_values(by="filename", ascending=True)
new_df.to_csv('submission.csv',index=False)


# In[ ]:




