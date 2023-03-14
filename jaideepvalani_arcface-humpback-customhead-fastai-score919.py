#!/usr/bin/env python
# coding: utf-8



get_ipython().run_cell_magic('time', '', '! nvidia-smi\n#! rm -rf resnet_324.pth\n#!echo c.ExecutePreprocessor.timeout')




from fastai import *
from fastai.vision import *
from fastai.callbacks import *

import warnings
warnings.filterwarnings("ignore")




get_ipython().system(' ls -l ../input/')




#path = Path('./data/')
path_t=Path('../input/humpback-whale-identification/')
path_b=Path('../input/')
path1='.'
df = pd.read_csv(path_t/'train.csv'); 

#!pip install fastai=='1.0.44'

import fastai
fastai.__version__




#df = pd.read_csv(LABELS).set_index('Image')
exclude_list=['0b1e39ff.jpg',
'0c11fa0c.jpg',
'1b089ea6.jpg',
'2a2ecd4b.jpg',
'2c824757.jpg',
'3e550c8a.jpg',
'56893b19.jpg',
'613539b4.jpg',
'6530809b.jpg',
'6b753246.jpg',
'6b9f5632.jpg',
'75c94986.jpg',
'7f048f21.jpg',
'7f7702dc.jpg',
'806cf583.jpg',
'95226283.jpg',
'a3e9070d.jpg',
'ade8176b.jpg',
'b1cfda8a.jpg',
'b24c8170.jpg',
'b7ea8be4.jpg',
'b9315c19.jpg',
'b985ae1e.jpg',
'baf56258.jpg',
'c4ad67d8.jpg',
'c5da34e7.jpg',
'c5e3df74.jpg',
'ced4a25c.jpg',
'd14f0126.jpg',
'e0b00a14.jpg',
'e6ce415f.jpg',
'e9bd2e9c.jpg',
'f4063698.jpg',
'f9ba7040.jpg']
new_whale_df = df[df.Id == "new_whale"] # only new_whale dataset
train_df = df[~(df.Id == "new_whale")] # no new_whale dataset, used for training
unique_labels = np.unique(train_df.Id.values)
trn_imgs=train_df.copy().reset_index(drop=True)
cnter = Counter(trn_imgs.Id.values)
trn_imgs['cnt']=trn_imgs['Id'].apply(lambda x: cnter[x])
#trn_imgs['target'] = 1
trn_imgs['target'] = 0 # 0 for same images
trn_imgs1 = trn_imgs.copy()
#trn_imgs1['target'] = 0
trn_imgs1['target'] = 1 # 1 for dissimilar images
#trn_imgs = trn_imgs.append(trn_imgs1)
target_col = 3
trn_imgs.head(1)
trn_imgs=trn_imgs[~trn_imgs.Image.isin(exclude_list)]









def read_img(fname,box_df,img,sz=224):
    
             
    x0,y0,x1,y1 = tuple(box_df.loc[fname,['x0','y0','x1','y1']].tolist())
    #print(img.shape)
    l1,l0  = img.shape[1],img.shape[2]
    b0,b1 = x1-x0, y1-y0
        #padding
    x0n,x1n = max(int(x0 - b0*0.05),0), min(int(x1 + b0*0.05),l0-1)
    y0n,y1n = max(int(y0 - b1*0.05),0), min(int(y1 + b1*0.05),l1-1)
    img=to_np(img)
    
    #print(img.shape,x0,y0,x1,y1)
    if not (x0 >= x1 or y0 >= y1):
        None
        
        #img = img[:,y0n:y1n, x0n:x1n]
        #print(img.shape,'img')
        #if self.tfms_g != None: img = self.tfms_g.augment_image(img)
    img = img[:,y0n:y1n, x0n:x1n]
    #print(img.T.shape)
    #img = cv2.resize(img.T, (sz,sz))
    return Image(pil2tensor(img.astype(np.float)/255, np.float32).float())




def crop_loose_bbox(img,area, val=0.2):
    #img=np.asarray(img)
    #print(img.shape)
    l1, l0,_ = img.shape
    #print(img.shape)
    b0 = area[2] - area[0]
    b1 = area[3] - area[1]
    x0n,x1n = max(int(area[0] - b0*0.05),0), min(int(area[2] + b0*0.05),l0-1)
    y0n,y1n = max(int(area[1] - b1*0.05),0), min(int(area[3] + b1*0.05),l1-1)
   
    #print(img[y0n:y1n,x0n:x1n,:].shape )
    #img = cv2.resize(img[y0n:y1n,x0n:x1n,:], (224,224))
    """
    area2 = (max(0, int(area[0] - 0.5*val*w)),
             max(0, int(area[1] - 0.5*val*h)),
             min(img_w, int(area[2] + 0.5*val*w)),
             min(img_h, int(area[3] + 0.5*val*h)))
    """
    return img[y0n:y1n,x0n:x1n,:] #img.crop(area2)




"""
  def __call__(self, fname):
      fname = os.path.basename(fname)
      #x0,y0,x1,y1 = tuple(self.boxes.loc[fname,['x0','y0','x1','y1']].tolist())
      img = open_image(os.path.join(self.path,fname))
      l1,l0,_ = img.shape
      b0,b1 = x1-x0, y1-y0
      #padding
      x0n,x1n = max(int(x0 - b0*0.05),0), min(int(x1 + b0*0.05),l0-1)
      y0n,y1n = max(int(y0 - b1*0.05),0), min(int(y1 + b1*0.05),l1-1)
       
      if self.tfms_g != None: img = self.tfms_g.augment_image(img)
      img = cv2.resize(img[y0n:y1n,x0n:x1n,:], (sz,sz))
      if self.tfms_px != None: img = self.tfms_px.augment_image(img)
      return img.astype(np.float)/255
  """




#bbox_df = pd.read_csv(path_b/'cropped-img'/'bounding_boxes.csv').set_index('Image')




def open_4_channel2(fname):
    fname = str(fname)
    bbox_df = pd.read_csv(path_b/'cropped-img'/'bounding_boxes.csv').set_index('Image')
    #print(fname)
    # strip extension before adding color
    x0,y0,x1,y1=bbox_df.loc[fname[fname.rfind('/')+1:]]
    area=(x0,y0,x1,y1)                        
    #print(fname)
    img     = cv2.imread(fname)
    #PIL.Image.open(fname)
    #print(img.size)
                            
    img=crop_loose_bbox(img,area)
    
    #img=np.asarray(img)
    #print(img.shape)
    #print(img.shape)
   
    #import time
    #a=time.time()
   
    
    return Image(pil2tensor(img/255, np.float32).float())




trn_imgs=trn_imgs.append(trn_imgs.loc[trn_imgs.cnt==2],ignore_index=True) 
trn_imgs=trn_imgs.append(trn_imgs.loc[trn_imgs.cnt==1],ignore_index=True) 





val_idx=[]
import random
for i in trn_imgs[trn_imgs.cnt>5].Id.unique():
  tmp=list(trn_imgs.loc[trn_imgs.Id==i].index.values)
  #print(tmp)
  val_idx=val_idx+(random.sample(tmp,1))
len(val_idx)
#since images less than 5 are less in number we dont select much from them 
for i in trn_imgs[(trn_imgs.cnt<5) &( trn_imgs.cnt>2)].Id.unique():
  
  tmp=list(trn_imgs.loc[trn_imgs.Id==i].index.values)
  #print(type(tmp))
  
  if len(val_idx) < 1300 :
        
        val_idx=val_idx+(random.sample(tmp,1))
len(val_idx)




#val_idx[0:5]
#train_idx=
#trn_imgs[trn_imgs.Id=='w_f48451c']




#bbox_df = pd.read_csv(path/'bounding_boxes.csv').set_index('Image')
#x0,y0,x1,y1=bbox_df.loc['72c3ce75c.jpg']
#crop_loo

#open_4_channel(path/'train'/'0001f9222.jpg')
 #crop_loose_bbox(img,area, val=0.2)
len(trn_imgs.Id.unique())

#trn_imgs[trn_imgs.cnt<3].Id.unique().shape




val_idx=list(trn_imgs.iloc[val_idx].index.values)
trn_idx=set(list(trn_imgs.index.values))-set(val_idx) # generating only trn idx to run
df_i=trn_imgs.iloc[list(trn_idx)].reset_index(drop=True) # this will be used latter on to run CV loop
#fn2label = {row[1].Image: row[1].Id for row in df.iterrows()}
#path2fn = lambda path: re.search('\w*\.jpg$', path).group(0)




#df_i.index.size 0000e88ab.jpg w_f48451c
#df_i.Id.nunique()
#trn_imgs.to_csv() 
#trn_imgs[trn_imgs.Image=='0001f9222.jpg']#w_c3d896a	
#df_i.head(2)




src1= (ImageList.from_df(trn_imgs[['Image','Id']],path_t, folder='train') #ImageList
       .split_by_idx(val_idx)
       .label_from_df( cols=1))




#trn_imgs.head(2)




"""
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
mlb = OneHotEncoder([i for i in range(5004)],sparse=False)
#MultiLabelBinarizer([i for i in range(5004)],sparse_output=False)
#y=mlb.fit_transform(np.array(list(1) ).reshape(-1,1))
#trn_imgs['hot']=trn_imgs.Image.apply(lambda i : y[trn_imgs[trn_imgs.Image==i].index.values])
#y[0]
#trn_imgs.head(1)

#np.array([1,2])
a=[one_hot(i,5004 )for i in range(5)]

np.array(a).reshape(5,-1).shape
"""




test_ids = list(sorted({fname for fname in os.listdir(path_t/'test')}))


#protein_stats = ([0.16258, 0.13877, 0.10067, 0.16358], [0.21966, 0.18559, 0.25573,0.22066])
test_fnames = [path_t/'test'/test_id for test_id in test_ids]

test_fnames[:3]




#np.where(list(trn_imgs.hot.values)[1]==[1])[1]




import cv2
src1.train.x.create_func = open_4_channel2
src1.train.x.open = open_4_channel2

src1.valid.x.create_func = open_4_channel2
src1.valid.x.open = open_4_channel2
src1.add_test(test_fnames);
src1.test.x.create_func = open_4_channel2
src1.test.x.open = open_4_channel2
# combine dataset/transform/dataloader into one dataobject called databunch in fastai
trn_tfms,_ = get_transforms(do_flip=False, flip_vert=True, max_rotate=5., max_zoom=1.08,
                      max_lighting=0.15, max_warp=0. )

data1 = (src1.transform((trn_tfms,trn_tfms), size=224,resize_method=ResizeMethod.SQUISH)
        .databunch(bs=64,num_workers=0).normalize(imagenet_stats))

data2 = (src1.transform((trn_tfms,trn_tfms), size=484,resize_method=ResizeMethod.SQUISH)
        .databunch(bs=36,num_workers=0).normalize(imagenet_stats))





#a=[one_hot(i.unsqueeze(-1),5004 ) for i in tensor(data1.train_ds.y.items[0:5])]
#listify(x)
#np.where(a[0]==[1])
#tensor(data1.train_ds.y.items[0:5])
#type(a)
#torch.from_numpy(np.array(a)).size()

#data1.show_batch(2)
#import pylot as plt
#i=PIL.Image('data/train/3ece2140f.jpg')
#print(i.shape)
#plt.imshow(i)




#data1.c
#data1.show_batch(2)
#!cp *.csv ./data/




from fastai.metrics import accuracy




from torchvision import models as m
def dense(pre):
    
    #model=nn.Sequential(body, head)
    model = m.densenet121(pretrained=pre)
  
    model.classifier = (nn.Linear(1024, 5004))

   
    return model
def _densenet_split(m): return   (m[0][0][6],m[1]) 




#dense(True)




def acc (input:Tensor, targs:Tensor)->Rank0Tensor:
  
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n,-1)
    targs = targs.view(n,-1)
    return (input==targs).float().mean()




i=torch.rand(3,2)
#j=torch.ones(48,1)
print(i)
#acc(i,j)
#F.softmax(i,1) 
#torch.empty(5004, 1024)
#nn.init.kaiming_normal_(torch.FloatTensor(5004, 1024))
#torch.randint(4, (3,), dtype=torch.int64)




class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features=5004):
        
      
        super(ArcMarginProduct, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()
        #nn.init.kaiming_uniform_(self.weight)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1)) # eq to input . This is more or less like Kaiming normal .
        #we use this to ensure values remain between 0 and 1 . Since std deviation reduces to almost half every layer
        # we can try some trick  multiplying it by 2 
        self.weight.data.uniform_(-stdv, stdv)
    
    def forward(self, features):
        #x=self.head(features)
        #print(self.weight.shape)
        #self.fc1.weight=nn.Parameter(F.normalize(self.fc1.weight)).cuda()
        cosine = F.linear(F.normalize(features), F.normalize(self.weight.cuda()))
        #cosine = cosine.clamp(-1, 1)
        #self.fc1(F.normalize(x))
        #F.linear(F.normalize(x), F.normalize(self.weight.cuda()))
        return cosine   

class Customhead(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features=5004):
        super(Customhead, self).__init__()
        #self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        #self.register_parameter('normweights',self.weight)
        # nn.init.xavier_uniform_(self.weight)
        #body = create_body(m.densenet121, True, -1)
        body = create_body(m.resnet50, True, -2)
#body = create_head(ArcMarginProduct, pretrained, 0)
        nf = num_features_model(nn.Sequential(*body.children())) * 2
        #head = 
        self.head=create_head(nf, 1024,[2048],  ps=0.5, bn_final=False) # 1024 no of classes
        self.arc_margin=ArcMarginProduct(in_features,out_features)
        #self.fc1=nn.Linear(1024,5004,bias=False)
        #self.custom=nn.Sequential(self.head,self.fc1)
        #self.reset_parameters()

   # def reset_parameters(self):
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)
       

    def forward(self, features):
        x=self.head(features)
        #w=self.fc1.weight
        #self.fc1.weight=nn.Parameter(F.normalize(self.fc1.weight)).cuda()
        cosine = self.arc_margin(x)
        #F.linear(F.normalize(x), F.normalize(w))
        #self.arc_margin(x)
        #F.linear(F.normalize(x), F.normalize(self.weight.cuda()))
        cosine = cosine.clamp(-1, 1)
        #self.fc1(F.normalize(x))
        #F.linear(F.normalize(x), F.normalize(self.weight.cuda()))
        return cosine




#for i in Customhead(1024,5004).parameters():
   #print( i.size())
#Customhead(1024,5004)

class CustomheadRes(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features=5004):
        super(CustomheadRes, self).__init__()
        #self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        #self.register_parameter('normweights',self.weight)
        # nn.init.xavier_uniform_(self.weight)
        body = create_body(m.densenet121, True, -1)
        #body = create_body(m.resnet50, True, -2)
#body = create_head(ArcMarginProduct, pretrained, 0)
        nf = num_features_model(nn.Sequential(*body.children())) * 2
        #head = 
        self.head=create_head(nf, 1024,  ps=0.5, bn_final=False) # 1024 no of classes
        self.arc_margin=ArcMarginProduct(in_features,out_features)
        #self.fc1=nn.Linear(1024,5004,bias=False)
        #self.custom=nn.Sequential(self.head,self.fc1)
        #self.reset_parameters()

   # def reset_parameters(self):
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)
       

    def forward(self, features):
        x=self.head(features)
        #w=self.fc1.weight
        #self.fc1.weight=nn.Parameter(F.normalize(self.fc1.weight)).cuda()
        cosine = self.arc_margin(x)
        #F.linear(F.normalize(x), F.normalize(w))
        #self.arc_margin(x)
        #F.linear(F.normalize(x), F.normalize(self.weight.cuda()))
        cosine = cosine.clamp(-1, 1)
        #self.fc1(F.normalize(x))
        #F.linear(F.normalize(x), F.normalize(self.weight.cuda()))
        return cosine




#data1.show_batch(2)
#src1.xtra.Id
class ArcFaceLoss(nn.modules.Module):
    def __init__(self,s=30.0,m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.s = s
        self.easy_margin = False
        self.cos_m = math.cos(m) 
        self.sin_m = math.sin(m) 
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, labels, epoch=0,reduction=None):
        cosine = inputs
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss1 = self.classify_loss(output, labels) # this is as per paper what is missing here is centralized features
        loss2 = self.classify_loss(cosine, labels)
        gamma=1
        loss=(loss1+gamma*loss2)/(1+gamma)
        return loss.mean()





def resnet501(pre):
    
    model = m.resnet50(pretrained=pre)
    #w=model.features[0].weight
    #model.features[0]=nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #model.fc = (nn.Linear(1024, 5004))

    #print(w.shape)
   # model.features[0].weight=torch.nn.Parameter(torch.cat((w, w[:,:1,:,:]),dim=1))
    #print(model.features[0].weight.shape)
    return model
def _resnet_split(m): return (m[0][6],m[1])




ar=ArcFaceLoss().cuda() # this may not be needed just try it out





#m1=nn.Sequential(body, head)
#m1[:-1]

#custom_head









f1_score = partial(fbeta, thresh=0.4, beta=1)
acc_02 = partial(accuracy_thresh, thresh=0.2)

custom_head=Customhead(1024,5004)
custom_headres=CustomheadRes(1024,5004)
ar=ArcFaceLoss().cuda()
from fastai.torch_core import *
from fastai.callbacks import *
from fastai.basic_train import *
from torch.autograd import Variable

#callbacks=[partial(GradientClipping, clip=1),partial(SaveModelCallback,monitor='trn_loss',mode='min')
#           ,ReduceLROnPlateauCallback(learn, min_delta=1e-5, patience=3)]
import gc
gc.collect()
learn1 = create_cnn(
    data2,
    #dense,
    resnet501,
    #dense,
    #cut=-1,
    cut=-2,
    split_on=_resnet_split,
    #_densenet_split,
    lin_ftrs=[1024],
    custom_head=custom_head,
   
    #lambda m: (m[0][11], m[1]),
    loss_func=ar,
    #torch.nn.MultiLabelSoftMarginLoss(),
    #F.binary_cross_entropy_with_logits,
    #FocalLoss(),
    #F.binary_cross_entropy_with_logits,
    path=path1,    
    metrics=[accuracy], callback_fns= partial(GradientClipping, clip=1))
 
#learn1 = Learner(data1, dense(), loss_func=torch.nn.MultiLabelSoftMarginLoss(),path=path,
               #metrics=[acc_02,f1_scorestd], callback_fns= partial(GradientClipping, clip=1))
learn1.callback_fns.append(partial(SaveModelCallback,monitor='val_loss',mode='min'))
learn1.callback_fns.append(partial(ReduceLROnPlateauCallback, min_delta=1e-5, patience=3))

# dense net used for building CNN.
learn2 = create_cnn(
    data2,
    dense,
    #resnet501,
    #dense,
    cut=-1,
    #cut=-2,
    #split_on=_resnet_split,
    split_on=_densenet_split,
    lin_ftrs=[512],
    custom_head=custom_headres,
    #lambda m: (m[0][11], m[1]),
    loss_func=ar,
    #torch.nn.MultiLabelSoftMarginLoss(),
    #F.binary_cross_entropy_with_logits,
    #FocalLoss(),
    #F.binary_cross_entropy_with_logits,
    path=path1,    
    metrics=[accuracy], callback_fns= partial(GradientClipping, clip=1))
 
#learn1 = Learner(data1, dense(), loss_func=torch.nn.MultiLabelSoftMarginLoss(),path=path,
               #metrics=[acc_02,f1_scorestd], callback_fns= partial(GradientClipping, clip=1))
#learn2.callback_fns.append(partial(SaveModelCallback,monitor='val_loss',mode='min'))
learn2.callback_fns.append(partial(ReduceLROnPlateauCallback, min_delta=1e-5, patience=3))
#learn2=learn2.to_fp16()




#learn1.model[1]
#data1.c
#!cp dens* ./data/models/
#learn2.model#[6].weight.shape
#learn2.save('save')
#learn2.model
#torch.FloatTensor(2,3)
get_ipython().system(' mkdir /kaggle/working/models')
#! ls -l /kaggle/input/
#!cp /kaggle/input/dense324/*.pth /kaggle/working/models/
#!cp /kaggle/input/arcface-humpback-customhead-fastai-score919/models/resnet_ar_c384.pth  /kaggle/working/models/
#!cp /kaggle/input/arcface-humpback-customhead-fastai/models/dense_ar_c1284.pth  /kaggle/working/models/
#!cp /kaggle/input/arcface-humpback-customhead-fastai-score919/models/resnet_ar_c484.pth  /kaggle/working/models/
get_ipython().system('cp /kaggle/input/arcface-humpback-customhead-fastai-score919/models/dense_ar_c424_1.pth  /kaggle/working/models/')

#!cp /kaggle/input/arcface-humpback-customhead-fastai-score919/models/resnet_ar_c424_2.pth  /kaggle/working/models/
get_ipython().system('cp /kaggle/input/arcface-humpback-customhead-fastai-score919/models/resnet_ar_c424_1.pth  /kaggle/working/models/')
#!mv /kaggle/working/models/dense_ar_c324 /kaggle/working/models/dense_ar_c324.pth
get_ipython().system(' ls -l   /kaggle/working/models')




import gc
gc.collect()
#learn2.lr_find()
#learn2.recorder.plot()
#!rm -rf ./data/models/
#len(data1.train_dl)
#learn1.loss_func
#data1.show_batch(2)
#learn1.save('dense_224')
#for i in learn2.model[1].parameters():
    #print(i.size())




#learn2.recorder.plot()
#push
#learn2.model[1]




#x,y=next(iter(learn1.data.train_dl))
#! rm -rf ./models
#learn1.recorder.plot()
#for i in trainable_params(learn2.model[1]):
    #print(i.size())
#push




#learn2.unfreeze()
#learn2.load('dense_arc1')
#learn2.fit_one_cycle(2,3e-2)




#!ls -l
#!cp  ./data/models/dense_arc1.pth ./

#learn1.save('dense_arc1')
for i in learn1.model[1].parameters():
  print(i.shape)
#learn2.model[1]
learn2.model[1]




"""
lr=1e-2 # ran stratified 224,284*2,now ffull
learn2.unfreeze()
#learn2.load('save')
#learn2.load('dense_ar_c324') #0.026030	0.656774	0.892308
learn2.fit_one_cycle(14,slice(2e-4,lr/2))
learn2.save('dense_ar_c324')
"""




#learn2.fit_one_cycle(8,slice(2e-4,lr/2)) # run for 30 epochs 
lr=2e-2 # ran stratified 224,284*2,now ffull
#learn2.unfreeze()
#learn2.load('save')
#learn2.load('dense_ar_c56') #0.026030	0.656774	0.892308
#learn1.fit_one_cycle(2,slice(2e-4,lr/2))
#learn1.save('resnet_ar_c224')




"""
lr=2e-2 # ran stratified 224,284*2,now ffulls

learn1.unfreeze()
#learn2.load('save')
learn1.load('resnet_ar_c424') #0.026030	0.656774	0.892308
learn1.fit_one_cycle(7,slice(2e-4,lr/2))
learn1.save('resnet_ar_c424_1')
"""
lr=3e-2
learn1.unfreeze()
#learn2.load('save')
learn1.load('resnet_ar_c424_1') #0.026030	0.656774	0.892308
learn1.fit_one_cycle(9,slice(2e-5,lr/2))
learn1.save('resnet_ar_c424_2')

 




""" 
lr=1e-2 # ran stratified 224,284*2,now ffulls
learn2.unfreeze()
#learn2.load('save')
learn2.load('dense_ar_c384') #0.026030	0.656774	0.892308
learn2.fit_one_cycle(11,slice(2e-5,lr/2))
learn2.save('dense_ar_c424_1')
 
"""





print('Train_loss',learn1.recorder.losses[-1])
print('Val loss',learn1.recorder.val_losses)

print('Accuracy',learn1.recorder.metrics)
 




print('Train_loss',learn2.recorder.losses[-1])
print('Val loss',learn2.recorder.val_losses)

print('Accuracy',learn2.recorder.metrics)




learn1.recorder.plot_losses()




#learn2.recorder.plot_losses()
#w=tensor([0,1,0])
#torch.nonzero(w == 0)
#y1t[487]

#len(set(data2.train_ds.y.items))




""" 
learn2.model.eval()
#!cp unfreeze_284_1.pth ./data/models/
learn2.load('dense_ar_c424_1')
preds2,y = learn2.TTA(ds_type=DatasetType.Test,beta=0.30,with_loss=False,scale=1.08)
"""
learn1.load('resnet_ar_c424_2')
preds1,y = learn1.get_preds(ds_type=DatasetType.Test)





 

learn1.model.eval()
learn1.load('resnet_ar_c424_2')
preds1,y1 = learn1.TTA(ds_type=DatasetType.Test,beta=0.30,with_loss=False,scale=1.1)

#preds1,y = learn1.get_preds(ds_type=DatasetType.Test)
 




""" 
learn1.model.eval()
learn1.load('resnet_ar_c424_1')
preds1t,y1t = learn1.TTA(ds_type=DatasetType.Train,beta=0.30,with_loss=False,scale=1.1)
""" 




""" 
# intra class mean
l=[]
for c in list(set(data2.train_ds.y.items)):
     l.append(torch.mean(preds1t[torch.nonzero(y1t == c)],dim=0))

trn_centre=torch.cat(l)
trn_centre.size()
 """




preds1.size()




"""
#learn1.load('resnet_ar_c424_1')
learn1.model.eval()
#%%time
sims = []
with torch.no_grad():
    
    for feat in preds1:
        dists = F.cosine_similarity(trn_centre, feat.unsqueeze(0).repeat(5004, 1))
        predicted_similarity = dists.cuda()#learn.model.head(dists.cuda())
        sims.append(predicted_similarity.squeeze().detach().cpu())
 """




#sims[20][sims[20].argsort(descending=True)[:5]]




#! cp /kaggle/working/models/resnet_ar_c356.pth /kaggle/working/
#!ls -l ./models/
#! cd models
#torch.max(preds1,preds2).shape

#torch.mean(preds1,preds2)
#FileLink('resnet_ar_c356.pth')
"""
learn1.model.eval()
learn1.load('resnet_ar_c356_1')
predsv,y_v = learn1.TTA(ds_type=DatasetType.Valid,beta=0.30,with_loss=False,scale=1.08)
"""




def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))
#preds_t = np.stack(preds1[0], axis=-1)
#print(preds_t.shape)
preds_tv = sigmoid_np(predsv )




#np.linspace(0.5, 1, 10)
i=preds_tv[99,:].argsort(descending=True)
probs = preds_tv[0,i]
probs[:5] 
np.stack(top_5s_v).shape
labels_list[2]
#y_v[0]




"""
classes = df.Id.unique()
new_whale_idx = np.where(classes == 'new_whale')[0][0]
#top_5s = []
for thresh in np.linspace(0.5, 1, 20):
    top_5s_v = []
    for sim in preds_tv:
        idxs = sim.argsort(descending=True)
        probs = sim[idxs]
        top_5 = []
        for i, p in zip(idxs, probs):
            #if 'new_whale' not in top_5 and p <thresh and len(top_5) < 5: 
              #top_5.append('new_whale')
            if len(top_5) == 5: break
            if i == new_whale_idx: continue
            predicted_class =idzxs #labels_list[i]
            if predicted_class not in top_5: top_5.append(predicted_class)
        top_5s_v.append(top_5)
    print(thresh, mapk(data2.valid_ds.y.items.reshape(-1,1), np.stack(top_5s_v), 5))
"""





def sigmoid_np(x):
   return 1.0/(1.0 + np.exp(-x))
#preds_t = np.stack(preds1[0], axis=-1)
#print(preds_t.shape)
preds_t = sigmoid_np(preds1)
#preds_t1 = sigmoid_np(preds1 )
#preds_t2 = sigmoid_np(preds2 )
#preds_t = sigmoid_np((preds1+preds2)/2)
#sigmoid_np(torch.max(preds1,preds2)) # ensembling part
#preds_t = torch.max(preds_t1,preds_t2)


#preds_t[90,i]




#((preds1+preds2)/2).shape
get_ipython().system(' nvidia -smi')




#preds1[:,0:10]
i=preds_t[99,:].argsort(descending=True)
probs = preds_t[99,i]
probs[:5] #0.7307, 0.5002, 0.5001, 0.5000, 0.5000])




unique_labels = np.unique(trn_imgs.Id.values)

labels_dict = dict()
labels_list = []
for i in range(len(unique_labels)):
    labels_dict[unique_labels[i]] = i
    labels_list.append(unique_labels[i])
labels_list[0]




#learn1.data
#data1.xtra.Id.values





classes = df.Id.unique()
new_whale_idx = np.where(classes == 'new_whale')[0][0]
top_5s = []
for sim in preds_t:
   idxs = sim.argsort(descending=True)
   probs = sim[idxs]
   top_5 = []
   for i, p in zip(idxs, probs):
       if 'new_whale' not in top_5 and p <0.585 and len(top_5) < 5: #615#575 res,.58 63 for dense .39
         top_5.append('new_whale')
       if len(top_5) == 5: break
       if i == new_whale_idx: continue
       predicted_class = labels_list[i]
       if predicted_class not in top_5: top_5.append(predicted_class)
   top_5s.append(top_5)






#top_5_classes
#top_5s




from IPython.display import FileLink

top_5_classes = []
for top_5 in top_5s:
    top_5_classes.append(' '.join([t for t in top_5]))
sub = pd.DataFrame({'Image': [path.name for path in data1.test_ds.x.items]})
sub['Id'] = top_5_classes

#sub.head(10)
sub.to_csv('pred_res484.csv',index=False)
FileLink('pred_res484.csv')
 
#!nvdia - smi




#sub.to_csv('resnetpred6.csv',index=False)
#sub.head(10)
#!kaggle competitions submit -c humpback-whale-identification -f 'resnetpred6.csv' -m "bestresnet324_525"




#df_i.head(2)
X = list(df_i.index.values)
set(df_i.iloc[X].Id)-set(trn_imgs.iloc[X].Id)

#(df_i.iloc[X].Id,df_i.Id.values)
len(trn_imgs)
#len(learn2.get_layer_group)









""" 
lr=1e-2
#learn1.fit_one_cycle(2,lr)
#learn1.summary
#learn1.save('freeez1')

# Go through folds
#for trn_idx, val_idx in folds.split(target, target):

#!pip install iterative-stratification
#from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from fastai.torch_core import *
from fastai.callbacks import *
from fastai.basic_train import *
from torchvision import models
import gc
gc.collect()
from sklearn.model_selection import KFold, StratifiedKFold
#df1=df.copy()
#train_labels = df1.apply(fill_targets, axis=1) # convert comma separated targets into list
#!cp *.pth ./data/models/
X = list(df_i.index.values)
y=list(df_i.Id.values)
#mlb = MultiLabelBinarizer( )
#y=mlb.fit_transform(df_i.Id.values)
#df['labels_v']=df.labels.apply(lambda x: mlb.fit_transform( x  )

#y=df.Target.values
#print(X.shape)
#np.array([[1,2], [3,4], [1,2], [3,4], [1,2], [3,4], [1,2], [3,4]])
#y = np.array([[0,0], [0,0], [0,1], [0,1], [1,1], [1,1], [1,0], [1,0]])

mskf = StratifiedKFold(n_splits=9, random_state=2)

#MultilabelStratifiedKFold(n_splits=9, random_state=2)
#val_idx= df.loc[df.Id.isin(val_n)].index
i=0
#!cp *.pth ./data/models/
#stage-1-rn50-f
#learn.load('stage-1-rn50-f')
#protein_stats =([0.08069, 0.05258, 0.05487, 0.08282],[0.13704, 0.10145, 0.15313, 0.13814])
for train_index, test_index in mskf.split(X, y):
    val_i=test_index

#ImageItemList.from_csv( path, 'train.csv',folder='train', suffix='.png')

    src= (ImageList.from_df(df_i[['Image','Id']],path_t, folder='train') #ImageList
       .split_by_idx(val_i)
       #..split_by_valid_func(lambda path: path2fn(path) in val_fns)
        #.label_from_func(lambda path: fn2label[path2fn(path)]))
       .label_from_df( cols=1))
    
   
    src.train.x.create_func = open_4_channel2
    src.train.x.open = open_4_channel2

    src.valid.x.create_func = open_4_channel2
    src.valid.x.open = open_4_channel2
    
    src.add_test(test_fnames);
    src.test.x.create_func = open_4_channel2
    src.test.x.open = open_4_channel2
   
    #data.c=5004
    
    if i>=0:
        
      
        
        trn_tfms,_ = get_transforms(do_flip=False, flip_vert=True, max_rotate=5., max_zoom=1.08,
                              max_lighting=0.15, max_warp=0. )
        #protein_stats = ([0.16258, 0.13877, 0.10067 ], [0.21966, 0.18559, 0.25573 ])
        

        data = (src.transform((trn_tfms,trn_tfms), size=484,resize_method=ResizeMethod.SQUISH)
        .databunch(bs=32,num_workers=0).normalize(imagenet_stats)) #40
        #data.c=5004
        learn = create_cnn(
                      data,
                      resnet501,
                     split_on=_resnet_split,
                      cut=-2,
                     # dense,
                      
                      #cut=-1,
                     # split_on=_densenet_split,
                    
                      lin_ftrs=[1024],
                       
                      #lambda m: (m[0][11], m[1]),
                      loss_func=ar,
                      #custom_head=custom_headres,
                       custom_head=custom_head,
            #custom_head,
                      #torch.nn.MultiLabelSoftMarginLoss(),
                      #F.binary_cross_entropy_with_logits,
                      #FocalLoss(),
                      #F.binary_cross_entropy_with_logits,
                      path=path1,    
                      metrics=[accuracy], callback_fns= partial(GradientClipping, clip=1))
        #learn.callback_fns.append(partial(SaveModelCallback,monitor='val_loss',mode='min'))
        learn.callback_fns.append(partial(ReduceLROnPlateauCallback, min_delta=1e-5, patience=3))
        #learn.to_fp16()
        #learn.load('stage-1-rn50')
        #print('load')
        
        lr=1e-2
        #4e-4 # every 3-4 epocs reduce by 1 2e-3,1e-3.slice (lr/10,lr )
        #learn.load('stage-1-rn50-u7datablocks')
        #learn.load('stage-1-rn50-u11_512')
        if i==0:
            
            learn.load('resnet_ar_c424_1')
            print('x')
        else :

            learn.load('resnet_ar_c484')  
        print(i)  
        learn.unfreeze()
        learn.fit_one_cycle(1, slice(2e-5,lr/2))
        learn.save('resnet_ar_c484')
    i=i+1
#re
 """




#learn.recorder.losses
""" 
lr=2e-2 # ran stratified 224,284*2,now ffulls

learn2.unfreeze()
#learn2.load('save')
learn2.load('dense_ar_c424') #0.026030	0.656774	0.892308
learn2.fit_one_cycle(7,slice(2e-4,lr/2))
learn2.save('dense_ar_c424')
"""
""" 
lr=1e-2 # ran stratified 224,284*2,now ffulls

learn1.unfreeze()
#learn2.load('save')
learn1.load('resnet_ar_c484') #0.026030	0.656774	0.892308
learn1.fit_one_cycle(7,slice(2e-5,lr/2))
learn1.save('resnet_ar_c484_2')
"""




#print('Train_loss',learn2.recorder.losses[-1])
#print('Val loss',learn2.recorder.val_losses)

#print('Accuracy',learn2.recorder.metrics)

