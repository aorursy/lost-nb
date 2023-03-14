#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install torch torchvision feather-format kornia pyarrow --upgrade   > /dev/null')
get_ipython().system('pip install git+https://github.com/fastai/fastai2 > /dev/null')


# In[2]:


from fastai2.basics           import *
from fastai2.vision.all       import *
from fastai2.medical.imaging  import *
from fastai2.callback.tracker import *
from fastai2.callback.all     import *

np.set_printoptions(linewidth=120)
matplotlib.rcParams['image.cmap'] = 'bone'


# In[3]:


path = Path('../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/')
path_trn = path/'stage_2_train'
path_tst = path/'stage_2_test'

path_inp = Path('../input')
path_xtra = path_inp/'rsna-hemorrhage-jpg'
path_meta = path_xtra/'meta'/'meta'
path_jpg = path_xtra/'train_jpg'/'train_jpg'


# In[4]:


df_comb = pd.read_feather(path_meta/'comb.fth').set_index('SOPInstanceUID')
df_tst  = pd.read_feather(path_meta/'df_tst.fth').set_index('SOPInstanceUID')
df_samp = pd.read_feather(path_meta/'wgt_sample.fth').set_index('SOPInstanceUID')
bins = (path_meta/'bins.pkl').load()


# In[5]:


set_seed(42)
patients = df_comb.PatientID.unique()
pat_mask = np.random.random(len(patients))<0.8
pat_trn = patients[pat_mask]


# In[6]:


def split_data(df):
    idx = L.range(df)
    mask = df.PatientID.isin(pat_trn)
    return idx[mask],idx[~mask]

splits = split_data(df_samp)


# In[7]:


df_trn = df_samp.iloc[splits[0]]
p1 = L.range(df_samp)[df_samp.PatientID==df_trn.PatientID[0]]
assert len(p1) == len(set(p1) & set(splits[0]))


# In[8]:


def filename(o): return os.path.splitext(os.path.basename(o))[0]

fns = L(list(df_samp.fname)).map(filename)
fn = fns[0]
fn


# In[9]:


def fn2image(fn): return PILCTScan.create((path_jpg/fn).with_suffix('.jpg'))
fn2image(fn).show();


# In[10]:


htypes = ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']
def fn2label(fn): return df_comb.loc[fn][htypes].values.astype(np.float32)
fn2label(fn)


# In[11]:


bs,nw = 128,4


# In[12]:


tfms = [[fn2image], [fn2label,EncodedMultiCategorize(htypes)]]
dsrc = Datasets(fns, tfms, splits=splits)
nrm = Normalize(tensor([0.6]).cuda(),tensor([0.25]).cuda())
aug = aug_transforms(p_lighting=0.)
batch_tfms = [IntToFloatTensor(), nrm, *aug]


# In[13]:


def get_data(bs, sz):
    return dsrc.dataloaders(bs=bs, num_workers=nw,after_item=[ToTensor],after_batch=batch_tfms+[AffineCoordTfm(size=sz)])


# In[14]:


dbch = get_data(128, 96)
xb,yb = to_cpu(dbch.one_batch())
dbch.show_batch(max_n=4, figsize=(9,6))
xb.mean(),xb.std(),xb.shape,len(dbch.train_ds)


# In[15]:


def accuracy_any(inp, targ, thresh=0.5, sigmoid=True):
    inp,targ = flatten_check(inp[:,0],targ[:,0])
    if sigmoid: inp = inp.sigmoid()
    return ((inp>thresh)==targ.bool()).float().mean()


# In[16]:


def get_loss(scale=1.0):
    loss_weights = tensor(2.0, 1, 1, 1, 1, 1).cuda()*scale
    return BaseLoss(nn.BCEWithLogitsLoss, pos_weight=loss_weights, floatify=True, flatten=False, 
        is_2d=False, activation=torch.sigmoid)


# In[17]:


loss_func = get_loss(0.14*2)
opt_func = partial(Adam, wd=0.01, eps=1e-3)
metrics=[accuracy_multi,accuracy_any]


# In[18]:


def get_learner():
    dbch = get_data(128,128)
    learn = cnn_learner(dbch, xresnet50, loss_func=loss_func, opt_func=opt_func, metrics=metrics)
    return learn.to_fp16()


# In[19]:


learn = get_learner()


# In[20]:


lrf = learn.lr_find()


# In[21]:


def do_fit(bs,sz,epochs,lr, freeze=True):
    learn.dbunch = get_data(bs, sz)
    if freeze:
        if learn.opt is not None: learn.opt.clear_state()
        learn.freeze()
        learn.fit_one_cycle(1, slice(lr))
    learn.unfreeze()
    learn.fit_one_cycle(epochs, slice(lr))


# In[22]:


do_fit(128, 96, 4, 1e-2)


# In[23]:


do_fit(128, 160, 3, 1e-3)


# In[24]:


fns = L(list(df_comb.fname)).map(filename)
splits = split_data(df_comb)


# In[25]:


def fix_pxrepr(dcm):
    if dcm.PixelRepresentation != 0 or dcm.RescaleIntercept<-100: return
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000


# In[26]:


def dcm_tfm(fn): 
    fn = (path_trn/fn).with_suffix('.dcm')
    try:
        x = fn.dcmread()
        fix_pxrepr(x)
    except Exception as e:
        print(fn,e)
        raise SkipItemException
    if x.Rows != 512 or x.Columns != 512: x.zoom_to((512,512))
    px = x.scaled_px
    return TensorImage(px.to_3chan(dicom_windows.brain,dicom_windows.subdural, bins=bins))


# In[27]:


dcm = dcm_tfm(fns[0])
show_images(dcm)
dcm.shape


# In[28]:


tfms = [[dcm_tfm], [fn2label,EncodedMultiCategorize(htypes)]]
dsrc = Datasets(fns, tfms, splits=splits)
batch_tfms = [nrm, *aug]


# In[29]:


def get_data(bs, sz):
    return dsrc.dataloaders(bs=bs, num_workers=nw, after_batch=batch_tfms+[AffineCoordTfm(size=sz)])


# In[30]:


dbch = get_data(64,256)
x,y = to_cpu(dbch.one_batch())
dbch.show_batch(max_n=4)
x.shape


# In[31]:


learn.loss_func = get_loss(1.0)


# In[32]:


def fit_tune(bs, sz, epochs, lr):
    dbch = get_data(bs, sz)
    learn.dbunch = dbch
    learn.opt.clear_state()
    learn.unfreeze()
    learn.fit_one_cycle(epochs, slice(lr))


# In[33]:


fit_tune(64, 256, 2, 1e-3)


# In[34]:


test_fns = [(path_tst/f'{filename(o)}.dcm').absolute() for o in df_tst.fname.values]


# In[35]:


testFns = L(list(test_fns)).map(filename)
tst = test_set(dsrc, testFns)


# In[36]:


learn.dls.device


# In[37]:


next(learn.model.parameters()).is_cuda


# In[38]:


#preds,targs = learn.get_preds(dl=tst.tls)
preds_clipped = preds.clamp(.0001, .999)


# In[39]:


ids = []
labels = []

for idx,pred in zip(df_tst.index, preds_clipped):
    for i,label in enumerate(htypes):
        ids.append(f"{idx}_{label}")
        predicted_probability = '{0:1.10f}'.format(pred[i].item())
        labels.append(predicted_probability)


# In[40]:


df_csv = pd.DataFrame({'ID': ids, 'Label': labels})
df_csv.to_csv(f'submission.csv', index=False)
df_csv.head()


# In[41]:


from IPython.display import FileLink, FileLinks
FileLink('submission.csv')


# In[ ]:




