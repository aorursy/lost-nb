#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.vision import *




bs = 64




train=pd.read_csv('../input/labels.csv')
train.head()




train.id = train.id+'.jpg'




train.head()




tfms = get_transforms(max_rotate = 20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4, p_affine = 1., p_lighting=1.)




src = (ImageList.from_df(train, '../input/', folder='train')
      .split_by_rand_pct()
      .label_from_df(cols = 'breed'))




def get_data(size, bs, padding_mode = 'reflection'):
    return(src.transform(tfms, size = size, padding_mode=padding_mode)
          .databunch(bs=bs, num_workers = 0).normalize(imagenet_stats))




data = get_data(224, bs, 'zeros')




def _plot(i,j,ax):
    x,y = data.train_ds[4]
    x.show(ax, y=y)
    
plot_multi(_plot, 3, 3, figsize=(8,8))




gc.collect()
learn = cnn_learner(data, models.resnet34, 
                    metrics = error_rate, bn_final = True, 
                    model_dir = "/tmp/model")




learn.fit_one_cycle(3, slice(1e-2), pct_start=0.8)




learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-3), pct_start=0.8)




data = get_data(352,bs)
learn.data = data




learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))




learn.save('352')




data = get_data(352, 16)




learn = cnn_learner(data, models.resnet34, metrics=error_rate, bn_final=True, model_dir = '/tmp/model').load('352')




#Right Sobel kernel
k = tensor ([
    [-1., 0, 1],
    [-2., 0, 2],
    [-1., 0 , 1],
]).expand(1,3,3,3)/6




#Sharpen Kernel
k = tensor ([
    [0., -1, 0],
    [-1, 5, -1],
    [0, -1 , 0],
]).expand(1,3,3,3)




k




k.shape




idx = 2
t = data.valid_ds[idx][0].data; t.shape #Pull out a single image sample, 0th is image and 1st is label




t[None].shape #t[None] is a trick to get a mini-batch of a tensor of size 1, this also works in numpy




edge = F.conv2d(t[None], k)




show_image(edge[0], figsize = (5,5))




x,y = data.valid_ds[idx]
x.show()
data.valid_ds.y[idx]




#Number of categories
data.c




#Details of the resnet model 
learn.model
#theres a lot going on in the first Conv layer, but there are 64 chanels and a stride of 2 for the first layer
#When you stride by 2 you can double the number of chanels (this preserves the complexity of model/memory)




print(learn.summary())









m = learn.model.eval();




xb,_ = data.one_item(x) #takes all the settings from our previously created data object
xb_im = Image(data.denorm(xb)[0])
xb = xb.cuda()




from fastai.callbacks.hooks import *




get_ipython().run_line_magic('pinfo2', 'hook_output')




def hooked_backward(cat=y):
    with hook_output(m[0]) as hook_a: #Get activations from the convolution layers
        with hook_output(m[0], grad = True) as hook_g: #Get gradient from convolution layers
            preds = m(xb) #DO foreward pass through model
            preds[0,int(cat)].backward()
    return hook_a, hook_g




hook_a, hook_g = hooked_backward()




hook_a.stored




acts = hook_a.stored[0].cpu()
acts.shape #Now we see our 512 chanels over the 11x11 sections of the image




avg_acts = acts.mean(0)
avg_acts.shape




def show_heatmap(hm):
    _,ax = plt.subplots()
    xb_im.show(ax) #fastai function to show the image
    ax.imshow(hm, alpha = 0.6, extent = (0,352,352, 0), #extent expands the 11x11 image to 352,352
             interpolation = 'bilinear', cmap = 'magma')




show_heatmap(avg_acts)

