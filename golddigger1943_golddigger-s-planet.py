#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')




get_ipython().system('pip install fastai==0.7.0 --no-deps')
get_ipython().system('pip install torch==0.4.1 torchvision==0.2.1')

from fastai.conv_learner import *




print(os.listdir("../input/"))
PATH = '../input/planet-understanding-the-amazon-from-space/'




TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model"




ls {PATH}




df = pd.read_csv(PATH+'sample_submission_v2.csv')
df.head(5)




from fastai.plots import *




def get_1st(path): return glob(f'{path}/*.*')[0]




dc_path = '../input/newdogscats/dogscats/dogscats/train/'
list_paths = [get_1st(f"{dc_path}cats"), get_1st(f"{dc_path}dogs")]
plots_from_files(list_paths, titles=["cat", "dog"], maintitle="Single-label classification")




list_paths = [f"{PATH}train-jpg/train_0.jpg", f"{PATH}train-jpg/train_1.jpg"]
titles=["haze primary", "agriculture clear primary water"]
plots_from_files(list_paths, titles=titles, maintitle="Multi-label classification")




# planet.py

from fastai.imports import *
from fastai.transforms import *
from fastai.dataset import *
from sklearn.metrics import fbeta_score
import warnings

def f2(preds, targs, start=0.17, end=0.24, step=0.01):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return max([fbeta_score(targs, (preds>th), 2, average='samples')
                    for th in np.arange(start,end,step)])

metrics=[f2]
f_model = resnet34




label_csv = f'{PATH}train_v2.csv'
n = len(list(open(label_csv)))-1
val_idxs = get_cv_idxs(n)




def get_data(sz):
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_top_down, max_zoom=1.05)
    return ImageClassifierData.from_csv(PATH, 'train-jpg', label_csv, tfms=tfms,
                    suffix='.jpg', val_idxs=val_idxs, test_name='test-jpg-v2')




data = get_data(256)




x,y = next(iter(data.val_dl))




y




list(zip(data.classes, y[0]))




plt.imshow(data.val_ds.denorm(to_np(x))[0]*1.4);




sz=64




data = get_data(sz)




data = data.resize(int(sz*1.3), TMP_PATH)




learn = ConvLearner.pretrained(f_model, data, metrics=metrics)




lrf=learn.lr_find()
learn.sched.plot()




lr = 0.2




learn.fit(lr, 3, cycle_len=1, cycle_mult=2)




lrs = np.array([lr/9,lr/3,lr])




learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)




learn.save(f'{sz}')




learn.sched.plot_loss()




sz=128




learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)




learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')




sz=256




learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)




learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')




multi_preds, y = learn.TTA()
preds = np.mean(multi_preds, 0)




f2(preds,y)




multi_preds, y = learn.TTA(is_test=True)




preds = np.mean(multi_preds, 0)
preds.shape




test_fnames = [os.path.basename(f).split(".")[0] for f in data.test_ds.fnames]
classes = np.array(data.classes, dtype=str)
res = [" ".join(classes[np.where(pp > 0.2)]) for pp in preds] 
test_df = pd.DataFrame(res, index=test_fnames, columns=['tags'])
test_df.head(5)
test_df.to_csv('submission.csv', index_label='image_name')
df = pd.read_csv('submission.csv')
df.head(20)

