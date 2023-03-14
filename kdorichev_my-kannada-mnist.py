#!/usr/bin/env python
# coding: utf-8



from fastai.vision import *




get_ipython().system('ls ../input/Kannada-MNIST')




path = Path('../input/Kannada-MNIST')
train_csv = path/'train.csv'




data = pd.read_csv(train_csv)




data.head()




y = data.label.values




X = torch.tensor(data.drop('label', axis = 1).values)




X[0].shape




tfms = get_transforms(do_flip=False)




rand_idx = torch.randperm(X.shape[0])
split_ratio = 0.8
split = int(X.shape[0] * split_ratio)
train_idxs = rand_idx[:split]
test_idxs  = rand_idx[split:]

X_train = X[train_idxs]
X_valid = X[test_idxs]
y_train = y[train_idxs]
y_valid = y[test_idxs]
X_train.shape, X_valid.shape




def tensor2Images(x):
    return [Image(x[i].reshape(-1,28,28).repeat(3, 1, 1)/255.) for i in range(x.shape[0])]




class MNISTImageList(ImageList):
    "`ImageList` of Images stored as in `items` as tensor."

    def open(self, fn):
        "No file associated to open"
        pass

    def get(self, i):
        res = self.items[i]
        self.sizes[i] = sys.getsizeof(res)
        return res




til = MNISTImageList(tensor2Images(X_train))




til[0]




train_ll = LabelList(MNISTImageList(tensor2Images(X_train)),CategoryList(y_train, ['0','1','2','3','4','5','6','7','8','9']))
valid_ll = LabelList(MNISTImageList(tensor2Images(X_valid)),CategoryList(y_valid, ['0','1','2','3','4','5','6','7','8','9']))




valid_ll[1][0]




valid_ll[1][1]




ll = LabelLists('.',train_ll,valid_ll)




data.head()




test_csv  = path/'test.csv'
data = pd.read_csv(test_csv)
Xtest = torch.tensor(data.drop('id', axis = 1).values)
test_il = ItemList(tensor2Images(Xtest))




ll.add_test(test_il)




assert len(ll.train.x)==len(ll.train.y)
assert len(ll.valid.x)==len(ll.valid.y)




ll.train.x[0]




dbch = ImageDataBunch.create_from_ll(ll)




dbch.sanity_check()




dbch




dbch.show_batch(rows=4, figsize=(6,6))




path = '/kaggle/working/'




learn = cnn_learner(dbch,models.resnet50,metrics=accuracy, pretrained=True)




learn.freeze()




learn.lr_find()
learn.recorder.plot()




learn.fit_one_cycle(10,slice(2e-3,2e-2))




learn.recorder.plot_losses()




learn.save('stage1')




learn.unfreeze()




learn.lr_find(start_lr=1e-9)
learn.recorder.plot()




learn.fit_one_cycle(10,slice(2e-7,2e-6))




learn.recorder.plot_losses()




learn.show_results(ds_type=DatasetType.Train, rows=4, figsize=(8,10))




learn.fit_one_cycle(5,max_lr=1e-6)




learn.summary()




learn.export()




preds,y = learn.get_preds(ds_type=DatasetType.Test)




y = preds.argmax(dim=1)




assert len(y)==len(test_il)




res = pd.DataFrame(y,columns=['label'],index=range(1, 5001))
res.index.name = 'id'




res.to_csv('submission.csv')

