#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')




from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics




#PATH = "data/bulldozers/"
#PATH = "C:/Users/jcat/fastai/data/bulldozers/"
PATH = '../input/'
get_ipython().system('ls {PATH}')




df_raw = pd.read_csv(f'{PATH}Train.csv', low_memory=False, 
                     parse_dates=["saledate"])




def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)




display_all(df_raw.tail().T)




display_all(df_raw.describe(include='all').T)




# replace price with log(price) so that rmse metric will compute rmsle
df_raw.SalePrice = np.log(df_raw.SalePrice)




m = RandomForestRegressor(n_jobs=-1)
# The following code is supposed to fail due to string values in the input data
m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)




add_datepart(df_raw, 'saledate')
df_raw.saleYear.head()




train_cats(df_raw)




df_raw.UsageBand.cat.categories




df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
df_raw.UsageBand = df_raw.UsageBand.cat.codes




display_all(df_raw.isnull().sum().sort_index()/len(df_raw))




get_ipython().system('conda install -c conda-forge feather-format ')




# make tmp directory to store results
os.makedirs('tmp', exist_ok=True)
# why doesn't this work?
df_raw.to_feather('tmp/bulldozers-raw')




# why doesn't this work?
import feather
feather.write_dataframe(df_raw, 'tmp/bulldozers-raw')




# why doesn't this work?
df_raw = pd.read_feather('tmp/bulldozers-raw')
#  why doesn't this work?
df = feather.read_dataframe(path)




# find out what proc_df does:
# as shown below, it fills nulls for all numerical columns, 
# bu adds indicator columns only for float features with nulls
get_ipython().run_line_magic('pinfo2', 'proc_df')




df, y, nas = proc_df(df_raw, 'SalePrice')




m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(df, y)')
m.score(df,y)




def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 12000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape




# m.score defaults to R^2
def rmse(x,y): return math.sqrt( ((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)




m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




# speed-up by taking a subset ~10x smaller than the original dataset
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice', subset=30000, na_dict=nas)
X_train, _ = split_vals(df_trn, 20000)
y_train, _ = split_vals(y_trn, 20000)




# doesn't achieve as good a score as using all the training examples
m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')

# Using same validation set as before!
print_score(m)




# much lower scores, and same for training and validation sets
m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




# find the depth of each tree!
[estimator.tree_.max_depth for estimator in m.estimators_]




# Install graphviz first!
# conda install -c conda-forge python-graphviz 
draw_tree(m.estimators_[0], df_trn, precision=3)




# One tree (of depth 35!) manages to fit nearly all the data because max_depth was not set
m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




# don't do it -- takes too long!!!!!
draw_tree(m.estimators_[0], df_trn, precision=3)




# instead, find the depth of each tree
[estimator.tree_.max_depth for estimator in m.estimators_]




# remember we are using a subset of 30,000 training examples

m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




len(m.estimators_)




# RandomForestRegressor defaults to 10 trees
# bootstrap defaults to True
get_ipython().run_line_magic('pinfo2', 'RandomForestRegressor')




preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:,0], np.mean(preds[:,0]), y_valid[0]




preds.shape




# plot mean r2 score vs number of estimators (trees)
plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)]);




m = RandomForestRegressor(n_estimators=20, n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




m = RandomForestRegressor(n_estimators=80, n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




# oob_score costs only ~5% overhead time
# m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
# with 40 trees, tune tree depth to mitigate overfitting
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, max_depth=6, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




# depths of each tree
depths = [estimator.tree_.max_depth for estimator in m.estimators_]




# start with all the data
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)




# sets RandomForestRegressor to use 20000 samples per tree
# this is a great idea. 
#     can this be done from a direct call to RandomForestRegressor?????
set_rf_samples(20000)




get_ipython().run_line_magic('pinfo2', 'set_rf_samples')




# this is a good result -- would be #74 on the leaderboard
# training, test and oob_error are very close
# first example uses default of 10 trees
m = RandomForestRegressor(n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




# using sample subsets has eliminated the overfitting
#     that was happening from not pruning the trees
# depths of each tree
[estimator.tree_.max_depth for estimator in m.estimators_]




# now increase to 40 trees -- get a slight improvement in score for
#     for both training and oob samples
# up to #60 on the leaderboard
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




# back to full bootstrap sample
reset_rf_samples()




# ????? not clear what this function does
def dectree_max_depth(tree):
    children_left = tree.children_left
    children_right = tree.children_right

    def walk(node_id):
        if (children_left[node_id] != children_right[node_id]):
            left_max = 1 + walk(children_left[node_id])
            right_max = 1 + walk(children_right[node_id])
            return max(left_max, right_max)
        else: # leaf
            return 1

    root_node_id = 0
    return walk(root_node_id)




# using entire bootstrap sample improves metric by quite a bit! 
#     and it looks like we are back to overfitting
#     because training score increased to 0.987...
# note-training score higher than validation score not necesarily
#    a sign of overfitting
# now #17 on the leaderboard!
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




t=m.estimators_[0].tree_




dectree_max_depth(t)




# see what's in a tree
dir(t)




# we are using all the features
print(X_train.shape)
print(t.n_features)




# increase minimum number of samples in a leaf from default of 1 to 5
# decreases wall time because there are fewer trees
# this seems better than directly limiting tree depth
# good for #11 on leaderboard
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=5, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




# first tree
t=m.estimators_[0].tree_




t.max_depth




# why does jeremy's code dectree_max_depth(t) return 53?????
dectree_max_depth(t)




# print tree depths
depths = np.array([estimator.tree_.max_depth for estimator in m.estimators_])
print(depths)
print(np.max(depths))




# my code says that the max tree depth is 41
np.max([estimator.tree_.max_depth for estimator in m.estimators_])




# min_samples of 3 is a good way to limit tree depth,
#     but this is not much different than min_samples_leaf = 5
#     above
# still #11
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




# use a random 50% of the features at each split
# reduces time by ~half!!!!!
# still overfitting?
# empirical rule is to sample sqrt(n_features) at each split
#    oob is worse than at max_featuures = 0.5
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt', n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)
# sampling half the features at each split gets better oob
#    and much better validation metric, 
#    now better than #1 on leaderboard!
#    
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




# tree depths with min_samples_leaf=3
depths = np.array([estimator.tree_.max_depth for estimator in m.estimators_])
print(depths)




# try min_sample_leaf = 10
# now training and oob are much closer, though
# now 'overfitting' is mitigated, 
#     because training and oob scores are closer, 
#     but oob score is not quite as good as before, 
#     validation error is #72 on leaderboard
#     !!!!! perhaps this is why jeremy says that training > oob is not
#         the definition of overfitting
# empirical rule is to sample sqrt(n_features) at each split
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=10, max_features='sqrt', n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




# tree depths with min_samples_leaf=10
# as expected, trees are shallower than for min_samples_leaf=3
depths = np.array([estimator.tree_.max_depth for estimator in m.estimators_])
print(depths)




# what happens if each tree samples subsets instead of entire sample
set_rf_samples(20000)




# back to min_samples_leaf = 3
# worse performance (oob) with 40 trees, 
# but now oob and training are much closer
# much worse validation score #82 on leaderboard
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt', n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




# doesn't improve much with 100 trees
m = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features='sqrt', n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




# revert to sampling the entire data set for each bootstrap sample
reset_rf_samples()




# best model 100 trees with all the samples, min_samples_leaf=3,
#    and max_features=0.5
#    oob score = 91.3
#    improved validation rmsle is 0.2259
#    tried running this again and got 91.29, so the variation is small
#    but note that training score is much higher than oob
#        perhaps that doesn't matter?
m = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




# mean tree depth is 40 with 3 samples in leaf
depths = np.array([estimator.tree_.max_depth for estimator in m.estimators_])
print(depths)
print(np.mean(depths))




# same as best model except min_samples_leaf=1
#    and max_features=0.5
#    oob score = 91.46
#    rmsle validation metric 0.2268, about ~0.001 worse than best model
m = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, max_features=0.5, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




# mean tree depth is 46 with 1 sample in leaf
depths = np.array([estimator.tree_.max_depth for estimator in m.estimators_])
print(depths)
print(np.mean(depths))




# Function to compute R2 score
def R2score(y_data,y_pred):
    dev_tot = y_data - np.mean(y_data)
    dev_res = y_data - y_pred
    SS_res = np.sum(dev_res**2)
    SS_tot = np.sum(dev_tot**2)
    R2 = 1 - SS_res/SS_tot
    return R2




# compute R2 scores for training and validation data
y_valid_pred = np.mean(np.stack([t.predict(X_valid) for t in m.estimators_]),axis=0)
y_train_pred = np.mean(np.stack([t.predict(X_train) for t in m.estimators_]),axis=0)
R2_train = R2score(y_train,y_train_pred)
R2_valid = R2score(y_valid,y_valid_pred)
print(R2_train,R2_valid)




# get and preprocess TrainAndValid set
df_train0 = pd.read_csv(f'{PATH}TrainAndValid.csv', low_memory=False, 
                     parse_dates=["saledate"])
# replace price with log(price) so that rmse metric will compute rmsle
df_train0.SalePrice = np.log(df_train0.SalePrice)
add_datepart(df_train0, 'saledate')
train_cats(df_train0)
# fix UsageBand
df_train0.UsageBand.cat.categories
df_train0.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
df_train0.UsageBand = df_train0.UsageBand.cat.codes
# proc_df takes a data frame df and splits off the response variable, and
# changes the df into an entirely numeric dataframe.
df_train, y_train, nas = proc_df(df_train0, 'SalePrice')
print(df_train.shape,df_train0.shape,len(y_train))




print(sum(df_train0.columns=='SalePrice'))
print(sum(df_train.columns=='SalePrice'))




# the TrainAndValid data set has labels for *all* the examples
df_train0.SalePrice




df_train0.columns[61:]#[51:60]#[41:50]#[31:40]#[21:30]#[11:20]#[2:10]




df_train.columns[60:]#[50:59]#[40:49]#[30:39]#[20:29]#[10:19]#[1:9]




print(sum(df_train.MachineHoursCurrentMeter_na))
print(sum(df_train.auctioneerID_na))




# looks like df_train is clean, no nulls
df_train.info()




# df_train0 is a different story,
#     many columns have nulls;
#     the first two of these are
#     auctioneerID and MachineHoursCurrentMeter
for column in df_train0.columns:
    print(column, df_train0[column].isnull().sum()) 




# get and preprocess Valid set. Has no labels
df_valid0 = pd.read_csv(f'{PATH}Valid.csv', low_memory=False, 
                     parse_dates=["saledate"])
add_datepart(df_valid0, 'saledate')
train_cats(df_valid0)
# fix UsageBand
df_valid0.UsageBand.cat.categories
df_valid0.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
df_valid0.UsageBand = df_valid0.UsageBand.cat.codes

df_valid, _, nas = proc_df(df_valid0)
print(df_valid.shape,df_valid0.shape)




df_valid0.info()




df_valid.columns




df_valid0.columns




# get and preprocess Test set
df_test0 = pd.read_csv(f'{PATH}Test.csv', low_memory=False, 
                     parse_dates=["saledate"])
add_datepart(df_test0, 'saledate')
train_cats(df_test0)
# fix UsageBand
df_test0.UsageBand.cat.categories
df_test0.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
df_test0.UsageBand = df_test0.UsageBand.cat.codes
# find out specifically what does proc_df do
df_test, _, _ = proc_df(df_test0)
print(df_test.shape,df_test0.shape)




df_test0.columns




df_test.columns




df_test0.info()




df_test.info()




# we need to add an extra column labeled auctioneerID_na to df_valid and df_test
#     since in these data sets, auctioneerID had no nulls, all the entries should be False
df_valid['auctioneerID_na']=False
df_test['auctioneerID_na']=False




n_valid




# get the features matrix for TrainAndValid data set
X_train = df_train.values
print(X_train.shape)




# m.score defaults to R^2
def rmse(x,y): return math.sqrt( ((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train),m.score(X_train, y_train)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)




# sample the entire data set for each bootstrap sample
reset_rf_samples()




# best model using only Train data -- n_estimators = 100, min_samples_leaf = 3,
#     max_features = 0.5
# validation metric is 0.2378 # 17 on leaderboard
#     notes
#         n_estimators = 200 gives metric to 0.2375
#         n_estimators = 200 gives metric to 0.2377
#         max_features = 0.25 gives metric 0.2416
#         max_features = 0.375 gives metric 0.2384
#         max_features = 0.5 gives metric 0.2378, benchmark       
#         max_features = 0.5 gives metric 0.2386, shows that variation ~0.001
#         max_features = 0.75 gives metric 0.2392
#         max_features = None gives metric 0.2446
#         min_samples_leaf = 1 gives metric 0.2396
#         min_samples_leaf = 5 gives metric 0.2388
# m = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




# get the features matrix for Train data set
X_test = df_test.values
print(X_test.shape)




df_test.info()




display_all(df_test.describe(include='all').T)




X_test.shape




y_test = m.predict(X_test)




df_solution = pd.DataFrame({'SalesID':df_test['SalesID'],'SalePrice':y_test})




df_solution




df_solution.to_csv('submission.csv',index=False)




# install latest version of kaggle api
#!pip install kaggle --upgrade
#!kaggle --version




# ????? shows all scores as zero
#!kaggle competitions leaderboard bluebook-for-bulldozers --show




#!kaggle kernels list -s bluebook-for-bulldozers




#!kaggle datasets status bluebook-for-bulldozers




#!kaggle competitions files bluebook-for-bulldozers




#!kaggle competitions submit bluebook-for-bulldozers -f submission.csv -m "using TrainAndValid data"






