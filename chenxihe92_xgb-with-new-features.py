#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection, preprocessing
import xgboost as xgb




total = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
macro = pd.read_csv('../input/macro.csv')

df_total = pd.merge(total, macro, on='timestamp', how='left')
df_total.drop('id', axis = 1, inplace = True)
df_total['price_doc'] = np.log(df_total['price_doc'])
Ytotal = df_total['price_doc']

df_test = pd.merge(test, macro, on='timestamp', how='left')
df_test.drop('id', axis = 1, inplace = True)
df_all = pd.concat([df_total,df_test], keys = ['total','test'])

print ('total: ', df_total.shape)
print ('test: ', df_test.shape)
print ('macro: ', macro.shape)
print ('all: ', df_all.shape)




def missingPattern(df):
    numGroup = list(df._get_numeric_data().columns)
    catGroup = list(set(df.columns) - set(numGroup))
    print ('Total categorical/numerical variables are %s/%s' % (len(catGroup), len(numGroup)))
    
    #missing data
    n = df.shape[0]
    count = df.isnull().sum()
    percent = 1.0 * count / n
    dtype = df.dtypes
    # correlation
    missing_data = pd.concat([count, percent,dtype], axis=1, keys=['Count', 'Percent', 'Type'])
    missing_data.sort_values('Count', ascending = False, inplace = True)
    missing_data = missing_data[missing_data['Count'] > 0]
    print ('Total missing columns is %s' % len(missing_data))

    return numGroup, catGroup, missing_data

numGroup, catGroup, missing_data = missingPattern(df_all)
# missing_data




import operator
def getCorr(df, numGroup, eps, *verbose):
    corr = df[numGroup].corr()
#     plt.figure(figsize=(8, 6))
#     plt.pcolor(corr, cmap=plt.cm.Blues)
#     plt.show()
    corr.sort_values(["price_doc"], ascending = False, inplace = True)
    highCorrList = list(corr.price_doc[abs(corr.price_doc)>eps].index)
    if verbose:
        print ("Find most important features relative to target")
        print (corr.price_doc[abs(corr.price_doc)>eps])
    return corr, highCorrList
corr, highCorrList = getCorr(df_all.ix['total',:], numGroup, 0.4, True)




# for numerical variable, draw scatter plot(x vs y) and histogram plot(total vs test)    
def scatterplotNum(df, varNum, ax):
    plt.scatter(df[varNum], df['price_doc'])
    plt.xlabel(varNum)
    plt.ylabel('Price_doc')

def hishplotNum(df, varNum, ax):
    plt.hist(df.ix['total',varNum], bins = 50, alpha = 0.4)
    plt.hist(df.ix['test',varNum], bins = 50, color = 'r', alpha = 0.4)
    plt.xlabel(varNum)
    plt.ylabel('Frequency')
    plt.legend(('total','test'))




high_missing_data = missing_data[missing_data['Percent'] > 0.5]
# print high_missing_data.index
XYcorr = corr['price_doc'].to_dict()

for i in XYcorr:
    if i != 'price_doc' and XYcorr[i] > -1 and i in high_missing_data.index:
        fig = plt.figure(i)
        ax1 = fig.add_subplot(1,1,1)
        scatterplotNum(df_all.ix['total'], i, ax1)
        plt.title('correlation is %.4f' %(XYcorr[i]))

#         ax2 = fig.add_subplot(1,2,2)
#         hishplotNum(pd.concat([total,test],keys = ['total','test']), i, ax2)
#         plt.title('correlation is %.4f' %(XYcorr[i]))
        plt.gcf().set_size_inches(6, 4)
        plt.show()




# remove the heavy missing features
for i in high_missing_data.index:
    df_all.drop(i, axis = 1, inplace = True)

print ('all: ', df_all.shape)




# total missing
# macro missing
basic_missing = list((set(missing_data.index) - set(high_missing_data.index)) & set(total.columns))
macro_missing = list((set(missing_data.index) - set(high_missing_data.index)) & set(macro.columns))
print ('missing in basic: ', len(basic_missing))
print ('missing in macro: ', len(macro_missing))




### for macro info, look at the missing value info(mean, std) groupby yr, year_month
df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
df_all['year'] = df_all.timestamp.dt.year
df_all['year_month'] = df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100




# life_sq and full_sq are highly related to price_doc
# life_sq <= full_sq and full_sq has no missing value

# life_sq or full_sq <= 5
df_all['life_sq'][df_all['life_sq']<=5] = df_all['full_sq'][df_all['life_sq']<=5]
df_all['full_sq'][df_all['full_sq']<=5] = df_all['life_sq'][df_all['full_sq']<=5]


# # life_sq or full_sq > 200 
df_all['life_sq'].ix['total'][1084] = 28.1
df_all['life_sq'].ix['total'][4385] = 42.6
df_all['life_sq'].ix['total'][9237] = 30.1
df_all['life_sq'].ix['total'][9256] = 45.8
df_all['life_sq'].ix['total'][9646] = 80.2
df_all['life_sq'].ix['total'][13546] = 74.78
df_all['life_sq'].ix['total'][13629] = 25.9
df_all['life_sq'].ix['total'][21080] = 34.9
df_all['life_sq'].ix['total'][26342] = 43.5

df_all['life_sq'].ix['test'][601] = 74.2
df_all['life_sq'].ix['test'][1896] = 36.1
df_all['life_sq'].ix['test'][2031] = 23.7
df_all['life_sq'].ix['test'][2791] = 86.9
df_all['life_sq'].ix['test'][5187] = 28.3

df_all['full_sq'].ix['total'][1478] = 35.3
df_all['full_sq'].ix['total'][1610] = 39.4
df_all['full_sq'].ix['total'][2425] = 41.2
df_all['full_sq'].ix['total'][2780] = 72.9
df_all['full_sq'].ix['total'][3527] = 53.3
df_all['full_sq'].ix['total'][5944] = 63.4
df_all['full_sq'].ix['total'][7207] = 46.1


# life_sq > full_sq
df_all['life_sq'][df_all.life_sq > df_all.full_sq] = df_all['full_sq'][df_all.life_sq > df_all.full_sq]

# kitch_sq > full_sq

df_all['kitch_sq'][df_all.kitch_sq > df_all.full_sq] =             df_all['full_sq'][df_all.kitch_sq > df_all.full_sq] - df_all['life_sq'][df_all.kitch_sq > df_all.full_sq]


# else
# floor > max_floor
df_all['max_floor'][df_all.floor > df_all.max_floor] =         df_all['floor'][df_all.floor > df_all.max_floor] + df_all['max_floor'][df_all.floor > df_all.max_floor]




# fill the missing value in train and test
def basicmissingFill(df):
    # num variables
    # pre-processing
    n = df.shape[0]
    
    
    df_all['life_sq'][df_all.life_sq.isnull()] = df_all['full_sq'][df_all.life_sq.isnull()]


    df['state'] = df['state'].replace({33:3})
    df['build_year'][df['build_year'] == 20052009] = 2005
    df['build_year'][df['build_year'] == 4965] = float('nan')
    df['build_year'][df['build_year'] == 0] = float('nan')
    df['build_year'][df['build_year'] == 1] = float('nan')
    df['build_year'][df['build_year'] == 3] = float('nan')
    df['build_year'][df['build_year'] == 71] = float('nan')
    df['build_year'][df['build_year'] == 20] = 2000
    df['build_year'][df['build_year'] == 215] = 2015
    df['build_year'].ix['total'][13117] = 1970


    
    # zero-filling count feature 
    zero_fil = ['build_count_brick','build_count_block','build_count_mix','build_count_before_1920',               'build_count_1921-1945','build_count_1946-1970','build_count_1971-1995','build_count_after_1995',               'build_count_monolith','build_count_slag','build_count_wood','build_count_panel','build_count_frame',               'build_count_foam','preschool_quota']
    for i in zero_fil:
        df[i] = df[i].fillna(0)
    
    # mode-filling: count feature and ID
    mode_fil = ['state','ID_railroad_station_walk','build_year','material','num_room']
    for i in mode_fil:
        df[i] = df[i].fillna(df[i].mode()[0]) 

    # mean-filling
    mean_fil = ['cafe_avg_price_500','cafe_avg_price_1000','cafe_avg_price_1500','cafe_avg_price_2000',               'cafe_avg_price_3000','cafe_avg_price_5000','cafe_sum_500_max_price_avg','cafe_sum_500_min_price_avg',               'cafe_sum_1000_max_price_avg','cafe_sum_1000_min_price_avg','cafe_sum_1500_max_price_avg',               'cafe_sum_1500_min_price_avg','cafe_sum_2000_max_price_avg','cafe_sum_2000_min_price_avg',               'cafe_sum_3000_max_price_avg','cafe_sum_3000_min_price_avg','cafe_sum_5000_max_price_avg',               'cafe_sum_5000_min_price_avg','railroad_station_walk_min','railroad_station_walk_km',               'school_quota','raion_build_count_with_material_info','prom_part_5000',               'raion_build_count_with_builddate_info','green_part_2000','metro_km_walk','metro_min_walk',               'hospital_beds_raion']
    for i in mean_fil:
        grouped = df[['year',i]].groupby('year')
        df[i] = grouped.transform(lambda x: x.fillna(x.mean()))
        
    # exception: 'kitch_sq','floor','max_floor'
    df['kitch_sq'][df.kitch_sq.isnull()] = df['full_sq'][df.kitch_sq.isnull()] - df['life_sq'][df.kitch_sq.isnull()]
    df['floor'] = df['floor'].fillna(df['floor'].mean())
    df['max_floor'][df.max_floor.isnull()] = df['floor'][df.max_floor.isnull()]
    
    #================
    # Cat. variables
    df['product_type'] = df['product_type'].fillna(df['product_type'].mode()[0])
    
    return df

df_all = basicmissingFill(df_all)




print ('basic_missing filling finished: ', df_all[basic_missing].isnull().sum().sum() == 7662)




# for Cat features in macro_missing
macro_missing_obj = []
for i in macro_missing:
    if df_all[i].dtype == object:
        grouped = df_all[['year',i]].groupby(['year',i])
        # print grouped.agg(len)
        macro_missing_obj.append(i)
        # print missing_data.ix[i]
        # print '\n'
# consider to drop macro_missing_obj
for i in macro_missing_obj:
    df_all.drop(i, axis = 1, inplace = True)
    macro_missing.remove(i)

print ('macro missing features count: ', len(macro_missing))
print ('df_all shape: ', df_all.shape)




# for num features in macro_missing
# filling strategy: for each feature->if 2015 is not null: fillna the mean(2015) else: fillna the mean(2014)
def macromissingFill(df):
    for i in macro_missing:
        fill2014 = np.nanmean(df[i][df['year']==2014])
        fill2015 = np.nanmean(df[i][df['year']==2015])
        # income_per_cap: the only macro_missing feature which is not agg by year
        if ~np.isnan(fill2015):
            df[i] = df[i].fillna(fill2015)
        else:
            df[i] = df[i].fillna(fill2014)

    return df

df_all = macromissingFill(df_all)
print ('macro_missing filling finished: ', df_all[macro_missing].isnull().sum().sum() == 0)

    




# running mean price vs timestamp
# len(df_all.timestamp.unique()) -> total 1435 timestamp

def running_mean(df, x, y, agg_list, k, *condition): 
    if condition:
        if condition[0] == 'Xless':
            grouped = df[[x,y]][df[x] <= condition[1]].ix['total'].groupby(x)
        elif condition[0] == 'Xlarger':
            grouped = df[[x,y]][df[x] >= condition[1]].ix['total'].groupby(x)
        elif condition[0] == 'Yless':
            grouped = df[[x,y]][df[y] <= condition[1]].ix['total'].groupby(x)
        elif condition[0] == 'Ylarger':
            grouped = df[[x,y]][df[y] >= condition[1]].ix['total'].groupby(x)  
        else: 
            assert 'Wrong conditions!'
    else:
        grouped = df[[x, y]].ix['total'].groupby(x)
    agg_y = grouped.agg({y: agg_list})[y]

    m, n = agg_y.shape
    px = list(agg_y.index)[:m+1-k]
    py = []
    for i in range(n):
        temp = []
        for j in range(m+1-k):
            temp.append(np.mean(agg_y.iloc[j:j+k,i]))
        py.append(temp)

    # plot
    colors = ['r','g','b','y']
    plt.figure()
    for i in range(len(agg_list)):
        if k == 1:
            plt.scatter(px, py[i], color = colors[i])
        else:
            plt.plot(px, py[i], color = colors[i])
    
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(agg_list)
    plt.title('running mean: ' + str(k))
    plt.gcf().set_size_inches(10, 6)
    plt.show()        
    return px, py

x = 'timestamp'
y = 'price_doc'
agg_list = ['mean','median']
k = 60
px, py = running_mean(df_all, x, y, agg_list, k)




# year and housing sq
df_all['used_yr'] = df_all['year'] - df_all['build_year']
df_all['used_yr'][df_all['used_yr'] < 0] = 0
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_life_sq'] = df_all['life_sq'] / df_all['full_sq'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)
# fillna
df_all['rel_floor'] = df_all['rel_floor'].fillna(df_all['rel_floor'].mean())
df_all['rel_life_sq'] = df_all['rel_life_sq'].fillna(df_all['rel_life_sq'].mean())
df_all['rel_kitch_sq'] = df_all['rel_kitch_sq'].fillna(df_all['rel_kitch_sq'].mean())




numGroup = list(df_all._get_numeric_data().columns)
corr, highCorrList = getCorr(df_all.ix['total',:], numGroup, 0.15)

cntGroup = [i for i in numGroup if re.match(r'\w+_count+',i)]
raionGroup = [i for i in numGroup if re.match(r'\w+_raion+',i)]
kmGroup = [i for i in numGroup if re.match(r'\w+_km+',i)]
minGroup = ['metro_min_avto', 'metro_min_walk', 'public_transport_station_min_walk', 'railroad_station_avto_min',            'railroad_station_walk_min']
print ('cntGroup:', len(cntGroup))
print ('raionGroup: ', len(raionGroup))
print ('kmGroup: ', len(kmGroup))
print ('minGroup: ', len(minGroup))

# for i in kmGroup+minGroup+cntGroup:
#     if sum(df_all[i] < 0) > 0:
#         print i
#     else:
#         print 'positive'
# all positive values in kmGroup, minGroup, cntGroup 




# km and minutes to neighborhood feats transformation
# house full_sq and distance to neighborhood combination

def min_km_trans(df, group):
    group = list(group)
    newFeats = []
    for i in group:
        df['fsq_'+i+'_inv1'] = df['full_sq'] / (df[i] + 0.1)
        df['fsq_'+i+'_inv5'] = df['full_sq'] / (df[i] + 0.5)
        df['fsq_'+i+'_inv10'] = df['full_sq'] / (df[i] + 1.0)
        df['fsq_'+i+'_invlg1'] = df['full_sq'] / (np.log1p(df[i]) + 0.1)
        df['fsq_'+i+'_invlg5'] = df['full_sq'] / (np.log1p(df[i]) + 0.5)
        df['fsq_'+i+'_invlg10'] = df['full_sq'] / (np.log1p(df[i]) + 1.0)
        # df['full_sq_'+i] = df['full_sq'] / (np.log1p(df[i]) + 0.1)
        newFeats += ['fsq_'+i+'_inv1','fsq_'+i+'_inv5','fsq_'+i+'_inv10','fsq_'+i+'_invlg1',                     'fsq_'+i+'_invlg5','fsq_'+i+'_invlg10']
    group.extend(newFeats)
    return df, group

df_all, kmFeats = min_km_trans(df_all, kmGroup)
df_all, minFeats = min_km_trans(df_all, minGroup)




cntFeats = list(set(cntGroup) & set(highCorrList))
extFeats = ['full_sq','life_sq','kitch_sq','floor','max_floor','num_room','build_year',            'used_yr','rel_life_sq','rel_kitch_sq','rel_floor',           'ppi','cpi','price_doc']
print (len(highCorrList),len(cntFeats),len(kmFeats)/7,len(minFeats)/7,len(extFeats))
# print highCorrList
numFeats = cntFeats + kmFeats + minFeats + extFeats




testId = list(test['id'])
drop_list = ['timestamp','year','year_month','price_doc']
for i in drop_list:
    df_all.drop(i, axis = 1, inplace = True)

numGroup,catGroup,_ = missingPattern(df_all)

# self-define numGroup
# numGroup = numFeats

df_total_num = df_all.ix['total',numGroup]
df_test_num = df_all.ix['test',numGroup]
df_total_cat = df_all.ix['total',catGroup]
df_test_cat = df_all.ix['test',catGroup]
print ('Current training numerical variables count is %d '  %(df_total_num.shape[1]))
print ('Current training categorical variables count is %d '  %(df_total_cat.shape[1]))
print ('Current test numerical variables count is %d '  %(df_test_num.shape[1]))
print ('Current test categorical variables count is %d '  %(df_test_cat.shape[1]))




# one-hot encoding for categorical variables
df_concat_cat = pd.concat([df_total_cat,df_test_cat],keys = ['total','test'])
df_total_cat = pd.get_dummies(df_concat_cat).ix['total',:]
df_test_cat = pd.get_dummies(df_concat_cat).ix['test',:]
print ('After one-hot encoding, total training cat variables are %d' %(df_total_cat.shape[1]))
print ('After one-hot encoding, total test cat variables are %d' %(df_test_cat.shape[1]))




#Xtotal = pd.concat([df_total_num,df_total_cat], axis = 1)
#Xtest = pd.concat([df_test_num,df_test_cat], axis = 1)
Xtotal = df_total_num
Xtest = df_test_num
dtrain = xgb.DMatrix(Xtotal, Ytotal)
dtest = xgb.DMatrix(Xtest)




xgb_params = {
    'eta': 0.05,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'lambda': 0.2,
    'alpha': 0.2,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1,
    'scale_pos_weight': 1,
}
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=50,
    verbose_eval=50, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()




num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)




fig, ax = plt.subplots(1, 1, figsize=(8, 12))
xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)




Ypred = model.predict(dtest)
Ypred = np.expm1(Ypred)
output = pd.DataFrame({"id": testId, "price_doc": Ypred})
output.head()




output.to_csv('xgb_submission.csv',index=False)

