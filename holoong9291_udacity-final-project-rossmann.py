#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


os.listdir('../input')


# In[3]:


base_path = '../input/'


# In[4]:


train_data = pd.read_csv(base_path+'train.csv')
test_data = pd.read_csv(base_path+'test.csv')
store_data = pd.read_csv(base_path+'store.csv')


# In[5]:


train_data.Date = pd.to_datetime(train_data.Date)
test_data.Date = pd.to_datetime(test_data.Date)
store_data['PromoInterval'] = store_data['PromoInterval'].astype(str)


# In[6]:


train_data.head(5)


# In[7]:


test_data.head(5)


# In[8]:


store_data.head(5)


# In[9]:


train_data.info()


# In[10]:


test_data.info()


# In[11]:


store_data.info()


# In[12]:


store_data[store_data['CompetitionDistance'].isnull()]


# In[13]:


store_data[store_data['CompetitionOpenSinceMonth'].isnull()][:5]


# In[14]:


store_data[store_data['Promo2SinceWeek'].isnull()][:5]


# In[15]:


test_data.Open.fillna(1, inplace=True)


# In[16]:


store_data.CompetitionOpenSinceYear.fillna(store_data.CompetitionOpenSinceYear.median(), inplace=True)
store_data.CompetitionOpenSinceMonth.fillna(store_data.CompetitionOpenSinceMonth.median(), inplace=True)


# In[17]:


store_data.CompetitionDistance.fillna(store_data.CompetitionDistance.median(), inplace=True)


# In[18]:


train_all = pd.merge(train_data, store_data)
train_all.head(5)


# In[19]:


train_all.info()


# In[20]:


test_all = pd.merge(test_data, store_data)


# In[21]:


test_all.info()


# In[22]:


train_all[train_all['Promo2']==0][:3]


# In[23]:


train_all = train_all.sort_values(['Date'],ascending = False)


# In[24]:


train_all[:10] # 排序成功的话，这十条数据应该都是2015年7月31号的不同商店的数据才对吧，妈呀，坑死人了。。。。。。


# In[25]:


def get_datetime_info(data):
    '''
    data:dataFrame
    return year,quarter,month,day,weekOfYear,isWorkDay
    '''
    return (data.Date.apply(lambda date:date.year), 
            data.Date.apply(lambda date:date.quarter), 
            data.Date.apply(lambda date:date.month), 
            data.Date.apply(lambda date:date.day), 
            data.Date.apply(lambda date:date.weekofyear), 
            data.DayOfWeek.apply(lambda dow:dow<=6)) # 周日不上班


# In[26]:


get_datetime_info(train_all[:1])


# In[27]:


train_data.Date.min()


# In[28]:


train_data.Date.max()


# In[29]:


train_data.Date.unique()[:10]


# In[30]:


print('国家放假平均销售额：'+str(train_data[train_data['SchoolHoliday'] == 1].Sales.mean()))
print('国家不放假平均销售额：'+str(train_data[train_data['SchoolHoliday'] == 0].Sales.mean()))


# In[31]:


print('国家放假平均销售额：'+str(train_data[train_data['StateHoliday'] != 0].Sales.mean()))
print('国家不放假平均销售额：'+str(train_data[train_data['StateHoliday'] == 0].Sales.mean()))


# In[32]:


X=[]
Y=[]
X_=[]
Y_=[]
train_data_store1 = train_all[train_all.Store==1][::-1].reset_index()
train_data_store1 = train_data_store1[:360]
holiday_start = False
holiday_end = False
x_temp = []
y_temp = []
plt.figure(figsize=(15,15))
for i in range(len(train_data_store1)):
    d = train_data_store1.loc[i]
    if d.SchoolHoliday==1 and (i==0 or train_data_store1.loc[i-1].SchoolHoliday==0):
        X.append(d.Date)
        Y.append(9000)
        holiday_start = True
        holiday_end = False
        x_temp = []
        y_temp = []
    if d.SchoolHoliday==0 and (i==len(train_data_store1)-1 or train_data_store1.loc[i-1].SchoolHoliday==1):
        X.append(d.Date)
        Y.append(9500)
        holiday_end = True
        holiday_start = False
    if holiday_start and not holiday_end:
        x_temp.append(d.Date)
        y_temp.append(d.Sales)
    if holiday_end and not holiday_start:
        X_.append(x_temp)
        Y_.append(y_temp)
        holiday_end = False

for i in range(len(X_)):
    plt.plot(X_[i], Y_[i])
for i in range(0, min(len(X), len(Y)), 2):
    plt.plot(X[i:i+2], Y[i:i+2])
        


# In[33]:


plt.figure(figsize=(15,15))
plt.plot(train_all[train_all.Store==1].Date[:365], train_all[train_all.Store==1].Sales[:365])
plt.plot(train_all[train_all.Store==1].Date[365:365+365], train_all[train_all.Store==1].Sales[365:365+365])


# In[34]:


plt.figure(figsize=(15,15))
plt.scatter(train_data[train_data.Store==1].DayOfWeek[365:365+365], train_data[train_data.Store==1].Sales[365:365+365])


# In[35]:


print('1~5平均销售额：'+str(train_data[train_data['DayOfWeek'] <=5].Sales.mean()))
print('6平均销售额：'+str(train_data[train_data['DayOfWeek'] ==6].Sales.mean()))
print('7平均销售额：'+str(train_data[train_data['DayOfWeek'] ==7].Sales.mean()))


# In[36]:


def get_week_month_season_halfyear_year(train_data_store):
    every_day = train_data_store1.Sales
    last_weeks = []
    last_months = []
    last_seasons = []
    last_halfyears = []
    last_years = []
    for i in range(len(every_day)):
        # week
        sales=0
        count=1
        for j in range(i, i-7 if i-7>0 else 0, -1):
            sales+=every_day[j]
            count+=1.
        last_weeks.append(sales/count)
        # month
        sales=0
        count=1
        for j in range(i, i-30 if i-30>0 else 0, -1):
            sales+=every_day[j]
            count+=1.
        last_months.append(sales/count)
        # season
        sales=0
        count=1
        for j in range(i, i-90 if i-90>0 else 0, -1):
            sales+=every_day[j]
            count+=1.
        last_seasons.append(sales/count)
        # halfyear
        sales=0
        count=1
        for j in range(i, i-180 if i-180>0 else 0, -1):
            sales+=every_day[j]
            count+=1.
        last_halfyears.append(sales/count)
        # year
        sales=0
        count=1
        for j in range(i, i-360 if i-360>0 else 0, -1):
            sales+=every_day[j]
            count+=1.
        last_years.append(sales/count)
    return every_day, last_weeks, last_months, last_seasons, last_halfyears, last_years

train_data_store1 = train_all[train_all['Store']==1][::-1].reset_index()
every_day, last_weeks, last_months, last_seasons, last_halfyears, last_years = get_week_month_season_halfyear_year(train_data_store1)
plt.figure(figsize=(15,15))
plt.plot(every_day, label='every day')
plt.plot(last_weeks, label='week')
plt.plot(last_months, label='month')
plt.plot(last_seasons, label='season')
plt.plot(last_halfyears, label='halfyear')
plt.plot(last_years, label='year')
plt.legend()
plt.show()


# In[37]:


pd.DataFrame([last_weeks, last_months, last_seasons, last_halfyears, last_years, 
              list(train_data_store1.Customers)]).corrwith(pd.Series(every_day), axis=1)


# In[38]:


def is_in_promo(data):
    '''
    data:DataFrame。
    return:返回bool值的Seris表示当前是否处于活动中。
    '''
    months_str = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return data.apply(lambda d:False if d.Promo2==0 else (months_str[int(d.Month-1)] in d.PromoInterval), axis=1)


# In[39]:


def get_promo_days(data):
    '''
    return:返回活动已经持续的天数的Series。
    '''
    return data.apply(lambda d:0 if not d.IsInPromo else d.Day, axis=1)


# In[40]:


plt.figure(figsize=(15,15))
plt.scatter(train_all[train_all['CompetitionDistance']<10000].CompetitionDistance, train_all[train_all['CompetitionDistance']<10000].Sales)


# In[41]:


def get_competition_openmonths(data):
    '''
    return:返回截止当前竞争对手的开张时间，月为单位。
    '''
    return data.apply(lambda d:(d.Year-d.CompetitionOpenSinceYear)*12+(d.Month-d.CompetitionOpenSinceMonth), axis=1)


# In[42]:


months = train_all.apply(lambda data:(data.Date.year - data.CompetitionOpenSinceYear)*12+(data.Date.month - data.CompetitionOpenSinceMonth), axis=1)
plt.figure(figsize=(15,15))
plt.scatter(months, train_all.Sales)


# In[43]:


train_all['Year'], train_all['Quarter'], train_all['Month'], train_all['Day'], train_all['WeekOfYear'], train_all['IsWorkDay'] = get_datetime_info(train_all)


# In[44]:


test_all['Year'], test_all['Quarter'], test_all['Month'], test_all['Day'], test_all['WeekOfYear'], test_all['IsWorkDay'] = get_datetime_info(test_all)


# In[45]:


plt.figure(figsize=(15,15))
train_all.groupby(['Quarter']).Sales.mean().plot()


# In[46]:


plt.figure(figsize=(15,15))
train_all.groupby(['Month']).Sales.mean().plot()


# In[47]:


plt.figure(figsize=(15,15))
train_all.groupby(['WeekOfYear']).Sales.mean().plot()


# In[48]:


train_all['IsInPromo'] = is_in_promo(train_all)
train_all.IsInPromo.unique()


# In[49]:


test_all['IsInPromo'] = is_in_promo(test_all)


# In[50]:


train_all['PromoDays'] = get_promo_days(train_all)
train_all.PromoDays.unique()


# In[51]:


test_all['PromoDays'] = get_promo_days(test_all)


# In[52]:


plt.figure(figsize=(15,15))
train_all.groupby(['IsInPromo']).Sales.mean().plot()


# In[53]:


plt.figure(figsize=(15,15))
train_all.groupby(['PromoDays']).Sales.mean().plot()


# In[54]:


train_all['CompetitionOpenMonths'] = get_competition_openmonths(train_all)
train_all.CompetitionOpenMonths.unique()[:10]


# In[55]:


test_all['CompetitionOpenMonths'] = get_competition_openmonths(test_all)


# In[56]:


drop_cols = ['Date', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 
             'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Store']
train_all.drop(drop_cols+['Customers'], axis=1, inplace=True)
test_all.drop(drop_cols, axis=1, inplace=True)


# In[57]:


train_all.columns


# In[58]:


test_all.columns


# In[59]:


# train_all.head(3)
# train_all = pd.get_dummies(train_all, columns=['StateHoliday', 'StoreType', 'Assortment'])
# train_all.info()
# test_all = pd.get_dummies(test_all, columns=['StateHoliday', 'StoreType', 'Assortment'])
# test_all.info()


# In[60]:


code_map = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, '0':0, 
           1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 0:0}
train_all.StateHoliday = train_all.StateHoliday.map(code_map)
train_all.StoreType = train_all.StoreType.map(code_map)
train_all.Assortment = train_all.Assortment.map(code_map)

test_all.StateHoliday = test_all.StateHoliday.map(code_map)
test_all.StoreType = test_all.StoreType.map(code_map)
test_all.Assortment = test_all.Assortment.map(code_map)

print(train_all.StateHoliday.unique())
print(test_all.Assortment.unique())


# In[61]:


#plt.figure(figsize=(15,15))
#plt.plot(train_all.CompetitionDistance)


# In[62]:


print('min:'+str(train_all.CompetitionDistance.min()))


# In[63]:


print('max:'+str(train_all.CompetitionDistance.max()))


# In[64]:


# train_all.CompetitionDistance = ((train_all.CompetitionDistance - train_all.CompetitionDistance.min())/(train_all.CompetitionDistance.max() - train_all.CompetitionDistance.min()))
# test_all.CompetitionDistance = ((test_all.CompetitionDistance - test_all.CompetitionDistance.min())/(test_all.CompetitionDistance.max() - test_all.CompetitionDistance.min()))
# print 'min:'+str(train_all.CompetitionDistance.min())
# print 'max:'+str(train_all.CompetitionDistance.max())
# print 'mean:'+str(train_all.CompetitionDistance.mean())


# In[65]:


target_all = train_all.Sales
target_all.head(5)


# In[66]:


train_all = train_all.drop('Sales', axis=1)
print('Sales' in train_all.columns)


# In[67]:


#from sklearn.cross_validation import train_test_split

#x_train, x_valid, y_train, y_valid = train_test_split(train_all, target_all, test_size=0.1)

x_valid = train_all[:1115*6*7]
x_train = train_all[1115*6*7:]
y_valid = target_all[:1115*6*7]
y_train = target_all[1115*6*7:]

# y做对数处理，数据分布转换
y_train = np.log1p(y_train)
y_valid = np.log1p(y_valid)


# In[68]:


x_train[:5]


# In[69]:


x_valid[:5]


# In[70]:


pred_base = np.expm1(y_train).mean()
print('基准模型预测值：'+str(pred_base))


# In[71]:


def rmspe(y_pred, y_real):
    y_pred = list(y_pred)
    y_real = list(y_real)
    for i in range(len(y_real)):
        if y_real[i]==0:
            y_real[i], y_pred[i] = 1., 1.
    return np.sqrt(np.mean((np.divide(np.subtract(y_real, y_pred),y_real))**2))


# In[72]:


print('基准模型的RMSPE：'+str(rmspe([pred_base]*len(y_valid), np.expm1(y_valid))))


# In[73]:


import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


# In[74]:


train_matrix = xgb.DMatrix(x_train, y_train)
valid_matrix = xgb.DMatrix(x_valid, y_valid)
watchlist = [(train_matrix, 'train'), (valid_matrix, 'valid')]


# In[75]:


ps_first = {
    'max_depth':5,
    'learning_rate':.3,
    'n_estimators':5000,
    'objective':'reg:linear',
    'booster':'gbtree',
    'gamma':0,
    'min_child_weight':1,
    'subsample':1,
    'colsample_bytree':1,
    'random_state':6,
    'silent':True
}

params_first = {
    "objective": "reg:linear",
    "booster" : "gbtree",
    "eta": 0.1,
    "max_depth": 5,
    "silent": 1,
    "seed": 6}
num_boost_round_first = 1000


# In[76]:


def train(params, num_boost_round):
    print('XGBoost Model Train Start....')
    start_time = time.time()
    model = xgb.train(params, train_matrix, num_boost_round, evals=watchlist, early_stopping_rounds=100)
    print('XGBoost Model Train End, Time: {:4f} s....'.format(time.time()-start_time))
    return model

def train2(ps, x, y, x_test, y_test):
    print('XGBRegressor Train Start....')
    start_time = time.time()
    model = XGBRegressor(max_depth=ps['max_depth'], learning_rate=ps['learning_rate'], 
                         n_estimators=ps['n_estimators'],objective=ps['objective'], silent=ps['silent'],
                         booster=ps['booster'], gamma=ps['gamma'], min_child_weight=ps['min_child_weight'],
                         subsample=ps['subsample'], colsample_bytree=ps['colsample_bytree'],
                        random_state=ps['random_state'], n_jobs=-1)
    model.fit(x, y, early_stopping_rounds=100, eval_set=[(x_test,y_test)], verbose=True)
    print('XGBRegressor Train End, Time: {:4f} s....'.format(time.time()-start_time))
    return model


# In[77]:


#model_first = train(params_first, num_boost_round_first)
model_first = train2(ps_first, x_train, y_train, x_valid, y_valid)


# In[78]:


def cv(params, num_boost_round):
    print('XGBoost Model cv Start....')
    start_time = time.time()
    cv = xgb.cv(params, train_matrix, num_boost_round, early_stopping_rounds=100)
    print('XGBoost Model cv End, Time: {:4f} s....'.format(time.time()-start_time))
    return cv


# In[79]:


def predict(model, x_valid, y_valid):
    print('XGBoost Model Valid Start....')
    start_time = time.time()
    pred_valid = model.predict(xgb.DMatrix(x_valid))
    rmspe_value = rmspe(np.expm1(pred_valid), np.expm1(y_valid))
    print('Valid RMSPE:'+str(rmspe_value))
    print('XGBoost Model Valid End, Time: {:4f} s....'.format(time.time()-start_time))
    return pred_valid, rmspe_value

def predict2(model, x_valid, y_valid):
    print('XGBoost Model Valid Start....')
    start_time = time.time()
    pred_valid = model.predict(x_valid)
    rmspe_value = rmspe(np.expm1(pred_valid), np.expm1(y_valid))
    print('Valid RMSPE:'+str(rmspe_value))
    print('XGBoost Model Valid End, Time: {:4f} s....'.format(time.time()-start_time))
    return pred_valid, rmspe_value
    
pred_valid_first, rmspe_first = predict2(model_first, x_valid, y_valid)


# In[80]:


model_first.save_model('./model/first.model')


# In[81]:


ps_opt_estimators = {
    'max_depth':5,
    'learning_rate':.3,
    'n_estimators':1100,
    'objective':'reg:linear',
    'booster':'gbtree',
    'gamma':0,
    'min_child_weight':1,
    'subsample':1,
    'colsample_bytree':1,
    'random_state':6,
    'silent':True,
}


# In[82]:


param_grid_maxdepth_minchildweight = {
 'max_depth':[5, 7, 9],
 'min_child_weight':[1, 3, 5]
}


# In[83]:


def gridSearch(ps, param_grid, X, Y):
    print('XGBRegressor Grid Search Start....')
    start_time = time.time()
    model = XGBRegressor(max_depth=ps['max_depth'], learning_rate=ps['learning_rate'], 
                         n_estimators=ps['n_estimators'],objective=ps['objective'], silent=ps['silent'],
                         booster=ps['booster'], gamma=ps['gamma'], min_child_weight=ps['min_child_weight'],
                         subsample=ps['subsample'], colsample_bytree=ps['colsample_bytree'], random_state=ps['random_state'], n_jobs=-1)
    
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=3)
    grid_result = grid_search.fit(X, Y)
    
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    
    print('XGBRegressor Grid Search End, Time: {:4f} s....'.format(time.time()-start_time))
    return grid_result

grid_result = gridSearch(ps_opt_estimators, param_grid_maxdepth_minchildweight, x_train, y_train)


# In[84]:


ps_opt = {
    'max_depth':9,
    'learning_rate':.03,
    'n_estimators':6000,
    'objective':'reg:linear',
    'booster':'gbtree',
    'subsample':.9,
    'colsample_bytree':.7,
    'random_state':6,
    'silent':True,
}

params_opt = {
    "objective": "reg:linear",
    "booster" : "gbtree",
    "eta": 0.03,
    "max_depth": 10,
    'subsample':.9,
    'colsample_bytree':.7,
    "silent": 1,
    "seed": 6}

model_opt = train2(ps_opt,  x_train, y_train, x_valid, y_valid)
pred_valid_opt, rmspe_opt = predict2(model_opt, x_valid, y_valid)

#model_opt = train(params_opt, 6000)


# In[85]:


model_opt.save_model('/home/kael/projects/model/opt.model


# In[86]:


plt.figure(figsize=(15,15))
plt.scatter(range(len(np.expm1(pred_valid)[::1115])), np.expm1(pred_valid)[::1115], color='red')
plt.scatter(range(len(np.expm1(y_valid)[::1115])), np.expm1(y_valid)[::1115], color='blue')


# In[87]:


np.mean(np.abs(np.expm1(pred_valid)-np.expm1(y_valid)))


# In[88]:


np.mean(np.expm1(pred_valid)-np.expm1(y_valid))


# In[89]:


def get_fix_actor(pred_valid, y_valid):
    results = {}
    for actor in [0.990+i/1000. for i in range(20)]:
        results[actor]=rmspe(y_pred=np.expm1(pred_valid)*actor, y_real=np.expm1(y_valid))
    return sorted(results.items(),key = lambda x:x[1],reverse = True)[-1]

print '校正前：'+str(rmspe(np.expm1(pred_valid), np.expm1(y_valid)))
print '校正后：'
actor_score = get_fix_actor(pred_valid, y_valid)
print actor_score


# In[90]:


plt.figure(figsize=(15,15))
plt.scatter(range(len(np.expm1(pred_valid)[::1115])), np.expm1(pred_valid)[::1115], color='red')
plt.scatter(range(len(np.expm1(pred_valid)[::1115])), np.expm1(pred_valid*actor_score[0])[::1115], color='blue')
plt.scatter(range(len(np.expm1(y_valid)[::1115])), np.expm1(y_valid)[::1115], color='green')


# In[91]:


num_model = 5

print 'XGBoost XModel Train Start....'
start_time = time.time()
models = []
for i in range(num_model):
    params = {
        'objective':'reg:linear', 
        'booster':'gbtree', # 注意此处的提升方式，保证了我们之前一些类别字段映射到01234也是可行的，而不需增加维度
        'eta':.03,
        'max_depth':10,
        'subsample':.9,
        'colsample_bytree':.7,
        'silent':1,
        'seed':10000+i
    }
    num_boost_round = 5000
    model = xgb.train(params, train_matrix, num_boost_round, evals=watchlist, 
                  early_stopping_rounds=100)
    models.append(model)

print 'XGBoost XModel Train End, Time: {:4f} s....'.format(time.time()-start_time)


# In[92]:


actor_scores = []
for i in range(len(models)):
    pred_valid = models[i].predict(xgb.DMatrix(x_valid))
    actor_scores.append(get_fix_actor(pred_valid, y_valid))


# In[93]:


actor_scores


# In[94]:


weights = []
for i in range(len(actor_scores)):
    weights.append(actor_scores[i][1])
weights = [sum(weights)-w for w in weights]
weights = [1.*w/sum(weights) for w in weights]


# In[95]:


weights


# In[96]:


def predict_x(x_valid):
    preds = []
    for m in models:
        preds.append(m.predict(xgb.DMatrix(x_valid)))
    for i in range(len(preds)):
        preds[i] = [p*actor_scores[i][0]*weights[i]for p in preds[i]]
    final_pred = []
    for i in range(len(preds[0])):
        p=0
        for j in range(len(preds)):
            p+=preds[j][i]
        final_pred.append(p)
    return final_pred

print 'X模型融合RMSPE:'+str(rmspe(np.expm1(predict_x(x_valid)), np.expm1(y_valid)))


# In[97]:


for i in range(len(models)):
    models[i].save_model('/home/kael/projects/model/model_'+str(i)+'.model')


# In[98]:


plt.figure(figsize=(15,15))
singal_model_pred_valid = model.predict(xgb.DMatrix(x_valid))*actor_score[0]
x_model_pred_valid = predict_x(x_valid)
plt.scatter(range(len(y_valid[:100])), np.expm1(singal_model_pred_valid[:100]), color='red')
plt.scatter(range(len(y_valid[:100])), np.expm1(x_model_pred_valid[:100]), color='blue')
plt.scatter(range(len(y_valid[:100])), np.expm1(y_valid[:100]), color='green')


# In[ ]:





# In[99]:


test_id = test_all.Id
test_all.drop(['Id'], axis=1, inplace=True)


# In[100]:


pd.DataFrame({'Id':test_id, 'Sales':pd.Series([pred_base]*len(test_id))}).to_csv('submission_base.csv', index=False)


# In[101]:


pred_test = model.predict(xgb.DMatrix(test_all))
pred_test = pred_test*actor_score[0] # 校正系数


# In[102]:


pd.DataFrame({'Id':test_id, 'Sales':np.expm1(pred_test)}).to_csv('submission.csv', index=False)


# In[103]:


pred_x_test = predict_x(test_all)


# In[104]:


pd.DataFrame({'Id':test_id, 'Sales':np.expm1(pred_x_test)}).to_csv('submission_x.csv', index=False)


# In[ ]:





# In[ ]:




