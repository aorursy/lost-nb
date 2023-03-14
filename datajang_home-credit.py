#!/usr/bin/env python
# coding: utf-8



import os
import gc
import time
import numpy as np
import pandas as pd
from contextlib import contextmanager
import multiprocessing as mp
from functools import partial
from scipy.stats import kurtosis, iqr, skew
from lightgbm import LGBMClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings




warnings.simplefilter(action = 'ignore', category = FutureWarning)




pd.set_option('display.max_rows', 60)
pd.set_option('display.max_columns', 100)




debug = True




num_rows = 30000 if debug else None




NUM_THREADS = 4
DATA_DIRECTORY = "../input/"
SUBMISSION_SUFIX = "_model2_04"




path = DATA_DIRECTORY
num_rows = num_rows




train =pd.read_csv(os.path.join(path, 'application_train.csv'), nrows = num_rows)
test = pd.read_csv(os.path.join(path, 'application_test.csv'), nrows = num_rows)




train.head()




# imbalance 한 Target값 확인
train['TARGET'].value_counts().plot.bar()




# trian과 test를 append 로 합침 (concat으로도 합칠수 있음)
df = train.append(test)
df




del train, test;




gc.collect()




# CODE_GENDER에 XNA(결측치)가 4개가 있으므로 결측치 처리를 해준다
df['CODE_GENDER'].value_counts().plot.bar()
df = df[df['CODE_GENDER'] != 'XNA']




# test 셋과 train set의 AMT_INCOME_TOTAL개수를 맞춤
df = df[df['AMT_INCOME_TOTAL']< 20000000 ]




# DAYS_EMPLOYED 일한 날을 나타내는 데이터인데 outlier값을 없에줌 -> 365243(의미없는 값이 11120개 있음)
(df['DAYS_EMPLOYED'] == 365243).sum()
# NAN 값으로 채움 
df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace = True)




# 핸드폰 바꾼일자를 나타내는 값인데 outlier 값을 없에줌 -> 0(의미없는 값을 NAN으로 처리)
(df['DAYS_LAST_PHONE_CHANGE']==0).sum()
df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace = True)




# docs 에  FLAG_DOC column 을 모두 저장 -> 유한님 EDA를 참고하면 FLAG_DOC의 값이 0~3 으로 나타나 있기 때문
docs = [f for f in df.columns if 'FLAG_DOC' in f]




df['DOCUMENT_COUNT'] = df[docs].sum(axis=1)




df['DOCUMENT_COUNT'].hist()




# kurtosis (분포가 어떻게 되어있는가를 확인하는 통계방법)
# 대부분이 특정값으로 몰려있는것을 확인할 수 있음
df[docs].kurtosis(axis = 1).hist()




df['NEW_DOC_KURT'] = df[docs].kurtosis(axis=1)




# DAYS_BIRTH 라는 값은 태어난 날인데 해당값은 음수이고, 정확한 수치가 아니기 때문에 아래와 같은 함수로 값을 처리한다
df['DAYS_BIRTH']




def get_age_label(days_birth):
    """ Return the age group label (int). """
    age_years = -days_birth / 365
    if age_years < 27: return 1
    elif age_years < 40: return 2
    elif age_years < 50: return 3
    elif age_years < 65: return 4
    elif age_years < 99: return 5
    else: return 0




# apply 와 lambda 함수를 이용하여 df에 적용
df['AGE_RANGE'] = df['DAYS_BIRTH'].apply(lambda x: get_age_label(x))




# df['EXT_SOURCE_1'] , df['EXT_SOURCE_2'] , df['EXT_SOURCE_3'] 값들은 EDA에서 항상 좋은값을 보여왔으므로 뭔진몰라도 다 곱해서 파생변수에 저장
df['EXT_SOURCE_PROD']=df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3'] 




# 어떤 사람이 이 값이 가장 좋다고 해서 가져다 씀..ㅋ
df['EXT_SOURCE_WEIGHTED']=df['EXT_SOURCE_1'] * 2 + df['EXT_SOURCE_2'] * 1 + df['EXT_SOURCE_3'] * 3




np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')




# eval 을 사용해서 반복작업을 해결함
for function_name in ['min', 'max', 'mean', 'nanmedian', 'var']:
    feature_name = 'EXT_SOURCES_{}'.format(function_name.upper())
    df[feature_name] = eval('np.{}'.format(function_name))(df[['EXT_SOURCE_1', 'EXT_SOURCE_2','EXT_SOURCE_3']], axis =1)




df




# 연금보험당 융자(빚)를 구한 파생변수를 생성
df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
# 상품가격당 융자(빚)를 구한 파생변수를 생성
df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']




df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
df['INCOME_TO_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
df['INCOME_TO_BIRTH_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']




df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
df['CAR_TO_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']




group = ['ORGANIZATION_TYPE', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'AGE_RANGE', 'CODE_GENDER']




df, group_cols, counted, agg_name = df, group, 'EXT_SOURCES_MEAN','GROUP_EXT_SOURCES_MEDIAN'




df[group_cols + [counted]]




df[group_cols + [counted]].groupby(group_cols)[counted].median().reset_index()




gp = df[group_cols + [counted]].groupby(group_cols)[counted].median().reset_index().rename(columns = {counted:agg_name})
gp




df = df.merge(gp, on = group_cols , how = 'left')
df




agg = {}
agg['counted'] = {'max','min','median','std','mean'}




F = df[group_cols + [counted]].groupby(group_cols)[counted].agg(agg).reset_index()




# aggregate을 활용한 값에 이름 넣기
F.columns = [''.join(col).strip() for col in F.columns.values]
F




def do_mean(df, group_cols, counted, agg_name):
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].mean().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def do_median(df, group_cols, counted, agg_name):
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].median().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def do_std(df, group_cols, counted, agg_name):
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].std().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def do_sum(df, group_cols, counted, agg_name):
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].sum().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df




df = do_median(df, group, 'EXT_SOURCES_MEAN', 'GROUP_EXT_SOURCES_MEDIAN')
df = do_std(df, group, 'EXT_SOURCES_MEAN', 'GROUP_EXT_SOURCES_STD')
df = do_mean(df, group, 'AMT_INCOME_TOTAL', 'GROUP_INCOME_MEAN')
df = do_std(df, group, 'AMT_INCOME_TOTAL', 'GROUP_INCOME_STD')
df = do_mean(df, group, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP_CREDIT_TO_ANNUITY_MEAN')
df = do_std(df, group, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP_CREDIT_TO_ANNUITY_STD')
df = do_mean(df, group, 'AMT_CREDIT', 'GROUP_CREDIT_MEAN')
df = do_mean(df, group, 'AMT_ANNUITY', 'GROUP_ANNUITY_MEAN')
df = do_std(df, group, 'AMT_ANNUITY', 'GROUP_ANNUITY_STD')




#라벨 인코더 함수 사용, sklearn 의 labelencoder 보다 pandas의 factorize가 더 빠름
def label_encoder(df, categorical_columns=None):
    """Encode categorical values as integers (0,1,2,3...) with pandas.factorize. """
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_columns:
        df[col], uniques = pd.factorize(df[col])
    return df, categorical_columns




df, le_encoded_cols = label_encoder(df, None)




# EDA를 통해 나온 결과값을 보고 필요없는 column 들을 삭제
def drop_application_columns(df):
    """ Drop features based on permutation feature importance. """
    drop_list = [
        'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'HOUR_APPR_PROCESS_START',
        'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 'FLAG_PHONE',
        'FLAG_OWN_REALTY', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
        'REG_CITY_NOT_WORK_CITY', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
        'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_YEAR', 
        'COMMONAREA_MODE', 'NONLIVINGAREA_MODE', 'ELEVATORS_MODE', 'NONLIVINGAREA_AVG',
        'FLOORSMIN_MEDI', 'LANDAREA_MODE', 'NONLIVINGAREA_MEDI', 'LIVINGAPARTMENTS_MODE',
        'FLOORSMIN_AVG', 'LANDAREA_AVG', 'FLOORSMIN_MODE', 'LANDAREA_MEDI',
        'COMMONAREA_MEDI', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'BASEMENTAREA_AVG',
        'BASEMENTAREA_MODE', 'NONLIVINGAPARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 
        'LIVINGAPARTMENTS_AVG', 'ELEVATORS_AVG', 'YEARS_BUILD_MEDI', 'ENTRANCES_MODE',
        'NONLIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'LIVINGAPARTMENTS_MEDI',
        'YEARS_BUILD_MODE', 'YEARS_BEGINEXPLUATATION_AVG', 'ELEVATORS_MEDI', 'LIVINGAREA_MEDI',
        'YEARS_BEGINEXPLUATATION_MODE', 'NONLIVINGAPARTMENTS_AVG', 'HOUSETYPE_MODE',
        'FONDKAPREMONT_MODE', 'EMERGENCYSTATE_MODE'
    ]
    for doc_num in [2,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20,21]:
        drop_list.append('FLAG_DOCUMENT_{}'.format(doc_num))
    df.drop(drop_list, axis=1, inplace=True)
    return df




# bureau는 다른 회사의 카드거래 data를 가져온것
bureau = pd.read_csv(os.path.join(path, 'bureau.csv'), nrows = num_rows)
bureau




# Credit duration and credit/account end date difference
bureau['CREDIT_DURATION'] = -bureau['DAYS_CREDIT'] + bureau['DAYS_CREDIT_ENDDATE']
bureau['ENDDATE_DIF'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']




# Credit to debt ratio and difference
bureau['DEBT_PERCENTAGE'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_CREDIT_SUM_DEBT']
bureau['DEBT_CREDIT_DIFF'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
bureau['CREDIT_TO_ANNUITY_RATIO'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_ANNUITY']




'''
Label encoder 함수와 one-hot encoder 함수는 거의 흡사하지만 one-hot 에서는 Null값을 encoder 해주거나 안해줄수 있음

현재 커널에서는 LGBM을 사용할거라서 Null값의 유무가 상관없으므로 Null값을 encoder 해주지 않는다.

'''

def one_hot_encoder(df, categorical_columns=None, nan_as_category=True):
    """Create a new column for each categorical value in categorical columns. """
    original_columns = list(df.columns)
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    categorical_columns = [c for c in df.columns if c not in original_columns]
    return df, categorical_columns




bureau, categorical_columns = one_hot_encoder(bureau, nan_as_category=False)




bb = pd.read_csv(os.path.join(path, 'bureau_balance.csv'), nrows= num_rows)
bb




bb, categorical_cols = one_hot_encoder(bb, nan_as_category= False)




bb_processed = bb.groupby('SK_ID_BUREAU')[categorical_cols].mean().reset_index()




agg = {'MONTHS_BALANCE' : ['min','max','mean','size']}
#bb_processed = group_and_merge(bb, bb_processed, '', agg, 'SK_ID_BUREAU')




df_to_agg = bb
df_to_merge = bb_processed
prefix = ''
aggregations = agg
aggregate_by = 'SK_ID_BUREAU'




def group(df_to_agg, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
    agg_df = df_to_agg.groupby(aggregate_by).agg(aggregations)
    agg_df.columns = pd.Index(['{}{}_{}'.format(prefix, e[0], e[1].upper())
                               for e in agg_df.columns.tolist()])
    return agg_df.reset_index()




def group_and_merge(df_to_agg, df_to_merge, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
    agg_df = group(df_to_agg, prefix, aggregations, aggregate_by= aggregate_by)
    return df_to_merge.merge(agg_df, how='left', on= aggregate_by)




def get_bureau_balance(path, num_rows= None):
    bb = pd.read_csv(os.path.join(path, 'bureau_balance.csv'), nrows= num_rows)
    bb, categorical_cols = one_hot_encoder(bb, nan_as_category= False)
    # Calculate rate for each category with decay
    bb_processed = bb.groupby('SK_ID_BUREAU')[categorical_cols].mean().reset_index()
    # Min, Max, Count and mean duration of payments (months)
    agg = {'MONTHS_BALANCE': ['min', 'max', 'mean', 'size']}
    bb_processed = group_and_merge(bb, bb_processed, '', agg, 'SK_ID_BUREAU')
    del bb; gc.collect()
    return bb_processed




bureau = bureau.merge(get_bureau_balance(path, num_rows), how='left', on='SK_ID_BUREAU')
bureau




bureau['STATUS_12345'] = 0
for i in range(1,6):
        bureau['STATUS_12345'] += bureau['STATUS_{}'.format(i)]




features = ['AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_CREDIT_SUM',
       'AMT_CREDIT_SUM_DEBT', 'DEBT_PERCENTAGE', 'DEBT_CREDIT_DIFF', 'STATUS_0', 'STATUS_12345']




# MONTHS_BALANCE_SIZE(이자가 남은 기간)을 기준으로 features의 평균을 전부 구한것
# 82개의 평균값이 또다른 데이터가 될 수 있다..?
agg_length = bureau.groupby('MONTHS_BALANCE_SIZE')[features].mean().reset_index()




# rename을 정말 똑똑하게 했내..
agg_length.rename({feat:'LL_' + feat for feat in features}, axis = 1, inplace = True)




{feat:'LL_' + feat for feat in features}




bureau = bureau.merge(agg_length, how = 'left', on = 'MONTHS_BALANCE_SIZE')




del agg_length; gc.collect()




BUREAU_AGG = {
    'SK_ID_BUREAU': ['nunique'],
    'DAYS_CREDIT': ['min', 'max', 'mean'],
    'DAYS_CREDIT_ENDDATE': ['min', 'max'],
    'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
    'AMT_ANNUITY': ['mean'],
    'DEBT_CREDIT_DIFF': ['mean', 'sum'],
    'MONTHS_BALANCE_MEAN': ['mean', 'var'],
    'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
    # Categorical
    'STATUS_0': ['mean'],
    'STATUS_1': ['mean'],
    'STATUS_12345': ['mean'],
    'STATUS_C': ['mean'],
    'STATUS_X': ['mean'],
    'CREDIT_ACTIVE_Active': ['mean'],
    'CREDIT_ACTIVE_Closed': ['mean'],
    'CREDIT_ACTIVE_Sold': ['mean'],
    'CREDIT_TYPE_Consumer credit': ['mean'],
    'CREDIT_TYPE_Credit card': ['mean'],
    'CREDIT_TYPE_Car loan': ['mean'],
    'CREDIT_TYPE_Mortgage': ['mean'],
    'CREDIT_TYPE_Microloan': ['mean'],
    # Group by loan duration features (months)
    'LL_AMT_CREDIT_SUM_OVERDUE': ['mean'],
    'LL_DEBT_CREDIT_DIFF': ['mean'],
    'LL_STATUS_12345': ['mean'],
}




bureau.groupby('SK_ID_CURR').agg(BUREAU_AGG)




agg_bureau = group(bureau, 'BUREAU_', BUREAU_AGG)
agg_bureau




BUREAU_ACTIVE_AGG = {
    'DAYS_CREDIT': ['max', 'mean'],
    'DAYS_CREDIT_ENDDATE': ['min', 'max'],
    'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
    'AMT_CREDIT_SUM': ['max', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
    'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean'],
    'DAYS_CREDIT_UPDATE': ['min', 'mean'],
    'DEBT_PERCENTAGE': ['mean'],
    'DEBT_CREDIT_DIFF': ['mean'],
    'CREDIT_TO_ANNUITY_RATIO': ['mean'],
    'MONTHS_BALANCE_MEAN': ['mean', 'var'],
    'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
}




active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
active




agg_bureau = group_and_merge(active, agg_bureau, 'BUREAU_ACTIVE_',BUREAU_ACTIVE_AGG)




closed = bureau[bureau['CREDIT_ACTIVE_Closed']==1]
closed




BUREAU_CLOSED_AGG = {
    'DAYS_CREDIT': ['max', 'var'],
    'DAYS_CREDIT_ENDDATE': ['max'],
    'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
    'AMT_CREDIT_SUM_OVERDUE': ['mean'],
    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['max', 'sum'],
    'DAYS_CREDIT_UPDATE': ['max'],
    'ENDDATE_DIF': ['mean'],
    'STATUS_12345': ['mean'],
}




del active, closed; gc.collect()




BUREAU_LOAN_TYPE_AGG = {
    'DAYS_CREDIT': ['mean', 'max'],
    'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
    'AMT_CREDIT_SUM': ['mean', 'max'],
    'AMT_CREDIT_SUM_DEBT': ['mean', 'max'],
    'DEBT_PERCENTAGE': ['mean'],
    'DEBT_CREDIT_DIFF': ['mean'],
    'DAYS_CREDIT_ENDDATE': ['max'],
}




# Aggregations for the main loan types
for credit_type in ['Consumer credit', 'Credit card', 'Mortgage', 'Car loan', 'Microloan']:
    type_df = bureau[bureau['CREDIT_TYPE_' + credit_type] == 1]
    prefix = 'BUREAU_' + credit_type.split(' ')[0].upper() + '_'
    agg_bureau = group_and_merge(type_df, agg_bureau, prefix, BUREAU_LOAN_TYPE_AGG)
    del type_df; gc.collect()




'BUREAU_' + credit_type.split(' ')[0].upper() + '_'




BUREAU_TIME_AGG = {
    'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
    'AMT_CREDIT_SUM_OVERDUE': ['mean'],
    'AMT_CREDIT_SUM': ['max', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
    'DEBT_PERCENTAGE': ['mean'],
    'DEBT_CREDIT_DIFF': ['mean'],
    'STATUS_0': ['mean'],
    'STATUS_12345': ['mean'],
}




# Time based aggregations: last x months
for time_frame in [6, 12]:
    prefix = "BUREAU_LAST{}M_".format(time_frame)
    time_frame_df = bureau[bureau['DAYS_CREDIT'] >= -30*time_frame]
    agg_bureau = group_and_merge(time_frame_df, agg_bureau, prefix, BUREAU_TIME_AGG)
    del time_frame_df; gc.collect()




sort_bureau = bureau.sort_values(by=['DAYS_CREDIT'])




#last()는 가장 최근것을 보여줌
gr = sort_bureau.groupby('SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE'].last().reset_index()




gr.rename(columns={'AMT_CREDIT_MAX_OVERDUE':'BUREAU_LAST_LOAN_MAX_OVERDUE'},inplace = True)




agg_bureau = agg_bureau.merge(gr, on = 'SK_ID_CURR', how = 'left')




agg_bureau['BUREAU_DEBT_OVER_CREDIT'] =     agg_bureau['BUREAU_AMT_CREDIT_SUM_DEBT_SUM']/agg_bureau['BUREAU_AMT_CREDIT_SUM_SUM']
agg_bureau['BUREAU_ACTIVE_DEBT_OVER_CREDIT'] =     agg_bureau['BUREAU_ACTIVE_AMT_CREDIT_SUM_DEBT_SUM']/agg_bureau['BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM']




df = df.merge(agg_bureau, on = 'SK_ID_CURR', how='left')




df.head()




prev = pd.read_csv(os.path.join(path, 'previous_application.csv'), nrows= num_rows)
pay = pd.read_csv(os.path.join(path, 'installments_payments.csv'), nrows= num_rows)




ohe_columns = [
        'NAME_CONTRACT_STATUS', 'NAME_CONTRACT_TYPE', 'CHANNEL_TYPE',
        'NAME_TYPE_SUITE', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION',
        'NAME_PRODUCT_TYPE', 'NAME_CLIENT_TYPE']




prev, categorical_cols = one_hot_encoder(prev, ohe_columns, nan_as_category=False)




# AMT_APPLICATION -> 신용등급?
prev['APPLICATION_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
prev['APPLICATION_CREDIT_RATIO'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
prev['CREDIT_TO_ANNUITY_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_ANNUITY']
prev['DOWN_PAYMENT_TO_CREDIT'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_CREDIT']




# Interest ratio on previous application
# 간단한 이자 공식 만듦
total_payment = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']
(total_payment/prev['AMT_CREDIT'] -1) / prev['CNT_PAYMENT']
prev['SIMPLE_INTERESTS'] = (total_payment/prev['AMT_CREDIT'] -1) / prev['CNT_PAYMENT']




approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
active_df = approved[approved['DAYS_LAST_DUE']== 365243]




# home credit data 설명 참고
# previous_application의 정보를 얻기 위해서는 pay의  installments_payments를 이용해야 하고, 그때 필요한것이 SK_ID_PREV이다.
# 여기에서 isin 을 사용해서 pay의 SK_ID_PREV가 active_df의 SK_ID_PREV와 일치하는지를 보는것, boolean 형식으로 나옴 -> True값만 뽑음
# Find how much was already payed in active loans (using installments csv)
active_pay = pay[pay['SK_ID_PREV'].isin(active_df['SK_ID_PREV'])]
active_pay_agg = active_pay.groupby('SK_ID_PREV')[['AMT_INSTALMENT', 'AMT_PAYMENT']].sum()
active_pay_agg.reset_index(inplace = True)




# Active loans: difference of what was payed and installments
active_pay_agg['INSTALMENT_PAYMENT_DIFF'] = active_pay_agg['AMT_INSTALMENT'] - active_pay_agg['AMT_PAYMENT']
# Merge with active_df
active_df = active_df.merge(active_pay_agg, on = 'SK_ID_PREV', how = 'left')
active_df['REMAINING_DEBT'] = active_df['AMT_CREDIT'] - active_df['AMT_PAYMENT']
active_df['REPAYMENT_RATIO'] = active_df['AMT_PAYMENT'] / active_df['AMT_CREDIT']




PREVIOUS_ACTIVE_AGG = {
    'SK_ID_PREV': ['nunique'],
    'SIMPLE_INTERESTS': ['mean'],
    'AMT_ANNUITY': ['max', 'sum'],
    'AMT_APPLICATION': ['max', 'mean'],
    'AMT_CREDIT': ['sum'],
    'AMT_DOWN_PAYMENT': ['max', 'mean'],
    'DAYS_DECISION': ['min', 'mean'],
    'CNT_PAYMENT': ['mean', 'sum'],
    'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
    # Engineered features
    'AMT_PAYMENT': ['sum'],
    'INSTALMENT_PAYMENT_DIFF': ['mean', 'max'],
    'REMAINING_DEBT': ['max', 'mean', 'sum'],
    'REPAYMENT_RATIO': ['mean'],
}




# Perform aggregations for active applications
active_agg_df = group(active_df,'PREV_ACTIVE_',PREVIOUS_ACTIVE_AGG)




active_agg_df['TOTAL_REPAYMENT_RATIO'] = active_agg_df['PREV_ACTIVE_AMT_PAYMENT_SUM']/                                         active_agg_df['PREV_ACTIVE_AMT_CREDIT_SUM']




del active_pay, active_pay_agg, active_df; gc.collect()




# change 364,243 values to nan
prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace = True)
prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace = True)
prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace = True)
prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace = True)
prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace = True)




# Days last due difference(유효기간)
prev['DAYS_LAST_DUE_DIFF'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_LAST_DUE']
approved['DAYS_LAST_DUE_DIFF'] = approved['DAYS_LAST_DUE_1ST_VERSION'] - approved['DAYS_LAST_DUE']




#Categorical features
categorical_agg = {key:['mean'] for key in categorical_cols}




PREVIOUS_AGG = {
    'SK_ID_PREV': ['nunique'],
    'AMT_ANNUITY': ['min', 'max', 'mean'],
    'AMT_DOWN_PAYMENT': ['max', 'mean'],
    'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
    'RATE_DOWN_PAYMENT': ['max', 'mean'],
    'DAYS_DECISION': ['min', 'max', 'mean'],
    'CNT_PAYMENT': ['max', 'mean'],
    'DAYS_TERMINATION': ['max'],
    # Engineered features
    'CREDIT_TO_ANNUITY_RATIO': ['mean', 'max'],
    'APPLICATION_CREDIT_DIFF': ['min', 'max', 'mean'],
    'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean', 'var'],
    'DOWN_PAYMENT_TO_CREDIT': ['mean'],
}




#Perform general aggregations
#{**PREVIOUS_AGG, **categorical_agg}는 두개를 합치는거임
agg_prev = group(prev, 'PREV_', {**PREVIOUS_AGG, **categorical_agg})




# Merge active loans dataframe on agg_prev
agg_prev = agg_prev.merge(active_agg_df, how='left', on = 'SK_ID_CURR')




del active_agg_df; gc.collect()




PREVIOUS_APPROVED_AGG = {
    'SK_ID_PREV': ['nunique'],
    'AMT_ANNUITY': ['min', 'max', 'mean'],
    'AMT_CREDIT': ['min', 'max', 'mean'],
    'AMT_DOWN_PAYMENT': ['max'],
    'AMT_GOODS_PRICE': ['max'],
    'HOUR_APPR_PROCESS_START': ['min', 'max'],
    'DAYS_DECISION': ['min', 'mean'],
    'CNT_PAYMENT': ['max', 'mean'],
    'DAYS_TERMINATION': ['mean'],
    # Engineered features
    'CREDIT_TO_ANNUITY_RATIO': ['mean', 'max'],
    'APPLICATION_CREDIT_DIFF': ['max'],
    'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean'],
    # The following features are only for approved applications
    'DAYS_FIRST_DRAWING': ['max', 'mean'],
    'DAYS_FIRST_DUE': ['min', 'mean'],
    'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
    'DAYS_LAST_DUE': ['max', 'mean'],
    'DAYS_LAST_DUE_DIFF': ['min', 'max', 'mean'],
    'SIMPLE_INTERESTS': ['min', 'max', 'mean'],
}




PREVIOUS_REFUSED_AGG = {
    'AMT_APPLICATION': ['max', 'mean'],
    'AMT_CREDIT': ['min', 'max'],
    'DAYS_DECISION': ['min', 'max', 'mean'],
    'CNT_PAYMENT': ['max', 'mean'],
    # Engineered features
    'APPLICATION_CREDIT_DIFF': ['min', 'max', 'mean', 'var'],
    'APPLICATION_CREDIT_RATIO': ['min', 'mean'],
    'NAME_CONTRACT_TYPE_Consumer loans': ['mean'],
    'NAME_CONTRACT_TYPE_Cash loans': ['mean'],
    'NAME_CONTRACT_TYPE_Revolving loans': ['mean'],
}




# Aggregations for approved and refused loans
agg_prev = group_and_merge(approved, agg_prev, 'APPROVED_', PREVIOUS_APPROVED_AGG)
refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
agg_prev = group_and_merge(refused, agg_prev, 'REFUSED_', PREVIOUS_REFUSED_AGG)
del approved, refused; gc.collect()




# Aggregations for Consumer loans and Cash loans
for loan_type in ['Consumer loans', 'Cash loans']:
    type_df = prev[prev['NAME_CONTRACT_TYPE_{}'.format(loan_type)] == 1]
    prefix = 'PREV_' + loan_type.split(" ")[0] + '_'
    agg_prev = group_and_merge(type_df, agg_prev, prefix, PREVIOUS_LOAN_TYPE_AGG)
    del type_df; gc.collect()




PREVIOUS_LOAN_TYPE_AGG = {
    'AMT_CREDIT': ['sum'],
    'AMT_ANNUITY': ['mean', 'max'],
    'SIMPLE_INTERESTS': ['min', 'mean', 'max', 'var'],
    'APPLICATION_CREDIT_DIFF': ['min', 'var'],
    'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean'],
    'DAYS_DECISION': ['max'],
    'DAYS_LAST_DUE_1ST_VERSION': ['max', 'mean'],
    'CNT_PAYMENT': ['mean'],
}




PREVIOUS_LATE_PAYMENTS_AGG = {
    'DAYS_DECISION': ['min', 'max', 'mean'],
    'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
    # Engineered features
    'APPLICATION_CREDIT_DIFF': ['min'],
    'NAME_CONTRACT_TYPE_Consumer loans': ['mean'],
    'NAME_CONTRACT_TYPE_Cash loans': ['mean'],
    'NAME_CONTRACT_TYPE_Revolving loans': ['mean'],
}




# Get the SK_ID_PREV for loans with late payments (days past due, 기한이 지난 날)
pay['LATE_PAYMENT'] = pay['DAYS_ENTRY_PAYMENT'] - pay['DAYS_INSTALMENT']
pay['LATE_PAYMENT'] = pay['LATE_PAYMENT'].apply(lambda x:1 if x>0 else 0)
# dpd_id는 연체가 있는 사람들의 정보를 알아냄
dpd_id = pay[pay['LATE_PAYMENT']>0]['SK_ID_PREV'].unique()




# prev에 연체가 있는 사람들만 추출
prev[prev['SK_ID_PREV'].isin(dpd_id)]




#Aggregations for loans with late payments
agg_dpd = group_and_merge(prev[prev['SK_ID_PREV'].isin(dpd_id)],agg_prev,'PREV_LATE_',PREVIOUS_LATE_PAYMENTS_AGG)
del agg_dpd, dpd_id; gc.collect




PREVIOUS_TIME_AGG = {
    'AMT_CREDIT': ['sum'],
    'AMT_ANNUITY': ['mean', 'max'],
    'SIMPLE_INTERESTS': ['mean', 'max'],
    'DAYS_DECISION': ['min', 'mean'],
    'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
    # Engineered features
    'APPLICATION_CREDIT_DIFF': ['min'],
    'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean'],
    'NAME_CONTRACT_TYPE_Consumer loans': ['mean'],
    'NAME_CONTRACT_TYPE_Cash loans': ['mean'],
    'NAME_CONTRACT_TYPE_Revolving loans': ['mean'],
}




#Aggregataions for loans in the last x months
#-30을 곱해주는 이유는 prev['DAYS_DECISION']의 값이 음수이고, 한달을 의미하기 때문, 30*12 = 360, 30*24 = 720
for time_frame in [12,24]:
    time_frame_df = prev[prev['DAYS_DECISION'] >= -30*time_frame]
    prefix = 'PREV_LAST{}M_'.format(time_frame)
    agg_prev = group_and_merge(time_frame_df, agg_prev, prefix , PREVIOUS_TIME_AGG)
    del time_frame_df; gc.collect()




POS_CASH_AGG = {
    'SK_ID_PREV': ['nunique'],
    'MONTHS_BALANCE': ['min', 'max', 'size'],
    'SK_DPD': ['max', 'mean', 'sum', 'var'],
    'SK_DPD_DEF': ['max', 'mean', 'sum'],
    'LATE_PAYMENT': ['mean']
}




pos = pd.read_csv(os.path.join(path,'POS_CASH_balance.csv'), nrows =num_rows)




# null value가 있어도 LGBM이 가능하므로 Null 안없엠
pos, categorical_cols = one_hot_encoder(pos, nan_as_category= False)




#Flag months with late payment
#pos data의 연체정보 파생변수 생성
pos['LATE_PAYMENT'] = pos['SK_DPD'].apply(lambda x:1 if x>0 else 0)




#Aggregate by SK_ID_CURR
categorical_agg = {key : ['mean'] for key in categorical_cols}
pos_agg = group(pos, 'POS_', {**POS_CASH_AGG, **categorical_agg})




#Sort and group by SK_ID_PREV
sort_pos = pos.sort_values(by = ['SK_ID_PREV', 'MONTHS_BALANCE'])
gp = sort_pos.groupby('SK_ID_PREV')
df = pd.DataFrame()
df['SK_ID_CURR'] = gp['SK_ID_CURR'].first()
df['MONTHS_BALANCE_MAX'] = gp['MONTHS_BALANCE'].max()




#Percentage of previous loans completed and completed before initial term
#CNT_INSTALMENT = 이전 CREDIT의 기간

df['POS_LOAN_COMPLETED_MEAN'] = gp['NAME_CONTRACT_STATUS_Completed'].mean()
df['POS_COMPLETED_BEFORE_MEAN'] = gp['CNT_INSTALMENT'].first() - gp['CNT_INSTALMENT'].last()
df['POS_COMPLETED_BEFORE_MEAN'] = df.apply(lambda x:1 if x['POS_COMPLETED_BEFORE_MEAN']>0 and x['POS_LOAN_COMPLETED_MEAN'] > 0 else 0, axis = 1)




# Number of remaining installments (future installments) and percentage from total
# 잔여 할부 건수(향후 할부) 및 총 할부 비율
df['POS_REMAINING_INSTALMENTS'] = gp['CNT_INSTALMENT_FUTURE'].last()
df['POS_REMAINING_INSTALMENTS_RATIO'] = gp['CNT_INSTALMENT_FUTURE'].last()/gp['CNT_INSTALMENT'].last()




# Group by SK_ID_CURR and merge
df_gp = df.groupby('SK_ID_CURR').sum().reset_index()
df_gp.drop(['MONTHS_BALANCE_MAX'], axis = 1, inplace = True)
pos_agg = pd.merge(pos_agg, df_gp, on = 'SK_ID_CURR', how= 'left')
del df, gp, df_gp, sort_pos; gc.collect()




'''
def do_sum(df, group_cols, counted, agg_name):
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].sum().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df
'''
pos = do_sum(pos, ['SK_ID_PREV'],'LATE_PAYMENT','LATE_PAYMENT_SUM')
# pos.groupby(['SK_ID_PREV'])['LATE_PAYMENT'].sum().reset_index().rename(columns = {'LATE_PAYMENT':'LATE_PAYMENT_SUM'})




# Last month of each application
# idxmax -> 최대값을 갖고있는 인덱스 반환 -> 제일 마지막에 있는 인덱스 반환함(오름차순일 경우)
last_month_df = pos.groupby('SK_ID_PREV')['MONTHS_BALANCE'].idxmax()




# Most recent applications (last 3)
sort_pos = pos.sort_values(by=['SK_ID_PREV','MONTHS_BALANCE'])
gp = sort_pos.iloc[last_month_df].groupby('SK_ID_CURR').tail(3)
gp_mean = gp.groupby('SK_ID_CURR').mean().reset_index()
pos_agg = pd.merge(pos_agg, gp_mean[['SK_ID_CURR','LATE_PAYMENT_SUM']], on='SK_ID_CURR', how='left')




# Drop some useless categorical features
drop_features = [
    'POS_NAME_CONTRACT_STATUS_Canceled_MEAN', 'POS_NAME_CONTRACT_STATUS_Amortized debt_MEAN',
    'POS_NAME_CONTRACT_STATUS_XNA_MEAN']
pos_agg.drop(drop_features, axis=1, inplace=True)































