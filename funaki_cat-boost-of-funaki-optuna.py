#!/usr/bin/env python
# coding: utf-8

# In[1]:


from kaggle.competitions import nflrush
import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier,Pool
from tqdm import tqdm

env = nflrush.make_env()


# In[2]:


from sklearn.model_selection import train_test_split


# In[3]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import warnings
warnings.simplefilter('ignore')


# In[5]:


df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)


# In[6]:


iter_test = env.iter_test()


# In[7]:


#ボールを持っている人のデータのみ抽出
rusher_df=df[df['NflId']==df['NflIdRusher']]


# In[8]:


def count_position(df,rusher_df):
    new_df=df.groupby(['PlayId','Position']).count()
    position_count=new_df['GameId'].unstack().fillna(0).astype(int)
    rusher_df=rusher_df.merge(position_count, on='PlayId')
    rusher_df=rusher_df.rename(columns={'S_x':'S','S_y':'S_position'})
    return rusher_df


# In[9]:


def preprocess(df):
    #StadiumTypeからおかしなデータを削除
    df=df[(df['StadiumType']!='Cloudy') & (df['StadiumType']!='Bowl')]
    #StadiumTypeの文字列を屋外内で分けてリスト化
    outdoor=['Outdoor', 'Outdoors','Open','Indoor, Open Roof','Outdoor Retr Roof-Open', 'Oudoor', 'Ourdoor','Retr. Roof-Open','Outdor','Retr. Roof - Open', 'Domed, Open', 'Domed, open', 'Outside','Heinz Field']
    indoor=['Indoors', 'RetractableRoof', 'Indoor','Retr. Roof-Closed','Dome', 'Domed, closed','Indoor, Roof Closed', 'Retr. Roof Closed','Closed Dome','Dome, closed','Domed']
    #StadiumTypeがoutdoorの時に１になるようにダミー変数化
    df['stadiumtype']=(df['StadiumType'].isin(outdoor)*1)
    #天候の悪い時だけリスト化
    rain=['Light Rain', 'Showers','Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.','Rain', 'Heavy lake effect snow','Snow', 'Cloudy, Rain','Rain shower','Rainy']
    #天気が悪くない時に１になるようにダミー変数化
    df['weather']=(~df['GameWeather'].isin(rain)*1)
    #身長をフィートからセンチに変換
    df['PlayerHeight']= df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
    #ゲームの経過時間を算出
    df['gameclock']=[ pd.Timedelta(val).total_seconds() for val in df['GameClock']]
    #Orientationを整える
    df.loc[df["Season"]==2017, "Orientation"] = (df.loc[df["Season"]==2017, "Orientation"] -90)%360
    #攻撃の向きを右を正として揃える
    df.loc[df['PlayDirection']=='left','Dir'] = 180 + df['Dir'] - 360
    df.loc[df['PlayDirection']=='left','Orientation'] = 180 + df['Orientation'] - 360
    df.loc[df['PlayDirection']=='left','X'] = 120 - df['X']
    df.loc[df['PlayDirection']=='left','Y'] = 53.3 - df['Y']
    #Orientationをx,y成分に分ける
    df['sin_Ori']=(df['Orientation']*np.pi/180).map(np.sin) 
    df['cos_Ori']=(df['Orientation']*np.pi/180).map(np.cos)
    #Dirをx,y成分に分けて速度をかける
    df['sin_Dir_S']=(df['Dir']*np.pi/180).map(np.sin)*df['S']
    df['cos_Dir_S']=(df['Dir']*np.pi/180).map(np.cos)*df['S']
    
    return df


# In[10]:


'''def add_team_yard(rusher_df):
    #チーム毎(home/away別)の獲得ヤード数の平均を見る
    team_yards_df = rusher_df.groupby(['Team','PossessionTeam']).mean()[['Yards']]
    team_yards_df = team_yards_df.rename(columns={'Yards':'team_yards'})
    #rusherのみのデータにチーム毎の平均獲得ヤード数を加える
    rusher_df = rusher_df.merge(team_yards_df,on='PossessionTeam',how="left")
    return rusher_df,team_yards_df'''


# In[11]:


def add_team_score(rusher_df):
    # 攻撃チームの得点
    rusher_df.loc[rusher_df["Team"]=="home", "rusherTeamScore"] = rusher_df["HomeScoreBeforePlay"]
    rusher_df.loc[rusher_df["Team"]=="away", "rusherTeamScore"] = rusher_df["VisitorScoreBeforePlay"]

    # 守備チームの得点
    rusher_df.loc[rusher_df["Team"]=="home", "defenceTeamScore"] = rusher_df["VisitorScoreBeforePlay"]
    rusher_df.loc[rusher_df["Team"]=="away", "defenceTeamScore"] = rusher_df["HomeScoreBeforePlay"]

    # 得点差
    rusher_df.loc[:, "diffScore"] = rusher_df["rusherTeamScore"] - rusher_df["defenceTeamScore"]
    return rusher_df


# In[12]:


def count_yard_to_touchdown(rusher_df):
    #タッチダウンまで何ヤードあるか
    rusher_df["yardsToTouchdown"] = 100-rusher_df['X']
    rusher_df["yardsToTouchdown"].clip(0,100,inplace=True)
    return rusher_df


# In[13]:


def add_personal_yard(rusher_df):
    # 選手毎の平均獲得ヤード
    rusher_yards = rusher_df[["NflId", "Yards"]].groupby("NflId").mean()[["Yards"]]
    rusher_yards.dropna(inplace=True)
    rusher_yards=rusher_yards.rename(columns={'Yards':'PersonalYard'})
    rusher_df = rusher_df.merge(rusher_yards, on="NflId", how="left")
    return rusher_df,rusher_yards


# In[14]:


def add_average_data(df,rusher_df):
    offence_position = ['WR', 'TE', 'T', 'QB', 'RB', 'G', 'C', 'FB', 'HB',  'OT', 'OG']
    df["offence"] = 0
    df.loc[df["Position"].isin(offence_position), "offence"] = 1
    # 攻撃,守備チーム平均 体重, 身長, S, A（PlayIdがキー）
    offence_av = df.loc[df["offence"]==1, ["PlayerHeight", "PlayerWeight", "S", "A", "PlayId"]].groupby("PlayId").mean()
    defence_av = df.loc[df["offence"]==0, ["PlayerHeight", "PlayerWeight", "S", "A", "PlayId"]].groupby("PlayId").mean()
    offence_av.columns = ['PlayerHeight_offence', 'PlayerWeight_offence', 'S_offence', 'A_offence']
    defence_av.columns = ['PlayerHeight_defence', 'PlayerWeight_defence', 'S_defence', 'A_defence']
    rusher_df = rusher_df.merge(offence_av, on="PlayId", how="left").merge(defence_av, on="PlayId", how="left")
    return rusher_df
    


# In[15]:


def create_datetime(df):
    #play開始時間のデータをdatetime型にする
    df['game_daytime']=pd.to_datetime(df['TimeSnap'],format='%Y-%m-%dT%H:%M:%S.000Z')
    
    return df
    


# In[16]:


def feature(df):
    features=pd.DataFrame(df,columns=[
        #数値型  
        'X', 'Y', 'S', 'A', 'Dis',
       'gameclock', 'Distance','HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'DefendersInTheBox','PlayerHeight',
       'PlayerWeight','Temperature', 'Humidity',
        'stadiumtype', 'weather', 
        'C', 'CB', 'DB','DE', 'DL', 'DT', 'FB', 'FS', 'G', 'HB', 'ILB', 'LB', 'MLB', 'NT', 'OG','OLB', 'OT', 'QB', 'RB', 'S_position', 'SAF', 'SS', 'T', 'TE', 'WR',
        "yardsToTouchdown",
       'PersonalYard',
       #'team_yards',
       #"rusherTeamScore","defenceTeamScore",
        "diffScore",
        'PlayerHeight_offence', 'PlayerWeight_offence', 'S_offence', 'A_offence',
        'PlayerHeight_defence', 'PlayerWeight_defence', 'S_defence', 'A_defence',
        'sin_Dir_S','cos_Dir_S','sin_Ori','cos_Ori',
        'game_daytime',
        # dtypes='object'
        'Team', 'DisplayName', 'PossessionTeam', 'FieldPosition',
       'OffenseFormation', 'Position', 'HomeTeamAbbr', 'VisitorTeamAbbr',
       'Stadium'])
    return features   


# In[17]:


rusher_df=count_position(df,rusher_df)


# In[18]:


df=preprocess(df)


# In[19]:


rusher_df=preprocess(rusher_df)


# In[20]:


#rusher_df,team_yards_df=add_team_yard(rusher_df)


# In[21]:


rusher_df=add_team_score(rusher_df)


# In[22]:


rusher_df=count_yard_to_touchdown(rusher_df)


# In[23]:


rusher_df,rusher_yards=add_personal_yard(rusher_df)


# In[24]:


rusher_df=add_average_data(df,rusher_df)


# In[25]:


rusher_df=create_datetime(rusher_df)


# In[26]:


rusher_df=rusher_df.dropna()


# In[27]:


#rusher_df.columns[rusher_df.dtypes=='object']


# In[28]:


features=feature(rusher_df)


# In[29]:


#train_mean=features.mean(axis=0)


# In[30]:


#train_std=features.std(axis=0)


# In[31]:


'''
def normalize(features):
    X=(features-train_mean)/train_std
    return X
'''


# In[32]:


X=features


# In[33]:


target=pd.Series(rusher_df['Yards'])


# In[34]:


train_X,test_X,train_y,test_y=train_test_split(X,target,test_size=0.2)


# In[35]:


cattegirical_features= np.where(X.dtypes == np.object)[0]


# In[36]:


train_pool=Pool(train_X,train_y,cat_features=cattegirical_features)
test_pool=Pool(test_X,test_y,cat_features=cattegirical_features)
all_pool=Pool(X,target,cat_features=cattegirical_features)


# In[37]:


import optuna


# In[38]:


'''
def objective(trial):
    # パラメータの指定
    params = {
        'iterations' : trial.suggest_int('iterations', 50, 300),                         
        'depth' : trial.suggest_int('depth', 6, 10),                                       
        'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.3),               
        'random_strength' :trial.suggest_loguniform('random_strength', 1, 10),                       
        'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01,100.00 ),
        'l2_leaf_reg' : trial.suggest_loguniform('l2_leaf_reg',0.1,30)
        #'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        #'od_wait' :trial.suggest_int('od_wait', 10, 50)
    }
    
    # 学習
    model = CatBoostClassifier(**params,
                               task_type='GPU',
                               has_time=True #時系列データとして扱う
                              )
    model.fit(train_pool,early_stopping_rounds=10,verbose=0)
    # 予測
    pred_y = model.predict_proba(test_pool)
    
    
    #実測値の累積確率のアレーを作成
    test_y_score=np.array([(i >= test_y)*1 for i in range(-99,100)])

    #予測値の累積確率のアレーを作成
    pred_prob_cdf=pd.DataFrame(pred_y,columns=[ "Yards"+str(i) for i in model.classes_])
    pred_prob_cdf=pd.DataFrame(pred_prob_cdf,columns=[ "Yards"+str(i) for i in range(-99,100)])
    pred_prob_cdf.fillna(0,inplace=True)
    pred_prob_cdf = pred_prob_cdf.cumsum(axis=1)
    pred_prob_cdf[pred_prob_cdf>1] = 1 
    pred_prob=np.array(pred_prob_cdf.values)
    
    #実測値と予測値の誤差で評価
    C=((pred_prob - test_y_score.T)**2).sum().sum()/(199*len(pred_y))
    
    return C
'''


# In[39]:


from optuna.pruners import SuccessiveHalvingPruner


# In[40]:


'''
# optimizeの第一引数に対象のメソッドを指定、n_trialsにプログラムが試行錯誤する回数を指定
# timeoutに試行する時間(秒)を指定
pruner=SuccessiveHalvingPruner(min_resource=100)
study = optuna.create_study(pruner=pruner)
study.optimize(objective, 
               #n_jobs=-1,
               #n_trials=100,
               timeout=600)
'''


# In[41]:


'''
#optunaで見つけた最適なパラメーターを出力
#best_params=study.best_params
study.best_params
'''


# In[42]:


#C=0.01300098884071051
best_params={'iterations': 236,
             'depth': 9,
             'learning_rate': 0.10501998335501668,
             'random_strength': 5.390945350348725,
             'bagging_temperature': 0.10569149527194119,
             'l2_leaf_reg': 2.296406589911536,
             'has_time' : True
            }


# In[43]:


model = CatBoostClassifier(**best_params)


# In[44]:


#モデルにさらに全データを学習
CBC=model.fit(all_pool)


# In[45]:


num_feat_imp=pd.DataFrame(CBC.get_feature_importance,index=X,select_dtypes([int,float]).columns)
num_feat_imp.sort_values(0,ascending=False)


# In[46]:


cat_feat_imp=pd.DataFrame(CBC.get_object_importance,index==X,select_dtypes('object').columns)
cat_feat_imp.sort_values(0,ascending=False)


# In[47]:


train_df=rusher_df.iloc[:0,:]


# In[48]:


for (test_df, sample_prediction_df) in tqdm(iter_test):
    new_df=test_df.groupby(['PlayId','Position']).count()
    position_count=new_df['GameId'].unstack().fillna(0).astype(int)
    rusher_df=test_df[test_df['NflId']==test_df['NflIdRusher']]
    rusher_df=preprocess(rusher_df)
    test_df=preprocess(test_df)
    rusher_df=count_position(rusher_df)
    #rusher_df=rusher_df.merge(rusher_yards,  on="NflId", how="left")
    #rusher_df = rusher_df.merge(team_yards_df,on='PossessionTeam',how="left")
    rusher_df=add_team_score(rusher_df)
    rusher_df=count_yard_to_touchdown(rusher_df)
    rusher_df=add_average_data(test_df,rusher_df)
    rusher_df=create_datetime(rusher_df)
    rusher_df=pd.concat([train_df,rusher_df],sort=False)
    test_X=feature(rusher_df)
    test_X.fillna(0,inplace=True)
    pred_prob=CBC.predict_proba(test_X)
    pred_prob_cdf=pd.DataFrame(pred_prob,columns=[ "Yards"+str(i) for i in CBC.classes_])
    pred_prob_cdf=pd.DataFrame(pred_prob_cdf, columns=[ "Yards"+str(i) for i in range(-99,100)])
    pred_prob_cdf.fillna(0,inplace=True)
    pred_prob_cdf = pred_prob_cdf.cumsum(axis=1)
    pred_prob_cdf[pred_prob_cdf>1]=1
    #pred_prob_cdf.loc[:, :"Yards-6"] = 0
    #pred_prob_cdf.loc[:, "Yards21":] = 1
    sample_prediction_df.iloc[0,:]=pred_prob_cdf.iloc[0,:]
    env.predict(sample_prediction_df)


# In[49]:


sample_prediction_df


# In[50]:


env.write_submission_file()


# In[51]:


import os
print([filename for filename in os.listdir('/kaggle/working') if '.csv' in filename])

