#!/usr/bin/env python
# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.ops import polygonize,unary_union
from shapely.geometry import LineString, MultiPolygon, MultiPoint, Point, Polygon
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
import math
import matplotlib.patches as patches




import os
TRAIN_ABLE_FALSE=True
if TRAIN_ABLE_FALSE:
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import numpy as np
import pandas as pd
import sklearn.metrics as mtr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda,BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from  keras.callbacks import EarlyStopping,ModelCheckpoint
import datetime

TRAIN_OFFLINE = False


pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 150)




from tqdm.auto import tqdm
tqdm.pandas()




def standardise_features(train):
    train['ToLeft'] = train.PlayDirection == "left"
    train['IsBallCarrier'] = train.NflId == train.NflIdRusher

    train.loc[train.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
    train.loc[train.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

    train.loc[train.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
    train.loc[train.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

    train.loc[train.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
    train.loc[train.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

    train.loc[train.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
    train.loc[train.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"

    train['Dir_rad'] = np.mod(90 - train.Dir, 360) * math.pi/180.0
    train['Rusher'] = train['NflIdRusher'] == train['NflId']
    train['TeamOnOffense'] = "home"
    train.loc[train.PossessionTeam != train.HomeTeamAbbr, 'TeamOnOffense'] = "away"
    train['IsOnOffense'] = train.Team == train.TeamOnOffense # Is player on offense?
    train['YardLine_std'] = 100 - train.YardLine
    train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  
              'YardLine_std'
             ] = train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  
              'YardLine']
    train['X_std'] = train.X
    train.loc[train.ToLeft, 'X_std'] = 120 - train.loc[train.ToLeft, 'X'] 
    train['Y_std'] = train.Y
    train.loc[train.ToLeft, 'Y_std'] = 160/3 - train.loc[train.ToLeft, 'Y'] 
    #train['Orientation_std'] = -90 + train.Orientation
    #train.loc[train.ToLeft, 'Orientation_std'] = np.mod(180 + train.loc[train.ToLeft, 'Orientation_std'], 360)
    train['Dir_std'] = train.Dir_rad
    train.loc[train.ToLeft, 'Dir_std'] = np.mod(np.pi + train.loc[train.ToLeft, 'Dir_rad'], 2*np.pi)

    train['Orientation_rad'] = np.mod(train.Orientation, 360) * math.pi/180.0
    train.loc[train.Season >= 2018, 'Orientation_rad'
             ] = np.mod(train.loc[train.Season >= 2018, 'Orientation'] - 90, 360) * math.pi/180.0
    train['Orientation_std'] = train.Orientation_rad
    train.loc[train.ToLeft, 'Orientation_std'] = np.mod(np.pi + train.loc[train.ToLeft, 'Orientation_rad'], 2*np.pi)

    return train



def euclidean_distance(x1,y1,x2,y2):
    x_diff = (x1-x2)**2
    y_diff = (y1-y2)**2

    return np.sqrt(x_diff + y_diff)

def back_direction(orientation):
    if orientation > 180.0:
        return 1
    else:
        return 0




def voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite Voronoi regions in a
    2D diagram to finite regions.
    Source:
    [https://stackoverflow.com/a/20678647/1595060](https://stackoverflow.com/a/20678647/1595060)
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()
    # Construct a map containing all ridges for a
    # given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points,
                                  vor.ridge_vertices):
        all_ridges.setdefault(
            p1, []).append((p2, v1, v2))
        all_ridges.setdefault(
            p2, []).append((p1, v1, v2))
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue
            # Compute the missing endpoint of an
            # infinite ridge
            t = vor.points[p2] -                 vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]].                 mean(axis=0)
            direction = np.sign(
                np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] +                 direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        # Sort region counterclockwise.
        vs = np.asarray([new_vertices[v]
                         for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(
            vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[
            np.argsort(angles)]
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)

def getVornoiAreaRusher(play1):
    def get_dx_dy_row(row):
        t= 5
        dist= t*row.S + 0.5*row.A*(t)**2
        radian_angle = row.Dir_std
        dx = dist * math.cos(radian_angle)
        dy = dist * math.sin(radian_angle)
        return dx, dy
    play1['X_std_n'],play1['Y_std_n'] = zip(*play1.apply(get_dx_dy_row,axis=1))
    play1['X_std_n'] += play1.X_std
    play1['Y_std_n'] += play1.Y_std
    
    play2 = play1[(~play1.Team.isin(play1[play1.Rusher==1].Team) | (play1.NflId.isin(play1[play1.Rusher==1].NflId)))]
    points=play2[['X_std','Y_std']].values
    points = np.array(points)
    
    points_n=play2[['X_std_n','Y_std_n']].values
    points_n = np.array(points_n)
    
    vor = Voronoi(points)
    n_points = points.shape[0]
    offense = play2.IsOnOffense.values
    regions,vertices=voronoi_finite_polygons_2d(vor)
    
    for r in range(n_points):
        region = regions[r]
        sp = points[r]
        npt = points_n[r]
        if offense[r]:
            int_pt_x=0
            int_pt_y=0
            
            pg = vertices[region]
            #print(pg)
            mp = MultiPoint([Point(i) for i in pg])
            pgn = Polygon(mp)
            
            lp1 = [sp[0],sp[1]]
            lp2 = [npt[0],npt[1]]
            
            line = [lp1,lp2]
            shapely_line = LineString(line)
            try:
                intersection_line = list(pgn.intersection(shapely_line).coords)
                pt_to_consider = 0
                if intersection_line[0][0] > sp[0]:
                    pt_to_consider = 0
                else:
                    pt_to_consider = 1


                int_pt_x,int_pt_y= intersection_line[pt_to_consider][0],intersection_line[pt_to_consider][1]
            except :
                print('Exception',play1.PlayId.values[0],pgn.intersection(shapely_line))
                int_pt_x,int_pt_y= sp[0],sp[1]
            
            
          
            less_xix = np.where(pg[:,0]>=sp[0])[0]
            
            mp = MultiPoint([Point(i) for i in pg])
            pgn = Polygon(mp)
            
            lp1 = [sp[0]-.1,sp[1]-100]
            lp2 = [sp[0]-.1,sp[1]+100]
            
            line = [lp1,lp2]
            shapely_line = LineString(line)
            
            intersection_line = list(pgn.intersection(shapely_line).coords)
            pg = pg[less_xix]
            pg = np.vstack((intersection_line[0],pg,intersection_line[1]))
            mp = MultiPoint([Point(i) for i in pg])
            pgn = Polygon(mp)
            
            #plt.fill(*zip(*pg), c='b', alpha=0.25)
            #print(playid,pgn.area,play1.Yards.values[0])
            #plt.show()
            
            return pgn.area,int_pt_x,int_pt_y

def getDistanceForVAreaEnd(play1):
    def get_dx_dy_row(row):
        t= 5
        dist= t*row.S + 0.5*row.A*(t)**2
        radian_angle = row.Dir_std
        dx = dist * math.cos(radian_angle)
        dy = dist * math.sin(radian_angle)
        return dx, dy
    play1['X_std_n'],play1['Y_std_n'] = zip(*play1.apply(get_dx_dy_row,axis=1))
    play1['X_std_n'] += play1.X_std
    play1['Y_std_n'] += play1.Y_std
    
    
    play2 = play1[(~play1.Team.isin(play1[play1.Rusher==1].Team) | (play1.NflId.isin(play1[play1.Rusher==1].NflId)))]
    points=play2[['X_std','Y_std']].values
    points = np.array(points)
    
    n_points=play2[['X_std_n','Y_std_n']].values
    n_points = np.array(points)
    
    vor = Voronoi(points)
    n_points = points.shape[0]
    offense = play2.IsOnOffense.values
    regions,vertices=voronoi_finite_polygons_2d(vor)
    
    
    defense = play1[~play1.Team.isin(play1[play1.Rusher==1].Team)][['GameId','PlayId','X_std','Y_std']]
    
    for r in range(n_points):
        region = regions[r]
        sp = points[r]
        npt = n_points[r]
        if offense[r]:
            
            pg = vertices[region]
            #print(pg)
          
            less_xix = np.where(pg[:,0]>=sp[0])[0]
            
            mp = MultiPoint([Point(i) for i in pg])
            pgn = Polygon(mp)
            
            lp1 = [sp[0],sp[1]]
            lp2 = [npt[0],npt[1]]
            
            line = [lp1,lp2]
            shapely_line = LineString(line)
            
            intersection_line = list(pgn.intersection(shapely_line).coords)
            pt_to_consider = 0
            if intersection_line[0][0] >= sp[0][0]:
                pt_to_consider = 0
            else:
                pt_to_consider = 1
            
            
            plt.fill(*zip(*pg), c='b', alpha=0.25)
            print(playid,pgn.area,play1.Yards.values[0])
            plt.show()
            return intersection_line[pt_to_consider]
            
    
def getVornoiAreaRusher_after_half_second(play1):
    def get_dx_dy_row(row):
        t= 0.5
        dist= t*row.S + 0.5*row.A*(t)**2
        radian_angle = row.Dir_std
        dx = dist * math.cos(radian_angle)
        dy = dist * math.sin(radian_angle)
        return dx, dy
    play1['X_std_n'],play1['Y_std_n'] = zip(*play1.apply(get_dx_dy_row,axis=1))
    play1['X_std_n'] += play1.X_std
    play1['Y_std_n'] += play1.Y_std
    
    play2 = play1[(~play1.Team.isin(play1[play1.Rusher==1].Team) | (play1.NflId.isin(play1[play1.Rusher==1].NflId)))]
    
    points=play2[['X_std_n','Y_std_n']].values
    points = np.array(points)
    vor = Voronoi(points)
    n_points = points.shape[0]
    offense = play2.IsOnOffense.values
    
    regions,vertices=voronoi_finite_polygons_2d(vor)
    
    for r in range(n_points):
        region = regions[r]
        sp = points[r]
        if offense[r]:
            
            pg = vertices[region]
            #print(pg)
          
            less_xix = np.where(pg[:,0]>=sp[0])[0]
            
            mp = MultiPoint([Point(i) for i in pg])
            pgn = Polygon(mp)
            
            lp1 = [sp[0],sp[1]-100]
            lp2 = [sp[0],sp[1]+100]
            
            line = [lp1,lp2]
            shapely_line = LineString(line)
            
            intersection_line = list(pgn.intersection(shapely_line).coords)
            pg = pg[less_xix]
            pg = np.vstack((intersection_line[0],pg,intersection_line[1]))
            mp = MultiPoint([Point(i) for i in pg])
            pgn = Polygon(mp)
            return pgn.area
            
            #plt.fill(*zip(*pg), c='b', alpha=0.25)
            #print(playid,pgn.area,play1.Yards.values[0])
            #plt.show()
            
            #playIds.append(playid)
            #rusher_cell_area.append(pgn.area)
            
def getVornoiAreaRusher_after_1_second(play1):
    def get_dx_dy_row1(row):
        t= 1
        dist= t*row.S + 0.5*row.A*(t)**2
        radian_angle = row.Dir_std
        dx = dist * math.cos(radian_angle)
        dy = dist * math.sin(radian_angle)
        return dx, dy
    play1['X_std_n'],play1['Y_std_n'] = zip(*play1.apply(get_dx_dy_row1,axis=1))
    play1['X_std_n'] += play1.X_std
    play1['Y_std_n'] += play1.Y_std
    
    play2 = play1[(~play1.Team.isin(play1[play1.Rusher==1].Team) | (play1.NflId.isin(play1[play1.Rusher==1].NflId)))]
    
    points=play2[['X_std_n','Y_std_n']].values
    points = np.array(points)
    vor = Voronoi(points)
    n_points = points.shape[0]
    offense = play2.IsOnOffense.values
    
    regions,vertices=voronoi_finite_polygons_2d(vor)
    
    for r in range(n_points):
        region = regions[r]
        sp = points[r]
        if offense[r]:
            
            pg = vertices[region]
            #print(pg)
          
            less_xix = np.where(pg[:,0]>=sp[0])[0]
            
            mp = MultiPoint([Point(i) for i in pg])
            pgn = Polygon(mp)
            
            lp1 = [sp[0],sp[1]-100]
            lp2 = [sp[0],sp[1]+100]
            
            line = [lp1,lp2]
            shapely_line = LineString(line)
            
            intersection_line = list(pgn.intersection(shapely_line).coords)
            pg = pg[less_xix]
            pg = np.vstack((intersection_line[0],pg,intersection_line[1]))
            mp = MultiPoint([Point(i) for i in pg])
            pgn = Polygon(mp)
            return pgn.area
            #plt.fill(*zip(*pg), c='b', alpha=0.25)
            #print(playid,pgn.area,play1.Yards.values[0])
            #plt.show()
            
            #playIds.append(playid)
            #rusher_cell_area.append(pgn.area)




#Features from other kernel
def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
    return ans

def strtofloat(x):
    try:
        return float(x)
    except:
        return -1

def map_weather(txt):
    ans = 1
    if pd.isna(txt):
        return 0
    if 'partly' in txt:
        ans*=0.5
    if 'climate controlled' in txt or 'indoor' in txt:
        return ans*3
    if 'sunny' in txt or 'sun' in txt:
        return ans*2
    if 'clear' in txt:
        return ans
    if 'cloudy' in txt:
        return -ans
    if 'rain' in txt or 'rainy' in txt:
        return -2*ans
    if 'snow' in txt:
        return -3*ans
    return 0

def OffensePersonnelSplit(x):
    dic = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0, 'QB' : 0, 'RB' : 0, 'TE' : 0, 'WR' : 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        dic[xxs[-1]] = int(xxs[-2])
    return dic

def DefensePersonnelSplit(x):
    dic = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        dic[xxs[-1]] = int(xxs[-2])
    return dic

def orientation_to_cat(x):
    x = np.clip(x, 0, 360 - 1)
    try:
        return str(int(x/15))
    except:
        return "nan"

def defense_features_next_second(df, next_duration=1):
    
    def get_dx_dy_row(row):
        t= next_duration
        dist= t*row.S + 0.5*row.A*(t)**2
        radian_angle = row.Dir_std
        dx = dist * math.cos(radian_angle)
        dy = dist * math.sin(radian_angle)
        return dx, dy
    
    df['X_std_n'],df['Y_std_n'] = zip(*df.apply(get_dx_dy_row,axis=1))
    df['X_std_n'] += df.X_std
    df['Y_std_n'] += df.Y_std

    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X_std_n','Y_std_n']]
    rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']


    defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
    defense = defense[defense['Team'] != defense['RusherTeam']][['GameId','PlayId','X_std_n','Y_std_n','RusherX','RusherY']]
    defense['def_dist_to_back'] = defense[['X_std_n','Y_std_n','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
    defense['change_x'] = defense.RusherX - defense.X_std_n
    defense['change_y'] = defense.RusherY - defense.Y_std_n
    defense = defense.groupby(['GameId','PlayId'])                     .agg({'def_dist_to_back':['min','max','mean','std'],
                          'change_x':['min','max','mean','std']})\
                     .reset_index()

    defense.columns = ['GameId','PlayId',f'def_min_dist_next_{next_duration}',
                       f'def_max_dist_next_{next_duration}',
                       f'def_mean_dist_{next_duration}',f'def_std_dist_{next_duration}',
                       f'def_min_chx_{next_duration}',f'def_max_chx_{next_duration}',
                       f'def_mean_chx_{next_duration}',f'def_std_chx_{next_duration}']

    return defense


def back_features(df):
    carriers = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','NflIdRusher',
                                                     'X_std','Y_std','Orientation_std','Dir_std','YardLine_std']]
    carriers['back_from_scrimmage'] = carriers['YardLine_std'] - carriers['X_std']
    carriers['back_oriented_down_field'] = carriers['Orientation_std'].apply(lambda x: back_direction(x))
    carriers['back_moving_down_field'] = carriers['Dir_std'].apply(lambda x: back_direction(x))
    carriers = carriers.rename(columns={'X_std':'back_X',
                                        'Y_std':'back_Y'})
    carriers = carriers[['GameId','PlayId','NflIdRusher','back_X','back_Y',
                         'back_from_scrimmage','back_oriented_down_field',
                         'back_moving_down_field']]

    return carriers

def features_relative_to_back(df, carriers):
    player_distance = df[['GameId','PlayId','NflId','X_std','Y_std']]
    player_distance = pd.merge(player_distance, carriers, on=['GameId','PlayId'], how='inner')
    player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
    player_distance['dist_to_back'] = player_distance[['X_std','Y_std','back_X','back_Y']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

    player_distance = player_distance.groupby(['GameId','PlayId','back_from_scrimmage',
                                               'back_oriented_down_field','back_moving_down_field'])\
                                     .agg({'dist_to_back':['min','max','mean','std']})\
                                     .reset_index()
    player_distance.columns = ['GameId','PlayId','back_from_scrimmage',
                               'back_oriented_down_field','back_moving_down_field',
                               'min_dist','max_dist','mean_dist','std_dist']

    return player_distance

def defense_features(df):
    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X_std','Y_std']]
    rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

    defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
    defense = defense[defense['Team'] != defense['RusherTeam']][['GameId','PlayId','X_std','Y_std','RusherX','RusherY']]
    defense['def_dist_to_back'] = defense[['X_std','Y_std','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
    defense['change_x'] = defense.RusherX - defense.X_std
    
    defense = defense.groupby(['GameId','PlayId'])                     .agg({'def_dist_to_back':['min','max','mean','std'],
                          'change_x':['min','max','mean','std']})\
                     .reset_index()
    defense.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_std_dist',
                      'def_min_chx','def_max_chx','def_mean_chx','def_std_chx',]

    return defense

def offense_features(df):
    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X_std','Y_std']]
    rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

    offense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
    #offense = offense[(offense['Team'] == offense['RusherTeam'])&(df['NflId'] != df['NflIdRusher'])][['GameId','PlayId','X_std','Y_std','RusherX','RusherY']]
    offense = offense[(offense.IsOnOffense)&(~offense.Rusher)][['GameId','PlayId','X_std','Y_std','RusherX','RusherY']]
    offense['def_dist_to_back'] = offense[['X_std','Y_std','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
    offense['change_x'] = offense.RusherX - offense.X_std
    offense = offense.groupby(['GameId','PlayId'])                     .agg({'def_dist_to_back':['min','max','mean','std'],
                          'change_x':['min','max','mean','std']})\
                     .reset_index()
    offense.columns = ['GameId','PlayId','off_min_dist','off_max_dist','off_mean_dist','off_std_dist',
                      'off_min_chx','off_max_chx','off_mean_chx','off_std_chx']

    return offense

def static_features(df):

    add_new_feas = []

    ## Height
    df['PlayerHeight_dense'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

    add_new_feas.append('PlayerHeight_dense')

    ## Time
    
    #df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    #df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    #df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
    #df['PlayerBirthDate'] =df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
    
    ## Age
    #seconds_in_year = 60*60*24*365.25
    #df['PlayerAge'] = df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
    #add_new_feas.append('PlayerAge')

    ## WindSpeed
    df['WindSpeed_ob'] = df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
    df['WindSpeed_ob'] = df['WindSpeed_ob'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
    df['WindSpeed_ob'] = df['WindSpeed_ob'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
    df['WindSpeed_dense'] = df['WindSpeed_ob'].apply(strtofloat)
    add_new_feas.append('WindSpeed_dense')

    ## Weather
    df['GameWeather_process'] = df['GameWeather'].str.lower()
    df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: "indoor" if not pd.isna(x) and "indoor" in x else x)
    df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
    df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
    df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
    df['GameWeather_dense'] = df['GameWeather_process'].apply(map_weather)
    add_new_feas.append('GameWeather_dense')
#         ## Rusher
#         train['IsRusher'] = (train['NflId'] == train['NflIdRusher'])
#         train['IsRusher_ob'] = (train['NflId'] == train['NflIdRusher']).astype("object")
#         temp = train[train["IsRusher"]][["Team", "PlayId"]].rename(columns={"Team":"RusherTeam"})
#         train = train.merge(temp, on = "PlayId")
#         train["IsRusherTeam"] = train["Team"] == train["RusherTeam"]

    ## dense -> categorical
#         train["Quarter_ob"] = train["Quarter"].astype("object")
#         train["Down_ob"] = train["Down"].astype("object")
#         train["JerseyNumber_ob"] = train["JerseyNumber"].astype("object")
#         train["YardLine_ob"] = train["YardLine"].astype("object")
    # train["DefendersInTheBox_ob"] = train["DefendersInTheBox"].astype("object")
    # train["Week_ob"] = train["Week"].astype("object")
    # train["TimeDelta_ob"] = train["TimeDelta"].astype("object")


    ## Orientation and Dir
    df["Orientation_ob"] = df["Orientation_std"].apply(lambda x : orientation_to_cat(x)).astype("object")
    df["Dir_ob"] = df["Dir_std"].apply(lambda x : orientation_to_cat(x)).astype("object")

    df["Orientation_sin"] = df["Orientation_std"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
    df["Orientation_cos"] = df["Orientation_std"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
    df["Dir_sin"] = df["Dir_std"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
    df["Dir_cos"] = df["Dir_std"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
    add_new_feas.append("Dir_sin")
    add_new_feas.append("Dir_cos")

    ## diff Score
    df["diffScoreBeforePlay"] = df["HomeScoreBeforePlay"] - df["VisitorScoreBeforePlay"]
    add_new_feas.append("diffScoreBeforePlay")



    static_features = df[df['NflId'] == df['NflIdRusher']][add_new_feas+['GameId','PlayId','X_std','Y_std',
                                                                         'S','A','Dis',
                                                                         'Orientation_std','Dir_std','Dir_rad',
                                                                         'YardLine_std','Quarter','Down','Distance',
                                                                         'DefendersInTheBox']].drop_duplicates()
#         static_features['DefendersInTheBox'] = static_features['DefendersInTheBox'].fillna(np.mean(static_features['DefendersInTheBox']))
    static_features.fillna(-999,inplace=True)
#         for i in add_new_feas:
#             static_features[i] = static_features[i].fillna(np.mean(static_features[i]))


    return static_features


def combine_features(relative_to_back, defense, static, offense, deploy=False):
    df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
    df = pd.merge(df,offense,on=['GameId','PlayId'],how='inner')
    df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')

    if not deploy:
        df = pd.merge(df, outcomes, on=['GameId','PlayId'], how='inner')

    return df

def getFarthestPointDefenseFeatures(df,farthest_points_df):
    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team']]
    rusher = rusher.merge(farthest_points_df,on='PlayId',how='left')
    rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

    defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
    defense = defense[defense['Team'] != defense['RusherTeam']][['GameId','PlayId','X_std','Y_std','RusherX','RusherY']]
    defense['def_dist_to_back'] = defense[['X_std','Y_std','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

    defense = defense.groupby(['GameId','PlayId'])                     .agg({'def_dist_to_back':['min','max','mean','std']})                     .reset_index()
    defense.columns = ['GameId','PlayId','def_min_dist_far','def_max_dist_far','def_mean_dist_far','def_std_dist_far']
    return defense
'''
def create_features(df, deploy=False):
    df = standardise_features(df)
    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    static_feats = static_features(df)
    basetable = combine_features(rel_back, def_feats, static_feats, deploy=deploy)
    
    df['SpeedForward'] = df['S']*np.cos(df['Dir_std'])

    sped_fw = df[df.Rusher][['PlayId','SpeedForward']].copy()
    basetable = basetable.merge(sped_fw,left_on=['PlayId'],right_on=['PlayId'],how='left')
    
    for dur in [0.5,1,2,2.5,3]:
        def_f_n = defense_features_next_second(df,next_duration=dur)
        basetable = basetable.merge(def_f_n,on=['GameId','PlayId'],how='left')
    
    #Voronoi area Rusher
    vor_area_ruhser_arr = []
    vor_area_ruhser_after_half_second_arr = []
    vor_area_ruhser_after_1_second_arr = []
    
    playids = []
    farthest_points_x = []
    farthest_points_y = []

    #for playid in tqdm(df.PlayId.unique()): #[20170907000395 ]: #dominance_df.PlayId.unique():
    for playid in df.PlayId.unique(): 
    #show_voronoi_rusher(playid,n=False)
        play1 =df[(df.PlayId ==playid )].copy()
        try:
            vor_area_ruhser,fpx,fpy = getVornoiAreaRusher(play1)
            
        except:
            vor_area_ruhser=0
            fpx=play1[play1.Rusher].X_std.values[0]
            fpy=play1[play1.Rusher].Y_std.values[0]
            
        try:
            vor_area_ruhser_after_half_second = getVornoiAreaRusher_after_half_second(play1)
        except:
            vor_area_ruhser_after_half_second=0
            
        try:
            vor_area_ruhser_after_1_second = getVornoiAreaRusher_after_1_second(play1)
        except:
            vor_area_ruhser_after_1_second=0
            
        vor_area_ruhser_arr.append(vor_area_ruhser)
        vor_area_ruhser_after_half_second_arr.append(vor_area_ruhser_after_half_second)
        vor_area_ruhser_after_1_second_arr.append(vor_area_ruhser_after_1_second)
        farthest_points_x.append(fpx)
        farthest_points_y.append(fpy)
        playids.append(playid)
        
    vor_df = pd.DataFrame({'PlayId':playids, 'vor_area_ruhser':vor_area_ruhser_arr, 
                           'vor_area_ruhser_after_half_second':vor_area_ruhser_after_half_second_arr,
                           'vor_area_ruhser_after_1_second':vor_area_ruhser_after_1_second_arr,
                           })
    basetable = basetable.merge(vor_df,on='PlayId',how='left')
    farthest_points_df = pd.DataFrame({'PlayId':playids,'farthest_point_x':farthest_points_x,
                              'farthest_point_y':farthest_points_y
                            })
    
    defensedf = getFarthestPointDefenseFeatures(df,farthest_points_df)
    basetable = basetable.merge(defensedf,on='PlayId',how='left')

    basetable = basetable.fillna(-199)
    
    return basetable
'''
print()




def getFeaturesAfterRemovingNearestDefense(play1,th1=0.5,th2=0.5):
    
    def get_dx_dy_row(row):
        t= th1
        dist= t*row.S + 0.5*row.A*(t)**2
        radian_angle = row.Dir_std
        dx = dist * math.cos(radian_angle)
        dy = dist * math.sin(radian_angle)
        return dx, dy
    play1['X_std_n'],play1['Y_std_n'] = zip(*play1.apply(get_dx_dy_row,axis=1))
    play1['X_std_n'] += play1.X_std
    play1['Y_std_n'] += play1.Y_std
    
    play1 =play1[['PlayId','NflId','NflIdRusher','S','A','X_std','Y_std',
                                                   'IsOnOffense','Rusher',
                                                   'X_std_n','Y_std_n']]
    #cp between defence and offense

    onDefense_play1 = play1[~play1.IsOnOffense].copy()
    #onOffense_play1 = play1[(play1.IsOnOffense) & (~play1.Rusher)]
    onOffense_play1 = play1[(play1.IsOnOffense)].copy()
    
    rusher_id = onOffense_play1[onOffense_play1.Rusher].NflId.values[0]

    del onDefense_play1['IsOnOffense'],onDefense_play1['NflIdRusher']
    del onOffense_play1['IsOnOffense'],onOffense_play1['NflIdRusher']
    
    def_odd_cp = onOffense_play1.merge(onDefense_play1,on='PlayId')
    
    #rint(def_odd_cp.NflId_x.unique(),rusher_id)
    def_odd_cp['dist_to_defense'] = def_odd_cp[['X_std_n_x','Y_std_n_x','X_std_n_y','Y_std_n_y']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
    #remove nearest defense players
    rusher_def_cp = def_odd_cp[def_odd_cp.NflId_x== rusher_id]
    def_odd_cp    = def_odd_cp[def_odd_cp.NflId_x!= rusher_id]
    
    
    to_rem_def_players = def_odd_cp[def_odd_cp.dist_to_defense<=th2].NflId_y.values
    
    play2 = play1[(play1.Rusher) | (~play1.IsOnOffense)] 
    play2 = play2[~play2.NflId.isin(to_rem_def_players)] #this is for vornoi

    play3 = play2[~play1.Rusher].copy()
    #this if for distance calcualtion and time calculation
    rusher_def_cp = rusher_def_cp[~rusher_def_cp.NflId_y.isin(to_rem_def_players)] 
    
    #print(to_rem_def_players)
    
    points=play2[['X_std_n','Y_std_n']].values
    rusher = onOffense_play1[onOffense_play1.Rusher]
    
    play3['RusherX'] = rusher.X_std.values[0]
    play3['RusherY'] = rusher.Y_std.values[0]
    
    play3['ls1_theta'] = np.arctan(((play3.Y_std_n - play3.Y_std)/(play3.X_std_n - play3.X_std)))
    play3['ls2_theta'] = np.arctan(((play3.Y_std - play3.RusherY)/(play3.X_std - play3.RusherX)))
    
    play3['bc_dir_target_end_zone'] = np.abs(play3['ls1_theta']-play3['ls2_theta'])
    #print(play2['bc_dir_target_end_zone'])
            
    #Calculate Distance statistics on remianing defence
    bc_dir_target_end_zone_mean = play3.bc_dir_target_end_zone.mean()
    bc_dir_target_end_zone_min = play3.bc_dir_target_end_zone.min()
    bc_dir_target_end_zone_max = play3.bc_dir_target_end_zone.max()
    bc_dir_target_end_zone_std = play3.bc_dir_target_end_zone.std()
   
    
    
    points = np.array(points)
    try:
        vor = Voronoi(points)
    except :
        print('QhullError:',play1.PlayId.values[0])
        return -1
    #voronoi_plot_2d(vor)
    
    n_points = points.shape[0]
    offense = play2.IsOnOffense.values
    
    #Calculate defens
    
    regions,vertices=voronoi_finite_polygons_2d(vor)
    
    for r in range(n_points):
        region = regions[r]
        sp = points[r]
        if offense[r]:
            
            pg = vertices[region]
            #print(pg)
          
            less_xix = np.where(pg[:,0]>=sp[0])[0]
            
            mp = MultiPoint([Point(i) for i in pg])
            pgn = Polygon(mp)
            
            lp1 = [sp[0],sp[1]-100]
            lp2 = [sp[0],sp[1]+100]
            
            line = [lp1,lp2]
            shapely_line = LineString(line)
            
            intersection_line = list(pgn.intersection(shapely_line).coords)
            pg = pg[less_xix]
            pg = np.vstack((intersection_line[0],pg,intersection_line[1]))
            mp = MultiPoint([Point(i) for i in pg])
            pgn = Polygon(mp)
            
            return pgn.area,rusher_def_cp.dist_to_defense.mean(),                    rusher_def_cp.dist_to_defense.min(),rusher_def_cp.dist_to_defense.max(),                    rusher_def_cp.dist_to_defense.std(), bc_dir_target_end_zone_mean,                    bc_dir_target_end_zone_min,bc_dir_target_end_zone_max,bc_dir_target_end_zone_std
            
def getMoreFeatures(basetable,df):
    basetable['SpeedForward'] = basetable.S*np.cos(basetable.Dir_std)
    basetable['hA'] = basetable.A*np.cos(basetable.Dir_std)
    basetable['hD'] = basetable.SpeedForward + 0.5*basetable.hA
    basetable['Distance'] = basetable.X_std-90
    basetable['dis_forward'] = basetable.Dis*np.cos(basetable.Dir_std)
    df_def = df[(~df.IsOnOffense)].copy()
    rusher_df = df[df.Rusher][['PlayId','X_std','Y_std','S']].copy()
    rusher_df.columns = ['PlayId','RusherX','RusherY','RusherS']
    df_def =df_def.merge(rusher_df,on='PlayId',how='left')
    df_def['rusherDistance'] = np.sqrt((df_def.X_std - df_def.Y_std)**2 + (df_def.X_std - df_def.RusherX)**2)
    df_def['time_tackle'] = df_def['rusherDistance']/df_def.S
    time_tackle_df = df_def.groupby(['PlayId'])['time_tackle'].min().reset_index()
    basetable = basetable.merge(time_tackle_df,on='PlayId',how='left') 
    basetable=basetable.fillna(-1)
    return basetable

def create_features(df, deploy=False):
    df = standardise_features(df)
    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    off_feats = offense_features(df)
    static_feats = static_features(df)
    basetable = combine_features(rel_back, def_feats, static_feats,off_feats, deploy=deploy)
    for dur in [0.5,1,2,2.5,3,4]:
        def_f_n = defense_features_next_second(df,next_duration=dur)
        basetable = basetable.merge(def_f_n,on=['GameId','PlayId'],how='left')
    
    #Voronoi area Rusher
    vor_area_ruhser_arr = []
    vor_area_ruhser_after_half_second_arr = []
    vor_area_ruhser_after_1_second_arr = []
    
    playids = []
    farthest_points_x = []
    farthest_points_y = []
    
    dist_to_defense_area=[]
    dist_to_defense_min=[]
    dist_to_defense_max=[]
    dist_to_defense_mean=[]
    dist_to_defense_std=[]
    bc_dir_target_end_zone_mean= []
    bc_dir_target_end_zone_min= []
    bc_dir_target_end_zone_max= []
    bc_dir_target_end_zone_std= []
    def_area = []
    
    playcnt=0
    for playid in df.PlayId.unique(): #[20170907000395 ]: #dominance_df.PlayId.unique():
        playcnt+=1
        if playcnt%1000==0:
            print('Porcessed Plays',playcnt)
        play1 =df[(df.PlayId ==playid )].copy()
        try:
            vor_area_ruhser,fpx,fpy = getVornoiAreaRusher(play1)
            
        except:
            vor_area_ruhser=0
            fpx=X_std
            fpy=Y_std
            
        try:
            vor_area_ruhser_after_half_second = getVornoiAreaRusher_after_half_second(play1)
        except:
            vor_area_ruhser_after_half_second=0
            
        try:
            vor_area_ruhser_after_1_second = getVornoiAreaRusher_after_1_second(play1)
        except:
            vor_area_ruhser_after_1_second=0
            
        try:
            a1,a2,a3,a4,a5,a6,a7,a8,a9 = getFeaturesAfterRemovingNearestDefense(play1,th1=1,th2=0.4)
        except:
            a1=0
            a2=0
            a3=0
            a4=0
            a5=0
            a6=0
            a7=0
            a8=0
            a9=0
        dist_to_defense_area.append(a1)
        dist_to_defense_mean.append(a2)
        dist_to_defense_min.append(a3)
        dist_to_defense_max.append(a4)
        dist_to_defense_std.append(a5)
        bc_dir_target_end_zone_mean.append(a6)
        bc_dir_target_end_zone_min.append(a7)
        bc_dir_target_end_zone_max.append(a8)
        bc_dir_target_end_zone_std.append(a9)
            
        vor_area_ruhser_arr.append(vor_area_ruhser)
        vor_area_ruhser_after_half_second_arr.append(vor_area_ruhser_after_half_second)
        vor_area_ruhser_after_1_second_arr.append(vor_area_ruhser_after_1_second)
        farthest_points_x.append(fpx)
        farthest_points_y.append(fpy)
        playids.append(playid)
        
        '''
        play2 = play1[(play1.Rusher)|((~play1.IsOnOffense)&(~play1.Position.isin(['MLB','ILB','DT','DE'])))].copy()
        try:
            a,x,y=getVornoiAreaRusher(play2)
        except :
            a=0
        def_area.append(a)
        '''
        
    vor_df = pd.DataFrame({'PlayId':playids, 'vor_area_ruhser':vor_area_ruhser_arr, 
                           'vor_area_ruhser_after_half_second':vor_area_ruhser_after_half_second_arr,
                           'vor_area_ruhser_after_1_second':vor_area_ruhser_after_1_second_arr,
                           'dist_to_defense_area':dist_to_defense_area,
                            'dist_to_defense_mean':dist_to_defense_mean,
                            'dist_to_defense_min':dist_to_defense_min,
                            'dist_to_defense_max':dist_to_defense_max,
                            'dist_to_defense_std':dist_to_defense_std,
                            'bc_dir_target_end_zone_mean':bc_dir_target_end_zone_mean,
                            'bc_dir_target_end_zone_min':bc_dir_target_end_zone_min,
                            'bc_dir_target_end_zone_max':bc_dir_target_end_zone_max,
                            'bc_dir_target_end_zone_std':bc_dir_target_end_zone_std
                           })
    basetable = basetable.merge(vor_df,on='PlayId',how='left')
    farthest_points_df = pd.DataFrame({'PlayId':playids,'farthest_point_x':farthest_points_x,
                              'farthest_point_y':farthest_points_y
                            })
    
    defensedf = getFarthestPointDefenseFeatures(df,farthest_points_df)
    basetable = basetable.merge(defensedf,on='PlayId',how='left')

    
    basetable = getMoreFeatures(basetable ,df)
    basetable = basetable.fillna(-199)
    
    return basetable




## Create top3 defences near the rusher, distnace
## Determine nearest offense near top3 defense

## Calculate Voronoi area of the rusher, excuding back
## Calculate Voronoi area of the rusher removing blocked defenders




if TRAIN_OFFLINE:
    train = pd.read_csv('./input/train.csv', dtype={'WindSpeed': 'object'})
else:
    train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})




train.loc[train['Season'] == 2017, 'S'] = (train['S'][train['Season'] == 2017] - 2.4355) / 1.2930 * 1.4551 + 2.7570




#trainH =train.head(11000)
#trainT =train.tail(11000)

#train = pd.concat([trainH,trainT]).reset_index(drop=True)




print(train.columns,train.shape)




outcomes = train[['GameId','PlayId','Yards']].drop_duplicates()




get_ipython().run_line_magic('time', 'train_basetable = create_features(train, False)')




print(train_basetable.corr()['Yards'])




wt= train[train.Rusher][['PlayId','PlayerWeight']].copy()
train_basetable = train_basetable.merge(wt,left_on=['PlayId'],right_on=['PlayId'],how='left')
train_basetable = train_basetable.fillna(-99)
train_basetable['AW'] = train_basetable.A / train_basetable.PlayerWeight




train_basetable.fillna(-199,inplace=True)




X = train_basetable.copy()
yards = X.Yards

y = np.zeros((yards.shape[0], 199))
for idx, target in enumerate(list(yards)):
    y[idx][99 + target] = 1

X.drop(['GameId_x','GameId_y'], axis=1, inplace=True)




crm = ['min_dist', 
       'max_dist', 'mean_dist', 'std_dist', 'def_max_dist', 'def_std_dist',
       'def_max_chx', 'def_mean_chx',  'WindSpeed_dense', 'GameWeather_dense', 'diffScoreBeforePlay',
       'Y_std', 'Orientation_std', 'Dir_std', 'Quarter', 'Down', 'def_max_dist_next_0.5',
       'def_std_dist_0.5', 'def_mean_chx_0.5', 'def_max_dist_next_1', 'def_std_dist_1',
       'def_mean_chx_1', 'def_max_dist_next_2', 'def_std_dist_2', 'def_min_chx_2',
       'def_min_dist_next_2.5', 'def_max_dist_next_2.5', 'def_std_dist_2.5',
       'def_min_chx_2.5', 'def_std_chx_2.5', 'def_min_dist_next_3', 'def_max_dist_next_3',
       'def_std_chx_3', 'vor_area_ruhser', 'def_max_dist_far', 'def_mean_dist_far', 
       'def_std_dist_far']
#X.drop(crm, axis=1, inplace=True)




tormcols=['back_oriented_down_field', 'back_moving_down_field', 'std_dist', 'def_std_dist',
 'def_min_chx', 'off_max_dist', 'PlayerHeight_dense', 'Dir_sin', 'Dir_cos', 
 'diffScoreBeforePlay', 'X_std', 'Y_std', 'Distance', 'DefendersInTheBox', 
 'def_max_dist_next_0.5', 'def_std_chx_1', 'def_min_chx_2', 'def_std_chx_2', 
 'def_mean_dist_2.5', 'def_min_chx_2.5', 'def_max_chx_2.5', 'def_min_dist_next_3', 
 'def_mean_dist_3', 'def_min_chx_3', 'def_max_chx_3', 'def_std_chx_3', 
 'def_min_dist_next_4', 'def_std_dist_4', 'def_std_chx_4', #'vor_area_ruhser', 
 'bc_dir_target_end_zone_max', 'bc_dir_target_end_zone_std', #'def_poly_area', 
 'def_mean_dist_far', 'def_std_dist_far',  'max_dist', 'mean_dist', 
 'off_min_chx', 'off_max_chx', 'GameWeather_dense', 'Dir_std', 'Down', 
 'def_min_chx_0.5', 'def_std_dist_2.5', 'def_max_dist_next_4', 
 'def_min_chx_4', 'bc_dir_target_end_zone_min']
X=X.drop(tormcols,axis=1)




for c in X.columns:
    X[c] = X[c].replace(-np.inf,199)
    X[c] = X[c].replace(np.inf,199)




scaler = StandardScaler()
X1 = scaler.fit_transform(X.drop(['PlayId','Yards'],axis=1))




#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=12345)
#print(X_train.shape, X_val.shape)
#print(y_train.shape, y_val.shape)




from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda
from keras.models import Model
import keras.backend as K
import re
from keras.losses import binary_crossentropy
from  keras.callbacks import EarlyStopping,ModelCheckpoint
import codecs

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import f1_score




class CRPSCallback(Callback):
    
    def __init__(self,validation, predict_batch_size=20, include_on_batch=False):
        super(CRPSCallback, self).__init__()
        self.validation = validation
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch
        
        print('validation shape',len(self.validation))

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        if not ('CRPS_score_val' in self.params['metrics']):
            self.params['metrics'].append('CRPS_score_val')

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['CRPS_score_val'] = float('-inf')

    def on_epoch_end(self, epoch, logs={}):
        logs['CRPS_score_val'] = float('-inf')
            
        if (self.validation):
            X_valid, y_valid = self.validation[0], self.validation[1]
            y_pred = self.model.predict(X_valid)
            y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
            y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
            val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid.shape[0])
            val_s = np.round(val_s, 6)
            logs['CRPS_score_val'] = val_s
    




def get_model(x_tr,y_tr,x_val,y_val):
    inp = Input(shape = (x_tr.shape[1],))
    x = Dense(1024, input_dim=X.shape[1], activation='relu')(inp)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    
    out = Dense(199, activation='softmax')(x)
    model = Model(inp,out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[])
    #add lookahead
#     lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
#     lookahead.inject(model) # add into model

    
    es = EarlyStopping(monitor='CRPS_score_val', 
                       mode='min',
                       restore_best_weights=True, 
                       verbose=1, 
                       patience=10)

    mc = ModelCheckpoint('best_model.h5',monitor='CRPS_score_val',mode='min',
                                   save_best_only=True, verbose=1, save_weights_only=True)
    
    bsz = 512
    steps = x_tr.shape[0]/bsz
    


    model.fit(x_tr, y_tr,callbacks=[CRPSCallback(validation = (x_val,y_val)),es,mc], epochs=100, batch_size=bsz,verbose=1)
    model.load_weights("best_model.h5")
    
    y_pred1 = model.predict(x_val)
    y_valid = y_val
    y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred1, axis=1), 0, 1)
    val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * x_val.shape[0])
    crps = np.round(val_s, 6)

    return model,crps,y_pred1




X_2017 = X[X.PlayId< 20180906000000].copy().reset_index(drop=True)
X_2018 = X[X.PlayId> 20180906000000].copy().reset_index(drop=True)




from sklearn.model_selection import train_test_split, KFold,StratifiedKFold
import time

losses = []
models = []
crps_csv = []

s_time = time.time()

nn_cv = np.zeros((len(X),199))
for k in range(3):
    kfold = KFold(5, random_state = 42 + k, shuffle = True)
    for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(X)):
        print("-----------")
        
        tr_x= X.loc[tr_inds]
        val_x = X.loc[val_inds]
        
        #tr_x = pd.concat([X_2017,tr_x])
        #tr_x = pd.concat([X_2017,tr_x])
        
        tr_y = np.zeros((tr_x.shape[0], 199))
        for idx, target in enumerate(list(tr_x['Yards'])):
            tr_y[idx][99 + target] = 1
            
        val_y = np.zeros((val_x.shape[0], 199))
        for idx, target in enumerate(list(val_x['Yards'])):
            val_y[idx][99 + target] = 1
        
        del tr_x['Yards'],tr_x['PlayId']
        del val_x['Yards'],val_x['PlayId']
        
        tr_x = scaler.transform(tr_x)
        val_x= scaler.transform(val_x)
        
        model,crps,y_pred = get_model(tr_x,tr_y,val_x,val_y)
        nn_cv[val_inds] = y_pred
        models.append(model)
        print("the %d fold crps is %f"%((k_fold+1),crps))
        crps_csv.append(crps)
        to_rm_cols = []
        
print("mean crps is %f"%np.mean(crps_csv))        




def predict(x_te):
    model_num = len(models)
    for k,m in enumerate(models):
        if k==0:
            y_pred = m.predict(x_te,batch_size=1024)
        else:
            y_pred+=m.predict(x_te,batch_size=1024)
            
    y_pred = y_pred / model_num
    
    return y_pred




print("mean crps is %f"%np.mean(crps_csv))




get_ipython().run_cell_magic('time', '', "if  TRAIN_OFFLINE==False:\n    from kaggle.competitions import nflrush\n    env = nflrush.make_env()\n    iter_test = env.iter_test()\n\n    for (test_df, sample_prediction_df) in iter_test:\n        print(test_df.shape)\n        basetable = create_features(test_df, deploy=True)\n        wt= test_df[test_df.Rusher][['PlayId','PlayerWeight']].copy()\n        basetable = basetable.merge(wt,left_on=['PlayId'],right_on=['PlayId'],how='left')\n        basetable = basetable.fillna(-99)\n        basetable['AW'] = basetable.A / basetable.PlayerWeight\n        \n        basetable.drop(['GameId_x','GameId_y','PlayId'], axis=1, inplace=True)#\n        basetable.drop(tormcols, axis=1, inplace=True)\n        #print(basetable.shape)\n        scaled_basetable = scaler.transform(basetable)\n\n        y_pred = predict(scaled_basetable)\n        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]\n\n        preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)\n        env.predict(preds_df)\n\n    env.write_submission_file()")

