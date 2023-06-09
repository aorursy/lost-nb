#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle, Arc
import seaborn as sns
plt.style.use('seaborn-dark-palette')
mypal = plt.rcParams['axes.prop_cycle'].by_key()['color'] # Grab the color pal
import os
import gc

MENS_DIR = '../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament'
WOMENS_DIR = '../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament'




def logloss(true_label, predicted, eps=1e-15):
    p = np.clip(predicted, eps, 1 - eps)
    if true_label == 1:
        return -np.log(p)
    return -np.log(1 - p)




print(f'Confident Wrong Prediction: \t\t {logloss(1, 0.01):0.4f}')
print(f'Confident Correct Prediction: \t\t {logloss(0, 0.01):0.4f}')
print(f'Non-Confident Wrong Prediction: \t {logloss(1, 0.49):0.4f}')
print(f'Non-Confident Correct Prediction: \t {logloss(0, 0.49):0.4f}')




Mss = pd.read_csv(f'{MENS_DIR}/MSampleSubmissionStage1_2020.csv')
Wss = pd.read_csv(f'{WOMENS_DIR}/WSampleSubmissionStage1_2020.csv')
Mss.head()




MTeams = pd.read_csv(f'{MENS_DIR}/MDataFiles_Stage1/MTeams.csv')
MTeams.sort_values('FirstD1Season', ascending=False).head(5)




# Womens' data does not contain years joined :(
WTeams = pd.read_csv(f'{WOMENS_DIR}/WDataFiles_Stage1/WTeams.csv')
WTeams.head()




MSeasons = pd.read_csv(f'{MENS_DIR}/MDataFiles_Stage1/MSeasons.csv')
WSeasons = pd.read_csv(f'{WOMENS_DIR}/WDataFiles_Stage1/WSeasons.csv')
MSeasons.head()




MNCAATourneySeeds = pd.read_csv(f'{MENS_DIR}/MDataFiles_Stage1/MNCAATourneySeeds.csv')
WNCAATourneySeeds = pd.read_csv(f'{WOMENS_DIR}/WDataFiles_Stage1/WNCAATourneySeeds.csv')




# As Lets join this with the teams data to see some of the past matchups
WNCAATourneySeeds.merge(WTeams, validate='many_to_one').head()




MRegularSeasonCompactResults = pd.read_csv(f'{MENS_DIR}/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')
WRegularSeasonCompactResults = pd.read_csv(f'{WOMENS_DIR}/WDataFiles_Stage1/WRegularSeasonCompactResults.csv')




# We have the team the won, lost and the score.
MRegularSeasonCompactResults.head(5)




# Lets Add the winning and losing team names to the results
MRegularSeasonCompactResults =     MRegularSeasonCompactResults     .merge(MTeams[['TeamName', 'TeamID']],
           left_on='WTeamID',
           right_on='TeamID',
           validate='many_to_one') \
    .drop('TeamID', axis=1) \
    .rename(columns={'TeamName': 'WTeamName'}) \
    .merge(MTeams[['TeamName', 'TeamID']],
           left_on='LTeamID',
           right_on='TeamID') \
    .drop('TeamID', axis=1) \
    .rename(columns={'TeamName': 'LTeamName'})

WRegularSeasonCompactResults =     WRegularSeasonCompactResults     .merge(WTeams[['TeamName', 'TeamID']],
           left_on='WTeamID',
           right_on='TeamID',
           validate='many_to_one') \
    .drop('TeamID', axis=1) \
    .rename(columns={'TeamName': 'WTeamName'}) \
    .merge(WTeams[['TeamName', 'TeamID']],
           left_on='LTeamID',
           right_on='TeamID') \
    .drop('TeamID', axis=1) \
    .rename(columns={'TeamName': 'LTeamName'})




WRegularSeasonCompactResults.head()




WRegularSeasonCompactResults['Score_Diff'] = WRegularSeasonCompactResults['WScore'] - WRegularSeasonCompactResults['LScore']
MRegularSeasonCompactResults['Score_Diff'] = MRegularSeasonCompactResults['WScore'] - MRegularSeasonCompactResults['LScore']




plt.style.use('fivethirtyeight')
MRegularSeasonCompactResults['Score_Diff']     .plot(kind='hist',
          bins=90,
          figsize=(15, 5),
          label='Mens',
          alpha=0.5)
WRegularSeasonCompactResults['Score_Diff']     .plot(kind='hist',
          bins=105,
          figsize=(15, 5),
          label='Womens',
          alpha=0.5)
plt.title('Score Differential')
plt.xlim(0,60)
plt.legend()
plt.show()




plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
WRegularSeasonCompactResults['counter'] = 1
WRegularSeasonCompactResults.groupby('WTeamName')['counter']     .count()     .sort_values()     .tail(20)     .plot(kind='barh',
          title='⬆️ Most Regular Season Wins (Womens)',
          figsize=(15, 8),
          xlim=(400, 680),
          color=mypal[0],
          ax=axs[0])
WRegularSeasonCompactResults.groupby('WTeamName')['counter']     .count()     .sort_values(ascending=False)     .tail(20)     .plot(kind='barh',
          title='⬇️ Least Regular Season Wins (Womens)',
          figsize=(15, 8),
          xlim=(0, 150),
          color=mypal[1],
          ax=axs[1])
plt.tight_layout()
plt.show()




plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
MRegularSeasonCompactResults['counter'] = 1
MRegularSeasonCompactResults.groupby('WTeamName')['counter']     .count()     .sort_values()     .tail(20)     .plot(kind='barh',
          title='⬆️ Most Regular Season Wins (Mens)',
          figsize=(15, 8),
          xlim=(600, 920),
          color=mypal[2],
         ax=axs[0])
MRegularSeasonCompactResults.groupby('WTeamName')['counter']     .count()     .sort_values(ascending=False)     .tail(20)     .plot(kind='barh',
          title='⬇️ Least Regular Season Wins (Mens)',
          figsize=(15, 8),
          xlim=(0, 150),
          color=mypal[3],
          ax=axs[1])
axs[1].set_ylabel('')
plt.tight_layout()
plt.show()




mens_events = []
for year in [2015, 2016, 2017, 2018, 2019]:
    mens_events.append(pd.read_csv(f'{MENS_DIR}/MEvents{year}.csv'))
MEvents = pd.concat(mens_events)
print(MEvents.shape)
MEvents.head()




womens_events = []
for year in [2015, 2016, 2017, 2018, 2019]:
    womens_events.append(pd.read_csv(f'{WOMENS_DIR}/WEvents{year}.csv'))
WEvents = pd.concat(womens_events)
print(WEvents.shape)
WEvents.head()




del mens_events
del womens_events
gc.collect()




# Event Types
plt.style.use('fivethirtyeight')
MEvents['counter'] = 1
MEvents.groupby('EventType')['counter']     .sum()     .sort_values(ascending=False)     .plot(kind='bar',
          figsize=(15, 5),
         color=mypal[2],
         title='Event Type Frequency (Mens)')
plt.xticks(rotation=0)
plt.show()




# Event Types
plt.style.use('fivethirtyeight')
WEvents['counter'] = 1
WEvents.groupby('EventType')['counter']     .sum()     .sort_values(ascending=False)     .plot(kind='bar',
          figsize=(15, 5),
         color=mypal[3],
         title='Event Type Frequency (Womens)')
plt.xticks(rotation=0)
plt.show()




area_mapping = {0: np.nan,
                1: 'under basket',
                2: 'in the paint',
                3: 'inside right wing',
                4: 'inside right',
                5: 'inside center',
                6: 'inside left',
                7: 'inside left wing',
                8: 'outside right wing',
                9: 'outside right',
                10: 'outside center',
                11: 'outside left',
                12: 'outside left wing',
                13: 'backcourt'}

MEvents['Area_Name'] = MEvents['Area'].map(area_mapping)




MEvents.groupby('Area_Name')['counter'].sum()     .sort_values()     .plot(kind='barh',
          figsize=(15, 8),
          title='Frequency of Event Area')
plt.show()




fig, ax = plt.subplots(figsize=(15, 8))
for i, d in MEvents.loc[~MEvents['Area_Name'].isna()].groupby('Area_Name'):
    d.plot(x='X', y='Y', style='.', label=i, ax=ax, title='Visualizing Event Areas')
    ax.legend()
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
plt.show()




# Normalize X, Y positions for court dimentions
# Court is 50 feet wide and 94 feet end to end.
MEvents['X_'] = (MEvents['X'] * (94/100))
MEvents['Y_'] = (MEvents['Y'] * (50/100))

WEvents['X_'] = (WEvents['X'] * (94/100))
WEvents['Y_'] = (WEvents['Y'] * (50/100))




def create_ncaa_full_court(ax=None, three_line='mens', court_color='#dfbb85',
                           lw=3, lines_color='black', lines_alpha=0.5,
                           paint_fill='blue', paint_alpha=0.4,
                           inner_arc=False):
    """
    Version 2020.2.19
    Creates NCAA Basketball Court
    Dimensions are in feet (Court is 97x50 ft)
    Created by: Rob Mulla / https://github.com/RobMulla

    * Note that this function uses "feet" as the unit of measure.
    * NCAA Data is provided on a x range: 0, 100 and y-range 0 to 100
    * To plot X/Y positions first convert to feet like this:
    ```
    Events['X_'] = (Events['X'] * (94/100))
    Events['Y_'] = (Events['Y'] * (50/100))
    ```
    
    ax: matplotlib axes if None gets current axes using `plt.gca`


    three_line: 'mens', 'womens' or 'both' defines 3 point line plotted
    court_color : (hex) Color of the court
    lw : line width
    lines_color : Color of the lines
    lines_alpha : transparency of lines
    paint_fill : Color inside the paint
    paint_alpha : transparency of the "paint"
    inner_arc : paint the dotted inner arc
    """
    if ax is None:
        ax = plt.gca()

    # Create Pathes for Court Lines
    center_circle = Circle((94/2, 50/2), 6,
                           linewidth=lw, color=lines_color, lw=lw,
                           fill=False, alpha=lines_alpha)
    hoop_left = Circle((5.25, 50/2), 1.5 / 2,
                       linewidth=lw, color=lines_color, lw=lw,
                       fill=False, alpha=lines_alpha)
    hoop_right = Circle((94-5.25, 50/2), 1.5 / 2,
                        linewidth=lw, color=lines_color, lw=lw,
                        fill=False, alpha=lines_alpha)

    # Paint - 18 Feet 10 inches which converts to 18.833333 feet - gross!
    left_paint = Rectangle((0, (50/2)-6), 18.833333, 12,
                           fill=paint_fill, alpha=paint_alpha,
                           lw=lw, edgecolor=None)
    right_paint = Rectangle((94-18.83333, (50/2)-6), 18.833333,
                            12, fill=paint_fill, alpha=paint_alpha,
                            lw=lw, edgecolor=None)
    
    left_paint_boarder = Rectangle((0, (50/2)-6), 18.833333, 12,
                           fill=False, alpha=lines_alpha,
                           lw=lw, edgecolor=lines_color)
    right_paint_boarder = Rectangle((94-18.83333, (50/2)-6), 18.833333,
                            12, fill=False, alpha=lines_alpha,
                            lw=lw, edgecolor=lines_color)

    left_arc = Arc((18.833333, 50/2), 12, 12, theta1=-
                   90, theta2=90, color=lines_color, lw=lw,
                   alpha=lines_alpha)
    right_arc = Arc((94-18.833333, 50/2), 12, 12, theta1=90,
                    theta2=-90, color=lines_color, lw=lw,
                    alpha=lines_alpha)
    
    leftblock1 = Rectangle((7, (50/2)-6-0.666), 1, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    leftblock2 = Rectangle((7, (50/2)+6), 1, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    ax.add_patch(leftblock1)
    ax.add_patch(leftblock2)
    
    left_l1 = Rectangle((11, (50/2)-6-0.666), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    left_l2 = Rectangle((14, (50/2)-6-0.666), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    left_l3 = Rectangle((17, (50/2)-6-0.666), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    ax.add_patch(left_l1)
    ax.add_patch(left_l2)
    ax.add_patch(left_l3)
    left_l4 = Rectangle((11, (50/2)+6), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    left_l5 = Rectangle((14, (50/2)+6), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    left_l6 = Rectangle((17, (50/2)+6), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    ax.add_patch(left_l4)
    ax.add_patch(left_l5)
    ax.add_patch(left_l6)
    
    rightblock1 = Rectangle((94-7-1, (50/2)-6-0.666), 1, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    rightblock2 = Rectangle((94-7-1, (50/2)+6), 1, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    ax.add_patch(rightblock1)
    ax.add_patch(rightblock2)

    right_l1 = Rectangle((94-11, (50/2)-6-0.666), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    right_l2 = Rectangle((94-14, (50/2)-6-0.666), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    right_l3 = Rectangle((94-17, (50/2)-6-0.666), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    ax.add_patch(right_l1)
    ax.add_patch(right_l2)
    ax.add_patch(right_l3)
    right_l4 = Rectangle((94-11, (50/2)+6), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    right_l5 = Rectangle((94-14, (50/2)+6), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    right_l6 = Rectangle((94-17, (50/2)+6), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    ax.add_patch(right_l4)
    ax.add_patch(right_l5)
    ax.add_patch(right_l6)
    
    # 3 Point Line
    if (three_line == 'mens') | (three_line == 'both'):
        # 22' 1.75" distance to center of hoop
        three_pt_left = Arc((6.25, 50/2), 44.291, 44.291, theta1=-78,
                            theta2=78, color=lines_color, lw=lw,
                            alpha=lines_alpha)
        three_pt_right = Arc((94-6.25, 50/2), 44.291, 44.291,
                             theta1=180-78, theta2=180+78,
                             color=lines_color, lw=lw, alpha=lines_alpha)

        # 4.25 feet max to sideline for mens
        ax.plot((0, 11.25), (3.34, 3.34),
                color=lines_color, lw=lw, alpha=lines_alpha)
        ax.plot((0, 11.25), (50-3.34, 50-3.34),
                color=lines_color, lw=lw, alpha=lines_alpha)
        ax.plot((94-11.25, 94), (3.34, 3.34),
                color=lines_color, lw=lw, alpha=lines_alpha)
        ax.plot((94-11.25, 94), (50-3.34, 50-3.34),
                color=lines_color, lw=lw, alpha=lines_alpha)
        ax.add_patch(three_pt_left)
        ax.add_patch(three_pt_right)

    if (three_line == 'womens') | (three_line == 'both'):
        # womens 3
        three_pt_left_w = Arc((6.25, 50/2), 20.75 * 2, 20.75 * 2, theta1=-85,
                              theta2=85, color=lines_color, lw=lw, alpha=lines_alpha)
        three_pt_right_w = Arc((94-6.25, 50/2), 20.75 * 2, 20.75 * 2,
                               theta1=180-85, theta2=180+85,
                               color=lines_color, lw=lw, alpha=lines_alpha)

        # 4.25 inches max to sideline for mens
        ax.plot((0, 8.3), (4.25, 4.25), color=lines_color,
                lw=lw, alpha=lines_alpha)
        ax.plot((0, 8.3), (50-4.25, 50-4.25),
                color=lines_color, lw=lw, alpha=lines_alpha)
        ax.plot((94-8.3, 94), (4.25, 4.25),
                color=lines_color, lw=lw, alpha=lines_alpha)
        ax.plot((94-8.3, 94), (50-4.25, 50-4.25),
                color=lines_color, lw=lw, alpha=lines_alpha)

        ax.add_patch(three_pt_left_w)
        ax.add_patch(three_pt_right_w)

    # Add Patches
    ax.add_patch(left_paint)
    ax.add_patch(left_paint_boarder)
    ax.add_patch(right_paint)
    ax.add_patch(right_paint_boarder)
    ax.add_patch(center_circle)
    ax.add_patch(hoop_left)
    ax.add_patch(hoop_right)
    ax.add_patch(left_arc)
    ax.add_patch(right_arc)
    
    if inner_arc:
        left_inner_arc = Arc((18.833333, 50/2), 12, 12, theta1=90,
                             theta2=-90, color=lines_color, lw=lw,
                       alpha=lines_alpha, ls='--')
        right_inner_arc = Arc((94-18.833333, 50/2), 12, 12, theta1=-90,
                        theta2=90, color=lines_color, lw=lw,
                        alpha=lines_alpha, ls='--')
        ax.add_patch(left_inner_arc)
        ax.add_patch(right_inner_arc)

    # Restricted Area Marker
    restricted_left = Arc((6.25, 50/2), 8, 8, theta1=-90,
                        theta2=90, color=lines_color, lw=lw,
                        alpha=lines_alpha)
    restricted_right = Arc((94-6.25, 50/2), 8, 8,
                         theta1=180-90, theta2=180+90,
                         color=lines_color, lw=lw, alpha=lines_alpha)
    ax.add_patch(restricted_left)
    ax.add_patch(restricted_right)
    
    # Backboards
    ax.plot((4, 4), ((50/2) - 3, (50/2) + 3),
            color=lines_color, lw=lw*1.5, alpha=lines_alpha)
    ax.plot((94-4, 94-4), ((50/2) - 3, (50/2) + 3),
            color=lines_color, lw=lw*1.5, alpha=lines_alpha)
    ax.plot((4, 4.6), (50/2, 50/2), color=lines_color,
            lw=lw, alpha=lines_alpha)
    ax.plot((94-4, 94-4.6), (50/2, 50/2),
            color=lines_color, lw=lw, alpha=lines_alpha)

    # Half Court Line
    ax.axvline(94/2, color=lines_color, lw=lw, alpha=lines_alpha)

    # Boarder
    boarder = Rectangle((0.3,0.3), 94-0.4, 50-0.4, fill=False, lw=3, color='black', alpha=lines_alpha)
    ax.add_patch(boarder)
    
    # Plot Limit
    ax.set_xlim(0, 94)
    ax.set_ylim(0, 50)
    ax.set_facecolor(court_color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    return ax


fig, ax = plt.subplots(figsize=(15, 8.5))
create_ncaa_full_court(ax, three_line='both', paint_alpha=0.4)
plt.show()




fig, ax = plt.subplots(figsize=(15, 7.8))
ms = 10
ax = create_ncaa_full_court(ax, paint_alpha=0.1)
MEvents.query('EventType == "turnover"')     .plot(x='X_', y='Y_', style='X',
          title='Turnover Locations (Mens)',
          c='red',
          alpha=0.3,
         figsize=(15, 9),
         label='Steals',
         ms=ms,
         ax=ax)
ax.set_xlabel('')
ax.get_legend().remove()
plt.show()




COURT_COLOR = '#dfbb85'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
# Where are 3 pointers made from? (This is really cool)
WEvents.query('EventType == "made3"')     .plot(x='X_', y='Y_', style='.',
          color='blue',
          title='3 Pointers Made (Womens)',
          alpha=0.01, ax=ax1)
ax1 = create_ncaa_full_court(ax1, lw=0.5, three_line='womens', paint_alpha=0.1)
ax1.set_facecolor(COURT_COLOR)
WEvents.query('EventType == "miss3"')     .plot(x='X_', y='Y_', style='.',
          title='3 Pointers Missed (Womens)',
          color='red',
          alpha=0.01, ax=ax2)
ax2.set_facecolor(COURT_COLOR)
ax2 = create_ncaa_full_court(ax2, lw=0.5, three_line='womens', paint_alpha=0.1)
ax1.get_legend().remove()
ax2.get_legend().remove()
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax1.set_xlabel('')
ax2.set_xlabel('')
plt.show()




COURT_COLOR = '#dfbb85'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
# Where are 3 pointers made from? (This is really cool)
WEvents.query('EventType == "made2"')     .plot(x='X_', y='Y_', style='.',
          color='blue',
          title='2 Pointers Made (Womens)',
          alpha=0.01, ax=ax1)
ax1.set_facecolor(COURT_COLOR)
ax1 = create_ncaa_full_court(ax1, lw=0.5, three_line='womens', paint_alpha=0.1)
WEvents.query('EventType == "miss2"')     .plot(x='X_', y='Y_', style='.',
          title='2 Pointers Missed (Womens)',
          color='red',
          alpha=0.01, ax=ax2)
ax2.set_facecolor(COURT_COLOR)
ax2 = create_ncaa_full_court(ax2, lw=0.5, three_line='womens', paint_alpha=0.1)
ax1.get_legend().remove()
ax2.get_legend().remove()
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax1.set_xlabel('')
ax2.set_xlabel('')
plt.show()




MPlayers = pd.read_csv(f'{MENS_DIR}/MPlayers.csv', error_bad_lines=False)
WPlayers = pd.read_csv(f'{WOMENS_DIR}/WPlayers.csv')




MPlayers.head()




# Merge Player name onto events
MEvents = MEvents.merge(MPlayers,
              how='left',
              left_on='EventPlayerID',
              right_on='PlayerID')

WEvents = WEvents.merge(WPlayers,
              how='left',
              left_on='EventPlayerID',
              right_on='PlayerID')




MEvents.loc[MEvents['PlayerID'] == 2825].groupby('EventType')['EventID'].count()     .sort_values()     .plot(kind='barh',
          figsize=(15, 5),
          title='Zion Williamson event type count',
          color=mypal[1])
plt.show()




ms = 10 # Marker Size
FirstName = 'Zion'
LastName = 'Williamson'
fig, ax = plt.subplots(figsize=(15, 8))
ax = create_ncaa_full_court(ax)
MEvents.query('FirstName == @FirstName and LastName == @LastName and EventType == "made2"')     .plot(x='X_', y='Y_', style='o',
          title='Shots (Zion Williamson)',
          alpha=0.5,
         figsize=(15, 8),
         label='Made 2',
         ms=ms,
         ax=ax)
plt.legend()
MEvents.query('FirstName == @FirstName and LastName == @LastName and EventType == "miss2"')     .plot(x='X_', y='Y_', style='X',
          alpha=0.5, ax=ax,
         label='Missed 2',
         ms=ms)
plt.legend()
MEvents.query('FirstName == @FirstName and LastName == @LastName and EventType == "made3"')     .plot(x='X_', y='Y_', style='o',
          c='brown',
          alpha=0.5,
         figsize=(15, 8),
         label='Made 3', ax=ax,
         ms=ms)
plt.legend()
MEvents.query('FirstName == @FirstName and LastName == @LastName and EventType == "miss3"')     .plot(x='X_', y='Y_', style='X',
          c='green',
          alpha=0.5, ax=ax,
         label='Missed 3',
         ms=ms)
ax.set_xlabel('')
plt.legend()
plt.show()




ms = 10 # Marker Size
FirstName = 'Katie Lou'
LastName = 'Samuelson'
fig, ax = plt.subplots(figsize=(15, 8))
ax = create_ncaa_full_court(ax, three_line='womens')
WEvents.query('FirstName == @FirstName and LastName == @LastName and EventType == "made2"')     .plot(x='X_', y='Y_', style='o',
          title='Shots (Katie Lou Samuelson)',
          alpha=0.5,
         figsize=(15, 8),
         label='Made 2',
         ms=ms,
         ax=ax)
plt.legend()
WEvents.query('FirstName == @FirstName and LastName == @LastName and EventType == "miss2"')     .plot(x='X_', y='Y_', style='X',
          alpha=0.5, ax=ax,
         label='Missed 2',
         ms=ms)
plt.legend()
WEvents.query('FirstName == @FirstName and LastName == @LastName and EventType == "made3"')     .plot(x='X_', y='Y_', style='o',
          c='brown',
          alpha=0.5,
         figsize=(15, 8),
         label='Made 3', ax=ax,
         ms=ms)
plt.legend()
WEvents.query('FirstName == @FirstName and LastName == @LastName and EventType == "miss3"')     .plot(x='X_', y='Y_', style='X',
          c='green',
          alpha=0.5, ax=ax,
         label='Missed 3',
         ms=ms)
ax.set_xlabel('')
plt.legend()
plt.show()




N_bins = 100
shot_events = MEvents.loc[MEvents['EventType'].isin(['miss3','made3','miss2','made2']) & (MEvents['X_'] != 0)]
fig, ax = plt.subplots(figsize=(15, 7))
ax = create_ncaa_full_court(ax,
                            paint_alpha=0.0,
                            three_line='mens',
                            court_color='black',
                            lines_color='white')
_ = plt.hist2d(shot_events['X_'].values + np.random.normal(0, 0.1, shot_events['X_'].shape), # Add Jitter to values for plotting
           shot_events['Y_'].values + np.random.normal(0, 0.1, shot_events['Y_'].shape),
           bins=N_bins, norm=mpl.colors.LogNorm(),
               cmap='plasma')

# Plot a colorbar with label.
cb = plt.colorbar()
cb.set_label('Number of shots')

ax.set_title('Shot Heatmap (Mens)')
plt.show()




N_bins = 100
shot_events = WEvents.loc[WEvents['EventType'].isin(['miss3','made3','miss2','made2']) & (WEvents['X_'] != 0)]
fig, ax = plt.subplots(figsize=(15, 7))
ax = create_ncaa_full_court(ax, three_line='womens', paint_alpha=0.0,
                            court_color='black',
                            lines_color='white')
_ = plt.hist2d(shot_events['X_'].values + np.random.normal(0, 0.2, shot_events['X_'].shape),
           shot_events['Y_'].values + np.random.normal(0, 0.2, shot_events['Y_'].shape),
           bins=N_bins, norm=mpl.colors.LogNorm(),
               cmap='plasma')

# Plot a colorbar with label.
cb = plt.colorbar()
cb.set_label('Number of shots')

ax.set_title('Shot Heatmap (Womens)')
plt.show()

