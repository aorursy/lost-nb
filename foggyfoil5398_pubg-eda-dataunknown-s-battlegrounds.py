#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




pubg = pd.read_csv('../input/train_V2.csv')




pubg.head()




#Every column there is !
pubg.columns




#I am going to work on the first 1million rows of data since my kernel can't really handle 4.2m otherwise
pubg = pubg[:1000000]




pubg.shape




pubg.describe().iloc[0,:] 




get_ipython().run_line_magic('matplotlib', "inline # so that I don't have to type plt.show() everytime")




sns.heatmap(pubg.isnull(),yticklabels=False)
f = plt.gcf()
f.set_size_inches(10,8)




pubg['matchType'].value_counts()




#converting them
pubg['matchType'].replace(['squad-fpp','duo-fpp','solo-fpp','normal-squad-fpp','crashfpp','normal-duo-fpp','flaretpp',
                           'normal-solo-fpp','flarefpp','normal-squad','crashtpp','normal-solo','normal-duo'],
                          ['squad','duo','solo','squad','others','duo','others','solo','others','squad',
                          'others','solo','duo'],inplace=True)
#others matchtype represent names of types which I can't personally distinguish as solo/squad/duo even after searching on kaggle




pubg['matchType'].value_counts()




pubg['assists'].value_counts().to_frame().style.background_gradient(cmap='summer_r')




sns.factorplot('assists','winPlacePerc',hue='matchType',data=pubg)
f = plt.gcf()
f.set_size_inches(15,8)
plt.title('Assists vs winPlacePerc')




pd.crosstab(pubg.boosts,pubg.matchType).style.background_gradient(cmap='summer_r')




x = pubg[pubg['boosts']<6].count()[0]/pubg.shape[0]*100
print('Percentage of players who used less than 6 boosts per game : ',x)




sns.barplot('boosts','winPlacePerc',hue='matchType',data=pubg)
plt.title('Boosts vs WinPlacePerc')
f = plt.gcf()
f.set_size_inches(15,8)




f,ax=plt.subplots(1,2,figsize=(18,6))
sns.distplot(pubg.damageDealt,kde=True,ax=ax[0])
plt.title('Density Graph of DamageDealt')
sns.scatterplot('damageDealt','winPlacePerc',hue='matchType',ax=ax[1],data=pubg)
plt.title('DamageDealt vs winPlacePerc')




pubg.groupby(['matchType'])['damageDealt'].mean()




pubg[(pubg['matchType']=='solo')&(pubg['winPlacePerc']==0)&(pubg['damageDealt']>500)].count()[0]




pubg[(pubg['matchType']=='solo')&(pubg['winPlacePerc']==0)&(pubg['damageDealt']>1500)].count()[0]




for i in range(5):    
    plt.title('DamageDealt with winPlacePerc = '+str(0.4+(i/10))+' - '+str(0.4+(i+1)/10))
    pubg[(pubg['winPlacePerc']>=0.4+(i/10))&(pubg['winPlacePerc']<0.4+((i+1)/10))].damageDealt.plot.hist(color='red',edgecolor='black',bins=20)
    plt.show()




pubg['headshotKills'].value_counts().to_frame().style.background_gradient(cmap='summer_r')




sns.factorplot('headshotKills','winPlacePerc',hue='matchType',data=pubg)
f = plt.gcf()
f.set_size_inches(10,8)
plt.title('Headshot Kills vs winPlacePerc')




x = pubg[(pubg['headshotKills']>5)&(pubg['winPlacePerc']>0.8)].count()[0]/pubg[pubg['headshotKills']>5].count()[0]*100
print('Percentage of people having more than 5 headshot kills and having winPlacePerc > 0.8 : ',x)




sns.factorplot('heals','winPlacePerc',hue='matchType',data=pubg)
f = plt.gcf()
f.set_size_inches(25,15)
plt.title('Heals vs winPlacePerc')




sns.scatterplot('killPlace','winPlacePerc',data=pubg[pubg['matchType']=='solo'])
f=plt.gcf()
f.set_size_inches(10,8)
plt.title('KillPlace vs winPlacePerc')




sns.heatmap(pubg[['killPlace','winPlacePerc']].corr(),annot=True)




pubg.drop(['killPoints','rankPoints'],axis=1,inplace=True)




pubg.shape[1] #drop success !




sns.factorplot('killStreaks','winPlacePerc',hue='matchType',data=pubg)
f = plt.gcf()
f.set_size_inches(12,9)
plt.title('KillStreaks vs winPlacePerc')




f,ax = plt.subplots(2,2,figsize=(15,10))
sns.barplot('kills','winPlacePerc',data=pubg[pubg['matchType']=='solo'],ax=ax[0,0],palette='Blues')
ax[0,0].set_title('Solos')
sns.barplot('kills','winPlacePerc',data=pubg[pubg['matchType']=='duo'],ax=ax[0,1],palette='Blues_r')
ax[0,1].set_title('Duos')
sns.barplot('kills','winPlacePerc',data=pubg[pubg['matchType']=='squad'],ax=ax[1,0],palette='OrRd')
ax[1,0].set_title('Squads')
sns.barplot('kills','winPlacePerc',data=pubg[pubg['matchType']=='others'],ax=ax[1,1],palette= 'OrRd_r')
ax[1,1].set_title('Others')




pubg['kills'].max()




pubg['kills'].argmax()




pubg.iloc[334400].to_frame()




sns.heatmap(pubg[['kills','winPlacePerc']].corr(),annot=True)




print('Longest Kill :',str(pubg['longestKill'].max()) + ' - AWM + 15x')
print('Shortest Kill :',str(pubg['longestKill'].min())+' - Pan Kill')




sns.scatterplot('longestKill','winPlacePerc',data=pubg)
f = plt.gcf()
f.set_size_inches(12,9)
plt.title('LongestKill vs WinPlacePerc')




x = pubg[(pubg['longestKill']>200)&(pubg['winPlacePerc']>0.8)].count()[0]/pubg[pubg['longestKill']>200].count()[0]*100
print('Percentage of people who had LongestKills > 200m and winPlacePerc > 0.8 :',x)




sns.distplot(pubg.matchDuration)
f = plt.gcf()
f.set_size_inches(10,8)
plt.title('Match duration Density Graph')




1400/60,1800/60




pubg.groupby(['matchType'])['revives'].value_counts().to_frame().style.background_gradient(cmap='summer_r')




sns.catplot('revives','winPlacePerc',data=pubg[pubg.matchType=='squad'],jitter=False)
f=plt.gcf()
f.set_size_inches(10,8)
plt.title('Squad revives vs winPlacePerc')




print('The highest ride distance by a player :',pubg['rideDistance'].max())




sns.scatterplot('rideDistance','winPlacePerc',data=pubg)
f=plt.gcf()
f.set_size_inches(10,8)
plt.title('rideDistance vs winPlacePerc')




for i in range(5):    
    plt.title('RideDistance with winPlacePerc = '+str(0.4+(i/10))+' - '+str(0.4+(i+1)/10))
    pubg[(pubg['winPlacePerc']>=0.4+(i/10))&(pubg['winPlacePerc']<0.4+((i+1)/10))&(pubg['rideDistance']<10000)].rideDistance.plot.hist(color='red',edgecolor='black',bins=20)
    plt.show()




sns.heatmap(pubg[['rideDistance','winPlacePerc']].corr(),annot=True)




pubg['roadKills'].value_counts()




sns.catplot('roadKills','winPlacePerc',kind='boxen',data=pubg)
f = plt.gcf()
f.set_size_inches(12,9)
plt.title('RoadKills vs winPlacePerc')




sns.violinplot('teamKills','winPlacePerc',hue='matchType',data=pubg[(pubg['matchType']=='squad')|(pubg['matchType']=='duo')],split=True)
f = plt.gcf()
f.set_size_inches(10,8)
plt.title('TeamKills vs winPlacePerc (duos and squads only)')




sns.factorplot('vehicleDestroys','winPlacePerc',hue='matchType',data=pubg)
f=plt.gcf()
f.set_size_inches(10,8)
plt.title('VehiclesDestroyed vs winPlacePerc')




#below represents people who haven't moved a single step but still had more than 0 kill
pubg[(pubg['walkDistance']==0)&(pubg['kills']>0)].count()[0]




pubg.drop(pubg[(pubg['walkDistance']==0)&(pubg['kills']>0)].index,inplace=True)




sns.scatterplot('walkDistance','winPlacePerc',data=pubg)
f = plt.gcf()
f.set_size_inches(8,6)
plt.title('WalkDistance vs winPlacePerc')




# Most of the data have walkdistance value < 10000m. So that's why I have put a limit on it in below code
pubg[(pubg['winPlacePerc']>0.9)&(pubg['walkDistance']<10000)].walkDistance.plot.hist(bins=50,edgecolor='black')
f = plt.gcf()
f.set_size_inches(12,9)
plt.title('WalkDistance Frequency')




pubg['walkDistance'].mean()




print(pubg['walkDistance'].max())
print("How the hell can someone walk a fu****g 25 Km. Let's see who he actually is !")
pubg[pubg['walkDistance']>25000].iloc[0,:].to_frame()




sns.catplot('weaponsAcquired','winPlacePerc',data=pubg, jitter=False)
f=plt.gcf()
f.set_size_inches(20,15)
plt.title('Weapons Acquired vs winPlacePerc')




sns.heatmap(pubg[['weaponsAcquired','winPlacePerc']].corr(),annot=True)




sns.heatmap(pubg.corr(),annot=True,linewidths=0.5,cmap='RdYlGn')
f = plt.gcf()
f.set_size_inches(25,20)




get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1548504873335' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;26&#47;266JW6WP2&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;266JW6WP2' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;26&#47;266JW6WP2&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1548504873335');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='1029px';vizElement.style.height='722px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                   \nvizElement.parentNode.insertBefore(scriptElement, vizElement);    \n</script>")




pubg['health'] = pubg['heals'] + pubg['boosts']




sns.heatmap(pubg[['health','heals','boosts','winPlacePerc']].corr(),annot=True)
f=plt.gcf()
f.set_size_inches(8,6)




def head(c):
    if(c[0]==0):
        return 0
    elif(c[1]==0):
        return 0
    else:
        return c[0]/c[1]
pubg['headshotSkill'] = pubg[['headshotKills','kills']].apply(head,axis=1)




pubg.describe()




sns.heatmap(pubg[['headshotKills','headshotSkill','kills','winPlacePerc']].corr(),annot=True)
f=plt.gcf()
f.set_size_inches(8,6)




pubg['distanceCovered'] = pubg['walkDistance']+pubg['rideDistance']+pubg['swimDistance']




sns.heatmap(pubg[['distanceCovered','walkDistance','rideDistance','swimDistance','winPlacePerc']].corr(),annot=True)
f=plt.gcf()
f.set_size_inches(8,6)




pubg['actualKills'] = pubg['kills'] - pubg['teamKills']




sns.heatmap(pubg[['kills','actualKills','teamKills','winPlacePerc']].corr(),annot=True)
f=plt.gcf()
f.set_size_inches(8,6)






