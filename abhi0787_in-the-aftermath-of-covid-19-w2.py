#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




#Analysis done by: Abhishe Mukherjee
#Queries/questions/contradictions can be directly addressed to: abhi0787@gmail.com or amukher3@rockets.utoledo.edu




COVID-19 Analysis/Exploratory Data Analysis Week-2 kaggle Data

Various locations across the world. Data includes the Week-2 data for the month of March along with January and February.

A note about using filters and link to the data used: https://github.com/amukher3/COVID-19/blob/master/README.md




# An assemblage of the entire analysis. 

#The world map shows the fatalities across the globe along with a cumulative representation of confirmed case 
#during the months of January,February and March. 
#China seems to have the highest numbers. 




get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1586128490041' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;CO&#47;COVID_19_Kaggle_Data&#47;World_Map_cases&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='COVID_19_Kaggle_Data&#47;World_Map_cases' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;CO&#47;COVID_19_Kaggle_Data&#47;World_Map_cases&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1586128490041');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")




get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1586128929322' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;83&#47;83W2SPDK5&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;83W2SPDK5' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;83&#47;83W2SPDK5&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1586128929322');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")




#**Role of Longitude:

#Longitude seems to have no apparent relation with the number of confirmed cases. The zone that seems to have some 
#apparent relation with the number of confirmed cases is the Euro zone. Neighnbouring countries in that zone seems to 
#grown at a steady pace following Italy. **




get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1586128981746' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;D4&#47;D4RKFC39H&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;D4RKFC39H' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;D4&#47;D4RKFC39H&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1586128981746');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")




#The curve with time: 

#The time series seems to paint a clear picture of the changes in the number of confirmed cases as a function of time
#for the months of January,February and March. 

#If analyzed carefully, the progression potrays the fact that East Asian countries like China,which had huge number of cases 
#and a very steep rate of increase in the number of cases had been able to "flatten the curve" by the begining the March indicated
#by a much slower rise in the number of confirmed cases. 

#On the flipside western countries like US which had a relatively much lesser number of cases and a much flatter for the months of
#Jan/Feb had started to grapple once the infection reached the levels of community transmission. 

#The rate of increase for the number of confirmed cases seems to have grown exponentially,without any signs of slowing down,
#surpassing countries the countries which were finding it difficult to deal with the pandemic.




get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1586135344971' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;X9&#47;X9R8DF2ZN&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;X9R8DF2ZN' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;X9&#47;X9R8DF2ZN&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1586135344971');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")




#Curve flexion(St.Patrick's day):

#A careful analysis for the month of March the curve for the US seems to have "flexed",attaining a much higher rate of 
#increase around March 18th indicatinig a large number of confirmed cases,approximately double of what was reported the previous
#day.

#This point in the history of the global pandemic, March 18th, seems to be the saddle point for the curve of US. 

#March 17th being celebrated widely across the US as St.Patrick's day could be a potential reason for the sudden change 
#in the characteristics of the curve. This is probably the pivotal point in the country's pandemic history where it entered 
#community transmission phase.




# February 12th: 

# China seems to have seen a surge in cases on February 12th this has apparently been attributed to 
# re-categorisation of patients
#More details: https://www.cnbc.com/2020/02/26/confusion-breeds-distrust-china-keeps-changing-how-it-counts-coronavirus-
#cases.html
# Another striking feature, along with the sudden surge is the plateauing and saturation of the curve within a week of Feb. 12th.
#Inspite of the sudden jump in the number of confirmed cases China seems to have flattened the curve within a week.  




#Italy: 

#Probably the most interesting countries in the the shifting Pandemic has been with Italy's temporal curve 
#of confirmed cases.

#In the initial phases of the pandemic,the first couple of weeks of Februay, the curve seems to have run 
#parallely with some of the south-asian countrires like Taiwan which seemingly have dealt with the crisis much
#better than some of its western counterparts.

#Around, Feb.21st Italy seems to have gotten the first jolt of the crisis after an increase of more than 60 reported cases
#within a day the country responds quickly by introducing a partial lockdown but strangely enough, even after that and multiple
#more lockdowns the steep rise in the number of confirmed cases never seems to have slowed down.
#Begining March the country had accumulated more than 1000 confirmed cases. 
#Reportedly, the lockdown measure introduced by Italy were one of the most radical comparing it with the lockdown measures of 
#mainlaind China.

#The country seemed to have entered the different phases of transmission under lockdown as well. One of the fathomable 
#reasons that might be attributed to this anamoly could be the festivities and congregations in Italy during that time frame. 
#Cellular phones network data-sets could bring in more insights as to how effective the lockdown turned out to be. 

#Iran: 

#Around March 23rd the curve for Iran seems to have pivoted into a steep rise indicated by a rapid rise in the 
#sum of confirmed cases. 
#This is the time around the Persian new year which is generally celebrated widely across the community. 

