#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


Kaggle

Search kaggle
Competitions
Datasets
Kernels
Discussion
Jobs


Santa's Uncertain Bags

♫ Bells are ringing, children singing, all is merry and bright. Santa's elves made a big mistake, now he needs your help tonight ♫

694 teams · 11 days ago
Overview
Data
Kernels
Discussion
Leaderboard
More
Submit Predictions
Training Data
2 files
sample_submission.cs…
gifts.csv.zip
sample_submission.csv
Download File
File size
67.25 KB
Data Introduction
Santa has 1000 bags to fill to fill with 9 types of gifts. Due to regulations at the North Pole workshop, no bag can contain more than 50 pounds of gifts. If a bag is overweight, it is confiscated by regulators from the North Pole Department of Labor without warning! Even Santa has to worry about throwing out his bad back.

Each present has a fixed weight, but the individual weights are unknown. The weights for each present type are not identical because the elves make them in many types and sizes.

Although the weights were deleted from the database, the elves still have the blueprints for each toy. After some complex volume integrals, the elves managed to give Santa a probability distribution for the weight of each type of toy. To simulate a single gift's weight in pounds, they came up with the following numpy distribution parameters:

horse = max(0, np.random.normal(5,2,1)[0])

ball = max(0, 1 + np.random.normal(1,0.3,1)[0])

bike = max(0, np.random.normal(20,10,1)[0])

train = max(0, np.random.normal(10,5,1)[0])

coal = 47 * np.random.beta(0.5,0.5,1)[0]

book = np.random.chisquare(2,1)[0]

doll = np.random.gamma(5,1,1)[0]

block = np.random.triangular(5,10,20,1)[0]

gloves = 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]
gifts.csv contains the GiftIds which you must sort into Santa's bags. The text of the GiftId contains the type of toy. You do not need to include all GiftIds or all bags when submitting. The evaluation page provides full details on scoring.
© 2017 Kaggle Inc
Our Team Careers Terms Privacy Contact/Support
  

