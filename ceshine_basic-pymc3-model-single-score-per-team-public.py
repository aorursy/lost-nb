#!/usr/bin/env python
# coding: utf-8



import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import pandas as pd
import numpy as np
import theano.tensor as tt
import matplotlib.pyplot as plt
import pymc3
from sklearn.preprocessing import LabelEncoder




df_regular = pd.read_csv("../input/RegularSeasonCompactResults.csv")
df_tourney = pd.read_csv("../input/NCAATourneyCompactResults.csv")
df_seeds = pd.read_csv("../input/NCAATourneySeeds.csv")




df_tourney["WLoc"].unique()




df_regular = df_regular[df_regular.Season >= 2014].copy().reset_index(drop=True)
df_regular.DayNum.describe()




df_regular.head()




df_regular["HomeScore"] = df_regular.apply(
    lambda x: x["WScore"] if x["WLoc"] == "H" or x["WLoc"] == "N" else x["LScore"], axis=1)
df_regular["AwayScore"] = df_regular.apply(
    lambda x: x["LScore"] if x["WLoc"] == "H" or x["WLoc"] == "N" else x["WScore"], axis=1)
df_regular["HomeTeam"] = df_regular.apply(
    lambda x: x["WTeamID"] if x["WLoc"] == "H" or x["WLoc"] == "N" else x["LTeamID"], axis=1)
df_regular["AwayTeam"] = df_regular.apply(
    lambda x: x["LTeamID"] if x["WLoc"] == "H" or x["WLoc"] == "N" else x["WTeamID"], axis=1)
assert all(df_regular["HomeTeam"] != df_regular["AwayTeam"])
assert all(df_regular["HomeScore"] != df_regular["AwayScore"])
df_regular["ScoreDiff"] = df_regular["HomeScore"] - df_regular["AwayScore"]




teams = sorted(list(set(df_regular["HomeTeam"]) | set(df_regular["AwayTeam"])))
team_encoder = LabelEncoder()
team_encoder.fit(teams)
df_regular["HomeTeamID"] = team_encoder.transform(df_regular["HomeTeam"])
df_regular["AwayTeamID"] = team_encoder.transform(df_regular["AwayTeam"])
df_regular["SeasonID"] = df_regular["Season"] - 2014
df_regular.head()




advantage = (df_regular["WLoc"] != "N").astype("int")
num_teams = len(team_encoder.classes_)
num_games = df_regular.shape[0]
num_teams, num_games




model = pymc3.Model()
with model:
    # global model parameters
    home = pymc3.Flat('home')
    sd_scores = pymc3.HalfStudentT('sd_scores', nu=10, sd=10)
    sd_points_diff = pymc3.HalfStudentT('sd_diff', nu=20, sd=10)
    
    # team-specific model parameters
    scores_ = pymc3.Normal(
        "scores_raw", mu=0, sd=sd_scores, shape=(5, num_teams))
    scores = pymc3.Deterministic('scores', scores_ - tt.mean(scores_))
    
    points_diff = pymc3.Normal('points_diff', 
        home * advantage + 
        scores[df_regular["SeasonID"].values, df_regular["HomeTeamID"].values] - 
        scores[df_regular["SeasonID"].values, df_regular["AwayTeamID"].values],
        sd=sd_points_diff,
        observed=df_regular["ScoreDiff"].values)




with model:
    trace = pymc3.sample(5000, tune=2000, njobs=4, chains=4)




pymc3.traceplot(trace)
plt.show()




def calculate_winning_probability(trace, season=0, team_1=0, team_2=1, sample_size=100):
    draw = np.random.randint(0, trace['scores'].shape[0], size=sample_size)
    scores_ = trace['scores'][draw]
    # Take a short cut
    sd_diff_ = np.mean(trace['sd_diff'][draw])
    noise = np.random.normal(
            loc=0,
            scale=sd_diff_,
            size=sample_size
    )
    points_diff = scores_[:, season, team_1] - scores_[:, season, team_2] + noise
    wins = points_diff > 0
    return (
        np.mean(wins), np.percentile(points_diff, 5), 
        np.percentile(points_diff, 50), np.percentile(points_diff, 95)
    )
calculate_winning_probability(trace, 3, 0, 3, 1000)




calculate_winning_probability(trace, 3, 2, 10, 1000)




df_tourney = df_tourney[df_tourney.Season >= 2014].copy().reset_index(drop=True)
df_tourney["WTeamID"] = team_encoder.transform(df_tourney.WTeamID)
df_tourney["LTeamID"] = team_encoder.transform(df_tourney.LTeamID)
df_tourney["SeasonID"] = df_tourney["Season"] - 2014




df_tourney.head()




calculate_winning_probability(trace, 0, 192, 348, 1000)




calculate_winning_probability(trace, 0, 6, 71, 1000)




get_ipython().run_line_magic('time', 'pred = df_tourney.apply(     lambda x: calculate_winning_probability(         trace, x["SeasonID"], x["WTeamID"], x["LTeamID"], 5000), axis=1)')




df_tourney["Pred"] = [x[0] for x in pred]
df_tourney["low"] = [x[1] for x in pred]
df_tourney["med"] = [x[2] for x in pred]
df_tourney["high"] = [x[3] for x in pred]




_ = plt.hist(df_tourney["Pred"], bins=20)
plt.show()




df_tourney["diff"] = df_tourney["WScore"] - df_tourney["LScore"]
print("Percentage of diff inside 90% confidence interval:", 
      sum((df_tourney["diff"] >= df_tourney["low"]) & (df_tourney["diff"] <= df_tourney["high"])) * 100 / df_tourney.shape[0])




sum(df_tourney[df_tourney.Season==2017]["Pred"] > 0.5) / df_tourney[df_tourney.Season==2017].shape[0]




sum(df_tourney["Pred"] > 0.5) / df_tourney.shape[0]




np.mean(-np.log(df_tourney[df_tourney.Season==2017]["Pred"].values))




np.mean(-np.log(df_tourney["Pred"].values))











