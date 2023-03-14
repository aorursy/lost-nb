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
    sd_att = pymc3.HalfStudentT('sd_att', nu=3, sd=2.5)
    sd_def = pymc3.HalfStudentT('sd_def', nu=3, sd=2.5)
    intercept = pymc3.Flat('intercept')
    
    # team-specific model parameters
    offs_star = pymc3.Normal('offs_star', mu=0, sd=sd_att, shape=(5, num_teams))
    defs_star = pymc3.Normal('defs_star', mu=0, sd=sd_def, shape=(5, num_teams))
    offs = pymc3.Deterministic('offs', offs_star - tt.mean(offs_star))
    defs = pymc3.Deterministic('defs', defs_star - tt.mean(defs_star))
    
    # derive the scoring intensity for a game
    home_theta = tt.exp(
        intercept + home * advantage + 
        offs[df_regular["SeasonID"].values, df_regular["HomeTeamID"].values] + 
        defs[df_regular["SeasonID"].values, df_regular["AwayTeamID"].values])
    away_theta = tt.exp(
        intercept + 
        offs[df_regular["SeasonID"].values, df_regular["AwayTeamID"].values] + 
        defs[df_regular["SeasonID"].values, df_regular["HomeTeamID"].values])
    
    # likelihood of observed data
    home_points = pymc3.Poisson('home_points', mu=home_theta, observed=df_regular["HomeScore"].values)
    away_points = pymc3.Poisson('away_points', mu=away_theta, observed=df_regular["AwayScore"].values)




with model:
    trace = pymc3.sample(2000, tune=1000)




pymc3.traceplot(trace)
plt.show()




team_mean_offs = np.mean(trace['offs'], axis=0)
team_mean_defs = np.mean(trace['defs'], axis=0)
def calculate_winning_probability(trace, season=0, team_1=0, team_2=1, sample_size=100):
    draw = np.random.randint(0, trace['intercept'].shape[0], size=sample_size)
    intercept_ = trace['intercept'][draw]
    offs_ = trace['offs'][draw]
    defs_ = trace['defs'][draw]
    home_theta_ = np.exp(intercept_ + offs_[:, season,  team_1] + defs_[:, season,  team_2])
    away_theta_ = np.exp(intercept_ + offs_[:, season,  team_2] + defs_[:, season,  team_1])
    home_score_ = np.random.poisson(home_theta_, sample_size)
    away_score_ = np.random.poisson(away_theta_, sample_size)   
    wins = np.mean((home_score_ - away_score_ > 0))
    return (
        wins,
        (np.percentile(home_score_, 5), np.percentile(home_score_, 50), np.percentile(home_score_, 95)), 
        (np.percentile(away_score_, 5), np.percentile(away_score_, 50), np.percentile(away_score_, 95)), 
        (team_mean_offs[season, team_1], team_mean_defs[season, team_1]), 
        (team_mean_offs[season, team_2], team_mean_defs[season, team_2])
    )
calculate_winning_probability(trace, 3, 0, 1, 5000)




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
df_tourney["wscore_low"] = [x[1][0] for x in pred]
df_tourney["wscore_med"] = [x[1][1] for x in pred]
df_tourney["wscore_high"] = [x[1][2] for x in pred]
df_tourney["lscore_low"] = [x[2][0] for x in pred]
df_tourney["lscore_med"] = [x[2][1] for x in pred]
df_tourney["lscore_high"] = [x[2][2] for x in pred]




_ = plt.hist(df_tourney["Pred"], bins=20)
plt.show()




print("Percentage of WScore inside 90% confidence interval:", 
      sum((df_tourney["WScore"] >= df_tourney["wscore_low"]) & (df_tourney["WScore"] <= df_tourney["wscore_high"])) * 100 / df_tourney.shape[0])




print("Percentage of LScore inside 90% confidence interval:", 
      sum((df_tourney["LScore"] >= df_tourney["lscore_low"]) & (df_tourney["LScore"] <= df_tourney["lscore_high"])) * 100 / df_tourney.shape[0])




sum(df_tourney[df_tourney.Season==2017]["Pred"] > 0.5) / df_tourney[df_tourney.Season==2017].shape[0]




sum(df_tourney["Pred"] > 0.5) / df_tourney.shape[0]




np.mean(-np.log(df_tourney[df_tourney.Season==2017]["Pred"].values))




np.mean(-np.log(df_tourney["Pred"].values))











