# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 12:47:56 2015

@author: jchen
"""

# Build a preditive model that predicts whether a player was inducted to the Baseball Hall of Fame
# before 2000 using their batting, pitching, and fielding results- not the number of votes they received. 
# Please make sure to use K-fold cross validaiton and grid search to find your best model.

from __future__ import division
# import packages
import numpy as np
import sqlite3
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt

CROSS_VALIDATION_AMOUNT=.2

# connect to SQLite database
conn = sqlite3.connect('/Users/jchen/Documents/SQLite/lahman2013.sqlite')
# SQL query object
# pull in all relevant 
sql = '''
select m.nameGiven as player_name,
	   h.inducted,
         sum(b.AB) as at_bats,
         sum(b.R) as runs,
         sum(b.H) as hits,
	   sum(b.RBI) as rbi,
	   sum(p.GS) as p_games_started,
	   sum(p.CG) as p_complete_games,
	   sum(p.SHO) as shutouts,
         sum(p.W) as p_wins,
         sum(p.IPOuts) as outs_pitched,
	   sum(f.PO) as putouts,
	   sum(f.A) as assists,
	   sum(f.E) as errors,
         (b.H+b.BB+b.HBP)*1.0/(b.AB+b.BB+b.SF+b.HBP) as OBP,
	   (b.H+b."2B"+(b."3B"*2)+(b.HR*3))*1.0/b.AB as SLG,
         (p.W + p.BB)/(p.IPOuts/3) as WHIP
from HallOfFame h
left join Batting b on h.playerID=b.playerID
left join Pitching p on h.playerID=p.playerID
left join Fielding f on h.playerID=f.playerID
left join Master m on h.playerID=m.playerID  
where h.yearID < 2000
	and h.category='Player'
group by nameGiven, inducted
order by player_name;
'''
# read into data frame
df = pd.read_sql(sql, conn)
# close out connection
conn.close()

# count up null values in each columns
df.isnull().sum() # lots of null values for pitching stats
# drop all-null rows
df.dropna(how='all', inplace=True)

# split up batters and pitchers, since not likely to get into hall of fame on performance of both batting and pitching
# use fielding stats for both

# designate relevant batting and pitching stats
batting_vars=['player_name', 'inducted', 'at_bats', 'runs', 'hits', 'rbi', 'putouts', 'assists', 'errors', 'OBP', 'SLG']
pitching_vars=['player_name', 'inducted', 'p_games_started', 'p_complete_games', 'shutouts', 'p_wins', 'outs_pitched', 'putouts', 'assists', 'errors', 'WHIP']

df_batting = df[pd.notnull(df['at_bats'])][batting_vars]
df_pitching = df[pd.notnull(df['p_games_started'])][pitching_vars]

# check new data frames
df_batting.describe()
df_pitching.describe()

# fill missing values with mean
df_batting.fillna(df_batting.mean(), inplace=True)
df_pitching.fillna(df_pitching.mean(), inplace=True)

# split out response and explanatory variables 
batting_response_series = df_batting.inducted
batting_explanatory_variables = df_batting[batting_vars[2:]] # all other variables

pitching_response_series = df_pitching.inducted
pitching_explanatory_variables = df_pitching[pitching_vars[2:]]

# Let's look at batting first
# break up data into test and train
batting_holdout_num = round(len(df_batting.index) * CROSS_VALIDATION_AMOUNT, 0)
batting_test_indices = np.random.choice(df_batting.index, batting_holdout_num, replace = False)
batting_train_indices = df_batting.index[~df_batting.index.isin(batting_test_indices)]
batting_response_train = batting_response_series.ix[batting_train_indices,]
batting_explanatory_train = batting_explanatory_variables.ix[batting_train_indices,]
batting_response_test = batting_response_series.ix[batting_test_indices,]
batting_explanatory_test = batting_explanatory_variables.ix[batting_test_indices,]

# instantiate KNN classifier, with p=2 for Euclidian distance
batting_knn = KNeighborsClassifier(n_neighbors=3, p = 2)
batting_knn.fit(batting_explanatory_train, batting_response_train) 

batting_predicted_response = batting_knn.predict(batting_explanatory_test)

# calculating accuracy
number_correct = len(batting_response_test[batting_response_test == batting_predicted_response])
total_in_test_set = len(batting_response_test)
accuracy = number_correct / total_in_test_set
print accuracy*100 
# not a stellar accuracy: 81.5%

# repeat for pitching
pitching_holdout_num = round(len(df_pitching.index) * CROSS_VALIDATION_AMOUNT, 0)
pitching_test_indices = np.random.choice(df_pitching.index, pitching_holdout_num, replace = False)
pitching_train_indices = df_pitching.index[~df_pitching.index.isin(batting_test_indices)]
pitching_response_train = pitching_response_series.ix[pitching_train_indices,]
pitching_explanatory_train = pitching_explanatory_variables.ix[pitching_train_indices,]
pitching_response_test = pitching_response_series.ix[pitching_test_indices,]
pitching_explanatory_test = pitching_explanatory_variables.ix[pitching_test_indices,]

# instantiate KNN classifier, with p=2 for Euclidian distance
pitching_knn = KNeighborsClassifier(n_neighbors=3, p = 2)
pitching_knn.fit(pitching_explanatory_train, pitching_response_train) 

pitching_predicted_response = pitching_knn.predict(pitching_explanatory_test)

number_correct = len(pitching_response_test[pitching_response_test == pitching_predicted_response])
total_in_test_set = len(pitching_response_test)
accuracy = number_correct / total_in_test_set
print accuracy*100
# roughly the same, at 79%

###########################
# K-fold CV
###########################

# reinstantiate batting classifier
batting_knn = KNeighborsClassifier(n_neighbors=3, p = 2) 
# compute scores
batting_scores = cross_val_score(batting_knn, batting_explanatory_variables, batting_response_series, cv=10, scoring='accuracy')
print batting_scores

batting_mean_accuracy = np.mean(batting_scores) 
print batting_mean_accuracy*100

# pitching
pitching_knn = KNeighborsClassifier(n_neighbors=3, p = 2) 
# compute scores
pitching_scores = cross_val_score(pitching_knn, pitching_explanatory_variables, pitching_response_series, cv=10, scoring='accuracy')
print pitching_scores

pitching_mean_accuracy = np.mean(pitching_scores) 
print pitching_mean_accuracy*100
# this time pitching is lower

############################
# Grid search for optimal k
############################

# instatiate classifier for batting
batting_knn = KNeighborsClassifier(p = 2)
batting_k_range = range(1, 60, 2)
batting_param_grid = dict(n_neighbors=batting_k_range)
batting_grid = GridSearchCV(batting_knn, batting_param_grid, cv=10, scoring='accuracy')
batting_grid.fit(batting_explanatory_variables, batting_response_series)

# get optimal estimator for batting
batting_grid_scores = batting_grid.grid_scores_
batting_grid_mean_scores = [result[1] for result in batting_grid_scores]
plt.figure()
plt.plot(batting_k_range, batting_grid_mean_scores)

batting_best_oob_score = batting_grid.best_score_
print batting_grid.best_params_ # best k-value at 27 - that's pretty high. might be overfitting here
print batting_best_oob_score
batting_knn_opt = batting_grid.best_estimator_

# repeat for pitching
pitching_knn = KNeighborsClassifier(p = 2)
pitching_k_range = range(1, 60, 2)
pitching_param_grid = dict(n_neighbors=pitching_k_range)
pitching_grid = GridSearchCV(pitching_knn, pitching_param_grid, cv=10, scoring='accuracy')
pitching_grid.fit(pitching_explanatory_variables, pitching_response_series)

# get optimal estimator for pitching
pitching_grid_scores = pitching_grid.grid_scores_
pitching_grid_mean_scores = [result[1] for result in pitching_grid_scores]
plt.figure()
plt.plot(pitching_k_range, pitching_grid_mean_scores)

pitching_best_oob_score = pitching_grid.best_score_
print pitching_grid.best_params_ # best k-value at 23
print pitching_best_oob_score
pitching_knn_opt = pitching_grid.best_estimator_


##############################
# Test model on post-2005 data
##############################

conn = sqlite3.connect('/Users/jchen/Documents/SQLite/lahman2013.sqlite')
sql = '''
select m.nameGiven as player_name,
	   h.inducted,
         sum(b.AB) as at_bats,
         sum(b.R) as runs,
         sum(b.H) as hits,
	   sum(b.RBI) as rbi,
	   sum(p.GS) as p_games_started,
	   sum(p.CG) as p_complete_games,
	   sum(p.SHO) as shutouts,
         sum(p.W) as p_wins,
         sum(p.IPOuts) as outs_pitched,
	   sum(f.PO) as putouts,
	   sum(f.A) as assists,
	   sum(f.E) as errors,
         (b.H+b.BB+b.HBP)*1.0/(b.AB+b.BB+b.SF+b.HBP) as OBP,
	   (b.H+b."2B"+(b."3B"*2)+(b.HR*3))*1.0/b.AB as SLG,
         (p.W + p.BB)/(p.IPOuts/3) as WHIP
from HallOfFame h
left join Batting b on h.playerID=b.playerID
left join Pitching p on h.playerID=p.playerID
left join Fielding f on h.playerID=f.playerID
left join Master m on h.playerID=m.playerID  
where h.yearID >= 2000
	and h.category='Player'
group by nameGiven, inducted
order by player_name;
'''
df = pd.read_sql(sql, conn)
conn.close()

df_batting = df[pd.notnull(df['at_bats'])][batting_vars]
df_pitching = df[pd.notnull(df['p_games_started'])][pitching_vars]

# fill missing values with mean
df_batting.fillna(df_batting.mean(), inplace=True)
df_pitching.fillna(df_pitching.mean(), inplace=True)

# set response and explanatory data
batting_response_series = df_batting.inducted
batting_explanatory_variables = df_batting[batting_vars[2:]] # all other variables

pitching_response_series = df_pitching.inducted
pitching_explanatory_variables = df_pitching[pitching_vars[2:]]

# predict batting
batting_opt_knn_preds = batting_knn_opt.predict(batting_explanatory_variables)

batting_number_correct = len(batting_response_series[batting_response_series == batting_opt_knn_preds])
batting_total_in_test_set = len(batting_response_series)
batting_accuracy = batting_number_correct / batting_total_in_test_set

## compare actual accuracy with accuracy anticipated by grid search.
print batting_accuracy* 100
print batting_best_oob_score* 100
# interestingly enough, higher accuracy on new data

# do the same with pitching
pitching_opt_knn_preds = pitching_knn_opt.predict(pitching_explanatory_variables)

pitching_number_correct = len(pitching_response_series[pitching_response_series == pitching_opt_knn_preds])
pitching_total_in_test_set = len(pitching_response_series)
pitching_accuracy = pitching_number_correct / pitching_total_in_test_set

print pitching_accuracy* 100
print pitching_best_oob_score * 100
# way higher accuracy on out of sample data
# perhaps best taken with a grain of salt

# could be that the metrics used are more influential in determining hall of fame chances in modern times compared with previous decades
# (or could be chance)
