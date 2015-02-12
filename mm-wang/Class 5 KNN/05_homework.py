# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 23:49:46 2015

@author: Margaret
"""


# importing division from the 'future' release of Python (i.e. Python 3)
from __future__ import division

# importing numpy and the KNN content in scikit-learn along with SQLite + pandas
import numpy
import sqlite3
import pandas
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# connect to the baseball database. I'm passing the full path to the SQLite file
conn = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')
# creating an objected constraining a string that has the SQL query
sql = """ 
SELECT h.playerID, h.inducted as inducted, bat_runs, bat_hits, at_bats, bat_homeruns, bat_strikes, bat_stolen,
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, f_putouts, f_assists, f_errors FROM HallofFame h
INNER JOIN
(SELECT f.playerID, f.PO as f_putouts, f.A as f_assists, f.E as f_errors, bat_runs, bat_hits, at_bats, bat_homeruns,
bat_strikes, bat_stolen, pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA FROM Fielding f
LEFT JOIN
(SELECT b.playerID, sum(b.R) as bat_runs, sum(b.H) as bat_hits, sum(b.AB) as at_bats, sum(b.HR) as bat_homeruns,
sum(b.SO) as bat_strikes, sum(b.SB) as bat_stolen,
sum(p.W) as pitch_wins, sum(p.SO) as pitch_strikes, sum(p.SHO) as pitch_shuts, avg(1/p.ERA) as pitch_ERA
FROM Batting b
LEFT JOIN Pitching p on p.playerID = b.playerID
GROUP BY b.playerID) batpitch on batpitch.playerID = f.playerID
GROUP BY batpitch.playerID) positions
ON positions.playerID = h.playerID
WHERE h.yearID < 2000
"""

# passing the connectiona nd the SQL string to pandas.read_sql
df = pandas.read_sql(sql, conn)

# closing the connection
conn.close()


df.fillna(value = 0, inplace = True)

## batting - could be home runs, hits/at bats (batting average), lack of strikes, stolen bases
df['bat_avg'] = df.bat_hits/df.at_bats

## fielding - fielding percentage as (PO + A)/(PO + A + E)
## taken from Wikipedia and http://www.csgnetwork.com/baseballdefensestatsformulae.html
df['f_perc'] = (df.f_putouts+df.f_assists)/(df.f_putouts+df.f_assists+df.f_errors)

df.fillna(value = 0, inplace = True)

# predicting inductions
response_series = df.inducted
# using all the variables and derived variables
explanatory_variables = df[['bat_runs','bat_homeruns','bat_hits','at_bats','bat_avg','bat_strikes','bat_stolen',
'pitch_wins','pitch_strikes','pitch_shuts','pitch_ERA','f_putouts','f_assists','f_errors','f_perc']]

# use 10-fold cross validation to score model
from sklearn.cross_validation import cross_val_score
# we need to reinstantiate the model
KNN_classifier = KNeighborsClassifier(n_neighbors=9,p=2)
# putting in the entire dataset
scores = cross_val_score(KNN_classifier, explanatory_variables,response_series,cv=10,scoring='accuracy')

# automatic grid search for an optimal value of K
from sklearn.grid_search import GridSearchCV
knn = KNeighborsClassifier(p=2)
k_range = range(1,30,2)
param_grid = dict(n_neighbors = k_range) #dictionary that links number of neighbors and the k range)
grid = GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')
# there is a KNN model, that is cross validated 10 times over the parameter grid, and then base it off accuracy
grid.fit(explanatory_variables,response_series)

grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_] #0 is the key, 1 is the first value
# params is the key
plt.figure()
plt.plot(k_range, grid_mean_scores)
best_oob_score = grid.best_score_
grid.best_params_
knn_optimal = grid.best_estimator_

# average accuracy of scores
mean_accuracy = numpy.mean(scores)
print mean_accuracy*100



# connect to the baseball database. I'm passing the full path to the SQLite file
conn = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')
# creating an objected constraining a string that has the SQL query
sql = """ 
SELECT h.playerID, h.inducted as inducted, bat_runs, bat_hits, at_bats, bat_homeruns, bat_strikes, bat_stolen,
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, f_putouts, f_assists, f_errors FROM HallofFame h
INNER JOIN
(SELECT f.playerID, f.PO as f_putouts, f.A as f_assists, f.E as f_errors, bat_runs, bat_hits, at_bats, bat_homeruns,
bat_strikes, bat_stolen, pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA FROM Fielding f
LEFT JOIN
(SELECT b.playerID, sum(b.R) as bat_runs, sum(b.H) as bat_hits, sum(b.AB) as at_bats, sum(b.HR) as bat_homeruns,
sum(b.SO) as bat_strikes, sum(b.SB) as bat_stolen,
sum(p.W) as pitch_wins, sum(p.SO) as pitch_strikes, sum(p.SHO) as pitch_shuts, avg(1/p.ERA) as pitch_ERA
FROM Batting b
LEFT JOIN Pitching p on p.playerID = b.playerID
GROUP BY b.playerID) batpitch on batpitch.playerID = f.playerID
GROUP BY batpitch.playerID) positions
ON positions.playerID = h.playerID
WHERE h.yearID >= 2000
"""

# passing the connectiona nd the SQL string to pandas.read_sql
df = pandas.read_sql(sql, conn)

# closing the connection
conn.close()

df.fillna(value = 0, inplace = True)

## batting - could be home runs, hits/at bats (batting average), lack of strikes, stolen bases
df['bat_avg'] = df.bat_hits/df.at_bats

## fielding - fielding percentage as (PO + A)/(PO + A + E)
## taken from Wikipedia and http://www.csgnetwork.com/baseballdefensestatsformulae.html
df['f_perc'] = (df.f_putouts+df.f_assists)/(df.f_putouts+df.f_assists+df.f_errors)

df.fillna(value = 0, inplace = True)

# predicting inductions
response_series = df.inducted
# using all the variables and derived variables
explanatory_variables = df[['bat_runs','bat_homeruns','bat_hits','at_bats','bat_avg','bat_strikes','bat_stolen',
'pitch_wins','pitch_strikes','pitch_shuts','pitch_ERA','f_putouts','f_assists','f_errors','f_perc']]

optimal_knn_preds = knn_optimal.predict(explanatory_variables)

number_correct = len(response_series[response_series==optimal_knn_preds])
total_in_test_set = len(response_series)
accuracy = number_correct / total_in_test_set

#compare actual accuracy with accuracy anticipated by grid search
print accuracy * 100
print best_oob_score
