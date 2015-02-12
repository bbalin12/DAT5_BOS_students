# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 17:39:14 2015

@author: megan
"""
from __future__ import division

import numpy
import sqlite3
import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

# Build a preditive model that predicts whether a player was inducted to the Baseball Hall of Fame before 2000 using their batting, pitching, and fielding results- not the number of votes they received. Please make sure to use K-fold cross validaiton and grid search to find your best model. 

conn = sqlite3.connect('/Users/megan/Documents/SQLite/lahman2013.sqlite')
sql = '''
SELECT hof.playerID, hof.yearID, hof.inducted, 
sum(b.AB) as atBats, sum(b.H) as hits,
sum(p.W) as wins, sum(p.L) as losses,
sum(f.PO) as putOuts, sum(f.A) as assists, sum(f.E) as errors
FROM HallOfFame hof
LEFT JOIN Batting b on hof.playerID = b.playerID
LEFT JOIN Pitching p on hof.playerID = p.playerID
LEFT JOIN Fielding f on hof.playerID = f.playerID
WHERE b.yearID <= hof.yearID and hof.yearID<2000
GROUP BY hof.playerID, hof.yearID, hof.inducted
'''
df = pandas.read_sql(sql, conn)
conn.close()
df.dropna(inplace = True)

df['batting_average'] = df.hits / df.atBats
df['winning_percentage'] = df.wins / (df.wins + df.losses)
df['fielding_percentage'] = (df.putOuts + df.assists) / (df.putOuts + df.assists + df.errors)

df = df.replace([np.inf, -np.inf], np.nan)
df.dropna(inplace = True)

response_series = df.inducted 
explanatory_variables = df[['batting_average', 'winning_percentage', 'fielding_percentage']]

knn = KNeighborsClassifier(p = 2)
k_range = range(1, 30, 2)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(explanatory_variables, response_series)

grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(k_range, grid_mean_scores)
best_oob_score = grid.best_score_
grid.best_params_
Knn_optimal = grid.best_estimator_

print best_oob_score* 100

########################################
## VERIFY RESULTS AGAINST HOLD OUT DATA
## (years >= 2000)
########################################

conn = sqlite3.connect('/Users/megan/Documents/SQLite/lahman2013.sqlite')
sql = '''
SELECT hof.playerID, hof.yearID, hof.inducted, 
sum(b.AB) as atBats, sum(b.H) as hits,
sum(p.W) as wins, sum(p.L) as losses,
sum(f.PO) as putOuts, sum(f.A) as assists, sum(f.E) as errors
FROM HallOfFame hof
LEFT JOIN Batting b on hof.playerID = b.playerID
LEFT JOIN Pitching p on hof.playerID = p.playerID
LEFT JOIN Fielding f on hof.playerID = f.playerID
WHERE b.yearID <= hof.yearID and hof.yearID >=2000
GROUP BY hof.playerID, hof.yearID, hof.inducted
'''
df = pandas.read_sql(sql, conn)
conn.close()
df.dropna(inplace = True)

df['batting_average'] = df.hits / df.atBats
df['winning_percentage'] = df.wins / (df.wins + df.losses)
df['fielding_percentage'] = (df.putOuts + df.assists) / (df.putOuts + df.assists + df.errors)

df = df.replace([np.inf, -np.inf], np.nan)
df.dropna(inplace = True)

response_series = df.inducted 
explanatory_variables = df[['batting_average', 'winning_percentage', 'fielding_percentage']]

optimal_knn_preds = Knn_optimal.predict(explanatory_variables)

number_correct = len(response_series[response_series == optimal_knn_preds])
total_in_test_set = len(response_series)
accuracy = number_correct / total_in_test_set

## compare actual accuracy with the accuracy anticipated by our grid search.
print accuracy * 100
print best_oob_score * 100