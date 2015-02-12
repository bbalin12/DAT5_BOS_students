# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:16:44 2015

@author: Haruo M
"""
from __future__ import division

import numpy
import sqlite3
import pandas
from sklearn.neighbors import KNeighborsClassifier

# the percent to hold out for the cross-validation.
CROSS_VALIDATION_AMOUNT = .2

conn = sqlite3.connect('C:\Users\mizutani\Documents\SQLite\lahman2013.sqlite')

# a SQL query that takes data from the two tables, HallofFame and Batting are combined
sql = """SELECT h.playerID, h.yearID, h.inducted, SUM(b.G_batting) AS total_games_batter, SUM(b.AB) AS total_at_bats, SUM(b.H) AS total_hits, SUM(b.HR) AS total_homeruns, SUM(b.SB) AS total_stolen_base
FROM HallofFame h
LEFT JOIN Batting b ON h.playerID = b.playerID
WHERE h.yearID <2000
GROUP BY h.playerID
ORDER BY h.yearID ASC;"""

df = pandas.read_sql(sql, conn)
conn.close()

# Batting averages are calurated and added to the dataframe
df['batting_average'] = df.total_hits / df.total_at_bats
df.dropna(inplace = True)

df.head()

response_series = df.inducted 
explanatory_variables = df[['total_games_batter','total_at_bats', 'total_hits', 'total_homeruns', 'total_stolen_base', 'batting_average']]

holdout_num = round(len(df.index) * CROSS_VALIDATION_AMOUNT, 0)

test_indices = numpy.random.choice(df.index, holdout_num, replace = False )
train_indices = df.index[~df.index.isin(test_indices)]

response_train = response_series.ix[train_indices,]
explanatory_train = explanatory_variables.ix[train_indices,]

response_test = response_series.ix[test_indices,]
explanatory_test = explanatory_variables.ix[test_indices,]

# KNN with k = 3, p = 2
KNN_classifier = KNeighborsClassifier(n_neighbors=3, p = 2)
KNN_classifier.fit(explanatory_train, response_train) 
predicted_response = KNN_classifier.predict(explanatory_test)

# Calculating accuracy
number_correct = len(response_test[response_test == predicted_response])
total_in_test_set = len(response_test)
accuracy = number_correct / total_in_test_set
print accuracy* 100


# 10-fold cross-validation to score the model(k = 3). 
from sklearn.cross_validation import cross_val_score
KNN_classifier = KNeighborsClassifier(n_neighbors=3, p = 2)
scores = cross_val_score(KNN_classifier, explanatory_variables, response_series, cv=10, scoring='accuracy')
print scores
mean_accuracy = numpy.mean(scores) 
print mean_accuracy * 100
print accuracy* 100


# tune the model for the optimal number of K, whose range is 1 to 100
k_range = range(1, 100, 2)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k,  p = 2)
    scores.append(numpy.mean(cross_val_score(knn, explanatory_variables, response_series, cv=10, scoring='accuracy')))
import matplotlib.pyplot as plt
plt.figure()
plt.plot(k_range, scores)

# Grid search
from sklearn.grid_search import GridSearchCV
knn = KNeighborsClassifier(p = 2)

param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(explanatory_variables, response_series)

import matplotlib.pyplot as plt
grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(k_range, grid_mean_scores)
best_oob_score = grid.best_score_
grid.best_params_
Knn_optimal = grid.best_estimator_

optimal_knn_preds = Knn_optimal.predict(explanatory_variables)

number_correct = len(response_series[response_series == optimal_knn_preds])
total_in_test_set = len(response_series)
accuracy = number_correct / total_in_test_set

print accuracy * 100
print best_oob_score *100

# Best optimized K-fold cross-validation to score our model. 
from sklearn.cross_validation import cross_val_score
KNN_classifier = KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors']
, p = 2)
scores = cross_val_score(KNN_classifier, explanatory_variables, response_series, cv=10, scoring='accuracy')
print scores
mean_accuracy = numpy.mean(scores) 
print mean_accuracy * 100
print accuracy* 100


