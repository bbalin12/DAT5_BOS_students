# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 21:11:14 2015

@author: MatthewCohen
"""

from __future__ import division


import numpy
import sqlite3
import pandas
from sklearn.neighbors import KNeighborsClassifier



conn = sqlite3.connect('/Users/MatthewCohen/Documents/SQLite/lahman2013.sqlite')

sql = 'SELECT b.R, b.H, b.RBI, p.SO, p.ERA, h.inducted FROM Batting b LEFT OUTER JOIN Pitching p ON p.playerID = b.playerID LEFT OUTER JOIN HallOfFame h ON h.playerID = p.playerID WHERE b.yearID <2000;'

df = pandas.read_sql(sql, conn)

conn.close()

df.dropna(inplace = True)

response_series = df.inducted
explanatory_variables = df[['R', 'H', 'RBI', 'SO', 'ERA']]

from sklearn.cross_validation import cross_val_score

KNN_classifier = KNeighborsClassifier(n_neighbors=3, p = 2)

scores = cross_val_score(KNN_classifier, explanatory_variables, response_series, cv=10, scoring='accuracy')

print scores

mean_accuracy = numpy.mean(scores) 
print mean_accuracy * 100

from sklearn.grid_search import GridSearchCV
knn = KNeighborsClassifier( p = 2)
k_range = range(1, 30, 2)
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


    
  # TEST ON DIFFERENT DATA  
    
    
conn = sqlite3.connect('/Users/MatthewCohen/Documents/SQLite/lahman2013.sqlite')

sql = 'SELECT b.R, b.H, b.RBI, p.SO, p.ERA, h.inducted FROM Batting b LEFT OUTER JOIN Pitching p ON p.playerID = b.playerID LEFT OUTER JOIN HallOfFame h ON h.playerID = p.playerID WHERE b.yearID >=2000;'

df = pandas.read_sql(sql, conn)

conn.close()

df.dropna(inplace = True)
    
response_series = df.inducted
explanatory_variables = df[['R', 'H', 'RBI', 'SO', 'ERA']]

optimal_knn_preds = Knn_optimal.predict(explanatory_variables)

number_correct = len(response_series[response_series == optimal_knn_preds])
total_in_test_set = len(response_series)
accuracy = number_correct / total_in_test_set

print accuracy * 100
print best_oob_score * 100










