# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 15:26:00 2015

@author: YihanLi
"""

from __future__ import division

import numpy
import sqlite3
import pandas
from sklearn.neighbors import KNeighborsClassifier

CROSS_VALIDATION_AMOUNT = .2




##Batting
conn = sqlite3.connect('/Users/YihanLi/Documents/SQLite/lahman2013.sqlite')

sql = '''select a.playerID, a.inducted, sum(b.G_batting) as game_as_batter, sum(b.AB) as At_bats, sum(b.HR) as homeruns, sum(b.R) as runs, sum(b.H) as hits, sum(b.BB) as baseonballs, sum(b.SO) as strikeouts from HallofFame a
join Batting b on a.playerID=b.playerID
where a.yearid<2000
group by 1;'''

df = pandas.read_sql(sql,conn)

conn.close()

df.dropna(inplace=True)
df.head()
df['AB_per_HR'] = df.At_bats/df.homeruns
df['BB_per_SO'] = df.baseonballs/df.strikeouts
df['Batting_average'] = df.hits/df.At_bats
df['game_per_AB'] = df.game_as_batter/df.At_bats
df['run_per_hit'] = df.runs/df.hits

df.replace(numpy.inf, numpy.nan)
df = df[numpy.isfinite(df['AB_per_HR'])]
df = df[numpy.isfinite(df['BB_per_SO'])]
df = df[numpy.isfinite(df['Batting_average'])]
df.dropna(inplace=True)
df.head()

response_series = df.inducted
explanatory_variables = df[['AB_per_HR', 'BB_per_SO', 'Batting_average', 'At_bats', 'homeruns', 'game_as_batter', 'run_per_hit', 'runs', 'hits']]


holdout_num = round(len(df.index)*CROSS_VALIDATION_AMOUNT, 0)

#Way1
test_indices = numpy.random.choice(df.index,holdout_num, replace=False)
train_indices = df.index[~df.index.isin(test_indices)]

response_train = response_series.ix[train_indices,]
explanatory_train = explanatory_variables.ix[train_indices,]

response_test = response_series.ix[test_indices,]
explanatory_test = explanatory_variables.ix[test_indices,]

KNN_classifier = KNeighborsClassifier(n_neighbors=7, p=2)
KNN_classifier.fit(explanatory_train, response_train)

predicted_response = KNN_classifier.predict(explanatory_test)

number_correct = len(response_test[response_test==predicted_response])
total_in_test_set = len(response_test)
accuracy = number_correct / total_in_test_set
print accuracy*100



from sklearn.cross_validation import cross_val_score
KNN_classifier = KNeighborsClassifier(n_neighbors=7, p=2)
scores = cross_val_score(KNN_classifier, explanatory_variables, response_series, cv=10, scoring='accuracy')

print scores

mean_accuracy = numpy.mean(scores)
print mean_accuracy * 100

print accuracy * 100

k_range = range(1, 30, 2)
scores=[]

from sklearn.grid_search import GridSearchCV
knn = KNeighborsClassifier(p=2)
k_range = range(1,30,2)
param_grid = dict(n_neighbors = k_range)
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(explanatory_variables, response_series)

grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(k_range, grid_mean_scores)
best_oob_score = grid.best_score_
grid.best_params_
knn_optimal = grid.best_estimator_
print knn_optimal



conn = sqlite3.connect('/Users/YihanLi/Documents/SQLite/lahman2013.sqlite')

sql = '''select a.playerID, a.inducted, sum(b.G_batting) as game_as_batter, sum(b.AB) as At_bats, sum(b.HR) as homeruns, sum(b.R) as runs, sum(b.H) as hits, sum(b.BB) as baseonballs, sum(b.SO) as strikeouts from HallofFame a
join Batting b on a.playerID=b.playerID
where a.yearid>=2000
group by 1;'''

df = pandas.read_sql(sql,conn)

conn.close()

df.dropna(inplace=True)

df.head()
df['AB_per_HR'] = df.At_bats/df.homeruns
df['BB_per_SO'] = df.baseonballs/df.strikeouts
df['Batting_average'] = df.hits/df.At_bats
df['game_per_AB'] = df.game_as_batter/df.At_bats
df['run_per_hit'] = df.runs/df.hits

df.replace(numpy.inf, numpy.nan)
df = df[numpy.isfinite(df['AB_per_HR'])]
df = df[numpy.isfinite(df['BB_per_SO'])]
df = df[numpy.isfinite(df['Batting_average'])]

response_series = df.inducted
explanatory_variables = df[['AB_per_HR', 'BB_per_SO', 'Batting_average', 'At_bats', 'homeruns', 'game_as_batter', 'run_per_hit', 'runs', 'hits']]


optimal_knn_preds = knn_optimal.predict(explanatory_variables)


number_correct = len(response_series[response_series == optimal_knn_preds])
total_in_test_set = len(response_series)
accuracy = number_correct / total_in_test_set
print accuracy*100


##Pitching 
conn = sqlite3.connect('/Users/YihanLi/Documents/SQLite/lahman2013.sqlite')

sql = '''select a.playerID, a.inducted,sum(b.W) as wins, sum(b.L) as loses, sum(b.G) as games, sum(b.SHO) as SHO, sum(b.ERA) as ERA, sum(b.GF) as GF, sum(b.WP) as wildpitch from HallofFame a
join Pitching b on a.playerID=b.playerID
where a.yearid<2000
group by 1;'''

df = pandas.read_sql(sql,conn)

conn.close()

df.dropna(inplace=True)
df.head()
df['win_average'] = df.wins/(df.wins+df.loses)
df['GF_per_game'] = df.GF/df.games
df['SHO_per_game'] = df.SHO/df.games

df.replace(numpy.inf, numpy.nan)
df = df[numpy.isfinite(df['win_average'])]
df = df[numpy.isfinite(df['GF_per_game'])]
df = df[numpy.isfinite(df['SHO_per_game'])]
df.dropna(inplace=True)
df.head()

response_series = df.inducted
explanatory_variables = df[['win_average', 'GF_per_game', 'SHO_per_game', 'ERA', 'wildpitch', 'games']]


holdout_num = round(len(df.index)*CROSS_VALIDATION_AMOUNT, 0)

#Way1
test_indices = numpy.random.choice(df.index,holdout_num, replace=False)
train_indices = df.index[~df.index.isin(test_indices)]

response_train = response_series.ix[train_indices,]
explanatory_train = explanatory_variables.ix[train_indices,]

response_test = response_series.ix[test_indices,]
explanatory_test = explanatory_variables.ix[test_indices,]

KNN_classifier = KNeighborsClassifier(n_neighbors=5, p=2)
KNN_classifier.fit(explanatory_train, response_train)

predicted_response = KNN_classifier.predict(explanatory_test)

number_correct = len(response_test[response_test==predicted_response])
total_in_test_set = len(response_test)
accuracy = number_correct / total_in_test_set
print accuracy*100



from sklearn.cross_validation import cross_val_score
KNN_classifier = KNeighborsClassifier(n_neighbors=7, p=2)
scores = cross_val_score(KNN_classifier, explanatory_variables, response_series, cv=10, scoring='accuracy')

print scores

mean_accuracy = numpy.mean(scores)
print mean_accuracy * 100

print accuracy * 100

k_range = range(1, 30, 2)
scores=[]

from sklearn.grid_search import GridSearchCV
knn = KNeighborsClassifier(p=2)
k_range = range(1,30,2)
param_grid = dict(n_neighbors = k_range)
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(explanatory_variables, response_series)

grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(k_range, grid_mean_scores)
best_oob_score = grid.best_score_
grid.best_params_
knn_optimal = grid.best_estimator_
print knn_optimal



conn = sqlite3.connect('/Users/YihanLi/Documents/SQLite/lahman2013.sqlite')

sql = '''select a.playerID, a.inducted,sum(b.W) as wins, sum(b.L) as loses, sum(b.G) as games, sum(b.SHO) as SHO, sum(b.ERA) as ERA, sum(b.GF) as GF, sum(b.WP) as wildpitch from HallofFame a
join Pitching b on a.playerID=b.playerID
where a.yearid>=2000
group by 1;'''

df = pandas.read_sql(sql,conn)

conn.close()

df.dropna(inplace=True)

df.head()
df['win_average'] = df.wins/(df.wins+df.loses)
df['GF_per_game'] = df.GF/df.games
df['SHO_per_game'] = df.SHO/df.games

df.replace(numpy.inf, numpy.nan)
df = df[numpy.isfinite(df['win_average'])]
df = df[numpy.isfinite(df['GF_per_game'])]
df = df[numpy.isfinite(df['SHO_per_game'])]
df.dropna(inplace=True)
df.head()
response_series = df.inducted
explanatory_variables = df[['win_average', 'GF_per_game', 'SHO_per_game', 'ERA', 'wildpitch', 'games']]

optimal_knn_preds = knn_optimal.predict(explanatory_variables)


number_correct = len(response_series[response_series == optimal_knn_preds])
total_in_test_set = len(response_series)
accuracy = number_correct / total_in_test_set
print accuracy*100


##Fielding
conn = sqlite3.connect('/Users/YihanLi/Documents/SQLite/lahman2013.sqlite')

sql = '''select a.playerID,a.inducted, sum(b.PO) as putouts, sum(b.A) as assists, sum(b.G) as games, sum(b.DP) as doubleplay, sum(b.E) as errors from HallofFame a
join Fielding b on a.playerID=b.playerID
where a.yearid<2000
group by 1;'''

df = pandas.read_sql(sql,conn)

conn.close()

df.dropna(inplace=True)
df.head()
df['DP_per_PO'] = df.doubleplay/(df.putouts+df.assists)
df['error_per_game'] = df.errors/df.games

df.replace(numpy.inf, numpy.nan)
df = df[numpy.isfinite(df['DP_per_PO'])]
df = df[numpy.isfinite(df['error_per_game'])]
df.dropna(inplace=True)
df.head()

response_series = df.inducted
explanatory_variables = df[['DP_per_PO', 'error_per_game', 'games', 'doubleplay', 'putouts', 'assists']]


holdout_num = round(len(df.index)*CROSS_VALIDATION_AMOUNT, 0)

#Way1
test_indices = numpy.random.choice(df.index,holdout_num, replace=False)
train_indices = df.index[~df.index.isin(test_indices)]

response_train = response_series.ix[train_indices,]
explanatory_train = explanatory_variables.ix[train_indices,]

response_test = response_series.ix[test_indices,]
explanatory_test = explanatory_variables.ix[test_indices,]

KNN_classifier = KNeighborsClassifier(n_neighbors=17, p=2)
KNN_classifier.fit(explanatory_train, response_train)

predicted_response = KNN_classifier.predict(explanatory_test)

number_correct = len(response_test[response_test==predicted_response])
total_in_test_set = len(response_test)
accuracy = number_correct / total_in_test_set
print accuracy*100



from sklearn.cross_validation import cross_val_score
KNN_classifier = KNeighborsClassifier(n_neighbors=17, p=2)
scores = cross_val_score(KNN_classifier, explanatory_variables, response_series, cv=10, scoring='accuracy')

print scores

mean_accuracy = numpy.mean(scores)
print mean_accuracy * 100

print accuracy * 100

k_range = range(1, 30, 2)
scores=[]

from sklearn.grid_search import GridSearchCV
knn = KNeighborsClassifier(p=2)
k_range = range(1,30,2)
param_grid = dict(n_neighbors = k_range)
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(explanatory_variables, response_series)

grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(k_range, grid_mean_scores)
best_oob_score = grid.best_score_
grid.best_params_
knn_optimal = grid.best_estimator_
print knn_optimal



conn = sqlite3.connect('/Users/YihanLi/Documents/SQLite/lahman2013.sqlite')

sql = '''select a.playerID,a.inducted, sum(b.PO) as putouts, sum(b.A) as assists, sum(b.G) as games, sum(b.DP) as doubleplay, sum(b.E) as errors from HallofFame a
join Fielding b on a.playerID=b.playerID
where a.yearid>=2000
group by 1;'''

df = pandas.read_sql(sql,conn)

conn.close()

df.dropna(inplace=True)

df.head()
df['DP_per_PO'] = df.doubleplay/(df.putouts+df.assists)
df['error_per_game'] = df.errors/df.games

df.replace(numpy.inf, numpy.nan)
df = df[numpy.isfinite(df['DP_per_PO'])]
df = df[numpy.isfinite(df['error_per_game'])]
df.dropna(inplace=True)
df.head()
response_series = df.inducted
explanatory_variables = df[['DP_per_PO', 'error_per_game', 'games', 'doubleplay', 'putouts', 'assists']]

optimal_knn_preds = knn_optimal.predict(explanatory_variables)


number_correct = len(response_series[response_series == optimal_knn_preds])
total_in_test_set = len(response_series)
accuracy = number_correct / total_in_test_set
print accuracy*100
