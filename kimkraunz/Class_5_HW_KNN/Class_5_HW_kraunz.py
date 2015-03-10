# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 10:13:00 2015

@author: kim kraunz

The first section uses KNN to predict induction into the hall of fame based on pitching parameters
(strikeouts, ERA, wins, saves, and years pitched)
"""

import numpy
import pandas
import sqlite3
from sklearn.neighbors import KNeighborsClassifier
from __future__ import division
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

CROSS_VALIDATION_AMOUNT = .2

conn = sqlite3.connect('/Users/jkraunz/Documents/SQLite/lahman2013.sqlite.crdownload')

sql = '''
Select a.playerID, a.final_year_voted, a.years_voted, a.inducted, a.total_hits, a.total_years_b, a.total_HRs,
a.total_SOs, a.avg_ERA, a.total_wins, a.total_saves, a.years_pitched, sum(E) as total_errors, sum(G) as games_played_fielding
FROM 
(SELECT m.playerID, m.final_year_voted, m.years_voted, m.inducted, m.total_hits, m.total_years_b, m.total_HRs,
sum(SO) as total_SOs, avg(ERA) as avg_ERA, sum(W) as total_wins, sum(SV) as total_saves, count(YearID) as years_pitched
FROM
(select h.playerID, h.final_year_voted, h.years_voted, h.inducted,
sum(H) as total_hits, count(yearID) as total_years_b, sum(HR) as total_HRs
FROM 
(SELECT playerID, max(yearID) as final_year_voted, count(yearID) as years_voted, inducted
FROM HallofFame 
GROUP BY playerID) h
LEFT JOIN Batting b on h.playerID = b.playerID
GROUP BY h.playerID) m
LEFT JOIN Pitching p on m.playerID = p.playerID
WHERE final_year_voted < 2000
GROUP BY m.playerID) a
LEFT JOIN Fielding f on a.playerID = f.playerID
GROUP BY a.playerID
'''

# names the dataframe that is read from the table (sql) in the sql database (conn)
df = pandas.read_sql(sql, conn)

# good to close database
conn.close()

# prints out first 5 rows to make sure table imported
df.head()

# converting Nans to means of series
df = df.fillna(df.mean())
df.head(30)

# creates new variable "error_rate" which is a function of total_errors and games_played_fielding
df['error_rate'] = df.total_errors / df.games_played_fielding

# creates new variable "years_played" which is a function of the years_pitched and total_years_b series
df['years_played'] = 0
df.years_played[df.years_pitched >= df.total_years_b] = df.years_pitched
df.years_played[df.years_pitched < df.total_years_b] = df.total_years_b
    
# checks to make sure new variables are correct
df.head(30)

###################################################################################

# K nearest neighbor
    
# defines the dependent variable (y variable) as the series inducted in the dataframe
response_series = df.inducted

# defines the independent variables in the dataframe
explanatory_variables = df[['years_voted', 'years_played', 'total_hits', 'total_HRs', 'total_SOs', 'avg_ERA', 'total_wins', 'total_saves', 'error_rate']]
 
# calculates the # to "holdout" of the prediction test to cross-validate
holdout_num = round(len(df.index) * CROSS_VALIDATION_AMOUNT, 0)

# prints the holdout number
holdout_num

# defines the test_indices using the random module in numpy based on the dataframe index and # to holdout
test_indices = numpy.random.choice(df.index, holdout_num, replace = False)

# defines the train_indices as the opposite of what is in the test_indices defined above
train_indices = df.index[~df.index.isin(test_indices)]

# defines the responses for the training  to the response series (inducted) from the randomly assigned training indices 
response_train = response_series.ix[train_indices,]

# checks the response_train series
response_train.head()

# defines the explanatory for the training to the explanatory variables in the train indices
explanatory_train = explanatory_variables.ix[train_indices,]

# tests that the response training and explanatory training are in the same indices
response_train.index == explanatory_train.index

# defines the response test series as the reponse variable (induction) in the test indices
response_test = response_series.ix[test_indices,]

# defines the explanatory test series as the explantory variables in the test indices
explanatory_test = explanatory_variables.ix[test_indices,]

# checks the response_test and explanatory_test series
response_test.head()
explanatory_test.head()


# determines the optimal number for k using the built in functions KNeighbors Classifier
# and GridSearchCV
knn = KNeighborsClassifier( p = 2)
k_range = range(1, 30, 2)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(explanatory_variables, response_series)

# plots the scores versus k
grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(k_range, grid_mean_scores)
best_oob_score = grid.best_score_
grid.best_params_
Knn_optimal = grid.best_estimator_

print best_oob_score
print k_range, grid_mean_scores


# tests accuracy of model above on test group
predicted_response = Knn_optimal.predict(explanatory_test)
predicted_response

# calculates accuracy of the KNN model
number_correct = len(response_test[response_test == predicted_response])
total_in_test_set = len(response_test)
accuracy = number_correct / total_in_test_set
print "The accuracy of my initial model is: %f" % (accuracy * 100)

#############################################################################

'''I decided to try a different model replacing error_rate with total_errors since
the rate has a different magnitude from the other variables
'''

# defines the dependent variable (y variable) as the series inducted in the dataframe
response_series = df.inducted

# defines the independent variables in the dataframe
explanatory_variables = df[['years_voted', 'years_played', 'total_hits', 'total_HRs', 'total_SOs', 'avg_ERA', 'total_wins', 'total_saves', 'total_errors']]
 
# calculates the # to "holdout" of the prediction test to cross-validate
holdout_num = round(len(df.index) * CROSS_VALIDATION_AMOUNT, 0)

# prints the holdout number
holdout_num

# defines the test_indices using the random module in numpy based on the dataframe index and # to holdout
test_indices = numpy.random.choice(df.index, holdout_num, replace = False)

# defines the train_indices as the opposite of what is in the test_indices defined above
train_indices = df.index[~df.index.isin(test_indices)]

# defines the responses for the training  to the response series (inducted) from the randomly assigned training indices 
response_train = response_series.ix[train_indices,]

# checks the response_train series
response_train.head()

# defines the explanatory for the training to the explanatory variables in the train indices
explanatory_train = explanatory_variables.ix[train_indices,]

# tests that the response training and explanatory training are in the same indices
response_train.index == explanatory_train.index

# defines the response test series as the reponse variable (induction) in the test indices
response_test = response_series.ix[test_indices,]

# defines the explanatory test series as the explantory variables in the test indices
explanatory_test = explanatory_variables.ix[test_indices,]

# checks the response_test and explanatory_test series
response_test.head()
explanatory_test.head()


# determines the optimal number for k using the built in functions KNeighbors Classifier
# and GridSearchCV
knn = KNeighborsClassifier( p = 2)
k_range = range(1, 30, 2)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(explanatory_variables, response_series)

# plots the scores versus k
grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(k_range, grid_mean_scores)
best_oob_score = grid.best_score_
grid.best_params_
Knn_optimal = grid.best_estimator_

print best_oob_score
print k_range, grid_mean_scores


# tests accuracy of model above on test group
predicted_response = Knn_optimal.predict(explanatory_test)
predicted_response

# calculates accuracy of the KNN model
number_correct = len(response_test[response_test == predicted_response])
total_in_test_set = len(response_test)
total_error_accuracy = number_correct / total_in_test_set
print "The accuracy of a model that uses total errors instead of error rate is: %f" % (100*total_error_accuracy)
print "The accuracy of my initial model is: %f" % (accuracy * 100)

# I see increased accuracy but a slightly lower mean accuracy by changing the way I input errors

###############################################################################

'''
I wanted to try one more model to see if simplifying it slightly would lead to 
higher accuracy.  I left out years_played and avg_ERA to see what kind of effect
it would have on accuracy.  I kept total_errors instead of error_rate because
the magnitude of total_errors is more similar to the other variables
'''

# defines the dependent variable (y variable) as the series inducted in the dataframe
response_series = df.inducted

# defines the independent variables in the dataframe
explanatory_variables = df[['years_voted', 'total_hits', 'total_HRs', 'total_SOs', 'total_wins', 'total_saves', 'total_errors']]
 
# calculates the # to "holdout" of the prediction test to cross-validate
holdout_num = round(len(df.index) * CROSS_VALIDATION_AMOUNT, 0)

# prints the holdout number
holdout_num

# defines the test_indices using the random module in numpy based on the dataframe index and # to holdout
test_indices = numpy.random.choice(df.index, holdout_num, replace = False)

# defines the train_indices as the opposite of what is in the test_indices defined above
train_indices = df.index[~df.index.isin(test_indices)]

# defines the responses for the training  to the response series (inducted) from the randomly assigned training indices 
response_train = response_series.ix[train_indices,]

# checks the response_train series
response_train.head()

# defines the explanatory for the training to the explanatory variables in the train indices
explanatory_train = explanatory_variables.ix[train_indices,]

# tests that the response training and explanatory training are in the same indices
response_train.index == explanatory_train.index

# defines the response test series as the reponse variable (induction) in the test indices
response_test = response_series.ix[test_indices,]

# defines the explanatory test series as the explantory variables in the test indices
explanatory_test = explanatory_variables.ix[test_indices,]

# checks the response_test and explanatory_test series
response_test.head()
explanatory_test.head()


# determines the optimal number for k using the built in functions KNeighbors Classifier
# and GridSearchCV
knn = KNeighborsClassifier( p = 2)
k_range = range(1, 30, 2)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(explanatory_variables, response_series)

# plots the scores versus k
grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(k_range, grid_mean_scores)
best_oob_score = grid.best_score_
grid.best_params_
Knn_optimal = grid.best_estimator_

print best_oob_score
print k_range, grid_mean_scores

# tests accuracy of model above on test group
predicted_response = Knn_optimal.predict(explanatory_test)
predicted_response

# calculates accuracy of the KNN model
number_correct = len(response_test[response_test == predicted_response])
total_in_test_set = len(response_test)
smaller_accuracy = number_correct / total_in_test_set
print "The accuracy of a model that doesn't include average ERA and years played is: %f" % (100*smaller_accuracy)
print "The accuracy of a model that uses total errors instead of error rate is: %f" % (100*total_error_accuracy)
print "The accuracy of my initial model is: %f" % (accuracy * 100)


####################################################################################

''' Interestingly, I find that my mean accuracies are almost indentical while my KNN accuracies 
are different.  I believe that the simplest model will end up being the best when I apply it to 
an outside group since it has almost as high of mean accuracy but has the least variables.
I'm now going to test it against those inducted in 2000 or later
'''

conn = sqlite3.connect('/Users/jkraunz/Documents/SQLite/lahman2013.sqlite.crdownload')

sql = '''
Select a.playerID, a.final_year_voted, a.years_voted, a.inducted, a.total_hits, a.total_years_b, a.total_HRs,
a.total_SOs, a.avg_ERA, a.total_wins, a.total_saves, a.years_pitched, sum(E) as total_errors, sum(G) as games_played_fielding
FROM 
(SELECT m.playerID, m.final_year_voted, m.years_voted, m.inducted, m.total_hits, m.total_years_b, m.total_HRs,
sum(SO) as total_SOs, avg(ERA) as avg_ERA, sum(W) as total_wins, sum(SV) as total_saves, count(YearID) as years_pitched
FROM
(select h.playerID, h.final_year_voted, h.years_voted, h.inducted,
sum(H) as total_hits, count(yearID) as total_years_b, sum(HR) as total_HRs
FROM 
(SELECT playerID, max(yearID) as final_year_voted, count(yearID) as years_voted, inducted
FROM HallofFame 
GROUP BY playerID) h
LEFT JOIN Batting b on h.playerID = b.playerID
GROUP BY h.playerID) m
LEFT JOIN Pitching p on m.playerID = p.playerID
WHERE final_year_voted >= 2000
GROUP BY m.playerID) a
LEFT JOIN Fielding f on a.playerID = f.playerID
GROUP BY a.playerID
'''

# names the dataframe that is read from the table (sql) in the sql database (conn)
df = pandas.read_sql(sql, conn)

# good to close database
conn.close()

# prints out first 5 rows to make sure table imported
df.head()

# converting Nans to means of series
df = df.fillna(df.mean())
df.head(30)

# creates new variable "years_played" which is a function of the years_pitched and total_years_b series
df['years_played'] = 0
df.years_played[df.years_pitched >= df.total_years_b] = df.years_pitched
df.years_played[df.years_pitched < df.total_years_b] = df.total_years_b
    
# checks to make sure new variables are correct
df.head(30)

response_series = df.inducted
explanatory_variables = df[['years_voted', 'total_hits', 'total_HRs', 'total_SOs', 'total_wins', 'total_saves', 'total_errors']]


optimal_knn_preds = Knn_optimal.predict(explanatory_variables)

number_correct = len(response_series[response_series == optimal_knn_preds])
total_in_test_set = len(response_series)
post_2000_accuracy = number_correct / total_in_test_set

print "The accuracy of the simplest model when I test on those inducted 2000 or later: %f" % (post_2000_accuracy * 100)


'''
I see in fact that the accuracy goes down when I test my model on those inducted in 2000 or later.
'''

