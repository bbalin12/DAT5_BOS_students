# -*- coding: utf-8 -*-
"""
Created on Fri Feb 06 13:48:01 2015

@author: jeppley
"""


#SETTING UP THE DATA EXERCISE 

# importing division from the 'future' release of Python (i.e. Python 3)
from __future__ import division

# importing numpy and the KNN content in scikit-learn 
# along with SQLite and pandas.
import numpy
import sqlite3
import pandas
from sklearn.neighbors import KNeighborsClassifier

# designating model constants at the top of the file per PEP8 
# see https://www.python.org/dev/peps/pep-0008/
# this is the percent we want to hold out for our cross-validation.
CROSS_VALIDATION_AMOUNT = .2


#WRANGLING DATA NEEDED FOR MODELING: LOOKING AT BATTING PREDICTIONS FIRST

#Work done in SQL before bringing in for python amendments


create table pitchingsub as
select playerID, yearID, ERA, HR, W, G from pitching
group by playerID, yearID

create table hallsub2 as
select playerID, yearID, inducted from halloffame
where yearID < 2000
group by playerID, yearID

create table pitchingmod as
select a.*, b.ERA, b.HR, b.W, b.G from hallsub2 a
left outer join
pitchingsub b on a.playerID=b.playerID and a.yearID=b.yearID




# connect to the baseball database. Notice I am passing the full path
# to the SQLite file.
conn = sqlite3.connect("C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite")
# creating an object contraining a string that has the SQL query. 
sql = '''select a.playerID as playerID, max(a.inducted) as inducted, 
sum(f.Games) as Games, sum(f.Hits) as Hits, sum(f.At_Bats) as At_Bats, 
sum(f.Homers) as HomeRuns, sum(f.Field_Errors) as Field_Errors
from HallOfFame a 
left outer join 
    (select b.G as Games, b.H as Hits, b.AB as At_Bats, b.HR as Homers,
    b.playerID,e.Field_Errors from Batting b
    left outer join 
        (select d.playerID, d.E as Field_Errors,
        d.DP as Double_Plays from Fielding d) e on b.playerID = e.playerID)f
on a.playerID = f.playerID
where yearID<2000
and f.Games is not null 
group by a.playerID
;'''
# passing the connection and the SQL string to pandas.read_sql.
test = pandas.read_sql(sql, conn)
# closing the connection.
conn.close()


#checking what dataset looks like

test.describe()
#surprised that there are only 7 cases of pitchers being inducted into hall of fame


# dropping ALL NaNs in the dataset.
test.dropna(inplace = True)

#Model

# seperate your response variable from your explanatory variables
response_series = test.inducted 
explanatory_variables = test[['Hits','HomeRuns', 'Field_Errors']]

# designating the number of observations we need to hold out.
# notice that I'm rounding down so as to get a whole number. 
holdout_num = round(len(test.index) * CROSS_VALIDATION_AMOUNT, 0)




# creating our training and text indices
test_indices = numpy.random.choice(test.index, holdout_num, replace = False )
train_indices = test.index[~test.index.isin(test_indices)]

# our training set
response_train = response_series.ix[train_indices,]
explanatory_train = explanatory_variables.ix[train_indices,]

# our test set
response_test = response_series.ix[test_indices,]
explanatory_test = explanatory_variables.ix[test_indices,]

# instantiating the KNN classifier, with p=2 for Euclidian distnace
# see http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier for more information.
KNN_classifier = KNeighborsClassifier(n_neighbors=3, p = 2)
# fitting the data to the training set
KNN_classifier.fit(explanatory_train, response_train) 

# predicting the data on the test set. 
predicted_response = KNN_classifier.predict(explanatory_test)

# calculating accuracy
number_correct = len(response_test[response_test == predicted_response])
total_in_test_set = len(response_test)
accuracy = number_correct / total_in_test_set
print accuracy* 100



######
## K-Fold CV
#####



# let's use 10-fold cross-validation to score our model. 
from sklearn.cross_validation import cross_val_score
# we need to re-instantiate the model 
KNN_classifier = KNeighborsClassifier(n_neighbors=3, p = 2)
# notice that instead of putting in my train and text groups, I'm putting 
# in the entire dataset -- the cross_val_score method automatically splits
# the data. 
scores = cross_val_score(KNN_classifier, explanatory_variables, response_series, cv=10, scoring='accuracy')
# let's print out the accuracy at each itration of cross-validation.
print scores
# now, let's get the average accuracy score. 
mean_accuracy = numpy.mean(scores) 
print mean_accuracy * 100
# look at hhow his differs from the previous two accuracies we computed. 
print new_accuracy * 100
print accuracy * 100

# now, let's tune the model for the optimal number of K. 
k_range = range(1, 30, 2)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k,  p = 2)
    scores.append(numpy.mean(cross_val_score(knn, explanatory_variables, response_series, cv=10, scoring='accuracy')))

# plot the K values (x-axis) versus the 5-fold CV score (y-axis)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(k_range, scores)
## so, the optimal value of K appears to be low -- under 5 or so. 

# automatic grid search for an optimal value of K
from sklearn.grid_search import GridSearchCV
knn = KNeighborsClassifier( p = 2)
k_range = range(1, 30, 2)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(explanatory_variables, response_series)

# check the results of the grid search and extract the optial estimator
grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(k_range, grid_mean_scores)
best_oob_score = grid.best_score_
grid.best_params_
Knn_optimal = grid.best_estimator_






