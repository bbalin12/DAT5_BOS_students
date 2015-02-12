# -*- coding: utf-8 -*-
"""
Created on Sun Feb 08 10:24:52 2015

@author: garauste
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 03 19:31:33 2015

@author: garauste
"""

# importing division from the future release of python (i.e. Python 3)
from __future__ import division

## Importing the correct libraries
import numpy
import sqlite3
import pandas
from sklearn.neighbors import KNeighborsClassifier

## Designating the model constraints at the top of the file
## This is the amount of the data that we want to hold out#
CROSS_VALIDATION_AMOUNT = .2

## CONNECT TO THE BASEBALL DATABASE
conn = sqlite3.connect('C:\Users\Gareth\Dropbox\General Assembly\Raw Data\lahman2013.sqlite')

## Create a large SQL object to simultaneously pull data from all 4 Tables in the Baseball database: HallOfFame, Batting, Pitching, Fielding
sql = """select a.playerID, a.inducted, f.Games, f.Hits, f.At_Bats, f.Homers, f.Pitcher_Ws, f.Pitcher_ShutOuts, f.Pitcher_StrikeOuts, f.Pitcher_Earned_Run_Avg, f.Field_Position,
f.Field_Errors from HallOfFame a 
left outer join (select b.G as Games, b.H as Hits, b.AB as At_Bats, b.HR as Homers, b.playerID, e.Pitcher_Ws, e.Pitcher_ShutOuts, e.Pitcher_StrikeOuts, e.Pitcher_Earned_Run_Avg,
e.Field_Position, e.Field_Errors  from Batting b
left outer join (select c.playerID, c.W as Pitcher_Ws, c.SHO as Pitcher_ShutOuts, c.SO as Pitcher_StrikeOuts, c.ERA as Pitcher_Earned_Run_Avg, 
d.Pos as Field_Position, d.E as Field_Errors from Pitching c left outer join Fielding d on c.playerID = d.playerID) e on b.playerID = e.playerID)f
on a.playerID = f.playerID
where yearID<2000;"""

## Pass the connection and the SQL String to a pandas.read file 
df = pandas.read_sql(sql,conn)

#Close the connection 
conn.close()

# Dropping all NaNs in the dataset
df.dropna(inplace=True)

## Comment: This SQL query does not work, replace the NA removes all non-pitchers from the table. Need to split the dataset into separate tables for pitchers and batters 

## CONNECT TO THE BASEBALL DATABASE
conn = sqlite3.connect('C:\Users\Gareth\Dropbox\General Assembly\Raw Data\lahman2013.sqlite')

## Create a large SQL object to simultaneously pull data from all 4 Tables in the Baseball database: HallOfFame, Batting, Pitching, Fielding
sql_batters = """select a.playerID as playerID, max(a.inducted) as inducted, sum(f.Games) as Games, sum(f.Hits) as Hits, sum(f.At_Bats) as At_Bats, sum(f.Homers) as Homers, sum(f.Double_Plays) as Double_Plays, sum(f.Fielder_Assists) as Fielder_Assists, sum(f.Field_Errors) as Field_Errors from HallOfFame a 
left outer join (select b.G as Games, b.H as Hits, b.AB as At_Bats, b.HR as Homers, b.playerID,e.Fielder_Assists, e.Double_Plays, e.Field_Errors  from Batting b
left outer join (select d.playerID,
d.A as Fielder_Assists, d.E as Field_Errors, d.DP as Double_Plays from  Fielding d) e on b.playerID = e.playerID)f
on a.playerID = f.playerID
where yearID<2000
and f.Games is not null 
group by a.playerID;"""

## Pass the connection and the SQL String to a pandas.read file 
df_batters = pandas.read_sql(sql_batters,conn)

#Close the connection 
conn.close()

# Dropping all NaNs in the dataset
df_batters.dropna(inplace=True)

## CONNECT TO THE BASEBALL DATABASE
conn = sqlite3.connect('C:\Users\garauste\Documents\SQLite\lahman2013.sqlite')

## Create a large SQL object to simultaneously pull data from all 4 Tables in the Baseball database: HallOfFame, Batting, Pitching, Fielding
sql_pitchers = """select a.playerID as playerID, max(a.inducted) as inducted, sum(f.Pitcher_Ws) as Pitcher_Ws, sum(f.Pitcher_ShutOuts) as Pitcher_ShutOuts, sum(f.Pitcher_StrikeOuts) as Pitcher_StrikeOuts, sum(f.HR_Allowed) as HR_Allowed, sum(f.Complete_Games) as Complete_Games 
from HallOfFame a inner join (select b.playerID,b.ERA as Pitcher_Earned_Run_Avg,b.W as Pitcher_Ws, b.SHO as Pitcher_ShutOuts, b.SO as Pitcher_StrikeOuts, b.HR as HR_Allowed, b.CG as Complete_Games from Pitching b) f on a.playerID = f.playerID where yearID<2000 group by a.playerID;"""

## Pass the connection and the SQL String to a pandas.read file 
df_pitchers = pandas.read_sql(sql_pitchers,conn)

#Close the connection 
conn.close()

# Dropping all NaNs in the dataset
df_pitchers.dropna(inplace=True)

##############################################################################
################## Begin KNN Modelling Process ###############################
##############################################################################

# separate your response variable from your explanatory variable for both the batters and pitchers datasets
response_series_batters = df_batters.inducted
response_series_pitchers = df_pitchers.inducted
explanatory_vars_batters = df_batters[['Games','Hits','At_Bats','Homers','Double_Plays','Fielder_Assists','Field_Errors']]
explanatory_vars_pitchers= df_pitchers[['Pitcher_Ws','Pitcher_ShutOuts','Pitcher_StrikeOuts','HR_Allowed','Complete_Games']]

## Designate Separate holdouts for both the batters and pitchers 
holdout_num_batters = round(len(df_batters.index)*CROSS_VALIDATION_AMOUNT,0)
holdout_num_pitchers = round(len(df_pitchers.index)*CROSS_VALIDATION_AMOUNT,0)

# creating our training and test indices for the batter and pitcher datasets #
test_indices_batters = numpy.random.choice(df_batters.index, holdout_num_batters, replace = False)
train_indices_batters = df_batters.index[~df_batters.index.isin(test_indices_batters)]
test_indices_pitchers = numpy.random.choice(df_pitchers.index, holdout_num_pitchers, replace = False)
train_indices_pitchers = df_pitchers.index[~df_pitchers.index.isin(test_indices_pitchers)] 

# create our training set for both datasets
response_train_batters = response_series_batters.ix[train_indices_batters,]
explanatory_train_batters = explanatory_vars_batters.ix[train_indices_batters,]
response_train_pitchers = response_series_pitchers.ix[train_indices_pitchers,]
explanatory_train_pitchers = explanatory_vars_pitchers.ix[train_indices_pitchers,]

# create our test set for both datasets
response_test_batters = response_series_batters.ix[test_indices_batters,]
explanatory_test_batters = explanatory_vars_batters.ix[test_indices_batters,]
response_test_pitchers= response_series_pitchers.ix[test_indices_pitchers,]
explanatory_test_pitchers = explanatory_vars_pitchers.ix[test_indices_pitchers,]

## Instantiating the KNN Classifier, with p = 2 for Euclidian distance
KNN_Classifier_batters = KNeighborsClassifier(n_neighbors=3,p=2)
KNN_Classifier_pitchers = KNeighborsClassifier(n_neighbors=3,p=2)
# fitting the data to the training set #
KNN_Classifier_batters.fit(explanatory_train_batters,response_train_batters)
KNN_Classifier_pitchers.fit(explanatory_train_pitchers,response_train_pitchers)

# predicting the data on the test set
predicted_response_batters = KNN_Classifier_batters.predict(explanatory_test_batters)
predicted_response_pitchers = KNN_Classifier_pitchers.predict(explanatory_test_pitchers)

# calculating accuracy
number_correct_batters = len(response_test_batters[response_test_batters == predicted_response_batters])
total_in_test_set_batters = len(response_test_batters)
accuracy_batters = number_correct_batters/total_in_test_set_batters
print accuracy_batters*100

number_correct_pitchers = len(response_test_pitchers[response_test_pitchers == predicted_response_pitchers])
total_in_test_set_pitchers = len(response_test_pitchers)
accuracy_pitchers = number_correct_pitchers/total_in_test_set_pitchers
print accuracy_pitchers*100

## Comment: It appears in this particular scenario that the batters model is more accurate than the pitching model. However Neither model is overly accurate compared to the In-Class model that had a 98% Accuracy rating. Our Batting model has accuracy of 79.6% accuracy while pitching only has 69.7% accuracy.

############################################
### Now let's do K-Fold Cross validation ###
############################################

# LET'S USE 10-FOLD CROSS-VALIDATION TO SCORE OUR MODEL
from sklearn.cross_validation import cross_val_score
# we need to re-instantiate the model
KNN_Classifier_batters = KNeighborsClassifier(n_neighbors=3,p=2)
KNN_Classifier_pitchers = KNeighborsClassifier(n_neighbors=3,p=2)

# Notice that instead of passing in the train and test sets we are passing 
# the entire dataset as method will auto split
scores_batters = cross_val_score(KNN_Classifier_batters, explanatory_vars_batters,response_series_batters,cv=10,scoring='accuracy')
scores_pitchers = cross_val_score(KNN_Classifier_pitchers, explanatory_vars_pitchers,response_series_pitchers,cv=10,scoring='accuracy')
                         
# print out scores object
print scores_batters 
print scores_pitchers
        
# now let's get the average accuary score 
mean_accuracy_batters = numpy.mean(scores_batters)
print mean_accuracy_batters*100
mean_accuracy_pitchers = numpy.mean(scores_pitchers)
print mean_accuracy_pitchers*100
# look at how this differes from the previous two accuarcies 
print accuracy_batters*100
print accuracy_pitchers*100

## Comment: Interestingly using cross validation our batting model accuracy has declined slightly while the pitching model has seen substantial improvements as a result of cross validation. Next we will find the optimal value for K.

# Tune the model for the optimal number of K
k_range = range(1,30,2)
scores_batters =[]
for k in k_range:
    knn_batters = KNeighborsClassifier(n_neighbors=k,p=2)
    scores_batters.append(numpy.mean(cross_val_score(knn_batters,explanatory_vars_batters,response_series_batters,cv=5,scoring='accuracy')))
    
k_range = range(1,30,2)
scores_pitchers =[]
for k in k_range:
    knn_pitchers = KNeighborsClassifier(n_neighbors=k,p=2)
    scores_pitchers.append(numpy.mean(cross_val_score(knn_pitchers,explanatory_vars_pitchers,response_series_pitchers,cv=5,scoring='accuracy')))

    
# Plot the K values (x-axis) versus the 5 fold CV Score
import matplotlib.pyplot as plt
plt.figure()
plt.plot(k_range,scores_batters)

plt.figure()
plt.plot(k_range,scores_pitchers)
# optimal value of K appears to be 3

# automatic grid search for an optimal value of K
from sklearn.grid_search import GridSearchCV

knn_batters = KNeighborsClassifier(p=2)
k_range = range(1,70,2)
param_grid_batters = dict(n_neighbors=k_range)
grid_batters = GridSearchCV(knn_batters,param_grid_batters,cv=5,scoring='accuracy')
grid_batters.fit(explanatory_vars_batters,response_series_batters)

knn_pitchers = KNeighborsClassifier(p=2)
k_range = range(1,70,2)
param_grid_pitchers = dict(n_neighbors=k_range)
grid_pitchers = GridSearchCV(knn_pitchers,param_grid_pitchers,cv=5,scoring='accuracy')
grid_pitchers.fit(explanatory_vars_pitchers,response_series_pitchers)

# Check the reults of the grid search and extract the optimal estimator 
grid_batters.grid_scores_
grid_mean_scores_batters = [results[1] for results in grid_batters.grid_scores_]
plt.figure()
plt.plot(k_range,grid_mean_scores_batters)
best_oob_score_batters = grid_batters.best_score_
grid_batters.best_params_
KNN_optimal_batters = grid_batters.best_estimator_

grid_pitchers.grid_scores_
grid_mean_scores_pitchers= [results[1] for results in grid_pitchers.grid_scores_]
plt.figure()
plt.plot(k_range,grid_mean_scores_pitchers)
best_oob_score_pitchers = grid_pitchers.best_score_
grid_pitchers.best_params_
KNN_optimal_pitchers = grid_pitchers.best_estimator_

########################################################################
############# Final step is to test the process out of sample ##########
########################################################################

## Pull in the Data from 2000 onwards ##
conn = sqlite3.connect('C:\Users\garauste\Documents\SQLite\lahman2013.sqlite')

## redo sql formula
sql_batters = """select a.playerID as playerID, max(a.inducted) as inducted, sum(f.Games) as Games, sum(f.Hits) as Hits, sum(f.At_Bats) as At_Bats, sum(f.Homers) as Homers, sum(f.Double_Plays) as Double_Plays, sum(f.Fielder_Assists) as Fielder_Assists, sum(f.Field_Errors) as Field_Errors from HallOfFame a 
left outer join (select b.G as Games, b.H as Hits, b.AB as At_Bats, b.HR as Homers, b.playerID,e.Fielder_Assists, e.Double_Plays, e.Field_Errors  from Batting b
left outer join (select d.playerID,
d.A as Fielder_Assists, d.E as Field_Errors, d.DP as Double_Plays from  Fielding d) e on b.playerID = e.playerID)f
on a.playerID = f.playerID
where yearID>2000
and f.Games is not null 
group by a.playerID;"""

df_batters = pandas.read_sql(sql_batters,conn)

conn.close()

df_batters.dropna(inplace=True)

response_series_batters = df_batters.inducted
explanatory_vars_batters = df_batters[['Games','Hits','At_Bats','Homers','Double_Plays','Fielder_Assists','Field_Errors']]

optimal_knn_preds_batters = KNN_optimal_batters.predict(explanatory_vars_batters)

number_correct_batters = len(response_series_batters[response_series_batters==optimal_knn_preds_batters])
total_in_test_set_batters = len(response_series_batters)
accuracy_batters = number_correct_batters/total_in_test_set_batters

# compare actual with the accuracy anticipated by our grid search
print accuracy_batters*100
print best_oob_score_batters*100

## Comment: Strangely in this case our model has outperformed the in-sample data when tested out of sample for players who were inducted into the hall of fame post 2000. This is highly unusually and I'm not sure of the significance of this result. 

### Repeat the out of sample testing for pitchers ##

## Pull in the Data from 2000 onwards ##
conn = sqlite3.connect('C:\Users\garauste\Documents\SQLite\lahman2013.sqlite')

## redo sql formula
sql_pitchers = """select a.playerID as playerID, max(a.inducted) as inducted, sum(f.Pitcher_Ws) as Pitcher_Ws, sum(f.Pitcher_ShutOuts) as Pitcher_ShutOuts, sum(f.Pitcher_StrikeOuts) as Pitcher_StrikeOuts, sum(f.HR_Allowed) as HR_Allowed, sum(f.Complete_Games) as Complete_Games 
from HallOfFame a inner join (select b.playerID,b.ERA as Pitcher_Earned_Run_Avg,b.W as Pitcher_Ws, b.SHO as Pitcher_ShutOuts, b.SO as Pitcher_StrikeOuts, b.HR as HR_Allowed, b.CG as Complete_Games from Pitching b) f on a.playerID = f.playerID where yearID<2000 group by a.playerID;"""

df_pitchers = pandas.read_sql(sql_pitchers,conn)

conn.close()

df_pitchers.dropna(inplace=True)

response_series_pitchers = df_pitchers.inducted
explanatory_vars_pitchers = df_pitchers[['Pitcher_Ws','Pitcher_ShutOuts','Pitcher_StrikeOuts','HR_Allowed','Complete_Games']]

optimal_knn_preds_pitchers = KNN_optimal_pitchers.predict(explanatory_vars_pitchers)

number_correct_pitchers = len(response_series_pitchers[response_series_pitchers==optimal_knn_preds_pitchers])
total_in_test_set_pitchers = len(response_series_pitchers)
accuracy_pitchers = number_correct_pitchers/total_in_test_set_pitchers

# compare actual with the accuracy anticipated by our grid search
print accuracy_pitchers*100
print best_oob_score_pitchers*100

## Comment: The pitchers model also performs better out of sample than in sample.