# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 21:09:44 2015

@author: Margaret
"""

import sqlite3
import pandas
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn import ensemble
import numpy
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import  GridSearchCV
from sklearn import tree
import matplotlib.pyplot as plt

def cleanup_data(df, cutoffPercent = .01):
    for col in df:
        sizes = df[col].value_counts(normalize = True)
        # get the names of the levels that make up less than 1% of the dataset
        values_to_delete = sizes[sizes<cutoffPercent].index
        df[col].ix[df[col].isin(values_to_delete)] = "Other"
    return df
#

def get_binary_values(data_frame):
    """encodes cateogrical features in Pandas.
    """
    all_columns = pandas.DataFrame( index = data_frame.index)
    for col in data_frame.columns:
        data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pandas.concat([all_columns, data], axis=1)
    return all_columns
#
def find_zero_var(df):
    """finds columns in the dataframe with zero variance -- ie those
        with the same value in every observation.
    """   
    toKeep = []
    toDelete = []
    for col in df:
        if len(df[col].value_counts()) > 1:
            toKeep.append(col)
        else:
            toDelete.append(col)
        ##
    return {'toKeep':toKeep, 'toDelete':toDelete} 
##
def find_perfect_corr(df):
    """finds columns that are eother positively or negatively perfectly correlated (with correlations of +1 or -1), and creates a dict 
        that includes which columns to drop so that each remaining column
        is independent
    """  
    corrMatrix = df.corr()
    corrMatrix.loc[:,:] =  numpy.tril(corrMatrix.values, k = -1)
    already_in = set()
    result = []
    for col in corrMatrix:
        perfect_corr = corrMatrix[col][abs(numpy.round(corrMatrix[col],10)) == 1.00].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    toRemove = []
    for item in result:
        toRemove.append(item[1:(len(item)+1)])
    toRemove = sum(toRemove, [])
    return {'corrGroupings':result, 'toRemove':toRemove}
###


# putting a setting into pandas that lets you print out the entire
# DataFrame when you use the .head() method
pandas.set_option('display.max_columns', None)


con = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')
query = """
SELECT (m.nameFirst||" "||nameLast) as p_name, m.height as height, m.weight as weight, m.bats as bats, m.throws as throws,
inducted, bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_RBI, bat_baseballs, 
bat_intentwalks, bat_strikes,
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, pitch_wild, pitch_games, pitch_saves, 
pitch_earnruns, pitch_runsallowed, pitch_finish, pitch_outs, 
pitch_hits, pitch_opp_BA, f_putouts, f_assists, f_errors FROM Master m
INNER JOIN
(SELECT pID, dom.teamID as dom_team, inducted, 
bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_RBI, bat_caught, bat_baseballs, 
bat_intentwalks, bat_doubles, bat_triples, bat_strikes, bat_stolen, 
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, pitch_wild, pitch_games, pitch_saves,
pitch_earnruns, pitch_runsallowed, pitch_finish, pitch_outs, pitch_hits, pitch_opp_BA, 
f_putouts, f_assists, f_errors FROM dominant_team_per_player dom
INNER JOIN
(SELECT h.playerID as pID, max(CASE WHEN h.inducted='Y' THEN 1 ELSE 0 END) as inducted, 
bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_RBI, bat_caught, bat_baseballs, 
bat_intentwalks, bat_doubles, bat_triples, bat_strikes, bat_stolen, 
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, pitch_wild, pitch_games, pitch_saves,
pitch_earnruns, pitch_runsallowed, pitch_finish, pitch_outs, pitch_hits, pitch_opp_BA, 
f_putouts, f_assists, f_errors FROM HallofFame h
INNER JOIN
(SELECT f.playerID, f.PO as f_putouts, f.A as f_assists, f.E as f_errors, 
bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_RBI, bat_caught, bat_baseballs, 
bat_intentwalks, bat_doubles, bat_triples, bat_strikes, bat_stolen, 
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, pitch_wild, pitch_games, pitch_saves,
pitch_earnruns, pitch_runsallowed, pitch_finish, pitch_outs, pitch_hits, pitch_opp_BA FROM Fielding f
LEFT JOIN
(SELECT b.playerID, b.lgID as bat_league, sum(b.R) as bat_runs, sum(b.H) as bat_hits, sum(b.CS) as bat_caught, sum(b.BB) as bat_baseballs,
sum(b.AB) as at_bats, sum(b.HR) as bat_homeruns, sum(b.RBI) as bat_RBI, sum(b.SO) as bat_strikes, sum(b.SB) as bat_stolen,
sum(b.IBB) as bat_intentwalks, sum(b.'2B') as bat_doubles, sum(b.'3B') as bat_triples,
sum(p.W) as pitch_wins, sum(p.SO) as pitch_strikes, sum(p.SHO) as pitch_shuts, avg(p.ERA) as pitch_ERA, sum(p.WP) as pitch_wild,
sum(p.G) as pitch_games, sum(p.SV) as pitch_saves, sum(p.ER) as pitch_earnruns, sum(p.R) as pitch_runsallowed, sum(p.GF) as pitch_finish, 
sum(p.IPOuts) as pitch_outs, sum(p.HBP) as pitch_hits, sum(p.BAOpp) as pitch_opp_BA
FROM Batting b
LEFT JOIN Pitching p on p.playerID = b.playerID
GROUP BY b.playerID) batpitch on batpitch.playerID = f.playerID
GROUP BY batpitch.playerID) positions
ON positions.playerID = h.playerID
WHERE h.yearID < 2000 AND h.yearID > 1965
GROUP BY h.playerID) all_features on pID = dom.playerID) all_data on pID = m.playerID
"""
df = pandas.read_sql(query, con)
con.close()

df.drop('p_name',  1, inplace = True)


#################
### Preprocessing
#################

# splitting out the explanatory features 
explanatory_features = [col for col in df.columns if col not in ['p_name', 'inducted']]
explanatory_df = df[explanatory_features]

# dropping rows with no data.
explanatory_df.dropna(how='all', inplace = True) 

# extracting column names 
explanatory_colnames = explanatory_df.columns

## doing the same for response
response_series = df.inducted
response_series.dropna(how='all', inplace = True) 

## seeing which explanatory feature rows got removed.  Looks like none.
response_series.index[~response_series.index.isin(explanatory_df.index)]

### now, let's seperate the numeric explanatory data from the string data
string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']


# that are all NANs, as they will show up as all 'Nothing' when we start binning or look for features with no variation)
string_features = string_features.fillna('Nothing')
# cleaning up string features
string_features = cleanup_data(string_features)
# binarizing string features 
encoded_data = get_binary_values(string_features)
## imputing features
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

## pulling together numeric and encoded data.
explanatory_df = pandas.concat([numeric_features, encoded_data],axis = 1)
explanatory_df.head()


#now, let's find features with no variance 
no_variation = find_zero_var(explanatory_df)
explanatory_df.drop(no_variation['toDelete'], inplace = True)

# deleting perfect correlation
no_correlation = find_perfect_corr(explanatory_df)
explanatory_df.drop(no_correlation['toRemove'], 1, inplace = True)

# scaling data
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)


#################
## RANDOM FORESTS
#################

# Random Forest object
# default values
rf = ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, 
    min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, 
    bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, min_density=None, 
    compute_importances=None)

# changing estimators
rf = ensemble.RandomForestClassifier(n_estimators = 500)

# computing ROC AUC of random forest
roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, 
                                scoring='roc_auc', n_jobs=-1)

# computing ROC AUC of decision tree just to have a benchmark                                
roc_score_tree = cross_val_score(tree.DecisionTreeClassifier(), explanatory_df, response_series,
                                 cv=10, scoring = 'roc_auc', n_jobs = -1)

# compare the mean ROC AUC
print "Random Forest Mean ROC AUC %f" % roc_scores_rf.mean()
print "Decision Tree Mean ROC AUC %f" % roc_score_tree.mean()
# by roc auc, 87% to 69%, much better


# perform grid search to find the optimal number of treees
trees_range = range(200, 500, 10)
param_grid = dict(n_estimators = trees_range)

grid = GridSearchCV(rf, param_grid, cv=10, scoring = 'roc_auc', n_jobs = -1)
grid.fit(explanatory_df, response_series)

# check out the scores
grid_mean_scores = [result[1] for result in grid.grid_scores_]

# plot results
plt.figure()
plt.plot(trees_range, grid_mean_scores)
# looks like about 250 trees


# pick best estimator and print ROC AUC
best_rf_est = grid.best_estimator_

# how many trees did the best estimator have?
print "Number of Trees in Best Estimator: %d" % best_rf_est.n_estimators
# what is the best score
print "Accuracy for Best Estimator : %f" % grid.best_score_
# 87.3%

##################
### BOOSTING TREES
##################

boosting_tree = ensemble.GradientBoostingClassifier()

roc_scores_gbm = cross_val_score(boosting_tree, explanatory_df, response_series, cv=10,
                                 scoring = 'roc_auc', n_jobs = -1)
                                 
# compare the mean ROC AUC
print "Boosting Tree Mean ROC AUC %f" % roc_scores_gbm.mean()   
print "Random Forest Mean ROC AUC %f" % roc_scores_rf.mean()
print "Decision Tree Mean ROC AUC %f" % roc_score_tree.mean()   


# tune for num_trees, learning rate, and subsampling percent
# import arange to create ranges for floats

from numpy import arange

# ranges suggested by scikit learn

learning_rate_range = arange(0.01, 0.4, 0.02) # most important feature/parameter to tune, over 0.04 isn't great
subsampling_range = arange(0.2, 1, 0.1) # 0 to 1
n_estimators_range = range(20, 100, 20)  # over 100 isn't usually helpful by experience

param_grid = dict(learning_rate = learning_rate_range, n_estimators = n_estimators_range, 
                  subsample = subsampling_range)

gbm_grid = GridSearchCV(boosting_tree, param_grid, cv=10, scoring='roc_auc', n_jobs = -1)
gbm_grid.fit(explanatory_df, response_series)

# small subsample = data is prone to overfitting
# low number of estimators = each has to have more importance, could mean edge cases have more meaning

# find the winning parameters
print gbm_grid.best_params_

# pull out the best score
print "Best Boosting Score: %f" % gbm_grid.best_score_


###################
### NEURAL NETWORKS
###################

# good for unsupervised neural network

# good for handwriting (China) simple
# face/voice recognition complex
# Watson complex

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model

logistic_classifier = linear_model.LogisticRegression()
neural_net = BernoulliRBM(random_state = 0, verbose = True)

# create pipeline of a neural net connected to a logistic regression
neural_classifier = Pipeline(steps=[('neural_net', neural_net), 
                    ('logistic_classifier', logistic_classifier)])
# they run in unison, puts result from one step as input to the next input

# cross-validate the entire pipeline like any old classifier
roc_scores_nn = cross_val_score(neural_classifier, explanatory_df, response_series, cv=10,
                scoring = 'roc_auc')                   

print "Neural Networks Mean ROC AUC %f" % roc_scores_nn.mean()

# let's do some grid search.
# i constrained this more than I should for the sake of time.
# i also commented out iteraton to speed things up -- 
# feel free to uncomment in your spare time.
learning_rate_range = arange(0.01, 0.2, 0.05)
iteration_range = range(30, 50, 5)
components_range = range(250, 500, 50)

# notice that I have the name of the item in the pipeline 
# followed by two underscores when I build the pipeline.
param_grid = dict(neural_net__n_components = components_range, 
                  neural_net__n_iter = iteration_range,                  
                  neural_net__learning_rate = learning_rate_range)
                  # for free time
                  
nn_grid = GridSearchCV(neural_classifier, param_grid, cv=5, scoring='roc_auc')
nn_grid.fit(explanatory_df, response_series)

print nn_grid.best_params_
# tuning parameters are not that intuitive
# learning rate is low, mostly very arbitrary

print "Best Neural Network Score: %f" % nn_grid.best_score_
# actual scores can be much better, currently went from 56% to 87%
print "Best Boosting Score: %f" % gbm_grid.best_score_
print "Best Random Tree Estimator Score : %f" % grid.best_score_


######################################################
### ROC Curve Accuracy of the GBM vs RF vs Tree Method
######################################################

from sklearn.cross_validation import train_test_split
from sklearn import metrics

xTrain, xTest, yTrain, yTest = train_test_split(explanatory_df, response_series, test_size=0.3)

tree_probabilities = pandas.DataFrame(tree.DecisionTreeClassifier().fit(xTrain, 
                yTrain).predict_proba(xTest))
rf_probabilities = pandas.DataFrame(best_rf_est.fit(xTrain, 
                yTrain).predict_proba(xTest))       
gbm_probabilities = pandas.DataFrame(gbm_grid.best_estimator_.fit(xTrain,
                yTrain).predict_proba(xTest)) 

tree_fpr, tree_tpr, thresholds = metrics.roc_curve(yTest, tree_probabilities[1])
rf_fpr, rf_tpr, thresholds = metrics.roc_curve(yTest, rf_probabilities[1])
gbm_fpr, gbm_tpr, thresholds = metrics.roc_curve(yTest, gbm_probabilities[1])                               

# tree probabilities = 1/0
# random forest probabilities = actual probabilities

# plotting
plt.figure()
plt.plot(tree_fpr, tree_tpr, color='g', label = "Decision Tree")
plt.plot(rf_fpr, rf_tpr, color = 'b', label = "Random Forest")
plt.plot(gbm_fpr, gbm_tpr, color='r', label = "Boosting Tree")
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve Accuracy of the GBM vs RF vs Tree Method')
plt.subplot()
plt.legend(bbox_to_anchor=(1.3,0.9))

# look at partial dependence plot on most important features for gbm

importances = pandas.DataFrame(gbm_grid.best_estimator_.feature_importances_, 
            index = explanatory_df.columns, columns = ['importance'])
            
importances.sort(columns = ['importance'], ascending = False, inplace = True)
print importances
# does not necessarily say whether it is a positive or negative importance

from sklearn.ensemble.partial_dependence import plot_partial_dependence

features = [i for i, j in enumerate(explanatory_df.columns.tolist()) 
            if j in importances.importance[0:3].index.tolist()]
# match feature importance for the first 3 importances
# i is index in list where the name occured - finds the feature
# j is the feature name

fig, axs = plot_partial_dependence(gbm_grid.best_estimator_, explanatory_df, features, 
                                   feature_names = explanatory_df.columns)
            

# compare the mean ROC AUC
print "Neural Networks Mean ROC AUC %f" % roc_scores_nn.mean()
print "Boosting Tree Mean ROC AUC %f" % roc_scores_gbm.mean()   
print "Random Forest Mean ROC AUC %f" % roc_scores_rf.mean()
print "Decision Tree Mean ROC AUC %f" % roc_score_tree.mean()   

