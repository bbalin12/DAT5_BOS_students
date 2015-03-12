# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:38:39 2015

@author: jchen
"""

import sqlite3
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn import ensemble
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import  GridSearchCV
from sklearn import tree
import matplotlib.pyplot as plt


# set option to display all columns
pd.set_option('display.max_columns', None)

# Functions from class for cleaning data
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
    all_columns = pd.DataFrame( index = data_frame.index)
    for col in data_frame.columns:
        data = pd.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pd.concat([all_columns, data], axis=1)
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
    corrMatrix.loc[:,:] =  np.tril(corrMatrix.values, k = -1)
    already_in = set()
    result = []
    for col in corrMatrix:
        perfect_corr = corrMatrix[col][abs(np.round(corrMatrix[col],10)) == 1.00].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    toRemove = []
    for item in result:
        toRemove.append(item[1:(len(item)+1)])
    toRemove = sum(toRemove, [])
    return {'corrGroupings':result, 'toRemove':toRemove}

#############
# Pull in same data as from hw 9, do the same preprocessing

# read in data
conn = sqlite3.connect('/Users/jchen/Documents/SQLite/lahman2013.sqlite')  
# pull in all metrics from hw #5 for players inducted into Hall of Fame
# optimize query for run time
# note that hall_of_fame_inductees already exists for players inducted before 2000
# pull in additional information about player's dominant team
sql = '''
select m.nameGiven as player_name,
       h.playerID,
       h.inducted,
       d.teamID,
       t.lgID as league,
       t.divID as division,
       b.*,
       p.*,
       f.*
from hall_of_fame_inductees h
left outer join Master m on m.playerID=h.playerID
left outer join
(select playerID, sum(AB) as at_bats, sum(R) as runs, sum(H) as hits, sum(RBI) as rbi, 
      (H+BB+HBP)*1.0/(AB+BB+SF+HBP) as OBP, (H+"2B"+("3B"*2)+(HR*3))*1.0/AB as SLG
 from Batting 
 group by playerID) as b on h.playerID=b.playerID
left outer join
(select playerID, sum(GS) as p_games_started, sum(CG) as p_complete_games, sum(SHO) as shutouts, sum(W) as p_wins, sum(IPOuts) as outs_pitched, (W + BB)/(IPOuts/3) as WHIP
 from Pitching
 group by playerID) as p on h.playerID=p.playerID
left outer join
(select playerID, sum(PO) as putouts, sum(A) as assists, sum(E) as errors
 from Fielding f
 group by playerID) as f on h.playerID=f.playerID
left outer join dominant_team_per_player d on h.playerID=d.playerID
left outer join Teams t on d.teamID=t.teamID
where b.playerID is not null
group by player_name, h.playerID, inducted, d.teamID
order by h.playerID;
'''
# read into data frame
df = pd.read_sql(sql, conn)
# close out connection
conn.close()

###############
# Preprocessing
###############

# drop duplicate playerID columns
df.drop('playerID',  1, inplace = True)

# Split out explanatory features
explanatory_features = [col for col in df.columns if col not in ['player_name', 'inducted']]
explanatory_df = df[explanatory_features]

explanatory_df.dropna(how='all', inplace = True) 

explanatory_colnames = explanatory_df.columns

# Response series
response_series = df.inducted
response_series.dropna(how='all', inplace = True) 

# Find rows where explanatory features were removed
response_series.index[~response_series.index.isin(explanatory_df.index)] 
# Looks like none.

# Separate numeric from string features
string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']

# Fill NaNs in string features
string_features = string_features.fillna('Nothing')

# Bin categorical features with threshold of 1%
string_features = cleanup_data(string_features, cutoffPercent = .01)
# Check the data
string_features.teamID.value_counts(normalize = True) 

# Encode categorical features
encoded_data = get_binary_values(string_features)
encoded_data.head()

# Impute vales for numeric features
from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features) # Store imputer in object
numeric_features = pd.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

# recombine numeric and categorical features into one df
explanatory_df = pd.concat([numeric_features, encoded_data],axis = 1)
explanatory_df.head()

# Find features with no variance
no_var = find_zero_var(explanatory_df)['toDelete']
explanatory_df.drop(no_var, inplace = True)
# nothing to drop

# Find features with perfect correlation
perf_cor = find_perfect_corr(explanatory_df)['toRemove']
explanatory_df.drop(perf_cor, 1, inplace = True)
# no features with perfect correlation

#############
# Scale data

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pd.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)

  
#################
## RANDOM FORESTS
#################

# Create a random forest classifier w/500 trees
rf = ensemble.RandomForestClassifier(n_estimators= 500)

# Compute ROC AUC of the random forest. 
roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

## let's compare the mean ROC AUC
print roc_scores_rf.mean()
# 88.9% - not bad

# Use grid search to find the optimal number of trees

trees_range = range(10, 600, 10)
param_grid = dict(n_estimators = trees_range)

grid = GridSearchCV(rf, param_grid, cv=10, scoring='roc_auc', n_jobs = -1)
grid.fit(explanatory_df, response_series)

grid_mean_scores = [result[1] for result in grid.grid_scores_]

# Plot grid search results
plt.figure()
plt.plot(trees_range, grid_mean_scores)
# Looks like it hits an early peak around 55, with a slightly higher peak at 405?

# let's pull out the best estimator and print its ROC AUC
best_rf_est = grid.best_estimator_
print best_rf_est.n_estimators # optimal n=410 
print grid.best_score_ # best accuracy score is 89% - not different from our default at n=500


#################
## BOOSTING TREES
#################
boosting_tree = ensemble.GradientBoostingClassifier()

roc_scores_gbm = cross_val_score(boosting_tree, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

print roc_scores_gbm.mean()
print roc_scores_rf.mean()
# Boosting tree mean ROC score: 88.7% 
# not significantly different from the random forest

# Tune for num trees, learning rate, and subsampling percent.

from numpy import arange # import arange to create ranges for floats

learning_rate_range = arange(0.01, 0.4, 0.02)
subsampling_range = arange(0.25, 1, 0.25)
n_estimators_range = range(10, 100, 10)

param_grid = dict(learning_rate = learning_rate_range, n_estimators = n_estimators_range, subsample = subsampling_range)

gbm_grid = GridSearchCV(boosting_tree, param_grid, cv=10, scoring='roc_auc', n_jobs = -1)
gbm_grid.fit(explanatory_df, response_series)

# Best params
print gbm_grid.best_params_
# {'n_estimators': 90, 'subsample': 0.5, 'learning_rate': 0.12999999999999998}

print gbm_grid.best_score_
# 0.893746842287 - pretty close to RF, but still below

# Plot ROC curve accuracy of the GBM vs RF
from sklearn.cross_validation import train_test_split
from sklearn import metrics

xTrain, xTest, yTrain, yTest = train_test_split(
                    explanatory_df, response_series, test_size =  0.3)

rf_probabilities = pd.DataFrame(best_rf_est.fit(xTrain, yTrain).predict_proba(xTest))
gbm_probabilities = pd.DataFrame(gbm_grid.best_estimator_.fit(xTrain, yTrain).predict_proba(xTest))

rf_fpr, rf_tpr, thresholds = metrics.roc_curve(yTest, rf_probabilities[1])
gbm_fpr, gbm_tpr, thresholds = metrics.roc_curve(yTest, gbm_probabilities[1])

plt.figure()
plt.plot(rf_fpr, rf_tpr, color = 'b')
plt.plot(gbm_fpr, gbm_tpr, color = 'r')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
# Interestingly, these curves seem to intersect
# It would appear that at some specificity levels, GBM is better for sensitivity than RF

# Partial dependence plot on the most important features for GBM
importances = pd.DataFrame(gbm_grid.best_estimator_.feature_importances_, index = explanatory_df.columns, columns =['importance'])
importances.sort(columns = ['importance'], ascending = False, inplace = True)
print importances
# assists, runs, errors most important
# teams least important

from sklearn.ensemble.partial_dependence import plot_partial_dependence
features = [i for i, j in enumerate(explanatory_df.columns.tolist()) if j in importances.importance[0:3].index.tolist()]
fig, axs = plot_partial_dependence(gbm_grid.best_estimator_, explanatory_df, features, feature_names = explanatory_df.columns)
# These are certainly not very smooth interactions
# Runs shows a very steep dependence curve

#################
# NEURAL NETWORKS
################

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model

logistic_classifier = linear_model.LogisticRegression()
neural_net = BernoulliRBM(random_state=0, verbose=True) 

# Set up pipeline
neural_classifier = Pipeline(steps=[('neural_net', neural_net), ('logistic_classifier', logistic_classifier)])

roc_scores_nn = cross_val_score(neural_classifier, explanatory_df, response_series, cv=10, scoring='roc_auc')

# compare scores
print roc_scores_nn.mean() 
# 0.761175217975 - pretty low in comparison
print roc_scores_gbm.mean()
print roc_scores_rf.mean()

# Grid search
learning_rate_range = arange(0.01, 0.4, 0.05)
iteration_range = range(30, 50, 5)
components_range = range(250, 600, 50)

param_grid = dict(neural_net__n_components = components_range, 
                  neural_net__learning_rate = learning_rate_range, 
                  neural_net__n_iter = iteration_range)

# Cross-validation
nn_grid = GridSearchCV(neural_classifier, param_grid, cv=10, scoring='roc_auc')
nn_grid.fit(explanatory_df, response_series)

# pull out best score
print nn_grid.best_score_
# 0.788142400659 - still pretty low in comparison
print gbm_grid.best_score_
print grid.best_score_

print nn_grid.best_params_

# In this grid search, the resulting accuracy is not much higher than the accuracy without any tuning
# With this data set it may not be worth it to run grid search as it is computationally expensive
# In this exercise, the GBM and RF are both comparable and better than the NN
# Our previously constructed Naive Bayes and Decision Tree did not produce very accuracte models