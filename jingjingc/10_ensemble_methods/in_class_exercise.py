# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 19:40:15 2015

@author: jchen
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


# including our functions from last week up here for use. 
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
    
    
## using the new table as part of the monster query from last class
monster_query = """
select m.nameGiven, d.teamID, m.weight, m.height, m.bats, m.throws, hfi.inducted, batting.*, pitching.*, fielding.* from hall_of_fame_inductees hfi 
left outer join master m on hfi.playerID = m.playerID
left outer join 
(
select playerID,  sum(AB) as total_at_bats, sum(H) as total_hits, sum(R) as total_runs, sum(HR) as total_home_runs, sum(SB) as total_stolen_bases,
sum(RBI) as total_RBI, sum(CS) as total_caught_stealing, sum(SO) as total_hitter_strikeouts, sum(IBB) as total_intentional_walks
from Batting
group by playerID
HAVING max(yearID) > 1950 and min(yearID) >1950 
)
batting on batting.playerID = hfi.playerID
left outer join
(
 select playerID, sum(G) as total_games_pitched, sum(SO) as total_shutouts, sum(sv) as total_saves, sum(IPouts) as total_outs_pitched, 
sum(H) as total_pitching_hits, sum(er) as total_pitching_earned_runs, sum(so) as total_pitcher_strikeouts, 
avg(ERA) as average_ERA, sum(WP) as total_wild_pitches, sum(HBP) as total_hit_by_pitch, sum(GF) as total_games_finished,
sum(R) as total_runs_allowed
from Pitching
group by playerID
) 
pitching on pitching.playerID = hfi.playerID 
LEFT OUTER JOIN
(
select playerID, sum(G) as total_games_fielded, sum(InnOuts) as total_time_in_field_with_outs, 
sum(PO) as total_putouts, sum(E) as total_errors, sum(DP) as total_double_plays
from Fielding
group by playerID
) 
fielding on fielding.playerID = hfi.playerID

LEFT OUTER JOIN dominant_team_per_player d on d.playerID = hfi.playerID
where batting.playerID is not null
"""

con = sqlite3.connect('/Users/jchen/Documents/SQLite/lahman2013.sqlite')
df = pandas.read_sql(monster_query, con)
con.close()

df.drop('playerID',  1, inplace = True)

#############
## repeating the same preprocessing from the previous lesson
############

## splitting out the explanatory features 
explanatory_features = [col for col in df.columns if col not in ['nameGiven', 'inducted']]
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

# create random forest object
rf = ensemble.RandomForestClassifier(n_estimators=500)

# compute ROC AUC of the random forest
roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring="roc_auc", n_jobs=-1)
# do the same for a decision tree
roc_score_tree = cross_val_score(tree.DecisionTreeClassifier(), explanatory_df, response_series, cv=10, scoring="roc_auc", n_jobs=-1)

# compare the mean ROC AUC
print roc_scores_rf.mean()
print roc_score_tree.mean()

# perform grid search to find the optimal number of trees
trees_range = range(10, 550, 10)
param_grid = dict(n_estimators = trees_range)

grid = GridSearchCV(rf, param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
grid.fit(explanatory_df, response_series)

# scores of the grid search
grid_mean_scores = [result[1] for result in grid.grid_scores_]

# plot the results of grid search
plt.figure()
plt.plot(trees_range, grid_mean_scores)

# pull out the best estimator
best_decision_tree_est = grid.best_estimator_
print best_decision_tree_est.n_estimators
print grid.best_score_

#################
## BOOSTING TREES
#################

boosting_tree = ensemble.GradientBoostingClassifier()

roc_scores_gbm = cross_val_score(boosting_tree, explanatory_df, response_series, cv=10, scoring="roc_auc", n_jobs=-1)

# comparing accuracies
print roc_scores_gbm.mean()
print roc_scores_rf.mean()
print roc_score_tree.mean()

# tune for num_trees, learning rate, and subsampling percent
from numpy import arange # need arange to create ranges for floats

learning_rate_range = arange(0.01, 0.4, 0.02) # most important feature to tune - bad learning rate will overfit
subsampling_range = arange(0.25, 1, .25)
n_estimators_range = range(25, 100, 25)


param_grid = dict(learning_rate = learning_rate_range, n_estimators = n_estimators_range,
                  subsample = subsampling_range)
                  
                  
gbm_grid = GridSearchCV(boosting_tree, param_grid, cv=10, scoring="roc_auc", n_jobs=-1)
gbm_grid.fit(explanatory_df, response_series)

# find the winning parameters
print gbm_grid.best_params_
# note default settings: estimators = 100, subsample = 1.0, learning_rate = 0.1

print gbm_grid.best_score_
print grid.best_score_ # random forest score
# gbm underperforms here slightly


# plot ROC curve accuracy of GBM vs RF vs Tree 
from sklearn.cross_validation import train_test_split
from sklearn import metrics

# split test and train sets
xTrain, xTest, yTrain, yTest = train_test_split(
                    explanatory_df, response_series, test_size =  0.3)


tree_probabilities = pandas.DataFrame(tree.DecisionTreeClassifier().fit(xTrain, yTrain).predict_proba(xTest))
rf_probabilities = pandas.DataFrame(best_decision_tree_est.fit(xTrain, yTrain).predict_proba(xTest))
gbm_probabilities = pandas.DataFrame(gbm_grid.best_estimator_.fit(xTrain, yTrain).predict_proba(xTest))
# note that these predict two classes of probabilities
# we are really only interested in the positive class

tree_fpr, tree_tpr, thresholds = metrics.roc_curve(yTest, tree_probabilities[1])
rf_fpr, rf_tpr, thresholds = metrics.roc_curve(yTest, rf_probabilities[1])
gbm_fpr, gbm_tpr, thresholds = metrics.roc_curve(yTest, gbm_probabilities[1])


plt.figure()
plt.plot(tree_fpr, tree_tpr, color = 'g')
plt.plot(rf_fpr, rf_tpr, color = 'b')
plt.plot(gbm_fpr, gbm_tpr, color = 'r')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

# create partial dependence plot on most important features for gbm

importances = pandas.DataFrame(gbm_grid.best_estimator_.feature_importances_, 
                               index=explanatory_df.columns, columns=['Importance'])

importances.sort(columns=['Importance'], ascending = False, inplace = True)
print importances # cannot tell from importance whether positive or negative


from sklearn.ensemble.partial_dependence import plot_partial_dependence

# iterate through explanatory columns, only return top 3 most important features 
features = [i for i, j in enumerate(explanatory_df.columns.tolist()) if j in importances.Importance[0:3].index.tolist()]

fig, axs = plot_partial_dependence(gbm_grid.best_estimator_, explanatory_df, features, feature_names = explanatory_df.columns)

#################
# NEURAL NETWORKS
#################

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model

logistic_classifier = linear_model.LogisticRegression()
neural_net = BernoulliRBM(random_state=0, verbose=True)

# create the pipeline - output of neural net passed to logistic classifier
neural_classifier = Pipeline(steps=[('neural_net', neural_net), ('logistic_classifier', logistic_classifier)])

# cross-validate the entire pipeline like any old classifier.
roc_scores_nn = cross_val_score(neural_classifier, explanatory_df, response_series, cv=10, scoring='roc_auc')

#let's compare our accuracies
print roc_scores_nn.mean()

print roc_scores_gbm.mean()
print roc_scores_rf.mean()
print roc_score_tree.mean()

# nn classifier pretty poor
# this is without tuning any parameters

# Grid search constrained for the sake of time.
learning_rate_range = arange(0.01, 0.2, 0.05)
# iteration_range = range(30, 50, 5) 
components_range = range(250, 500, 50)

# Now we are tuning in a pipeline
# so we need the name of the item in the pipeline followed by two underscores 
param_grid = dict(neural_net__n_components = components_range, neural_net__learning_rate = learning_rate_range
#, neural_net__n_iter = iteration_range
)

# 5-fold CV here for reasons of time
nn_grid = GridSearchCV(neural_classifier, param_grid, cv=5, scoring='roc_auc')
nn_grid.fit(explanatory_df, response_series)

# pull out best score
print nn_grid.best_score_
# compare to other grid best scores
print gbm_grid.best_score_
print grid.best_score_
print nn_grid.best_params_
# so the grid seacrch best score is tons better than the 
# original, but lags rf and gbm.  You can probably meet or surpass
# rf or GBM with a full grid search, but this will take a lot of time.

