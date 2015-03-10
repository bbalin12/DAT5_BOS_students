# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 08:36:33 2015

@author: garauste
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

# Putting a setting into pandas that lets you print out the entire 
# Datframe when you use the .head() method
pandas.set_option('display.max_columns',None)

conn = sqlite3.connect('C:\Users\garauste\Documents\SQLite\lahman2013.sqlite')

sql = '''
select coalesce(a.nameFirst ,"") || coalesce(a.nameLast,"") as PlayerName, hfi.inducted,
a.playerID, a.birthState, b.*, c.*,d.*
from hall_of_fame_inductees hfi
left outer join master a
on a.playerID = hfi.playerID
left outer join 
(
select sum(b.G) as Games, sum(b.H) as Hits, sum(b.AB) as At_Bats, sum(b.HR) as Homers,
b.playerID from Batting b group by b.playerID
) b
on hfi.playerID = b.playerID
left outer join 
(
select d.playerID, sum(d.A) as Fielder_Assists, sum(d.E) as Field_Errors, 
sum(d.DP) as Double_Plays from Fielding d group by d.playerID 
) c
on hfi.playerID = c.playerID
left outer join
(
select b.playerID, sum(b.ERA) as Pitcher_Earned_Run_Avg, sum(b.W) as Pitcher_Ws, sum(b.SHO) as Pitcher_ShutOuts, sum(b.SO) as Pitcher_StrikeOuts,
sum(b.HR) as HR_Allowed, sum(b.CG) as Complete_Games 
from Pitching b 
group by b.playerID
) d 
on hfi.playerID = d.playerID;
'''

df = pandas.read_sql(sql, conn)
conn.close()

## getting an initial view of the data for validation
df.head(10)
df.columns

# Dropping duplicate playerIDs columns
df.drop('playerID',1,inplace=True)

###################################################################
######## Importing Functions created in inclass exercise ##########
###################################################################

# Function to cut off data that has less than 1% of all volume
def cleanup_data(df, cutoffPercent = .01):
    for col in df:    
        sizes = df[col].value_counts(normalize=True) # normalize = True gives percentages
        # get the names of the levels that make up less than 1% of the dataset
        values_to_delete = sizes[sizes<cutoffPercent].index
        df[col].ix[df[col].isin(values_to_delete)] = 'Other'
    return df

# Function to create binary dummies for catergorical data    
def get_binary_values(data_frame):
    """encodes the categorical features in Pandas
    """
    all_columns = pandas.DataFrame(index=data_frame.index)
    for col in data_frame.columns:
        data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii','replace'))
        all_columns = pandas.concat([all_columns,data],axis=1)
    return all_columns

# Function to find variables with no variance at all - Need to Impute before this step
def find_zero_var(df):
    """ find the columns in the dataframe with zero variance -- ie those 
        with the same value in every observation
    """
    toKeep = []
    toDelete = []
    for col in df:
        if len(df[col].value_counts())>1:
            toKeep.append(col)
        else:
            toDelete.append(col)
    #
    return {'toKeep':toKeep, 'toDelete':toDelete}
    
# Function to find the variables with perfect correlation
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
    

#############
## repeating the same preprocessing from the previous lesson
############

## splitting out the explanatory features 
explanatory_features = [col for col in df.columns if col not in ['PlayerName', 'inducted']]
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

## Return the dataframe, scalar and imputer

##############
## Random Forests
##############

# Creating a random forest object
rf = ensemble.RandomForestClassifier(n_estimators=500)

## let's compute ROC AUC of the randome forest
roc_scores_rf = cross_val_score(rf,explanatory_df, response_series, cv=10, 
                                scoring = 'roc_auc')
                            
# DO the same for a decision tree
roc_score_tree = cross_val_score(tree.DecisionTreeClassifier(),explanatory_df,
 response_series, cv=10, scoring='roc_auc')
 
# Compare the mean
print roc_scores_rf.mean()
print roc_score_tree.mean()
# RF massive outperformance

# perform grid search to find the optimal number of trees

trees_range = range(10,550,10)
param_grid = dict(n_estimators = trees_range)

grid = GridSearchCV(rf, param_grid, cv=10, scoring = 'roc_auc')
grid.fit(explanatory_df, response_series)

# check out the scores of the grid search
grid_mean_scores = [result[1] for result in grid.grid_scores_]

# plot the results of the grid search
plt.figure()
plt.plot(trees_range, grid_mean_scores)

# pull out the best estimator and print it's ROC AUC
best_descision_tree_est = grid.best_estimator_
# how many trees did the best estimator have
print best_descision_tree_est.n_estimators
###
# Interesting to note that only 40 trees is the optimal number of estimators
###
# how accurate was the best estimator
print grid.best_score_
# Accuracy only increased by 1% after using gridsearch

#########################
## Boosting Trees 
#########################
boosting_tree = ensemble.GradientBoostingClassifier()

roc_scores_gbm = cross_val_score(boosting_tree, explanatory_df, response_series,
                                 cv = 10, scoring = 'roc_auc')
                                 
# Compare the accuracies
print roc_scores_gbm.mean()
print roc_scores_rf.mean()
print roc_score_tree.mean()
## Random Forests are still the best predicter but they only slightly outperform
## Boosting Trees


# Let's tune for num_trees, learning_rate, and subsampling percent.
# need to import arrange to create ranges for floats
from numpy import arange

learning_rate_range = arange (0.01, 0.4, 0.02)
subsampling_range = arange(0.25, 1, 0.25)
n_estimators_range = arange(25,100,25)

param_grid = dict(learning_rate = learning_rate_range, n_estimators=
n_estimators_range, subsample = subsampling_range)

gbm_grid = GridSearchCV(boosting_tree, param_grid, cv=10, scoring = 'roc_auc')
gbm_grid.fit(explanatory_df, response_series)

# find the winning paramters
print gbm_grid.best_params_
# Best Gradient Booster has 50 estimators

# pull out best score
print gbm_grid.best_score_
print grid.best_score_
# The RF still Slightly outperforms GBM

# ROC Curve of GBM vs RF vs Tree method
from sklearn.cross_validation import train_test_split
from sklearn import metrics

xTrain, xTest, yTrain, yTest = train_test_split(explanatory_df, response_series, test_size=0.3)

tree_probabilities = pandas.DataFrame(tree.DecisionTreeClassifier().fit(xTrain,yTrain).predict_proba(xTest))
rf_probabilities = pandas.DataFrame(best_descision_tree_est.fit(xTrain,yTrain).predict_proba(xTest))
gbm_probabilities = pandas.DataFrame(gbm_grid.best_estimator_.fit(xTrain,yTrain).predict_proba(xTest))

tree_fpr, tree_tpr, thresholds = metrics.roc_curve(yTest, tree_probabilities[1])
rf_fpr, rf_tpr, thresholds = metrics.roc_curve(yTest, rf_probabilities[1])
gbm_fpr, gbm_tpr, thresholds = metrics.roc_curve(yTest, gbm_probabilities[1])

plt.figure()
plt.plot(tree_fpr, tree_tpr, color = 'g')
plt.plot(rf_fpr, rf_tpr, color = 'b')
plt.plot(gbm_fpr, gbm_tpr, color = 'r')
plt.xlabel('False Positive Rates (1-Specificity)')
plt.ylabel('True Positives Rate (Sesitivity)')
# from this chart it seems like GBM will outperform RF at lower TPR-FPR rates


####################
## Neural Networks
####################

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model

logistic_classifier = linear_model.LogisticRegression()
neural_net = BernoulliRBM(random_state=0,verbose=True)

# create a pipeline of a neural net connected to a logistic regression
neural_classifier = Pipeline(steps=[('neural_net',neural_net),('logistic_classifier',
logistic_classifier)])

# you can cross-validate the entire pipeline like any old classifier
roc_scores_nn = cross_val_score(neural_classifier, explanatory_df, response_series,
cv=10, scoring='roc_auc')

# lET'S compare accracies
print roc_scores_nn.mean()
print roc_scores_gbm.mean()
print roc_scores_rf.mean()
print roc_score_tree.mean()
# The Neural Network performs worse than all other classifiers. RF still has the
# highest ROC score

## Ranges
learning_rate_range = arange(0.01,0.2,0.05)
# iteration_range = range(30,50,5)
components_range = range(250,500,50)


# Grid Search time
param_grid = dict(neural_net__n_components = components_range,
neural_net__learning_rate = learning_rate_range
# , neural_net__n_iter = iteration_range
)

# do 5 fold grid search
nn_grid = GridSearchCV(neural_classifier, param_grid, cv=5, scoring = 'roc_auc')
nn_grid.fit(explanatory_df,response_series)

# Pull out best score
print nn_grid.best_score_
# compare to other grid best scores
print gbm_grid.best_score_
print grid.best_score_
print nn_grid.best_params_

# Neural networks even with GridSearch did not perform well. 

## NOTE: Not comparing to Naive Bayes or linear regression HWs as I have changed 
## the dataset since then. I will compare to the logistic regression HW once that 
## has been completed