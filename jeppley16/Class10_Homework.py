# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 14:13:41 2015

@author: jeppley
"""

import pandas as pandas
import sqlite3 as sql
from sklearn.preprocessing import Imputer as imp
import matplotlib.pyplot as plt
import numpy as numpy
from sklearn import preprocessing as pre
from sklearn.feature_selection import RFECV
from sklearn import tree
from sklearn.grid_search import  GridSearchCV
from sklearn.cross_validation import cross_val_score as cv
from sklearn import ensemble
from numpy import arange
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model


#########################
#### Data from HW 9  ####
#########################




con = sql.connect('C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite')

modeldata = '''select h.*, 
  b.b_atbat, b.b_hits, p.p_wins, f.f_puts, t.teamID, o.pos
  from 
  (select playerid, inducted
  from hall_of_fame_inductees_3 
   where category = 'Player'
   group by playerid) h
left outer join 
  (select playerid,
    count(distinct yearid) as b_years,
    sum(ab) as b_atbat, 
    sum(r) as b_runs, 
    sum(h) as b_hits, 
    sum(hr) as b_hruns, 
    sum(sb) as b_stbas,
    sum(so) as b_strik
  from batting
  group by playerid 
  HAVING max(yearID) > 1950 and min(yearID) >1950 ) b
  on h.playerid = b.playerid
left outer join
  (select playerid,
    count(distinct yearid) as p_years,
    sum(w) as p_wins,
    sum(l) as p_loss,
    sum(sho) as p_shout,
    sum(sv) as p_saves,
    sum(er) as p_eruns,
    sum(so) as p_stout
  from pitching
  group by playerid) p
  on h.playerid = p.playerid
left outer join
  (select playerid,
    max(teamID) as teamID
  from dominant_team_per_player
  group by playerid) t
  on h.playerid = t.playerid
left outer join
  (select playerid,
    max(POS) as pos
  from dominant_pos_per_player
  group by playerid) o
  on h.playerid = o.playerid    
left outer join
  (select playerid,
     count(distinct yearid) as f_years,
     sum(po) as f_puts,
     sum(a) as f_assis,
     sum(dp) as f_dplay,
     sum(pb) as f_pass
  from fielding
  group by playerid) f
  on h.playerid = f.playerid
  where b.playerID is not null
;'''


fore = pd.read_sql(modeldata, con)
con.close()

fore.head(10)
fore.columns
fore.describe()

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
    

fore.drop('playerid',  1, inplace = True)
    
## splitting out the explanatory features 
explanatory_featuresfore = [col for col in fore.columns if col not in ['nameGiven', 'inducted']]
explanatory_dffore = fore[explanatory_featuresfore]

# dropping rows with no data.
explanatory_dffore.dropna(how='all', inplace = True)

# extracting column names 
explanatory_colnamesfore = explanatory_dffore.columns 

## doing the same for response
response_seriesfore = fore.inducted
response_seriesfore.dropna(how='all', inplace = True) 

## seeing which explanatory feature rows got removed.  Looks like none.
response_seriesfore.index[~response_seriesfore.index.isin(explanatory_dffore.index)]

### now, let's seperate the numeric explanatory data from the string data
string_featuresfore = explanatory_dffore.ix[:, explanatory_dffore.dtypes == 'object']
numeric_featuresfore = explanatory_dffore.ix[:, explanatory_dffore.dtypes != 'object']

# that are all NANs, as they will show up as all 'Nothing' when we start binning or look for features with no variation)
string_featuresfore = string_featuresfore.fillna('Nothing')
# cleaning up string features
string_featuresfore = cleanup_data(string_featuresfore)
# binarizing string features 
encoded_datafore = get_binary_values(string_featuresfore)
## imputing features
imputer_object = imp(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_featuresfore)
numeric_featuresfore = pandas.DataFrame(imputer_object.transform(numeric_featuresfore), columns = numeric_featuresfore.columns)

## pulling together numeric and encoded data.
explanatory_dffore = pandas.concat([numeric_featuresfore, encoded_datafore],axis = 1)
explanatory_dffore.head()


#now, let's find features with no variance 
no_variationfore = find_zero_var(explanatory_dffore)
explanatory_dffore.drop(no_variationfore['toDelete'], inplace = True)

# deleting perfect correlation
no_correlationfore = find_perfect_corr(explanatory_dffore)
explanatory_dffore.drop(no_correlationfore['toRemove'], 1, inplace = True)

# scaling data
scalerfore = pre.StandardScaler()
scalerfore.fit(explanatory_dffore)
explanatory_dffore = pandas.DataFrame(scalerfore.transform(explanatory_dffore), columns = explanatory_dffore.columns)


#################
## RANDOM FORESTS
#################


# creating a random forest object.
## these are the default values of the classifier
rfhw = ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, min_density=None, compute_importances=None)

# I'm going to change this a bit. Instantiates the object
rfhw = ensemble.RandomForestClassifier(n_estimators= 500)

roc_scores_rfhw = cv(rfhw, explanatory_dffore, response_seriesfore, cv=10, scoring='roc_auc')

# let's do the same for the decision tree
roc_score_treehw = cv(tree.DecisionTreeClassifier(), explanatory_dffore, response_seriesfore, cv=10, scoring='roc_auc')

## let's compare the mean ROC AUC
print roc_scores_rfhw.mean()
print roc_score_treehw.mean()
#The random forest indeed is much better in accuracy here than the regular decision tree.

## perform grid search to find the optimal number of trees (tuning some parameters)

trees_rangehw = range(10, 550, 10) #see what accuracy is like
param_gridhw = dict(n_estimators = trees_rangehw)#tuning parameters is number estimators

gridhw = GridSearchCV(rfhw, param_gridhw, cv=10, scoring='roc_auc') 
gridhw.fit(explanatory_dffore, response_seriesfore) # often will want to do this after night, and after feature selection 

# Check out the scores of the grid search
grid_mean_scoreshw = [result[1] for result in gridhw.grid_scores_]



# Plot the results of the grid search
plt.figure()
plt.plot(trees_rangehw, grid_mean_scoreshw)


best_rf_tree_esthw = gridhw.best_estimator_
# how many trees did the best estiator have? 
print best_rf_tree_esthw.n_estimators
# how accurate was the best estimator?
print gridhw.best_score_

#This did improve accuracy a bit from the 88% it was before to 92% now with grid search



#################
## BOOSTING TREES
#################
boosting_treehw = ensemble.GradientBoostingClassifier()

roc_scores_gbmhw = cv(boosting_treehw, explanatory_dffore, response_seriesfore, cv=10, scoring='roc_auc')

#let's compare our accuracies
print roc_scores_gbmhw.mean()
print roc_scores_rfhw.mean()
print roc_score_treehw.mean()
# boosting tree does ever so slightly better than the random forest (91% to 90%)


# let's tune for num_trees, learning rate, and subsampling percent.
# need to import arange to create ranges for floats
from numpy import arange #pythons range function doesn't allow you to do floats

learning_rate_rangehw = arange(0.01, 0.4, 0.02)
subsampling_rangehw = arange(0.25, 1, 0.25)
n_estimators_rangehw = range(25, 100, 25) #less than RF because by definition you are boosting

param_gridhw = dict(learning_rate = learning_rate_rangehw, n_estimators = n_estimators_rangehw, subsample = subsampling_rangehw)

gbm_gridhw = GridSearchCV(boosting_treehw, param_gridhw, cv=10, scoring='roc_auc')
gbm_gridhw.fit(explanatory_dffore, response_seriesfore)

# find the winning parameters
print gbm_gridhw.best_params_
# how does this compare to the default settings
# estimators = 75, subsample = 0.75, learning_rate = 0.21

# pull out the best score
print gbm_gridhw.best_score_
print gridhw.best_score_
## GBM actually performs better here than the RF

## ROC curve accuracy of the GBM vs RF vs Tree Method

#not doing on all the CV splits
from sklearn.cross_validation import train_test_split
from sklearn import metrics

xTrain, xTest, yTrain, yTest = train_test_split(
                    explanatory_dffore, response_seriesfore, test_size =  0.3)


#comparing ROCs of our best estimators that came out of grid search
tree_probabilitieshw = pandas.DataFrame(tree.DecisionTreeClassifier().fit(xTrain, yTrain).predict_proba(xTest))#wrap in data frame because 2 columns of probabilities, one for 0 class and 1 class, pandas data frame easy to extract
rf_probabilitieshw = pandas.DataFrame(best_rf_tree_esthw.fit(xTrain, yTrain).predict_proba(xTest))
gbm_probabilitieshw = pandas.DataFrame(gbm_gridhw.best_estimator_.fit(xTrain, yTrain).predict_proba(xTest))


tree_fpr, tree_tpr, thresholds = metrics.roc_curve(yTest, tree_probabilitieshw[1])
rf_fpr, rf_tpr, thresholds = metrics.roc_curve(yTest, rf_probabilitieshw[1])
gbm_fpr, gbm_tpr, thresholds = metrics.roc_curve(yTest, gbm_probabilitieshw[1])


plt.figure()
plt.plot(tree_fpr, tree_tpr, color = 'g')
plt.plot(rf_fpr, rf_tpr, color = 'b')
plt.plot(gbm_fpr, gbm_tpr, color = 'r')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
## GBM does better up to a particular  point, but if we are maximizing true positive while minimizing
#false positives, it looks like random forest frankly wins out


#################
# NEURAL NETWORKS
################

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model

logistic_classifierhw = linear_model.LogisticRegression()
neural_nethw = BernoulliRBM(random_state=0, verbose=True) 

# create the pipeline of a neural net connected to a logistic regression.
neural_classifierhw = Pipeline(steps=[('neural_net', neural_nethw), ('logistic_classifier', logistic_classifierhw)])
#both run in unison with one another, pipeline returns result from one thing in pipeline to next thing in pipeline

## you can cross-validate the entire pipeline like any old classifier.
roc_scores_nnhw = cv(neural_classifierhw, explanatory_dffore, response_seriesfore, cv=10, scoring='roc_auc')


#let's compare our accuracies
print roc_scores_nnhw.mean()

print roc_scores_gbmhw.mean()
print roc_scores_rfhw.mean()
print roc_score_treehw.mean()
# not so great, eh?

# let's do some grid search.
# i constrained this more than I should for the sake of time.
# i also commented out iteraton to speed things up -- 
# feel free to uncomment in your spare time.
learning_rate_rangehw = arange(0.01, 0.2, 0.05)
#iteration_range = range(30, 50, 5)
components_rangehw = range(250, 500, 50)

# notice that I have the name of the item in the pipeline 
# followed by two underscores when I build the pipeline.
param_gridhw = dict(neural_net__n_components = components_rangehw, neural_net__learning_rate = learning_rate_rangehw
#, neural_net__n_iter = iteration_range
)

# doing 5-fold CV here for reasons of time; feel free to do 10-fold 
# in your own leisure.
nn_gridhw = GridSearchCV(neural_classifierhw, param_gridhw, cv=5, scoring='roc_auc')
nn_gridhw.fit(explanatory_dffore, response_seriesfore)

## pull out best score
print nn_gridhw.best_score_
## compare to other grid best scores
print gbm_gridhw.best_score_
print gridhw.best_score_
# so the grid seacrch best score is tons better than the 
# original, but lags rf and gbm.  You can probably meet or surpass
# rf or GBM with a full grid search, but this will take a lot of time.


################################
#### METHOD COMPARISON #########
################################

#roc

#Neural Nets: 86%
#Boosting: 93%
#RF: 91%
#Decision Tree: 88%
#Naive Bayes: 47%, from Homework 8




