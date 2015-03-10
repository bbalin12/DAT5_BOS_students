# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 21:30:09 2015

@author: melaccor
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
###
    
# putting a setting into pandas that lets you print out the entire
# DataFrame when you use the .head() method
pandas.set_option('display.max_columns', None)

#categorical feature that shows the dominant team played per player
database = r'C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite'
conn = sqlite3.connect(database)
query = 'select playerID, teamID from Batting;'
df = pandas.read_sql(query, conn)
conn.close()

# use pandas.DataFrame.groupby and an annonymous lambda function to pull the mode team for each player
majority_team_by_player = df.groupby(['playerID']).agg(lambda x:x.value_counts().idxmax())

## write the data back to the database
conn = sqlite3.connect(database)
majority_team_by_player.to_sql('dominant_team_per_player', conn, if_exists = 'replace')
# closing the connection.
conn.close()

##Create another categorical variable
database = r'C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite'
conn = sqlite3.connect(database)

query = 'select playerID, lgID from Batting'
df = pandas.read_sql(query, conn)
conn.close()

# use pandas.DataFrame.groupby and an annonymous lambda function
# to pull the mode team for each player
majority_league_by_player = df.groupby(['playerID']).agg(lambda x:x.value_counts().idxmax())

## write the data back to the database
database = r'C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite'
conn = sqlite3.connect(database)
majority_league_by_player.to_sql('dominant_league_per_player', conn, if_exists = 'replace')
# closing the connection.
conn.close()
    
## using the new table as part of the query from last homework
monster_query = """
select a.playerID, a.inducted as inducted, batting.*, pitching.*, fielding.*, lg.*, player.* from
(select playerID, case when avginducted = 0 then 0 else 1 end as inducted from 
(select playerID, avg(case when inducted = 'Y' then 1 else 0 end ) as avginducted from HallOfFame 
where yearid < 2000
group by playerID)) a 
left outer join
(select playerID,  sum(AB) as atbats, sum(H) as totalhits, sum(R) as totalruns, sum(HR) as totalhomeruns, sum(SB) as stolenbases, sum(RBI) as totalRBI, sum(SO) as strikeouts, sum(IBB) as intentionalwalks
from Batting
group by playerID) batting on batting.playerID = a.playerID
left outer join(select playerID, sum(G) as totalgames, sum(SO) as shutouts, sum(sv) as totalsaves, sum(er) as earnedruns, sum(WP) as wildpitches
from Pitching
group by playerID) pitching on pitching.playerID = a.playerID 
left outer join
(select playerID, sum(InnOuts) as timewithouts, sum(PO) as putouts, sum(E) as errors, sum(DP) as doubleplays
from Fielding
group by playerID) fielding on fielding.playerID = a.playerID
left outer join
dominant_team_per_player player on player.playerID = a.playerID
left outer join 
dominant_league_per_player lg on lg.playerID = a.playerID;
"""

##Create another categorical variable
database = r'C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite'
conn = sqlite3.connect(database)
df = pandas.read_sql(monster_query, conn)
conn.close()

## getting an intial view of the data for validation
df.head(10)
df.columns

# dropping duplicate playerID columns
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
print no_variation
explanatory_df.drop(no_variation['toDelete'], inplace = True)
#Nothing to delete

# deleting perfect correlation
no_correlation = find_perfect_corr(explanatory_df)
explanatory_df.drop(no_correlation['toRemove'], 1, inplace = True)

# scaling data
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)

explanatory_df.isnull().sum()
#atbats              0
#totalhits           0
#totalruns           0
#totalhomeruns       0
#stolenbases         0
#totalRBI            0
#strikeouts          0
#intentionalwalks    0
#totalgames          0
#shutouts            0
#totalsaves          0
#earnedruns          0
#wildpitches         0
#timewithouts        0
#putouts             0
#errors              0
#doubleplays         0
#a.playerID_Other    0
#lgID_AL             0
#lgID_NL             0
#lgID_Other          0
#teamID_BAL          0
#teamID_BOS          0
#teamID_BRO          0
#teamID_BSN          0
#teamID_CAL          0
#teamID_CHA          0
#teamID_CHN          0
#teamID_CIN          0
#teamID_CLE          0
#teamID_DET          0
#teamID_HOU          0
#teamID_KCA          0
#teamID_LAN          0
#teamID_MIN          0
#teamID_ML4          0
#teamID_MON          0
#teamID_NY1          0
#teamID_NYA          0
#teamID_NYN          0
#teamID_Nothing      0
#teamID_Other        0
#teamID_PHA          0
#teamID_PHI          0
#teamID_PIT          0
#teamID_SFN          0
#teamID_SLA          0
#teamID_SLN          0
#teamID_WS1          0
#dtype: int64

explanatory_df[explanatory_df.atbats.isnull()]

# dropping rows with no data in any of the columns.
explanatory_df.dropna(how='any', inplace = True)
explanatory_df.isnull().sum()
#atbats              0
#totalhits           0
#totalruns           0
#totalhomeruns       0
#stolenbases         0
#totalRBI            0
#strikeouts          0
#intentionalwalks    0
#totalgames          0
#shutouts            0
#totalsaves          0
#earnedruns          0
#wildpitches         0
#timewithouts        0
#putouts             0
#errors              0
#doubleplays         0
#a.playerID_Other    0
#lgID_AL             0
#lgID_NL             0
#lgID_Other          0
#teamID_BAL          0
#teamID_BOS          0
#teamID_BRO          0
#teamID_BSN          0
#teamID_CAL          0
#teamID_CHA          0
#teamID_CHN          0
#teamID_CIN          0
#teamID_CLE          0
#teamID_DET          0
#teamID_HOU          0
#teamID_KCA          0
#teamID_LAN          0
#teamID_MIN          0
#teamID_ML4          0
#teamID_MON          0
#teamID_NY1          0
#teamID_NYA          0
#teamID_NYN          0
#teamID_Nothing      0
#teamID_Other        0
#teamID_PHA          0
#teamID_PHI          0
#teamID_PIT          0
#teamID_SFN          0
#teamID_SLA          0
#teamID_SLN          0
#teamID_WS1          0
#dtype: int64
response_series.head()
response_series.value_counts() 
#0    704
#1    244
#dtype: int64

print explanatory_df.index
print response_series.index
#Int64Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, ...], dtype='int64')
#Int64Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, ...], dtype='int64')

missing_rows = response_series.index[~response_series.index.isin(explanatory_df.index)]
print missing_rows

response_series2 = response_series.drop(missing_rows,axis=0)
missing_rows = response_series2.index[~response_series2.index.isin(explanatory_df.index)]
print missing_rows

resp_missing = explanatory_df.index[~explanatory_df.index.isin(explanatory_df.index)]
print resp_missing

# scaling data
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)
explanatory_df.describe()

response_series = None
response_series = response_series2
print response_series.describe()
print response_series.value_counts()
#count    948.000000
#mean       0.257384
#std        0.437423
#min        0.000000
#25%        0.000000
#50%        0.000000
#75%        1.000000
#max        1.000000
#dtype: float64
#0    704
#1    244
#dtype: int64

#################
## RANDOM FORESTS
#################

# creating a random forest object.
## these are the default values of the classifier
rf = ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, min_density=None, compute_importances=None)

# I'm going to change this a bit.
rf = ensemble.RandomForestClassifier(n_estimators= 500)

# let's compute ROC AUC of the random forest. 
roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring='roc_auc')

# let's do the same for the decision tree
roc_score_tree = cross_val_score(tree.DecisionTreeClassifier(), explanatory_df, response_series, cv=10, scoring='roc_auc')

## let's compare the mean ROC AUC
print roc_scores_rf.mean()
#0.914078722334
print roc_score_tree.mean()
#0.760437290409
## RF does really well, eh? 

## perform grid search to find the optimal number of trees

trees_range = range(10, 550, 10)
param_grid = dict(n_estimators = trees_range)

grid = GridSearchCV(rf, param_grid, cv=10, scoring='roc_auc')
grid.fit(explanatory_df, response_series)

# Check out the scores of the grid search
grid_mean_scores = [result[1] for result in grid.grid_scores_]

# Plot the results of the grid search
plt.figure()
plt.plot(trees_range, grid_mean_scores)
# where do you think we should put the tree cut-off at?

# let's pull out the best estimator and print its ROC AUC
best_decision_tree_est = grid.best_estimator_
# how many trees did the best estiator have? 
print best_decision_tree_est.n_estimators
# how accurate was the best estimator?
#360
print grid.best_score_
## did accuracy improve? 
#92%, yes a little better than the random forest

#################
## BOOSTING TREES
#################
boosting_tree = ensemble.GradientBoostingClassifier()

roc_scores_gbm = cross_val_score(boosting_tree, explanatory_df, response_series, cv=10, scoring='roc_auc')

#let's compare our accuracies
print roc_scores_gbm.mean()
#0.895137491616
print roc_scores_rf.mean()
#0.914078722334
print roc_score_tree.mean()
#0.760437290409

# let's tune for num_trees, learning rate, and subsampling percent.
# need to import arange to create ranges for floats
from numpy import arange

learning_rate_range = arange(0.01, 0.4, 0.02)
subsampling_range = arange(0.25, 1, 0.25)
n_estimators_range = range(25, 100, 25)

param_grid = dict(learning_rate = learning_rate_range, n_estimators = n_estimators_range, subsample = subsampling_range)

gbm_grid = GridSearchCV(boosting_tree, param_grid, cv=10, scoring='roc_auc')
gbm_grid.fit(explanatory_df, response_series)

# find the winning parameters
print gbm_grid.best_params_
# how does this compare to the default settings
# estimators = 75, subsample = .5, learning_rate = 0.07

# pull out the best score
print gbm_grid.best_score_
#0.905180055367
print grid.best_score_
#0.916196121879
## so, GBMs get close to RFs, but underpreform here.

## ROC curve accuracy of the GBM vs RF vs Tree Method
from sklearn.cross_validation import train_test_split
from sklearn import metrics

xTrain, xTest, yTrain, yTest = train_test_split(
                    explanatory_df, response_series, test_size =  0.3)

tree_probabilities = pandas.DataFrame(tree.DecisionTreeClassifier().fit(xTrain, yTrain).predict_proba(xTest))
rf_probabilities = pandas.DataFrame(best_decision_tree_est.fit(xTrain, yTrain).predict_proba(xTest))
gbm_probabilities = pandas.DataFrame(gbm_grid.best_estimator_.fit(xTrain, yTrain).predict_proba(xTest))


tree_fpr, tree_tpr, thresholds = metrics.roc_curve(yTest, tree_probabilities[1])
rf_fpr, rf_tpr, thresholds = metrics.roc_curve(yTest, rf_probabilities[1])
gbm_fpr, gbm_tpr, thresholds = metrics.roc_curve(yTest, gbm_probabilities[1])


plt.figure()
plt.plot(tree_fpr, tree_fpr, color = 'g')
plt.plot(rf_fpr, rf_tpr, color = 'b')
plt.plot(gbm_fpr, gbm_tpr, color = 'r')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
## what does this tell us for this sample?
#this tells us that random forest is the best model

## create partial dependence plot on most important features for gbm.

importances = pandas.DataFrame(gbm_grid.best_estimator_.feature_importances_, index = explanatory_df.columns, columns =['importance'])

importances.sort(columns = ['importance'], ascending = False, inplace = True)
print importances

from sklearn.ensemble.partial_dependence import plot_partial_dependence

features = [i for i, j in enumerate(explanatory_df.columns.tolist()) if j in importances.importance[0:3].index.tolist()]

fig, axs = plot_partial_dependence(gbm_grid.best_estimator_, explanatory_df, features, feature_names = explanatory_df.columns)

#                  importance
#totalruns           0.156319
#shutouts            0.085167
#errors              0.077250
#teamID_Nothing      0.071030
#totalRBI            0.064376
#earnedruns          0.061927
#stolenbases         0.049815
#atbats              0.045446
#totalhomeruns       0.044935
#totalgames          0.042369
#timewithouts        0.036490
#doubleplays         0.036110
#totalhits           0.036078
#strikeouts          0.032670
#putouts             0.027523
#intentionalwalks    0.024861
#teamID_CLE          0.018374
#lgID_Other          0.016115
#wildpitches         0.014075
#teamID_MIN          0.009700
#totalsaves          0.009699
#teamID_SLA          0.007456
#teamID_CHA          0.005530
#teamID_CHN          0.005182
#teamID_NYA          0.004148
#teamID_Other        0.003070
#teamID_PIT          0.002294
#teamID_SLN          0.002274
#teamID_BOS          0.002180
#lgID_NL             0.001989
#teamID_WS1          0.001543
#teamID_BRO          0.001110
#teamID_DET          0.001065
#teamID_NYN          0.000957
#teamID_CIN          0.000426
#teamID_SFN          0.000331
#teamID_BSN          0.000120
#teamID_PHA          0.000000
#a.playerID_Other    0.000000
#lgID_AL             0.000000
#teamID_BAL          0.000000
#teamID_PHI          0.000000
#teamID_HOU          0.000000
#teamID_KCA          0.000000
#teamID_NY1          0.000000
#teamID_MON          0.000000
#teamID_ML4          0.000000
#teamID_LAN          0.000000
#teamID_CAL          0.000000


#################
# NEURAL NETWORKS
################
#good for building a model for unclear relataionships such as text analysis or handwriting

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model

logistic_classifier = linear_model.LogisticRegression()
neural_net = BernoulliRBM(random_state=0, verbose=True) 

# create the pipeline of a neural net connected to a logistic regression.
neural_classifier = Pipeline(steps=[('neural_net', neural_net), ('logistic_classifier', logistic_classifier)])


## you can cross-validate the entire pipeline like any old classifier.
roc_scores_nn = cross_val_score(neural_classifier, explanatory_df, response_series, cv=10, scoring='roc_auc')


#let's compare our accuracies
print roc_scores_nn.mean()
#0.59340417505
print roc_scores_gbm.mean()
#0.895137491616
print roc_scores_rf.mean()
#0.914078722334
print roc_score_tree.mean()
#0.760437290409
# not so great, eh?

# let's do some grid search.
# i constrained this more than I should for the sake of time.
# i also commented out iteraton to speed things up -- 
# feel free to uncomment in your spare time.
learning_rate_range = arange(0.01, 0.2, 0.05)
#iteration_range = range(30, 50, 5)
components_range = range(250, 500, 50)

# notice that I have the name of the item in the pipeline 
# followed by two underscores when I build the pipeline.
param_grid = dict(neural_net__n_components = components_range, neural_net__learning_rate = learning_rate_range)
#, neural_net__n_iter = iteration_range

# doing 5-fold CV here for reasons of time; feel free to do 10-fold 
# in your own leisure.
nn_grid = GridSearchCV(neural_classifier, param_grid, cv=5, scoring='roc_auc')
nn_grid.fit(explanatory_df, response_series)

## pull out best score
print nn_grid.best_score_
#0.65855746655
## compare to other grid best scores
print gbm_grid.best_score_
#0.905180055367
print grid.best_score_
#0.916196121879

print nn_grid.best_params_
#neural_net__n_components': 400, 'neural_net__learning_rate': 0.01
# so the grid search best score is tons better than the 
# original, and slightly better than rf and gbm!

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

#Add in scores for logistic regression and naive bayes Logistic Regression
# create dataframes with an intercept column and dummy variables for occupation and occupation_husb
#y, X = dmatrices('inducted ~ outs_pitched',df, return_type="dataframe")
#print X.columns

# flatten y into a 1-D array
y = np.ravel(response_series)
X=explanatory_df
# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X, y)

# check the accuracy on the training set
print model.score(X, y)
#Accuracy is 87%

# what percentage were inducted
print y.mean()
#It appears that 26% of players were inducted

# evaluate the model using 10-fold cross-validation
acc_scores_log = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print acc_scores_log
#[ 0.82291667  0.85416667  0.86458333  0.875       0.75531915  0.80851064
 # 0.88297872  0.82978723  0.88297872  0.87234043]

roc_scores_log = cross_val_score(LogisticRegression(), X, y, scoring='roc_auc', cv=10)
print roc_scores_log

#[ 0.8428169   0.9228169   0.8828169   0.90309859  0.8422619   0.78988095
#  0.87380952  0.78333333  0.93095238  0.85654762]

print roc_scores_log.mean()
#With cross validation we see that the model is performing at 86% accuracy

#Also do Naive Bayes Classifier
# create a naive Bayes classifier and get it cross-validated accuracy score. 
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score

# creating the naive bayes classifier object 
naive_bayes_classifier = MultinomialNB()

#Also do Naive Bayes Classifier
#Note that MultinomialNB had a scores must be non-negative error
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.cross_validation import cross_val_score

# creating the naive bayes classifier object 
naive_bayes_classifier = GaussianNB()

# running a cross-validates score on accuracy
acc_scores_naive = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='accuracy')

# let's see how accurate the model is, on average.
print acc_scores_naive.mean()
#0.420744680851

## calculating the ROC area under the curve score. 
roc_scores_naive = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='roc_auc')
print roc_scores_naive.mean()
#0.719704309188

#Compare Scores using ROC_AUC
#RF
print roc_scores_rf.mean()
#0.914078722334

#GBM
print roc_scores_gbm.mean()
#0.895137491616

#NN
print roc_scores_nn.mean()
#0.59340417505

#Decision Tree
print roc_score_tree.mean()
#0.760437290409

#Logistic Regression
print roc_scores_log.mean()
#0.862833501006

#Naive Bayes
roc_scores_naive.mean()
#0.71970430918846418

#Comparing the models, it looks like random forest is the most accurate (measured by ROC AUC)

# let's do some grid search.
# i constrained this more than I should for the sake of time.
# i also commented out iteraton to speed things up -- 
# feel free to uncomment in your spare time.
learning_rate_range = arange(0.01, 0.2, 0.05)
iteration_range = range(30, 50, 5)
components_range = range(250, 500, 50)

# notice that I have the name of the item in the pipeline 
# followed by two underscores when I build the pipeline.
param_grid = dict(neural_net__n_components = components_range, neural_net__learning_rate = learning_rate_range, neural_net__n_iter = iteration_range)

# doing 5-fold CV here for reasons of time; feel free to do 10-fold in your own leisure.
nn_grid = GridSearchCV(neural_classifier, param_grid, cv=10, scoring='roc_auc')
nn_grid.fit(explanatory_df, response_series)

## pull out best score
print nn_grid.best_score_
#0.615856050508
## compare to other grid best scores
print gbm_grid.best_score_
#0.905180055367
print grid.best_score_
#0.916196121879

print nn_grid.best_params_
#'neural_net__n_components': 450, 'neural_net__learning_rate': 0.11, 'neural_net__n_iter': 30
#grid search best score is better











