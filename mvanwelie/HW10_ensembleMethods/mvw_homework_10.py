# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 21:32:03 2015

@author: megan
"""

import sqlite3
import pandas
import numpy
import matplotlib.pyplot as plt 

from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn import ensemble
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import  GridSearchCV
from sklearn import tree

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model

from sklearn.naive_bayes import MultinomialNB

################
# Homework 10 - Ensemble methods and neural networks
# Assignment:
# Run a Random Forest (RF), Boosting Trees (GBM), and Neural Network (NN) classifier on the data you assembled in the homework from class 9.
# See which of the methods you've used so far (RF, GBM, NN, Decision Tree, Logistic Regression, Naive Bayes) is the most accurate (measured by ROC AUC).
# Use grid seach to optimize your NN's tuning parameters for learning_rate, iteration_range, and compoents, as well as any others you'd like to test.
################

################
# Define helper methods for cleaning and manipulating
# the data into a useful set
################

def cleanup_data(df, cutoffPercent = .01):
    for col in df:
        sizes = df[col].value_counts(normalize = True)
        # get the names of the levels that make up less than 1% of the dataset
        values_to_delete = sizes[sizes<cutoffPercent].index
        df[col].ix[df[col].isin(values_to_delete)] = "Other"
    return df

def get_binary_values(data_frame):
    """encodes cateogrical features in Pandas.
    """
    all_columns = pandas.DataFrame( index = data_frame.index)
    for col in data_frame.columns:
        data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pandas.concat([all_columns, data], axis=1)
    return all_columns
    
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
    
# create new class with a .coef_ attribute.
class TreeClassifierWithCoef(tree.DecisionTreeClassifier):
    def fit(self, *args, **kwargs):
        super(tree.DecisionTreeClassifier, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

################
# Query databases for training data before the year 2000
################

# Create a table with a categorical feature that shows the dominant team played per player
con = sqlite3.connect('/Users/megan/Documents/SQLite/lahman2013.sqlite')
query = 'select playerID, teamID from Batting'
df = pandas.read_sql(query, con)
con.close()
majority_team_by_player = df.groupby(['playerID']).agg(lambda x:x.value_counts().idxmax())
conn = sqlite3.connect('/Users/megan/Documents/SQLite/lahman2013.sqlite')
majority_team_by_player.to_sql('dominant_team_per_player', conn, if_exists = 'replace')

# Create a table to indicate if a player was ever inducted into the hall of fame
cur = conn.cursor()    
table_creation_query = """
CREATE TABLE hall_of_fame_inductees_all as  
select playerID, inductedYear, category, case when average_inducted = 0 then 0 else 1 end as inducted from (
select playerID, max(yearid) as inductedYear, category, avg(case when inducted = 'Y' then 1 else 0 end ) as average_inducted from  HallOfFame hf
group by playerID
) bb;"""
cur.execute(table_creation_query)
cur.close()
conn.close()

# Query the DB for all predicive columns for inducted into the hall of fame
# Add birth state and dominant team player as the extra categorical features
conn = sqlite3.connect('/Users/megan/Documents/SQLite/lahman2013.sqlite')
sql = '''
SELECT hofi.playerID, hofi.inductedYear, hofi.inducted, dtpp.teamID,
batting.atBats, batting.hits,
pitching.wins, pitching.losses,
fielding.putOuts, fielding.assists, fielding.errors,
master.birthState
FROM hall_of_fame_inductees_all hofi
LEFT JOIN
(
SELECT dtpp.playerID, dtpp.teamID
from dominant_team_per_player dtpp
)
dtpp on dtpp.playerID = hofi.playerID
LEFT JOIN 
(
SELECT b.playerID, sum(b.AB) as atBats, sum(b.H) as hits
FROM Batting b
GROUP BY b.playerID
)
batting on batting.playerID = hofi.playerID
LEFT JOIN 
(
SELECT p.playerID, sum(p.W) as wins, sum(p.L) as losses
FROM Pitching p 
GROUP BY p.playerID
)
pitching on hofi.playerID = pitching.playerID
LEFT JOIN 
(
SELECT f.playerID, sum(f.PO) as putOuts, sum(f.A) as assists, sum(f.E) as errors
FROM Fielding f 
GROUP BY f.playerID
)
fielding on hofi.playerID = fielding.playerID
LEFT JOIN
(
SELECT m.playerID, m.birthState
FROM Master m
)
master on master.playerID = hofi.playerID
WHERE hofi.inductedYear < 2000 and hofi.category = 'Player';
'''
df = pandas.read_sql(sql, conn)
conn.close()

############
# Clean and prepare the data
############

# Add composite feature columns
df['batting_average'] = df.hits / df.atBats
df['winning_percentage'] = df.wins / (df.wins + df.losses)
df['fielding_percentage'] = (df.putOuts + df.assists) / (df.putOuts + df.assists + df.errors)

# dropping duplicate playerID columns
df.drop('playerID',  1, inplace = True)

## splitting out the explanatory features 
explanatory_features = [col for col in df.columns if col not in ['inducted']]
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

# Split the numeric explanatory data from the string data
# This is done to clean and manipulate the different types of data correctly
string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']

# Fill NaNs in our string features
string_features = string_features.fillna('Nothing')

# Bin and encode the categorical string features
string_features = cleanup_data(string_features)
string_features.teamID.value_counts(normalize = True)
encoded_data = get_binary_values(string_features)
encoded_data.head()

# Fill NaNs in our numeric features
from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

# Merge the numeric and encoded categorical features back together
explanatory_df = pandas.concat([numeric_features, encoded_data],axis = 1)
explanatory_df.head()

# Find and remove features with no variance 
toKeepDelete = find_zero_var(explanatory_df)
print toKeepDelete # nothing to delete

# Find and remove columns with perfect correlation
corrMatrixToRemove = find_perfect_corr(explanatory_df)
print corrMatrixToRemove['toRemove'] # nothing to remove
# explanatory_df.drop(corrMatrixToRemove['toRemove'], 1, inplace = True)

# Scale data
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)

############
# Naive Bayes
############
naive_bayes_classifier = MultinomialNB()

# Compute ROC AUC of naive bayes
roc_scores_nb = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='accuracy')

# Average is 59%
print roc_scores_nb.mean()

############
# Random Forest
############
random_forest = ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, min_density=None, compute_importances=None)

# Compute ROC AUC of the random forest
roc_scores_rf = cross_val_score(random_forest, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

# Accuracy is 90%
print roc_scores_rf.mean()

# Perform grid search to find the number of optimal trees
trees_range = range(10, 550, 10)
param_grid = dict(n_estimators = trees_range)
grid = GridSearchCV(random_forest, param_grid, cv=10, scoring='roc_auc', n_jobs = -1)
grid.fit(explanatory_df, response_series)

# Plot the results of the grid search 
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(trees_range, grid_mean_scores)

# Pull out the best estimator
best_random_forest = grid.best_estimator_

# 60 trees is optimal
print best_random_forest.n_estimators

# Accuracy increased to 94%
print grid.best_score_

############
# Decision Tree
############
decision_tree = tree.DecisionTreeClassifier()

# Compute ROC AUC of the decision tree
roc_score_tree = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

# Accuracy is 81%
print roc_score_tree.mean()

############
# Boosting Tree
############
boosting_tree = ensemble.GradientBoostingClassifier()

# Compute ROC AUC of the boosting tree
roc_scores_gbm = cross_val_score(boosting_tree, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

# Accuracy is 93%
print roc_scores_gbm.mean()

############
# Neural Networks
############

logistic_classifier = linear_model.LogisticRegression()
neural_net = BernoulliRBM(random_state=0, verbose=True) 
# create the pipeline of a neural net connected to a logistic regression.
neural_classifier = Pipeline(steps=[('neural_net', neural_net), ('logistic_classifier', logistic_classifier)])

# Compute ROC AUC of the neural network
roc_scores_nn = cross_val_score(neural_classifier, explanatory_df, response_series, cv=10, scoring='roc_auc')

# Accuracy is 50%
print roc_scores_nn.mean()

############
# Compare baseline results
############

print roc_scores_nb.mean()      # 58.6%
print roc_scores_rf.mean()      # 90.4%
print roc_score_tree.mean()     # 80.9%
print roc_scores_gbm.mean()     # 93.1%
print roc_scores_nn.mean()      # 50%

# Based on the results of the baseline classifiers with no tuning
# Boosting Tree is the current winner with 93% accuracy

############
# Optimze the Neural Network
############

# Create value ranges for grid search
learning_rate_range = arange(0.01, 0.2, 0.05)
iteration_range = range(30, 50, 5)
components_range = range(250, 500, 50)

param_grid = dict(neural_net__n_components = components_range, neural_net__learning_rate = learning_rate_range, 
neural_net__n_iter = iteration_range
)

# Run the grid search
nn_grid = GridSearchCV(neural_classifier, param_grid, cv=10, scoring='roc_auc')
nn_grid.fit(explanatory_df, response_series)

# Pull out best score
print nn_grid.best_score_
# Even after tuning, the best neural net ROC AUC score remains at 50%

