# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 14:46:50 2015

@author: jkraunz
"""

import pandas
import sqlite3
import numpy

pandas.set_option('display.max_columns', None)

con = sqlite3.connect('/Users/jkraunz/Documents/SQLite/lahman2013.sqlite.crdownload')

query = 'select playerID, yearID from AllstarFull'
df = pandas.read_sql(query, con)
con.close()

# for some reason there were multiple games played in the same year.  Need to drop those duplicates
df.drop_duplicates()

conn = sqlite3.connect('/Users/jkraunz/Documents/SQLite/lahman2013.sqlite.crdownload')
df.to_sql('num_of_allstar', conn, if_exists = 'replace')

conn.close()

conn = sqlite3.connect('/Users/jkraunz/Documents/SQLite/lahman2013.sqlite.crdownload')

sql= '''
Select i.*, count(yearID) as num_of_allstar_games
FROM
(Select f.*, birthCountry
FROM
(Select d.*, e.teamID
FROM
(Select c.*, sum(H) as total_post_hits, sum(HR) as total_post_HRs, sum(RBI) as total_post_RBIs
FROM
(Select a.*, sum(E) as total_errors
FROM
(SELECT m.*,
sum(SO) as total_SOs, avg(ERA) as avg_ERA, sum(W) as total_wins, sum(SV) as total_saves, count(YearID) as years_pitched
FROM
(select h.*, sum(RBI) as total_RBIs, sum(SB) as total_stolen_bases, sum(R) as total_runs, sum(H) as total_hits, count(yearID) as years_batted, sum(HR) as total_HRs, sum('2B') as total_2B, sum('3B') as total_3B
FROM 
(SELECT playerID, max(yearID) as final_year_voted, count(yearID) as years_voted, inducted
FROM HallofFame 
Where yearID < 2000
GROUP BY playerID) h
LEFT JOIN Batting b on h.playerID = b.playerID
GROUP BY h.playerID) m
LEFT JOIN Pitching p on m.playerID = p.playerID
group by m.playerID) a
LEFT JOIN Fielding f on a.playerID = f.playerID
GROUP BY a.playerID) c
Left Join BattingPost bp on c.playerID = bp.playerID
Group By c.playerID) d
Left Join dominant_team_per_player e on d.playerID = e.playerID
Group by d.playerID) f
Left Join Master g on f.playerID = g.playerID
Group by f.playerID) i
Left Join num_of_allstar j on i.playerID = j.playerID
Group by i.playerID
'''

df = pandas.read_sql(sql, conn)
conn.close()

df.head()

df.columns

# Functions
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

# Data manipulation

df['inducted1'] = 0
df.inducted1[df.inducted == 'Y'] = 1

df['years_played'] = 0
df.years_played[df.years_pitched >= df.years_batted] = df.years_pitched
df.years_played[df.years_pitched < df.years_batted] = df.years_batted

df.drop(['playerID', 'inducted', 'years_pitched', 'years_batted', 'final_year_voted'],  1, inplace = True)

# dropped final_year_voted since it's not a good variable to use

new_columns = df.columns.values
new_columns[19] = 'inducted'
df.columns = new_columns

df.head()

# Set up explanatory and response features

# using features identified in the decision tree recursive feature elimination in HW 9.  Would orginarily run new rfe for each method but want to be able to compare models.

explanatory_df = df[['years_voted', 'total_RBIs', 'total_runs', 'total_hits', 'total_3B', 'total_wins', 'total_errors', 'total_post_hits', 'birthCountry']]

explanatory_df.dropna(how = 'all', inplace = True)

explanatory_col_names = explanatory_df.columns

response_series = df.inducted

response_series.dropna(how = 'all', inplace = True)

response_series.index[~response_series.index.isin(explanatory_df.index)]

explanatory_df.describe()

# 1. Split data into categorical and numeric data

string_features = explanatory_df.ix[: , explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[: , explanatory_df.dtypes != 'object']

string_features.head()
numeric_features.head()

# 2. Fill numeric NaNs through imputation

from sklearn.preprocessing import Imputer

imputer_object = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

numeric_features.head()

numeric_features.describe()

# 3. Fill categorical NaNs with ‘Nothing’

string_features = string_features.fillna('Nothing')

# 4. Detect low-frequency levels in categorical features and bin them under ‘other’

string_features['birthCountry_USA'] = 0
string_features.birthCountry_USA[df.birthCountry == 'USA'] = 1

string_features.drop(['birthCountry'],  1, inplace = True)

# 5. Encode each categorical variable into a sequence of binary variables.

# Not needed since I only have one and I did it in the previous step


# 6. Merge your encoded categorical data with your numeric data

explanatory_df = pandas.concat([numeric_features, string_features], axis = 1)
explanatory_df.head()

explanatory_df.describe()

# 7. Remove features with no variation
   
# Did in HW 9

# No features had zero variance

# 8. Remove perfectly correlated features


# Function to look at all correlation
   
# Did in HW ()

# 9. Scale your data with zero mean and unit variance

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)

explanatory_df.describe()

# Random Forest
from sklearn import ensemble
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

rf = ensemble.RandomForestClassifier(n_estimators = 500)

roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring = 'roc_auc', n_jobs = -1)

roc_scores_tree = cross_val_score(tree.DecisionTreeClassifier(), explanatory_df, response_series, cv=10, scoring = 'roc_auc', n_jobs = -1)

print roc_scores_rf.mean()
print roc_scores_tree.mean()

# Random forest ROC score .922 and decision tree is .785

# Use grid search to optimize Random Forest
from sklearn.grid_search import GridSearchCV

trees_range = range(10, 550, 10)
param_grid = dict(n_estimators = trees_range)

grid = GridSearchCV(rf, param_grid, cv=10, scoring = 'roc_auc', n_jobs = -1)
grid.fit(explanatory_df, response_series)

grid_mean_scores = [result[1] for result in grid.grid_scores_]

# increases size of graph
from pylab import rcParams
import matplotlib.pyplot as plt

rcParams['figure.figsize'] = 10, 5

plt.figure()
plt.plot(trees_range, grid_mean_scores)

best_decision_tree_est = grid.best_estimator_

print best_decision_tree_est.n_estimators
# 300

print grid.best_score_
# 0.926543560528


#If I rerurn with n_estimators = 210 then my roc scores go down slightly

rf = ensemble.RandomForestClassifier(n_estimators = 210)
 
roc_scores_rf_opt = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring = 'roc_auc', n_jobs = -1)

print roc_scores_rf_opt.mean()
print roc_scores_rf.mean()
print roc_scores_tree.mean()

#0.919830397913
#0.921950424005
#0.785359959675


# Boosting tree

boosting_tree = ensemble.GradientBoostingClassifier()

roc_scores_gbm = cross_val_score(boosting_tree, explanatory_df, response_series, cv=10, scoring = 'roc_auc', n_jobs = -1)

print roc_scores_gbm.mean()
print roc_scores_rf_opt.mean()
print roc_scores_rf.mean()
print roc_scores_tree.mean()

#0.921613888395
#0.919830397913
#0.921950424005
#0.785359959675


from numpy import arange

# learning rate range is most important to tune for
learning_rate_range = arange(0.01, 0.4, 0.02)
subsampling_range = arange(0.25, 1, 0.25)
n_estimators_range = arange(200, 300, 25)

param_grid = dict(learning_rate = learning_rate_range, n_estimators = n_estimators_range, subsample = subsampling_range)

gbm_grid = GridSearchCV(boosting_tree, param_grid, cv=10, scoring = 'roc_auc', n_jobs = -1)

gbm_grid.fit(explanatory_df, response_series)

print gbm_grid.best_params_
 #{'n_estimators': 225, 'subsample': 0.75, 'learning_rate': 0.029999999999999999}


print gbm_grid.best_score_
# 0.926270745804


boosting_tree = ensemble.GradientBoostingClassifier(learning_rate = 0.029999999999999999, subsample = 0.75, n_estimators = 225)

roc_scores_gbm_opt = cross_val_score(boosting_tree, explanatory_df, response_series, cv=10, scoring = 'roc_auc', n_jobs = -1)

print roc_scores_gbm_opt.mean()
print roc_scores_gbm.mean()
print roc_scores_rf_opt.mean()
print roc_scores_rf.mean()
print roc_scores_tree.mean()

#0.923815453952
#0.921613888395
#0.919830397913
#0.921950424005
#0.785359959675


from sklearn.cross_validation import train_test_split
from sklearn import metrics

xTrain, xTest, yTrain, yTest = train_test_split(explanatory_df, response_series, test_size = 0.3)

# Create pandas dataframe
tree_probabilities = pandas.DataFrame(tree.DecisionTreeClassifier().fit(xTrain, yTrain).predict_proba(xTest))
rf_probabilities = pandas.DataFrame(best_decision_tree_est.fit(xTrain, yTrain).predict_proba(xTest))
gbm_probabilities = pandas.DataFrame(gbm_grid.best_estimator_.fit(xTrain, yTrain).predict_proba(xTest))

# plot with Y and 2nd column of pandas dataframe
tree_fpr, tree_tpr, thresholds = metrics.roc_curve(yTest, tree_probabilities[1])
rf_fpr, rf_tpr, thesholds = metrics.roc_curve(yTest, rf_probabilities[1])
gbm_fpr, gbm_tpr, thesholds = metrics.roc_curve(yTest, gbm_probabilities[1])

plt.figure()
plt.plot(tree_fpr, tree_tpr, color = 'g')
plt.plot(rf_fpr, rf_tpr, color = 'b')
plt.plot(gbm_fpr, gbm_tpr, color = 'r')
plt.xlabel('False Positive Rate(1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

importances = pandas.DataFrame(gbm_grid.best_estimator_.feature_importances_, index = explanatory_df.columns, columns =['importance'])

importances.sort(columns = ['importance'], ascending = False, inplace = True)
print importances

#                  importance
#total_3B            0.202059
#total_wins          0.178061
#years_voted         0.151448
#total_runs          0.133153
#total_RBIs          0.119498
#total_hits          0.090283
#total_errors        0.075008
#total_post_hits     0.047740
#birthCountry_USA    0.002750



# Boosting tree and Random Forest both look good but Random Forest looks slightly better

from sklearn.ensemble.partial_dependence import plot_partial_dependence

features = [i for i, j in enumerate(explanatory_df.columns.tolist()) if j in importances.importance[0:3].index.tolist()]

fig, axs = plot_partial_dependence(gbm_grid.best_estimator_, explanatory_df, features, feature_names = explanatory_df.columns)

# Neural Networks

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model

logistic_classifier = linear_model.LogisticRegression()
neural_net = BernoulliRBM(random_state=0, verbose=True)

neural_classifier = Pipeline(steps=[('neural_net', neural_net), ('logistic_classifier', logistic_classifier)])

roc_scores_nn = cross_val_score(neural_classifier, explanatory_df, response_series, cv=10, scoring = 'roc_auc')

print roc_scores_nn.mean()
print roc_scores_gbm_opt.mean()
print roc_scores_gbm.mean()
print roc_scores_rf_opt.mean()
print roc_scores_rf.mean()
print roc_scores_tree.mean()

#0.559621656882
#0.923815453952
#0.921613888395
#0.919830397913
#0.921950424005
#0.785359959675


learning_rate_range = arange(0.01, 0.2, 0.05)
components_range = range(250, 500, 50)

param_grid = dict(neural_net__n_components = components_range, neural_net__learning_rate = learning_rate_range)
#, neural_net__n_iter = iteration_range)

# doing 5-fold CV here for reasons of time; feel free to do 10-fold 
# in your own leisure.
nn_grid = GridSearchCV(neural_classifier, param_grid, cv=5, scoring='roc_auc')
nn_grid.fit(explanatory_df, response_series)

print nn_grid.best_score_
# 0.746507109599

print nn_grid.best_params_
# {'neural_net__n_components': 450, 'neural_net__learning_rate': 0.01}


