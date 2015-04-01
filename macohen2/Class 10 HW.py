# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 23:47:22 2015

@author: MatthewCohen
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

def get_binary_values(data_frame):
    all_columns = pandas.DataFrame( index = data_frame.index)
    for col in data_frame.columns:
        data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pandas.concat([all_columns, data], axis=1)
    return all_columns

def find_zero_var(df):
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
    
pandas.set_option('display.max_columns', None)


monster_query = """
select m.nameGiven, d.teamID, hfi.inducted, batting.*, pitching.*, fielding.* from hall_of_fame_inductees3 hfi 
left outer join master m on hfi.playerID = m.playerID
left outer join 
(
select playerID,  sum(AB) as total_at_bats, sum(H) as total_hits, sum(R) as total_runs, sum(HR) as total_home_runs, sum(SB) as total_stolen_bases,
sum(RBI) as total_RBI, sum(IBB) as total_intentional_walks
from Batting
group by playerID
HAVING max(yearID) > 1950 and min(yearID) >1950 
)
batting on batting.playerID = hfi.playerID
left outer join
(
 select playerID, sum(G) as total_games_pitched, sum(SO) as total_shutouts, sum(sv) as total_saves, sum(IPouts) as total_outs_pitched, sum(er) as total_pitching_earned_runs, sum(so) as total_strikeouts, 
avg(ERA) as average_ERA,
sum(R) as total_runs_allowed
from Pitching
group by playerID
) 
pitching on pitching.playerID = hfi.playerID 
LEFT OUTER JOIN
(
select playerID, sum(G) as total_games_fielded, sum(E) as total_errors, sum(DP) as total_double_plays
from Fielding
group by playerID
) 
fielding on fielding.playerID = hfi.playerID

LEFT OUTER JOIN dominant_team_per_player d on d.playerID = hfi.playerID
where batting.playerID is not null"""

    
conn = sqlite3.connect('/Users/MatthewCohen/Documents/SQLite/lahman2013.sqlite')
df = pandas.read_sql(monster_query, conn)
conn.close()

df.drop('playerID',  1, inplace = True)

explanatory_features = [col for col in df.columns if col not in ['nameGiven', 'inducted']]
explanatory_df = df[explanatory_features]

explanatory_df.dropna(how='all', inplace = True) 

explanatory_colnames = explanatory_df.columns

response_series = df.inducted
response_series.dropna(how='all', inplace = True) 

response_series.index[~response_series.index.isin(explanatory_df.index)]

string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']

string_features = string_features.fillna('Nothing')

string_features = cleanup_data(string_features)

encoded_data = get_binary_values(string_features)

imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

explanatory_df = pandas.concat([numeric_features, encoded_data],axis = 1)
explanatory_df.head()

no_variation = find_zero_var(explanatory_df)
explanatory_df.drop(no_variation['toDelete'], inplace = True)

no_correlation = find_perfect_corr(explanatory_df)
explanatory_df.drop(no_correlation['toRemove'], 1, inplace = True)

scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)
    
    
rf = ensemble.RandomForestClassifier(n_estimators= 500)

roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)    
roc_score_tree = cross_val_score(tree.DecisionTreeClassifier(), explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

trees_range = range(10, 550, 10)
param_grid = dict(n_estimators = trees_range)

grid = GridSearchCV(rf, param_grid, cv = 10, scoring = 'roc_auc', n_jobs = -1)
grid.fit(explanatory_df, response_series)

grid_mean_scores = [result[1] for result in grid.grid_scores_]
best_decision_tree_est = grid.best_estimator_


boosting_tree = ensemble.GradientBoostingClassifier()
roc_scores_gbm = cross_val_score(boosting_tree, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)  
    
from numpy import arange

learning_rate_range = arange(0.01, 0.4, 0.02)
subsampling_range = arange(0.25, 1, 0.25)
n_estimators_range = range(25, 100, 25)

param_grid = dict(learning_rate = learning_rate_range, n_estimators = n_estimators_range, subsample = subsampling_range)

gbm_grid = GridSearchCV(boosting_tree, param_grid, cv=10, scoring='roc_auc', n_jobs = -1)
gbm_grid.fit(explanatory_df, response_series)  
    
    
  
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model

logistic_classifier = linear_model.LogisticRegression()
neural_net = BernoulliRBM(random_state=0, verbose=True) 
neural_classifier = Pipeline(steps=[('neural_net', neural_net), ('logistic_classifier', logistic_classifier)])
roc_scores_nn = cross_val_score(neural_classifier, explanatory_df, response_series, cv=10, scoring='roc_auc')  
 



################################################################################



   
learning_rate_range = arange(0.01, 0.2, 0.05)
components_range = range(250, 500, 50)
param_grid = dict(neural_net__n_components = components_range, neural_net__learning_rate = learning_rate_range)
nn_grid = GridSearchCV(neural_classifier, param_grid, cv=5, scoring='roc_auc')
nn_grid.fit(explanatory_df, response_series)
    
    
print nn_grid.best_score_
print gbm_grid.best_score_
print grid.best_score_
print nn_grid.best_params_ 
    
    
   
   
# let's do some grid search.
# i constrained this more than I should for the sake of time.
# i also commented out iteraton to speed things up -- 
# feel free to uncomment in your spare time.
learning_rate_range = arange(0.01, 0.4, 0.05)
iteration_range = range(30, 50, 5)
components_range = range(50, 500, 50)

# notice that I have the name of the item in the pipeline 
# followed by two underscores when I build the pipeline.
param_grid = dict(neural_net__n_components = components_range, neural_net__learning_rate = learning_rate_range, neural_net__n_iter = iteration_range)

# doing 5-fold CV here for reasons of time; feel free to do 10-fold 
# in your own leisure.
nn_grid = GridSearchCV(neural_classifier, param_grid, cv=10, scoring='roc_auc')
nn_grid.fit(explanatory_df, response_series)
   
   
   
print nn_grid.best_score_
print gbm_grid.best_score_
print grid.best_score_
print nn_grid.best_params_
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    
    
    
    
    
    
    
    