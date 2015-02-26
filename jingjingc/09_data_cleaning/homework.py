# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 23:01:43 2015

@author: jchen
"""
# import packages
import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

# create all functions used to simplify script

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
    all_columns = pd.DataFrame( index = data_frame.index)
    for col in data_frame.columns:
        data = pd.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pd.concat([all_columns, data], axis=1)
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
    corrMatrix.loc[:,:] = np.tril(corrMatrix.values, k = -1)
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


# set option to display all columns
pd.set_option('display.max_columns', None)

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

# Check where explanatory features got removed. Looks like none.
response_series.index[~response_series.index.isin(explanatory_df.index)] 

# Separate numeric from string features
string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']

# Fill NaNs in string features
string_features = string_features.fillna('Nothing')

# Bin categorical features with threshold of 1%
string_features = cleanup_data(string_features, cutoffPercent = .01)
string_features.teamID.value_counts(normalize = True) # check data 

# Encode categorical features
encoded_data = get_binary_values(string_features)
encoded_data.head()

# Impute vales for numeric features
from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features) # store imputer in object
numeric_features = pd.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

# recombine numeric and categorical features into one df
explanatory_df = pd.concat([numeric_features, encoded_data],axis = 1)
explanatory_df.head()

# Find features with no variance
delete = find_zero_var(explanatory_df)['toDelete']
print delete
# nothing to drop

# Find features with perfect correlation
remove = find_perfect_corr(explanatory_df)['toRemove']
print remove
# no features with perfect correlation

#############
# Scale data

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pd.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)

###############################
# Recursive feature elimination
###############################

from sklearn.feature_selection import RFECV
from sklearn import tree

# create new class with a .coef_ attribute.
class TreeClassifierWithCoef(tree.DecisionTreeClassifier):
    def fit(self, *args, **kwargs):
        super(tree.DecisionTreeClassifier, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

# create your tree based estimator
decision_tree = TreeClassifierWithCoef(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)

## set up the estimator. Score by AUC
rfe_cv = RFECV(estimator=decision_tree, step=1, cv=10,
              scoring='roc_auc', verbose = 1)
rfe_cv.fit(explanatory_df, response_series)

print "Optimal number of features :{0} of {1} considered".format(rfe_cv.n_features_,len(explanatory_df.columns))

print rfe_cv.grid_scores_

## let's plot out the results
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (ROC_AUC)")
plt.plot(range(1, len(rfe_cv.grid_scores_) + 1), rfe_cv.grid_scores_)
plt.show()
# Looks like 5 features is good enough

# Print features used and extract final model
features_used = explanatory_df.columns[rfe_cv.get_support()]
print features_used

final_estimator_used = rfe_cv.estimator_

#############
# Grid search
#############

from sklearn.grid_search import  GridSearchCV

depth_range = range(1, 10)
param_grid = dict(estimator__max_depth=depth_range)
rfe_grid_search = GridSearchCV(rfe_cv, param_grid, cv=10, scoring='roc_auc')
rfe_grid_search.fit(explanatory_df, response_series)

print rfe_grid_search.grid_scores_
print rfe_grid_search.best_params_

grid_mean_scores = [score[1] for score in rfe_grid_search.grid_scores_]

plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.plot(rfe_grid_search.best_params_['estimator__max_depth'], rfe_grid_search.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')
plt.grid(True)

# pull out best estimator
best_decision_tree_rfe_grid = rfe_grid_search.best_estimator_

features_used_rfecv_grid = explanatory_df.columns[best_decision_tree_rfe_grid.get_support()]

print features_used_rfecv_grid

#########################
# Test on data post-2000
#########################

# Pull in data from 2000 onward
conn = sqlite3.connect('/Users/jchen/Documents/SQLite/lahman2013.sqlite')
# create hall of fame inductees table for post-2000
cur = conn.cursor()    
table_creation_query = """
CREATE TABLE hall_of_fame_inductees_post_2000 as  
select playerID, case when average_inducted = 0 then 0 else 1 end as inducted from (
select playerID, avg(case when inducted = 'Y' then 1 else 0 end ) as average_inducted from  HallOfFame hf
where yearID >= 2000
group by playerID
) bb;"""
cur.execute(table_creation_query)
cur.close()

# same SQL query as above
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
from hall_of_fame_inductees_post_2000 h
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
df_post_2000 = pd.read_sql(sql, conn)
# close out connection
conn.close()

## Perform the same preprocessing ##
df_post_2000.drop('playerID',  1, inplace = True)

# Split out explanatory features
explanatory_features_post_2000 = [col for col in df_post_2000.columns if col not in ['player_name', 'inducted']]
explanatory_df_post_2000 = df_post_2000[explanatory_features_post_2000]

explanatory_df_post_2000.dropna(how='all', inplace = True) 

explanatory_colnames_post_2000 = explanatory_df_post_2000.columns

# Response series
response_series_post_2000 = df_post_2000.inducted
response_series_post_2000.dropna(how='all', inplace = True) 

# Check where explanatory features got removed. Looks like none.
response_series_post_2000.index[~response_series_post_2000.index.isin(explanatory_df_post_2000.index)] 

# Separate numeric from string features
string_features_post_2000 = explanatory_df_post_2000.ix[:, explanatory_df_post_2000.dtypes == 'object']
numeric_features_post_2000 = explanatory_df_post_2000.ix[:, explanatory_df_post_2000.dtypes != 'object']

# Fill NaNs in string features
string_features_post_2000 = string_features_post_2000.fillna('Nothing')


# Get binned categories from training data
team_bins = np.unique(string_features.teamID.values)
league_bins = np.unique(string_features.league.values)
division_bins = np.unique(string_features.division.values)

# compare these string values to training data
print np.unique(string_features_post_2000.teamID.values)
print team_bins

print np.unique(string_features_post_2000.league.values)
print league_bins

print np.unique(string_features_post_2000.division.values)
print division_bins

# Replace values not found in training bins with Other
string_features_post_2000.teamID[~string_features_post_2000.teamID.isin(team_bins)]= 'Other'
string_features_post_2000.league[~string_features_post_2000.league.isin(league_bins)]= 'Other'
string_features_post_2000.division[~string_features_post_2000.division.isin(division_bins)]= 'Other'

# Encode categorical features
encoded_data_post_2000 = get_binary_values(string_features_post_2000)
encoded_data_post_2000.head()

# Impute vales for numeric features
from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features_post_2000) # store imputer in object
numeric_features_post_2000 = pd.DataFrame(imputer_object.transform(numeric_features_post_2000), columns = numeric_features_post_2000.columns)

# recombine numeric and categorical features into one df
explanatory_df_post_2000 = pd.concat([numeric_features_post_2000, encoded_data_post_2000],axis = 1)
explanatory_df_post_2000.head()

# Find features with no variance
delete = find_zero_var(explanatory_df_post_2000)['toDelete']
print delete
# nothing to drop

# Find features with perfect correlation
remove = find_perfect_corr(explanatory_df_post_2000)['toRemove']
print remove
# American League indicator needs removing
# This makes sense as we already have indicators for all other leagues

explanatory_df_post_2000.drop('league_AL', inplace=True, axis=1)

## Scale data ##
# First keep only columns used in training
explanatory_df_post_2000 = explanatory_df_post_2000[explanatory_df.columns]
# seems like we are missing columns in OOS data that were in training set
# add them in
add_columns = [col for col in explanatory_df.columns if col not in explanatory_df_post_2000.columns]
add_df = pd.DataFrame(0, index = explanatory_df_post_2000.index, columns = add_columns)

explanatory_df_post_2000_new = pd.concat([explanatory_df_post_2000, add_df], axis = 1)

# rearrange columns
explanatory_df_post_2000 = explanatory_df_post_2000_new[explanatory_df.columns]

explanatory_df_post_2000 = pd.DataFrame(scaler.transform(explanatory_df_post_2000), columns = explanatory_df_post_2000.columns)

#################################
# Predict on transformed OOS data
#################################

from sklearn import metrics

predicted_values = best_decision_tree_rfe_grid.predict(explanatory_df_post_2000)

cm = pd.crosstab(response_series_post_2000, predicted_values, rownames=['True Label'], colnames=['Predicted Label'], margins=True)

print cm
# Looks like the model correctly predicts 88% of players not inducted into Hall of Fame
# But only predicts 43% of players who actually get inducted into the Hall of Fame
# So very poor prediction for true positives - worse than luck :(

# Extract decision tree probabilities
predicted_probs = pd.DataFrame(best_decision_tree_rfe_grid.predict_proba(explanatory_df_post_2000))

# Plot the ROC curve
fpr, tpr, thresholds_cart = metrics.roc_curve(response_series_post_2000, predicted_probs[1])
plt.figure()
plt.plot(fpr, tpr, color = 'b')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
# Really not a stellar ROC curve
# Sensitivity is pretty poor 


