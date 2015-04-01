# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 21:25:04 2015

@author: MatthewCohen
"""

import pandas
import sqlite3

pandas.set_option('display.max_columns', None)
conn = sqlite3.connect('/Users/MatthewCohen/Documents/SQLite/lahman2013.sqlite')

query = 'select playerID, teamID from Batting'
df = pandas.read_sql(query, conn)
conn.close

majority_team_by_player = df.groupby(['playerID']).agg(lambda x:x.value_counts().idxmax())


conn = sqlite3.connect('/Users/MatthewCohen/Documents/SQLite/lahman2013.sqlite')
majority_team_by_player.to_sql('dominant_team_per_player', conn, if_exists = 'replace')
conn.close()

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

string_features.teamID.value_counts(normalize = True)

def cleanup_data(df, cutoffPercent = .01):
    for col in df:
        sizes = df[col].value_counts(normalize = True)
        values_to_delete = sizes[sizes < cutoffPercent].index
        df[col].ix[df[col].isin(values_to_delete)] = 'Other'
    return df

encoded_data = pandas.DataFrame(index = string_features.index)
for col in string_features.columns:
    data = pandas.get_dummies(string_features[col], prefix=col.encode('ascii', 'replace'))
    encoded_data = pandas.concat([encoded_data, data], axis=1)

encoded_data.head()

def get_binary_values(data_frame):
    """encodes cateogrical features in Pandas.
    """
    all_columns = pandas.DataFrame( index = data_frame.index)
    for col in data_frame.columns:
        data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pandas.concat([all_columns, data], axis=1)
    return all_columns
    
from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

explanatory_df = pandas.concat([numeric_features, encoded_data],axis = 1)
explanatory_df.head()
explanatory_df['no_variation'] = 1

toKeep = []
toDelete = []

for col in explanatory_df:
    if len(explanatory_df[col].value_counts()) > 1:
        toKeep.append(col)
    else:
        toDelete.append(col)

print toKeep
print toDelete


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


import matplotlib.pyplot as plt
import numpy

corr_matrix = explanatory_df.corr()
corr_matrix.ix[:,:] =  numpy.tril(corr_matrix.values, k = -1)

already_in = set()
result = []
for col in corr_matrix:
    perfect_corr = corr_matrix[col][abs(numpy.round(corr_matrix[col],10)) == 1.00].index.tolist()
    if perfect_corr and col not in already_in:
        already_in.update(set(perfect_corr))
        perfect_corr.append(col)
        result.append(perfect_corr)
        
print result

toRemove = []
for item in result:
    toRemove.append(item[1:(len(item)+1)])
toRemove = sum(toRemove, [])

explanatory_df.drop(toRemove, 1, inplace = True)


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

    
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)

from sklearn.preprocessing import Imputer
numeric_features = df.ix[:, df.dtypes != 'object']
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), 
                                    columns = numeric_features.columns)
                                    
from sklearn.feature_selection import RFECV
from sklearn import tree

class TreeClassifierWithCoef(tree.DecisionTreeClassifier):
    def fit(self, *args, **kwargs):
        super(tree.DecisionTreeClassifier, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

decision_tree = TreeClassifierWithCoef(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)

rfe_cv = RFECV(estimator=decision_tree, step=1, cv=10,
              scoring='roc_auc', verbose = 1)
rfe_cv.fit(explanatory_df, response_series)

print "Optimal number of features :{0} of {1} considered".format(rfe_cv.n_features_,len(explanatory_df.columns))
print rfe_cv.grid_scores_

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (ROC_AUC)")
plt.plot(range(1, len(rfe_cv.grid_scores_) + 1), rfe_cv.grid_scores_)
plt.show()

features_used = explanatory_df.columns[rfe_cv.get_support()]
print features_used

final_estimator_used = rfe_cv.estimator_



### Bringing in data post 2000 and transforming
conn = sqlite3.connect('/Users/MatthewCohen/Documents/SQLite/lahman2013.sqlite')

cur = conn.cursor()

table_creation_query = """
CREATE TABLE hall_of_fame_inductees4 as  
select playerID, case when average_inducted = 0 then 0 else 1 end as inducted from (
select playerID, avg(case when inducted = 'Y' then 1 else 0 end ) as average_inducted from  HallOfFame hf
where yearid > 2000
group by playerID
) bb;"""

cur.execute(table_creation_query)

cur.close()

monster_query2 = """
select m.nameGiven, d.teamID, hfi.inducted, batting.*, pitching.*, fielding.* from hall_of_fame_inductees4 hfi 
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
df2 = pandas.read_sql(monster_query2, conn)
conn.close()

df2.drop('playerID',  1, inplace = True)

explanatory_features = [col for col in df2.columns if col not in ['nameGiven', 'inducted']]
explanatory_df2 = df2[explanatory_features]

explanatory_df2.dropna(how='all', inplace = True) 

explanatory_colnames = explanatory_df2.columns

response_series = df2.inducted
response_series.dropna(how='all', inplace = True) 

response_series.index[~response_series.index.isin(explanatory_df2.index)]

string_features = explanatory_df2.ix[:, explanatory_df2.dtypes == 'object']
numeric_features = explanatory_df2.ix[:, explanatory_df2.dtypes != 'object']

string_features = string_features.fillna('Nothing')

for col in string_features:
    sizes = string_features[col].value_counts(normalize = True)
    values_to_delete = sizes[sizes<0.01].index
    string_features[col].ix[string_features[col].isin(values_to_delete)] = "Other"

string_features.teamID.value_counts(normalize = True)

def cleanup_data(df2, cutoffPercent = .01):
    for col in df2:
        sizes = df2[col].value_counts(normalize = True)
        values_to_delete = sizes[sizes < cutoffPercent].index
        df2[col].ix[df2[col].isin(values_to_delete)] = 'Other'
    return df2

encoded_data = pandas.DataFrame(index = string_features.index)
for col in string_features.columns:
    data = pandas.get_dummies(string_features[col], prefix=col.encode('ascii', 'replace'))
    encoded_data = pandas.concat([encoded_data, data], axis=1)

encoded_data.head()

def get_binary_values(data_frame):
    """encodes cateogrical features in Pandas.
    """
    all_columns = pandas.DataFrame( index = data_frame.index)
    for col in data_frame.columns:
        data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pandas.concat([all_columns, data], axis=1)
    return all_columns

    
from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

explanatory_df2 = pandas.concat([numeric_features, encoded_data],axis = 1)
explanatory_df2.head()
explanatory_df2['no_variation'] = 1

toKeep = []
toDelete = []

for col in explanatory_df2:
    if len(explanatory_df2[col].value_counts()) > 1:
        toKeep.append(col)
    else:
        toDelete.append(col)

print toKeep
print toDelete


def find_zero_var(df2):
    """finds columns in the dataframe with zero variance -- ie those
        with the same value in every observation.
    """   
    toKeep = []
    toDelete = []
    for col in df2:
        if len(df2[col].value_counts()) > 1:
            toKeep.append(col)
        else:
            toDelete.append(col)
        ##
    return {'toKeep':toKeep, 'toDelete':toDelete} 


import matplotlib.pyplot as plt
import numpy

corr_matrix = explanatory_df2.corr()
corr_matrix.ix[:,:] =  numpy.tril(corr_matrix.values, k = -1)

already_in = set()
result = []
for col in corr_matrix:
    perfect_corr = corr_matrix[col][abs(numpy.round(corr_matrix[col],10)) == 1.00].index.tolist()
    if perfect_corr and col not in already_in:
        already_in.update(set(perfect_corr))
        perfect_corr.append(col)
        result.append(perfect_corr)
        
print result

toRemove = []
for item in result:
    toRemove.append(item[1:(len(item)+1)])
toRemove = sum(toRemove, [])

explanatory_df2.drop(toRemove, 1, inplace = True)


def find_perfect_corr(df2):
    """finds columns that are eother positively or negatively perfectly correlated (with correlations of +1 or -1), and creates a dict 
        that includes which columns to drop so that each remaining column
        is independent
    """  
    corrMatrix = df2.corr()
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

    
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df2)
explanatory_df2 = pandas.DataFrame(scaler.transform(explanatory_df2), columns = explanatory_df2.columns)

from sklearn.preprocessing import Imputer
numeric_features = df2.ix[:, df2.dtypes != 'object']
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), 
                                    columns = numeric_features.columns)

from sklearn.feature_selection import RFECV
from sklearn import tree

rfe_cv.fit(explanatory_df2, response_series)
rfe_cv.predict(explanatory_df2)

print "Optimal number of features :{0} of {1} considered".format(rfe_cv.n_features_,len(explanatory_df2.columns))
print rfe_cv.grid_scores_

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (ROC_AUC)")
plt.plot(range(1, len(rfe_cv.grid_scores_) + 1), rfe_cv.grid_scores_)
plt.show()

features_used = explanatory_df.columns[rfe_cv.get_support()]
print features_used





#               IGNORE                  #

################################################################################
################################################################################
###############################################################################
class TreeClassifierWithCoef(tree.DecisionTreeClassifier):
    def fit(self, *args, **kwargs):
        super(tree.DecisionTreeClassifier, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

decision_tree = TreeClassifierWithCoef(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)

rfe_cv = RFECV(estimator=decision_tree, step=1, cv=10,
              scoring='roc_auc', verbose = 1)
rfe_cv.fit(explanatory_df, response_series)

print "Optimal number of features :{0} of {1} considered".format(rfe_cv.n_features_,len(explanatory_df.columns))
print rfe_cv.grid_scores_

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (ROC_AUC)")
plt.plot(range(1, len(rfe_cv.grid_scores_) + 1), rfe_cv.grid_scores_)
plt.show()

features_used = explanatory_df.columns[rfe_cv.get_support()]
print features_used

final_estimator_used = rfe_cv.estimator_
################################################################################
################################################################################
################################################################################
































