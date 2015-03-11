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

  
