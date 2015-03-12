# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:20:34 2015

@author: jchen
"""

import pandas
import sqlite3
from sklearn import preprocessing
import numpy as np

pandas.set_option('display.max_columns', None)

# look at salaries and wins by team
query = """
select s.teamID, 
    sum(s.salary) as total_salaries, 
    sum(t.W) as total_wins
from salaries s
inner join teams t on s.teamID=t.teamID
where t.yearID > 1990
group by s.teamID
order by total_wins desc
"""
con = sqlite3.connect('/Users/jchen/Documents/SQLite/lahman2013.sqlite')
df = pandas.read_sql(query, con)
con.close()

# scale data
data = df[['total_salaries', 'total_wins']]
scaler = preprocessing.StandardScaler()
scaler.fit(data)
data = pandas.DataFrame(scaler.transform(data), columns = data.columns)

# plot the scaled data
plt = df.plot(x='total_salaries', y='total_wins', kind='scatter')

# annotate with team names
for i, txt in enumerate(df.teamID):
    plt.annotate(txt, (df.total_salaries[i],df.total_wins[i]))
plt.show()
# looks like there's a small cluster at the bottom end of wins
# majority cluster in the mid-range of salaries
# and some outliers - NY and a few others in between clusters.

#########
# K-Means
#########

from sklearn.cluster import KMeans
import matplotlib.pylab as plt
kmeans_est = KMeans(n_clusters=3)

kmeans_est.fit(data)
labels = kmeans_est.labels_

plt.scatter(df.total_salaries, df.total_wins, s=60, c=labels)
# K-means forms 3 clusters
# two seem pretty distinct, however the third (including NY) seems close to the middle cluster

########
# DBSCAN
########

from sklearn.cluster import DBSCAN

dbsc = DBSCAN().fit(np.array(data))
labels = dbsc.labels_

plt.scatter(df.total_salaries, df.total_wins, s=60, c=labels)
# This one looks better, in my opinion, with 4 clusters, one of which is only NY

# Of K-means and DBSCAN, DBSCAN is better at identifying outliers

############
# Dendrogram
############
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, fclusterdata

distanceMatrix = pdist(data)

# print dendrogram
dend = dendrogram(linkage(distanceMatrix, method='complete'), 
           color_threshold=1, 
           leaf_font_size=10,
           labels = df.teamID.tolist())
# This give us 7 clusters
              
# let's set the cutoff at 2 for 4 clusters
dend = dendrogram(linkage(distanceMatrix, method='complete'), 
           color_threshold=2, 
           leaf_font_size=10,
           labels = df.teamID.tolist())

# get cluster assignments
assignments = fcluster(linkage(distanceMatrix, method='complete'),2,'distance')

cluster_output = pandas.DataFrame({'team':df.teamID.tolist() , 'cluster':assignments})

# change the colors of the graph
colors = cluster_output.cluster
colors[colors == 1] = 'b'
colors[colors == 2] = 'g'
colors[colors == 3] = 'r'
colors[colors == 4] = 'y'

# Plot 
plt.scatter(df.total_salaries, df.total_wins, s=100, c=colors,  lw=0)

#########
# Part II
#########

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

# Functions from class

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
    
###############
# Pull in the data from hw 9, do the same preprocessing
conn = sqlite3.connect('/Users/jchen/Documents/SQLite/lahman2013.sqlite')  
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

# Find rows where explanatory features were removed
response_series.index[~response_series.index.isin(explanatory_df.index)] 
# Looks like none.

# Separate numeric from string features
string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']

# Fill NaNs in string features
string_features = string_features.fillna('Nothing')

# Bin categorical features with threshold of 1%
string_features = cleanup_data(string_features, cutoffPercent = .01)
# Check the data
string_features.teamID.value_counts(normalize = True) 

# Encode categorical features
encoded_data = get_binary_values(string_features)
encoded_data.head()

# Impute vales for numeric features
from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features) # Store imputer in object
numeric_features = pd.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

# recombine numeric and categorical features into one df
explanatory_df = pd.concat([numeric_features, encoded_data],axis = 1)
explanatory_df.head()

# Find features with no variance
no_var = find_zero_var(explanatory_df)['toDelete']
explanatory_df.drop(no_var, inplace = True)
# nothing to drop

# Find features with perfect correlation
perf_cor = find_perfect_corr(explanatory_df)['toRemove']
explanatory_df.drop(perf_cor, 1, inplace = True)
# no features with perfect correlation

# Scale data
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pd.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)

######
# PCA
######
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pca.fit(explanatory_df)

# extract the components
pca_df = pandas.DataFrame(pca.transform(explanatory_df))

# plotting the first two principal components
pca_df.plot(x = 0, y= 1, kind = 'scatter')


# Scree plot
variance_df = pandas.DataFrame({'variance': pca.explained_variance_, 'principal component': pca_df.columns.tolist()})
# renaming the nth component
variance_df['principal component'] = variance_df['principal component'] + 1
variance_df.plot(x = 'principal component', y= 'variance')
# Looks like there are 'crooks' at both the 2nd and 6th componensts
# Let's only take the first two
pca_df_small = pca_df.ix[:,0:1]

# Get RF cross-val score of transformed data
rf = ensemble.RandomForestClassifier(n_estimators= 500)
roc_scores_rf_pca = cross_val_score(rf, pca_df_small, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_rf_pca.mean()
# 79% accuracy. (0.786488598256)

# Let's compare this to the original adata
roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_rf.mean()
# 89% (0.888310026828) - so PCA here created information loss

# Now get boosting tree CV scores on small data
boosting_tree = ensemble.GradientBoostingClassifier()
roc_scores_gbm_pca = cross_val_score(boosting_tree, pca_df_small, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_gbm_pca.mean()
# 80% accuracy. (0.806741784038) Similar to RF

# Original data
roc_scores_gbm = cross_val_score(boosting_tree, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_gbm.mean()
# 89% accuracy (0.888386317907) 


######
# SVM
######
from sklearn.svm import SVC

# first, run quadratic kernel without PCA
svm = SVC(kernel='poly')

roc_scores_svm = cross_val_score(svm, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_svm.mean()
# 82% acccuracy (0.823437290409)

# Now with PCA
roc_scores_svm_pca = cross_val_score(svm, pca_df_small, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_svm_pca.mean()
# 79% acccuracy (0.78752045607) - so PCA did worse here as well

# Grid search for the optimal kernel
param_grid = dict(kernel = ['linear','poly','rbf','sigmoid'])

svm_grid = GridSearchCV(svm, param_grid, cv=10, scoring='roc_auc', n_jobs = -1)
svm_grid.fit(explanatory_df, response_series)
best_estimator = svm_grid.best_estimator_
print best_estimator.kernel
# Linear is the best estimator
print svm_grid.best_score_
# Best estimator was 87% accurate - just below RF abd GBM

########################
# Test on post-2000 data
########################
# Pull in data from 2000 onward
conn = sqlite3.connect('/Users/jchen/Documents/SQLite/lahman2013.sqlite')
# Note hall of fame inductees table for post-2000 already exists
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
# Keep only columns used in training
# First add in missing columns 
add_columns = [col for col in explanatory_df.columns if col not in explanatory_df_post_2000.columns]
add_df = pd.DataFrame(0, index = explanatory_df_post_2000.index, columns = add_columns)

explanatory_df_post_2000_new = pd.concat([explanatory_df_post_2000, add_df], axis = 1)

# rearrange columns
explanatory_df_post_2000 = explanatory_df_post_2000_new[explanatory_df.columns]
explanatory_df_post_2000 = pd.DataFrame(scaler.transform(explanatory_df_post_2000), columns = explanatory_df_post_2000.columns)

###########################
# Predict on post-2000 data

from sklearn import metrics

predicted_svm = best_estimator.predict(explanatory_df_post_2000)
cm_svm = pd.crosstab(response_series_post_2000, predicted_svm, rownames=['True Label'], colnames=['Predicted Label'], margins=True)
print cm_svm
'''
Predicted Label    0  1  All
True Label                  
0                213  5  218
1                 36  2   38
All              249  7  256
'''
# This is pretty good at predicting induction on OOS data

