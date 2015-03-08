# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:40:50 2015

@author: Margaret
"""

import pandas
import sqlite3
from sklearn import preprocessing
import numpy as np

# putting a setting into pandas that lets you print out the entire
# DataFrame when you use the .head() method
pandas.set_option('display.max_columns', None)


query = """
SELECT t.teamID, sum(t.W) as total_wins, sum(b.R) as total_runs FROM Teams t
 INNER JOIN Batting b on t.teamID = b.teamID 
 WHERE b.R is not null AND t.yearID > 2000
GROUP BY t.teamID
ORDER BY total_runs desc
"""

con = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')
df = pandas.read_sql(query, con)
con.close()

### SCALE THE DATA
data = df[['total_runs', 'total_wins']]
scaler = preprocessing.StandardScaler()
scaler.fit(data)
data = pandas.DataFrame(scaler.transform(data), columns = data.columns)

## increaseing figure size of plots fo better readability
#from pylab import rcParams
#rcParams['figure.figsize'] = 10, 5

## plot the scaled data
plt = df.plot(x='total_runs', y='total_wins', kind='scatter')

## annotating with team names
for i, txt in enumerate(df.teamID):
    plt.annotate(txt, (df.total_runs[i], df.total_wins[i])) # txt, x, y for that point
    
######################
### K-MEANS CLUSTERING
######################

# plotting the data, it looks like there are 3 clusters
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
kmeans_est = KMeans(n_clusters = 3)

kmeans_est.fit(data)
labels = kmeans_est.labels_
# teams = len(labels)

plt.scatter(df.total_runs, df.total_wins, s=60, c=labels)

##########
### DBSCAN
##########

from sklearn.cluster import DBSCAN

# getting around a bug that doesn't let you fit to a dataframe, make it a NumPy array
dbsc = DBSCAN().fit(np.array(data))
labels = dbsc.labels_

plt.scatter(df.total_runs, df.total_wins, s=60, c=labels)


###########################
### HIERARCHICAL CLUSTERING
###########################

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, fclusterdata

distanceMatrix = pdist(data)

# move threhold to 3 to get 3 clusters
threshold = 3
dend = dendrogram(linkage(distanceMatrix, method = 'complete'),
                  color_threshold = threshold,
                  leaf_font_size = 10,
                  labels = df.teamID.tolist())

## get cluster assignments
assignments = fcluster(linkage(distanceMatrix, method = 'complete'), threshold, 'distance')

# create DataFrame with cluster assignments and names
cluster_output = pandas.DataFrame({'team': df.teamID.tolist(), 'cluster':assignments})

## plot the results
plt.scatter(df.total_runs, df.total_wins, s=60, c=cluster_output.cluster)

## make it prettier
colors = cluster_output.cluster
colors[colors==1]='g'
colors[colors==2]='r'
colors[colors==3]='y'

plt.scatter(df.total_runs, df.total_wins, s=60, c=colors, lw=0)

#####################
### PCA Preprocessing
#####################

import sqlite3
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


con = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')
query = """
SELECT (m.nameFirst||" "||nameLast) as p_name, m.height as height, m.weight as weight, m.bats as bats, m.throws as throws,
inducted, bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_strikes, bat_stolen, bat_baseballs, bat_intentwalks
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, pitch_wild, pitch_saves,
f_putouts, f_assists, f_errors FROM Master m
INNER JOIN
(SELECT pID, dom.teamID as dom_team, inducted, 
bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_RBI, bat_caught, bat_baseballs, 
bat_intentwalks, bat_doubles, bat_triples, bat_strikes, bat_stolen, 
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, pitch_wild, pitch_games, pitch_saves,
pitch_earnruns, pitch_runsallowed, pitch_finish, pitch_outs, pitch_hits, pitch_opp_BA, 
f_putouts, f_assists, f_errors FROM dominant_team_per_player dom
INNER JOIN
(SELECT h.playerID as pID, max(CASE WHEN h.inducted='Y' THEN 1 ELSE 0 END) as inducted, 
bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_RBI, bat_caught, bat_baseballs, 
bat_intentwalks, bat_doubles, bat_triples, bat_strikes, bat_stolen, 
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, pitch_wild, pitch_games, pitch_saves,
pitch_earnruns, pitch_runsallowed, pitch_finish, pitch_outs, pitch_hits, pitch_opp_BA, 
f_putouts, f_assists, f_errors FROM HallofFame h
INNER JOIN
(SELECT f.playerID, f.PO as f_putouts, f.A as f_assists, f.E as f_errors, 
bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_RBI, bat_caught, bat_baseballs, 
bat_intentwalks, bat_doubles, bat_triples, bat_strikes, bat_stolen, 
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, pitch_wild, pitch_games, pitch_saves,
pitch_earnruns, pitch_runsallowed, pitch_finish, pitch_outs, pitch_hits, pitch_opp_BA FROM Fielding f
LEFT JOIN
(SELECT b.playerID, b.lgID as bat_league, sum(b.R) as bat_runs, sum(b.H) as bat_hits, sum(b.CS) as bat_caught, sum(b.BB) as bat_baseballs,
sum(b.AB) as at_bats, sum(b.HR) as bat_homeruns, sum(b.RBI) as bat_RBI, sum(b.SO) as bat_strikes, sum(b.SB) as bat_stolen,
sum(b.IBB) as bat_intentwalks, sum(b.'2B') as bat_doubles, sum(b.'3B') as bat_triples,
sum(p.W) as pitch_wins, sum(p.SO) as pitch_strikes, sum(p.SHO) as pitch_shuts, avg(p.ERA) as pitch_ERA, sum(p.WP) as pitch_wild,
sum(p.G) as pitch_games, sum(p.SV) as pitch_saves, sum(p.ER) as pitch_earnruns, sum(p.R) as pitch_runsallowed, sum(p.GF) as pitch_finish, 
sum(p.IPOuts) as pitch_outs, sum(p.HBP) as pitch_hits, sum(p.BAOpp) as pitch_opp_BA
FROM Batting b
LEFT JOIN Pitching p on p.playerID = b.playerID
GROUP BY b.playerID) batpitch on batpitch.playerID = f.playerID
GROUP BY batpitch.playerID) positions
ON positions.playerID = h.playerID
WHERE h.yearID < 2000 AND h.yearID > 1965
GROUP BY h.playerID) all_features on pID = dom.playerID) all_data on pID = m.playerID
"""
df = pandas.read_sql(query, con)
con.close()

df.drop('p_name',  1, inplace = True)


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
explanatory_df.drop(no_variation['toDelete'], inplace = True)

# deleting perfect correlation
no_correlation = find_perfect_corr(explanatory_df)
explanatory_df.drop(no_correlation['toRemove'], 1, inplace = True)

# scaling data
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)

#######
### PCA
#######

from sklearn.decomposition import PCA
pca = PCA(n_components = 6)
pca.fit(explanatory_df)

# extracting components
pca_df = pandas.DataFrame(pca.transform(explanatory_df))

## plotting first to principal components
pca_df.plot(x=0, y=1, kind='scatter')
# can see orthogonality by the right hand
# getting feature set as uncorrelated as possible


## making the scree plot
variance_df = pandas.DataFrame({'variance': pca.explained_variance_, 
'principal component': pca_df.columns.tolist()})
variance_df['principal component'] = variance_df['principal component'] +1
variance_df.plot(x='principal component', y='variance')
# looks like variance stops getting explained after the first two principal components

pca_df_small = pca_df.ix[:,0:1]
# orthogonal relationship between your data

## getting cross-val score of transformed data
rf = ensemble.RandomForestClassifier(n_estimators = 500)
roc_scores_rf_pca = cross_val_score(rf, pca_df_small, response_series, cv=10, scoring='roc_auc',
                                    n_jobs = -1)
                                    
print 'Mean ROC Scores for the Small Group: %f' % roc_scores_rf_pca.mean()

## getting cross val score for just raw value
roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring='roc_auc',
                                n_jobs = -1)
print 'Mean ROC Scores for the Raw Data: %f' % roc_scores_rf.mean()

## getting cross-val score of large group of pca_df
bt = ensemble.GradientBoostingClassifier(n_estimators = 500)
roc_scores_rf_pca = cross_val_score(bt, pca_df, response_series, cv=10, scoring='roc_auc',
                                    n_jobs = -1)
                                    
print 'Mean ROC Scores for the Whole PCA Group: %f' % roc_scores_rf_pca.mean()
# there are some nonlinear relationships that are not being explained by PCA


###########################
### SUPPORT VECTOR MACHINES
###########################

from sklearn.svm import SVC

## running quadratic kernel without PCA
svm = SVC(kernel = 'rbf')

roc_scores_svm = cross_val_score(svm, explanatory_df, response_series, cv=10, scoring='roc_auc',
                                 n_jobs = -1)
print 'Mean ROC Scores for the Support Vector Machine: %f' % roc_scores_svm.mean()

# try with PCA
roc_scores_svm_pca = cross_val_score(svm, pca_df_small, response_series, cv=10, scoring='roc_auc',
                                     n_jobs=01)
                                     
print 'Mean ROC Scores for the Small Group SVM: %f' % roc_scores_svm_pca.mean()

# let's do grid search on optimal kernel
param_grid = dict(kernel = ['linear', 'poly', 'rbf', 'sigmoid'])

svm_grid = GridSearchCV(svm, param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
svm_grid.fit(explanatory_df, response_series)
best_estimator = svm_grid.best_estimator_
print "Best Kernel Function: %s" %best_estimator.kernel
print "Best ROC Scores for SVM: %f" %svm_grid.best_score_



######################
### PREDICTION METHODS
######################

# putting a setting into pandas that lets you print out the entire
# DataFrame when you use the .head() method
pandas.set_option('display.max_columns', None)


con = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')
query = """
SELECT (m.nameFirst||" "||nameLast) as p_name, m.height as height, m.weight as weight, m.bats as bats, m.throws as throws,
inducted, bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_strikes, bat_stolen, bat_baseballs, bat_intentwalks
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, pitch_wild, pitch_saves,
f_putouts, f_assists, f_errors FROM Master m
INNER JOIN
(SELECT pID, dom.teamID as dom_team, inducted, 
bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_RBI, bat_caught, bat_baseballs, 
bat_intentwalks, bat_doubles, bat_triples, bat_strikes, bat_stolen, 
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, pitch_wild, pitch_games, pitch_saves,
pitch_earnruns, pitch_runsallowed, pitch_finish, pitch_outs, pitch_hits, pitch_opp_BA, 
f_putouts, f_assists, f_errors FROM dominant_team_per_player dom
INNER JOIN
(SELECT h.playerID as pID, max(CASE WHEN h.inducted='Y' THEN 1 ELSE 0 END) as inducted, 
bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_RBI, bat_caught, bat_baseballs, 
bat_intentwalks, bat_doubles, bat_triples, bat_strikes, bat_stolen, 
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, pitch_wild, pitch_games, pitch_saves,
pitch_earnruns, pitch_runsallowed, pitch_finish, pitch_outs, pitch_hits, pitch_opp_BA, 
f_putouts, f_assists, f_errors FROM HallofFame h
INNER JOIN
(SELECT f.playerID, f.PO as f_putouts, f.A as f_assists, f.E as f_errors, 
bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_RBI, bat_caught, bat_baseballs, 
bat_intentwalks, bat_doubles, bat_triples, bat_strikes, bat_stolen, 
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, pitch_wild, pitch_games, pitch_saves,
pitch_earnruns, pitch_runsallowed, pitch_finish, pitch_outs, pitch_hits, pitch_opp_BA FROM Fielding f
LEFT JOIN
(SELECT b.playerID, b.lgID as bat_league, sum(b.R) as bat_runs, sum(b.H) as bat_hits, sum(b.CS) as bat_caught, sum(b.BB) as bat_baseballs,
sum(b.AB) as at_bats, sum(b.HR) as bat_homeruns, sum(b.RBI) as bat_RBI, sum(b.SO) as bat_strikes, sum(b.SB) as bat_stolen,
sum(b.IBB) as bat_intentwalks, sum(b.'2B') as bat_doubles, sum(b.'3B') as bat_triples,
sum(p.W) as pitch_wins, sum(p.SO) as pitch_strikes, sum(p.SHO) as pitch_shuts, avg(p.ERA) as pitch_ERA, sum(p.WP) as pitch_wild,
sum(p.G) as pitch_games, sum(p.SV) as pitch_saves, sum(p.ER) as pitch_earnruns, sum(p.R) as pitch_runsallowed, sum(p.GF) as pitch_finish, 
sum(p.IPOuts) as pitch_outs, sum(p.HBP) as pitch_hits, sum(p.BAOpp) as pitch_opp_BA
FROM Batting b
LEFT JOIN Pitching p on p.playerID = b.playerID
GROUP BY b.playerID) batpitch on batpitch.playerID = f.playerID
GROUP BY batpitch.playerID) positions
ON positions.playerID = h.playerID
WHERE h.yearID >= 2000
GROUP BY h.playerID) all_features on pID = dom.playerID) all_data on pID = m.playerID
"""
df = pandas.read_sql(query, con)
con.close()

df.drop('p_name',  1, inplace = True)


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
explanatory_df.drop(no_variation['toDelete'], inplace = True)

# deleting perfect correlation
no_correlation = find_perfect_corr(explanatory_df)
explanatory_df.drop(no_correlation['toRemove'], 1, inplace = True)

# scaling data
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)



#################
## RANDOM FORESTS
#################

# Random Forest object
# default values
rf = ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, 
    min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, 
    bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, min_density=None, 
    compute_importances=None)

# changing estimators
rf = ensemble.RandomForestClassifier(n_estimators = 500)

# computing ROC AUC of random forest
roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, 
                                scoring='roc_auc', n_jobs=-1)

# computing ROC AUC of decision tree just to have a benchmark                                
roc_score_tree = cross_val_score(tree.DecisionTreeClassifier(), explanatory_df, response_series,
                                 cv=10, scoring = 'roc_auc', n_jobs = -1)

# compare the mean ROC AUC
print "Random Forest Mean ROC AUC %f" % roc_scores_rf.mean()
print "Decision Tree Mean ROC AUC %f" % roc_score_tree.mean()


# perform grid search to find the optimal number of treees
trees_range = range(200, 500, 10)
param_grid = dict(n_estimators = trees_range)

grid = GridSearchCV(rf, param_grid, cv=10, scoring = 'roc_auc', n_jobs = -1)
grid.fit(explanatory_df, response_series)

# check out the scores
grid_mean_scores = [result[1] for result in grid.grid_scores_]

# plot results
plt.figure()
plt.plot(trees_range, grid_mean_scores)


# pick best estimator and print ROC AUC
best_rf_est = grid.best_estimator_

# how many trees did the best estimator have?
print "Number of Trees in Best Estimator: %d" % best_rf_est.n_estimators
# what is the best score
print "Accuracy for Best Estimator : %f" % grid.best_score_

##################
### BOOSTING TREES
##################

boosting_tree = ensemble.GradientBoostingClassifier()

roc_scores_gbm = cross_val_score(boosting_tree, explanatory_df, response_series, cv=10,
                                 scoring = 'roc_auc', n_jobs = -1)
                                 
# compare the mean ROC AUC
print "Boosting Tree Mean ROC AUC %f" % roc_scores_gbm.mean()   
print "Random Forest Mean ROC AUC %f" % roc_scores_rf.mean()
print "Decision Tree Mean ROC AUC %f" % roc_score_tree.mean()   


# tune for num_trees, learning rate, and subsampling percent
# import arange to create ranges for floats

from numpy import arange

# ranges suggested by scikit learn

learning_rate_range = arange(0.01, 0.4, 0.02) # most important feature/parameter to tune, over 0.04 isn't great
subsampling_range = arange(0.2, 1, 0.1) # 0 to 1
n_estimators_range = range(20, 100, 20)  # over 100 isn't usually helpful by experience

param_grid = dict(learning_rate = learning_rate_range, n_estimators = n_estimators_range, 
                  subsample = subsampling_range)

gbm_grid = GridSearchCV(boosting_tree, param_grid, cv=10, scoring='roc_auc', n_jobs = -1)
gbm_grid.fit(explanatory_df, response_series)

# small subsample = data is prone to overfitting
# low number of estimators = each has to have more importance, could mean edge cases have more meaning

# find the winning parameters
print gbm_grid.best_params_

# pull out the best score
print "Best Boosting Score: %f" % gbm_grid.best_score_


###########################
### SUPPORT VECTOR MACHINES
###########################

from sklearn.svm import SVC

## running quadratic kernel without PCA
svm = SVC(kernel = 'rbf')

roc_scores_svm = cross_val_score(svm, explanatory_df, response_series, cv=10, scoring='roc_auc',
                                 n_jobs = -1)
print 'Mean ROC Scores for the Support Vector Machine: %f' % roc_scores_svm.mean()

# let's do grid search on optimal kernel
param_grid = dict(kernel = ['linear', 'poly', 'rbf', 'sigmoid'])

svm_grid = GridSearchCV(svm, param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
svm_grid.fit(explanatory_df, response_series)
best_estimator = svm_grid.best_estimator_
print "Best Kernel Function: %s" %best_estimator.kernel
print "Best ROC Scores for SVM: %f" %svm_grid.best_score_