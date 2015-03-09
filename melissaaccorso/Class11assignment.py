# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 17:57:38 2015

@author: melaccor
"""

import pandas
import sqlite3
from sklearn import preprocessing
import numpy as np

pandas.set_option('display.max_columns', None)

conn=sqlite3.connect('C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite')
# open a cursor as we are executing a SQL statement that does not produce a pandas DataFrame
cur = conn.cursor()
# writing the query to simplify creating our response feature. 
sql = """
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
(select playerID, sum(InnOuts) as timewithouts, sum(PO) as putouts, sum(E) as errors, sum(DP) as doubleplays, sum(a) as assists
from Fielding
group by playerID) fielding on fielding.playerID = a.playerID
left outer join
dominant_team_per_player player on player.playerID = a.playerID
left outer join 
dominant_league_per_player lg on lg.playerID = a.playerID;"""
df = pandas.read_sql(sql, conn)
# closing the connection.
conn.close()

# dropping duplicate playerID columns
df.drop('playerID',  1, inplace = True)

df.head(10)
df.columns

#Count NAs
df.dropna(how='any', inplace = True) 

# scaling data
data = df[['doubleplays', 'totalhits']]
scaler = preprocessing.StandardScaler()
scaler.fit(data)
data = pandas.DataFrame(scaler.transform(data), columns = data.columns)

## increasing the figure size of my plots for better readabaility.
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5

# plot the scaled data
plt = df.plot(x='doubleplays', y='totalhits', kind='scatter')

########
# K-Means
########

# plotting the data, it looks like there's 3 clusters -- one 
# 'big' cluster, another of low-preforming teams, and the Yankees.
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
kmeans_est = KMeans(n_clusters=3)

kmeans_est.fit(data)
labels = kmeans_est.labels_

plt.scatter(df.doubleplays, df.totalhits, s=60, c=labels)
#eh, it's okay.  Not great.

########
# DBSCAN
########

from sklearn.cluster import DBSCAN

## getting around a bug that doesn't let you fit to a dataframe
# by coercing it to a NumPy array.
dbsc = DBSCAN().fit(np.array(data))
labels = dbsc.labels_

plt.scatter(df.doubleplays, df.totalhits, s=60, c=labels)
# blah, not a ton better.

########
# hirearchical clustering
########
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, fclusterdata

## increasing the figure size of my plots for better readabaility.
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5

distanceMatrix = pdist(data)

## adjust color_threshold to get the number of clusters you want.
dend = dendrogram(linkage(distanceMatrix, method='complete'), 
           color_threshold=1, 
           leaf_font_size=10,
           labels = df.teamID.tolist())

# notice I moved the threshold to 4 so I can get 3 clusters. 
dend = dendrogram(linkage(distanceMatrix, method='complete'), 
           color_threshold=4, 
           leaf_font_size=10,
           labels = df.teamID.tolist())

## get cluster assignments
assignments = fcluster(linkage(distanceMatrix, method='complete'),4,'distance')

#create DataFrame with cluster assignments and names
cluster_output = pandas.DataFrame({'team':df.teamID.tolist() , 'cluster':assignments})

# let's plot the results
plt.scatter(df.doubleplays, df.totalhits, s=60, c=cluster_output.cluster)
## there we go -- this plot makes more sense than K-Means or DBSCAN.

## let's make the plot prettier with better assingments.  
colors = cluster_output.cluster
colors[colors == 1] = 'b'
colors[colors == 2] = 'g'
colors[colors == 3] = 'r'

plt.scatter(df.doubleplays, df.totalhits, s=100, c=colors,  lw=0)


#############
###  PCA  ###
#############

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


#Using the data from the last homework (above), perform principal component analysis on your data.

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
# PCA
######
from sklearn.decomposition import PCA

pca = PCA(n_components=6)
pca.fit(explanatory_df)

# extracting the components
pca_df = pandas.DataFrame(pca.transform(explanatory_df))

## plotting the first to principal components
pca_df.plot(x = 0, y= 1, kind = 'scatter')

# making a screen plot
variance_df = pandas.DataFrame({'variance': pca.explained_variance_, 'principal component': pca_df.columns.tolist()})
# adding one to principal components (since there is no 0th component)
variance_df['principal component'] = variance_df['principal component'] + 1
variance_df.plot(x = 'principal component', y= 'variance')
## looks like variance stops getting explained after the first two principal components.

pca_df_small = pca_df.ix[:,0:1]

## getting cross-val score of transformed data
rf = ensemble.RandomForestClassifier(n_estimators= 500)
roc_scores_rf_pca = cross_val_score(rf, pca_df_small, response_series, cv=10, scoring='roc_auc')

print roc_scores_rf_pca.mean()
## 92% accuracy.

# Let's compare this to the original adata
roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring='roc_auc')
print roc_scores_rf.mean()
## 96% accuracy - so PCA actually created information LOSS!


#######################################
#Run a Boosting Tree on the components 
#######################################

boosting_treehw = ensemble.GradientBoostingClassifier()

roc_scores_gbm = cross_val_score(boosting_treehw, explanatory_df, response_series, cv=10, scoring='roc_auc')
print roc_scores_gbm.mean()
#91% accuracy without PCA

roc_scores_gbm_pca = cross_val_score(boosting_treehw, pca_df_small, response_series, cv=10, scoring='roc_auc')
print roc_scores_gbm_pca.mean()
#85% accuracy with PCA

#########################
# SUPPORT VECTOR MACHINES
#########################
from sklearn.svm import SVC

## first, running quadratic kernel without PCA
svm = SVC(kernel='poly')

roc_scores_svm = cross_val_score(svm, explanatory_df, response_series, cv=10, scoring='roc_auc')
print roc_scores_svm.mean()
## 92% acccuracy


# let's try with PCA
roc_scores_svm_pca = cross_val_score(svm, pca_df_small, response_series, cv=10, scoring='roc_auc')
print roc_scores_svm_pca.mean()
## 86% acccuracy -- so PCA did worse AGAIN


# let's do a grid search on the optimal kernel
param_grid = dict(kernel = ['linear','poly','rbf','sigmoid'])

svm_grid = GridSearchCV(svm, param_grid, cv=10, scoring='roc_auc')
svm_grid.fit(explanatory_df, response_series)
best_estimator = svm_grid.best_estimator_
print best_estimator.kernel
## looks like rbf won out
print svm_grid.best_score_
## best estiamtor was 94% accurate -- so just a hair below RFs.
# Note: remember, SVMs are more accurate than RFs with trending data!
