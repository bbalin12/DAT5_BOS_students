# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 08:16:46 2015

@author: jeppley
"""

from sklearn.svm import SVC
from sklearn.decomposition import PCA
import sqlite3
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn import ensemble
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import  GridSearchCV
from sklearn import tree
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, fclusterdata
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
from pylab import rcParams
import pandas

###############################################################################
#Find some sort of attribute in the Baseball dataset that sits on a 
#two-dimenstional plane and has discrete clusters.
###############################################################################

#Will use players as the examples we will cluster, 

con = sqlite3.connect('C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite')

modeldata = '''select b.playerID,
b.b_hits, f.f_assis
  from 
  (select playerid, inducted
  from hall_of_fame_inductees_3 
   where category = 'Player'
   group by playerid) h
left outer join 
  (select playerid,
    count(distinct yearid) as b_years,
    sum(ab) as b_atbat, 
    sum(r) as b_runs, 
    sum(h) as b_hits, 
    sum(hr) as b_hruns, 
    sum(sb) as b_stbas,
    sum(so) as b_strik
  from batting
  group by playerid 
  HAVING max(yearID) > 1950 and min(yearID) >1950 ) b
  on h.playerid = b.playerid
left outer join
  (select playerid,
    count(distinct yearid) as p_years,
    sum(w) as p_wins,
    sum(l) as p_loss,
    sum(sho) as p_shout,
    sum(sv) as p_saves,
    sum(er) as p_eruns,
    sum(so) as p_stout
  from pitching
  group by playerid) p
  on h.playerid = p.playerid
left outer join
  (select playerid,
    max(teamID) as teamID
  from dominant_team_per_player
  group by playerid) t
  on h.playerid = t.playerid
left outer join
  (select playerid,
    max(POS) as pos
  from dominant_pos_per_player
  group by playerid) o
  on h.playerid = o.playerid    
left outer join
  (select playerid,
     count(distinct yearid) as f_years,
     sum(po) as f_puts,
     sum(a) as f_assis,
     sum(dp) as f_dplay,
     sum(pb) as f_pass
  from fielding
  group by playerid) f
  on h.playerid = f.playerid
  where b.playerID is not null
  group by b.playerID
;'''

dfhw = pandas.read_sql(modeldata, con)
con.close()

dfhw.head(10)


###############################################################################
#Perform K-Means and DBSCAN clustering.
###############################################################################


dfhw.dropna(how='all', inplace = True) 
dfhw.describe()


#extracting only numeric features
numeric_features = dfhw.ix[:, dfhw.dtypes != 'object']
string_features = dfhw.ix[:, dfhw.dtypes == 'object']

#imputing NaNs
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), 
                                    columns = numeric_features.columns)
                                    
#Merged imputed columns back with playerIds
dfhw = pandas.concat([string_features, numeric_features],axis = 1)
dfhw.head()
                                    

# scaling data
data = dfhw[['b_hits', 'f_assis']]
scaler = preprocessing.StandardScaler()
scaler.fit(data)
data = pandas.DataFrame(scaler.transform(data), columns = data.columns)


## increasing the figure size of my plots for better readabaility.
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5

# plot the scaled data
plt = dfhw.plot(x='b_hits', y='f_assis', kind='scatter')



kmeans_est = KMeans(n_clusters=3)

kmeans_est.fit(data)
labels = kmeans_est.labels_

plt.scatter(dfhw.b_hits, dfhw.f_assis, s=60, c=labels)

########
# DBSCAN
########


## getting around a bug that doesn't let you fit to a dataframe
# by coercing it to a NumPy array.
dbsc = DBSCAN().fit(np.array(data))
labels = dbsc.labels_

plt.scatter(dfhw.b_hits, dfhw.f_assis, s=60, c=labels)
#yikes, this is terribly awry


#Determine which better represents your data and the intiution behind why the model was the best for your dataset.
#K Means was better for my model because the data is a bit linear, whereas DBScan seems to work better for more non-linear data



###############################################################################
#Plot a dendrogram of your dataset.
###############################################################################


distanceMatrix = pdist(data)

## adjust color_threshold to get the number of clusters you want.
dend = dendrogram(linkage(distanceMatrix, method='complete'), 
           color_threshold=1, 
           leaf_font_size=10,
           labels = dfhw.playerid.tolist())

# notice I moved the threshold to 4 so I can get 3 clusters. 
dend = dendrogram(linkage(distanceMatrix, method='complete'), 
           color_threshold=4, 
           leaf_font_size=10,
           labels = dfhw.playerid.tolist())


###############################################################################
#Using the data from the last homework, perform principal component analysis on your data.
###############################################################################


con = sqlite3.connect('C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite')

modeldata = '''select h.*, 
  b.b_atbat, b.b_hits, p.p_wins, f.f_puts, t.teamID, o.pos
  from 
  (select playerid, inducted
  from hall_of_fame_inductees_3 
   where category = 'Player'
   group by playerid) h
left outer join 
  (select playerid,
    count(distinct yearid) as b_years,
    sum(ab) as b_atbat, 
    sum(r) as b_runs, 
    sum(h) as b_hits, 
    sum(hr) as b_hruns, 
    sum(sb) as b_stbas,
    sum(so) as b_strik
  from batting
  group by playerid 
  HAVING max(yearID) > 1950 and min(yearID) >1950 ) b
  on h.playerid = b.playerid
left outer join
  (select playerid,
    count(distinct yearid) as p_years,
    sum(w) as p_wins,
    sum(l) as p_loss,
    sum(sho) as p_shout,
    sum(sv) as p_saves,
    sum(er) as p_eruns,
    sum(so) as p_stout
  from pitching
  group by playerid) p
  on h.playerid = p.playerid
left outer join
  (select playerid,
    max(teamID) as teamID
  from dominant_team_per_player
  group by playerid) t
  on h.playerid = t.playerid
left outer join
  (select playerid,
    max(POS) as pos
  from dominant_pos_per_player
  group by playerid) o
  on h.playerid = o.playerid    
left outer join
  (select playerid,
     count(distinct yearid) as f_years,
     sum(po) as f_puts,
     sum(a) as f_assis,
     sum(dp) as f_dplay,
     sum(pb) as f_pass
  from fielding
  group by playerid) f
  on h.playerid = f.playerid
  where b.playerID is not null
;'''

dfhw = pandas.read_sql(modeldata, con)
con.close()

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



# dropping duplicate playerID columns
dfhw.drop('playerid',  1, inplace = True)


## splitting out the explanatory features 
explanatory_features = [col for col in dfhw.columns if col not in ['nameGiven', 'inducted']]
explanatory_dfhw = dfhw[explanatory_features]

# dropping rows with no data.
explanatory_dfhw.dropna(how='all', inplace = True) 

# extracting column names 
explanatory_colnames = explanatory_dfhw.columns

## doing the same for response
response_series = dfhw.inducted
response_series.dropna(how='all', inplace = True) 

## seeing which explanatory feature rows got removed.  Looks like none.
response_series.index[~response_series.index.isin(explanatory_dfhw.index)]

### now, let's seperate the numeric explanatory data from the string data
string_features = explanatory_dfhw.ix[:, explanatory_dfhw.dtypes == 'object']
numeric_features = explanatory_dfhw.ix[:, explanatory_dfhw.dtypes != 'object']


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
explanatory_dfhw = pandas.concat([numeric_features, encoded_data],axis = 1)
explanatory_dfhw.head()


#now, let's find features with no variance 
no_variation = find_zero_var(explanatory_dfhw)
explanatory_dfhw.drop(no_variation['toDelete'], inplace = True)

# deleting perfect correlation
no_correlation = find_perfect_corr(explanatory_dfhw)
explanatory_dfhw.drop(no_correlation['toRemove'], 1, inplace = True)

# scaling data
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_dfhw)
explanatory_dfhw = pandas.DataFrame(scaler.transform(explanatory_dfhw), columns = explanatory_dfhw.columns)


#######
# PCA
######
from sklearn.decomposition import PCA

pca = PCA(n_components=6)
pca.fit(explanatory_dfhw)

# extracting the components
pca_dfhw = pandas.DataFrame(pca.transform(explanatory_dfhw))

## plotting the first to principal components
pca_dfhw.plot(x = 0, y= 1, kind = 'scatter')


# making a scree plot
variance_dfhw = pandas.DataFrame({'variance': pca.explained_variance_, 'principal component': pca_dfhw.columns.tolist()})
# adding one to principal components (since there is no 0th component)
variance_dfhw['principal component'] = variance_dfhw['principal component'] + 1
variance_dfhw.plot(x = 'principal component', y= 'variance')
## looks like variance stops getting explained after the first 
#principal components

pca_dfhw_small = pca_dfhw.ix[:,0:1]

## getting cross-val score of transformed data
rf = ensemble.RandomForestClassifier(n_estimators= 500)
roc_scores_rf_pca = cross_val_score(rf, pca_dfhw_small, response_series, cv=10, scoring='roc_auc')

print roc_scores_rf_pca.mean()
## crazy that with just one component, 84% accuracy is achieved, yet that is worse than we had originally

# Let's compare this to the original adata
roc_scores_rf = cross_val_score(rf, explanatory_dfhw, response_series, cv=10, scoring='roc_auc')
print roc_scores_rf.mean()
## 90% accuracy from randomforest versus 83% for PCA -- worse off with PCA again



###############################################################################
#Run a Boosting Tree on the components and see if in-sample accuracy beats a classifier with the raw data not trasnformed by PCA.
###############################################################################


boosting_treehw = ensemble.GradientBoostingClassifier()

roc_scores_gbmhw = cross_val_score(boosting_treehw, explanatory_dfhw, response_series, cv=10, scoring='roc_auc')
print roc_scores_gbmhw.mean()
#91% accuracy without PCA

roc_scores_gbmhw_pca = cross_val_score(boosting_treehw, pca_dfhw_small, response_series, cv=10, scoring='roc_auc')
print roc_scores_gbmhw_pca.mean()
#85% accuracy with PCA

###############################################################################
#Run a support vector machine on your data
###############################################################################


svm = SVC(kernel='poly')

roc_scores_svm = cross_val_score(svm, explanatory_dfhw, response_series, cv=10, scoring='roc_auc')
print roc_scores_svm.mean()
## 77% acccuracy, pretty terrible compared to other methods


# let's try with PCA
roc_scores_svm_pca = cross_val_score(svm, pca_dfhw_small, response_series, cv=10, scoring='roc_auc')
print roc_scores_svm_pca.mean()
## 52% acccuracy -- so PCA did worse AGAIN, and this is just as bad as it's gotten on any iteration


# let's do a grid search on the optimal kernel
param_grid = dict(kernel = ['linear','poly','rbf','sigmoid'])

svm_grid = GridSearchCV(svm, param_grid, cv=10, scoring='roc_auc')
svm_grid.fit(explanatory_dfhw, response_series)
best_estimator = svm_grid.best_estimator_
print best_estimator.kernel
## looks like linear won out in this instance
print svm_grid.best_score_
#Fortunately, accuracy went up when tuning parameters to 89%, likely as the data was very linear and
#not, as originally stated, polynomial

###############################################################################
#Like what you did in class 9, bring in data after 2000, and preform the same transformations on the data you did with your training data.
#Compare Random Forest, Boosting Tree, and SVM accuracy on the data after 2000.
###############################################################################



