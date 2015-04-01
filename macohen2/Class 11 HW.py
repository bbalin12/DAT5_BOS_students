# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:22:21 2015

@author: MatthewCohen
"""

import pandas
import sqlite3
from sklearn import preprocessing
import numpy as np

pandas.set_option('display.max_columns', None)

query = """
select t.teamID, avg(W) as average_wins, avg(t.HR) as average_home_runs, avg(t.R) as average_runs, sum(t.R) as total_runs, avg(t.HR)/avg(t.G) as average_game_homeruns from teams t
 inner join Batting b on t.teamId = b.teamID 
 where W is not null and t.yearID > 1995
group by t.teamID
order by average_wins desc;
"""

conn = sqlite3.connect('/Users/MatthewCohen/Documents/SQLite/lahman2013.sqlite')
df = pandas.read_sql(query, conn)
conn.close()

data = df[['average_game_homeruns', 'average_wins']]
scaler = preprocessing.StandardScaler()
scaler.fit(data)
data = pandas.DataFrame(scaler.transform(data), columns = data.columns)

plt = df.plot(x='average_game_homeruns', y='average_wins', kind='scatter')

for i, txt in enumerate(df.teamID):
    plt.annotate(txt, (df.average_game_homeruns[i],df.average_wins[i]))
plt.show()


from sklearn.cluster import KMeans
import matplotlib.pylab as plt
kmeans_est = KMeans(n_clusters=3)

kmeans_est.fit(data)
labels = kmeans_est.labels_

plt.scatter(df.average_game_homeruns, df.average_wins, s=60, c=labels)



from sklearn.cluster import DBSCAN

dbsc = DBSCAN().fit(np.array(data))
labels = dbsc.labels_

plt.scatter(df.average_game_homeruns, df.average_wins, s=60, c=labels)



from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, fclusterdata

distanceMatrix = pdist(data)

dend = dendrogram(linkage(distanceMatrix, method='complete'), 
           color_threshold=1, 
           leaf_font_size=10,
           labels = df.teamID.tolist())

dend = dendrogram(linkage(distanceMatrix, method='complete'), 
           color_threshold=4, 
           leaf_font_size=10,
           labels = df.teamID.tolist())

assignments = fcluster(linkage(distanceMatrix, method='complete'),4,'distance')

cluster_output = pandas.DataFrame({'team':df.teamID.tolist() , 'cluster':assignments})

print cluster_output


plt.scatter(df.average_game_homeruns, df.average_wins, s=60, c=cluster_output.cluster)

colors = cluster_output.cluster
colors[colors == 1] = 'b'
colors[colors == 2] = 'g'
colors[colors == 3] = 'r'
plt.scatter(df.average_game_homeruns, df.average_wins, s=100, c=colors,  lw=0)






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


## using the new table as part of the monster query from last class
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
where batting.playerID is not null
"""

conn = sqlite3.connect('/Users/MatthewCohen/Documents/SQLite/lahman2013.sqlite')
df = pandas.read_sql(monster_query, conn)
conn.close()

## getting an intial view of the data for validation
df.head(10)
df.columns


df.drop('playerID',  1, inplace = True)

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

no_variation = find_zero_var(explanatory_df)
explanatory_df.drop(no_variation['toDelete'], inplace = True)

# deleting perfect correlation
no_correlation = find_perfect_corr(explanatory_df)
explanatory_df.drop(no_correlation['toRemove'], 1, inplace = True)

# scaling data
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)


from sklearn.decomposition import PCA
pca = PCA(n_components=6)
pca.fit(explanatory_df)

pca_df = pandas.DataFrame(pca.transform(explanatory_df))
pca_df.plot(x = 0, y= 1, kind = 'scatter')

variance_df = pandas.DataFrame({'variance': pca.explained_variance_, 'principal component': pca_df.columns.tolist()})
variance_df['principal component'] = variance_df['principal component'] + 1
variance_df.plot(x = 'principal component', y= 'variance')

pca_df_small = pca_df.ix[:,0:1]
rf = ensemble.RandomForestClassifier(n_estimators= 500)
roc_scores_rf_pca = cross_val_score(rf, pca_df_small, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

print roc_scores_rf_pca.mean()

roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_rf.mean()


from sklearn.svm import SVC

svm = SVC(kernel='poly')
roc_scores_svm = cross_val_score(svm, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_svm.mean()


roc_scores_svm_pca = cross_val_score(svm, pca_df_small, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_svm_pca.mean()


param_grid = dict(kernel = ['linear','poly','rbf','sigmoid'])
svm_grid = GridSearchCV(svm, param_grid, cv=10, scoring='roc_auc', n_jobs = -1)
svm_grid.fit(explanatory_df, response_series)
best_estimator = svm_grid.best_estimator_
print best_estimator.kernel
print svm_grid.best_score_





### Bringing in New Data

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
where batting.playerID is not null
"""

conn = sqlite3.connect('/Users/MatthewCohen/Documents/SQLite/lahman2013.sqlite')
df2 = pandas.read_sql(monster_query2, conn)
conn.close()

## getting an intial view of the data for validation
df2.head(10)
df2.columns


df2.drop('playerID',  1, inplace = True)

#############
## repeating the same preprocessing from the previous lesson
############

## splitting out the explanatory features 
explanatory_features = [col for col in df2.columns if col not in ['nameGiven', 'inducted']]
explanatory_df2 = df2[explanatory_features]

# dropping rows with no data.
explanatory_df2.dropna(how='all', inplace = True) 

# extracting column names 
explanatory_colnames = explanatory_df2.columns

## doing the same for response
response_series = df2.inducted
response_series.dropna(how='all', inplace = True) 

## seeing which explanatory feature rows got removed.  Looks like none.
response_series.index[~response_series.index.isin(explanatory_df2.index)]

### now, let's seperate the numeric explanatory data from the string data
string_features = explanatory_df2.ix[:, explanatory_df2.dtypes == 'object']
numeric_features = explanatory_df2.ix[:, explanatory_df2.dtypes != 'object']


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
explanatory_df2 = pandas.concat([numeric_features, encoded_data],axis = 1)
explanatory_df2.head()

no_variation = find_zero_var(explanatory_df2)
explanatory_df2.drop(no_variation['toDelete'], inplace = True)

# deleting perfect correlation
no_correlation = find_perfect_corr(explanatory_df2)
explanatory_df2.drop(no_correlation['toRemove'], 1, inplace = True)

# scaling data
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df2)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df2), columns = explanatory_df2.columns)


from sklearn.decomposition import PCA
pca = PCA(n_components=6)
pca.fit(explanatory_df2)

pca_df2 = pandas.DataFrame(pca.transform(explanatory_df2))
pca_df2.plot(x = 0, y= 1, kind = 'scatter')

variance_df2 = pandas.DataFrame({'variance': pca.explained_variance_, 'principal component': pca_df.columns.tolist()})
variance_df2['principal component'] = variance_df2['principal component'] + 1
variance_df2.plot(x = 'principal component', y= 'variance')

pca_df2_small = pca_df2.ix[:,0:2]
rf = ensemble.RandomForestClassifier(n_estimators= 500)
roc_scores_rf_pca = cross_val_score(rf, pca_df2_small, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

print roc_scores_rf_pca.mean()

roc_scores_rf = cross_val_score(rf, explanatory_df2, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_rf.mean()


from sklearn.svm import SVC

svm = SVC(kernel='poly')
roc_scores_svm = cross_val_score(svm, explanatory_df2, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_svm.mean()


roc_scores_svm_pca = cross_val_score(svm, pca_df2_small, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_svm_pca.mean()


param_grid = dict(kernel = ['linear','poly','rbf','sigmoid'])
svm_grid = GridSearchCV(svm, param_grid, cv=10, scoring='roc_auc', n_jobs = -1)
svm_grid.fit(explanatory_df2, response_series)
best_estimator = svm_grid.best_estimator_
print best_estimator.kernel
print svm_grid.best_score_


















