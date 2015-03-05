'''
jonblum
2015-02-26
datbos05
class 11 hw
'''


# division
from __future__ import division

# basics
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# i/o
import sqlite3

# models
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# tools
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, fclusterdata
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import  GridSearchCV
from numpy import arange
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score



# Hits v. Salary for 2013

query='''
SELECT s.playerID, s.yearID, m.nameFirst || " " ||  m.nameLast as name, SUM(b.H) as hits, SUM(s.salary) as salary
FROM Salaries s
INNER JOIN Batting b
ON b.playerID = s.playerID AND b.yearID = s.yearID AND b.teamID = s.teamID
LEFT JOIN Master m
ON m.playerID = s.playerID
WHERE s.yearID = 2013
GROUP BY s.playerID, s.yearID
'''



conn = sqlite3.connect('/Users/jon/Documents/Code/datbos05/data/lahman2013.sqlite')
df = pd.read_sql(query, conn)
conn.close()

df.sort(columns='salary')

# if not in batting table, 0 hits
df.fillna(0,inplace=True)
# scaling data
data = df[['hits', 'salary']]
scaler = preprocessing.StandardScaler()
scaler.fit(data)
data = pd.DataFrame(scaler.transform(data), columns = data.columns)

# plot the scaled data
scatter = df.plot(x='hits', y='salary', kind='scatter', alpha=0.2)

# name the extremes.
for i, txt in enumerate(df.name):
	if (df.hits[i] > 175 and df.salary[i] > 750000) or df.salary[i] > 20000000:
		scatter.annotate(txt, (df.hits[i],df.salary[i]))

#########
# K-MEANS
#########


kmeans_est = KMeans(n_clusters=5)

kmeans_est.fit(data)
labels = kmeans_est.labels_

plt.scatter(df.hits, df.salary, alpha=0.3, c=labels)
# Pretty decent except it lumps together modestly-paid pitchers and low-paid low-hit batters. (both near 0,0)

########
# DBSCAN
########


# getting around a bug that doesn't let you you fit to a pandas dataframe. by coercing it  to a numpy array
dbsc = DBSCAN().fit(np.array(data))
labels = dbsc.labels_

plt.scatter(df.hits, df.salary, alpha=0.3, c=labels)
# Because DBSCAN is all about density and separation, it separates only
# A) Highly-paid pitchers
# B) Highest-salary outlier batters
# C) Everyone else


# For this data, KMeans works better, because even the dense data set can be split up into a requested n categories.

### HIERARCHICAL CLUSTERING / DENDROGRAM


############
# DENDROGRAM
############

distance_matrix = pdist(data)
dend = dendrogram(linkage(distance_matrix, method='complete'),color_threshold=2, leaf_font_size=2, labels=df.name.tolist())


############################################################


# Data from HW 9/10


############################
# Helper methods & classes #
############################

def bin_categorical(df, cutoffPercent = .01):
    """Lumps categorcal features with frequency < cutoffPercent into 'Other'."""
    for col in df:
        sizes = df[col].value_counts(normalize = True)
        # get the names of the levels that make up less than 1% of the dataset
        values_to_delete = sizes[sizes<cutoffPercent].index
        df[col].ix[df[col].isin(values_to_delete)] = "Other"
    return df

def get_binary_values(df):
    """Binary encoding of categorical features using Pandas' get_dummies."""
    all_columns = pd.DataFrame(index = df.index)
    for col in df.columns:
        data = pd.get_dummies(df[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pd.concat([all_columns, data], axis=1)
    return all_columns

def find_zero_var(df):
    """Finds columns in the dataframe with zero variance -- ie those
        with the same value in every observation."""
    to_keep = []
    to_delete = []
    for col in df:
        if len(df[col].value_counts()) > 1:
            to_keep.append(col)
        else:
            to_delete.append(col)
        ##
    return {'to_keep':to_keep, 'to_delete':to_delete}

def find_perfect_corr(df):
    """Finds columns that are eother positively or negatively perfectly correlated (with correlations of +1 or -1), and creates a dict
        that includes which columns to drop so that each remaining column is independent."""
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

class TreeClassifierWithCoef(DecisionTreeClassifier):
    """Tree Classifer with coeff_ (derived from feature_importances_) for use with RFECV."""
    def fit(self, *args, **kwargs):
        super(DecisionTreeClassifier, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


#########################


#############################
# DATA / PREP FROM HW9/HW10 #
#############################

# same data as HW09/10
# grab familiar features, joined with primary team from class and additional categorical feature 'school'
conn = sqlite3.connect('/Users/jon/Documents/code/datbos05/data/lahman2013.sqlite')
sql = '''
SELECT hof.playerID, dtpp.teamID, sp.schoolID, b.totalCareerHits, b.careerBattingAvg, p.avgCareerERA, f.careerFieldingPercentage, MAX(hof.inducted) AS inducted
FROM HallOfFame hof
LEFT JOIN
        (SELECT playerID, SUM(H) as totalCareerHits, (SUM(H)*1.0) / SUM(AB) as careerBattingAvg
        FROM Batting
        GROUP BY playerID) b
ON b.playerID = hof.playerID
LEFT JOIN
        (SELECT playerID, AVG(ERA) as avgCareerERA
        FROM Pitching
        GROUP BY playerID) p
ON  p.playerID = hof.playerID
LEFT JOIN
        (SELECT playerID, 1.0 * (SUM(PO) + SUM(A)) / (SUM(PO) + SUM(A) + SUM(E)) as careerFieldingPercentage
        FROM Fielding
        GROUP BY playerID) f
ON f.playerID = hof.playerID
LEFT JOIN dominant_team_per_player dtpp
ON dtpp.playerID = hof.playerID
LEFT JOIN SchoolsPlayers sp
ON sp.playerID = hof.playerID
WHERE hof.yearID < 2000 AND hof.category = 'Player'
GROUP BY hof.playerID;
'''

df = pd.read_sql(sql,conn)
conn.close()


# Same cleanup / prep as HW09/HW10


# Y->1 N->0
df['inducted_boolean'] = 0
df.inducted_boolean[df.inducted == 'Y'] = 1
df.drop('inducted',1,inplace=True)

explanatory_features = [col for col in df.columns if col not in ['playerID', 'inducted_boolean']]
explanatory_df = df[explanatory_features]

# drop rows with no data at all (older players from other leauges)
explanatory_df.dropna(how='all', inplace = True)

# doing the same for response
response_series = df.inducted_boolean
response_series.dropna(how='all', inplace = True)

# dropping from response anything that was dropped from explanatory and vice versa
response_series.drop(response_series.index[~response_series.index.isin(explanatory_df.index)],0,inplace=True)
explanatory_df.drop(explanatory_df.index[~explanatory_df.index.isin(response_series.index)],0,inplace=True)

len(response_series) == len(explanatory_df)
# True

# split features
string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']

print string_features.columns
# Index([u'teamID', u'schoolID'], dtype='object')
print numeric_features.columns
# Index([u'totalCareerHits', u'careerBattingAvg', u'avgCareerERA', u'careerFieldingPercentage'], dtype='object')

# impute and put back into a dataframe with the columns and indices from before
imputer = Imputer(missing_values='NaN', strategy='median',axis=0)
imputer.fit(numeric_features)
numeric_features = pd.DataFrame(imputer.transform(numeric_features), columns=numeric_features.columns, index=numeric_features.index)

# fill categorical NaNs
string_features = string_features.fillna('Nothing')

# detect low-frequency features and bin as other
# use 0.5% to keep at least a few schools as separate bins
string_features = bin_categorical(string_features,cutoffPercent=0.005)

string_features.columns[0]
string_features.schoolID.unique()

# grab categories for use with the out-of-sample data
orig_categories = 	{}
for col in string_features.columns:
	orig_categories[col] = string_features[col].unique()

# encode categorical features
encoded_string_features  =  get_binary_values(string_features)

# merge features
explanatory_df = pd.concat([numeric_features, encoded_string_features], axis=1)

# remove features with no variation
keep_delete = find_zero_var(explanatory_df)
print keep_delete['to_delete']
# []   (all features have some variation)

# remove features with perfect correlation
corr = find_perfect_corr(explanatory_df)
print corr['corrGroupings'], corr['toRemove']
# [] [] (no perfectly-correlated features)

# scale features
scaler = StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pd.DataFrame(scaler.transform(explanatory_df),columns=explanatory_df.columns,index=explanatory_df.index)

explanatory_df.totalCareerHits.describe()

#######
# PCA #
#######


##########
## PCA
##########

pca = PCA(n_components=10)
pca.fit(explanatory_df)

# extracting the components
pca_df = pd.DataFrame(pca.transform(explanatory_df))


# scree plot
variance_df = pd.DataFrame({'variance': pca.explained_variance_, 'principal component': pca_df.columns.tolist()})


# adding one to principal components (since there is no 0th component)
variance_df['principal component'] = variance_df['principal component'] + 1
variance_df.plot(x = 'principal component', y= 'variance')

# Most variance in first 5 components


pca_df_small = pca_df.ix[:,0:4]


## getting cross-val score of transformed data
gbm = GradientBoostingClassifier()
roc_scores_gbm_pca = cross_val_score(gbm, pca_df_small, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_gbm_pca.mean()
# 0.765855995296


# Let's compare this to the original data
roc_scores_gbm = cross_val_score(gbm, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_gbm.mean()
## 0.857441955175

# So PCA was caused information loss in this case.


#######
# SVM #
#######


# first, running quadratic kernel without PCA

svm = SVC(kernel='poly')

roc_scores_svm = cross_val_score(svm, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

print roc_scores_svm.mean()
# 0.679078009316

# Not great


# Find optimal kernel

svm_grid_params = dict(kernel = ['linear','poly','rbf','sigmoid'])

svm_grid = GridSearchCV(svm, svm_grid_params, cv=10, scoring='roc_auc', n_jobs = -1)
svm_grid.fit(explanatory_df, response_series)
svm_estimator = svm_grid.best_estimator_
print svm_estimator.kernel, svm_grid.best_score_
# kernel: rbf
# 0.724776615437

# Slightly better with rbf kernel


# Bring back our optimal RF and GBMs:

rf = RandomForestClassifier()
trees_range = range(10, 600, 10)
rf_grid_params = dict(n_estimators = trees_range)
rf_grid = GridSearchCV(rf, rf_grid_params, cv=10, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(explanatory_df,response_series)
rf_estimator = rf_grid.best_estimator_


gbm = GradientBoostingClassifier()
learning_rate_range = arange(0.01, 0.4, 0.02)
subsampling_range = arange(0.25, 1, 0.25)
n_estimators_range = range(25, 100, 25)
gbm_grid_params = dict(learning_rate = learning_rate_range, n_estimators = n_estimators_range, subsample = subsampling_range)
gbm_grid = GridSearchCV(gbm, gbm_grid_params, cv=10, scoring='roc_auc', n_jobs = -1)
gbm_grid.fit(explanatory_df, response_series)
gbm_estimator = gbm_grid.best_estimator_


rf_roc_scores = cross_val_score(rf_estimator, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print rf_roc_scores.mean()
# 0.870790749206


gbm_roc_scores = cross_val_score(gbm_estimator, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print gbm_roc_scores.mean()
# 0.855835483989

svm_roc_scores = cross_val_score(svm_estimator, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print svm_roc_scores.mean()
# 0.724999828405



# Bring in post-2000 data

conn = sqlite3.connect('/Users/jon/Documents/code/datbos05/data/lahman2013.sqlite')
sql = '''
SELECT hof.playerID, dtpp.teamID, sp.schoolID, b.totalCareerHits, b.careerBattingAvg, p.avgCareerERA, f.careerFieldingPercentage, MAX(hof.inducted) AS inducted
FROM HallOfFame hof
LEFT JOIN
        (SELECT playerID, SUM(H) as totalCareerHits, (SUM(H)*1.0) / SUM(AB) as careerBattingAvg
        FROM Batting
        GROUP BY playerID) b
ON b.playerID = hof.playerID
LEFT JOIN
        (SELECT playerID, AVG(ERA) as avgCareerERA
        FROM Pitching
        GROUP BY playerID) p
ON  p.playerID = hof.playerID
LEFT JOIN
        (SELECT playerID, 1.0 * (SUM(PO) + SUM(A)) / (SUM(PO) + SUM(A) + SUM(E)) as careerFieldingPercentage
        FROM Fielding
        GROUP BY playerID) f
ON f.playerID = hof.playerID
LEFT JOIN dominant_team_per_player dtpp
ON dtpp.playerID = hof.playerID
LEFT JOIN SchoolsPlayers sp
ON sp.playerID = hof.playerID
WHERE hof.yearID > 2000 AND hof.category = 'Player'
GROUP BY hof.playerID;
'''

df_2k = pd.read_sql(sql,conn)
conn.close()


# Same cleanup / prep as HW09/HW10


# Y->1 N->0
df_2k['inducted_boolean'] = 0
df_2k.inducted_boolean[df_2k.inducted == 'Y'] = 1
df_2k.drop('inducted',1,inplace=True)

explanatory_features_2k = [col for col in df_2k.columns if col not in ['playerID', 'inducted_boolean']]
explanatory_df_2k = df_2k[explanatory_features]

# drop rows with no data at all (older players from other leauges)
explanatory_df_2k.dropna(how='all', inplace = True)

# doing the same for response
response_series_2k = df_2k.inducted_boolean
response_series_2k.dropna(how='all', inplace = True)

# dropping from response anything that was dropped from explanatory and vice versa
response_series_2k.drop(response_series_2k.index[~response_series_2k.index.isin(explanatory_df_2k.index)],0,inplace=True)
explanatory_df_2k.drop(explanatory_df_2k.index[~explanatory_df_2k.index.isin(response_series_2k.index)],0,inplace=True)

len(response_series_2k) == len(explanatory_df_2k)
# True

# split features
string_features_2k = explanatory_df_2k.ix[:, explanatory_df_2k.dtypes == 'object']
numeric_features_2k = explanatory_df_2k.ix[:, explanatory_df_2k.dtypes != 'object']


# impute numeric from original imputer
numeric_features_2k = pd.DataFrame(imputer.transform(numeric_features_2k), columns=numeric_features_2k.columns, index=numeric_features_2k.index)


# bin and encode
string_features_2k = string_features_2k.fillna('Nothing')

# for each string feature, if there is a value that is not in the original dataset, make it 'other'.
for col in string_features_2k:
	string_features_2k[col].ix[~string_features_2k[col].isin(orig_categories[col])] = "Other"

encoded_string_features_2k  =  get_binary_values(string_features_2k)

# post-encoding, add any dummy features that were in the original
for col in encoded_string_features:
	if col not in encoded_string_features_2k:
		encoded_string_features_2k[col] = 0


# reorder columns to match original

encoded_string_features_2k = encoded_string_features_2k[encoded_string_features.columns]


# combine
explanatory_df_2k = pd.concat([numeric_features_2k, encoded_string_features_2k], axis=1)


# scale using original scaler
explanatory_df_2k = pd.DataFrame(scaler.transform(explanatory_df_2k),columns=explanatory_df_2k.columns,index=explanatory_df_2k.index)


# Run models against post-2000 data


rf_predicted_post_2000_inductions = rf_estimator.predict(explanatory_df_2k)
gbm_predicted_post_2000_inductions = gbm_estimator.predict(explanatory_df_2k)
svm_predicted_post_2000_inductions = svm_estimator.predict(explanatory_df_2k)

total = len(response_series_2k)


rf_number_correct = len(response_series_2k[response_series_2k == rf_predicted_post_2000_inductions])
gbm_number_correct = len(response_series_2k[response_series_2k == gbm_predicted_post_2000_inductions])
svm_number_correct = len(response_series_2k[response_series_2k == svm_predicted_post_2000_inductions])


rf_accuracy = rf_number_correct / total
print rf_accuracy
# 0.887931034483


gbm_accuracy = gbm_number_correct / total
print gbm_accuracy
# 0.883620689655

svm_accuracy = svm_number_correct / total
print svm_accuracy
# 0.883620689655


rf_cm = pd.crosstab(response_series_2k, rf_predicted_post_2000_inductions, rownames=['True Label'], colnames=['Predicted Label'], margins=True)
print rf_cm
'''
Predicted Label    0   1  All
True Label
0                197   8  205
1                 18   9   27
All              215  17  232
'''

gbm_cm = pd.crosstab(response_series_2k, gbm_predicted_post_2000_inductions, rownames=['True Label'], colnames=['Predicted Label'], margins=True)
print gbm_cm
'''
Predicted Label    0   1  All
True Label
0                193  12  205
1                 15  12   27
All              208  24  232
'''


svm_cm = pd.crosstab(response_series_2k, svm_predicted_post_2000_inductions, rownames=['True Label'], colnames=['Predicted Label'], margins=True)
print svm_cm

'''
Predicted Label    0  All
True Label
0                205  205
1                 27   27
All              232  232
'''

# SVM just predicted no one gets inducted!   And in doing so, matched GBM's accuracy!

