# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 13:38:32 2015

@author: jkraunz

"""
import pandas
import sqlite3
from sklearn import preprocessing
import matplotlib as plt

from __future__ import division


conn= sqlite3.connect('/Users/jkraunz/Documents/SQLite/lahman2013.sqlite.crdownload')

query = '''
select s.yearID, sum(salary) as total_salaries, sum(R) as total_runs from salaries s
inner join Batting b on s.playerId = b.playerID 
where R is not null and s.yearID  > 1985
group by s.yearID
order by total_runs desc
'''


df = pandas.read_sql(query, conn)
conn.close()

pandas.set_option('display.max_columns', None)
df.head()

df = df.dropna()
df.head()

plt = df.plot(x = 'total_salaries', y = 'total_runs', kind = 'scatter')

for i, txt in enumerate(df.yearID):
    plt.annotate(txt, (df.total_salaries[i], df.total_runs[i]))
plt.show()

# Scales data
data = df[['total_salaries', 'total_runs']]
scaler = preprocessing.StandardScaler()
scaler.fit(data)
data = pandas.DataFrame(scaler.transform(data), columns = data.columns)

# Kmeans
from sklearn.cluster import KMeans
import matplotlib.pylab as plt

kmeans_est = KMeans(n_clusters=3)
kmeans_est.fit(data)

labels = kmeans_est.labels_

plt = df.plot(x = 'total_salaries', y = 'total_runs', s=60, c=labels, kind = 'scatter')
for i, txt in enumerate(df.yearID):
    plt.annotate(txt, (df.total_salaries[i], df.total_runs[i]))
plt.show()

kmeans_est = KMeans(n_clusters=4)
kmeans_est.fit(data)

labels = kmeans_est.labels_

plt = df.plot(x = 'total_salaries', y = 'total_runs', s=60, c=labels, kind = 'scatter')
for i, txt in enumerate(df.yearID):
    plt.annotate(txt, (df.total_salaries[i], df.total_runs[i]))
plt.show()

# DBSCAN

from sklearn.cluster import DBSCAN
import numpy as np

dbsc = DBSCAN().fit(np.array(data))
labels = dbsc.labels_

plt = df.plot(x = 'total_salaries', y = 'total_runs', s=60, c=labels, kind = 'scatter')
for i, txt in enumerate(df.yearID):
    plt.annotate(txt, (df.total_salaries[i], df.total_runs[i]))
plt.show()

# Dendrogram
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, fclusterdata

distanceMatrix = pdist(data)

dend = dendrogram(linkage(distanceMatrix, method='complete'), color_threshold=2, leaf_font_size=10, labels = df.yearID.tolist())
          
assignments = fcluster(linkage(distanceMatrix, method = 'complete'), 2, 'distance')

cluster_output = pandas.DataFrame({'team':df.yearID.tolist(), 'cluster':assignments})
cluster_output

plt.scatter(df.total_salaries, df.total_runs, s=60, c=cluster_output.cluster)

# Got the following code when I tried to improve the plot
# AttributeError: 'int' object has no attribute 'view'

#colors = cluster_output.cluster
#colors[colors == 1] = 'b'
#colors[colors == 2] = 'g'
#colors[colors == 3] = 'r'
#
#plt.scatter(df.total_salaries, df.total_runs, s=100, c=colors,  lw=0)

############################################################################

# Principal component analysis

import pandas
import sqlite3
import numpy
from sklearn.preprocessing import Imputer
from sklearn import ensemble
from sklearn.cross_validation import cross_val_score

conn = sqlite3.connect('/Users/jkraunz/Documents/SQLite/lahman2013.sqlite.crdownload')

sql= '''
Select i.*, count(yearID) as num_of_allstar_games
FROM
(Select f.*, birthCountry
FROM
(Select d.*, e.teamID
FROM
(Select c.*, sum(H) as total_post_hits, sum(HR) as total_post_HRs, sum(RBI) as total_post_RBIs
FROM
(Select a.*, sum(E) as total_errors
FROM
(SELECT m.*,
sum(SO) as total_SOs, avg(ERA) as avg_ERA, sum(W) as total_wins, sum(SV) as total_saves, count(YearID) as years_pitched
FROM
(select h.*, sum(RBI) as total_RBIs, sum(SB) as total_stolen_bases, sum(R) as total_runs, sum(H) as total_hits, count(yearID) as years_batted, sum(HR) as total_HRs, sum('2B') as total_2B, sum('3B') as total_3B
FROM 
(SELECT playerID, max(yearID) as final_year_voted, count(yearID) as years_voted, inducted
FROM HallofFame 
Where yearID < 2000
GROUP BY playerID) h
LEFT JOIN Batting b on h.playerID = b.playerID
GROUP BY h.playerID) m
LEFT JOIN Pitching p on m.playerID = p.playerID
group by m.playerID) a
LEFT JOIN Fielding f on a.playerID = f.playerID
GROUP BY a.playerID) c
Left Join BattingPost bp on c.playerID = bp.playerID
Group By c.playerID) d
Left Join dominant_team_per_player e on d.playerID = e.playerID
Group by d.playerID) f
Left Join Master g on f.playerID = g.playerID
Group by f.playerID) i
Left Join num_of_allstar j on i.playerID = j.playerID
Group by i.playerID
'''

df = pandas.read_sql(sql, conn)
conn.close()

df.head()

df.columns

# Functions
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
   all_columns = pandas.DataFrame( index = data_frame.index)
   for col in data_frame.columns:
       data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
       all_columns = pandas.concat([all_columns, data], axis=1)
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

## Manipulate data

df['inducted1'] = 0
df.inducted1[df.inducted == 'Y'] = 1

df['years_played'] = 0
df.years_played[df.years_pitched >= df.years_batted] = df.years_pitched
df.years_played[df.years_pitched < df.years_batted] = df.years_batted

df.drop(['playerID', 'inducted', 'years_pitched', 'years_batted', 'final_year_voted'],  1, inplace = True)

df.head()

# Set up explanatory and response features

explanatory_features = [col for col in df.columns if col not in ['inducted1']]
explanatory_df = df[explanatory_features]

explanatory_df.dropna(how = 'all', inplace = True)

explanatory_col_names = explanatory_df.columns

response_series = df.inducted1

response_series.dropna(how = 'all', inplace = True)

response_series.index[~response_series.index.isin(explanatory_df.index)]

# 1. Split data into categorical and numeric data

string_features = explanatory_df.ix[: , explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[: , explanatory_df.dtypes != 'object']

string_features.head()
numeric_features.head()

# 2. Fill numeric NaNs through imputation

from sklearn.preprocessing import Imputer

imputer_object = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

numeric_features.head()

numeric_features.describe()

# 3. Fill categorical NaNs with ‘Nothing’

len(string_features.teamID)
string_features.teamID.value_counts(normalize = False)

string_features = string_features.fillna('Nothing')

# 4. Detect low-frequency levels in categorical features and bin them under ‘other’

string_features.teamID.value_counts(normalize = True)
string_features.birthCountry.value_counts(normalize = True)

cleanup_data(string_features)
    
string_features.teamID.value_counts(normalize = True)
string_features.birthCountry.value_counts(normalize = True)

len(string_features.teamID)

# defined my column names so that they can be matched to the testing data
string_features_cat = 	{}
for col in string_features.columns:
	string_features_cat[col] = string_features[col].unique()

# 5. Encode each categorical variable into a sequence of binary variables.

string_features = get_binary_values(string_features)


# 6. Merge your encoded categorical data with your numeric data

explanatory_df = pandas.concat([numeric_features, string_features], axis = 1)
explanatory_df.head()

explanatory_df.describe()

# 7. Remove features with no variation
   
find_zero_var(explanatory_df)

# No features had zero variance

# 8. Remove perfectly correlated features
   
find_perfect_corr(explanatory_df)

# total_2B and total_3B are correlated so will drop total_2B

explanatory_df.drop('total_2B', 1, inplace=True)
explanatory_df.head()

# 9. Scale your data with zero mean and unit variance

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)

explanatory_df.describe()

#PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=6)
pca.fit(explanatory_df)

# extracting the components
pca_df = pandas.DataFrame(pca.transform(explanatory_df))

## plotting the first to principal components
pca_df.plot(x = 0, y= 1, kind = 'scatter')


# making a scree plot
variance_df = pandas.DataFrame({'variance': pca.explained_variance_, 'principal component': pca_df.columns.tolist()})
# adding one to principal components (since there is no 0th component)
variance_df['principal component'] = variance_df['principal component'] + 1
variance_df.plot(x = 'principal component', y= 'variance')
# variance stops being explaned after first five components

pca_df_small = pca_df.ix[:,0:4]

boosting_tree = ensemble.GradientBoostingClassifier()

roc_scores_gbm_pca = cross_val_score(boosting_tree, pca_df_small, response_series, cv=10, scoring = 'roc_auc', n_jobs = -1)

print roc_scores_gbm_pca.mean()
# 89.7% accuracy

roc_scores_gbm = cross_val_score(boosting_tree, explanatory_df, response_series, cv=10, scoring = 'roc_auc', n_jobs = -1)

print roc_scores_gbm.mean()
# 93.6% accuracy

# Support vector machines

from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score

svm = SVC(kernel='poly')

roc_scores_svm = cross_val_score(svm, pca_df_small, response_series, cv=10, scoring = 'roc_auc', n_jobs= -1)
print roc_scores_svm.mean()
# .8454

roc_scores_svm = cross_val_score(svm, explanatory_df, response_series, cv=10, scoring = 'roc_auc', n_jobs= -1)
print roc_scores_svm.mean()
# .8551

# Optimizing SVM
from sklearn.grid_search import  GridSearchCV
from numpy import arange

svm_grid_params = dict(kernel = ['linear','poly','rbf','sigmoid'])

svm_grid = GridSearchCV(svm, svm_grid_params, cv=10, scoring='roc_auc', n_jobs = -1)
svm_grid.fit(explanatory_df, response_series)
svm_estimator = svm_grid.best_estimator_
print svm_estimator.kernel, svm_grid.best_score_
# rbf
# .9028

# Compare to Random Forest and Boosting Tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


rf = RandomForestClassifier()
trees_range = range(10, 600, 10)
rf_grid_params = dict(n_estimators = trees_range)
rf_grid = GridSearchCV(rf, rf_grid_params, cv=10, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(explanatory_df,response_series)
rf_estimator = rf_grid.best_estimator_

rf_roc_scores = cross_val_score(rf_estimator, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print rf_roc_scores.mean()
# .9346

gbm = GradientBoostingClassifier()
learning_rate_range = arange(0.01, 0.4, 0.02)
subsampling_range = arange(0.25, 1, 0.25)
n_estimators_range = range(25, 100, 25)
gbm_grid_params = dict(learning_rate = learning_rate_range, n_estimators = n_estimators_range, subsample = subsampling_range)
gbm_grid = GridSearchCV(gbm, gbm_grid_params, cv=10, scoring='roc_auc', n_jobs = -1)
gbm_grid.fit(explanatory_df, response_series)
gbm_estimator = gbm_grid.best_estimator_

gbm_roc_scores = cross_val_score(gbm_estimator, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print gbm_roc_scores.mean()
# .9400

# Testing the models on post 2000 data

conn = sqlite3.connect('/Users/jkraunz/Documents/SQLite/lahman2013.sqlite.crdownload')

sql= '''
Select i.*, count(yearID) as num_of_allstar_games
FROM
(Select f.*, birthCountry
FROM
(Select d.*, e.teamID
FROM
(Select c.*, sum(H) as total_post_hits, sum(HR) as total_post_HRs, sum(RBI) as total_post_RBIs
FROM
(Select a.*, sum(E) as total_errors
FROM
(SELECT m.*,
sum(SO) as total_SOs, avg(ERA) as avg_ERA, sum(W) as total_wins, sum(SV) as total_saves, count(YearID) as years_pitched
FROM
(select h.*, sum(RBI) as total_RBIs, sum(SB) as total_stolen_bases, sum(R) as total_runs, sum(H) as total_hits, count(yearID) as years_batted, sum(HR) as total_HRs, sum('2B') as total_2B, sum('3B') as total_3B
FROM 
(SELECT playerID, max(yearID) as final_year_voted, count(yearID) as years_voted, inducted
FROM HallofFame 
Where yearID >= 2000
GROUP BY playerID) h
LEFT JOIN Batting b on h.playerID = b.playerID
GROUP BY h.playerID) m
LEFT JOIN Pitching p on m.playerID = p.playerID
group by m.playerID) a
LEFT JOIN Fielding f on a.playerID = f.playerID
GROUP BY a.playerID) c
Left Join BattingPost bp on c.playerID = bp.playerID
Group By c.playerID) d
Left Join dominant_team_per_player e on d.playerID = e.playerID
Group by d.playerID) f
Left Join Master g on f.playerID = g.playerID
Group by f.playerID) i
Left Join num_of_allstar j on i.playerID = j.playerID
Group by i.playerID
'''

post2000_df = pandas.read_sql(sql, conn)
conn.close()

post2000_df.head()
post2000_df.describe()
post2000_df.columns

# Manipulated data
post2000_df['inducted1'] = 0
post2000_df.inducted1[post2000_df.inducted == 'Y'] = 1

post2000_df['years_played'] = 0
post2000_df.years_played[post2000_df.years_pitched >= post2000_df.years_batted] = post2000_df.years_pitched
post2000_df.years_played[post2000_df.years_pitched < post2000_df.years_batted] = post2000_df.years_batted

post2000_df.drop(['playerID', 'inducted', 'years_pitched', 'years_batted', 'final_year_voted'],  1, inplace = True)

post2000_df.head()

# Defined explanatory and response variables
post2000_explanatory_features = [col for col in post2000_df.columns if col not in ['inducted1']]
post2000_explanatory_df = post2000_df[post2000_explanatory_features]

post2000_explanatory_df.dropna(how = 'all', inplace = True)

post2000_explanatory_col_names = post2000_explanatory_df.columns

post2000_response_series = post2000_df.inducted1

post2000_response_series.dropna(how = 'all', inplace = True)

post2000_response_series.index[~post2000_response_series.index.isin(post2000_explanatory_df.index)]

post2000_response_series.describe()

# 12. Reclean, encode, and scale model


#       1. Split data into categorical and numerical data

post2000_string_features = post2000_explanatory_df.ix[: , post2000_explanatory_df.dtypes == 'object']
post2000_numeric_features = post2000_explanatory_df.ix[: , post2000_explanatory_df.dtypes != 'object']

post2000_string_features.describe()
post2000_numeric_features.head()


#       2. Fill numeric NaNs through imputation (from pre-2000 fit)

post2000_numeric_features = pandas.DataFrame(imputer_object.transform(post2000_numeric_features), columns = post2000_numeric_features.columns)

post2000_numeric_features.head()

#       3. Fill categorical NaNs with ‘Nothing’

post2000_string_features = post2000_string_features.fillna('Nothing')

#       4. Matches categorical data to pre-2000 data

# If there is a value that is not in the training data set, replaces value with "Other"
for col in post2000_string_features:
    post2000_string_features[col].ix[~post2000_string_features[col].isin(string_features_cat[col])] = "Other"

#       5. Encode each categorical variable into a sequence of binary variables.

post2000_string_features = get_binary_values(post2000_string_features)

# must make sure that there are dummy variables for variables in the training data that is not in the testing data already

for col in string_features:
	if col not in post2000_string_features:
		post2000_string_features[col] = 0
 
# Make sure that the string data is sorted the same as the training data 
post2000_string_features = post2000_string_features[string_features.columns]

#       6. Merge encoded categorical data with numeric data

post2000_explanatory_df = pandas.concat([post2000_numeric_features, post2000_string_features], axis = 1)
post2000_explanatory_df.head()

#       7. Remove features with no variation

#           There were none

#       8. Remove features with perfect correlation

# Found that total_2B was perfectly correlated in the training data so dropped.
post2000_explanatory_df.drop(['total_2B'], 1, inplace = True)

#       9. Scale data with zero mean and unit variance (from pre-2000 fit)

post2000_explanatory_df = pandas.DataFrame(scaler.transform(post2000_explanatory_df), columns = post2000_explanatory_df.columns, index = post2000_explanatory_df.index)

post2000_explanatory_df.describe()

# Predicting Induction using SVM, Boosting Tree, and Random Forest

svm_pred_post2000_inductions = svm_estimator.predict(post2000_explanatory_df)
rf_pred_post2000_inductions = rf_estimator.predict(post2000_explanatory_df)
gbm_pred_post2000_inductions = gbm_estimator.predict(post2000_explanatory_df)

svm_number_correct = len(post2000_response_series[post2000_response_series == svm_pred_post2000_inductions])
total = len(post2000_response_series)
svm_accuracy = svm_number_correct / total

print svm_accuracy
# .8363

rf_number_correct = len(post2000_response_series[post2000_response_series == rf_pred_post2000_inductions])
total = len(post2000_response_series)
rf_accuracy = rf_number_correct / total

print rf_accuracy
# .8576

gbm_number_correct = len(post2000_response_series[post2000_response_series == gbm_pred_post2000_inductions])
total = len(post2000_response_series)
gbm_accuracy = gbm_number_correct / total

print gbm_accuracy
# .8612

svm_cm = pandas.crosstab(post2000_response_series, svm_pred_post2000_inductions, rownames=['True Label'], colnames=['Predicted Label'], margins=True)

print svm_cm
'''
Predicted Label    0   1  All
True Label                   
0                202  17  219
1                 29  33   62
All              231  50  281


'''

rf_cm = pandas.crosstab(post2000_response_series, rf_pred_post2000_inductions, rownames=['True Label'], colnames=['Predicted Label'], margins=True)

print rf_cm
'''
Predicted Label    0   1  All
True Label                   
0                200  19  219
1                 21  41   62
All              221  60  281
'''

gbm_cm = pandas.crosstab(post2000_response_series, gbm_pred_post2000_inductions, rownames=['True Label'], colnames=['Predicted Label'], margins=True)

print gbm_cm
'''
Predicted Label    0   1  All
True Label                   
0                200  19  219
1                 20  42   62
All              220  61  281
'''