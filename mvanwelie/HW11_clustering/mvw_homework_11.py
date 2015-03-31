# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 23:42:31 2015

@author: megan
"""

import pandas
import sqlite3
import numpy
import matplotlib.pylab as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import Imputer
from sklearn import ensemble
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import  GridSearchCV
from sklearn import tree
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, fclusterdata

################
# Homework 11 - Unsupervised learning
# Find some sort of attribute in the Baseball dataset that sits on a two-dimenstional plane and has discrete clusters.
# Perform K-Means and DBSCAN clustering.
# Determine which better represents your data and the intiution behind why the model was the best for your dataset.
# Plot a dendrogram of your dataset.
# Using the data from the last homework, perform principal component analysis on your data.
# Decide how many components to keep
# Run a Boosting Tree on the components and see if in-sample accuracy beats a classifier with the raw data not trasnformed by PCA.
# Run a support vector machine on your data
# Tune your SVM to optimize accuracy
# Like what you did in class 9, bring in data after 2000, and preform the same transformations on the data you did with your training data.
# Compare Random Forest, Boosting Tree, and SVM accuracy on the data after 2000.
# If you wish, see if using PCA improves the accuracy of any of your models.
################

################
# Define helper methods for cleaning and manipulating
# the data into a useful set
################

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
    
# create new class with a .coef_ attribute.
class TreeClassifierWithCoef(tree.DecisionTreeClassifier):
    def fit(self, *args, **kwargs):
        super(tree.DecisionTreeClassifier, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

################
# Query database for training data
################

# A query to grab the total number of players that came from each state
# and the total numbers of players from each state that were inducted
# into the hall of fame
query = """
select master.playerID, master.birthState as birthState, count(master.playerID) as total_players, sum(hofi.inducted) as total_inducted from master
INNER JOIN
(
select hofi.playerID, hofi.inducted from
hall_of_fame_inductees as hofi
)
hofi on hofi.playerID = master.playerID
group by master.birthState
"""
conn = sqlite3.connect('/Users/megan/Documents/SQLite/lahman2013.sqlite')
df = pandas.read_sql(query, conn)
conn.close()

# Scale the data
data = df[['total_players', 'total_inducted']]
scaler = preprocessing.StandardScaler()
scaler.fit(data)
data = pandas.DataFrame(scaler.transform(data), columns = data.columns)


# Plot the data
plt = df.plot(x='total_players', y='total_inducted', kind='scatter')

# Annotate with state names
for i, txt in enumerate(df.birthState):
    plt.annotate(txt, (df.total_players[i],df.total_inducted[i]))
plt.show()

########
# K-Means Clustering
########
kmeans_est = KMeans(n_clusters=5)
kmeans_est.fit(data)
kmeans_labels = kmeans_est.labels_

# Plot the data to see the clusters
plt.scatter(df.total_players, df.total_inducted, s=60, c=kmeans_labels)
# Forced it to select 5 clusters based on viewing the data previously

########
# DBSCAN Clustering
########
dbsc = DBSCAN().fit(np.array(data))
dbsc_labels = dbsc.labels_

plt.scatter(df.total_players, df.total_inducted, s=60, c=dbsc_labels)
# Found 2 clusters

# Based on the above plots, I believe K-Means clustering better fits
# the data. Based on viewing the graphs there appears to be more than 2
# clusters, which K-Means is able to find by DBSCAN is not

########
# Dendrogram - Hirearchical clustering
########

distanceMatrix = pdist(data)

# adjust color_threshold to get the number of clusters you want.
dend = dendrogram(linkage(distanceMatrix, method='complete'), 
           color_threshold=1, 
           leaf_font_size=10,
           labels = df.birthState.tolist())
# The threshold set to 1 actually looks good as it gives us the 5 desired clusters

################
# Query databases for training data before the year 2000
################

# Create a table with a categorical feature that shows the dominant team played per player
con = sqlite3.connect('/Users/megan/Documents/SQLite/lahman2013.sqlite')
query = 'select playerID, teamID from Batting'
df = pandas.read_sql(query, con)
con.close()
majority_team_by_player = df.groupby(['playerID']).agg(lambda x:x.value_counts().idxmax())
conn = sqlite3.connect('/Users/megan/Documents/SQLite/lahman2013.sqlite')
majority_team_by_player.to_sql('dominant_team_per_player', conn, if_exists = 'replace')

# Create a table to indicate if a player was ever inducted into the hall of fame
cur = conn.cursor()    
table_creation_query = """
CREATE TABLE hall_of_fame_inductees_all as  
select playerID, inductedYear, category, case when average_inducted = 0 then 0 else 1 end as inducted from (
select playerID, max(yearid) as inductedYear, category, avg(case when inducted = 'Y' then 1 else 0 end ) as average_inducted from  HallOfFame hf
group by playerID
) bb;"""
cur.execute(table_creation_query)
cur.close()
conn.close()

# Query the DB for all predicive columns for inducted into the hall of fame
# Add birth state and dominant team player as the extra categorical features
conn = sqlite3.connect('/Users/megan/Documents/SQLite/lahman2013.sqlite')
sql = '''
SELECT hofi.playerID, hofi.inductedYear, hofi.inducted, dtpp.teamID,
batting.atBats, batting.hits,
pitching.wins, pitching.losses,
fielding.putOuts, fielding.assists, fielding.errors,
master.birthState
FROM hall_of_fame_inductees_all hofi
LEFT JOIN
(
SELECT dtpp.playerID, dtpp.teamID
from dominant_team_per_player dtpp
)
dtpp on dtpp.playerID = hofi.playerID
LEFT JOIN 
(
SELECT b.playerID, sum(b.AB) as atBats, sum(b.H) as hits
FROM Batting b
GROUP BY b.playerID
)
batting on batting.playerID = hofi.playerID
LEFT JOIN 
(
SELECT p.playerID, sum(p.W) as wins, sum(p.L) as losses
FROM Pitching p 
GROUP BY p.playerID
)
pitching on hofi.playerID = pitching.playerID
LEFT JOIN 
(
SELECT f.playerID, sum(f.PO) as putOuts, sum(f.A) as assists, sum(f.E) as errors
FROM Fielding f 
GROUP BY f.playerID
)
fielding on hofi.playerID = fielding.playerID
LEFT JOIN
(
SELECT m.playerID, m.birthState
FROM Master m
)
master on master.playerID = hofi.playerID
WHERE hofi.inductedYear < 2000 and hofi.category = 'Player';
'''
df = pandas.read_sql(sql, conn)
conn.close()

############
# Clean and prepare the data
############

# Add composite feature columns
df['batting_average'] = df.hits / df.atBats
df['winning_percentage'] = df.wins / (df.wins + df.losses)
df['fielding_percentage'] = (df.putOuts + df.assists) / (df.putOuts + df.assists + df.errors)

# dropping duplicate playerID columns
df.drop('playerID',  1, inplace = True)

## splitting out the explanatory features 
explanatory_features = [col for col in df.columns if col not in ['inducted']]
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

# Split the numeric explanatory data from the string data
# This is done to clean and manipulate the different types of data correctly
string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']

# Fill NaNs in our string features
string_features = string_features.fillna('Nothing')

# Bin and encode the categorical string features
string_features = cleanup_data(string_features)
string_features.teamID.value_counts(normalize = True)
encoded_data = get_binary_values(string_features)
encoded_data.head()

# Fill NaNs in our numeric features
from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

# Merge the numeric and encoded categorical features back together
explanatory_df = pandas.concat([numeric_features, encoded_data],axis = 1)
explanatory_df.head()

# Find and remove features with no variance 
toKeepDelete = find_zero_var(explanatory_df)
print toKeepDelete # nothing to delete

# Find and remove columns with perfect correlation
corrMatrixToRemove = find_perfect_corr(explanatory_df)
print corrMatrixToRemove['toRemove'] # nothing to remove
# explanatory_df.drop(corrMatrixToRemove['toRemove'], 1, inplace = True)

# Scale data
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)

############
# PCA - Principal Component Analysis
############
pca = PCA(n_components=6)
pca.fit(explanatory_df)

# Extract the components
pca_df = pandas.DataFrame(pca.transform(explanatory_df))

# Plot the first two principal components
pca_df.plot(x = 0, y= 1, kind = 'scatter')

# Make a scree plot
variance_df = pandas.DataFrame({'variance': pca.explained_variance_, 'principal component': pca_df.columns.tolist()})
# Adding one to principal components (since there is no 0th component)
variance_df['principal component'] = variance_df['principal component'] + 1
variance_df.plot(x = 'principal component', y= 'variance')
# Most of the variance stops getting explained after the first two components
# This is based on the elbow in the plot

pca_df_small = pca_df.ix[:,0:1]

# Run a Boosting Tree on the principal components
boosting_tree = ensemble.GradientBoostingClassifier()
roc_scores_gbm_pca = cross_val_score(boosting_tree, pca_df_small, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

# Accuracy is 73%
print roc_scores_gbm_pca.mean()

# Compare against Boosting Tree on original data
roc_scores_gbm = cross_val_score(boosting_tree, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

# Accuracy is 93%
print roc_scores_gbm.mean()

# Comparing the 73% accuracy with the principal components and the 93% with the original data, it seems that PCA actually created information loss

############
# SVM - Support Vector Machine
############
from sklearn.svm import SVC

# Create an SVM with a quadratic kernel
svm = SVC(kernel='poly')

# Run the SVM on the original data
roc_scores_svm = cross_val_score(svm, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

# Accuracy is 75%
print roc_scores_svm.mean()

# Run the SVM on the principal components
roc_scores_svm_pca = cross_val_score(svm, pca_df_small, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

# Accuracy is 64%
print roc_scores_svm_pca.mean()

# So, PCA did worse again when used with the SVM

# Optimize the SVM by running grid search on the optimal kernel
param_grid = dict(kernel = ['linear','poly','rbf','sigmoid'])

svm_grid = GridSearchCV(svm, param_grid, cv=10, scoring='roc_auc', n_jobs = -1)
svm_grid.fit(explanatory_df, response_series)
svm_best_estimator = svm_grid.best_estimator_

# The best kernel was linear
print svm_best_estimator.kernel
# Accuracy is 88%
print svm_grid.best_score_

# The SVM performed worse than the Boosting Tree we ran before

#############
# Import hold out data past the year 2000 to further test the effectiveness
# of the classifiers
#############
conn = sqlite3.connect('/Users/megan/Documents/SQLite/lahman2013.sqlite')

sql = '''
SELECT hofi.playerID, hofi.inductedYear, hofi.inducted, dtpp.teamID,
batting.atBats, batting.hits,
pitching.wins, pitching.losses,
fielding.putOuts, fielding.assists, fielding.errors,
master.birthState
FROM hall_of_fame_inductees_all hofi
LEFT JOIN
(
SELECT dtpp.playerID, dtpp.teamID
from dominant_team_per_player dtpp
)
dtpp on dtpp.playerID = hofi.playerID
LEFT JOIN 
(
SELECT b.playerID, sum(b.AB) as atBats, sum(b.H) as hits
FROM Batting b
GROUP BY b.playerID
)
batting on batting.playerID = hofi.playerID
LEFT JOIN 
(
SELECT p.playerID, sum(p.W) as wins, sum(p.L) as losses
FROM Pitching p 
GROUP BY p.playerID
)
pitching on hofi.playerID = pitching.playerID
LEFT JOIN 
(
SELECT f.playerID, sum(f.PO) as putOuts, sum(f.A) as assists, sum(f.E) as errors
FROM Fielding f 
GROUP BY f.playerID
)
fielding on hofi.playerID = fielding.playerID
LEFT JOIN
(
SELECT m.playerID, m.birthState
FROM Master m
)
master on master.playerID = hofi.playerID
WHERE hofi.inductedYear >= 2000 and hofi.category = 'Player';
'''
df_post2000 = pandas.read_sql(sql, conn)
conn.close()

# Add composite feature columns
df_post2000['batting_average'] = df_post2000.hits / df_post2000.atBats
df_post2000['winning_percentage'] = df_post2000.wins / (df_post2000.wins + df_post2000.losses)
df_post2000['fielding_percentage'] = (df_post2000.putOuts + df_post2000.assists) / (df_post2000.putOuts + df_post2000.assists + df_post2000.errors)

# Initial cleanup of data
# Split into explanatory and response features
df_post2000.drop('playerID',  1, inplace = True)
explanatory_features_post2000 = [col for col in df_post2000.columns if col not in ['inducted']]
explanatory_df_post2000 = df_post2000[explanatory_features_post2000]
explanatory_df_post2000.dropna(how='all', inplace = True) 
explanatory_colnames_post2000 = explanatory_df_post2000.columns
response_series_post2000  = df_post2000.inducted
response_series_post2000.dropna(how='all', inplace = True) 
response_series_post2000.index[~response_series_post2000.index.isin(explanatory_df_post2000.index)]

# Split the data into numeric and categorical parts
string_features_post2000 = explanatory_df_post2000.ix[:, explanatory_df_post2000.dtypes == 'object']
numeric_features_post2000 = explanatory_df_post2000.ix[:, explanatory_df_post2000.dtypes != 'object']

# Bin the categorical features
string_features_post2000 = string_features_post2000.fillna('Nothing')
string_features_post2000 = cleanup_data(string_features_post2000)
string_features_post2000.teamID.value_counts(normalize = True)

# Encode the categorical features
encoded_data_post2000 = get_binary_values(string_features_post2000)
encoded_data_post2000.head()

# Impute missing numeric data
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features_post2000)
numeric_features_post2000 = pandas.DataFrame(imputer_object.transform(numeric_features_post2000), 
                                    columns = numeric_features_post2000.columns)

# Merge columns back together
explanatory_df_post2000 = pandas.concat([numeric_features_post2000, encoded_data_post2000],axis = 1)
explanatory_df_post2000.dropna(how='any', inplace = True) 

# Find features with no variance
toKeepDelete_post2000 = find_zero_var(explanatory_df_post2000)
print toKeepDelete_post2000 # nothing to delete

# Delete columns with perfect correlation
corrMatrixToRemove_post2000 = find_perfect_corr(explanatory_df_post2000)
print corrMatrixToRemove_post2000['toRemove'] # nothing to delete

# Match columns in training and testing data by adding missing columns and deleting extra columns from testing data
add_columns = [col for col in explanatory_df.columns if col not in explanatory_df_post2000.columns]
add_df = pandas.DataFrame(0, index = explanatory_df_post2000.index, columns = add_columns)
explanatory_df_post2000 = pandas.concat([explanatory_df_post2000, add_df], axis = 1)
del_columns = [col for col in explanatory_df_post2000.columns if col not in explanatory_df.columns]
explanatory_df_post2000.drop(del_columns, 1, inplace=True)

# Reorder columns to match pre2000 column order
explanatory_df_post2000 = explanatory_df_post2000[explanatory_df.columns]

# Scale data with previously created scaler object
explanatory_df_post2000 = pandas.DataFrame(scaler.transform(explanatory_df_post2000), columns = explanatory_df_post2000.columns)

###############
# Predict using Random Forest
###############
rf = ensemble.RandomForestClassifier(n_estimators= 500)
rf.fit(explanatory_df, response_series)
predicted_rf_post2000 = rf.predict(explanatory_df_post2000)

# Accuracy = 86%
number_correct = len(explanatory_df_post2000[response_series_post2000 == predicted_rf_post2000])
rf_accuracy = number_correct / float(len(explanatory_df_post2000))
print rf_accuracy

###############
# Predict using Boosting Tree
###############
boosting_tree.fit(explanatory_df, response_series)
predicted_bt_post2000 = boosting_tree.predict(explanatory_df_post2000)

# Accuracy = 88%
number_correct = len(explanatory_df_post2000[response_series_post2000 == predicted_bt_post2000])
bt_accuracy = number_correct / float(len(explanatory_df_post2000))
print bt_accuracy

##############
# Predict using SVM
##############
predicted_svm_post2000 = svm_grid.predict(explanatory_df_post2000)

# Accuracy = 83%
number_correct = len(explanatory_df_post2000[response_series_post2000 == predicted_svm_post2000])
svm_accuracy = number_correct / float(len(explanatory_df_post2000))
print svm_accuracy

# So, Boosting Tree predicts with the highest accuracy on the holdout test set

###########
# Run PCA on hold out test set and re-do the estimators to see if results improve
###########
pca_post2000 = PCA(n_components=6)
pca_post2000.fit(explanatory_df_post2000)

# Extract the first two components to match the training set
pca_df_post2000 = pandas.DataFrame(pca_post2000.transform(explanatory_df_post2000))
pca_df_small_post2000 = pca_df_post2000.ix[:,0:1]

# Random Forest
rf.fit(pca_df_small, response_series)
predicted_rf_post2000_pca = rf.predict(pca_df_small_post2000)
# Accuracy = 66% compared to 86% for original data
number_correct = len(pca_df_small_post2000[response_series_post2000 == predicted_rf_post2000_pca])
rf_accuracy = number_correct / float(len(pca_df_small_post2000))
print rf_accuracy

# Boosting Tree
boosting_tree.fit(pca_df_small, response_series)
predicted_bt_post2000_pca = boosting_tree.predict(pca_df_small_post2000)
# Accuracy = 70% for PCA compared to 88% for original data
number_correct = len(pca_df_small_post2000[response_series_post2000 == predicted_bt_post2000_pca])
bt_accuracy = number_correct / float(len(explanatory_df_post2000))
print bt_accuracy

# SVM
svm_grid.fit(pca_df_small, response_series)
predicted_svm_post2000_pca = svm_grid.predict(pca_df_small_post2000)
# Accuracy = 82% for PCA compared to the 83% for original data
number_correct = len(pca_df_small_post2000[response_series_post2000 == predicted_svm_post2000_pca])
svm_accuracy = number_correct / float(len(explanatory_df_post2000))
print svm_accuracy