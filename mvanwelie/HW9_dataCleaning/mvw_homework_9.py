# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 19:25:24 2015

@author: megan
"""
import pandas
import sqlite3

################
# Homework 9 - Data Cleaning and Manipulation
# Assignment:
# Join your SQL query used in last class' homework (to predict Baseball Hall of Fame indution) with the table we created in today's class (called dominant_team_per_player).
# Pick at least one additional categorical feature to include in your data.
# Bin and encode your categorical features.
# Remove features with perfect correlation and/or no variation.
# Scale your data and impute for your numeric NaNs.
# Perform recursive feature elimination on the data.
# Decide whether to use grid search to find your 'optimal' model.
# Bring in data after the year 2000, and preform the same transformations on the data you did with your training data.
# Predict Hall of Fame induction after the year 2000.
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

############
# Find and remove features with no variance 
############
toKeepDelete = find_zero_var(explanatory_df)
print toKeepDelete # nothing to delete

############
# Find and remove columns with perfect correlation
############
corrMatrixToRemove = find_perfect_corr(explanatory_df)
print corrMatrixToRemove['toRemove'] # nothing to remove
# explanatory_df.drop(corrMatrixToRemove['toRemove'], 1, inplace = True)

##############
# Scale data
#############
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)

########
# Recursive feature elimination
#######
from sklearn.feature_selection import RFECV
from sklearn import tree

# Create your tree based estimator
decision_tree = TreeClassifierWithCoef(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)

# Set up the estimator
# Score by AUC
rfe_cv = RFECV(estimator=decision_tree, step=1, cv=10,
              scoring='roc_auc', verbose = 1)
rfe_cv.fit(explanatory_df, response_series)

print "Optimal number of features :{0} of {1} considered".format(rfe_cv.n_features_,len(explanatory_df.columns))

# Examine the results of the estimator
# printing out scores as we increase the number of features -- the farther
# down the list, the higher the number of features considered.
print rfe_cv.grid_scores_

# Plot the results
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (ROC_AUC)")
plt.plot(range(1, len(rfe_cv.grid_scores_) + 1), rfe_cv.grid_scores_)
plt.show()

# Pull out the features used (just to see)
features_used = explanatory_df.columns[rfe_cv.get_support()]
print features_used

# Extract the final selected model object this way
final_estimator_used = rfe_cv.estimator_

##############
# Tune the model parameters
##############
from sklearn.grid_search import  GridSearchCV
depth_range = range(4, 6)
param_grid = dict(estimator__max_depth=depth_range)
rfe_grid_search = GridSearchCV(rfe_cv, param_grid, cv=10, scoring='roc_auc')
rfe_grid_search.fit(explanatory_df, response_series)

print rfe_grid_search.grid_scores_
rfe_grid_search.best_params_

# Plot the results.
grid_mean_scores = [score[1] for score in rfe_grid_search.grid_scores_]

plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.plot(rfe_grid_search.best_params_['estimator__max_depth'], rfe_grid_search.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')
plt.grid(True)

# Pull out the winning estimator.
best_decision_tree_rfe_grid = rfe_grid_search.best_estimator_

features_used_rfecv_grid = explanatory_df.columns[best_decision_tree_rfe_grid.get_support()]
print features_used_rfecv_grid

#############
# Test against hold out data past the year 2000
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

# Predict using the best decision tree tuned with training data
predicted_post2000 = best_decision_tree_rfe_grid.predict(explanatory_df_post2000)

# See the accuracy of the predictions
number_correct = len(explanatory_df_post2000[response_series_post2000 == predicted_post2000])
accuracy = number_correct / float(len(explanatory_df_post2000))

# Accuracy = 77%
print accuracy

