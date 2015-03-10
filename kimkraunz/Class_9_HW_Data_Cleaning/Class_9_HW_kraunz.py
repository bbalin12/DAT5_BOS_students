# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:03:56 2015

"""

import pandas
import sqlite3

pandas.set_option('display.max_columns', None)

con = sqlite3.connect('/Users/jkraunz/Documents/SQLite/lahman2013.sqlite.crdownload')

query = 'select playerID, yearID from AllstarFull'
df = pandas.read_sql(query, con)
con.close()

# for some reason there were multiple games played in the same year.  Need to drop those duplicates
df.drop_duplicates()

conn = sqlite3.connect('/Users/jkraunz/Documents/SQLite/lahman2013.sqlite.crdownload')
df.to_sql('num_of_allstar', conn, if_exists = 'replace')

conn.close()

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

# Data manipulation

df['inducted1'] = 0
df.inducted1[df.inducted == 'Y'] = 1

df['years_played'] = 0
df.years_played[df.years_pitched >= df.years_batted] = df.years_pitched
df.years_played[df.years_pitched < df.years_batted] = df.years_batted

df.drop(['playerID', 'inducted', 'years_pitched', 'years_batted', 'final_year_voted'],  1, inplace = True)

# dropped final_year_voted since it's not a good variable to use

new_columns = df.columns.values
new_columns[19] = 'inducted'
df.columns = new_columns

df.head()

# Set up explanatory and response features

explanatory_features = [col for col in df.columns if col not in ['inducted']]
explanatory_df = df[explanatory_features]

explanatory_df.dropna(how = 'all', inplace = True)

explanatory_col_names = explanatory_df.columns

response_series = df.inducted

response_series.dropna(how = 'all', inplace = True)

response_series.index[~response_series.index.isin(explanatory_df.index)]

explanatory_df.describe()

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

# Create list of column names for when used on testing data
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

# Color chart to look at correlation (only first 25)
toChart = explanatory_df.ix[:, 0:25].corr()
toChart.head()

import matplotlib.pyplot as plt
import numpy
plt.pcolor(toChart)
plt.yticks(numpy.arange(0.5, len(toChart.index), 1), toChart.index)
plt.xticks(numpy.arange(0.5, len(toChart.columns), 1), toChart.columns, rotation = -90)
plt.colorbar()
plt.show()

# Function to look at all correlation
   
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


# 10. Perform grid search and RFE on your data to find the optimal estimator for your data.

# Recursive Feature Elimination (without grid search)

from sklearn.feature_selection import RFECV
from sklearn import tree

class TreeClassifierWithCoef(tree.DecisionTreeClassifier):
   def fit(self, *args, **kwargs):
       super(tree.DecisionTreeClassifier, self).fit(*args, **kwargs)
       self.coef_ = self.feature_importances_

# these are the default settings for the tree based classifier
decision_tree = TreeClassifierWithCoef(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)

rfe_cv = RFECV(estimator=decision_tree, step=1, cv=10,
              scoring='roc_auc', verbose = 1)
rfe_cv.fit(explanatory_df, response_series)

print "Optimal number of features :{0} of {1} considered".format(rfe_cv.n_features_, len(explanatory_df.columns))
# Optimal number of features :6 of 48 considered

rfe_cv.n_features_

print rfe_cv.grid_scores_
#[ 0.71767479  0.73645555  0.80918431  0.80729704  0.80815691  0.82146712
#  0.790365    0.79967681  0.80072051  0.79699342  0.78881872  0.79984285
#  0.79920091  0.7840494   0.78297011  0.79266293  0.77987013  0.79018561
#  0.78167141  0.79648639  0.79017968  0.78059064  0.79439157  0.79211439
#  0.80158483  0.7959171   0.7812311   0.7970972   0.79501868  0.79661685
#  0.79715798  0.79315365  0.79942774  0.78824053  0.79753306  0.79178823
#  0.80416444  0.78612347  0.79438119  0.80060932  0.79057256  0.78093311
#  0.7923516   0.79152583  0.79225079  0.78993803  0.79679328  0.79108107]


## Plot the results
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (ROC_AUC)")
plt.plot(range(1, len(rfe_cv.grid_scores_) + 1), rfe_cv.grid_scores_)
plt.show()

features_used = explanatory_df.columns[rfe_cv.get_support()]
print features_used

# Index([u'years_voted', u'total_RBIs', u'total_runs', u'total_hits', u'total_wins', u'years_played'], dtype='object')


# Combine RFE with grid search to identify the best parameters

from sklearn.grid_search import GridSearchCV

depth_range = range(3, 6)
param_grid = dict(estimator__max_depth=depth_range)

rfe_grid_search = GridSearchCV(rfe_cv, param_grid, cv = 10, scoring = 'roc_auc')
rfe_grid_search.fit(explanatory_df, response_series)

print rfe_grid_search.grid_scores_
# [mean: 0.79559, std: 0.06544, params: {'estimator__max_depth': 3}, mean: 0.86625, std: 0.04474, params: {'estimator__max_depth': 4}, mean: 0.86418, std: 0.05680, params: {'estimator__max_depth': 5}]

rfe_grid_search.best_params_
# {'estimator__max_depth': 4}

grid_mean_scores = [score[1] for score in rfe_grid_search.grid_scores_]

# Plot max_depth vs. ROC score
plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.plot(rfe_grid_search.best_params_['estimator__max_depth'], rfe_grid_search.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')
plt.grid(True)

# pull out the winning estimator.
best_decision_tree_rfe_grid = rfe_grid_search.best_estimator_

features_used_rfecv_grid = explanatory_df.columns[best_decision_tree_rfe_grid.get_support()]

print features_used_rfecv_grid

# Index([u'years_voted', u'total_RBIs', u'total_runs', u'total_hits', u'total_3B', u'total_wins', u'total_errors', u'total_post_hits', u'birthCountry_USA'], dtype='object')

best_features = explanatory_df[features_used_rfecv_grid]


# Use different methods to check accuracy
from sklearn.cross_validation import cross_val_score

accuracy_scores_cart_pre2000 = cross_val_score(decision_tree, best_features, response_series, cv=10, scoring='accuracy', n_jobs = -1)

print "The 10 fold accuracy is: %f " % accuracy_scores_cart_pre2000.mean()
# accuracy is 84%

# Cohen's Kappa
mean_accuracy_score_cart_pre2000 = accuracy_scores_cart_pre2000.mean()
largest_class_percent_of_total_pre2000 = response_series.value_counts(normalize = True)[0]

kappa_cart_pre2000 = (mean_accuracy_score_cart_pre2000 - largest_class_percent_of_total_pre2000) / (1-largest_class_percent_of_total_pre2000)

print "Cohen's Kappa is: %f " %  kappa_cart_pre2000
# Cohen's Kappa is .307- not great

# F1 score
f1_scores_cart_pre2000 = cross_val_score(decision_tree, best_features, response_series, cv=10, scoring='f1', n_jobs = -1)

print "The mean F1 score is: %f" % f1_scores_cart_pre2000.mean()
# F1 score is .637, again not great

# ROC
roc_scores_cart_pre2000 = cross_val_score(decision_tree, best_features, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

print "The mean ROC scores is : %f" % roc_scores_cart_pre2000.mean()
# ROC score is .784, OK

##########################################################################

# 11. Test model on post 2000 data

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

post2000_df.describe()

# recreating categorical data
post2000_df['inducted1'] = 0
post2000_df.inducted1[post2000_df.inducted == 'Y'] = 1

post2000_df['years_played'] = 0
post2000_df.years_played[post2000_df.years_pitched >= post2000_df.years_batted] = post2000_df.years_pitched
post2000_df.years_played[post2000_df.years_pitched < post2000_df.years_batted] = post2000_df.years_batted

post2000_df.drop(['playerID', 'inducted', 'years_pitched', 'years_batted', 'final_year_voted'],  1, inplace = True)

# dropped final_year_voted since it's not a good variable to use

post2000_df.head()

# Set up explanatory and response features

post2000_explanatory_features = [col for col in post2000_df.columns if col not in ['inducted1']]
post2000_explanatory_df = post2000_df[explanatory_features]

post2000_explanatory_df.dropna(how = 'all', inplace = True)

post2000_explanatory_col_names = post2000_explanatory_df.columns

post2000_response_series = post2000_df.inducted1

post2000_response_series.dropna(how = 'all', inplace = True)

post2000_response_series.index[~post2000_response_series.index.isin(post2000_explanatory_df.index)]

# 12. Reclean, encode, and scale model


#       1. Split data into categorical and numerical data

post2000_string_features = post2000_explanatory_df.ix[: , post2000_explanatory_df.dtypes == 'object']
post2000_numeric_features = post2000_explanatory_df.ix[: , post2000_explanatory_df.dtypes != 'object']

post2000_string_features.head()
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

post2000_explanatory_df.drop(['total_2B'], 1, inplace = True)

#       9. Scale data with zero mean and unit variance (from pre-2000 fit)

post2000_explanatory_df = pandas.DataFrame(scaler.transform(post2000_explanatory_df), columns = post2000_explanatory_df.columns, index = post2000_explanatory_df.index)



#       10. Test data using optimized Decision Tree model (trained on pre-2000 data)

pred_post2000_inductions = best_decision_tree_rfe_grid.predict(post2000_explanatory_df)

from __future__ import division

number_correct = len(post2000_response_series[post2000_response_series == pred_post2000_inductions])
total = len(post2000_response_series)
accuracy = number_correct / total


print accuracy
# 85.8%

cm = pandas.crosstab(post2000_response_series, pred_post2000_inductions, rownames=['True Label'], colnames=['Predicted Label'], margins=True)

print cm
'''
Predicted Label    0   1  All
True Label                   
0                203  16  219
1                 24  38   62
All              227  54  281

'''


