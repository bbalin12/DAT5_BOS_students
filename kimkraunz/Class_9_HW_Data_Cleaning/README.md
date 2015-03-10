 __Kim Kraunz__
# Class 9 Homework - Data Cleaning and Manipulation


## Introduction
I used the Lahman Baseball Database for all analysis. In this homework I added a categorical feature, bin and encoded the categorical data, replaced the Nans, removed features with no variance or perfect correlation, performed recursive feature elimination, and evaluated the data using an optimized Decision Tree model.  Lastly, I evaluated model prediction of Hall of Fame induction on a testing data set of 2000 or later.

I used the following code to pull the total salaries and total runs grouped by year from the SQLite database.

```
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
```

#### Functions used
```
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
```

####Data Manipulation
I manipulated the data in the following ways:
1. I created a binary variable inducted1 from the inducted variable
2. I created a years_played variable from the years_pitched and years_batted variables
3. I dropped unneccessary variables (playerID, inducted, years_pitched, years_batter), variables with perfect correlation(total_2B), and variables that did not add to the model (birthCountry, total_post_RBIs) as determined in Homework 9
4. I also dropped final_year_voted since it is a time series variable and we haven't covered it yet.

```
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

```

#### Response and explanatory variables

I used the features I identified in HW 9 through Recursive Feature Elimination in my model.  I defined the response and explanatory variables using the following code:
```
explanatory_df = df[['years_voted', 'total_RBIs', 'total_runs', 'total_hits', 'total_3B', 'total_wins', 'total_errors', 'total_post_hits', 'birthCountry']]

explanatory_df.dropna(how = 'all', inplace = True)

explanatory_col_names = explanatory_df.columns

response_series = df.inducted

response_series.dropna(how = 'all', inplace = True)

response_series.index[~response_series.index.isin(explanatory_df.index)]

explanatory_df.describe()

```

#### Data Cleaning
I cleaned the data by first splitting the explanatory variables in to string and numeric data.  I then filled any Nans in the categorical data with 'Nothing' and created the birthCountry_USA variable as a binary variable.  I filled any Nans in the numerical data with the feature median.  Finally, I merged the string and numerical data back together into a Pandas dataframe.  I used the features indentified with Recursive Feature Elimination in HW 9.

#####1. Split data into categorical and numeric data
```
string_features = explanatory_df.ix[: , explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[: , explanatory_df.dtypes != 'object']

string_features.head()
numeric_features.head()
```

####2. Fill numeric NaNs through imputation
```
from sklearn.preprocessing import Imputer

imputer_object = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

numeric_features.head()
numeric_features.describe()
```

####3. Fill categorical NaNs with ‘Nothing’
```
string_features = string_features.fillna('Nothing')
```

####4. Detect low-frequency levels in categorical features and bin them under ‘other’
```
string_features.teamID.value_counts(normalize = True)
string_features.birthCountry.value_counts(normalize = True)

cleanup_data(string_features)
    
string_features.teamID.value_counts(normalize = True)
string_features.birthCountry.value_counts(normalize = True)

len(string_features.teamID)

# Create list of column names for when used on testing data
string_features_cat =   {}
for col in string_features.columns:
  string_features_cat[col] = string_features[col].unique()

```
####5. Encode each categorical variable into a sequence of binary variables.
```
string_features = get_binary_values(string_features)
```

####6. Merge your encoded categorical data with your numeric data
```
explanatory_df = pandas.concat([numeric_features, string_features], axis = 1)
explanatory_df.head()

explanatory_df.describe()
```
####7. Remove features with no variation
 ```  
find_zero_var(explanatory_df)
```
No features had zero variance

####8. Remove perfectly correlated features
```
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
```
![ColorCorrelationMatrix]()

The color correlation matrix was only used with the first 25 variables.  You can see clear correlation between total_2B and total_3B.  You can also see a strong correlation between years_played and total_2B and total_3B.

To calculate all the correlations, I used a function from class.

```   
find_perfect_corr(explanatory_df)
```
It found that total_2B and total_3B were perfectly correlated so I dropped total_2B.  It did not find that years_played was correlated with either total_2B or total_3B so I left it in.
```
explanatory_df.drop('total_2B', 1, inplace=True)
explanatory_df.head()
```
####9. Scale your data with zero mean and unit variance
```
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)

explanatory_df.describe()
```

####10. Perform grid search and RFE on your data to find the optimal estimator for your data.

I first performed Recursive Feature Elimination without Grid Search
```
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

rfe_cv.n_features_

print rfe_cv.grid_scores_

Optimal number of features :6 of 48 considered
```

I then plotted the results

```
## Plot the results
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (ROC_AUC)")
plt.plot(range(1, len(rfe_cv.grid_scores_) + 1), rfe_cv.grid_scores_)
plt.show()
```
![DecisionTreeFeaturesUsed]()

There is a clear spike at 6 features used.

I then printed out the features to use moving forward.

```
features_used = explanatory_df.columns[rfe_cv.get_support()]
print features_used


Index([u'years_voted', u'total_RBIs', u'total_runs', u'total_hits', u'total_wins', u'years_played'], dtype='object')
```

I combined recursive feature elimination with Grid Search to identify the best parameters.
```
# Combine RFE with grid search

from sklearn.grid_search import GridSearchCV

depth_range = range(3, 6)
param_grid = dict(estimator__max_depth=depth_range)

rfe_grid_search = GridSearchCV(rfe_cv, param_grid, cv = 10, scoring = 'roc_auc')
rfe_grid_search.fit(explanatory_df, response_series)

print rfe_grid_search.grid_scores_

[mean: 0.79559, std: 0.06544, params: {'estimator__max_depth': 3}, mean: 0.86625, std: 0.04474, params: {'estimator__max_depth': 4}, mean: 0.86418, std: 0.05680, params: {'estimator__max_depth': 5}]

rfe_grid_search.best_params_

{'estimator__max_depth': 4}
```

I found that the optimal depth was four and confirmed it by plotting the depth vs the mean ROC scores.

```
grid_mean_scores = [score[1] for score in rfe_grid_search.grid_scores_]

# Plot max_depth vs. ROC score
plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.plot(rfe_grid_search.best_params_['estimator__max_depth'], rfe_grid_search.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')
plt.grid(True)
```
![DecisionTreeMaxDepthOpt]()

I printed out the best features.
```
# pull out the winning estimator.
best_decision_tree_rfe_grid = rfe_grid_search.best_estimator_

features_used_rfecv_grid = explanatory_df.columns[best_decision_tree_rfe_grid.get_support()]

print features_used_rfecv_grid

Index([u'years_voted', u'total_RBIs', u'total_runs', u'total_hits', u'total_3B', u'total_wins', u'total_errors', u'total_post_hits', u'birthCountry_USA'], dtype='object')

best_features = explanatory_df[features_used_rfecv_grid]
```

####Model Evaluation

I evaluated the Decision Tree model using accuracy, Cohen's Kappa score, F1 score, and ROC scores.
```
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
```
The accuracy (what did the model get correct) was 84%.  Using Cohen's Kappa (.307) and the ROC scores (.784), the model would be defined as "fair" by both metrics.  After evaluating the Decision Tree model with different metrics, I think there might be a better method out there to predict Hall of Fame induction.

####11. Test model on post 2000 data

I then imported the 2000 and later data using the following code.  I manipulated the data and defined the explanatory and response variables the same as I had for the training data.

```
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
```

####12. Reclean, encode, and scale model

As I had done for the training data, I split the data into categorical and numerical data, filled the numeric NaNs through imputation, and filled the categorical Nans with feature medians.
```
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
```
I then matched the categorical data in the testing set to the training set.

```
#       4. Matches categorical data to pre-2000 data

# If there is a value that is not in the training data set, replaces value with "Other"
for col in post2000_string_features:
    post2000_string_features[col].ix[~post2000_string_features[col].isin(string_features_cat[col])] = "Other"
```

I encoded the categorical variables into binary variables.  I if there were dummy variable in the training set that weren't in the testing set, I added them to the testing set.  Finally I sorted the columns to make sure that they matched the training set.

```
#       5. Encode each categorical variable into a sequence of binary variables.

post2000_string_features = get_binary_values(post2000_string_features)

# must make sure that there are dummy variables for variables in the training data that is not in the testing data already

for col in string_features:
	if col not in post2000_string_features:
		post2000_string_features[col] = 0
 
# Make sure that the string data is sorted the same as the training data 
post2000_string_features = post2000_string_features[string_features.columns]
```
I merged the numerical and categorical data back together, dropped total_2B that was perfectly correlated in the training set, and scaled the data using the training transformation.  I would have also removed any features that had been identified as having zero variance in the training set but there were none.
```

#       6. Merge encoded categorical data with numeric data

post2000_explanatory_df = pandas.concat([post2000_numeric_features, post2000_string_features], axis = 1)
post2000_explanatory_df.head()

#       7. Remove features with no variation

#           There were none

#       8. Remove features with perfect correlation

post2000_explanatory_df.drop(['total_2B'], 1, inplace = True)

#       9. Scale data with zero mean and unit variance (from pre-2000 fit)

post2000_explanatory_df = pandas.DataFrame(scaler.transform(post2000_explanatory_df), columns = post2000_explanatory_df.columns, index = post2000_explanatory_df.index)
```
Finally, I evaluated the accuracy of the Decision Tree model I developed using training data with my testing data.
```
#       10. Test data using optimized Decision Tree model (trained on pre-2000 data)

pred_post2000_inductions = best_decision_tree_rfe_grid.predict(post2000_explanatory_df)

from __future__ import division

number_correct = len(post2000_response_series[post2000_response_series == pred_post2000_inductions])
total = len(post2000_response_series)
accuracy = number_correct / total


print accuracy
85.8%
```
The accuracy remained decent at 85.8%.  Not bad but still not great.  It was actually slight higher than the accuracy of the training data (84%) which shows at least that the model is consistent.

I printed out the confusion matrix to calculate the specificity and sensitivity.

```
cm = pandas.crosstab(post2000_response_series, pred_post2000_inductions, rownames=['True Label'], colnames=['Predicted Label'], margins=True)

print cm
'''
Predicted Label    0   1  All
True Label                   
0                203  16  219
1                 24  38   62
All              227  54  281
'''
```

sensitivity = 38/(38+24) = 61%

specificity = 203/(203 + 16) = 93%

The model is not very sensitive, meaning that it doesn't do a great job of predicting those that are inducted into the Hall of Fame.  However, it is fairly specific, meaning that it does a good job of predicting those not inducted into the Hall of Fame.

