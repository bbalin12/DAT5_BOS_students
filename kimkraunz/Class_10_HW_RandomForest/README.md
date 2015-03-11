# Kim Kraunz
# Class 10 Homework - Ensemble Methods and Neural Networks


## Introduction
I used the Lahman Baseball Database for all analysis. In this homework I used Random Forest (RF), Boosting Trees (GBM), and Neural Network (NN) classifier to predict Hall of Fame Induction.  Using the ROC scores, I compared the accuracy of Random Forests, Boosting Trees, and Neural Networks, as well as Decision Trees, Logistic Regression, and Naive Bayes.  I also used grid seach to optimize ythe Neural Network tuning parameters for learning_rate, iteration_range, and components.


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

```
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

string_features = string_features.fillna('Nothing')

# 4. Detect low-frequency levels in categorical features and bin them under ‘other’

string_features['birthCountry_USA'] = 0
string_features.birthCountry_USA[df.birthCountry == 'USA'] = 1

string_features.drop(['birthCountry'],  1, inplace = True)

# 5. Encode each categorical variable into a sequence of binary variables.

# Not needed


# 6. Merge your encoded categorical data with your numeric data

explanatory_df = pandas.concat([numeric_features, string_features], axis = 1)
explanatory_df.head()

explanatory_df.describe()

# 7. Remove features with no variation
   
# Did in HW 9

# No features had zero variance

# 8. Remove perfectly correlated features

# Function to look at all correlation
   
# Did in HW 9

# 9. Scale your data with zero mean and unit variance

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)

explanatory_df.describe()
```

####Random Forest
I performed Random Forest using 10-fold cross validation to predict induction into the Hall of Fame using the following code:

```
# Random Forest

rf = ensemble.RandomForestClassifier(n_estimators = 500)

roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring = 'roc_auc', n_jobs = -1)

print roc_scores_rf.mean()
```
I found a ROC score of 0.9236

I then compared that to the ROC score from the 10-fold cross validated Decision Tree model.

```
roc_scores_tree = cross_val_score(tree.DecisionTreeClassifier(), explanatory_df, response_series, cv=10, scoring = 'roc_auc', n_jobs = -1)

print roc_scores_tree.mean()
```

The ROC score for the Decision Tree was 0.7819.  The Random Forest model has much higher accuracy.

I then optimized the number of estimators used in the Random Forest using Grid Search, plotted the numbers of estimators versus the ROC score, and printed out the optimal number of estimators and best ROC score.

```
# grid search to optimize Random Forest
trees_range = range(10, 550, 10)
param_grid = dict(n_estimators = trees_range)

grid = GridSearchCV(rf, param_grid, cv=10, scoring = 'roc_auc', n_jobs = -1)
grid.fit(explanatory_df, response_series)

grid_mean_scores = [result[1] for result in grid.grid_scores_]

# increases size of graph
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5

plt.figure()
plt.plot(trees_range, grid_mean_scores)

best_decision_tree_est = grid.best_estimator_
print best_decision_tree_est.n_estimators
print grid.best_score_
``` 

I found that the optimal number of estimators was 200 and the best ROC score was 0.9257.

![RFestimators](https://github.com/bbalin12/DAT5_BOS_students/blob/master/kimkraunz/Class_10_HW_RandomForest/RandomForestEstimatorPlot.png)

####Boosting Tree
I ran the Boosting Tree using 10-fold cross validation to predict induction into the Hall of Fame using the following code:

```
# Boosting tree

boosting_tree = ensemble.GradientBoostingClassifier()

roc_scores_gbm = cross_val_score(boosting_tree, explanatory_df, response_series, cv=10, scoring = 'roc_auc', n_jobs = -1)
```

The ROC score for the Boosting Tree method was 0.9211, slightly better than for the Random Forest model.

I then optimized the learning rate, subsampling range, and number of estimators for the Boosting Tree.  I chose a range of 200 to 300 for the number of estimators since the previously optimized number of estimators was 300.

```
from numpy import arange

# learning rate range is most important to tune for
learning_rate_range = arange(0.01, 0.4, 0.02)
subsampling_range = arange(0.25, 1, 0.25)
n_estimators_range = arange(200, 300, 25)

param_grid = dict(learning_rate = learning_rate_range, n_estimators = n_estimators_range, subsample = subsampling_range)

gbm_grid = GridSearchCV(boosting_tree, param_grid, cv=10, scoring = 'roc_auc', n_jobs = -1)

gbm_grid.fit(explanatory_df, response_series)

print gbm_grid.best_params_
print gbm_grid.best_score_
```

The optimized parameters were: {’n_estimators': 225, 'subsample': 0.75, 'learning_rate': 0.029999999999999999}
 
The ROC score for the 10-fold cross validated and optimized Boosting Tree model was .9263.  The optimized Boosting Tree has a slightly higher accuracy.

I then wanted to compare the ROC curves for the Decision Tree, optimized Random Forest, and optimized Boosting Tree models.

```
from sklearn.cross_validation import train_test_split
from sklearn import metrics

xTrain, xTest, yTrain, yTest = train_test_split(explanatory_df, response_series, test_size = 0.3)

# Create pandas dataframe
tree_probabilities = pandas.DataFrame(tree.DecisionTreeClassifier().fit(xTrain, yTrain).predict_proba(xTest))
rf_probabilities = pandas.DataFrame(best_decision_tree_est.fit(xTrain, yTrain).predict_proba(xTest))
gbm_probabilities = pandas.DataFrame(gbm_grid.best_estimator_.fit(xTrain, yTrain).predict_proba(xTest))

# plot with Y and 2nd column of pandas dataframe
tree_fpr, tree_tpr, thresholds = metrics.roc_curve(yTest, tree_probabilities[1])
rf_fpr, rf_tpr, thesholds = metrics.roc_curve(yTest, rf_probabilities[1])
gbm_fpr, gbm_tpr, thesholds = metrics.roc_curve(yTest, gbm_probabilities[1])

plt.figure()
plt.plot(tree_fpr, tree_tpr, color = 'g')
plt.plot(rf_fpr, rf_tpr, color = 'b')
plt.plot(gbm_fpr, gbm_tpr, color = 'r')
plt.xlabel('False Positive Rate(1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
```
![ROCscores](https://github.com/bbalin12/DAT5_BOS_students/blob/master/kimkraunz/Class_10_HW_RandomForest/ROCscoresPlot.png)

Random Forest and Boosting Tree both look very similar with Boosting Tree being slightly better.  It mimics the mean ROC score that I found during 10-fold cross-validation and optimization.

I wanted to look at the features with the highest importance in the Boosting Tree optimized model.

```
importances = pandas.DataFrame(gbm_grid.best_estimator_.feature_importances_, index = explanatory_df.columns, columns =['importance'])

importances.sort(columns = ['importance'], ascending = False, inplace = True)
print importances

#                  importance
#total_3B            0.202059
#total_wins          0.178061
#years_voted         0.151448
#total_runs          0.133153
#total_RBIs          0.119498
#total_hits          0.090283
#total_errors        0.075008
#total_post_hits     0.047740
#birthCountry_USA    0.002750
```

The three features with the highest importance were total_3B, total_wins, and years_voted.  The teamID categorical data did not contribute very much to the model.

I then looked at the partial dependence plots for the three features with the most importance.  Partial dependence plots allow us to look at the relationship between a feature and hall of fame induction while controlling for the other features.

```
from sklearn.ensemble.partial_dependence import plot_partial_dependence

features = [i for i, j in enumerate(explanatory_df.columns.tolist()) if j in importances.importance[0:3].index.tolist()]

fig, axs = plot_partial_dependence(gbm_grid.best_estimator_, explanatory_df, features, feature_names = explanatory_df.columns)
```

![PartialDependence](https://github.com/bbalin12/DAT5_BOS_students/blob/master/kimkraunz/Class_10_HW_RandomForest/PartialDependencePlots.png)

There is a positive relationship between years voted and induction which indicates that the more years that a player was voted on the more likely he was to be inducted while controlling for other features.  There was a positive relationship between total_wins and induction.  Interestingly, there was a negative relationship between total_3B and induction.  This may be a result of total_3B being correlated to another feature and is a sign that the model could be improved.

####Neural Networks

Lastly, I wanted to test my features using the Neural Networks model.

```
# Neural Networks

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model

logistic_classifier = linear_model.LogisticRegression()
neural_net = BernoulliRBM(random_state=0, verbose=True)

neural_classifier = Pipeline(steps=[('neural_net', neural_net), ('logistic_classifier', logistic_classifier)])

roc_scores_nn = cross_val_score(neural_classifier, explanatory_df, response_series, cv=10, scoring = 'roc_auc')
```
The ROC score for the Neural Network model was 0.5596.  Not very good.

I wanted to see if I could increase the accuracy by optimizing the Neural Network parameters.

```
learning_rate_range = arange(0.01, 0.2, 0.05)
components_range = range(250, 500, 50)

param_grid = dict(neural_net__n_components = components_range, neural_net__learning_rate = learning_rate_range)
#, neural_net__n_iter = iteration_range)

# doing 5-fold CV here for reasons of time; feel free to do 10-fold 
# in your own leisure.
nn_grid = GridSearchCV(neural_classifier, param_grid, cv=5, scoring='roc_auc')
nn_grid.fit(explanatory_df, response_series)

print nn_grid.best_score_
print nn_grid.best_params_
```
The best ROC score was 0.7465 using the following optimal parameters:
{'neural_net__n_components': 450, 'neural_net__learning_rate': 0.01}

The Neural Network model is not a good model to represent the data.

##Conclusion
Both the optimized Random Forest and Boosting Tree models were good predictors of induction into the Hall of Fame.  Depending on the computation resources available, I would choose either of those two models.  The Neural Network model was a poor predictor of induction.