# __Kim Kraunz__
# Class 11 Homework - Unsupervised Learning and Dimensionality Reduction


## Introduction
I used the Lahman Baseball Database for all analysis. In this homework I used KMeans,  DBSCAN, and Hierarchical Clustering to analyze for clustering between total salaries vs. total runs per year in 1954 and later.   I also used Principal Component Analysis to try to find the best model components for prediction using Support Vector Machines, Random Forest, and Boosting Tree.

## Unsupervised learning using KMeans, DBSCAN, and Hierarchical Clustering
I used the following code to pull the total salaries and total runs grouped by year from the SQLite database.

```
query = '''
select s.yearID, sum(salary) as total_salaries, sum(R) as total_runs from salaries s
inner join Batting b on s.playerId = b.playerID 
where R is not null and s.yearID  > 1954
group by s.yearID
order by total_runs desc
'''

df = pandas.read_sql(query, conn)
conn.close()
```
#### Data exploration 

I plotted total salaies vs. total runs by year to visualize the relationship.

```
plt = df.plot(x = 'total_salaries', y = 'total_runs', kind = 'scatter')

for i, txt in enumerate(df.yearID):
    plt.annotate(txt, (df.total_salaries[i], df.total_runs[i]))
plt.show()
```

The plot clearly shows a relationship between total salaries and total runs over the years.
![Total salaries vs. Total runs](https://github.com/GarAust89/DAT5_BOS_students/blob/master/GarAust89/Class_09%20HW/DecisionTreeOptimalFeatureSelection.png)

#### Scaling data

The next step was to use StandardScaler from scikit to scale total salaries and total runs.

```
# Scales data
from sklearn import preprocessing

data = df[['total_salaries', 'total_runs']]
scaler = preprocessing.StandardScaler()
scaler.fit(data)
data = pandas.DataFrame(scaler.transform(data), columns = data.columns)
```

#### KMeans

I then ran KMeans with the number of clusters equal to three, based on my inital viusal of the scatter plot, and then plotted the result.
```
from sklearn.cluster import KMeans
import matplotlib.pylab as plt

kmeans_est = KMeans(n_clusters=3)
kmeans_est.fit(data)

labels = kmeans_est.labels_

plt = df.plot(x = 'total_salaries', y = 'total_runs', s=60, c=labels, kind = 'scatter')
for i, txt in enumerate(df.yearID):
    plt.annotate(txt, (df.total_salaries[i], df.total_runs[i]))
plt.show()
```
![KMeans 3 clusters](https://github.com/GarAust89/DAT5_BOS_students/blob/master/GarAust89/Class_09%20HW/DecisionTreeOptimalFeatureSelection.png)

The plot shows the three clusters but I wondered what it would look like if I changed the number of clusters to four.  I found using four clusters to be a better clustering of the data.

![KMeans 4 clusters](https://github.com/GarAust89/DAT5_BOS_students/blob/master/GarAust89/Class_09%20HW/DecisionTreeOptimalFeatureSelection.png)

#### DBSCAN

My next step was to use the follwing code to run DBSCAN to cluster the data and plot the results.  
```
from sklearn.cluster import DBSCAN
import numpy as np

dbsc = DBSCAN().fit(np.array(data))
labels = dbsc.labels_

plt = df.plot(x = 'total_salaries', y = 'total_runs', s=60, c=labels, kind = 'scatter')
for i, txt in enumerate(df.yearID):
    plt.annotate(txt, (df.total_salaries[i], df.total_runs[i]))
plt.show()
```

The following plot shows a nearly identical clustering of the data using DBSCAN as KMeans.

![DBSCAN](https://github.com/GarAust89/DAT5_BOS_students/blob/master/GarAust89/Class_09%20HW/DecisionTreeOptimalFeatureSelection.png)

#### Hierarchical Clustering

Finally I wanted to use Hierarchical Clustering to look at the clustering by plotting the dendrogram.  I used a depth of 2 and plotted the results.

```
# Dendrogram
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, fclusterdata

distanceMatrix = pdist(data)

dend = dendrogram(linkage(distanceMatrix, method='complete'), color_threshold=2, leaf_font_size=10, labels = df.yearID.tolist())
          
assignments = fcluster(linkage(distanceMatrix, method = 'complete'), 2, 'distance')

cluster_output = pandas.DataFrame({'team':df.yearID.tolist(), 'cluster':assignments})
cluster_output

plt.scatter(df.total_salaries, df.total_runs, s=60, c=cluster_output.cluster)
```

![Dendrogram](https://github.com/GarAust89/DAT5_BOS_students/blob/master/GarAust89/Class_09%20HW/GridSearchOptimalFeatureSelection.png)

Interestingly, using the dendrogram to cluster, it assigned five clusters instead of the four that DSCAN identified.  

![DBSCAN](https://github.com/GarAust89/DAT5_BOS_students/blob/master/GarAust89/Class_09%20HW/DecisionTreeOptimalFeatureSelection.png)

When we compare it to the DBSCAN clustering, we see that the dendrogram clusters years 1993-2005 together and creates a separate cluster for years 2011-2013 while DBSCAN creates four distinct clusters of 1985-1992, 1993-1999, 2000-2008, and 2009-2013.  The clustering using DBSCAN produced the most clearly defined results.  Interestingly, there are clear trends within the clusters.  For instance, since 2009, both total salaries and total runs are trending downwards.

## Dimensionality Reduction

I used the features from my previous homeworks to predict induction into the Baseball Hall of Fame for those inducted prior to 2000.  I used Principal Component Analysis to optimize the best combination of features and then tested the model using the Support Vector Machines, Boosting Tree, and Random Forest. Lastly I tested my Support Vector Machine, Boosting Tree, and Random Forest models that used Principal Component Analysis to identify the best combination to features on those inducted into the Hall of Fame in 2000 or later.

I imported my data from the SQLite using the following query.
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
```
####Functions used
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

#### Data Manipulation
I manipulated the data in the following ways:
1. I created a binary variable inducted1 from the inducted variable
2. I created a years_played variable from the years_pitched and years_batted variables
3. Finally, I dropped unneccessary variables (playerID, inducted, years_pitched, years_batter) and variables with perfect correlation(total_2B) as determined in Homework 9.
4. I also dropped final_year_voted as it is a time series variable and we don't know how to deal with them yet.
 
```
df['inducted1'] = 0
df.inducted1[df.inducted == 'Y'] = 1

df['years_played'] = 0
df.years_played[df.years_pitched >= df.years_batted] = df.years_pitched
df.years_played[df.years_pitched < df.years_batted] = df.years_batted

df.drop(['playerID', 'inducted', 'years_pitched', 'years_batted', 'final_year_voted'],  1, inplace = True)

df.head()
```

#### Response and explanatory variables

I defined the response and explanatory variables using the following code:
```
explanatory_features = [col for col in df.columns if col not in ['inducted1']]
explanatory_df = df[explanatory_features]
explanatory_df.dropna(how = 'all', inplace = True)
explanatory_col_names = explanatory_df.columns

response_series = df.inducted1
response_series.dropna(how = 'all', inplace = True)

response_series.index[~response_series.index.isin(explanatory_df.index)]

```

#### Data Cleaning
I cleaned the data by first splitting the explanatory variables in to string and numeric data.  I then filled any Nans in the categorical data with 'Nothing' and created dummy variables from the categorical data.  I filled any Nans in the numerical data with the feature median.  Finally, I merged the string and numerial data back together into a Pandas dataframe.

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
string_features_cat =   {}
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

```

#### Principal Component Analysis
I performed principal component analysis on the explanatory features and plotted the principal features using the following code:
```
from sklearn.decomposition import PCA

pca = PCA(n_components=6)
pca.fit(explanatory_df)

# extracting the components
pca_df = pandas.DataFrame(pca.transform(explanatory_df))

## plotting the first to principal components
pca_df.plot(x = 0, y= 1, kind = 'scatter')
```
![PCA Scatter]()

I then created a scree plot of the principal components versus the variance to understand the number of principal components that explain the variance.  From the plot, it is clear that the variance is reduced after 5 principal components.

![PCA Scree]()

####Boosting Tree using features transformed with Principal Component Analysis
I defined my explanatory features to the first two principal components and ran the Boosting Tree using 10 fold cross validation on the data.  I then compared the accuracy from the PCA transformed explanatory features to my non-transformed explanatory features using the Boosting Tree method.

```
pca_df_small = pca_df.ix[:,0:1]

boosting_tree = ensemble.GradientBoostingClassifier()

roc_scores_gbm_pca = cross_val_score(boosting_tree, pca_df_small, response_series, cv=10, scoring = 'roc_auc', n_jobs = -1)
print roc_scores_gbm_pca.mean()

roc_scores_gbm = cross_val_score(boosting_tree, explanatory_df, response_series, cv=10, scoring = 'roc_auc', n_jobs = -1)
print roc_scores_gbm.mean()
```

Interestingly, the Principal Component Analysis transformed data had a ROC score of 0.897 while the non-transformed data had a ROC score of 0.936.  In this case, not transforming the data using Principal Component Analysis yielded a better result.

####Support Vector Machines
I then used SUpport Vector Machines to predict Hall of Fame induction from both my non-PCA transformed data and my PCA transformed data.

```
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score

svm = SVC(kernel='poly')

roc_scores_svm = cross_val_score(svm, pca_df_small, response_series, cv=10, scoring = 'roc_auc', n_jobs= -1)
print roc_scores_svm.mean()
# .8454

roc_scores_svm = cross_val_score(svm, explanatory_df, response_series, cv=10, scoring = 'roc_auc', n_jobs= -1)
print roc_scores_svm.mean()
# .8551
```
The non-PCA transformed data (ROC = .855) performed better than the PCA transformed data (ROC = .845) using Support Vector Machines.  So far we've seen that the non-PCA transformed data has outperformed the PCA transformed data using two methods.

### Optimizing Support Vector Machines
I then used Grid Search to optimize the kernel type for Support Vector Machines.  Since then non-PCA transformed data was performing better, I used the non-transformed data moving forward.

```
from sklearn.grid_search import  GridSearchCV
from numpy import arange

svm_grid_params = dict(kernel = ['linear','poly','rbf','sigmoid'])

svm_grid = GridSearchCV(svm, svm_grid_params, cv=10, scoring='roc_auc', n_jobs = -1)
svm_grid.fit(explanatory_df, response_series)
svm_estimator = svm_grid.best_estimator_
print svm_estimator.kernel, svm_grid.best_score_
# rbf
# .9028
```
I found that the optimized kernel of rbf produced a ROC score of .9028.

####Comparing Support Vector Machines to Boosting Tree and Random Forests.
I wanted to see how well the optimized Support Vector Machines model performed as compared to the optimized Boosting Tree and Random Forest models.

```
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
```
The optimized Boosting Tree performed the best with a ROC score of .9400, while the Random Forest was second with a ROC score of .9346, as compared to the ROC score of .9028 for the Support Vector Machine.  The optimized Boosting Tree model is the best predicter of the data so far.

#### Testing the data
I then used the following code to import data from 2000 and later to test the accuracy of the Support Vector Machines, Boosting Tree, and Random Forest methods using non-PCA transformed data.  Again, I imported the data from the SQLite database, manipulated it, defined explanatory and response variables, and cleaned the data.  When filling Nans in the numeric data, I transformed the data using the fit from the training (pre 2000) data.

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
```

####Predicting Induction using SVM, Boosting Tree, and Random Forest

Finally, I tested the Support Vector Machine, Boosting Tree, and Random Forest models using the data from 2000 and later.

```
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
```
I found that, again, the Boosting Tree had the highest accuracy of the models with an accuracy equal to .8612, the Random Forest was second highest with an accuracy of .8576, and the Support Vector Machine was lowest with an accuracy of .8363.

####Confusion Matrices
I wanted compare the confusion matrices to see which models had the best sensitivity and specificity.

```
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
```
I calculated the specificity and sensitivity:

######Support Vector Machines

sensitivity = 33/(29 + 33) = 53%

specificity = 202/(202 + 17) = 92%

######Random Forests

sensitivity = 41/(21 + 41) = 66%

specificity = 200/(200 + 19) = 91%

######Boosting Tree

sensitivity = 42/(20 + 42) = 68%

specificity = 200/(200 + 19) = 91%

##Conclusions
This homework can be broken up into three parts: 1. comparing clustering methods,  2. testing the accuracy of Principal Component Analysis, and 3. comparing Support Vector Machines, Boosting Tree, and Random Forest.

1. Comparing clustering methods.
DBSCAN produced the clearest clusters, although KMeans also produced a similar cluster.  

2. Testing the accuracy of Principal Component Analysis
I used both the Boosting Tree and Support Vector Machines model to test whether Principal Component Analysis improved the accuracy of predicting Hall of Fame induction.  With both models, I found that the non-PCA transformed data was more accurate.  I would move forward without transforming the data with PCA.

3. Comparing Support Vector Machines, Boosting Tree, and Random Forest
I found using both my training data (pre2000) and testing data (post 2000) that the optimized Boosting Tree outperformed both Support Vector Machines and Random Forest.  In addition, it had the highest sensitivity of the three methods, although the Random Forest was close.  All three models had high specificity but moderate sensitivity.  Interestingly, the Support Vector Machines had the highest specificity of the three methods.  Therefore, I would make a decision on which model to use depending on whether I was more concerned with false positives or false negatives and the time needed to perform the analysis.  