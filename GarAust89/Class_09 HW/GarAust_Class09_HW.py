# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 19:27:20 2015

@author: garauste
"""

import pandas
import sqlite3
from sklearn.feature_selection import RFECV
from sklearn import tree
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy
from sklearn.cross_validation import cross_val_score

# Putting a setting into pandas that lets you print out the entire 
# Datframe when you use the .head() method
pandas.set_option('display.max_columns',None)

# Connecting to database and pull data
conn = sqlite3.connect('C:\Users\garauste\Documents\SQLite\lahman2013.sqlite')
sql = '''
select coalesce(a.nameFirst ,"") || coalesce(a.nameLast,"") as PlayerName, hfi.inducted,
a.playerID, a.birthState, b.*, c.*,d.*
from hall_of_fame_inductees hfi
left outer join master a
on a.playerID = hfi.playerID
left outer join 
(
select sum(b.G) as Games, sum(b.H) as Hits, sum(b.AB) as At_Bats, sum(b.HR) as Homers,
b.playerID from Batting b group by b.playerID
) b
on hfi.playerID = b.playerID
left outer join 
(
select d.playerID, sum(d.A) as Fielder_Assists, sum(d.E) as Field_Errors, 
sum(d.DP) as Double_Plays from Fielding d group by d.playerID 
) c
on hfi.playerID = c.playerID
left outer join
(
select b.playerID, sum(b.ERA) as Pitcher_Earned_Run_Avg, sum(b.W) as Pitcher_Ws, sum(b.SHO) as Pitcher_ShutOuts, sum(b.SO) as Pitcher_StrikeOuts,
sum(b.HR) as HR_Allowed, sum(b.CG) as Complete_Games 
from Pitching b 
group by b.playerID
) d 
on hfi.playerID = d.playerID;
'''

df = pandas.read_sql(sql, conn)
conn.close()

df.head(10)
df.columns

# Dropping duplicate playerIDs columns
df.drop('playerID',1,inplace=True)

df_colnames = df.columns

###################################################################
######## Importing Functions created in inclass exercise ##########
###################################################################

# Function to cut off data that has less than 1% of all volume
def cleanup_data(df, cutoffPercent = .01):
    for col in df:    
        sizes = df[col].value_counts(normalize=True) # normalize = True gives percentages
        # get the names of the levels that make up less than 1% of the dataset
        values_to_delete = sizes[sizes<cutoffPercent].index
        df[col].ix[df[col].isin(values_to_delete)] = 'Other'
    return df

# Function to create binary dummies for catergorical data    
def get_binary_values(data_frame):
    """encodes the categorical features in Pandas
    """
    all_columns = pandas.DataFrame(index=data_frame.index)
    for col in data_frame.columns:
        data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii','replace'))
        all_columns = pandas.concat([all_columns,data],axis=1)
    return all_columns

# Function to find variables with no variance at all - Need to Impute before this step
def find_zero_var(df):
    """ find the columns in the dataframe with zero variance -- ie those 
        with the same value in every observation
    """
    toKeep = []
    toDelete = []
    for col in df:
        if len(df[col].value_counts())>1:
            toKeep.append(col)
        else:
            toDelete.append(col)
    #
    return {'toKeep':toKeep, 'toDelete':toDelete}
    
# Function to find the variables with perfect correlation
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
        
##############################################################
########### Preprocessing the Data and splitting #############
##############################################################
        
## Split out data by explanatory features 
explan_features = [col for col in df_colnames if col not in ('PlayerName','inducted')]
df_explanatory = df[explan_features]

# drop rows 
df_explanatory.dropna(how='all',inplace=True)

# create resposne variable
response_series = df.inducted
response_series.dropna(how='all',inplace=True)

# seeing which explanatory feature rows got removed 
response_series.index[~response_series.index.isin(df_explanatory.index)]

# Create Categorical and Numeric Datasets
string_features = df_explanatory.ix[:,df_explanatory.dtypes == 'object']
numeric_features = df_explanatory.ix[:,df_explanatory.dtypes != 'object']

# Determine if categorical data requires binning
string_features = string_features.fillna('Nothing')

# Use function defined earlier to clean the data
string_features = cleanup_data(string_features)

# let's verify if the replacement happened
string_features.birthState.value_counts(normalize = True)

# Let's encode the categorical variables
# use function 

encoded_data = get_binary_values(string_features)

#######
## Now to check the numeric features
#######

# Use imputed to fill missing values
imputer_object = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features),
                                    columns = numeric_features.columns)

# Merging encoded and numeric                                    
explanatory_df = pandas.concat([numeric_features, encoded_data],axis=1)
explanatory_df.head()

# Now find the features with zero variance
no_var_features = find_zero_var(explanatory_df)
# No Features have zero variance

# Create a correlation matrix to check data
toChart = explanatory_df.ix[:,0:13].corr()
toChart.head()
plt.pcolor(toChart)
plt.yticks(numpy.arange(0.5, len(toChart.index),1),toChart.index)
plt.xticks(numpy.arange(0.5, len(toChart.columns),1),toChart.columns,rotation = -90)
plt.colorbar()
plt.show()

## Find the perfectly correlated features 
perfect_corr = find_perfect_corr(df_explanatory)
# no features are perfectly correlated


########
## Time to scale data
########

scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df),columns = 
explanatory_df.columns)

# Imputing the missing values using the median
numeric_features= df.ix[:,df.dtypes != 'object']
# imputting the median observation
imputer_object = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features),
                                    columns = numeric_features.columns)
                                  
########
# Recursive Feature Elimination
########
class TreeClassifierWithCoef(tree.DecisionTreeClassifier):
   def fit(self, *args, **kwargs):
       super(tree.DecisionTreeClassifier, self).fit(*args, **kwargs)
       self.coef_ = self.feature_importances_
       
# create your tree based estimator
decision_tree = TreeClassifierWithCoef(criterion = 'gini', splitter = 'best',
max_features=None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2,
max_leaf_nodes = None, random_state = 1)

## set up the estimator. Score by AUC
rfe_cv = RFECV(estimator = decision_tree, step = 1, cv = 10, scoring = 'roc_auc',verbose = 1)
rfe_cv.fit(explanatory_df, response_series)

print "Optimal number of features: {0} of {1} considered".format(rfe_cv.n_features_,
len(explanatory_df.columns))

print rfe_cv.grid_scores_

# let's plot out the results
plt.figure()
plt.xlabel('Number of Features selected')
plt.ylabel('Cross Validation score (ROC_AUC)')
plt.plot(range(1, len(rfe_cv.grid_scores_)+1),rfe_cv.grid_scores_)
plt.show()

# you can pull out the features used this way: 
features_used = explanatory_df.columns[rfe_cv.get_support()]
print features_used

# you can extract the final selected model object his way
final_estimator_used = rfe_cv.estimator_

# you can also combine RFE with grid search to find the tuning 
# parameters and features that optimize model accuracy metrics
# do this by passing the RFECV object to GridSearchCV
from sklearn.grid_search import GridSearchCV

depth_range = range(4,10)
# notice that in paramgrid need prefix estimator
param_grid = dict(estimator__max_depth=depth_range)
# notice that this will take quite a bit longer to compute
rfe_grid_search = GridSearchCV(rfe_cv, param_grid, cv = 10, scoring = 'roc_auc')
rfe_grid_search.fit(explanatory_df, response_series)

print rfe_grid_search.grid_scores_
rfe_grid_search.best_params_

# let's plot the results
grid_mean_scores = [score[1] for score in rfe_grid_search.grid_scores_]

plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.plot(rfe_grid_search.best_params_['estimator__max_depth'],
         rfe_grid_search.best_score_,'ro',markersize=12, markeredgewidth=1.5,
         markerfacecolor='None',markeredgecolor='r')
plt.grid(True)

# now let's pull out the winning estimator
best_decision_tree_rfe_grid = rfe_grid_search.best_estimator_

print best_decision_tree_rfe_grid

features_used_rfecv_grid = explanatory_df.columns[best_decision_tree_rfe_grid.get_support()]

print features_used_rfecv_grid

predicted_values_old = best_decision_tree_rfe_grid.predict(explanatory_df)

cm = pandas.crosstab(response_series,predicted_values_old,rownames=['True Label'],
                     colnames = ['Predicted Label'],margins = True)
                     
print cm

## Testing cross val accuracy score
accuracy_scores_best_oldData = cross_val_score(best_decision_tree_rfe_grid,
explanatory_df, response_series, cv=10, scoring='accuracy')

print accuracy_scores_best_oldData.mean()


####################################################
############# Out of Sample Testing ################
####################################################

conn = sqlite3.connect('C:\Users\garauste\Documents\SQLite\lahman2013.sqlite')
cur = conn.cursor()

## writing a query to simply creating our repsonse feature. 
# notice I ahve to aggregate at the player level as players can be entered into voting
# for numerous years in a row
table_creation_query = """
CREATE TABLE hall_of_fame_inductees_post2000 as  

select playerID, case when average_inducted = 0 then 0 else 1 end as inducted from (

select playerID, avg(case when inducted = 'Y' then 1 else 0 end ) as average_inducted from  HallOfFame hf
where yearid > 2000
group by playerID

) bb; """

# executing the query
cur.execute(table_creation_query)
# closing the cursor
cur.close()
conn.close()    

conn = sqlite3.connect('C:\Users\garauste\Documents\SQLite\lahman2013.sqlite')
# new query to pull data post 2000 
sql = '''
select coalesce(a.nameFirst ,"") || coalesce(a.nameLast,"") as PlayerName, hfi.inducted,
a.playerID, a.birthState, b.*, c.*,d.*
from hall_of_fame_inductees_post2000 hfi
left outer join master a
on a.playerID = hfi.playerID
left outer join 
(
select sum(b.G) as Games, sum(b.H) as Hits, sum(b.AB) as At_Bats, sum(b.HR) as Homers,
b.playerID from Batting b group by b.playerID
) b
on hfi.playerID = b.playerID
left outer join 
(
select d.playerID, sum(d.A) as Fielder_Assists, sum(d.E) as Field_Errors, 
sum(d.DP) as Double_Plays from Fielding d group by d.playerID 
) c
on hfi.playerID = c.playerID
left outer join
(
select b.playerID, sum(b.ERA) as Pitcher_Earned_Run_Avg, sum(b.W) as Pitcher_Ws, sum(b.SHO) as Pitcher_ShutOuts, sum(b.SO) as Pitcher_StrikeOuts,
sum(b.HR) as HR_Allowed, sum(b.CG) as Complete_Games 
from Pitching b 
group by b.playerID
) d 
on hfi.playerID = d.playerID;
'''

df_post2000 = pandas.read_sql(sql,conn)
conn.close()

# Dropping duplicate playerIDs columns
df_post2000.drop('playerID',1,inplace=True)
df_post_colnames = df_post2000.columns


## Split out data by explanatory features 
explan_features_post2000 = [col for col in df_post_colnames if col not in ('PlayerName','inducted')]
df_explanatory_post2000 = df_post2000[explan_features_post2000]

# drop rows 
df_explanatory_post2000.dropna(how='all',inplace=True)

# create resposne variable
response_series_post2000 = df_post2000.inducted
response_series_post2000.dropna(how='all',inplace=True)

# seeing which explanatory feature rows got removed 
response_series_post2000.index[~response_series_post2000.index.isin(df_explanatory_post2000.index)]

# Create Categorical and Numeric Datasets
string_features_post2000 = df_explanatory_post2000.ix[:,df_explanatory_post2000.dtypes == 'object']
numeric_features_post2000 = df_explanatory_post2000.ix[:,df_explanatory_post2000.dtypes != 'object']

# Determine if categorical data requires binning
string_features_post2000 = string_features_post2000.fillna('Nothing')

# Use function defined earlier to clean the data
string_features_post2000 = cleanup_data(string_features_post2000)

# let's verify if the replacement happened
string_features_post2000.birthState.value_counts(normalize = True)

# Let's encode the categorical variables
# use function 

## Prior to binning need to check to see if unique features are included
## here that are not in pre-2000 data
unique_states = list(set(e for e in string_features.birthState))
unique_states_post2000 = list(set(e for e in string_features_post2000.birthState))
extra_states = list(set(e for e in unique_states_post2000 if e not in unique_states))

# Check string features post 2000
string_features_post2000.birthState.index[~string_features_post2000.index.isin(unique_states)]
# There are a few features that do not occur in the pre-2000 dataset

## Replace these with other
string_features_post2000.birthState.replace(to_replace=extra_states,value = 'Other',inplace=True)

# Creating Dummy Categorical variables
encoded_data_post2000 = get_binary_values(string_features_post2000)

# Resetting the index
encoded_data_post2000.reset_index(inplace=True)

## Removing extra columns introduced by reset index function
del encoded_data_post2000['index']
del encoded_data_post2000['level_0']


#######
## Now to check the numeric features
#######

# Use imputed to fill missing values
imputer_object = Imputer(missing_values = 'NaN', strategy = 'median', axis=0)
imputer_object.fit(numeric_features_post2000)
numeric_features_post2000 = pandas.DataFrame(imputer_object.transform(numeric_features_post2000),
                                    columns = numeric_features_post2000.columns)

# Merging encoded and numeric                                    
new_df_post_2000 = pandas.concat([numeric_features_post2000, encoded_data_post2000],axis=1)
new_df_post_2000.head()

# Check to see if there are any columns missing from the new data set that are in the old data
missing_columns=[e for e in explanatory_df.columns if e not in new_df_post_2000.columns]
other_columns = [e for e in new_df_post_2000.columns if e not in explanatory_df.columns]

# Create dummies for missing columns 
for e in missing_columns:
    new_df_post_2000[e] = 0

# Scale the data 
new_df_post_2000= pandas.DataFrame(scaler.transform(new_df_post_2000),columns = 
new_df_post_2000.columns)

# Predict values for the output
predicted_values = best_decision_tree_rfe_grid.predict(new_df_post_2000)

# Create a confusion matrix to examine the results
cm = pandas.crosstab(response_series_post2000,predicted_values,rownames=['True Label'],
                     colnames = ['Predicted Label'],margins = True)
                     
print cm

# Calculate cross val accuracy scores
accuracy_scores_best_OOS = cross_val_score(best_decision_tree_rfe_grid,
new_df_post_2000, response_series_post2000, cv=10, scoring='accuracy')

print accuracy_scores_best_OOS.mean()
print accuracy_scores_best_oldData.mean()