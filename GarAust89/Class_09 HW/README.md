# Gareth Austen Homework for Class 09 - Data Manipulation

#### Introduction
I pulled data from the SQLite database using the below query. I decided to pull in the 
state that each player was born in to determine if it had any impact on whether a player 
was inducted into the Hall of Fame. 

```
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
select b.playerID, sum(b.ERA) as Pitcher_Earned_Run_Avg, sum(b.W) as Pitcher_Ws,
sum(b.SHO) as Pitcher_ShutOuts, sum(b.SO) as Pitcher_StrikeOuts,
sum(b.HR) as HR_Allowed, sum(b.CG) as Complete_Games 
from Pitching b 
group by b.playerID
) d 
on hfi.playerID = d.playerID;
```

In my previous homeworks I had split the data out by Pitchers and Batters. However this is very
time consuming and therefore I have to re-combine the datasets using the above sql query. 

#### Importing the Functions create in class

The next step was to reinstiate the functions that we had created during class to clean and 
manipulate the data. The following code segment was used:

```
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
    """finds columns that are eother positively or negatively 
    perfectly correlated (with correlations of +1 or -1), and creates a dict 
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

#### Cleaning and Manipulating the Data

Next we cleaned and manipulated the data using the functions listed above. We found that no variables were perfectly correlated and there were no variables with zero variance. 

However, some of the features show quite high correlation amongst the explanatory variables. 
This can be seen in the Heat Map chart below:

![CorrelationHeatMap](https://github.com/GarAust89/DAT5_BOS_students/blob/master/GarAust89/Class_09%20HW/CorrelationHeatMap.png)

We can see from this heat map that the variables selected from the batting table are highly correlated as are the variables selected from the pitching table. There is little or no correlation between batting and pitching statistics which is unsurprising.

#### Recursive Feature Elimination and Cross-Validation (RFECV)

Next we used RFECV to determine the optimal number of features to be used. This was achieved using the 
following code: 
```
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
```

The below plot clearly illustrates the optimal number of features. 

![DecisionTreeOptimalFeatureSelection](https://github.com/GarAust89/DAT5_BOS_students/blob/master/GarAust89/Class_09%20HW/DecisionTreeOptimalFeatureSelection.png)

#### Combining RFECV with GridSearchCV 

We combine RFECV with GridSearchCV to fine tune our model and obtain the optimal parameters. For this
process we used a decision tree depth range of 2-10.

```
from sklearn.grid_search import GridSearchCV

depth_range = range(2,10)
# notice that in paramgrid need prefix estimator
param_grid = dict(estimator__max_depth=depth_range)
# notice that this will take quite a bit longer to compute
rfe_grid_search = GridSearchCV(rfe_cv, param_grid, cv = 10, scoring = 'roc_auc')
rfe_grid_search.fit(explanatory_df, response_series)

print rfe_grid_search.grid_scores_
rfe_grid_search.best_params_
```

This leads to the following plot: 

![GridSearchOptimalFeatureSelection](https://github.com/GarAust89/DAT5_BOS_students/blob/master/GarAust89/Class_09%20HW/GridSearchOptimalFeatureSelection.png)

In this case the optimal decision tree depth is 4. We extract the optimal parameters and test the models
accuracy using cross-validation and create a confusion matrix.

```
Predicted Label    0    1  All
True Label                    
0                684   20  704
1                 98  146  244
All              782  166  948
```

Overall Accuracy : (146+684)/(98+20+146+684) = 88%

Sesnsitivity: (146)/(146+98) = 59%

Using the following code we predict accuracy: 
```
accuracy_scores_best_oldData = cross_val_score(best_decision_tree_rfe_grid,
explanatory_df, response_series, cv=10, scoring='roc_auc')

print accuracy_scores_best_oldData.mean()
```

The model has an accuracy rating of 81% which is rated as a good score under the following guidelines:
``` 
# .90 -1= excellent
# .8 - .9 = good
# .7 - .8 = fair
# .6 - .7 = poor 
# .5 - .6 = fail
```

However the poor sensitivity result is a cause of concern as we are trying to predict whether a player will
be inducted into the hall of fame or not. 

## Out-Of-Sample Testing

