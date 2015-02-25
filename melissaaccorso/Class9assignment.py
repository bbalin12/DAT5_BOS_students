# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 18:15:06 2015

@author: melaccor
"""

import pandas
import sqlite3

# putting a setting into pandas that lets you print out the entire
# DataFrame when you use the .head() method
pandas.set_option('display.max_columns', None)

# first, let's create a categorical feature that shows the dominant team 
# played per player
conn = sqlite3.connect('C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite')
query = 'select playerID, teamID from Batting'
player = pandas.read_sql(query, conn)
conn.close()

majority_team_by_player = player.groupby(['playerID']).agg(lambda x:x.value_counts().idxmax())

## write the data back to the database
conn = sqlite3.connect('C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite')
majority_team_by_player.to_sql('dominant_team_per_player', conn, if_exists = 'replace')
# closing the connection.
conn.close()


conn = sqlite3.connect('C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite')
query1 = 'select playerID, lgID from Batting'
lg = pandas.read_sql(query1, conn)
conn.close()

majority_league_by_player = lg.groupby(['playerID']).agg(lambda x:x.value_counts().idxmax())

conn = sqlite3.connect('C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite')
majority_league_by_player.to_sql('dominant_league_per_player', conn, if_exists = 'replace')
# closing the connection.
conn.close()

#query from last homework
conn=sqlite3.connect('C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite')
# open a cursor as we are executing a SQL statement that does not produce a pandas DataFrame
cur = conn.cursor()
# writing the query to simplify creating our response feature. 
sql = """
select a.playerID, a.inducted as inducted, batting.*, pitching.*, fielding.*, lg.*, player.* from
(select playerID, case when avginducted = 0 then 0 else 1 end as inducted from 
(select playerID, avg(case when inducted = 'Y' then 1 else 0 end ) as avginducted from HallOfFame 
where yearid < 2000
group by playerID)) a 
left outer join
(select playerID,  sum(AB) as atbats, sum(H) as totalhits, sum(R) as totalruns, sum(HR) as totalhomeruns, sum(SB) as stolenbases, sum(RBI) as totalRBI, sum(SO) as strikeouts, sum(IBB) as intentionalwalks
from Batting
group by playerID) batting on batting.playerID = a.playerID
left outer join(select playerID, sum(G) as totalgames, sum(SO) as shutouts, sum(sv) as totalsaves, sum(er) as earnedruns, sum(WP) as wildpitches
from Pitching
group by playerID) pitching on pitching.playerID = a.playerID 
left outer join
(select playerID, sum(InnOuts) as timewithouts, sum(PO) as putouts, sum(E) as errors, sum(DP) as doubleplays
from Fielding
group by playerID) fielding on fielding.playerID = a.playerID
left outer join
dominant_team_per_player player on player.playerID = a.playerID
left outer join 
dominant_league_per_player lg on lg.playerID = a.playerID;"""
df = pandas.read_sql(sql, conn)
# closing the connection.
conn.close()

## getting an intial view of the data for validation
df.head(10)
df.columns

# dropping duplicate playerID and strikeouts columns
df.drop('playerID',  1, inplace = True)

#############
## repeating the same preprocessing from the previous lesson
############

## splitting out the explanatory features 
explanatory_features = [col for col in df.columns if col not in ['playerID', 'inducted']]
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

### now, let's seperate the numeric explanatory data from the string data
string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']

######
# now, let's find if any of the categorical features need 'binnng'
#####
# first, fill the NANs in the feature (this lets us see if there are features
# that are all NANs, as they will show up as all 'Nothing' when we start binning or look for features with no variation)
string_features = string_features.fillna('Nothing')

# lets' create the heuristic that a level in the feature must exist in more than 1% of the training data to be retained. 
for col in string_features:
    # get the value_count of the column
    sizes = string_features[col].value_counts(normalize = True)
    # get the names of the levels that make up less than 1% of the dataset
    values_to_delete = sizes[sizes<0.01].index
    string_features[col].ix[string_features[col].isin(values_to_delete)] = "Other"

# let's verify if the replacement happened
string_features.teamID.value_counts(normalize = True)
#Other      0.080169
#NYA        0.070675
#PIT        0.067511
#CHN        0.059072
#SLN        0.054852
#PHI        0.052743
#NY1        0.051688
#CIN        0.048523
#DET        0.044304
#Nothing    0.042194
#BOS        0.041139
#CLE        0.041139
#BRO        0.037975
#PHA        0.036920
#WS1        0.030591
#CHA        0.029536
#LAN        0.028481
#BSN        0.027426
#BAL        0.023207
#HOU        0.022152
#SFN        0.018987
#NYN        0.016878
#SLA        0.013713
#KCA        0.013713
#CAL        0.012658
#MIN        0.012658
#ML4        0.010549
#MON        0.010549
#dtype: float64

string_features.lgID.value_counts(normalize = True)
#NL         0.530591
#AL         0.413502
#Nothing    0.042194
#Other      0.013713
#dtype: float64


## let's wrap that in a function for re-use 
def cleanup_data(df, cutoffPercent = .01):
    for col in df:
        sizes = df[col].value_counts(normalize = True)
        # get the names of the levels that make up less than 1% of the dataset
        values_to_delete = sizes[sizes<cutoffPercent].index
        df[col].ix[df[col].isin(values_to_delete)] = "Other"
    return df
##
######
## now, let's encode the categorical features.
######
# creating the 'catcher' data frame that will hold the encoded data
encoded_data = pandas.DataFrame(index = string_features.index)
for col in string_features.columns:
    ## calling pandas.get_dummies to turn the column into a sequene of 
    ## binary variables. Notice I'm using the 'prefix' feature to include the 
    ## original name of the column
    data = pandas.get_dummies(string_features[col], prefix=col.encode('ascii', 'replace'))
    encoded_data = pandas.concat([encoded_data, data], axis=1)

# let's verify that the encoding occured.
encoded_data.head()

## let's also wrap this into a function.
def get_binary_values(data_frame):
    """encodes cateogrical features in Pandas.
    """
    all_columns = pandas.DataFrame( index = data_frame.index)
    for col in data_frame.columns:
        data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pandas.concat([all_columns, data], axis=1)
    return all_columns

## now, let's fill the NANs in our numeric features.
# as before, let's impute using the mean strategy.
from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

## now that we've encoded our qualitative variables and filled the NaNs in our numeric variables, let's merge both DataFrames back together.

explanatory_df = pandas.concat([numeric_features, encoded_data],axis = 1)
explanatory_df.head()

#now, let's find features with no variance 
toKeep = []
toDelete = []
## loop through the DataFrame's columns
for col in explanatory_df:
    ## if the value_counts method returns more than one uniqe entity,
    ## append the column name to 'toKeep'
    if len(explanatory_df[col].value_counts()) > 1:
        toKeep.append(col)
    ## if not, append to 'toDelete'.
    else:
        toDelete.append(col)
# let's see if there's zero variance in an features
print toKeep
#['atbats', 'totalhits', 'totalruns', 'totalhomeruns', 'stolenbases', 'totalRBI', 'strikeouts', 'intentionalwalks', 'totalgames', 'shutouts', 'totalsaves', 'earnedruns', 'sum(so)', 'wildpitches', 'timewithouts', 'putouts', 'errors', 'doubleplays', u'lgID_AL', u'lgID_NL', 'lgID_Nothing', 'lgID_Other', u'teamID_BAL', u'teamID_BOS', u'teamID_BRO', u'teamID_BSN', u'teamID_CAL', u'teamID_CHA', u'teamID_CHN', u'teamID_CIN', u'teamID_CLE', u'teamID_DET', u'teamID_HOU', u'teamID_KCA', u'teamID_LAN', u'teamID_MIN', u'teamID_ML4', u'teamID_MON', u'teamID_NY1', u'teamID_NYA', u'teamID_NYN', 'teamID_Nothing', 'teamID_Other', u'teamID_PHA', u'teamID_PHI', u'teamID_PIT', u'teamID_SFN', u'teamID_SLA', u'teamID_SLN', u'teamID_WS1']
print toDelete
#['a.playerID_Other']

## let's wrap this into a function for future use. 
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
##

########
# now, let's look for columns with perfect correlation
#######

# first, let's create a correlation matrix diagram for the first 26 features.
toChart = explanatory_df.ix[:,0:25].corr()
toChart.head()

import matplotlib.pyplot as plt
import numpy
plt.pcolor(toChart)
plt.yticks(numpy.arange(0.5, len(toChart.index), 1), toChart.index)
plt.xticks(numpy.arange(0.5, len(toChart.columns), 1), toChart.columns, rotation=-90)
plt.colorbar()
plt.show()
# if you want to be audacious, try plotting the entire dataset.

# let's use an automated method to see what's perfectly correlated,
# either positively or negatively.
corr_matrix = explanatory_df.corr()
# substitude the entire matrix for a triangular matrix for faster
# computation
corr_matrix.ix[:,:] =  numpy.tril(corr_matrix.values, k = -1)
## create catcher objects to find lists of what is perfectly correlated
already_in = set()
result = []
for col in corr_matrix:
    perfect_corr = corr_matrix[col][abs(numpy.round(corr_matrix[col],10)) == 1.00].index.tolist()
    if perfect_corr and col not in already_in:
        already_in.update(set(perfect_corr))
        perfect_corr.append(col)
        result.append(perfect_corr)
# notice that throws R and throws L are perfectly correlated -- they should  be.
print result
#[['teamID_Nothing', 'lgID_Nothing']]

# creating a list of what to remove as all but the first column to appear
# in each correlation grouping.
toRemove = []
for item in result:
    toRemove.append(item[1:(len(item)+1)])
# flattenign the list of lists
toRemove = sum(toRemove, [])

#now, let's drop the columns we've identified from our explanatory features. 
explanatory_df.drop(toRemove, 1, inplace = True)

# let's combine all of this into a nice function.
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
###
    
##############
# scaling data
#############
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)

########
# Imputing missing values
#######
# recall that we used a 'mean' strategy for imputation before. This created some strange results for our values.  So, let's try out another method.
from sklearn.preprocessing import Imputer
## re-creating the numeric_features dataframe.
numeric_features = df.ix[:, df.dtypes != 'object']
## inputting the median observation
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), 
                                    columns = numeric_features.columns)


########
# Recursive feature elimination
#######
from sklearn.feature_selection import RFECV
from sklearn import tree

# create new class with a .coef_ attribute.
class TreeClassifierWithCoef(tree.DecisionTreeClassifier):
    def fit(self, *args, **kwargs):
        super(tree.DecisionTreeClassifier, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

# create your tree based estimator
decision_tree = TreeClassifierWithCoef(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)

## set up the estimator. Score by AUC
rfe_cv = RFECV(estimator=decision_tree, step=1, cv=10,
              scoring='roc_auc', verbose = 1)
rfe_cv.fit(explanatory_df, response_series)

print "Optimal number of features :{0} of {1} considered".format(rfe_cv.n_features_,len(explanatory_df.columns))
#Optimal number of features :4 of 49 considered

# printing out scores as we increase the number of features -- the farther down the list, the higher the number of features considered.
print rfe_cv.grid_scores_
#[ 0.71016985  0.72468872  0.73219391  0.78390853  0.78302037  0.77811519
#  0.78115837  0.77602666  0.7776255   0.77416709  0.7714305   0.76969123
#  0.76906841  0.76814386  0.77213296  0.76905743  0.76586066  0.77382746
#  0.76839747  0.76600654  0.77199388  0.77547426  0.77052507  0.77130055
#  0.77250771  0.76517396  0.76806145  0.76616247  0.76053706  0.76307671
#  0.76280097  0.76322745  0.7575083   0.77663112  0.76865233  0.75216851
#  0.77320439  0.77538867  0.75253823  0.76103865  0.77505282  0.75834188
#  0.757514    0.76883208  0.77124053  0.7578164   0.76844945  0.76673323
#  0.76369039]


## let's plot out the results
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (ROC_AUC)")
plt.plot(range(1, len(rfe_cv.grid_scores_) + 1), rfe_cv.grid_scores_)
plt.show()
# notice you could have just as well have included the 4 most important 
# features and received similar accuracy.

# you can pull out the features used this way:
features_used = explanatory_df.columns[rfe_cv.get_support()]
print features_used
#Index([u'atbats', u'totalruns', u'shutouts', u'teamID_Nothing'], dtype='object')

#you can extract the final selected model object this way:
final_estimator_used = rfe_cv.estimator_

# you can also combine RFE with grid search to find the tuning 
# parameters and features that optimize model accuracy metrics.
# do this by passing the RFECV object to GridSearchCV.
from sklearn.grid_search import  GridSearchCV

# doing this for a small range so I can show you the answer in a reasonable amount of time.
depth_range = range(4, 6)
# notice that in param_grid, I need to prefix estimator__ to my paramerters.
param_grid = dict(estimator__max_depth=depth_range)
# notice that this will take quite a bit longer to compute.
rfe_grid_search = GridSearchCV(rfe_cv, param_grid, cv=10, scoring='roc_auc')
rfe_grid_search.fit(explanatory_df, response_series)

print rfe_grid_search.grid_scores_
#[mean: 0.80810, std: 0.05151, params: {'estimator__max_depth': 4}, mean: 0.79594, std: 0.08245, params: {'estimator__max_depth': 5}]
rfe_grid_search.best_params_
#estimator__max_depth': 4

# let's plot out the results.
grid_mean_scores = [score[1] for score in rfe_grid_search.grid_scores_]

plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.plot(rfe_grid_search.best_params_['estimator__max_depth'], rfe_grid_search.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')
plt.grid(True)

# now let's pull out the winning estimator.
best_decision_tree_rfe_grid = rfe_grid_search.best_estimator_

features_used_rfecv_grid=explanatory_df.columns[best_decision_tree_rfe_grid.get_support()]


##########################
###  Test Data   ###
##########################

conn=sqlite3.connect('C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite')
# open a cursor as we are executing a SQL statement that does not produce a pandas DataFrame
cur = conn.cursor()
table_creation_query = """
CREATE TABLE hall_of_fame_inductees_post2000 as 
select playerID, yearid, category, case when average_inducted = 0 then 0 else 1 end as inducted from (
select playerID, yearid, category, avg(case when inducted = 'Y' then 1 else 0 end ) as average_inducted from  HallOfFame hf
where yearid > 2000
group by playerID
) bb; """
# executing the query
cur.execute(table_creation_query)
# closing the cursor
cur.close()
conn.close()

conn=sqlite3.connect('C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite')
cur = conn.cursor()
post2000data = '''select a.playerID, a.inducted as inducted, batting.*, pitching.*, fielding.*, lg.*, player.* from
(select playerID, case when avginducted = 0 then 0 else 1 end as inducted from 
(select playerID, avg(case when inducted = 'Y' then 1 else 0 end ) as avginducted from HallOfFame 
where yearid > 2000
group by playerID)) a 
left outer join
(select playerID,  sum(AB) as atbats, sum(H) as totalhits, sum(R) as totalruns, sum(HR) as totalhomeruns, sum(SB) as stolenbases, sum(RBI) as totalRBI, sum(SO) as strikeouts, sum(IBB) as intentionalwalks
from Batting
group by playerID) batting on batting.playerID = a.playerID
left outer join(select playerID, sum(G) as totalgames, sum(SO) as shutouts, sum(sv) as totalsaves, sum(er) as earnedruns, sum(WP) as wildpitches
from Pitching
group by playerID) pitching on pitching.playerID = a.playerID 
left outer join
(select playerID, sum(InnOuts) as timewithouts, sum(PO) as putouts, sum(E) as errors, sum(DP) as doubleplays
from Fielding
group by playerID) fielding on fielding.playerID = a.playerID
left outer join
dominant_team_per_player player on player.playerID = a.playerID
left outer join 
dominant_league_per_player lg on lg.playerID = a.playerID;'''
post2000data = pandas.read_sql(post2000data, conn)
conn.close()

post2000data.head(10)
post2000data.columns
post2000data.describe()

# dropping duplicate playerID columns
post2000data.drop('playerID',  1, inplace = True)

## splitting out the explanatory features 
explanatory_features2000 = [col for col in post2000data.columns if col not in ['playerID', 'inducted']]
explanatory_df2000 = post2000data[explanatory_features2000]

# dropping rows with no data.
explanatory_df2000.dropna(how='all', inplace = True) 

# extracting column names 
explanatory_colnames2000 = explanatory_df2000.columns

## doing the same for response
response_series2000 = post2000data.inducted
response_series2000.dropna(how='all', inplace = True) 

## seeing which explanatory feature rows got removed.  Looks like none.
response_series2000.index[~response_series2000.index.isin(explanatory_df2000.index)]

### now, let's seperate the numeric explanatory data from the string data
string_features2000 = explanatory_df2000.ix[:, explanatory_df2000.dtypes == 'object'] # using the indexes only taking columns where data type is an object, : means all the rows
numeric_features2000 = explanatory_df2000.ix[:, explanatory_df2000.dtypes != 'object']


string_features2000 = string_features2000.fillna('Nothing')

for col in string_features2000:
    # get the value_count of the column
    sizes = string_features2000[col].value_counts(normalize = True)
    # get the names of the levels that make up less than 1% of the dataset
    values_to_delete = sizes[sizes<0.01].index
    string_features2000[col].ix[string_features2000[col].isin(values_to_delete)] = "Other"
    
string_features2000.teamID.value_counts(normalize = True)
post2000data.teamID.value_counts(normalize=True)

string_features2000.teamID.value_counts(normalize = True)
#Nothing    0.091255
#TOR        0.049430
#SLN        0.049430
#NYN        0.045627
#SFN        0.045627
#Other      0.045627
#DET        0.041825
#OAK        0.041825
#NYA        0.041825
#CIN        0.038023
#TEX        0.038023
#MIN        0.034221
#BAL        0.030418
#ATL        0.030418
#BOS        0.030418
#CHA        0.030418
#LAN        0.030418
#MON        0.030418
#PIT        0.030418
#CAL        0.026616
#CHN        0.026616
#CLE        0.026616
#PHI        0.026616
#KCA        0.022814
#HOU        0.022814
#SEA        0.022814
#SDN        0.019011
#COL        0.015209
#ML4        0.015209
string_features2000.lgID.value_counts(normalize=True)
#AL         0.456274
#NL         0.452471
#Nothing    0.091255


## Prior to binning need to check to see if unique features are included
## here that are not in pre-2000 data
unique_teamID = list(set(e for e in string_features.teamID))
unique_teamID_post2000 = list(set(e for e in string_features2000.teamID))
extra_teams = list(set(e for e in unique_teamID_post2000 if e not in unique_teamID))

# Check string features post 2000
string_features.teamID.index[~string_features2000.index.isin(unique_teamID)]
# There are a few features that do not occur in the pre-2000 dataset

## Replace these with other
string_features2000.teamID.replace(to_replace=extra_teams,value = 'Other',inplace=True)

encoded_data2000 = get_binary_values(string_features2000)

encoded_data2000.reset_index(inplace=True)

########
# Imputing missing values
#######

## re-creating the numeric_features dataframe.
numeric_features2000 = post2000data.ix[:, post2000data.dtypes != 'object']
## inputting the median observation
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features2000)
numeric_features2000 = pandas.DataFrame(imputer_object.transform(numeric_features2000), 
                                    columns = numeric_features2000.columns)
                    
##Merging the two datasets
explanatory_df2000 = pandas.concat([numeric_features2000, encoded_data2000],axis = 1)
explanatory_df2000.head()

#Keeping only variables selected earlier
explanatory_df20002 = [col for col in explanatory_df2000.columns if col in ['atbats', 'totalruns', 'shutouts', 'teamID_Nothing']] 
df2000_slim = explanatory_df2000[explanatory_df20002]


#scaling data
scaler = preprocessing.StandardScaler()
scaler.fit(df2000_slim)
df2000_slim = pandas.DataFrame(scaler.transform(df2000_slim), columns = df2000_slim.columns)
df2000_slim.describe()

# Predict values for the output
predicted_values = final_estimator_used.predict(df2000_slim)

# Create a confusion matrix to examine the results
cm = pandas.crosstab(response_series2000,predicted_values,rownames=['True Label'], colnames = ['Predicted Label'],margins = True)
                     
print cm
#Predicted Label    0    1  All
#True Label                    
#0                131   75  206
#1                 22   35   57
#All              153  110  263

# Calculate cross val accuracy scores
from sklearn.cross_validation import cross_val_score as cv
accuracy_scores_best_OOS = cv(final_estimator_used, df2000_slim, response_series2000, cv=10, scoring='accuracy')

accuracy_scores_best_oldData = cv(final_estimator_used, explanatory_df, response_series, cv=10, scoring='accuracy')

print accuracy_scores_best_OOS.mean()
#Accuracy of 81% 
print accuracy_scores_best_oldData.mean()
#Accuracy of 80% 
