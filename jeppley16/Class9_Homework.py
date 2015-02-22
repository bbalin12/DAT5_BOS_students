# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 16:27:31 2015

@author: jeppley
"""
##########################
###  HOMEWORK: Class 9 ###
##########################


import pandas as pd
import sqlite3 as sql
from sklearn.preprocessing import Imputer as imp
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing as pre
from sklearn.feature_selection import RFECV
from sklearn import tree
from sklearn.grid_search import  GridSearchCV
from sklearn.cross_validation import cross_val_score as cv


pd.set_option('display.max_columns', None)


##########################
###  Extracting Data   ###
##########################


## Creating categorical feature for team position ##

con = sql.connect('C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite')
query = 'select playerID, POS from Fielding'
dfhw = pd.read_sql(query, con)
con.close()

# use pandas.DataFrame.groupby and an annonymous lambda function
# to pull the mode position for each player
majority_pos_by_player = dfhw.groupby(['playerID']).agg(lambda x:x.value_counts().idxmax())

## write the data back to the database
conn = sql.connect('C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite')
majority_pos_by_player.to_sql('dominant_pos_per_player', conn, if_exists = 'replace')
conn.close()

con = sql.connect('C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite')

modeldata = '''select h.*, 
  b.b_atbat, b.b_hits, p.p_wins, f.f_puts, t.teamID, o.pos
  from 
  (select playerid, inducted
  from hall_of_fame_inductees_3 
   where category = 'Player'
   group by playerid) h
left outer join 
  (select playerid,
    count(distinct yearid) as b_years,
    sum(ab) as b_atbat, 
    sum(r) as b_runs, 
    sum(h) as b_hits, 
    sum(hr) as b_hruns, 
    sum(sb) as b_stbas,
    sum(so) as b_strik
  from batting
  group by playerid 
  HAVING max(yearID) > 1950 and min(yearID) >1950 ) b
  on h.playerid = b.playerid
left outer join
  (select playerid,
    count(distinct yearid) as p_years,
    sum(w) as p_wins,
    sum(l) as p_loss,
    sum(sho) as p_shout,
    sum(sv) as p_saves,
    sum(er) as p_eruns,
    sum(so) as p_stout
  from pitching
  group by playerid) p
  on h.playerid = p.playerid
left outer join
  (select playerid,
    max(teamID) as teamID
  from dominant_team_per_player
  group by playerid) t
  on h.playerid = t.playerid
left outer join
  (select playerid,
    max(POS) as pos
  from dominant_pos_per_player
  group by playerid) o
  on h.playerid = o.playerid    
left outer join
  (select playerid,
     count(distinct yearid) as f_years,
     sum(po) as f_puts,
     sum(a) as f_assis,
     sum(dp) as f_dplay,
     sum(pb) as f_pass
  from fielding
  group by playerid) f
  on h.playerid = f.playerid
  where b.playerID is not null
;'''


dfhw = pd.read_sql(modeldata, con)
con.close()

dfhw.head(10)
dfhw.columns
dfhw.describe()


# dropping duplicate playerID columns
dfhw.drop('playerid',  1, inplace = True)


##########################
###  Processing Data   ###
##########################


## splitting out the explanatory features 
explanatory_featureshw = [col for col in dfhw.columns if col not in ['nameGiven', 'inducted']]
explanatory_dfhw = dfhw[explanatory_featureshw]

# dropping rows with no data.



explanatory_dfhw.dropna(how='all', inplace = True) 

# extracting column names 
explanatory_colnameshw = explanatory_dfhw.columns

## doing the same for response
response_serieshw = dfhw.inducted
response_serieshw.dropna(how='all', inplace = True) 

## seeing which explanatory feature rows got removed.  Looks like none.
response_serieshw.index[~response_serieshw.index.isin(explanatory_dfhw.index)]

### now, let's seperate the numeric explanatory data from the string data
string_featureshw = explanatory_dfhw.ix[:, explanatory_dfhw.dtypes == 'object'] # , using the indexes only taking columns where data type is an object, : means all the rows
numeric_featureshw = explanatory_dfhw.ix[:, explanatory_dfhw.dtypes != 'object']

string_featureshw = string_featureshw.fillna('Nothing')

for col in string_featureshw:
    # get the value_count of the column
    sizes = string_featureshw[col].value_counts(normalize = True)
    # get the names of the levels that make up less than 1% of the dataset
    values_to_delete = sizes[sizes<0.01].index
    string_featureshw[col].ix[string_featureshw[col].isin(values_to_delete)] = "Other"
    
string_featureshw.teamID.value_counts(normalize = True)
dfhw.teamID.value_counts(normalize=True)

string_featureshw.pos.value_counts(normalize = True)
dfhw.pos.value_counts(normalize=True)

## ENCODING CATEGORICAL FEATURES ##

# creating the 'catcher' data frame that will hold the encoded data
encoded_data = pd.DataFrame(index = string_featureshw.index)
for col in string_featureshw.columns:
    ## calling pandas.get_dummies to turn the column into a sequene of 
    ## binary variables. Notice I'm using the 'prefix' feature to include the 
    ## original name of the column
    data = pd.get_dummies(string_featureshw[col], prefix=col.encode('ascii', 'replace'))
    encoded_data = pd.concat([encoded_data, data], axis=1)

encoded_data.head()

## now, let's fill the NANs in our nuemeric features.
# as before, let's impute using the mean strategy.

imputer_object = imp(missing_values='NaN', strategy='mean', axis=0)
imputer_object.fit(numeric_featureshw) #extracts means from all columns, doesn't transform until transform feature is ran on it #fit on original data, and then pull in new data, will use the mean of the original model
numeric_featureshw = pd.DataFrame(imputer_object.transform(numeric_featureshw), columns = numeric_featureshw.columns)


explanatory_dfhw = pd.concat([numeric_featureshw, encoded_data],axis = 1)
explanatory_dfhw.head()

##########################
###  Selecting Data   ###
##########################


#Remove features with perfect correlation and/or no variation.

toKeep = []
toDelete = []
## loop through the DataFrame's columns
for col in explanatory_dfhw:
    ## if the value_counts method returns more than one uniqe entity,
    ## append the column name to 'toKeep'
    if len(explanatory_dfhw[col].value_counts()) > 1:
        toKeep.append(col)
    ## if not, append to 'toDelete'.
    else:
        toDelete.append(col)

# let's see if there's zero variance in an features
print toKeep
print toDelete

########
# now, let's look for columns with perfect correlation
#######

def find_perfect_corr(df):
    """finds columns that are eother positively or negatively perfectly correlated (with correlations of +1 or -1), and creates a dict 
        that includes which columns to drop so that each remaining column
        is independent
    """  
    corrMatrix = df.corr()
    corrMatrix.loc[:,:] =  np.tril(corrMatrix.values, k = -1)
    already_in = set()
    result = []
    for col in corrMatrix:
        perfect_corr = corrMatrix[col][abs(np.round(corrMatrix[col],10)) == 1.00].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    toRemove = []
    for item in result:
        toRemove.append(item[1:(len(item)+1)])
    toRemove = sum(toRemove, [])
    return {'corrGroupings':result, 'toRemove':toRemove}
    
find_perfect_corr(explanatory_dfhw)


##Scale your data and impute for your numeric NaNs

scaler = pre.StandardScaler()
scaler.fit(explanatory_dfhw)
explanatory_dfhw = pd.DataFrame(scaler.transform(explanatory_dfhw), columns = explanatory_dfhw.columns)
explanatory_dfhw.describe()

########
# Imputing missing values
#######

## re-creating the numeric_features dataframe.
numeric_featureshw = dfhw.ix[:, dfhw.dtypes != 'object']
## inputting the median observation
imputer_object = imp(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_featureshw)
numeric_featureshw = pd.DataFrame(imputer_object.transform(numeric_featureshw), 
                                    columns = numeric_featureshw.columns)


##Perform recursive feature elimination on the data.

class TreeClassifierWithCoef(tree.DecisionTreeClassifier): #create an attribute called coefficient, object oriented python
    def fit(self, *args, **kwargs):
        super(tree.DecisionTreeClassifier, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

# create your tree based estimator
decision_tree = TreeClassifierWithCoef(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)

## set up the estimator. Score by AUC
rfe_cvhw = RFECV(estimator=decision_tree, step=1, cv=10, #step, how many features at each step, cv amount of cross validation, scoring is how we'll determine which set of features is most accurate
              scoring='roc_auc', verbose = 1)
rfe_cvhw.fit(explanatory_dfhw, response_serieshw)

print "Optimal number of features :{0} of {1} considered".format(rfe_cvhw.n_features_,len(explanatory_dfhw.columns))

# printing out scores as we increase the number of features -- the farter
# down the list, the higher the number of features considered.
print rfe_cvhw.grid_scores_



## let's plot out the results
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (ROC_AUC)")
plt.plot(range(1, len(rfe_cvhw.grid_scores_) + 1), rfe_cvhw.grid_scores_)
plt.show()

features_used = explanatory_dfhw.columns[rfe_cvhw.get_support()]
print features_used


#you can extract the final selected model object this way:
final_estimator_used = rfe_cvhw.estimator_

#grid search likely won't make a difference in my model with the few variables that exist and
#the complete loss of any additional accuracy after those few features


##########################
###  Test Data   ###
##########################


conn = sql.connect('C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite')
cur = conn.cursor()

## writing a query to simply creating our repsonse feature. 

table_creation_query = """
CREATE TABLE hall_of_fame_inductees_post20003 as 
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

con = sql.connect('C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite')


post2000data = '''select h.*, 
  b.b_atbat, b.b_hits, p.p_wins, f.f_puts, t.teamID, o.pos
  from 
  (select playerid, inducted
  from hall_of_fame_inductees_post20003 
   where category = 'Player'
   group by playerid) h
left outer join 
  (select playerid,
    count(distinct yearid) as b_years,
    sum(ab) as b_atbat, 
    sum(r) as b_runs, 
    sum(h) as b_hits, 
    sum(hr) as b_hruns, 
    sum(sb) as b_stbas,
    sum(so) as b_strik
  from batting
  group by playerid 
  HAVING max(yearID) > 1950 and min(yearID) >1950 ) b
  on h.playerid = b.playerid
left outer join
  (select playerid,
    count(distinct yearid) as p_years,
    sum(w) as p_wins,
    sum(l) as p_loss,
    sum(sho) as p_shout,
    sum(sv) as p_saves,
    sum(er) as p_eruns,
    sum(so) as p_stout
  from pitching
  group by playerid) p
  on h.playerid = p.playerid
left outer join
  (select playerid,
    max(teamID) as teamID
  from dominant_team_per_player
  group by playerid) t
  on h.playerid = t.playerid
left outer join
  (select playerid,
    max(POS) as pos
  from dominant_pos_per_player
  group by playerid) o
  on h.playerid = o.playerid    
left outer join
  (select playerid,
     count(distinct yearid) as f_years,
     sum(po) as f_puts,
     sum(a) as f_assis,
     sum(dp) as f_dplay,
     sum(pb) as f_pass
  from fielding
  group by playerid) f
  on h.playerid = f.playerid
  where b.playerID is not null
;'''

df2000 = pd.read_sql(post2000data, con)
con.close()

df2000.head(10)
df2000.columns
df2000.describe()


# dropping duplicate playerID columns
df2000.drop('playerid',  1, inplace = True)

## splitting out the explanatory features 
explanatory_features2000 = [col for col in df2000.columns if col not in ['nameGiven', 'inducted']]
explanatory_df2000 = df2000[explanatory_features2000]

# dropping rows with no data.
explanatory_df2000.dropna(how='all', inplace = True) 

# extracting column names 
explanatory_colnames2000 = explanatory_df2000.columns

## doing the same for response
response_series2000 = df2000.inducted
response_series2000.dropna(how='all', inplace = True) 

## seeing which explanatory feature rows got removed.  Looks like none.
response_series2000.index[~response_series2000.index.isin(explanatory_df2000.index)]

### now, let's seperate the numeric explanatory data from the string data
string_features2000 = explanatory_df2000.ix[:, explanatory_df2000.dtypes == 'object'] # , using the indexes only taking columns where data type is an object, : means all the rows
numeric_features2000 = explanatory_df2000.ix[:, explanatory_df2000.dtypes != 'object']


string_features2000 = string_features2000.fillna('Nothing')

for col in string_features2000:
    # get the value_count of the column
    sizes = string_features2000[col].value_counts(normalize = True)
    # get the names of the levels that make up less than 1% of the dataset
    values_to_delete = sizes[sizes<0.01].index
    string_features2000[col].ix[string_features2000[col].isin(values_to_delete)] = "Other"
    
string_features2000.teamID.value_counts(normalize = True)
df2000.teamID.value_counts(normalize=True)

string_featureshw.pos.value_counts(normalize = True)
dfhw.pos.value_counts(normalize=True)


## Prior to binning need to check to see if unique features are included
## here that are not in pre-2000 data
unique_teamID = list(set(e for e in string_featureshw.teamID))
unique_teamID_post2000 = list(set(e for e in string_features2000.teamID))
extra_teams = list(set(e for e in unique_teamID_post2000 if e not in unique_teamID))

# Check string features post 2000
string_featureshw.teamID.index[~string_features2000.index.isin(unique_teamID)]
# There are a few features that do not occur in the pre-2000 dataset

## Replace these with other
string_features2000.teamID.replace(to_replace=extra_teams,value = 'Other',inplace=True)

encoded_data2000 = get_binary_values(string_features2000)

encoded_data2000.reset_index(inplace=True)

## Removing extra columns introduced by reset index function
del encoded_data2000['index']
del encoded_data2000['level_0']

########
# Imputing missing values
#######

## re-creating the numeric_features dataframe.
numeric_features2000 = df2000.ix[:, df2000.dtypes != 'object']
## inputting the median observation
imputer_object = imp(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features2000)
numeric_features2000 = pd.DataFrame(imputer_object.transform(numeric_features2000), 
                                    columns = numeric_features2000.columns)
                    
##Merging the two datasets
explanatory_df2000 = pd.concat([numeric_features2000, encoded_data2000],axis = 1)
explanatory_df2000.head()


#Keeping only variables selected earlier
explanatory_df20002 = [col for col in explanatory_df2000.columns if col in ['b_atbat', 'b_hits', 'p_wins', 'f_puts']] 
df2000_slim = explanatory_df2000[explanatory_df20002]


#scaling data
scaler = pre.StandardScaler()
scaler.fit(df2000_slim)
df2000_slim = pd.DataFrame(scaler.transform(df2000_slim), columns = df2000_slim.columns)
df2000_slim.describe()


# Predict values for the output
predicted_values = final_estimator_used.predict(df2000_slim)

# Create a confusion matrix to examine the results
cm = pd.crosstab(response_series2000,predicted_values,rownames=['True Label'],
                     colnames = ['Predicted Label'],margins = True)
                     
print cm

# Calculate cross val accuracy scores
accuracy_scores_best_OOS = cv(final_estimator_used,
df2000_slim, response_series2000, cv=10, scoring='accuracy')

accuracy_scores_best_oldData = cv(final_estimator_used,
explanatory_dfhw, response_serieshw, cv=10, scoring='accuracy')

print accuracy_scores_best_OOS.mean()
#Accuracy of 87% here
print accuracy_scores_best_oldData.mean()
#Accuracy of 92% here










