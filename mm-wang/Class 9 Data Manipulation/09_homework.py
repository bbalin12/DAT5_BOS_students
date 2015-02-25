# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 18:55:58 2015

@author: Margaret
"""

import pandas
import sqlite3

# putting a setting into pandas that lets you print out the entire
# DataFrame when you use the .head() method
pandas.set_option('display.max_columns', None)


# first, let's create a categorical feature that shows the dominant team 
# played per player, replace if it exists
con = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')
query = 'select playerID, teamID from Batting'
df = pandas.read_sql(query, con)
con.close()

# to pull the mode team for each player
majority_team_by_player = df.groupby(['playerID']).agg(lambda x:x.value_counts().idxmax())
# trying to find all instances of each player ID, then running a mode operation on each playerID

# write this table back to the database, replace if it exists so no error pops up
conn = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')
majority_team_by_player.to_sql('dominant_team_per_player', conn, if_exists = 'replace')
conn.close()
# sqlite does not have the same kind of mode here
# relational databases are "mean" databases



# first, let's create a categorical feature that shows the dominant team 
# played per player
con = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')
query = """
SELECT (m.nameFirst||" "||nameLast) as p_name, m.height as height, m.weight as weight, m.bats as bats, m.throws as throws,
dom_team, inducted, bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_strikes, bat_stolen, bat_baseballs, bat_intentwalks
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, pitch_wild, pitch_saves,
f_putouts, f_assists, f_errors FROM Master m
INNER JOIN
(SELECT pID, dom.teamID as dom_team, inducted, 
bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_strikes, bat_stolen, bat_RBI, bat_caught, bat_baseballs,
bat_intentwalks, pitch_wild, pitch_games, pitch_saves, pitch_earnruns, pitch_runsallowed,
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, f_putouts, f_assists, f_errors FROM dominant_team_per_player dom
INNER JOIN
(SELECT h.playerID as pID, max(CASE WHEN h.inducted='Y' THEN 1 ELSE 0 END) as inducted, 
bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_strikes, bat_stolen, bat_RBI, bat_caught, bat_baseballs,
bat_intentwalks, pitch_wild, pitch_games, pitch_saves, pitch_earnruns, pitch_runsallowed,
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, f_putouts, f_assists, f_errors FROM HallofFame h
INNER JOIN
(SELECT f.playerID, f.PO as f_putouts, f.A as f_assists, f.E as f_errors, 
bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_RBI, bat_caught, bat_baseballs, bat_intentwalks,
bat_strikes, bat_stolen, pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, pitch_wild, pitch_games, pitch_saves,
pitch_earnruns, pitch_runsallowed FROM Fielding f
LEFT JOIN
(SELECT b.playerID, b.lgID as bat_league, sum(b.R) as bat_runs, sum(b.H) as bat_hits, sum(b.CS) as bat_caught, sum(b.BB) as bat_baseballs,
sum(b.AB) as at_bats, sum(b.HR) as bat_homeruns, sum(b.RBI) as bat_RBI, sum(b.SO) as bat_strikes, sum(b.SB) as bat_stolen,
sum(b.IBB) as bat_intentwalks,
sum(p.W) as pitch_wins, sum(p.SO) as pitch_strikes, sum(p.SHO) as pitch_shuts, avg(1/p.ERA) as pitch_ERA, sum(p.WP) as pitch_wild,
sum(p.G) as pitch_games, sum(p.SV) as pitch_saves, sum(p.ER) as pitch_earnruns, sum(p.R) as pitch_runsallowed 
FROM Batting b
LEFT JOIN Pitching p on p.playerID = b.playerID
GROUP BY b.playerID) batpitch on batpitch.playerID = f.playerID
GROUP BY batpitch.playerID) positions
ON positions.playerID = h.playerID
WHERE h.yearID < 2000 and h.yearID > 1950
GROUP BY h.playerID) all_features on pID = dom.playerID) all_data on pID = m.playerID
"""
df = pandas.read_sql(query, con)
con.close()

df.drop('p_name',  1, inplace = True)


#################
### Preprocessing
#################

## splitting out the explanatory features 
explanatory_features = [col for col in df.columns if col not in ['p_name', 'inducted']]
explanatory_df = df[explanatory_features]

# dropping rows with no data.
explanatory_df.dropna(how = 'all', inplace = True) 

# extracting column names 
explanatory_colnames = explanatory_df.columns

## doing the same for response
response_series = df.inducted
response_series.dropna(how = 'all', inplace = True) 
# copy warning - operating on slice of data, but not on data. If you include inplace = True, should be the data

## seeing which explanatory feature rows got removed.  Looks like none.
response_series.index[~response_series.index.isin(explanatory_df.index)]


##########################################
### Splitting Data into Numeric and String
##########################################

string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object'] #data inside is string data if object, all rows
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']


#######################
### Binning/Make Binary
#######################

# first, fill the NANs in the feature (this lets us see if there are features
# that are all NANs, as they will show up as all 'Nothing' when we start binning or look for features with no variation)
string_features = string_features.fillna('Nothing')

## make into a function
def cleanup_data(df,cutoffPercent = 0.01):
    for col in df:
        sizes = df[col].value_counts(normalize = True)
        values_to_delete = sizes[sizes<cutoffPercent].index
        df[col].ix[df[col].isin(values_to_delete)] = "Other"
    return df

string_features = cleanup_data(string_features)


#################################
### Encoding Categorical Features
#################################

# creating catcher data frame that will hold the encoded data
encoded_data = pandas.DataFrame(index = string_features.index) # empty data frame
for col in string_features.columns:
    ## calling pandas.get_dummies to turn the column into a sequence of 
    ## binary variables. Notice I'm using the 'prefix' feature to include the 
    ## original name of the column
    data = pandas.get_dummies(string_features[col], prefix=col.encode('ascii', 'replace'))
    # creates dummy variables, can create a prefix - it is the column name, ascii is just a way to encode it
    encoded_data = pandas.concat([encoded_data, data], axis=1)
    # concatenating new dataFrame to encoded dataFrame
    

def get_binary_values(data_frame):
    """Encodes categorical features in Pandas with get_dummies.
    Includes prefix of column name.
    """
    all_columns = pandas.DataFrame(index = data_frame.index)
    for col in data_frame.columns:
        data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pandas.concat([all_columns, data], axis=1)
    return all_columns

encoded_data = get_binary_values(string_features)

# verify that encoding occurred
encoded_data.head()


##########################################
### Filling in NaNs for Numerical Features
##########################################

# Impute using mean strategy

from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy = 'median', axis = 0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), 
                                    columns = numeric_features.columns)
                                    

############
### Merging
###########

explanatory_df = pandas.concat([numeric_features, encoded_data], axis = 1)
explanatory_df.head()


#############################
### Features with No Variance
#############################
toKeep = []
toDelete = []

for col in explanatory_df:
    # if value counts method returns more than 1 unique entity, append to toKeep
    if len(explanatory_df[col].value_counts())>1:
        toKeep.append(col)
    # otherwise, append it to "toDelete"
    else:
        toDelete.append(col)

        
#####################################
### Features with Perfect Correlation
#####################################

# first, let's create a correlation matrix diagram for the first 26 features.
toChart = explanatory_df.ix[:,0:25].corr() # reads the first 26 columns
toChart.head()

import matplotlib.pyplot as plt
import numpy as np

#heatmap
plt.pcolor(toChart)
plt.yticks(np.arange(0.5,len(toChart.index),1), toChart.index)
plt.xticks(np.arange(0.5,len(toChart.columns),1),toChart.columns,rotation=-90)
plt.colorbar()
plt.show()

# whole dataset correlation matrix
# corr_matrix = explanatory_df.corr()
# heatmap
#plt.pcolor(corr_matrix)
#plt.yticks(np.arange(0.5,len(corr_matrix.index),1), corr_matrix.index)
#plt.xticks(np.arange(0.5,len(corr_matrix.columns),1),corr_matrix.columns,rotation=-90)
#plt.colorbar()
#plt.show()



# let's combine all of this into a nice function.
def find_perfect_corr(df):
    """Finds columns that are either positively or negatively perfectly correlated (with correlations of +1 or -1), 
        and creates a dict that includes which columns to drop so that each remaining column
        is independent.
    """  
    corrMatrix = df.corr()
    corrMatrix.ix[:,:] =  np.tril(corrMatrix.values, k = -1)
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

correlations = find_perfect_corr(explanatory_df)
print correlations
# no perfect correlations here

explanatory_df.drop(correlations['toRemove'],1,inplace=True)

################
### Scaling Data
################

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), 
                                  columns = explanatory_df.columns)
# standard deviations are all now 1


###########################
### Imputing Missing Values
###########################

from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy = 'median', axis = 0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), 
                                    columns = numeric_features.columns)
                                    

#################################
### Recursive Feature Elimination                                   
#################################

from sklearn.feature_selection import RFECV
from sklearn import tree

# create new class with a .coef_ attribute.
class TreeClassifierWithCoef(tree.DecisionTreeClassifier):
    def fit(self, *args, **kwargs):
        super(tree.DecisionTreeClassifier,self).fit(*args,**kwargs)
        self.coef_ = self.feature_importances_

# create tree-based estimator
decision_tree = TreeClassifierWithCoef(criterion = 'gini', splitter = 'best', 
                max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2,
                max_leaf_nodes = None, random_state = 1)
# same as decision_tree = TreeClassifierWithCoef()
                
# set up estimator, score by AUC
# Recursive Feature Elimination with Cross Validation                
rfe_cv = RFECV(estimator = decision_tree, step=1, cv=10, scoring = 'roc_auc', verbose = 0)
# verbose prints out progress
rfe_cv.fit(explanatory_df, response_series)

print "Optimal number of features {0} of {1} considered.".format(rfe_cv.n_features_,
        len(explanatory_df.columns))

# print scores as we increase the number of features - farther down the list, higher the number of features
print "Mean Grid Score: %f" %rfe_cv.grid_scores_.mean()

# let's plot the results
plt.figure()
plt.xlabel("Number of Selected Features")
plt.ylabel("Cross Validation Score (ROC_AUC)")
plt.plot(range(1,len(rfe_cv.grid_scores_)+1),rfe_cv.grid_scores_)

features_used = explanatory_df.columns[rfe_cv.support_]
# print features_used

#you can extract the final selected model object this way:
final_estimator_used = rfe_cv.estimator_


##########################################
### Combining Grid Search (Depth) with RFE
##########################################

# you can also combine RFE with grid search to find the tuning 
# parameters and features that optimize model accuracy metrics.
# do this by passing the RFECV object to GridSearchCV.
###from sklearn.grid_search import GridSearchCV

# small range for reasonable time
###depth_range = range(1,4)
# for param_grid, you need estimator__ as a prefix to parameters
###param_grid = dict(estimator__max_depth=depth_range)
# notice that this will take longer to compute
###rfe_grid_search = GridSearchCV(rfe_cv, param_grid, cv=10, scoring = 'roc_auc')
###rfe_grid_search.fit(explanatory_df, response_series)

###print rfe_grid_search.grid_scores_
###rfe_grid_search.best_params_
# 300-400 model runs right now
# ONLY gridsearch tuning parameters that are most essential, and only search over discrete number

# let's plot out
###grid_mean_scores = [score[1] for score in rfe_grid_search.grid_scores_]

###plt.figure()
###plt.plot(depth_range,grid_mean_scores)
###plt.hold(True)
###plt.plot(rfe_grid_search.best_params_['estimator__max_depth'],
###         rfe_grid_search.best_score_,
###         'ro', markersize = 12, markeredgewidth=1.5, markerfacecolor='None', markeredgecolor='r')
###plt.grid(True)

# pull out the winning estimator
###best_decision_tree_rfe_grid = rfe_grid_search.best_estimator_

###features_used_rfecv_grid = explanatory_df.columns[best_decision_tree_rfe_grid.get_support()]
###print "Features Used in Best Decision Tree Grid: %s" % features_used_rfecv_grid

# Not needed in prediction



######################################
### Testing Data from Year 2000 Onward
######################################

con = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')
query = """
SELECT (m.nameFirst||" "||nameLast) as p_name, m.height as height, m.weight as weight, m.bats as bats, m.throws as throws,
dom_team, inducted, bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_strikes, bat_stolen, bat_baseballs, bat_intentwalks,
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, pitch_wild, pitch_saves,
f_putouts, f_assists, f_errors FROM Master m
INNER JOIN
(SELECT pID, dom.teamID as dom_team, inducted, 
bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_strikes, bat_stolen, bat_RBI, bat_caught, bat_baseballs,
bat_intentwalks, pitch_wild, pitch_games, pitch_saves, pitch_earnruns, pitch_runsallowed,
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, f_putouts, f_assists, f_errors FROM dominant_team_per_player dom
INNER JOIN
(SELECT h.playerID as pID, max(CASE WHEN h.inducted='Y' THEN 1 ELSE 0 END) as inducted, 
bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_strikes, bat_stolen, bat_RBI, bat_caught, bat_baseballs,
bat_intentwalks, pitch_wild, pitch_games, pitch_saves, pitch_earnruns, pitch_runsallowed,
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, f_putouts, f_assists, f_errors FROM HallofFame h
INNER JOIN
(SELECT f.playerID, f.PO as f_putouts, f.A as f_assists, f.E as f_errors, 
bat_league, bat_runs, bat_hits, at_bats, bat_homeruns, bat_RBI, bat_caught, bat_baseballs, bat_intentwalks,
bat_strikes, bat_stolen, pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, pitch_wild, pitch_games, pitch_saves,
pitch_earnruns, pitch_runsallowed FROM Fielding f
LEFT JOIN
(SELECT b.playerID, b.lgID as bat_league, sum(b.R) as bat_runs, sum(b.H) as bat_hits, sum(b.CS) as bat_caught, sum(b.BB) as bat_baseballs,
sum(b.AB) as at_bats, sum(b.HR) as bat_homeruns, sum(b.RBI) as bat_RBI, sum(b.SO) as bat_strikes, sum(b.SB) as bat_stolen,
sum(b.IBB) as bat_intentwalks,
sum(p.W) as pitch_wins, sum(p.SO) as pitch_strikes, sum(p.SHO) as pitch_shuts, avg(1/p.ERA) as pitch_ERA, sum(p.WP) as pitch_wild,
sum(p.G) as pitch_games, sum(p.SV) as pitch_saves, sum(p.ER) as pitch_earnruns, sum(p.R) as pitch_runsallowed 
FROM Batting b
LEFT JOIN Pitching p on p.playerID = b.playerID
GROUP BY b.playerID) batpitch on batpitch.playerID = f.playerID
GROUP BY batpitch.playerID) positions
ON positions.playerID = h.playerID
WHERE h.yearID > 2000
GROUP BY h.playerID) all_features on pID = dom.playerID) all_data on pID = m.playerID
"""
df = pandas.read_sql(query, con)
con.close()

df.drop('p_name',  1, inplace = True)


#################
### Preprocessing
#################

#explanatory_df = pandas.DataFrame(columns = features_used)


## splitting out the explanatory features
#explanatory_features = [col for col in features_used]
#xplanatory_features = [col for col in df.columns if col not in ['p_name', 'inducted']]
explanatory_df = df[explanatory_features]


# dropping rows with no data.
explanatory_df.dropna(how = 'all', inplace = True) 

# extracting column names 
explanatory_colnames = explanatory_df.columns

## doing the same for response
response_series = df.inducted
response_series.dropna(how = 'all', inplace = True) 
# copy warning - operating on slice of data, but not on data. If you include inplace = True, should be the data

## seeing which explanatory feature rows got removed.  Looks like none.
response_series.index[~response_series.index.isin(explanatory_df.index)]


##########################################
### Splitting Data into Numeric and String
##########################################

string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object'] #data inside is string data if object, all rows
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']


#######################
### Binning/Make Binary
#######################

# first, fill the NANs in the feature (this lets us see if there are features
# that are all NANs, as they will show up as all 'Nothing' when we start binning or look for features with no variation)
string_features = string_features.fillna('Nothing')


## make into a function
def cleanup_data(df,cutoffPercent = 0.01):
    for col in df:
        sizes = df[col].value_counts(normalize = True)
        values_to_delete = sizes[sizes<cutoffPercent].index
        df[col].ix[df[col].isin(values_to_delete)] = "Other"
    return df

string_features = cleanup_data(string_features)


#################################
### Encoding Categorical Features
#################################

# creating catcher data frame that will hold the encoded data
encoded_data = pandas.DataFrame(index = string_features.index) # empty data frame
for col in string_features.columns:
    ## calling pandas.get_dummies to turn the column into a sequence of 
    ## binary variables. Notice I'm using the 'prefix' feature to include the 
    ## original name of the column
    data = pandas.get_dummies(string_features[col], prefix=col.encode('ascii', 'replace'))
    # creates dummy variables, can create a prefix - it is the column name, ascii is just a way to encode it
    encoded_data = pandas.concat([encoded_data, data], axis=1)
    # concatenating new dataFrame to encoded dataFrame
    

def get_binary_values(data_frame):
    """Encodes categorical features in Pandas with get_dummies.
    Includes prefix of column name.
    """
    all_columns = pandas.DataFrame(index = data_frame.index)
    for col in data_frame.columns:
        data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pandas.concat([all_columns, data], axis=1)
    return all_columns

encoded_data = get_binary_values(string_features)

# verify that encoding occurred
encoded_data.head()


##########################################
### Filling in NaNs for Numerical Features
##########################################

# Impute using mean strategy

from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy = 'median', axis = 0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), 
                                    columns = numeric_features.columns)
                                    

############
### Merging
###########

explanatory_df = pandas.concat([numeric_features, encoded_data], axis = 1)
explanatory_df.head()
#explanatory_df.drop(explanatory_df.columns[~rfe_cv.support_], axis=1, inplace=True)


################
### Scaling Data
################

scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), 
                                  columns = explanatory_df.columns)
# standard deviations are all now 1


#######################
### Prediction Accuracy
#######################

from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score

decision_tree = final_estimator_used

cv = StratifiedKFold(response_series, n_folds=10)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    decision_tree = final_estimator_used
    decision_tree.fit(explanatory_df.ix[train,], response_series.ix[train,])
    probabilities = pandas.DataFrame(decision_tree.predict_proba(explanatory_df.ix[test,]))
    # Confusion Matrix
    predicted_values = decision_tree.predict(explanatory_df.ix[test])    
    #cm = pandas.crosstab(response_series[test],predicted_values, rownames = ['True Label'], colnames = ['Predicted Label'], margins = True)
    #print "Decision Tree Confusion Matrix: %d" % (i+1)
    #print cm
    #print '\n'
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(response_series.ix[test], probabilities[1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Decision Tree 10-Fold Cross Validation ROC')
plt.subplot()
plt.legend(bbox_to_anchor=(1.65,1.07))
plt.show()
importances = pandas.DataFrame(decision_tree.feature_importances_, 
            index = explanatory_df.columns, columns = ['importance'])
            
importances.sort(columns = ['importance'], ascending = False, inplace = True)
print importances

# calculating accuracy
accuracy_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring = 'accuracy')

# calculating Cohen's Kappa
mean_accuracy_score_cart = accuracy_scores_cart.mean()
# recall we already calculated the largest_class_percent_of_total above.
largest_class_percent_of_total = response_series.value_counts(normalize = True)[0]
kappa_cart = (mean_accuracy_score_cart - largest_class_percent_of_total) / (1-largest_class_percent_of_total)

# calculating f1 score
f1_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring = 'f1')

# calculating the ROC area under the curve score. 
roc_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='roc_auc')

print 'Cross Validated Score using Decision Tree Accuracy: %f' % accuracy_scores_cart.mean()
print "Decision Tree Cohen's Kappa is: %f" % kappa_cart
print "Decision Tree F1 Score is: %f" % f1_scores_cart.mean()
print "Decision Tree ROC Score is: %f" % roc_scores_cart.mean()