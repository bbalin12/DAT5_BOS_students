# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 21:11:39 2015

@author: tdong1
"""

import pandas
import sqlite3

# putting a setting into pandas that lets you print out the entire
# DataFrame when you use the .head() method
pandas.set_option('display.max_columns', None)

# first, let's create a categorical feature that shows the dominant team 
# played per player
DATABASE = r'D:\Training\DataScience\Class3\lahman2013.sqlite'
con = sqlite3.connect(DATABASE)
query = 'select playerID, teamID from Batting'
df = pandas.read_sql(query, con)
con.close()

# use pandas.DataFrame.groupby and an annonymous lambda function
# to pull the mode team for each player
majority_team_by_player = df.groupby(['playerID']).agg(lambda x:x.value_counts().idxmax())

## write the data back to the database
conn = sqlite3.connect(DATABASE)
majority_team_by_player.to_sql('dominant_team_per_player', conn, if_exists = 'replace')
# closing the connection.
conn.close()

##Create another categorical variable
DATABASE = r'D:\Training\DataScience\Class3\lahman2013.sqlite'
con = sqlite3.connect(DATABASE)

query = 'select playerID, lgID from Batting'
df = pandas.read_sql(query, con)
con.close()

# use pandas.DataFrame.groupby and an annonymous lambda function
# to pull the mode team for each player
majority_league_by_player = df.groupby(['playerID']).agg(lambda x:x.value_counts().idxmax())

## write the data back to the database
conn = sqlite3.connect(DATABASE)
majority_league_by_player.to_sql('dominant_league_per_player', conn, if_exists = 'replace')
# closing the connection.
conn.close()


## using the new table as part of the monster query from last class
monster_query = """
SELECT hfi.playerID,hfi.inducted, batting.*, pitching.*, fielding.*,t.*,l.* FROM 
        (SELECT playerID, CASE WHEN average_inducted = 0 then 0 else 1 end as inducted from 
            (
            select playerID, avg(case when inducted = 'Y' then 1 else 0 end ) as average_inducted from  HallOfFame hf
            where yearid < 2000
            group by playerID
            )
        ) hfi 
    left outer join 
    (
    select playerID,  sum(AB) as total_at_bats, sum(H) as total_hits, sum(R) as total_runs, sum(HR) as total_home_runs, sum(SB) as total_stolen_bases,
    sum(RBI) as total_RBI, sum(CS) as total_caught_stealing, sum(SO) as total_hitter_strikeouts, sum(IBB) as total_intentional_walks
    from Batting
    group by playerID
    )
    batting on batting.playerID = hfi.playerID
    left outer join
    (
     select playerID, sum(G) as total_games_pitched, sum(SO) as total_shutouts, sum(sv) as total_saves, sum(IPouts) as total_outs_pitched, 
    sum(H) as total_pitching_hits, sum(er) as total_pitching_earned_runs, sum(so) as total_pitcher_strikeouts, 
    avg(ERA) as average_ERA, sum(WP) as total_wild_pitches, sum(HBP) as total_hit_by_pitch, sum(GF) as total_games_finished,
    sum(R) as total_runs_allowed
    from Pitching
    group by playerID
    ) 
    pitching on pitching.playerID = hfi.playerID 
    LEFT OUTER JOIN
    (
    select playerID, sum(G) as total_games_fielded, sum(InnOuts) as total_time_in_field_with_outs, 
    sum(PO) as total_putouts, sum(E) as total_errors, sum(DP) as total_double_plays
    from Fielding
    group by playerID
    ) 
    fielding on fielding.playerID = hfi.playerID
    LEFT OUTER JOIN 
        dominant_team_per_player t on t.playerID = hfi.playerID
    LEFT OUTER JOIN 
        dominant_league_per_player l on l.playerID = hfi.playerID
"""

con = sqlite3.connect(DATABASE)
df = pandas.read_sql(monster_query, con)
con.close()

## getting an intial view of the data for validation
df.head(10)
df.columns

# dropping duplicate playerID columns
df.drop('playerID',  1, inplace = True)

df.columns

#rename columns
df.rename(columns={'hfi.playerID': 'playerID', 'hfi.inducted': 'inducted'}, inplace=True)
df.head()

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



missing_rows = response_series.index[~response_series.index.isin(explanatory_df.index)]
print missing_rows
## seeing which explanatory feature rows got removed. 
# Looks like a bunch. possible problem?

#Int64Index([34, 37, 50, 104, 135, 145, 148, 153, 156, 176, 206, 217, 228, 263, 289, 
#            292, 298, 314, 315, 366, 403, 406, 422, 426, 461, 480, 497, 504, 524, 525, 560,
#            570, 729, 770, 876, 896, 897, 899, 915, 935], dtype='int64')


#maybe match up explanatory_df and response_series by removing missing rows)
response_series = response_series.drop(response_series.index[missing_rows])


#check missing_rows now
missing_rows = response_series.index[~response_series.index.isin(explanatory_df.index)]
print missing_rows
#Int64Index([], dtype='int64')



### now, let's seperate the numeric explanatory data from the string data
string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']

######
# now, let's find if any of the categorical features need 'binnng'
#####
# first, fill the NANs in the feature (this lets us see if there are features
# that are all NANs, as they will show up as all 'Nothing' when we start binning or look for features with no variation)
string_features = string_features.fillna('Nothing')

# lets' create the heuristic that a level in the feature must exist in more
# than 1% of the training data to be retained. 
for col in string_features:
    # get the value_count of the column
    sizes = string_features[col].value_counts(normalize = True)
    # get the names of the levels that make up less than 1% of the dataset
    values_to_delete = sizes[sizes<0.01].index
    string_features[col].ix[string_features[col].isin(values_to_delete)] = "Other"

# let's verify if the replacement happened
string_features.teamID.value_counts(normalize = True)
#Other    0.083700
#NYA      0.073789
#PIT      0.070485
#CHN      0.061674
#SLN      0.057269
#PHI      0.055066
#NY1      0.053965
#CIN      0.050661
#DET      0.046256
#BOS      0.042952
#CLE      0.042952
#BRO      0.039648
#PHA      0.038546
#WS1      0.031938
#CHA      0.030837
#LAN      0.029736
#BSN      0.028634
#BAL      0.024229
#HOU      0.023128
#SFN      0.019824
#NYN      0.017621
#SLA      0.014317
#KCA      0.014317
#CAL      0.013216
#MIN      0.013216
#ML4      0.011013
#MON      0.011013
#dtype: float64



string_features.lgID.value_counts(normalize = True)
#NL       0.553965
#AL       0.431718
#Other    0.014317
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
#Encoding has occured
encoded_data.columns
#Index([u'teamID_BAL', u'teamID_BOS', u'teamID_BRO', u'teamID_BSN', u'teamID_CAL',
#       u'teamID_CHA', u'teamID_CHN', u'teamID_CIN', u'teamID_CLE', u'teamID_DET', 
#       u'teamID_HOU', u'teamID_KCA', u'teamID_LAN', u'teamID_MIN', u'teamID_ML4',
#       u'teamID_MON', u'teamID_NY1', u'teamID_NYA', u'teamID_NYN', u'teamID_Other', 
#       u'teamID_PHA', u'teamID_PHI', u'teamID_PIT', u'teamID_SFN', u'teamID_SLA', u'teamID_SLN', 
#       u'teamID_WS1', u'lgID_AL', u'lgID_NL', u'lgID_Other'], dtype='object')



## let's also wrap this into a function.
def get_binary_values(data_frame):
    """encodes cateogrical features in Pandas.
    """
    all_columns = pandas.DataFrame( index = data_frame.index)
    for col in data_frame.columns:
        data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pandas.concat([all_columns, data], axis=1)
    return all_columns

print numeric_features.columns
#Index([u'total_at_bats', u'total_hits', u'total_runs', u'total_home_runs', u'total_stolen_bases',
#       u'total_RBI', u'total_caught_stealing', u'total_hitter_strikeouts', u'total_intentional_walks',
#       u'total_games_pitched', u'total_shutouts', u'total_saves', u'total_outs_pitched', u'total_pitching_hits',
#       u'total_pitching_earned_runs', u'total_pitcher_strikeouts', u'average_ERA', u'total_wild_pitches', 
#       u'total_hit_by_pitch', u'total_games_finished', u'total_runs_allowed', u'total_games_fielded', 
#       u'total_time_in_field_with_outs', u'total_putouts', u'total_errors', u'total_double_plays'], dtype='object')

## now, let's fill the NANs in our nuemeric features.
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
#    print explanatory_df[col].value_counts()
    ## if the value_counts method returns more than one uniqe entity,
    ## append the column name to 'toKeep'

    if len(explanatory_df[col].value_counts()) > 1:
        toKeep.append(col)
    ## if not, append to 'toDelete'.
    else:
        toDelete.append(col)



# let's see if there's zero variance in an features
print toKeep
#['total_at_bats', 'total_hits', 'total_runs', 'total_home_runs', 'total_stolen_bases', 'total_RBI', 
#'total_caught_stealing', 'total_hitter_strikeouts', 'total_intentional_walks', 'total_games_pitched', 
#'total_shutouts', 'total_saves', 'total_outs_pitched', 'total_pitching_hits', 'total_pitching_earned_runs', 
#'total_pitcher_strikeouts', 'average_ERA', 'total_wild_pitches', 'total_hit_by_pitch', 'total_games_finished', 
#'total_runs_allowed', 'total_games_fielded', 'total_time_in_field_with_outs', 'total_putouts', 'total_errors', 
#'total_double_plays', u'teamID_BAL', u'teamID_BOS', u'teamID_BRO', u'teamID_BSN', u'teamID_CAL', u'teamID_CHA', 
#u'teamID_CHN', u'teamID_CIN', u'teamID_CLE', u'teamID_DET', u'teamID_HOU', u'teamID_KCA', u'teamID_LAN', 
#u'teamID_MIN', u'teamID_ML4', u'teamID_MON', u'teamID_NY1', u'teamID_NYA', u'teamID_NYN', 'teamID_Other', 
#u'teamID_PHA', u'teamID_PHI', u'teamID_PIT', u'teamID_SFN', u'teamID_SLA', u'teamID_SLN', u'teamID_WS1', 
#u'lgID_AL', u'lgID_NL', 'lgID_Other']


print toDelete
# doesn't look like it.
#[]

#There appear to be no variables to delete

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
#Variables with perfect correlation are total_pitcher_strikeouts and total_shutouts

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
#got error ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
#Check for rows in explanatory_df and not in response_series
missing_rows = response_series.index[~response_series.index.isin(explanatory_df.index)]
missing_rows = explanatory_df.index[~explanatory_df.index.isin(response_series.index)]
#Int64Index([34, 37, 50, 104, 135, 145, 148, 153, 156, 176, 206, 217, 228, 263, 289, 
#292, 298, 314, 315, 366, 403, 406, 422, 426, 461, 480, 497, 504, 524, 525, 560, 570, 
#729, 770, 876, 896, 897, 899], dtype='int64')
#drop those rows
explanatory_df = explanatory_df.drop(df.index[missing_rows])
#drop any rows with NaN
explanatory_df = explanatory_df.dropna(axis=0)
missing_rows = response_series.index[~response_series.index.isin(explanatory_df.index)]
#Int64Index([908, 909, 910, 911, 912, 913, 914, 916, 917, 918, 919, 920, 921, 922, 
#923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 936, 937, 938, 939, 
#940, 941, 942, 943, 944, 945, 946, 947], dtype='int64')
response_series = response_series.drop(df.index[missing_rows])

#decided to fill nan's with zero
#explanatory_df = explanatory_df.fillna(0)
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
decision_tree = TreeClassifierWithCoef(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, 
                                       min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)

## set up the estimator. Score by AUC
rfe_cv = RFECV(estimator=decision_tree, step=1, cv=10,
              scoring='roc_auc', verbose = 1)
rfe_cv.fit(explanatory_df, response_series)

print "Optimal number of features :{0} of {1} considered".format(rfe_cv.n_features_,len(explanatory_df.columns))
#Optimal number of features :4 of 55 considered

# printing out scores as we increase the number of features -- the farter
# down the list, the higher the number of features considered.
print rfe_cv.grid_scores_
#[ 0.50979195  0.50124471  0.52410292  0.53473335  0.52607091  0.51631715
#  0.50092622  0.51234676  0.48869969  0.50493721  0.52083727  0.51181117
#  0.49511053  0.51000589  0.48488026  0.48133739  0.47327989  0.47336413
#  0.47122288  0.4652248   0.5005208   0.50227774  0.50224533  0.49330386
#  0.47834671  0.47919692  0.46419178  0.46640246  0.47870235  0.47372557
#  0.47277228  0.47621176  0.48498458  0.48492988  0.48089445  0.47819974
#  0.49257604  0.49193874  0.4802951   0.47096411  0.48771799  0.47174036
#  0.48443628  0.4863159   0.48145956  0.46551806  0.48712493  0.48374584
#  0.48214581  0.48784838  0.48293557  0.47438029  0.49738933  0.4984413
#  0.47931995]

## let's plot out the results
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (ROC_AUC)")
plt.plot(range(1, len(rfe_cv.grid_scores_) + 1), rfe_cv.grid_scores_)
plt.show()
# notice you could have just as well have included the 10 most important 
# features and received similar accuracy.

# you can pull out the features used this way:
features_used = explanatory_df.columns[rfe_cv.get_support()]
print features_used
#Index([u'total_hits', u'total_games_fielded', u'total_putouts', u'total_double_plays'], dtype='object')

#you can extract the final selected model object this way:
final_estimator_used = rfe_cv.estimator_
#TreeClassifierWithCoef(compute_importances=None, criterion='gini',
#            max_depth=None, max_features=None, max_leaf_nodes=None,
#            min_density=None, min_samples_leaf=2, min_samples_split=2,
#            random_state=1, splitter='best')

# you can also combine RFE with grid search to find the tuning 
# parameters and features that optimize model accuracy metrics.
# do this by passing the RFECV object to GridSearchCV.
from sklearn.grid_search import  GridSearchCV

# doing this for a small range so I can show you the answer in a reasonable
# amount of time.
depth_range = range(4, 6)
# notice that in param_grid, I need to prefix estimator__ to my paramerters.
param_grid = dict(estimator__max_depth=depth_range)
# notice that this will take quite a bit longer to compute.
rfe_grid_search = GridSearchCV(rfe_cv, param_grid, cv=10, scoring='roc_auc')
rfe_grid_search.fit(explanatory_df, response_series)

print rfe_grid_search.grid_scores_
#[mean: 0.52123, std: 0.07191, params: {'estimator__max_depth': 4}, 
#mean: 0.49032, std: 0.05068, params: {'estimator__max_depth': 5}]


rfe_grid_search.best_params_
#{'estimator__max_depth': 4}

# let's plot out the results.
grid_mean_scores = [score[1] for score in rfe_grid_search.grid_scores_]
#[0.52123453273825171, 0.49031615436217008]

plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.plot(rfe_grid_search.best_params_['estimator__max_depth'], rfe_grid_search.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')
plt.grid(True)
#see rfe_grid_search_scores.png

# now let's pull out the winning estimator.
best_decision_tree_rfe_grid = rfe_grid_search.best_estimator_
#RFECV(cv=10,
#   estimator=TreeClassifierWithCoef(compute_importances=None, criterion='gini',
#            max_depth=4, max_features=None, max_leaf_nodes=None,
#            min_density=None, min_samples_leaf=2, min_samples_split=2,
#            random_state=1, splitter='best'),
#   estimator_params={}, loss_func=None, scoring='roc_auc', step=1,
#   verbose=1)


features_used_rfecv_grid = explanatory_df.columns[best_decision_tree_rfe_grid.get_support()]
#Index([u'total_games_fielded'], dtype='object')
#The best estimator for hall of fame induction is total_games_fielded


#Read in post 2000 data
monster_query = """
SELECT hfi.playerID,hfi.inducted, batting.*, pitching.*, fielding.*,t.*,l.* FROM 
        (SELECT playerID, CASE WHEN average_inducted = 0 then 0 else 1 end as inducted from 
            (
            select playerID, avg(case when inducted = 'Y' then 1 else 0 end ) as average_inducted from  HallOfFame hf
            where yearid >= 2000
            group by playerID
            )
        ) hfi 
    left outer join 
    (
    select playerID,  sum(AB) as total_at_bats, sum(H) as total_hits, sum(R) as total_runs, sum(HR) as total_home_runs, sum(SB) as total_stolen_bases,
    sum(RBI) as total_RBI, sum(CS) as total_caught_stealing, sum(SO) as total_hitter_strikeouts, sum(IBB) as total_intentional_walks
    from Batting
    group by playerID
    )
    batting on batting.playerID = hfi.playerID
    left outer join
    (
     select playerID, sum(G) as total_games_pitched, sum(SO) as total_shutouts, sum(sv) as total_saves, sum(IPouts) as total_outs_pitched, 
    sum(H) as total_pitching_hits, sum(er) as total_pitching_earned_runs, sum(so) as total_pitcher_strikeouts, 
    avg(ERA) as average_ERA, sum(WP) as total_wild_pitches, sum(HBP) as total_hit_by_pitch, sum(GF) as total_games_finished,
    sum(R) as total_runs_allowed
    from Pitching
    group by playerID
    ) 
    pitching on pitching.playerID = hfi.playerID 
    LEFT OUTER JOIN
    (
    select playerID, sum(G) as total_games_fielded, sum(InnOuts) as total_time_in_field_with_outs, 
    sum(PO) as total_putouts, sum(E) as total_errors, sum(DP) as total_double_plays
    from Fielding
    group by playerID
    ) 
    fielding on fielding.playerID = hfi.playerID
    LEFT OUTER JOIN 
        dominant_team_per_player t on t.playerID = hfi.playerID
    LEFT OUTER JOIN 
        dominant_league_per_player l on l.playerID = hfi.playerID
"""

con = sqlite3.connect(DATABASE)
df_p2k = pandas.read_sql(monster_query, con)
con.close()

## getting an intial view of the data for validation
df_p2k.head(10)
df_p2k.columns

# dropping duplicate playerID columns
df_p2k.drop('playerID',  1, inplace = True)

df_p2k.columns
#Index([u'hfi.playerID', u'hfi.inducted', u'playerID', u'total_at_bats', 
#u'total_hits', u'total_runs', u'total_home_runs', u'total_stolen_bases', 
#u'total_RBI', u'total_caught_stealing', u'total_hitter_strikeouts', 
#u'total_intentional_walks', u'playerID', u'total_games_pitched', u'total_shutouts', 
#u'total_saves', u'total_outs_pitched', u'total_pitching_hits', u'total_pitching_earned_runs',
#u'total_pitcher_strikeouts', u'average_ERA', u'total_wild_pitches', u'total_hit_by_pitch',
# u'total_games_finished', u'total_runs_allowed', u'playerID', u'total_games_fielded', 
# u'total_time_in_field_with_outs', u'total_putouts', u'total_errors', u'total_double_plays', 
# u'playerID', u'teamID', u'playerID', u'lgID'], dtype='object')

#rename columns
df_p2k.rename(columns={'hfi.playerID': 'playerID', 'hfi.inducted': 'inducted'}, inplace=True)
df_p2k.head()

#############
## repeating the same preprocessing from the previous lesson
############

## splitting out the explanatory features 
explanatory_features = [col for col in df_p2k.columns if col not in ['playerID', 'inducted']]
explanatory_df = df_p2k[explanatory_features]

# dropping rows with no data.
explanatory_df.dropna(how='all', inplace = True) 

# extracting column names 
explanatory_colnames = explanatory_df.columns
print explanatory_colnames

## doing the same for response
response_series = df_p2k.inducted
response_series.dropna(how='all', inplace = True) 



missing_rows = response_series.index[~response_series.index.isin(explanatory_df.index)]
print missing_rows
#Int64Index([27, 49, 61, 83, 91, 100, 111, 136, 145, 147, 158, 175, 178, 190, 191, 203,
#            212, 220, 228, 237, 239, 247, 266, 267, 273], dtype='int64')



#maybe match up explanatory_df and response_series by removing missing rows)
response_series = response_series.drop(response_series.index[missing_rows])

#check missing_rows now
missing_rows = response_series.index[~response_series.index.isin(explanatory_df.index)]
print missing_rows
#Int64Index([], dtype='int64')

### now, let's seperate the numeric explanatory data from the string data
string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']

######
# now, let's find if any of the categorical features need 'binnng'
#####
# first, fill the NANs in the feature (this lets us see if there are features
# that are all NANs, as they will show up as all 'Nothing' when we start binning or look for features with no variation)
string_features = string_features.fillna('Nothing')


#use the cleanup functions previously defined    
string_features = cleanup_data(string_features)

#check normalized variables
string_features.teamID.value_counts(normalize = True)
#NYN      0.050781
#TOR      0.050781
#SLN      0.050781
#SFN      0.046875
#Other    0.046875
#CIN      0.046875
#LAN      0.042969
#DET      0.042969
#OAK      0.042969
#NYA      0.042969
#MIN      0.039062
#MON      0.039062
#TEX      0.039062
#ATL      0.035156
#BOS      0.035156
#CHA      0.035156
#PHI      0.035156
#BAL      0.031250
#CHN      0.031250
#PIT      0.031250
#KCA      0.027344
#SEA      0.027344
#CLE      0.027344
#CAL      0.027344
#HOU      0.023438
#SDN      0.019531
#COL      0.015625
#ML4      0.015625
#dtype: float64

string_features.lgID.value_counts(normalize = True)
#NL    0.507812
#AL    0.492188
#dtype: float64

#Lookin' good onto next step


encoded_data = get_binary_values(string_features)
# let's verify that the encoding occured.
encoded_data.head()
#Encoding has occured
encoded_data.columns
#Index([u'teamID_ATL', u'teamID_BAL', u'teamID_BOS', u'teamID_CAL', u'teamID_CHA', 
#u'teamID_CHN', u'teamID_CIN', u'teamID_CLE', u'teamID_COL', u'teamID_DET', 
#u'teamID_HOU', u'teamID_KCA', u'teamID_LAN', u'teamID_MIN', u'teamID_ML4', 
#u'teamID_MON', u'teamID_NYA', u'teamID_NYN', u'teamID_OAK', u'teamID_Other', 
#u'teamID_PHI', u'teamID_PIT', u'teamID_SDN', u'teamID_SEA', u'teamID_SFN', 
#u'teamID_SLN', u'teamID_TEX', u'teamID_TOR', u'lgID_AL', u'lgID_NL'], dtype='object')


print numeric_features.columns
#Index([u'total_at_bats', u'total_hits', u'total_runs', u'total_home_runs', 
#u'total_stolen_bases', u'total_RBI', u'total_caught_stealing', u'total_hitter_strikeouts', 
#u'total_intentional_walks', u'total_games_pitched', u'total_shutouts', u'total_saves',
# u'total_outs_pitched', u'total_pitching_hits', u'total_pitching_earned_runs', u'total_pitcher_strikeouts', 
# u'average_ERA', u'total_wild_pitches', u'total_hit_by_pitch', u'total_games_finished', u'total_runs_allowed', 
# u'total_games_fielded', u'total_time_in_field_with_outs', u'total_putouts', 
#u'total_errors', u'total_double_plays'], dtype='object')


from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

## now that we've encoded our qualitative variables and filled the NaNs in our numeric variables, let's merge both DataFrames back together.

explanatory_df = pandas.concat([numeric_features, encoded_data],axis = 1)
explanatory_df.head()
explanatory_df.columns
#Index([u'total_at_bats', u'total_hits', u'total_runs', u'total_home_runs', 
#u'total_stolen_bases', u'total_RBI', u'total_caught_stealing', 
#u'total_hitter_strikeouts', u'total_intentional_walks', u'total_games_pitched', 
#u'total_shutouts', u'total_saves', u'total_outs_pitched', u'total_pitching_hits', 
#u'total_pitching_earned_runs', u'total_pitcher_strikeouts', u'average_ERA', 
#u'total_wild_pitches', u'total_hit_by_pitch', u'total_games_finished', 
#u'total_runs_allowed', u'total_games_fielded', u'total_time_in_field_with_outs', 
#u'total_putouts', u'total_errors', u'total_double_plays', u'teamID_ATL', 
#u'teamID_BAL', u'teamID_BOS', u'teamID_CAL', u'teamID_CHA', u'teamID_CHN', 
#u'teamID_CIN', u'teamID_CLE', u'teamID_COL', u'teamID_DET', u'teamID_HOU', 
#u'teamID_KCA', u'teamID_LAN', u'teamID_MIN', u'teamID_ML4', u'teamID_MON', 
#u'teamID_NYA', u'teamID_NYN', u'teamID_OAK', u'teamID_Other', u'teamID_PHI', 
#u'teamID_PIT', u'teamID_SDN', u'teamID_SEA', u'teamID_SFN', u'teamID_SLN', u'teamID_TEX', 
#u'teamID_TOR', u'lgID_AL', u'lgID_NL'], dtype='object')


    
zero_var_dict = find_zero_var(explanatory_df)

print zero_var_dict['toKeep']
#['total_at_bats', 'total_hits', 'total_runs', 'total_home_runs', 
#'total_stolen_bases', 'total_RBI', 'total_caught_stealing', 
#'total_hitter_strikeouts', 'total_intentional_walks', 
#'total_games_pitched', 'total_saves', 'total_outs_pitched', 'total_pitching_hits', 
#'total_pitching_earned_runs', 'total_pitcher_strikeouts', 'average_ERA',
#'total_wild_pitches', 'total_hit_by_pitch', 'total_games_finished', 'total_runs_allowed', 
#'total_games_fielded', 'total_time_in_field_with_outs', 'total_putouts', 'total_errors', 
#'total_double_plays', u'teamID_BAL', u'teamID_BOS', u'teamID_BRO', u'teamID_BSN',
# u'teamID_CAL', u'teamID_CHA', u'teamID_CHN', u'teamID_CIN', u'teamID_CLE', u'teamID_DET',
# u'teamID_HOU', u'teamID_KCA', u'teamID_LAN', u'teamID_MIN', u'teamID_ML4', u'teamID_MON',
# u'teamID_NY1', u'teamID_NYA', u'teamID_NYN', 'teamID_Other', u'teamID_PHA', u'teamID_PHI', 
# u'teamID_PIT', u'teamID_SFN', u'teamID_SLA', u'teamID_SLN', u'teamID_WS1', u'lgID_AL', 
# u'lgID_NL', 'lgID_Other']



print zero_var_dict['toDelete']
#[]

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


corr_dict = find_perfect_corr(explanatory_df)
print corr_dict
#{'corrGroupings': [['total_pitcher_strikeouts', 'total_shutouts'], [u'lgID_NL', u'lgID_AL']], 
#'toRemove': ['total_shutouts', u'lgID_AL']}

 
explanatory_df.drop(corr_dict['toRemove'], 1, inplace = True)


##############
# scaling data
#############

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
#got error ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
#Check for rows in explanatory_df and not in response_series
missing_rows = response_series.index[~response_series.index.isin(explanatory_df.index)]
print missing_rows
#Int64Index([], dtype='int64')
#Nothing in response series that isn't in explanatory_df

missing_rows = explanatory_df.index[~explanatory_df.index.isin(response_series.index)]
print missing_rows
#Int64Index([27, 49, 61, 83, 91, 100, 111, 136, 145, 147, 158, 175, 178, 190,
#            191, 203, 212, 220, 228, 237, 239, 247], dtype='int64')


#drop those rows
explanatory_df = explanatory_df.drop(df.index[missing_rows])
#drop any rows with NaN
explanatory_df = explanatory_df.dropna(axis=0)
missing_rows = response_series.index[~response_series.index.isin(explanatory_df.index)]
print missing_rows
#Int64Index([256, 257, 258, 259, 260, 261, 262, 263, 264, 
#265, 268, 269, 270, 271, 272, 274, 275, 276, 277, 278, 279, 280], dtype='int64')

response_series = response_series.drop(df.index[missing_rows])

#decided to fill nan's with zero
#explanatory_df = explanatory_df.fillna(0)
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


# create your tree based estimator
decision_tree = TreeClassifierWithCoef(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, 
                                       min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)

## set up the estimator. Score by AUC
rfe_cv = RFECV(estimator=decision_tree, step=1, cv=10,
              scoring='roc_auc', verbose = 1)
rfe_cv.fit(explanatory_df, response_series)

#RFECV(cv=10,
#   estimator=TreeClassifierWithCoef(compute_importances=None, criterion='gini',
#            max_depth=None, max_features=None, max_leaf_nodes=None,
#            min_density=None, min_samples_leaf=2, min_samples_split=2,
#            random_state=1, splitter='best'),
#   estimator_params={}, loss_func=None, scoring='roc_auc', step=1,
#   verbose=1)



print "Optimal number of features :{0} of {1} considered".format(rfe_cv.n_features_,len(explanatory_df.columns))
#Optimal number of features :32 of 55 considered

# printing out scores as we increase the number of features -- the farter
# down the list, the higher the number of features considered.
print rfe_cv.grid_scores_
#[ 0.55190789  0.48630482  0.49857456  0.46311404  0.43032895  0.44591009
#  0.45746711  0.45850877  0.52163377  0.47472588  0.54025219  0.51894737
#  0.53924342  0.54566886  0.53998904  0.54058114  0.51432018  0.55164474
#  0.53924342  0.56498904  0.52935307  0.56539474  0.55561404  0.54662281
#  0.55692982  0.53509868  0.5551864   0.55630482  0.5510307   0.53093202
#  0.5483443   0.59656798  0.51610746  0.54697368  0.58277412  0.52677632
#  0.53269737  0.53172149  0.5389693   0.5477193   0.58986842  0.55122807
#  0.54938596  0.53188596  0.56572368  0.51947368  0.55423246  0.57769737
#  0.56194079  0.55838816  0.59588816  0.5922807   0.55089912  0.53548246]



## let's plot out the results
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (ROC_AUC)")
plt.plot(range(1, len(rfe_cv.grid_scores_) + 1), rfe_cv.grid_scores_)
plt.show()
# notice you could have just as well have included the 10 most important 
# features and received similar accuracy.
#see hw9_p2k_rfe_grid_search_scores.png

# you can pull out the features used this way:
features_used = explanatory_df.columns[rfe_cv.get_support()]
print features_used
#Index([u'total_at_bats', u'total_hits', u'total_runs', u'total_home_runs', 
#u'total_stolen_bases', u'total_RBI', u'total_caught_stealing', u'total_hitter_strikeouts', 
#u'total_intentional_walks', u'total_games_pitched', u'total_saves', 
#u'total_pitcher_strikeouts', u'total_hit_by_pitch', u'total_games_finished', 
#u'total_games_fielded', u'total_time_in_field_with_outs', u'total_putouts', 
#u'total_errors', u'teamID_CHA', u'teamID_CHN', u'teamID_CIN', u'teamID_CLE', 
#u'teamID_COL', u'teamID_DET', u'teamID_HOU', u'teamID_KCA', u'teamID_MIN', 
#u'teamID_MON', u'teamID_Other', u'teamID_SLN', u'teamID_TOR', u'lgID_NL'], dtype='object')


#you can extract the final selected model object this way:
final_estimator_used = rfe_cv.estimator_
print final_estimator_used
#TreeClassifierWithCoef(compute_importances=None, criterion='gini',
#            max_depth=None, max_features=None, max_leaf_nodes=None,
#            min_density=None, min_samples_leaf=2, min_samples_split=2,
#            random_state=1, splitter='best')


# you can also combine RFE with grid search to find the tuning 
# parameters and features that optimize model accuracy metrics.
# do this by passing the RFECV object to GridSearchCV.
from sklearn.grid_search import  GridSearchCV

# doing this for a small range so I can show you the answer in a reasonable
# amount of time.
depth_range = range(4, 6)
# notice that in param_grid, I need to prefix estimator__ to my paramerters.
param_grid = dict(estimator__max_depth=depth_range)
# notice that this will take quite a bit longer to compute.
rfe_grid_search = GridSearchCV(rfe_cv, param_grid, cv=10, scoring='roc_auc')
rfe_grid_search.fit(explanatory_df, response_series)

print rfe_grid_search.grid_scores_

#[mean: 0.40852, std: 0.16571, params: {'estimator__max_depth': 4}, 
# mean: 0.43497, std: 0.16575, params: {'estimator__max_depth': 5}]



rfe_grid_search.best_params_
#{'estimator__max_depth': 5}

# let's plot out the results.
grid_mean_scores = [score[1] for score in rfe_grid_search.grid_scores_]
print grid_mean_scores
#[0.40851514469935518, 0.43497150997150996]


plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.plot(rfe_grid_search.best_params_['estimator__max_depth'], rfe_grid_search.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')
plt.grid(True)
#hw9_p2k_tree_cv_scores.png

# now let's pull out the winning estimator.
best_decision_tree_rfe_grid = rfe_grid_search.best_estimator_
print best_decision_tree_rfe_grid
#RFECV(cv=10,
#   estimator=TreeClassifierWithCoef(compute_importances=None, criterion='gini',
#            max_depth=5, max_features=None, max_leaf_nodes=None,
#            min_density=None, min_samples_leaf=2, min_samples_split=2,
#            random_state=1, splitter='best'),
#   estimator_params={}, loss_func=None, scoring='roc_auc', step=1,
#   verbose=1)



features_used_rfecv_grid = explanatory_df.columns[best_decision_tree_rfe_grid.get_support()]
print features_used_rfecv_grid
#Index([u'total_runs', u'total_games_finished', u'total_putouts', u'teamID_SLN'], dtype='object')

#The best estimators for hall of fame induction are total_runs, total_games_finished, total_putouts, and teamID_SLN