'''
jonblum
2015-02-17
datbos05
class 9 hw
'''



# basics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# i/o
import sqlite3

# models
from sklearn.tree import DecisionTreeClassifier

# tools
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import RFECV


############################
# Helper methods & classes #
############################

def bin_categorical(df, cutoffPercent = .01):
    """Lumps categorcal features with frequency < cutoffPercent into 'Other'."""
    for col in df:
        sizes = df[col].value_counts(normalize = True)
        # get the names of the levels that make up less than 1% of the dataset
        values_to_delete = sizes[sizes<cutoffPercent].index
        df[col].ix[df[col].isin(values_to_delete)] = "Other"
    return df

def get_binary_values(df):
    """Binary encoding of categorical features using Pandas' get_dummies."""
    all_columns = pd.DataFrame(index = df.index)
    for col in df.columns:
        data = pd.get_dummies(df[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pd.concat([all_columns, data], axis=1)
    return all_columns

def find_zero_var(df):
    """Finds columns in the dataframe with zero variance -- ie those
        with the same value in every observation."""
    to_keep = []
    to_delete = []
    for col in df:
        if len(df[col].value_counts()) > 1:
            to_keep.append(col)
        else:
            to_delete.append(col)
        ##
    return {'to_keep':to_keep, 'to_delete':to_delete}

def find_perfect_corr(df):
    """Finds columns that are eother positively or negatively perfectly correlated (with correlations of +1 or -1), and creates a dict
        that includes which columns to drop so that each remaining column is independent."""
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

class TreeClassifierWithCoef(DecisionTreeClassifier):
    """Tree Classifer with coeff_ (derived from feature_importances_) for use with RFECV."""
    def fit(self, *args, **kwargs):
        super(DecisionTreeClassifier, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


#########################




# grab familiar features, joined with primary team from class and additional categorical feature 'school'
conn = sqlite3.connect('/Users/jon/Documents/code/datbos05/data/lahman2013.sqlite')
sql = '''
SELECT hof.playerID, dtpp.teamID, sp.schoolID, b.totalCareerHits, b.careerBattingAvg, p.avgCareerERA, f.careerFieldingPercentage, MAX(hof.inducted) AS inducted
FROM HallOfFame hof
LEFT JOIN
        (SELECT playerID, SUM(H) as totalCareerHits, (SUM(H)*1.0) / SUM(AB) as careerBattingAvg
        FROM Batting
        GROUP BY playerID) b
ON b.playerID = hof.playerID
LEFT JOIN
        (SELECT playerID, AVG(ERA) as avgCareerERA
        FROM Pitching
        GROUP BY playerID) p
ON  p.playerID = hof.playerID
LEFT JOIN
        (SELECT playerID, 1.0 * (SUM(PO) + SUM(A)) / (SUM(PO) + SUM(A) + SUM(E)) as careerFieldingPercentage
        FROM Fielding
        GROUP BY playerID) f
ON f.playerID = hof.playerID
LEFT JOIN dominant_team_per_player dtpp
ON dtpp.playerID = hof.playerID
LEFT JOIN SchoolsPlayers sp
ON sp.playerID = hof.playerID
WHERE hof.yearID < 2000 AND hof.category = 'Player'
GROUP BY hof.playerID;
'''

df = pd.read_sql(sql,conn)
conn.close()


# Cleanup


# Y->1 N->0
df['inducted_boolean'] = 0
df.inducted_boolean[df.inducted == 'Y'] = 1
df.drop('inducted',1,inplace=True)

explanatory_features = [col for col in df.columns if col not in ['playerID', 'inducted_boolean']]
explanatory_df = df[explanatory_features]

# drop rows with no data at all (older players from other leauges)
explanatory_df.dropna(how='all', inplace = True)

# doing the same for response
response_series = df.inducted_boolean
response_series.dropna(how='all', inplace = True)

# dropping from response anything that was dropped from explanatory and vice versa
response_series.drop(response_series.index[~response_series.index.isin(explanatory_df.index)],0,inplace=True)
explanatory_df.drop(explanatory_df.index[~explanatory_df.index.isin(response_series.index)],0,inplace=True)

len(response_series) == len(explanatory_df)
# True


# 1. SPLIT DATA INTO CATEGORICAL AND NUMERIC DATA

string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']

print string_features.columns
# Index([u'teamID', u'schoolID'], dtype='object')
print numeric_features.columns
# Index([u'totalCareerHits', u'careerBattingAvg', u'avgCareerERA', u'careerFieldingPercentage'], dtype='object')


# 2. FILL NUMERIC NANS THROUGH IMPUTATION

imputer = Imputer(missing_values='NaN', strategy='median',axis=0)
imputer.fit(numeric_features)

# impute and put back into a dataframe with the columns and indices from before
numeric_features = pd.DataFrame(imputer.transform(numeric_features), columns=numeric_features.columns, index=numeric_features.index)


# 3. FILL CATEGORICAL NANS WITH 'NOTHING'

string_features = string_features.fillna('Nothing')

# 4. DETECT LOW-FREQUENCY LEVELS IN CATEGORICAL FEATURES AND BIN THEM UNDER 'OTHER'

# use 0.5% to keep at least a few schools as separate bins
string_features = bin_categorical(string_features,cutoffPercent=0.005)
print string_features.schoolID.value_counts(normalize = True)
'''
Nothing      0.748337
Other        0.232816
usc          0.007761
stmarysca    0.005543
arizonast    0.005543
'''

# 5. ENCODE EACH CATEGORICAL VARIABLE INTO A SEQUENCE OF BINARY VARIABLES

encoded_string_features  =  get_binary_values(string_features)

print encoded_string_features.columns
'''
Index([u'teamID_ATL', u'teamID_BAL', u'teamID_BOS', u'teamID_BRO', u'teamID_BSN', u'teamID_CAL', u'teamID_CHA', u'teamID_CHN', u'teamID_CIN', u'teamID_CLE', u'teamID_DET', u'teamID_HOU', u'teamID_KCA', u'teamID_LAN', u'teamID_MIN', u'teamID_ML1', u'teamID_ML4', u'teamID_MON', u'teamID_NY1', u'teamID_NYA', u'teamID_NYN', u'teamID_OAK', u'teamID_Other', u'teamID_PHA', u'teamID_PHI', u'teamID_PIT', u'teamID_SDN', u'teamID_SFN', u'teamID_SLA', u'teamID_SLN', u'teamID_TEX', u'teamID_WS1', u'schoolID_Nothing', u'schoolID_Other', u'schoolID_arizonast', u'schoolID_stmarysca', u'schoolID_usc'], dtype='object')
'''

# 6. MERGE YOUR ENCODED CATEGORICAL DATA WITH YOUR NUMERIC DATA

explanatory_df = pd.concat([numeric_features, encoded_string_features], axis=1)

explanatory_df.columns
'''
Index([u'totalCareerHits', u'careerBattingAvg', u'avgCareerERA', u'careerFieldingPercentage', u'teamID_ATL', u'teamID_BAL', u'teamID_BOS', u'teamID_BRO', u'teamID_BSN', u'teamID_CAL', u'teamID_CHA', u'teamID_CHN', u'teamID_CIN', u'teamID_CLE', u'teamID_DET', u'teamID_HOU', u'teamID_KCA', u'teamID_LAN', u'teamID_MIN', u'teamID_ML1', u'teamID_ML4', u'teamID_MON', u'teamID_NY1', u'teamID_NYA', u'teamID_NYN', u'teamID_OAK', u'teamID_Other', u'teamID_PHA', u'teamID_PHI', u'teamID_PIT', u'teamID_SDN', u'teamID_SFN', u'teamID_SLA', u'teamID_SLN', u'teamID_TEX', u'teamID_WS1', u'schoolID_Nothing', u'schoolID_Other', u'schoolID_arizonast', u'schoolID_stmarysca', u'schoolID_usc'], dtype='object')
'''

# 7. REMOVE FEATURES WITH NO VARIATION

keep_delete = find_zero_var(explanatory_df)
print keep_delete['to_delete']
# []   (all features have some variation)


# 8. REMOVE PERFECTLY-CORELATED FEATURES

corr = find_perfect_corr(explanatory_df)
print corr['corrGroupings'], corr['toRemove']
# [] [] (no perfectly-correlated features)

# 9. SCALE YOUR DATA WITH ZERO MEAN AND UNIT VARIANCE

scaler = StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pd.DataFrame(scaler.transform(explanatory_df),columns=explanatory_df.columns,index=explanatory_df.index)

explanatory_df.totalCareerHits.describe()
'''
count    9.020000e+02
mean    -2.363224e-17
std      1.000555e+00
min     -1.250037e+00
25%     -1.026367e+00
50%      2.516063e-03
75%      7.510474e-01
max      3.480102e+00
Name: totalCareerHits, dtype: float64
'''


# 10. PERFORM GRID SEARCH AND RFE ON YOUR DATA TO FIND THE OPTIMAL ESTIMATOR FOR YOUR DATA
# 11. TRAIN AND TEST YOUR MODEL ON THE DATA

decision_tree = TreeClassifierWithCoef()
rfecv = RFECV(estimator=decision_tree, step=1,cv=10,scoring='roc_auc')

# tuning parameters
depth_range = range(1,7) #21
min_split_range = range(2,11) #11
tuning_params = dict(estimator__max_depth=depth_range, estimator__min_samples_split=min_split_range)

# GridSearh on the outside, RFECV on the inside
rfecv_grid_search = GridSearchCV(rfecv, tuning_params, cv=10, scoring='roc_auc', n_jobs=-1)

rfecv_grid_search.fit(explanatory_df, response_series)
# Good long think...


print rfecv_grid_search.best_params_
# {'estimator__max_depth': 4, 'estimator__min_samples_split': 5}

best_dt_model = rfecv_grid_search.best_estimator_

# let's plot out the resul
used_features = explanatory_df.columns[best_dt_model.get_support()]
print used_features
# Index([u'totalCareerHits', u'careerBattingAvg', u'avgCareerERA'], dtype='object')

best_dt_model = rfecv_grid_search.best_estimator_


# 12 RECLEAN, SCALE AND AND ENCODE INCOMING UNLABELED DATA

# Get >= 2000 data

conn = sqlite3.connect('/Users/jon/Documents/code/datbos05/data/lahman2013.sqlite')
sql = '''
SELECT hof.playerID, dtpp.teamID, sp.schoolID, b.totalCareerHits, b.careerBattingAvg, p.avgCareerERA, f.careerFieldingPercentage, MAX(hof.inducted) AS inducted
FROM HallOfFame hof
LEFT JOIN
        (SELECT playerID, SUM(H) as totalCareerHits, (SUM(H)*1.0) / SUM(AB) as careerBattingAvg
        FROM Batting
        GROUP BY playerID) b
ON b.playerID = hof.playerID
LEFT JOIN
        (SELECT playerID, AVG(ERA) as avgCareerERA
        FROM Pitching
        GROUP BY playerID) p
ON  p.playerID = hof.playerID
LEFT JOIN
        (SELECT playerID, 1.0 * (SUM(PO) + SUM(A)) / (SUM(PO) + SUM(A) + SUM(E)) as careerFieldingPercentage
        FROM Fielding
        GROUP BY playerID) f
ON f.playerID = hof.playerID
LEFT JOIN dominant_team_per_player dtpp
ON dtpp.playerID = hof.playerID
LEFT JOIN SchoolsPlayers sp
ON sp.playerID = hof.playerID
WHERE hof.yearID >= 2000 AND hof.category = 'Player'
GROUP BY hof.playerID;
'''

post_2000_df = pd.read_sql(sql,conn)
conn.close()

# Y->1 N->0
post_2000_df['inducted_boolean'] = 0
post_2000_df.inducted_boolean[post_2000_df.inducted == 'Y'] = 1
post_2000_df.drop('inducted',1,inplace=True)

post_2000_explanatory_features = [col for col in post_2000_df.columns if col not in ['playerID', 'inducted_boolean']]
post_2000_explanatory_df = post_2000_df[explanatory_features]

# drop rows with no data at all (older players from other leauges)
post_2000_explanatory_df.dropna(how='all', inplace = True)

# doing the same for response
post_2000_response_series = post_2000_df.inducted_boolean
post_2000_response_series.dropna(how='all', inplace = True)

# dropping from response anything that was dropped from explanatory and vice versa
post_2000_response_series.drop(post_2000_response_series.index[~post_2000_response_series.index.isin(post_2000_explanatory_df.index)],0,inplace=True)
post_2000_explanatory_df.drop(post_2000_explanatory_df.index[~post_2000_explanatory_df.index.isin(post_2000_response_series.index)],0,inplace=True)

len(post_2000_response_series) == len(post_2000_explanatory_df)
# True

# TODO: THE REST

