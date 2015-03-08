'''
jonblum
2015-02-24
datbos05
class 10 hw
'''

# division
from __future__ import division

# basics
import numpy as np
import pandas as pd

# i/o
import sqlite3

# models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# tools
from numpy import arange
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score



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



# same data as HW09
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


# Same cleanup / prep as HW09


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


# split features
string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']

print string_features.columns
# Index([u'teamID', u'schoolID'], dtype='object')
print numeric_features.columns
# Index([u'totalCareerHits', u'careerBattingAvg', u'avgCareerERA', u'careerFieldingPercentage'], dtype='object')

# impute and put back into a dataframe with the columns and indices from before
imputer = Imputer(missing_values='NaN', strategy='median',axis=0)
imputer.fit(numeric_features)
numeric_features = pd.DataFrame(imputer.transform(numeric_features), columns=numeric_features.columns, index=numeric_features.index)

# fill categorical NaNs
string_features = string_features.fillna('Nothing')

# detect low-frequency features and bin as other
# use 0.5% to keep at least a few schools as separate bins
string_features = bin_categorical(string_features,cutoffPercent=0.005)

string_features.columns[0]
string_features.schoolID.unique()

# grab categories for use with the out-of-sample data
orig_categories = 	{}
for col in string_features.columns:
	orig_categories[col] = string_features[col].unique()


print string_features.schoolID.value_counts(normalize = True)
'''
Nothing      0.748337
Other        0.232816
usc          0.007761
stmarysca    0.005543
arizonast    0.005543
'''

# encode categorical features
encoded_string_features  =  get_binary_values(string_features)

print encoded_string_features.columns
'''
Index([u'teamID_ATL', u'teamID_BAL', u'teamID_BOS', u'teamID_BRO', u'teamID_BSN', u'teamID_CAL', u'teamID_CHA', u'teamID_CHN', u'teamID_CIN', u'teamID_CLE', u'teamID_DET', u'teamID_HOU', u'teamID_KCA', u'teamID_LAN', u'teamID_MIN', u'teamID_ML1', u'teamID_ML4', u'teamID_MON', u'teamID_NY1', u'teamID_NYA', u'teamID_NYN', u'teamID_OAK', u'teamID_Other', u'teamID_PHA', u'teamID_PHI', u'teamID_PIT', u'teamID_SDN', u'teamID_SFN', u'teamID_SLA', u'teamID_SLN', u'teamID_TEX', u'teamID_WS1', u'schoolID_Nothing', u'schoolID_Other', u'schoolID_arizonast', u'schoolID_stmarysca', u'schoolID_usc'], dtype='object')
'''

# merge features
explanatory_df = pd.concat([numeric_features, encoded_string_features], axis=1)

print explanatory_df.columns
'''
Index([u'totalCareerHits', u'careerBattingAvg', u'avgCareerERA', u'careerFieldingPercentage', u'teamID_ATL', u'teamID_BAL', u'teamID_BOS', u'teamID_BRO', u'teamID_BSN', u'teamID_CAL', u'teamID_CHA', u'teamID_CHN', u'teamID_CIN', u'teamID_CLE', u'teamID_DET', u'teamID_HOU', u'teamID_KCA', u'teamID_LAN', u'teamID_MIN', u'teamID_ML1', u'teamID_ML4', u'teamID_MON', u'teamID_NY1', u'teamID_NYA', u'teamID_NYN', u'teamID_OAK', u'teamID_Other', u'teamID_PHA', u'teamID_PHI', u'teamID_PIT', u'teamID_SDN', u'teamID_SFN', u'teamID_SLA', u'teamID_SLN', u'teamID_TEX', u'teamID_WS1', u'schoolID_Nothing', u'schoolID_Other', u'schoolID_arizonast', u'schoolID_stmarysca', u'schoolID_usc'], dtype='object')
'''

# remove features with no variation
keep_delete = find_zero_var(explanatory_df)
print keep_delete['to_delete']
# []   (all features have some variation)


# remove features with perfect correlation
corr = find_perfect_corr(explanatory_df)
print corr['corrGroupings'], corr['toRemove']
# [] [] (no perfectly-correlated features)

# scale features
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

###############
# CLASSIFIERS #
###############

# RANDOM FOREST

rf = RandomForestClassifier()
trees_range = range(10, 600, 10)
rf_grid_params = dict(n_estimators = trees_range)
rf_grid = GridSearchCV(rf, rf_grid_params, cv=10, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(explanatory_df,response_series)
print rf_grid.best_params_
# {'n_estimators': 190}
rf_estimator = rf_grid.best_estimator_


# GRADIENT BOOSTING

gbm = GradientBoostingClassifier()
learning_rate_range = arange(0.01, 0.4, 0.02)
subsampling_range = arange(0.25, 1, 0.25)
n_estimators_range = range(25, 100, 25)
gbm_grid_params = dict(learning_rate = learning_rate_range, n_estimators = n_estimators_range, subsample = subsampling_range)
gbm_grid = GridSearchCV(gbm, gbm_grid_params, cv=10, scoring='roc_auc', n_jobs = -1)
gbm_grid.fit(explanatory_df, response_series)
print gbm_grid.best_params_
# {'n_estimators': 75, 'subsample': 0.75, 'learning_rate': 0.14999999999999999}
gbm_estimator = gbm_grid.best_estimator_


# NEURAL NETWORK

lc = LogisticRegression()
nn = BernoulliRBM()
nc = Pipeline(steps=[('neural_net', nn), ('logistic_classifier', lc)])
learning_rate_range = arange(0.01, 0.3, 0.1)
iteration_range = range(30, 60, 10)
components_range = range(200, 600, 100)
nn_grid_params = dict(neural_net__n_components = components_range, neural_net__learning_rate = learning_rate_range, neural_net__n_iter = iteration_range)
nn_grid = GridSearchCV(nc, nn_grid_params, cv=5, scoring='roc_auc', n_jobs=-1)
nn_grid.fit(explanatory_df, response_series)
print nn_grid.best_params_
# {'neural_net__n_components': 500, 'neural_net__learning_rate': 0.01, 'neural_net__n_iter': 40}

nn_estimator = nn_grid.best_estimator_


# DECISION TREE

dt = DecisionTreeClassifier()
depth_range = range(1,21)
min_split_range = range(2,11)
dt_grid_params = dict(max_depth=depth_range, min_samples_split=min_split_range)
dt_grid = GridSearchCV(dt,dt_grid_params, cv=10, scoring='roc_auc', n_jobs=-1)
dt_grid.fit(explanatory_df, response_series)
print dt_grid.best_params_
# {'min_samples_split': 2, 'max_depth': 3}
dt_estimator = dt_grid.best_estimator_

# LOGISTIC REGRESSION

lc2 = LogisticRegression()
c_range = arange(0.1,2.0,0.05)
lc_grid_params = dict(C=c_range)
lc_grid = GridSearchCV(lc2,lc_grid_params, cv=10, scoring='roc_auc', n_jobs=-1)
lc_grid.fit(explanatory_df, response_series)
print lc_grid.best_params_
# {'C': 0.15000000000000002}
lc_estimator = lc_grid.best_estimator_


# K NEAREST NEIGHBORS

knn = KNeighborsClassifier()
k_range = range(1,100,2)
knn_grid_params = dict(n_neighbors=k_range)
knn_grid = GridSearchCV(knn,knn_grid_params, cv=10, scoring='roc_auc', n_jobs=-1)
knn_grid.fit(explanatory_df, response_series)
print knn_grid.best_params_
# {'n_neighbors': 63}
knn_estimator = knn_grid.best_estimator_


###############
# ACCURACY    #
###############


rf_roc_scores = cross_val_score(rf_estimator, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print rf_roc_scores.mean()
# 0.87326417488

gbm_roc_scores = cross_val_score(gbm_estimator, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print gbm_roc_scores.mean()
# 0.862432906405

nn_roc_scores = cross_val_score(nn_estimator, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print nn_roc_scores.mean()
# 0.83144972136

dt_roc_scores = cross_val_score(dt_estimator, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print dt_roc_scores.mean()
# 0.795136200569

lc_roc_scores = cross_val_score(lc_estimator, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print lc_roc_scores.mean()
# 0.714901355828

knn_roc_scores = cross_val_score(knn_estimator, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print knn_roc_scores.mean()
# 0.732464634297

# Random Forest is best in this case, particularly considering the minimal tuning it requires.
