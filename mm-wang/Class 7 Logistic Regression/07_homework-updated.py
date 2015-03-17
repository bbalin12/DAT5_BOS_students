# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:50:12 2015

@author: Margaret
"""

import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import logit
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import  GridSearchCV


def cleanup_data(df, cutoffPercent = .01):
    for col in df:
        sizes = df[col].value_counts(normalize = True)
        # get the names of the levels that make up less than 1% of the dataset
        values_to_delete = sizes[sizes<cutoffPercent].index
        df[col].ix[df[col].isin(values_to_delete)] = "Other"
    return df
#

def get_binary_values(data_frame):
    """encodes cateogrical features in Pandas.
    """
    all_columns = pd.DataFrame( index = data_frame.index)
    for col in data_frame.columns:
        data = pd.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pd.concat([all_columns, data], axis=1)
    return all_columns
#
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
###


# putting a setting into pandas that lets you print out the entire
# DataFrame when you use the .head() method
pd.set_option('display.max_columns', None)


con = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')
query = """
SELECT (m.nameFirst||" "||nameLast) as p_name, m.height as height, m.weight as weight, m.bats as bats, 
m.throws as throws, inducted, bat_runs, bat_hits, at_bats, bat_homeruns, bat_strikes, 
bat_stolen, bat_baseballs, bat_intentwalks, pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA,
pitch_wild, pitch_saves, f_putouts, f_assists, f_errors FROM Master m
INNER JOIN
(SELECT pID, dom.teamID as dom_team, inducted, 
all_features.* FROM dominant_team_per_player dom
INNER JOIN
(SELECT h.playerID as pID, max(CASE WHEN h.inducted='Y' THEN 1 ELSE 0 END) as inducted, 
positions.* FROM HallofFame h
INNER JOIN
(SELECT f.playerID, f.PO as f_putouts, f.A as f_assists, f.E as f_errors, batpitch.* FROM Fielding f
LEFT JOIN
(SELECT b.playerID, b.lgID as bat_league, sum(b.R) as bat_runs, sum(b.H) as bat_hits, sum(b.CS) as bat_caught, 
sum(b.BB) as bat_baseballs, sum(b.AB) as at_bats, sum(b.HR) as bat_homeruns, sum(b.RBI) as bat_RBI, 
sum(b.SO) as bat_strikes, sum(b.SB) as bat_stolen, sum(b.IBB) as bat_intentwalks, sum(b.'2B') as bat_doubles, 
sum(b.'3B') as bat_triples,
sum(p.W) as pitch_wins, sum(p.SO) as pitch_strikes, sum(p.SHO) as pitch_shuts, avg(p.ERA) as pitch_ERA, 
sum(p.WP) as pitch_wild, sum(p.G) as pitch_games, sum(p.SV) as pitch_saves, sum(p.ER) as pitch_earnruns, 
sum(p.R) as pitch_runsallowed, sum(p.GF) as pitch_finish, sum(p.IPOuts) as pitch_outs, 
sum(p.HBP) as pitch_hits, sum(p.BAOpp) as pitch_opp_BA FROM Batting b
LEFT JOIN Pitching p on p.playerID = b.playerID
GROUP BY b.playerID) batpitch on batpitch.playerID = f.playerID
GROUP BY batpitch.playerID) positions
ON positions.playerID = h.playerID
WHERE h.yearID < 2000 AND h.yearID > 1965
GROUP BY h.playerID) all_features on pID = dom.playerID) all_data on pID = m.playerID
"""
df = pd.read_sql(query, con)
con.close()

df.drop('p_name',  1, inplace = True)


#################
### Preprocessing
#################

# splitting out the explanatory features 
explanatory_features = [col for col in df.columns if col not in ['p_name', 'inducted']]
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


# that are all NANs, as they will show up as all 'Nothing' when we start binning or look for features with no variation)
string_features = string_features.fillna('Nothing')
# cleaning up string features
string_features = cleanup_data(string_features)
# binarizing string features 
encoded_data = get_binary_values(string_features)
## imputing features
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pd.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

## pulling together numeric and encoded data.
explanatory_df = pd.concat([numeric_features, encoded_data],axis = 1)
explanatory_df.head()


#now, let's find features with no variance 
no_variation = find_zero_var(explanatory_df)
explanatory_df.drop(no_variation['toDelete'], inplace = True)

# deleting perfect correlation
no_correlation = find_perfect_corr(explanatory_df)
explanatory_df.drop(no_correlation['toRemove'], 1, inplace = True)

# scaling data
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pd.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)

# concatenanting response series and explanatory df
transform_df = pd.concat([response_series, explanatory_df], axis=1)

##################
# LOGISTICAL MODEL
##################
largey, largeX = dmatrices('inducted ~ height+weight+bat_runs+bat_hits+at_bats+bat_homeruns+\
                bat_stolen+bat_baseballs+pitch_wins+pitch_strikes+pitch_shuts+pitch_ERA+\
                pitch_wild+pitch_saves+f_putouts+f_assists+f_errors', 
                transform_df, return_type = "dataframe")
print largeX.columns
large_col = largeX.columns[1::]
largey = np.ravel(largey)

lg_explanatory_df = explanatory_df[large_col]

smally, smallX = dmatrices('inducted ~ bat_runs+at_bats+pitch_shuts+pitch_wins', 
                transform_df, return_type = "dataframe")
print smallX.columns
small_col = smallX.columns[1::]
smally = np.ravel(smally)

sm_explanatory_df = explanatory_df[small_col]

lg_model = LogisticRegression()
lg_model = lg_model.fit(largeX, largey)
sm_model = LogisticRegression()
sm_model = sm_model.fit(smallX, smally)

lg_model_score = lg_model.score(largeX, largey)
print "Large Model Score: %f" % lg_model_score
sm_model_score = sm_model.score(smallX, smally)
print "Small Model Score: %f" % sm_model_score

lg_coefficients = pd.DataFrame(zip(largeX.columns, np.transpose(lg_model.coef_)))
print "Large Coefficients"
print lg_coefficients
sm_coefficients = pd.DataFrame(zip(smallX.columns, np.transpose(sm_model.coef_)))
print "Small Coefficients"
print sm_coefficients


####################
### CROSS VALIDATION
####################

# Large Model
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold

cv = StratifiedKFold(response_series, n_folds=10)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    lg_model.fit(lg_explanatory_df.ix[train,], response_series.ix[train,])
    probabilities = pd.DataFrame(lg_model.predict_proba(lg_explanatory_df.ix[test,]))
    # Confusion Matrix
    predicted_values = lg_model.predict(lg_explanatory_df.ix[test])    
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
plt.title('Logistical Regression Large Model 10-Fold Cross Validation ROC')
plt.subplot()
plt.legend(bbox_to_anchor=(1.65,1.07))
plt.show()

# Small Model
cv = StratifiedKFold(response_series, n_folds=10)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    sm_model.fit(sm_explanatory_df.ix[train,], response_series.ix[train,])
    probabilities = pd.DataFrame(sm_model.predict_proba(sm_explanatory_df.ix[test,]))
    # Confusion Matrix
    predicted_values = sm_model.predict(sm_explanatory_df.ix[test])    
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
plt.title('Logistical Regression Small Model 10-Fold Cross Validation ROC')
plt.subplot()
plt.legend(bbox_to_anchor=(1.65,1.07))
plt.show()



##################################
### CLASS SEPARABLE CLASSIFICATION
##################################

model_small = logit('inducted ~ at_bats+bat_homeruns', 
               data = transform_df).fit()

logit_pars = model_small.params
intercept = -logit_pars[0] / logit_pars[1]
slope = -logit_pars[1] / logit_pars[1]

at_bats_i = transform_df['at_bats'][transform_df['inducted'] == 1]
at_bats_noi = transform_df['at_bats'][transform_df['inducted'] == 0]
hr_i = transform_df['bat_homeruns'][transform_df['inducted'] == 1]
hr_noi = transform_df['bat_homeruns'][transform_df['inducted'] == 0]
plt.figure(figsize = (12, 8))
plt.plot(at_bats_i, hr_i, '.', mec = 'purple', mfc = 'None', 
         label = 'Inducted')
plt.plot(at_bats_noi, hr_noi, '.', mec = 'orange', mfc = 'None', 
         label = 'Not Inducted')
plt.plot(np.arange(0, 10, 1), intercept + slope * np.arange(0, 10, 1),
         '-k', label = 'Separating line')
plt.ylim(0, 5)
plt.xlabel('At Bats')
plt.ylabel('Home Runs')
plt.legend(loc = 'best')