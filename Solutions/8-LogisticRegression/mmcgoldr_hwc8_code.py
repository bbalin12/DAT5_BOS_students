# -*- coding: utf-8 -*-
"""
Created on Thu Mar 05 18:55:58 2015
@author: mmcgoldr

GA CLASS 8 HOMEWORK (Logistic Regression)

Using the baseball dataset, build a logistic regression model that predicts who 
is likely to be inducted into Hall of Fame. Start with considering as many 
explanatory variables. What factors are signficant?  Reduce your explanatory 
variables to the ones that are significant. Compare performance between having 
many variables vs having a smaller set. Cross validate your model and print 
out the coeffecients of the best model. Considering any two features, generate 
a scatter plot with a class separable line showing the classification.

RESULTS SUMMARY

I start by considering the following variables: 
   Batting - at bats, runs, hits, home runs, strikeouts, 
   Pitching - wins, losses, shutouts, saves, earned runs, strikeouts
   Fielding - puts, double plays
   Dominant (most frequent) position
   Dominant (most frequent) team

I split my data into two sets: one before year 2000 for training and the other 
on/after 2000 for future predictions. From both sets, I remove records with 
all missing variables. I then bin and binarize the two categorical features, 
impute numeric missings with 0, and scale all variabless -- separately for the 
two datasets.

An initial recursive feature search with logistic regression and 10-fold 
cross-validation on the training set suggests that 18 features are important.  
The mean ROC AUC from cross-validation is 94.5%.

I then run logistic regression on the 18 features from the training set to obtain 
model summary results.  One team feature is removed immediately because it 
causes a single matrix error. The remaining 17 features, some of which are 
insignificant (p-value >= .05), yields a mean predicted probablity on future 
cases of 17.3%, which is greater than the actual induction rate (on/after 2000) 
of 12.1%.  Using a predicted probability of 0.5 as a cutoff, I classify future 
records into inducted (1) or not inducted (0) and then cross these class 
predictions with actual inductions. This yields an overall accuracy rate of 
85% and a true positive rate of 60%.

Finally, I reduce my model to only 13 significant predictors with p-values <0.05 
(shown below), which yields no change in pseudo-Rsq from the full model (both 
0.5808). The reduced model also yields a mean predicted probability on future 
cases of 21.6% (even higher than actual), an overall accuracy of 83.9% and a 
true positive rate of 56.7%. So there is a slight loss in accuracy and TPR when 
including only statistically significant predictors. However, if parsimony is 
impprovement, then the reduced model may be good enough.

                           Logit Regression Results                           
==============================================================================
Dep. Variable:               inducted   No. Observations:                  882
Model:                          Logit   Df Residuals:                      868
Method:                           MLE   Df Model:                           13
Date:                Sun, 08 Mar 2015   Pseudo R-squ.:                  0.5808
Time:                        22:38:16   Log-Likelihood:                -189.90
converged:                       True   LL-Null:                       -453.02
                                        LLR p-value:                3.894e-104
================================================================================
                   coef    std err          z      P>|z|      [95.0% Conf. Int.]
--------------------------------------------------------------------------------
Intercept       -2.6176      0.196    -13.361      0.000        -3.002    -2.234
b_atbat         -8.0738      1.167     -6.920      0.000       -10.360    -5.787
b_runs           2.8827      0.621      4.645      0.000         1.666     4.099
b_hits           8.1097      1.265      6.412      0.000         5.631    10.589
b_hruns          0.7147      0.176      4.059      0.000         0.370     1.060
p_wins           5.8466      0.931      6.278      0.000         4.021     7.672
p_saves          0.4207      0.146      2.877      0.004         0.134     0.707
p_eruns         -3.1113      0.706     -4.408      0.000        -4.495    -1.728
p_stout          1.3788      0.362      3.808      0.000         0.669     2.088
f_puts          -0.8823      0.212     -4.153      0.000        -1.299    -0.466
f_dplay          0.7878      0.176      4.480      0.000         0.443     1.132
POS_C            0.9812      0.183      5.365      0.000         0.623     1.340
POS_P           -1.4553      0.700     -2.079      0.038        -2.827    -0.083
teamID_Other    -0.5154      0.193     -2.668      0.008        -0.894    -0.137
================================================================================

The attached plots of career batting hits and career batting runs for training
and future prediction datasets show good separation of induction classes.

"""
#IMPORT PACKAGES---------------------------------------------------------------

import sqlite3 as sq
import pandas as pd
import numpy as np
from sklearn import preprocessing as pp
from sklearn.cross_validation import cross_val_score as cv
from sklearn.grid_search import GridSearchCV as gscv
from sklearn.feature_selection import RFECV as rfe
from sklearn import linear_model as lm
from statsmodels.formula.api import logit
import matplotlib.pyplot as plt
from patsy import dmatrix, dmatrices
import matplotlib.pylab as plt
from pylab import rcParams



#LOAD USER-DEFINED FUNCTIONS FOR DATA MANIPULATION-----------------------------

#convert low-freq categorical feature values to 'Other'
def cleanup_data(df, cutoffPercent = .01):
    for col in df:
        sizes = df[col].value_counts(normalize = True)
        values_to_delete = sizes[sizes<cutoffPercent].index
        df[col].ix[df[col].isin(values_to_delete)] = "Other"
    return df

#binazrize catergoical feature values into individual variables
def get_binary_values(data_frame):
    """encodes categorical features in Pandas."""
    all_columns = pd.DataFrame(index = data_frame.index)
    for col in data_frame.columns:
        data = pd.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pd.concat([all_columns, data], axis=1)
    return all_columns

#find and remove variables with zero variance
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
    return {'toKeep':toKeep, 'toDelete':toDelete}
    
#find and remove variables with perfect correlation
def find_perfect_corr(df):
    """finds columns that are eother positively or negatively perfectly 
        correlated (with correlations of +1 or -1), and creates a dict that 
        includes which columns to drop so that each remaining column is independent
    """
    corrMatrix = df.corr()
    corrMatrix.loc[:,:] = np.tril(corrMatrix.values, k = -1)
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



#GET DATA FROM SQL DB INTO PANDAS DATA FRAME-----------------------------------

#reconnect to SQLite DB
conn = sq.connect('C:\Users\mmcgoldr\Dropbox\GA\DataScience\SQLite\lahman2013.sqlite')

#get position, dominant team and performance stats
query2 = """select h.*, 
  b.b_atbat, b.b_runs, b.b_hits, b.b_hruns, b.b_stbas, b.b_strik,
  p.p_wins, p.p_loss, p.p_shout, p.p_saves, p.p_eruns, p.p_stout, 
  f.f_puts, f.f_assis, f.f_dplay, f.f_pass, o.pos, t.teamid
from 
  (select playerid, max(case when inducted = 'Y' then 1 else 0 end) as inducted, max(yearid) as year
   from halloffame where category = 'Player' group by playerid) h
left outer join 
  (select playerid, sum(ab) as b_atbat, sum(r) as b_runs, sum(h) as b_hits, 
    sum(hr) as b_hruns, sum(sb) as b_stbas, sum(so) as b_strik
  from batting group by playerid) b on h.playerid = b.playerid
left outer join
  (select playerid, sum(w) as p_wins, sum(l) as p_loss, sum(sho) as p_shout,
    sum(sv) as p_saves, sum(er) as p_eruns, sum(so) as p_stout
  from pitching group by playerid) p on h.playerid = p.playerid
left outer join
  (select playerid, sum(po) as f_puts, sum(a) as f_assis, sum(dp) as f_dplay, sum(pb) as f_pass 
  from fielding group by playerid) f on h.playerid = f.playerid
left outer join
  (select * from dominant_position_per_player) o on h.playerid = o.playerid
left outer join
  (select * from dominant_team_per_player) t on h.playerid = t.playerid
;"""

df = pd.read_sql(query2, conn)

#close connection
conn.close()

#check data
df.head()
df.tail()
df.shape

#split data before and on/after year 2000 (training vs future predictions)
pre2000 = df[df.year < 2000.00]
post2000 = df[df.year >= 2000.00]



#TRANSFORM DATA FOR TRAINING (< 2000)------------------------------------------

#drop records where all explanatory variables are missing (13 dropped)
pre2000.dropna(thresh=4,inplace=True)
pre2000.shape #882 players

#split dataset into id, explanatory and response features
exp = [col for col in pre2000.columns if col not in ['playerid','inducted','year']]
pre2000_exp = pre2000[exp]
pre2000_res = pre2000.inducted

#split explanatory features into categorical vs numeric
catfeat = pre2000_exp.ix[:, pre2000_exp.dtypes == 'object']
numfeat= pre2000_exp.ix[:, pre2000_exp.dtypes != 'object']

#fill cat NaNs with 'None'
catfeat = catfeat.fillna('None')

#bin and binarize cat vars
cleanup_data(catfeat, cutoffPercent = .01)
catfeat_bin = get_binary_values(catfeat)

pos_list = set(catfeat.POS)
team_list = set(catfeat.teamID)

# fill NANs in nuemeric features with zero
numfeat.fillna(0, inplace=True)

# Merge categorical and numeric dataframes
pre2000_exp = pd.concat([numfeat, catfeat_bin],axis = 1)

# find features with no variance (NONE / KEEP ALL)
find_zero_var(pre2000_exp)

#find features with perfect correlation (NONE / KEEP ALL)
find_perfect_corr(pre2000_exp)

#scale features
scaler = pp.StandardScaler()
scaler.fit(pre2000_exp)
pre2000_exp_scaled = pd.DataFrame(scaler.transform(pre2000_exp), 
                                  index = pre2000_exp.index, 
                                  columns = pre2000_exp.columns)

#retain unscaled features
pre2000_exp = pd.concat([numfeat, catfeat_bin],axis = 1)
find_zero_var(pre2000_exp)
find_perfect_corr(pre2000_exp)



#TRANSFORM DATA FOR FUTURE PREDICTIONS (>= 2000)-------------------------------

#drop records where all explanatory variables are missing (14 dropped)
post2000.dropna(thresh=4,inplace=True)
post2000.shape

#split dataset into id, explanatory and response features
post2000_exp = post2000[exp]
post2000_res = post2000.inducted

#split explanatory features into categorical vs numeric
catfeat_post = post2000_exp.ix[:, post2000_exp.dtypes == 'object']
numfeat_post= post2000_exp.ix[:, post2000_exp.dtypes != 'object']

#fill categorical missings with 'None'
catfeat_post.fillna('None', inplace=True)

#replace low-frequency categorical values with 'Other' using
#same breaks as training data
catfeat_post.POS[~catfeat_post.POS.isin(pos_list)] = 'Other'
catfeat_post.teamID[~catfeat_post.teamID.isin(team_list)] = 'Other'

#binarize categorical feature values into individual variables
catfeat_post_bin = get_binary_values(catfeat_post)

#identify binarized variables in training set that are not in future
#prediction set and add these columns (all set to 0) in future set
missing_col = catfeat_bin.columns[~catfeat_bin.columns.isin(catfeat_post_bin.columns)]
for c in missing_col:
    catfeat_post_bin[c] = 0
catfeat_post_bin = catfeat_post_bin[catfeat_bin.columns]

#impute numerical missings with 0
numfeat_post.fillna(0, inplace=True)

#merge categorical and numeric features
post2000_exp = pd.concat([numfeat_post, catfeat_post_bin],axis = 1)

#make sure final columns and their order are identical to training set
post2000_exp = post2000_exp[pre2000_exp.columns]

#scale data
scaler = pp.StandardScaler()
scaler.fit(post2000_exp)
post2000_exp_scaled = pd.DataFrame(scaler.transform(post2000_exp), 
                                   index = post2000_exp.index, 
                                   columns = post2000_exp.columns)
                                   
#retain unscaled features
post2000_exp = pd.concat([numfeat_post, catfeat_post_bin],axis = 1)
post2000_exp = post2000_exp[pre2000_exp.columns]



#IDENTIFY POTENTIAL FEATURES WITH RECURSIVE FEATURE SEARCH AND 10-FOLD CV------

#run recursive feature search with 10-fold cv to identify potential features
lr = lm.LogisticRegression()
lr_rfe_cv = rfe(estimator=lr, step=1, cv=10, scoring='roc_auc', verbose = 1)
lr_rfe_cv.fit(pre2000_exp_scaled, pre2000_res)

#identify features
features = pre2000_exp_scaled.columns[lr_rfe_cv.get_support()]
print features

#run 10-fold CV to get scores with selected features (ROC_AUC = 0.9451)
lr_cv = cv(lr, pre2000_exp_scaled[features], pre2000_res, cv=10, scoring='roc_auc')
lr_cv.mean()

#create dataset with response and selected features
lrset = pd.concat([pre2000_exp_scaled[features], pre2000_res], axis=1)



#BUILD FULL LOGISTIC REGRESSION MODEL------------------------------------------

#get model summary with ALL variables (except teamID_CAL because it leads to singular matrix)
model_all = logit('inducted ~ b_atbat + b_runs + b_hits + b_hruns + b_strik + p_wins + p_loss + p_shout + p_saves + p_eruns + p_stout + f_puts + f_dplay + POS_C + POS_P  + teamID_NYN  + teamID_Other', 
               data = lrset).fit(maxiter=5000)
print model_all.summary()

#get predicted probabilities for future cases >= 2000
pred_prob = pd.Series(model_all.predict(post2000_exp_scaled[features]))

pred_prob.describe()
pred_prob[pred_prob >= .5].count()
pred_prob[pred_prob >= .5].count()/248.0

#get predicted (0,1) for future cases >= 2000
pred = pred_prob
pred[pred < .5] = 0
pred[pred >= .5] = 1

#get actual induction rate for future cases >= 2000
post2000_res.describe()

#cross actual and predicted (accuracy = 0.85, TPR = .6 )
post2000_res.index = pred.index
pd.crosstab(post2000_res, pred, rownames=['Actual'], 
            colnames=['Predicted'], margins=True)

#BUILD REDUCED LOGISTIC REGRESSION MODEL WITH ONLY SIGNFICANT PREDICTORS-------

#get model summary with significant variables ONLY 
model_sig = logit('inducted ~ b_atbat + b_runs + b_hits + b_hruns + p_wins + p_saves + p_eruns + p_stout + f_puts + f_dplay + POS_C + POS_P  + teamID_Other', 
               data = lrset).fit(maxiter=5000)
print model_sig.summary()

#get predicted probabilities for future cases >= 2000
pred_prob_sig = pd.Series(model_sig.predict(post2000_exp_scaled[features]))

pred_prob_sig.describe()
pred_prob_sig[pred_prob_sig >= .5].count()
pred_prob_sig[pred_prob_sig >= .5].count()/248.0

#get predicted (0,1) for future cases >= 2000
pred_sig = pred_prob_sig
pred_sig[pred_sig < .5] = 0
pred_sig[pred_sig >= .5] = 1

#cross actual and predicted (accuracy = 0.84, TPR = .57 )
post2000_res.index = pred.index
pd.crosstab(post2000_res, pred_sig, rownames=['Actual'], 
            colnames=['Predicted'], margins=True)



#PLOT CLASS SEPARATION WITH REDUCED MODEL PARAMETERS---------------------------

#get parameters for intercept, batting hits and runs using full model
logit_parms = model_sig.params
intercept = logit_parms[0] / logit_parms[2]
slope = logit_parms[3] / logit_parms[2]

#plot batting hits vs runs with class separation line for training data
bhits_yes = pre2000_exp_scaled.b_hits[pre2000_res == 1]
bhits_no = pre2000_exp_scaled.b_hits[pre2000_res == 0]
bruns_yes = pre2000_exp_scaled.b_runs[pre2000_res == 1]
bruns_no = pre2000_exp_scaled.b_runs[pre2000_res == 0]
plt.figure(figsize = (12, 8))
plt.plot(bhits_yes, bruns_yes, '.', mec = 'purple', mfc = 'None', 
         label = 'Inducted')
plt.plot(bhits_no, bruns_no, '.', mec = 'orange', mfc = 'None', 
         label = 'Not Inducted')
plt.plot(np.arange(-2, 4, 1), intercept + slope * np.arange(-2, 4, 1), '-k', label = 'Separating line')
plt.ylim(-2, 4)
plt.xlabel('Career Batting Hits')
plt.ylabel('Career Batting Runs')
plt.legend(loc = 'best')

#plot batting hits vs runs with class separation line for test data
post2000_res.index = post2000_exp_scaled.index
bhits_yes = post2000_exp_scaled.b_hits[post2000_res == 1]
bhits_no = post2000_exp_scaled.b_hits[post2000_res == 0]
bruns_yes = post2000_exp_scaled.b_runs[post2000_res == 1]
bruns_no = post2000_exp_scaled.b_runs[post2000_res == 0]
plt.figure(figsize = (12, 8))
plt.plot(bhits_yes, bruns_yes, '.', mec = 'purple', mfc = 'None', 
         label = 'Inducted')
plt.plot(bhits_no, bruns_no, '.', mec = 'orange', mfc = 'None', 
         label = 'Not Inducted')
plt.plot(np.arange(-2, 4, 1), intercept + slope * np.arange(-2, 4, 1), '-k', label = 'Separating line')
plt.ylim(-2, 4)
plt.xlabel('Career Batting Hits')
plt.ylabel('Career Batting Runs')
plt.legend(loc = 'best')