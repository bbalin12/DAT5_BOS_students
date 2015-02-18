# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 09:49:23 2015
@author: mmcgoldr
"""

"""
GA DATA SCIENCE CLASS 7 HOMEWORK: NAIVE BAYES AND DECISION TREES

  1.Build a Naive Bayes and decision tree model for your set of features 
    you have extracted from the Baseball Dataset to predict whether a player was 
    inducted into the Baseball Hall of Fame before the year 2000.
  2.Compare the n-fold cross-validated accuracy, F1 score, and AUC 
    for your two new models, and compare it to the KNN model you 
    built for class 5. Decide which of these models is the most accurate.
  3.For your best performing model, print a confusion matrix and ROC curve 
    in your iPython interpreter for all k cross validation slices.

SUMMARY

Using SQLite3, I pulled in career totals for selected batting, pitching and 
fielding variables among 1,157 players who were ever eligible for the Hall of 
Fame. These variables include at bats, hits, runs, home runs, stolen bases and
strikeouts; pitching # years, wins, losses, shoutouts, saves, earned runs and stolen 
base outs; fielding # years, puts, assists, double plays and passes; and finally,
binary variables for position played, including catcher, pictcher, designated 
hitter, out fielder and center fielder.  The response variable is whether or 
not a player was ever inducted into the Hall of Fame (inducted = Y vs N)

I dropped 27 players whose stats were missing entirely and filled the remaining NaNs 
with zeros.  This netted 1,130 players on which the models are based.

For each of the models -- KNN (KN), Naive Bayes (NB), Decision Tree (DT) --
I used 10-fold cross validation and tested three scoring methods: 
accuracy, F1 and ROC AUC.  Also for KN and DT, I used Grid Search to identify 
the optimal k-neighbor and max depth parameters, respectively. The following 
are mean scores for each model.

     Accurary  F1       AUC
KN=  86.37%    57.06%   85.74%
NB=  61.63%    32.99%   57.16%
DT=  88.05%    63.91%   81.47%

Naive Bayes performs the worst based on all three scoring metrics.  Decision
Tree has the highest accuracy and F1 while KNN has the highest AUC.  Since the
Decision Tree method performs best on 2 of 3 metrics, I select this as my best
model.

Below are confusion matrices for each cross validation fold of the Decision
Tree model.  The True Positive Rate is > 50% for 8 of 10 folds.

----FOLD #0----
Predicted   0   1  All
Actual                
0          83   9   92
1           7  15   22
All        90  24  114

----FOLD #1----
Predicted   0   1  All
Actual                
0          87   5   92
1           7  15   22
All        94  20  114

----FOLD #2----
Predicted   0   1  All
Actual                
0          80  12   92
1          15   7   22
All        95  19  114

----FOLD #3----
Predicted   0   1  All
Actual                
0          84   8   92
1           8  14   22
All        92  22  114

----FOLD #4----
Predicted    0   1  All
Actual                 
0           89   3   92
1           11  11   22
All        100  14  114

----FOLD #5----
Predicted   0   1  All
Actual                
0          82   9   91
1          11  10   21
All        93  19  112

----FOLD #6----
Predicted   0   1  All
Actual                
0          87   4   91
1           9  12   21
All        96  16  112

----FOLD #7----
Predicted   0   1  All
Actual                
0          85   6   91
1          10  11   21
All        95  17  112

----FOLD #8----
Predicted   0   1  All
Actual                
0          87   4   91
1           7  14   21
All        94  18  112

----FOLD #9----
Predicted   0   1  All
Actual                
0          82   9   91
1           8  13   21
All        90  22  112

For ROC Curve plots by fold, see attache file "mmcgoldr_hwc7_plot.png"

For code, see below...

"""

#import packages and functions
import pandas as pd
import sqlite3 as sq
import numpy as np
from sklearn.cross_validation import cross_val_score as cvs
from sklearn.cross_validation import StratifiedKFold as skf
from sklearn.grid_search import GridSearchCV as gscv
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn import tree as tr
from sklearn import metrics as mt
from scipy import interp
import matplotlib.pyplot as plt


#get data from sqlite db into pandas dataframe and close connection
conn = sq.connect('C:\Users\mmcgoldr\Dropbox\GA\DataScience\SQLite\lahman2013.sqlite')

query = """
select h.*, 
  b.b_atbat, b.b_runs, b.b_hits, b.b_hruns, b.b_stbas, b.b_strik,
  p.p_years, p.p_wins, p.p_loss, p.p_shout, p.p_saves, p.p_eruns, p.p_stout, 
  f.f_years, f.f_puts, f.f_assis, f.f_dplay, f.f_pass, 
  f.catcher, f.pitcher, f.dhitter, f.ofielder, f.cfielder
from 
  (select playerid, max(case when inducted = 'Y' then 1 else 0 end) as inducted, max(yearid) as year
   from halloffame 
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
  group by playerid) b
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
     count(distinct yearid) as f_years,
     sum(po) as f_puts,
     sum(a) as f_assis,
     sum(dp) as f_dplay,
     sum(pb) as f_pass,
     max(case when pos = 'C' then 1 else 0 end) as catcher,
     max(case when pos = 'P' then 1 else 0 end) as pitcher,
     max(case when pos = 'DH' then 1 else 0 end) as dhitter,
     max(case when pos = 'OF' then 1 else 0 end) as ofielder,
     max(case when pos = 'CF' then 1 else 0 end) as cfielder
  from fielding
  group by playerid) f
  on h.playerid = f.playerid
;"""

df = pd.read_sql(query, conn)

conn.close()

# setting in pandas to print entire dataset
pd.set_option('display.max_columns', None)

#examine data
df.shape #players=1157
df.columns

#drop 27 players where all B/P/F stats are missing
df.dropna(thresh=8, inplace=True) 
df.shape #players=1130

#set missings to 0
df.fillna(value=0, inplace=True)
df.isnull().sum()

#set explanatory and response variables
explanatory = [col for col in df.columns if col not in ['playerid', 'inducted','year']]
df_exp = df[explanatory]
df_res = df.inducted

#KNN
knn=knc(p = 2) #specify Euclidean distance

param_grid = dict(n_neighbors=range(1,30, 2)) #set up grid for results
kn_accuracy=gscv(knn, param_grid, cv=10, scoring='accuracy').fit(df_exp, df_res)

param_grid = dict(n_neighbors=range(1,30, 2)) #set up grid for results
kn_f1=gscv(knn, param_grid, cv=10, scoring='f1').fit(df_exp, df_res)

param_grid = dict(n_neighbors=range(1,30, 2)) #set up grid for results
kn_auc=gscv(knn, param_grid, cv=10, scoring='roc_auc').fit(df_exp, df_res)

#Naive Bayes
nb = mnb()
nb_accuracy = cvs(nb, df_exp, df_res, cv=10, scoring='accuracy')
nb_f1 = cvs(nbclass, df_exp, df_res, cv=10, scoring='f1')
nb_auc = cvs(nbclass, df_exp, df_res, cv=10, scoring='roc_auc')

#Decision Tree
dtree = tr.DecisionTreeClassifier(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None,min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None)

param_grid = dict(max_depth=range(1, 23))
dt_accuracy = gscv(dtree, param_grid, cv=10, scoring='accuracy').fit(df_exp, df_res)

param_grid = dict(max_depth=range(1, 23))
dt_f1 = gscv(dtree, param_grid, cv=10, scoring='f1').fit(df_exp, df_res)

param_grid = dict(max_depth=range(1, 23))
dt_auc = gscv(dtree, param_grid, cv=10, scoring='roc_auc').fit(df_exp, df_res)

#Compare models: accuracy
print round(kn_accuracy.best_score_*100,2) 
print round(nb_accuracy.mean()*100,2) 
print round(dt_accuracy.best_score_*100,2) 

#Compare models: F1
print round(kn_f1.best_score_*100,2) 
print round(nb_f1.mean()*100,2) 
print round(dt_f1.best_score_*100,2) 

#Compare models: ROC AUC
print round(kn_auc.best_score_*100,2) 
print round(nb_auc.mean()*100,2) 
print round(dt_auc.best_score_*100,2) 


#Confusion Matrix, ROC Curve by K-fold Slice
folds = skf(df_res, 10, indices=False)

#produce confusion matrices for each fold
for i, (train, test) in enumerate(folds):    
    preds = dtree.fit(df_exp.ix[train,], df_res[train]).predict(df_exp.ix[test,])
    print '----FOLD #%d----' % i 
    print pd.crosstab(df_res[test], preds, rownames=['Actual'], colnames=['Predicted'], margins=True)

#produce ROC curves for each fold
dtree = tr.DecisionTreeClassifier()
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
for i, (train, test) in enumerate(folds):    
    preds = dtree.fit(df_exp.ix[train,], df_res[train]).predict(df_exp.ix[test,])
    fpr, tpr, thresholds = mt.roc_curve(df_res[test], preds)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, mt.auc(fpr, tpr)))
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= len(folds)
mean_tpr[-1] = 1.0
mean_auc = mt.auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(bbox_to_anchor=(1.7, 1.05))
plt.show()

plt.legend(show='right')
