# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 22:38:08 2015

@author: jchen
"""

import pandas as pd
import sqlite3
import numpy as np

pd.set_option('display.max_columns', None)

conn = sqlite3.connect('/Users/jchen/Documents/SQLite/lahman2013.sqlite')  

# pull in all metrics from hw #5 for players inducted into Hall of Fame
# note that hall_of_fame_inductees table already exists
sql = '''
select m.nameGiven as player_name,
         h.inducted,
         sum(b.AB) as at_bats,
         sum(b.R) as runs,
         sum(b.H) as hits,
	   sum(b.RBI) as rbi,
	   sum(p.GS) as p_games_started,
	   sum(p.CG) as p_complete_games,
	   sum(p.SHO) as shutouts,
         sum(p.W) as p_wins,
         sum(p.IPOuts) as outs_pitched,
	   sum(f.PO) as putouts,
	   sum(f.A) as assists,
	   sum(f.E) as errors,
         (b.H+b.BB+b.HBP)*1.0/(b.AB+b.BB+b.SF+b.HBP) as OBP,
	   (b.H+b."2B"+(b."3B"*2)+(b.HR*3))*1.0/b.AB as SLG,
         (p.W + p.BB)/(p.IPOuts/3) as WHIP
from hall_of_fame_inductees h
left join Batting b on h.playerID=b.playerID
left join Pitching p on h.playerID=p.playerID
left join Fielding f on h.playerID=f.playerID
left join Master m on h.playerID=m.playerID  
group by player_name, inducted
order by player_name;
'''
# read into data frame
df = pd.read_sql(sql, conn)
# close out connection
conn.close()


# separate explanatory features
explanatory_features = [col for col in df.columns if col not in ['player_name', 'inducted']]
explanatory_df = df[explanatory_features]

# explanatory column names
explanatory_colnames = explanatory_df.columns

# drop rows with no data from explanatory
explanatory_df.dropna(how='all', inplace = True) 

# drop rows with no data from response
response_series = df.inducted
response_series.dropna(how='all', inplace = True) 

# check to see if the new indices are different
response_series.index[~response_series.index.isin(explanatory_df.index)]
# keep only the indicies of the response series and explanatory df that are in both
common_indices = response_series.index[response_series.index.isin(explanatory_df.index)] 
response_series_new = response_series[response_series.index.isin(common_indices)]
explanatory_df = explanatory_df[explanatory_df.index.isin(common_indices)]

# impute NaNs with the mean value for each column
from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_object.fit(explanatory_df)
explanatory_df = imputer_object.transform(explanatory_df)

########################
# Naive Bayes classifier
########################

from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score

# creating the naive bayes classifier object 
naive_bayes_classifier = MultinomialNB()

# Run cross-validated score on accuracy
accuracy_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series_new, cv=10, scoring='accuracy', n_jobs = -1)
mean_accuracy_score = accuracy_scores.mean()
print mean_accuracy_score
# 67% - not stellar

# let's calculate Cohen's Kappa
largest_class_percent_of_total = response_series_new.value_counts(normalize = True)[0]
# largest_class_percent_total is around 77%.
kappa = (mean_accuracy_score - largest_class_percent_of_total) / (1-largest_class_percent_of_total)
print kappa
# kappa is negative - information loss

# calculate F1 score
f1_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series_new, cv=10, scoring='f1', n_jobs = -1)
print f1_scores.mean()
# F1 score 33% - not too good

# calculate ROC AUC score. 
roc_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series_new, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores.mean()
# 0.56 = fail


##################
# Decision Tree
##################

from sklearn import tree

# Decision tree classifier instance
decision_tree = tree.DecisionTreeClassifier(random_state=1)

# Run cross-validated score on accuracy
tree_accuracy_scores = cross_val_score(decision_tree, explanatory_df, response_series_new, cv=10, scoring='accuracy', n_jobs = -1)
tree_mean_accuracy_score = tree_accuracy_scores.mean()
print tree_mean_accuracy_score
# 74% - a little better than naive bayes

# Cohen's Kappa
tree_kappa = (tree_mean_accuracy_score - largest_class_percent_of_total) / (1-largest_class_percent_of_total)
print tree_kappa
# -0.125 still negative

# calculate F1 score
tree_f1_scores = cross_val_score(decision_tree, explanatory_df, response_series_new, cv=10, scoring='f1', n_jobs = -1)
print tree_f1_scores.mean()
# F1 score 42% - not too good but slightly better

# calculate ROC AUC score. 
tree_roc_scores = cross_val_score(decision_tree, explanatory_df, response_series_new, cv=10, scoring='roc_auc', n_jobs = -1)
print tree_roc_scores.mean()
# 0.63 = poor, but better than naive bayes


# Seems like KNN from HW #5 was the better model compared to these two
# But - we did split up batting and pitching stats and built separate KNN models
# Perhaps re-run these models on split up data

###########################
# ROC curves for decision tree
###########################

from sklearn.cross_validation import StratifiedKFold

from scipy import interp
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

# turn response series into np array for consistency
response_series = response_series_new.values

# Set up cross-validation
cv = StratifiedKFold(response_series, n_folds=10)

# Run classifier with cross-validation and plot ROC curves
mean_tpr = 0.0 
mean_fpr = np.linspace(0, 1, 100) # array of 100 evenly spaced values from 0 to 1
all_tpr = []
 
plt.figure()    
for i, (train, test) in enumerate(cv):
    y_probabilities = decision_tree.fit(explanatory_df[train], response_series[train]).predict_proba(explanatory_df[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(response_series[test], y_probabilities[:, 1])
    # update mean_tpr with array of interpolated values for each increment in mean fpr array
    mean_tpr += interp(mean_fpr, fpr, tpr)  
    mean_tpr[0] = 0.0 
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc)) 
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck') # reference line for randomness
# Hm - many of these ROC curves do not look very good

mean_tpr /= len(cv) # calculate mean by dividing by number of folds
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves for Decision Tree Classifier')
plt.legend(loc="lower right",prop={'size':7})
plt.show()
