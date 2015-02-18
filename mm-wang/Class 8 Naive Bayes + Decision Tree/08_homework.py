# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 12:22:44 2015

@author: Margaret
"""

# importing division from the 'future' release of Python (i.e. Python 3)
from __future__ import division

# importing numpy and the KNN content in scikit-learn along with SQLite + pandas

import sqlite3
import pandas
import matplotlib.pyplot as plt

# connect to the baseball database. I'm passing the full path to the SQLite file
conn = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')
# creating an objected constraining a string that has the SQL query
sql = """ 
SELECT h.playerID, max(CASE WHEN h.inducted='Y' THEN 1 ELSE 0 END) as inducted, bat_runs, bat_hits, at_bats, bat_homeruns, bat_strikes, bat_stolen,
pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA, f_putouts, f_assists, f_errors FROM HallofFame h
INNER JOIN
(SELECT f.playerID, f.PO as f_putouts, f.A as f_assists, f.E as f_errors, bat_runs, bat_hits, at_bats, bat_homeruns,
bat_strikes, bat_stolen, pitch_wins, pitch_strikes, pitch_shuts, pitch_ERA FROM Fielding f
LEFT JOIN
(SELECT b.playerID, sum(b.R) as bat_runs, sum(b.H) as bat_hits, sum(b.AB) as at_bats, sum(b.HR) as bat_homeruns,
sum(b.SO) as bat_strikes, sum(b.SB) as bat_stolen,
sum(p.W) as pitch_wins, sum(p.SO) as pitch_strikes, sum(p.SHO) as pitch_shuts, avg(1/p.ERA) as pitch_ERA
FROM Batting b
LEFT JOIN Pitching p on p.playerID = b.playerID
GROUP BY b.playerID) batpitch on batpitch.playerID = f.playerID
GROUP BY batpitch.playerID) positions
ON positions.playerID = h.playerID
WHERE h.yearID < 2000
GROUP BY h.playerID
"""

# passing the connectiona nd the SQL string to pandas.read_sql
df = pandas.read_sql(sql, conn)

# closing the connection
conn.close()



## batting - could be home runs, hits/at bats (batting average), lack of strikes, stolen bases
df['bat_avg'] = df.bat_hits/df.at_bats

## fielding - fielding percentage as (PO + A)/(PO + A + E)
## taken from Wikipedia and http://www.csgnetwork.com/baseballdefensestatsformulae.html
df['f_perc'] = (df.f_putouts+df.f_assists)/(df.f_putouts+df.f_assists+df.f_errors)

df.drop_duplicates('playerID',inplace=True)

# print out entire dataframe when using .head()
pandas.set_option('display.max_columns',None)


# predicting inductions
response_series = df.inducted
# using all the variables and derived variables
explanatory_df = df[['bat_runs','bat_homeruns','bat_hits','at_bats','bat_avg','bat_strikes','bat_stolen',
'pitch_wins','pitch_strikes','pitch_shuts','pitch_ERA','f_putouts','f_assists','f_errors','f_perc']]

explanatory_df.dropna(how='all',inplace=True)

# extracting column names 
explanatory_colnames = explanatory_df.columns

response_series.index[~response_series.index.isin(explanatory_df.index)]



# imputing with NaNs
from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# fitting the object on our data, done so we can save the fit for our new data
imputer_object.fit(explanatory_df)
explanatory_df = imputer_object.transform(explanatory_df)


######################
### NAIVE BAYES METHOD
######################

# create a naive Bayes classifier and get it cross-validated accuracy score. 
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score

# creating the naive bayes classifier object 
naive_bayes_classifier = MultinomialNB()

# running a cross-validates score on accuracy.  Notice I set 
# n_jobs to -1, which means I'm going to use all my computer's 
# cores to find the result.
accuracy_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='accuracy')


# let's calculate Cohen's Kappa
mean_accuracy_score = accuracy_scores.mean()
largest_class_percent_of_total = response_series.value_counts(normalize = True)[0]
kappa = (mean_accuracy_score - largest_class_percent_of_total) / (1-largest_class_percent_of_total)

# calculating F1 score
f1_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='f1')

# calculating ROC area under the curve score
roc_scores = cross_val_score(naive_bayes_classifier,explanatory_df,response_series,cv=10,scoring='roc_auc')

## here's the interpretability of AUC
#.90-1 = excellent 
#.80-.90 = good 
#.70-.80 = fair 
#.60-.70 = poor
#.50-.60 = fail


## Stratified K-Fold ROC Curve
import numpy as np

from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold

cv = StratifiedKFold(response_series, n_folds=10)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas_ = naive_bayes_classifier.fit(explanatory_df[train], response_series[train]).predict_proba(explanatory_df[test])
    predicted_values = naive_bayes_classifier.predict(explanatory_df[test])    
    cm = pandas.crosstab(response_series[test],predicted_values, rownames = ['True Label'], colnames = ['Predicted Label'], margins = True)
    print "Naive Bayes Confusion Matrix: %d" % (i+1)
    print cm
    print '\n'
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(response_series[test], probas_[:, 1])
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
plt.title('Naive Bayes 10-Fold Cross Validation ROC')
plt.subplot()
plt.legend(bbox_to_anchor=(1.65,1.07))
plt.show()

# create confusion matrix
cm = pandas.crosstab(response_series[test],probas_[:,1], rownames = ['True Label'], colnames = ['Predicted Label'], margins = True)
print "Naive Bayes Confusion Matrix: "
print cm



##############################################
### CART 
### Classification and Regression Trees Method
##############################################

# create trees
from sklearn import tree
# create instance of classifier
decision_tree = tree.DecisionTreeClassifier(random_state = 1)

# realize that the above code is the exact same as the code below
# shows the object's default values
# values can be changed to tune the tree
# no restrictions of features or the depth of the tree

decision_tree = tree.DecisionTreeClassifier(criterion = 'gini', splitter = 'best',
                max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2,
                max_leaf_nodes = None, random_state = 1)

## Stratified K Fold ROC
cv = StratifiedKFold(response_series, n_folds=10)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas_ = decision_tree.fit(explanatory_df[train], response_series[train]).predict_proba(explanatory_df[test])
    # Confusion Matrix
    predicted_values = decision_tree.predict(explanatory_df[test])    
    cm = pandas.crosstab(response_series[test],predicted_values, rownames = ['True Label'], colnames = ['Predicted Label'], margins = True)
    print "Decision Tree Confusion Matrix: %d" % (i+1)
    print cm
    print '\n'
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(response_series[test], probas_[:, 1])
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


importances_df = pandas.DataFrame(explanatory_colnames)
importances_df['importances'] = decision_tree.feature_importances_


# calculating accuracy
accuracy_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring = 'accuracy')

# calculating Cohen's Kappa
mean_accuracy_score_cart = accuracy_scores_cart.mean()
# recall we already calculated the largest_class_percent_of_total above.
kappa_cart = (mean_accuracy_score_cart - largest_class_percent_of_total) / (1-largest_class_percent_of_total)

# calculating f1 score
f1_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring = 'f1')

# calculating the ROC area under the curve score. 
roc_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='roc_auc')


###############################
### K-NEAREST NEIGHBORS METHOD
###############################

from sklearn.neighbors import KNeighborsClassifier

KNN_classifier = KNeighborsClassifier(n_neighbors=9,p=2)
# putting in the entire dataset
accuracy_scores_knn = cross_val_score(KNN_classifier, explanatory_df,response_series,cv=10,scoring='accuracy')

# calculating Cohen's Kappa
mean_accuracy_score_knn = accuracy_scores_knn.mean()
# recall we already calculated the largest_class_percent_of_total above.
kappa_knn = (mean_accuracy_score_knn - largest_class_percent_of_total) / (1-largest_class_percent_of_total)

# calculating f1 score
f1_scores_knn = cross_val_score(KNN_classifier, explanatory_df, response_series, cv=10, scoring = 'f1')

# calculating the ROC area under the curve score. 
roc_scores_knn = cross_val_score(KNN_classifier, explanatory_df, response_series, cv=10, scoring='roc_auc')

## Stratified K Fold ROC
cv = StratifiedKFold(response_series, n_folds=10)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas_ = KNN_classifier.fit(explanatory_df[train], response_series[train]).predict_proba(explanatory_df[test])
    # Confusion Matrix
    predicted_values = KNN_classifier.predict(explanatory_df[test])    
    cm = pandas.crosstab(response_series[test],predicted_values, rownames = ['True Label'], colnames = ['Predicted Label'], margins = True)
    print "K Nearest Neighbor Confusion Matrix: %d" % (i+1)
    print cm
    print '\n'
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(response_series[test], probas_[:, 1])
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
plt.title('K Nearest Neighbors 10-Fold Cross Validation ROC')
plt.subplot()
plt.legend(bbox_to_anchor=(1.65,1.07))
plt.show()



## COMPARISONS

print "Cross Validated Score using Naive Bayes is: %f " % accuracy_scores.mean()
print 'Cross Validated Score using Decision Tree Accuracy: %f' % accuracy_scores_cart.mean()
print 'Cross Validated Score using K Nearest Neighbors Accuracy: %f' % accuracy_scores_knn.mean()

print "Naive Bayes Cohen's Kappa is: %f" % kappa
print "Decision Tree Cohen's Kappa is: %f" % kappa_cart
print "K Nearest Neighbors Cohen's Kappa is: %f" % kappa_knn


print "Naive Bayes F1 Score is: %f" % f1_scores.mean()
print "Decision Tree F1 Score is: %f" % f1_scores_cart.mean()
print "K Nearest Neighbors F1 Score is: %f" % f1_scores_knn.mean()

print "Naive Bayes ROC Score is: %f" % roc_scores.mean()
print "Decision Tree ROC Score is: %f" % roc_scores_cart.mean()
print "K Nearest Neighbors ROC Score is: %f" % roc_scores_knn.mean()