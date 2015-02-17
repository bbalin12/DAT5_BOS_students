# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 14:34:15 2015

@author: megan
"""

import pandas
import sqlite3
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

conn = sqlite3.connect('/Users/megan/Documents/SQLite/lahman2013.sqlite')
cur = conn.cursor()    
table_creation_query = """
CREATE TABLE hall_of_fame_inductees as  

select playerID, case when average_inducted = 0 then 0 else 1 end as inducted from (

select playerID, avg(case when inducted = 'Y' then 1 else 0 end ) as average_inducted from  HallOfFame hf
where yearid < 2000
group by playerID

) bb;"""
cur.execute(table_creation_query)
cur.close()
conn.close()

conn = sqlite3.connect('/Users/megan/Documents/SQLite/lahman2013.sqlite')
sql = '''
SELECT hofi.playerID, hofi.inducted, 
batting.atBats, batting.hits,
pitching.wins, pitching.losses,
fielding.putOuts, fielding.assists, fielding.errors
FROM hall_of_fame_inductees hofi
LEFT JOIN 
(
SELECT b.playerID, max(b.yearID) maxYear, sum(b.AB) as atBats, sum(b.H) as hits
FROM Batting b
GROUP BY b.playerID
)
batting on batting.playerID = hofi.playerID
LEFT JOIN 
(
SELECT p.playerID, sum(p.W) as wins, sum(p.L) as losses
FROM Pitching p 
GROUP BY p.playerID
)
pitching on hofi.playerID = pitching.playerID
LEFT JOIN 
(
SELECT f.playerID, sum(f.PO) as putOuts, sum(f.A) as assists, sum(f.E) as errors
FROM Fielding f 
GROUP BY f.playerID
)
fielding on hofi.playerID = fielding.playerID
WHERE batting.maxYear < 2000;
'''
df = pandas.read_sql(sql, conn)
conn.close()

pandas.set_option('display.max_columns', None)

df.drop('playerID',  1, inplace = True)

# Add composite feature columns
df['batting_average'] = df.hits / df.atBats
df['winning_percentage'] = df.wins / (df.wins + df.losses)
df['fielding_percentage'] = (df.putOuts + df.assists) / (df.putOuts + df.assists + df.errors)

df = df.replace([np.inf, -np.inf], np.nan)

# Extract explanatory features
explanatory_features = [col for col in df.columns if col not in ['inducted']]
explanatory_df = df[explanatory_features]

explanatory_df.dropna(how='all', inplace=True)
explanatory_colnames = explanatory_df.columns

# Extract repsonse features
response_series = df.inducted
response_series.dropna(how='all', inplace = True) 
# See which explanatory feature rows got removed.  Looks like none.
response_series.index[~response_series.index.isin(explanatory_df.index)]

# Fill NaNs with the mean value for that column.  
from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_object.fit(explanatory_df)
explanatory_df = imputer_object.transform(explanatory_df)

largest_class_percent_of_total = response_series.value_counts(normalize = True)[0]

##################
# KNN 
##################
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(p = 2)

# Find best k value: k = 23
k_range = range(1, 30, 2)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(explanatory_df, response_series)

grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(k_range, grid_mean_scores)
best_oob_score = grid.best_score_
grid.best_params_
knn = grid.best_estimator_

# Average accuracy of model = 0.816
accuracy_scores_knn = cross_val_score(knn, explanatory_df, response_series, cv=10, scoring='accuracy', n_jobs = -1)
print accuracy_scores_knn.mean()

# Cohen's Kappa = 0.182
mean_accuracy_score_knn = accuracy_scores_knn.mean()
kappa_knn = (mean_accuracy_score_knn - largest_class_percent_of_total) / (1-largest_class_percent_of_total)
print kappa_knn

# F1 Score = 0.424
f1_scores_knn = cross_val_score(knn, explanatory_df, response_series, cv=10, scoring='f1', n_jobs = -1)
print f1_scores_knn.mean()

# ROC area under the curve score = 0.797
roc_scores_knn = cross_val_score(knn, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_knn.mean()

##################
# NAIVE BAYES
##################
from sklearn.naive_bayes import MultinomialNB

naive_bayes_classifier = MultinomialNB()

# Average accuracy of model = 0.594
accuracy_scores_nb = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='accuracy', n_jobs = -1)
print accuracy_scores_nb.mean()

# Cohen's Kappa = -0.802
mean_accuracy_score_nb = accuracy_scores_nb.mean()
kappa_nb = (mean_accuracy_score_nb - largest_class_percent_of_total) / (1-largest_class_percent_of_total)
print kappa_nb

# F1 Score = 0.348
f1_scores_nb = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='f1', n_jobs = -1)
print f1_scores_nb.mean()

# ROC area under the curve score = 0.554
roc_scores_nb = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_nb.mean()

####################
# DECISION TREE
####################
from sklearn import tree

decision_tree = tree.DecisionTreeClassifier(random_state=1)

# Find the best depth for the deicison tree: max_depth = 3
depth_range = range(1, 20)
param_grid = dict(max_depth=depth_range)
grid = GridSearchCV(decision_tree, param_grid, cv=10, scoring='roc_auc')
grid.fit(explanatory_df, response_series)

# Plot the results of the grid search
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.plot(grid.best_params_['max_depth'], grid.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')
plt.grid(True)

decision_tree = grid.best_estimator_

# Average accuracy of model = 0.840
accuracy_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='accuracy', n_jobs = -1)
print accuracy_scores_cart.mean()

# Cohen's Kappa = 0.290
mean_accuracy_score_cart = accuracy_scores_cart.mean()
kappa_cart = (mean_accuracy_score_cart - largest_class_percent_of_total) / (1-largest_class_percent_of_total)
print kappa_cart

# F1 Score = 0.574
f1_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='f1', n_jobs = -1)
print f1_scores_cart.mean()

# ROC area under the curve score = 0.774
roc_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_cart.mean()

# View importance of explanatory features
importances_df = pandas.DataFrame(explanatory_colnames)
importances_df['importances'] = decision_tree.feature_importances_

######################
# RESULTS
######################
# KNN
print "KNN Results"
print "Mean accuracy: " + str(accuracy_scores_knn.mean())
print "Cohen's Kappa: " + str(kappa_knn)
print "F1 Score: " + str(f1_scores_knn.mean())
print "ROC AUC Score: " + str(roc_scores_knn.mean())
print "\n"

# Naive Bayes
print "Naive Bayes Results"
print "Mean accuracy: " + str(accuracy_scores_nb.mean())
print "Cohen's Kappa: " + str(kappa_nb)
print "F1 Score: " + str(f1_scores_nb.mean())
print "ROC AUC Score: " + str(roc_scores_nb.mean())
print "\n"

# Decision Tree
print "Decision Tree Results"
print "Mean accuracy: " + str(accuracy_scores_cart.mean())
print "Cohen's Kappa: " + str(kappa_cart)
print "F1 Score: " + str(f1_scores_cart.mean())
print "ROC AUC Score: " + str(roc_scores_cart.mean())
print "\n"

# Based on these results, the Decision Tree is the best classifier
# For your best performing model, print a confusion matrix and ROC curve in your iPython interpreter for all k cross validation slices. 
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold

x = explanatory_df
y = response_series
classifier = decision_tree

cv = StratifiedKFold(y, n_folds=6)

# Print Confusion Matrices
for i, (train, test) in enumerate(cv):
    # get predictions on the test slice of the classifier. 
    y_predicted = classifier.fit(x[train], y[train]).predict(x[test])
    # create confusion matrix for the data
    cm = pandas.crosstab(y[test], y_predicted, rownames=['True Label'],
                         colnames=['Predicted Label'], margins=True)
    print cm
    print "\n"

# Plot ROC Curves
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(x[train], y[train]).predict_proba(x[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('ROC_kfold.png')