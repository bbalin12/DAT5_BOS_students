# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 08:57:24 2015

@author: jkraunz
"""

import pandas
import numpy
import sqlite3
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score

# import sql data
conn = sqlite3.connect('/Users/jkraunz/Documents/SQLite/lahman2013.sqlite.crdownload')

sql = '''
Select d.*, sum(H) as total_post_hits, sum(HR) as total_post_HRs, sum(RBI) as total_post_RBIs
FROM
(Select c.*, sum(W) as total_post_wins, sum(SV) as total_post_saves, avg(ERA) as avg_post_ERA
FROM
(Select a.*, sum(E) as total_errors
FROM
(SELECT m.*,
sum(SO) as total_SOs, avg(ERA) as avg_ERA, sum(W) as total_wins, sum(SV) as total_saves, count(YearID) as years_pitched
FROM
(select h.*, sum(RBI) as total_RBIs, sum(SB) as total_stolen_bases, sum(BB) as total_walks,
sum(R) as total_runs, sum(H) as total_hits, count(yearID) as years_batted, sum(HR) as total_HRs, sum('2B') as total_2B, sum('3B') as total_3B
FROM 
(SELECT playerID, max(yearID) as final_year_voted, count(yearID) as years_voted, inducted
FROM HallofFame 
Where yearID < 2000
GROUP BY playerID) h
LEFT JOIN Batting b on h.playerID = b.playerID
GROUP BY h.playerID) m
LEFT JOIN Pitching p on m.playerID = p.playerID
group by m.playerID) a
LEFT JOIN Fielding f on a.playerID = f.playerID
GROUP BY a.playerID) c
Left Join PitchingPost pp on c.playerID = pp.playerID
GROUP BY c.playerID) d
Left Join BattingPost bp on d.playerID = bp.playerID
Group By d.playerID
'''

df = pandas.read_sql(sql, conn)

conn.close()

# Cleans up imported data
df.head()
pandas.set_option('display.max_columns', None)
df.head()

df.describe()

df.dropna(how = 'all', inplace = True)

df['inducted1'] = 0
df.inducted1[df.inducted == 'Y'] = 1

df['years_played'] = 0
df.years_played[df.years_pitched >= df.years_batted] = df.years_pitched
df.years_played[df.years_pitched < df.years_batted] = df.years_batted

df.drop(['playerID', 'inducted', 'years_pitched', 'years_batted', 'final_year_voted'],  1, inplace = True)

df.head(10)

df.describe()

# Sets up explanatory and reponse variables
explanatory_features = [col for col in df.columns if col not in ['inducted1']]
explanatory_df = df[explanatory_features]

explanatory_df.head()

explanatory_col_names = explanatory_df.columns

response_series = df.inducted1

response_series.index[~response_series.index.isin(explanatory_df.index)]

# Replace Nans with column means
from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='mean', axis=0)

imputer_object.fit(explanatory_df)
explanatory_df = imputer_object.transform(explanatory_df)

naive_bayes_classifier = MultinomialNB()

# Confusion matrix

from sklearn.cross_validation import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(
                    explanatory_df, response_series, test_size =  0.3)

# get predictions on the test group 
y_predicted = naive_bayes_classifier.fit(xTrain, yTrain).predict(xTest)

cm = pandas.crosstab(yTest, y_predicted, rownames=['True Label'], colnames=['Predicted Label'], margins=True)

print cm

# Accuracy
accuracy_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring = 'accuracy', n_jobs = -1)

print accuracy_scores.mean()

# Cohen's Kappa
mean_accuracy_score = accuracy_scores.mean()
largest_class_percent_of_total = response_series.value_counts(normalize = True)[0]

kappa = (mean_accuracy_score - largest_class_percent_of_total) / (1 - largest_class_percent_of_total) 
print kappa


# F1 scores
f1_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv = 10, scoring = 'f1', n_jobs = -1)

print f1_scores.mean()

# Roc scores
roc_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv = 10, scoring = 'roc_auc', n_jobs = 01)

print roc_scores.mean()


###############################################################################

# Decision Tree

from sklearn import tree

decision_tree = tree.DecisionTreeClassifier(random_state=1)

decision_tree.fit(xTrain, yTrain)

# Confusion matrix
predicted_values = decision_tree.predict(xTest)

cm = pandas.crosstab(yTest, predicted_values, rownames=['True Label'], colnames=['Predicted Label'], margins=True)

print cm

# Finds the importances of each feature
importances_df = pandas.DataFrame(explanatory_col_names)
importances_df['importances'] = decision_tree.feature_importances_

print importances_df

# Accuracy
accuracy_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='accuracy', n_jobs = -1)

print accuracy_scores_cart.mean()
print accuracy_scores.mean()

# Cohen's Kappa
mean_accuracy_score_cart = accuracy_scores_cart.mean()

kappa_cart = (mean_accuracy_score_cart - largest_class_percent_of_total) / (1-largest_class_percent_of_total)

print kappa_cart
print kappa


# F1 score
f1_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='f1', n_jobs = -1)

print f1_scores_cart.mean()
print f1_scores.mean()

# ROC
roc_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

print roc_scores_cart.mean()
print roc_scores.mean()

# KNN

from sklearn.neighbors import KNeighborsClassifier
from __future__ import division
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV


KNN = KNeighborsClassifier(p = 2)

KNN.fit(xTrain, yTrain)

# Confusion matrix
predicted_values = KNN.predict(xTest)

cm = pandas.crosstab(yTest, predicted_values, rownames=['True Label'], colnames=['Predicted Label'], margins=True)

print cm

accuracy_scores_KNN = cross_val_score(KNN, explanatory_df, response_series, cv=10, scoring='accuracy', n_jobs = -1)

print accuracy_scores_KNN.mean()

# Cohen's Kappa
mean_accuracy_score_KNN = accuracy_scores_KNN.mean()

kappa_KNN = (mean_accuracy_score_KNN - largest_class_percent_of_total) / (1-largest_class_percent_of_total)

print kappa_KNN


# F1 score
f1_scores_KNN = cross_val_score(KNN, explanatory_df, response_series, cv=10, scoring='f1', n_jobs = -1)

print f1_scores_KNN.mean()

# ROC
roc_scores_KNN = cross_val_score(KNN, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

print roc_scores_KNN.mean()

# ROC curve
predicted_probs_cart = pandas.DataFrame(decision_tree.predict_proba(xTest))
predicted_probs_NB = pandas.DataFrame(naive_bayes_classifier.predict_proba(xTest))
predicted_probs_KNN = pandas.DataFrame(KNN.predict_proba(xTest))

from sklearn import metrics
import matplotlib.pyplot as plt

fpr_cart, tpr_cart, thresholds_cart = metrics.roc_curve(yTest, predicted_probs_cart[1])
fpr_NB, tpr_NB, thresholds_NB = metrics.roc_curve(yTest, predicted_probs_NB[1])
fpr_KNN, tpr_KNN, thresholds_KNN = metrics.roc_curve(yTest, predicted_probs_KNN[1])

plt.figure()
plt.plot(fpr_cart, tpr_cart, color = 'r')
plt.plot(fpr_KNN, tpr_KNN, color = 'b')
plt.plot(fpr_NB, tpr_NB, color = 'g')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')




##############################################################################

# Print ROC curve on 10 fold CV
# Was not able to get it to work

#print(__doc__)
#
#import numpy as np
#from scipy import interp
#
#from sklearn import svm, datasets
#from sklearn.metrics import roc_curve, auc
#from sklearn.cross_validation import StratifiedKFold
#
#X = explanatory_df
#y = response_series
#X, y = X[y != 2], y[y != 2]
#n_samples, n_features = X.shape
#
## Add noisy features
#random_state = np.random.RandomState(0)
#X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
#
################################################################################
## Classification and ROC analysis
#
## Run classifier with cross-validation and plot ROC curves
#cv = StratifiedKFold(y, n_folds=10)
#classifier = svm.SVC(kernel='linear', probability=True,
#                     random_state=random_state)
#
#mean_tpr = 0.0
#mean_fpr = np.linspace(0, 1, 100)
#all_tpr = []
#
#for i, (train, test) in enumerate(cv):
#    predicted_probs = pandas.DataFrame(decision_tree.predict_proba(xTest))
#    # Compute ROC curve and area the curve
#    fpr, tpr, thresholds_cart = metrics.roc_curve(yTest, predicted_probs[1])
#    mean_tpr += interp(mean_fpr, fpr, tpr)
#    mean_tpr[0] = 0.0
#    roc_auc = auc(fpr, tpr)
#    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
#
#plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
#
#mean_tpr /= len(cv)
#mean_tpr[-1] = 1.0
#mean_auc = auc(mean_fpr, mean_tpr)
#plt.plot(mean_fpr, mean_tpr, 'k--',
#         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
#
#plt.xlim([-0.05, 1.05])
#plt.ylim([-0.05, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
##plt.legend(loc="lower right")
#plt.show()
#
