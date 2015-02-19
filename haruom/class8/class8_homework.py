# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 17:25:41 2015

@author: Haruo M
"""
import numpy
import pandas
import sqlite3
from sklearn.neighbors import KNeighborsClassifier

con = sqlite3.connect('C:\Users\mizutani\Documents\SQLite\lahman2013.sqlite')

sql = """SELECT h.playerID, h.inducted, SUM(b.G_batting) AS total_games_batter, SUM(b.AB) AS total_at_bats, SUM(b.H) AS total_hits, SUM(b.HR) AS total_homeruns, SUM(b.SB) AS total_stolen_base
FROM hall_of_fame_inductees h
LEFT JOIN Batting b ON h.playerID = b.playerID
GROUP BY h.playerID;"""

df = pandas.read_sql(sql, con)
con.close()

# Batting averages are calurated and added to the dataframe
df['batting_average'] = df.total_hits / df.total_at_bats
df.dropna(inplace = True)

pandas.set_option('display.max_columns', None)

df.head(10)
df.columns

response_series = df.inducted 
explanatory_df = df[['total_games_batter','total_at_bats', 'total_hits', 'total_homeruns', 'total_stolen_base', 'batting_average']]

# Imputer fills Nas
from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_object.fit(explanatory_df)
explanatory_df = imputer_object.transform(explanatory_df)

# Naive bayes classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score
naive_bayes_classifier = MultinomialNB()
accuracy_scores_bayse = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='accuracy', n_jobs = 1)
mean_accuracy_score_bayse = accuracy_scores_bayse.mean()
print mean_accuracy_score_bayse


## Cohen's Kappa
largest_class_percent_of_total = response_series.value_counts(normalize = True)[0]
kappa_bayse = (mean_accuracy_score_bayse - largest_class_percent_of_total) / (1-largest_class_percent_of_total)
print kappa_bayse
# F1 score
f1_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='f1', n_jobs = 1)
mean_f1_score_bayse = f1_scores.mean()
print mean_f1_score_bayse
## ROC area under the curve score. 
roc_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = 1)
mean_roc_score_bayse = roc_scores.mean()
print mean_roc_score_bayse

#Cross Validation
from sklearn.cross_validation import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(explanatory_df, response_series, test_size =  0.2)
## All training data
xxTrain, xxTest, yyTrain, yyTest = train_test_split(explanatory_df, response_series, test_size =  0)
y_predicted = naive_bayes_classifier.fit(xTrain, yTrain).predict(xTest)

#Confusion Matrix
cm = pandas.crosstab(yTest, y_predicted, rownames=['True Label'], colnames=['Predicted Label'], margins=True)
print cm

y_probabilities = pandas.DataFrame(naive_bayes_classifier.fit(xTrain, yTrain).predict_proba(xTest))

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(yTest, y_probabilities[1])
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
#********************************************

# Decistion tree classification
from sklearn import tree
decision_tree = tree.DecisionTreeClassifier(random_state=1)
decision_tree = tree.DecisionTreeClassifier(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)
decision_tree.fit(xTrain, yTrain)

predicted_values = decision_tree.predict(xTest)
cm = pandas.crosstab(yTest, predicted_values, rownames=['True Label'], colnames=['Predicted Label'], margins=True)
print cm

predicted_probs = pandas.DataFrame(decision_tree.predict_proba(xTest))

fpr_cart, tpr_cart, thresholds_cart = metrics.roc_curve(yTest, predicted_probs[1])
plt.figure()
plt.plot(fpr, tpr, color = 'g')
plt.plot(fpr_cart, tpr_cart, color = 'b')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

accuracy_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='accuracy', n_jobs = 1)
mean_accuracy_score_cart = accuracy_scores_cart.mean()
print mean_accuracy_score_cart

mean_accuracy_score_cart = accuracy_scores_cart.mean()
kappa_cart = (mean_accuracy_score_cart - largest_class_percent_of_total) / (1-largest_class_percent_of_total)
f1_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='f1', n_jobs = 1)
mean_f1_score_cart = f1_scores_cart.mean()
roc_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = 1)
mean_roc_score_cart = roc_scores_cart.mean()

# KNN classification in class 5 homework
from sklearn.cross_validation import cross_val_score
KNN_classifier = KNeighborsClassifier(n_neighbors=37, p = 2)
scores = cross_val_score(KNN_classifier, explanatory_df, response_series, cv=10, scoring='accuracy')
print scores
mean_accuracy_score_KNN = numpy.mean(scores) 

f1_scores_knn = cross_val_score(KNN_classifier, explanatory_df, response_series, cv=10, scoring='f1', n_jobs = 1)
mean_f1_score_knn = f1_scores_knn.mean()
roc_scores_knn = cross_val_score(KNN_classifier, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = 1)
mean_roc_score_knn = roc_scores_knn.mean()

y_predicted_knn = KNN_classifier.fit(xTrain, yTrain).predict(xTest)
cm = pandas.crosstab(yTest, y_predicted_knn, rownames=['True Label'], colnames=['Predicted Label'], margins=True)
print cm

predicted_probs_knn = pandas.DataFrame(KNN_classifier.fit(xTrain, yTrain).predict_proba(xTest))
fpr_knn, tpr_knn, thresholds_knn = metrics.roc_curve(yTest, predicted_probs_knn[1])

# Accuracy, f1, roc scores comparison between naive bayse, decision tree and KNN
print 'Accuracy Comparison'
print 'Naive Bayse: {0}%'.format(round(mean_accuracy_score_bayse * 100, 2))
print 'Decistion Tree: {0}%'.format(round(mean_accuracy_score_cart * 100, 2))
print 'KNN: {0}%'.format(round(mean_accuracy_score_KNN * 100, 2))

print mean_f1_score_bayse
print mean_f1_score_cart
print mean_f1_score_knn

print mean_roc_score_bayse
print mean_roc_score_cart
print mean_roc_score_knn


# ROC comparison between naive bayse, decision tree and KNN
plt.figure()
plt.plot(fpr, tpr, color = 'g')
plt.plot(fpr_cart, tpr_cart, color = 'b')
plt.plot(fpr_knn, tpr_knn, color = 'r')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

print 'KNN classification is the best!!'


##################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves

from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import StratifiedKFold

cv = StratifiedKFold(response_series, n_folds=10)
KNN_classifier = KNeighborsClassifier(n_neighbors=37, p = 2)

mean_tpr = 0.0
mean_fpr = numpy.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas_ = KNN_classifier.fit(xxTrain[train], yyTrain[train]).predict_proba(xxTrain[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(yyTrain[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


