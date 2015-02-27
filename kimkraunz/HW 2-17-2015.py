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

df.drop(['playerID', 'inducted', 'years_pitched', 'years_batted'],  1, inplace = True)

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

# ROC curve
predicted_probs = pandas.DataFrame(decision_tree.predict_proba(xTest))

from sklearn import metrics
import matplotlib.pyplot as plt

fpr_cart, tpr_cart, thresholds_cart = metrics.roc_curve(yTest, predicted_probs[1])
plt.figure()
plt.plot(fpr_cart, tpr_cart, color = 'b')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

#########################################################################################

'''
I wanted to see if I could create a more robust model using Naive Bayes and Decision Trees than I had with the KNN model.  I had limited the features in the KNN model because I wanted to keep my accuracy consistent between my training and testing groups.  I will now repeat the Naive Bayes and Decision Tree exercise on my KNN model.
'''

conn = sqlite3.connect('/Users/jkraunz/Documents/SQLite/lahman2013.sqlite.crdownload')

sql2 = '''
Select a.playerID, a.final_year_voted, a.years_voted, a.inducted, a.total_hits, a.total_years_b, a.total_HRs,
a.total_SOs, a.avg_ERA, a.total_wins, a.total_saves, a.years_pitched, sum(E) as total_errors, sum(G) as games_played_fielding
FROM 
(SELECT m.playerID, m.final_year_voted, m.years_voted, m.inducted, m.total_hits, m.total_years_b, m.total_HRs,
sum(SO) as total_SOs, avg(ERA) as avg_ERA, sum(W) as total_wins, sum(SV) as total_saves, count(YearID) as years_pitched
FROM
(select h.playerID, h.final_year_voted, h.years_voted, h.inducted,
sum(H) as total_hits, count(yearID) as total_years_b, sum(HR) as total_HRs
FROM 
(SELECT playerID, max(yearID) as final_year_voted, count(yearID) as years_voted, inducted
FROM HallofFame 
Where yearID < 2000
GROUP BY playerID) h
LEFT JOIN Batting b on h.playerID = b.playerID
GROUP BY h.playerID) m
LEFT JOIN Pitching p on m.playerID = p.playerID
GROUP BY m.playerID) a
LEFT JOIN Fielding f on a.playerID = f.playerID
GROUP BY a.playerID
'''

df2 = pandas.read_sql(sql2, conn)

conn.close()

# Cleans up imported data
df2.head()
pandas.set_option('display.max_columns', None)
df2.head()

df2.describe()

df2.dropna(how = 'all', inplace = True)

df2['inducted1'] = 0
df2.inducted1[df2.inducted == 'Y'] = 1

df2['years_played'] = 0
df2.years_played[df2.years_pitched >= df2.total_years_b] = df2.years_pitched
df2.years_played[df2.years_pitched < df2.total_years_b] = df2.total_years_b

df2.drop(['playerID', 'inducted', 'years_pitched', 'total_years_b'],  1, inplace = True)

df2.head(10)

df2.describe()

# Sets up explanatory and reponse variables
explanatory_features = [col for col in df2.columns if col not in ['inducted1']]
explanatory_df2 = df2[explanatory_features]

explanatory_df2.head()

explanatory_col_names2 = explanatory_df2.columns

response_series2 = df2.inducted1

response_series2.index[~response_series2.index.isin(explanatory_df2.index)]

# Replace Nans with column means
from sklearn.preprocessing import Imputer
imputer_object2 = Imputer(missing_values='NaN', strategy='mean', axis=0)

imputer_object2.fit(explanatory_df2)
explanatory_df2 = imputer_object2.transform(explanatory_df2)

naive_bayes_classifier2 = MultinomialNB()

# Confusion matrix

from sklearn.cross_validation import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(
                    explanatory_df2, response_series2, test_size =  0.3)

# get predictions on the test group 
y_predicted = naive_bayes_classifier2.fit(xTrain, yTrain).predict(xTest)

cm = pandas.crosstab(yTest, y_predicted, rownames=['True Label'], colnames=['Predicted Label'], margins=True)

print cm

# Accuracy
accuracy_scores2 = cross_val_score(naive_bayes_classifier2, explanatory_df2, response_series, cv=10, scoring = 'accuracy', n_jobs = -1)

print accuracy_scores2.mean()

# Cohen's Kappa
mean_accuracy_score2 = accuracy_scores2.mean()
largest_class_percent_of_total2 = response_series2.value_counts(normalize = True)[0]

kappa2 = (mean_accuracy_score2 - largest_class_percent_of_total2) / (1 - largest_class_percent_of_total2) 
print kappa2


# F1 scores
f1_scores2 = cross_val_score(naive_bayes_classifier2, explanatory_df2, response_series, cv = 10, scoring = 'f1', n_jobs = -1)

print f1_scores2.mean()

# Roc scores
roc_scores2 = cross_val_score(naive_bayes_classifier2, explanatory_df2, response_series, cv = 10, scoring = 'roc_auc', n_jobs = 01)

print roc_scores2.mean()


###############################################################################

# Decision Tree

from sklearn import tree

decision_tree2 = tree.DecisionTreeClassifier(random_state=1)

decision_tree2.fit(xTrain, yTrain)

# Confusion matrix
predicted_values = decision_tree2.predict(xTest)

cm = pandas.crosstab(yTest, predicted_values, rownames=['True Label'], colnames=['Predicted Label'], margins=True)

print cm

# Finds the importances of each feature
importances_df = pandas.DataFrame(explanatory_col_names2)
importances_df['importances'] = decision_tree2.feature_importances_

print importances_df

# Accuracy
accuracy_scores_cart2 = cross_val_score(decision_tree2, explanatory_df2, response_series, cv=10, scoring='accuracy', n_jobs = -1)

print accuracy_scores_cart2.mean()
print accuracy_scores2.mean()
print accuracy_scores_cart.mean()
print accuracy_scores.mean()

# Cohen's Kappa
mean_accuracy_score_cart2 = accuracy_scores_cart2.mean()

kappa_cart2 = (mean_accuracy_score_cart2 - largest_class_percent_of_total2) / (1-largest_class_percent_of_total2)

print kappa_cart2
print kappa2
print kappa_cart
print kappa


# F1 score
f1_scores_cart2 = cross_val_score(decision_tree2, explanatory_df2, response_series, cv=10, scoring='f1', n_jobs = -1)

print f1_scores_cart2.mean()
print f1_scores2.mean()
print f1_scores_cart.mean()
print f1_scores.mean()

# ROC
roc_scores_cart2 = cross_val_score(decision_tree2, explanatory_df2, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

print roc_scores_cart2.mean()
print roc_scores2.mean()
print roc_scores_cart.mean()
print roc_scores.mean()

# ROC curve
predicted_probs = pandas.DataFrame(decision_tree2.predict_proba(xTest))

from sklearn import metrics
import matplotlib.pyplot as plt

fpr_cart, tpr_cart, thresholds_cart = metrics.roc_curve(yTest, predicted_probs[1])
plt.figure()
plt.plot(fpr_cart, tpr_cart, color = 'b')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

###############################################################################

'''
My scores went up marginally for all parameters.  I still found that the Decision Tree was a better tool to predict induction.  Since model #1 was had so many features and model #2 did not have very many and the Decision Tree was clearly the better tool, I decided to use the importances from the decision tree to try to refine the model even more.  I started with model #2 and added in feature importances from model #1 that had an importance factor greater than 0.001.
'''

conn = sqlite3.connect('/Users/jkraunz/Documents/SQLite/lahman2013.sqlite.crdownload')

sql3 = '''
Select c.*, sum(H) as total_post_hits, sum(HR) as total_post_HRs, sum(RBI) as total_post_RBIs
FROM
(Select a.*, sum(E) as total_errors
FROM
(SELECT m.*,
sum(SO) as total_SOs, avg(ERA) as avg_ERA, sum(W) as total_wins, sum(SV) as total_saves, count(YearID) as years_pitched
FROM
(select h.*, sum(RBI) as total_RBIs, sum(SB) as total_stolen_bases, sum(R) as total_runs, sum(H) as total_hits, count(yearID) as years_batted, sum(HR) as total_HRs, sum('2B') as total_2B, sum('3B') as total_3B
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
Left Join BattingPost bp on c.playerID = bp.playerID
Group By c.playerID
'''

df = pandas.read_sql(sql3, conn)

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

df.drop(['playerID', 'inducted', 'years_pitched', 'years_batted'],  1, inplace = True)

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


###############################################################################
# Naive Bayes

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
accuracy_scores_cart3 = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='accuracy', n_jobs = -1)

print accuracy_scores_cart3.mean()
print accuracy_scores_cart2.mean()
print accuracy_scores_cart.mean()

# Cohen's Kappa
mean_accuracy_score_cart3 = accuracy_scores_cart3.mean()
largest_class_percent_of_total3 = response_series.value_counts(normalize = True)[0]
kappa_cart3 = (mean_accuracy_score_cart3 - largest_class_percent_of_total3) / (1-largest_class_percent_of_total3)

print kappa_cart3
print kappa_cart2
print kappa_cart


# F1 score
f1_scores_cart3 = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='f1', n_jobs = -1)

print f1_scores_cart3.mean()
print f1_scores_cart2.mean()
print f1_scores_cart.mean()

# ROC
roc_scores_cart3 = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

print roc_scores_cart3.mean()
print roc_scores_cart2.mean()
print roc_scores_cart.mean()

################################################################################
# KNN on last model
from sklearn.neighbors import KNeighborsClassifier
from __future__ import division
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

response_series = df.inducted1

# defines the independent variables in the dataframe
explanatory_variables = df[['final_year_voted', 'years_voted', 'total_RBIs', 'total_stolen_bases', 'total_runs', 'total_hits', 'total_HRs', 'total_2B', 'total_3B', 'total_SOs', 'avg_ERA', 'total_wins', 'total_saves', 'total_errors', 'total_post_hits', 'total_post_HRs', 'total_post_RBIs', 'years_played']]

explanatory_variables.head()
explanatory_variables = explanatory_variables.fillna(explanatory_variables.mean())
explanatory_variables.head()

CROSS_VALIDATION_AMOUNT = .2
holdout_num = round(len(df.index) * CROSS_VALIDATION_AMOUNT, 0)

# prints the holdout number
holdout_num

# defines the test_indices using the random module in numpy based on the dataframe index and # to holdout
test_indices = numpy.random.choice(df.index, holdout_num, replace = False)

# defines the train_indices as the opposite of what is in the test_indices defined above
train_indices = df.index[~df.index.isin(test_indices)]

# defines the responses for the training  to the response series (inducted) from the randomly assigned training indices 
response_train = response_series.ix[train_indices,]

# checks the response_train series
response_train.head()

# defines the explanatory for the training to the explanatory variables in the train indices
explanatory_train = explanatory_variables.ix[train_indices,]

# tests that the response training and explanatory training are in the same indices
response_train.index == explanatory_train.index

# defines the response test series as the reponse variable (induction) in the test indices
response_test = response_series.ix[test_indices,]

# defines the explanatory test series as the explantory variables in the test indices
explanatory_test = explanatory_variables.ix[test_indices,]

# checks the response_test and explanatory_test series
response_test.head()
explanatory_test.head()


# determines the optimal number for k using the built in functions KNeighbors Classifier
# and GridSearchCV
knn = KNeighborsClassifier( p = 2)
k_range = range(1, 30, 2)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(explanatory_variables, response_series)

# plots the scores versus k
grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(k_range, grid_mean_scores)
best_oob_score = grid.best_score_
grid.best_params_
Knn_optimal = grid.best_estimator_

print best_oob_score
print k_range, grid_mean_scores

# creates model based on K nearest neighbors (optimal k found above to be 9) on
# the training group
KNN_classifier = KNeighborsClassifier(n_neighbors = 13, p = 2)
KNN_classifier.fit(explanatory_train, response_train)

# tests accuracy of model above on test group
predicted_response = KNN_classifier.predict(explanatory_test)
predicted_response

# calculates accuracy of the KNN model
number_correct = len(response_test[response_test == predicted_response])
total_in_test_set = len(response_test)
smaller_accuracy = number_correct / total_in_test_set
print "The KNN accuracy is: %f" % (100*smaller_accuracy)

##############################################################################

# Print ROC curve on 10 fold CV

print(__doc__)

import numpy as np
from scipy import interp

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold

X = explanatory_df
y = response_series
X, y = X[y != 2], y[y != 2]
n_samples, n_features = X.shape

# Add noisy features
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

###############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(y, n_folds=10)
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    predicted_probs = pandas.DataFrame(decision_tree.predict_proba(xTest))
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds_cart = metrics.roc_curve(yTest, predicted_probs[1])
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
#plt.legend(loc="lower right")
plt.show()

'''
I tested three different models using Naive Bayes and the Decision Tree model.  In all three models, accuracy, Cohen's Kappa, F1 scores, and ROC scores were higher for the Decision Tree than for Naive Bayes.  

My first model included the following features to predict Hall of Fame induction: 

total postseason hits
total postseason home runs
total postseason RBIs
total postseason wins
total postseason saves
average postseason ERA
total regular season errors
total regular season strikeouts
average regular season ERA
total regular season wins
total regular season saves
total regular season RBIs
total regular season stolen bases
total regular season walks
total regular season runs
total regular season hits
total regular season home runs
total regular season doubles
total regular season triples
number of years played
number of year voted for
final year voted

The accuracy was 0.829, Cohen's Kappa was 0.257, the F1 scores was 0.632, and the ROC score was 0.767 for the Decision Tree model.  The Naive Bayes had much lower model measurements.

I wanted to test my KNN model using Naive Bayes and the Decision Tree model.  That model included the following features:

total regular season errors
total regular season strikeouts
average regular season ERA
total regular season wins
total regular season saves
total regular season hits
total regular season home runs
number of years played
number of year voted for
final year voted

All model measurements increased modestly.  The accuracy was 0.842, Cohen's Kappa was 0.312, the F1 scores was 0.654, and the ROC score was 0.780 for the Decision Tree model.  The Naive Bayes had much lower model measurements.

Lastly, I wanted to test whether I could include more features than my basic KNN model but still increase the model measurements to have a more predictive model.  I included any features that had an importance greater than 0.001 in my first model.

My final model included the following features:

total postseason hits
total postseason home runs
total postseason RBIs
total regular season errors
total regular season strikeouts
average regular season ERA
total regular season wins
total regular season saves
total regular season RBIs
total regular season stolen bases
total regular season runs
total regular season hits
total regular season home runs
total regular season doubles
total regular season triples
number of years played
number of year voted for
final year voted

All model measurements increased modestly.  The accuracy was 0.844, Cohen's Kappa was 0.321, the F1 scores was 0.658, and the ROC score was 0.781 for the Decision Tree model.  The Naive Bayes had much lower model measurements.  The Cohen's Kappa model measurement indicates that it is substantial.

The confusion matrix for the final model using Naive Bayes was:
Predicted Label    0    1  All
True Label                    
0                132   96  228
1                 28   29   57
All              160  125  285
It has low specificity.

The confusion matrix for the final model using the Decision Tree model was:
Predicted Label    0   1  All
True Label                   
0                207  20  227
1                 24  34   58
All              231  54  285
The specificity (False Positives) decreased by using the Decision Tree model.

Lastly, I reran my new model with K Nearest Neighbor to see the effect on the KNN accuracy.  I found that in my final model, that the accuracy of KNN was .868 as compared to the Decision Tree model which was 0.844.  Therefore, KNN would be the best model to use for the data to predict Hall of Fame induction.
'''
