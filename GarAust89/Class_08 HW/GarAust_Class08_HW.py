# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 10:54:51 2015

@author: garauste
"""

import pandas
import sqlite3

## CONNECT TO THE BASEBALL DATABASE
conn = sqlite3.connect('C:\Users\garauste\Documents\SQLite\lahman2013.sqlite')

## Create a large SQL object to simultaneously pull data from all 4 Tables in the Baseball database: HallOfFame, Batting, Pitching, Fielding
sql_batters = """select a.playerID as playerID, a.inducted as inducted, b.*, c.*
from hall_of_fame_inductees a 
left outer join 
(
select sum(b.G) as Games, sum(b.H) as Hits, sum(b.AB) as At_Bats, sum(b.HR) as Homers,
b.playerID from Batting b group by b.playerID
) b
on a.playerID = b.playerID
left outer join 
(
select d.playerID, sum(d.A) as Fielder_Assists, sum(d.E) as Field_Errors, 
sum(d.DP) as Double_Plays from Fielding d group by d.playerID
) c
on a.playerID = c.playerID
"""

## Pass the connection and the SQL String to a pandas.read file 
df_batters = pandas.read_sql(sql_batters,conn)

#Close the connection 
conn.close()

# drop duplicate playerIDs
df_batters.drop('playerID',1,inplace=True)

df_batters.head()
df_batters.describe()

## CONNECT TO THE BASEBALL DATABASE
conn = sqlite3.connect('C:\Users\garauste\Documents\SQLite\lahman2013.sqlite')

## Create a large SQL object to simultaneously pull data from all 4 Tables in the Baseball database: HallOfFame, Batting, Pitching, Fielding
sql_pitchers = """select a.playerID, a.inducted, b.* 
from  hall_of_fame_inductees a 
left outer join
(
select b.playerID, sum(b.ERA) as Pitcher_Earned_Run_Avg, sum(b.W) as Pitcher_Ws, sum(b.SHO) as Pitcher_ShutOuts, sum(b.SO) as Pitcher_StrikeOuts,
sum(b.HR) as HR_Allowed, sum(b.CG) as Complete_Games 
from Pitching b 
group by b.playerID
) b 
on a.playerID = b.playerID 
where Pitcher_Ws is not null;"""

## Pass the connection and the SQL String to a pandas.read file 
df_pitchers = pandas.read_sql(sql_pitchers,conn)

#Close the connection 
conn.close()


# Dropping all NaNs in the dataset
df_batters.dropna(inplace=True)
df_pitchers.dropna(inplace=True)

df_batters.describe()
df_pitchers.describe()

### Splitting out the explanatory features
explan_batters = [col for col in df_batters.columns if col not in ('playerID','inducted')]
df_batters_explanatory = df_batters[explan_batters]

explan_pitchers = [col for col in df_pitchers.columns if col not in ('playerID','inducted')]
df_pitchers_explanatory = df_pitchers[explan_pitchers]

## Drop the rows with no data
df_batters_explanatory.dropna(how='all',inplace = True)
df_pitchers_explanatory.dropna(how='all',inplace = True)

# extracting colnames 
df_batters_explanatory_colnames = df_batters_explanatory.columns
df_pitchers_explanatory_colnames = df_pitchers_explanatory.columns

## Repeat the above for the response
response_batters = df_batters.inducted
response_pitchers = df_pitchers.inducted


#response_batters.dropna(how='all',inplace=True)
#response_pitchers.dropna(how='all', inplace=True)

# imputing NaNs with the mean value for that column. We will go over this in further 
# detail in next weeks class
from sklearn.preprocessing import Imputer

imputer_object = Imputer(missing_values='NaN',strategy = 'mean', axis=0)
# fitting the object on our data - we do this so that we can save the fit for our new data
imputer_object.fit(df_batters_explanatory)
df_batters_explanatory = imputer_object.transform(df_batters_explanatory)

imputer_object.fit(df_pitchers_explanatory)
df_pitchers_explanatory = imputer_object.transform(df_pitchers_explanatory)

# create a naive Bayes classifier and get it cross validated accuracy score
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score

# creating the naive bayes classifier object 
NB_Classifier = MultinomialNB()

# running a cross - validates score on accuracy. Notice set n_jobs = -1 
# this means that we are going to use all of the computers cores to find the results
accuracy_scores_batters = cross_val_score(NB_Classifier, 
df_batters_explanatory, response_batters, cv=10,scoring = 'accuracy')

accuracy_scores_pitchers = cross_val_score(NB_Classifier, 
df_pitchers_explanatory, response_pitchers, cv=10,scoring = 'accuracy')

# let's see how accurate the model is, on average. 
print accuracy_scores_batters.mean()
print accuracy_scores_pitchers.mean()

# let's calculate Cohen's Kappa
mean_accuracy_score_batters = accuracy_scores_batters.mean()
mean_accuracy_score_pitchers = accuracy_scores_pitchers.mean()

largest_class_percent_of_total_batters = response_batters.value_counts(normalize = True)[0]
largest_class_percent_of_total_pitchers = response_pitchers.value_counts(normalize = True)[0]

kappa_batters = (mean_accuracy_score_batters - largest_class_percent_of_total_batters) / (1 -
largest_class_percent_of_total_batters)
print kappa_batters

kappa_pitchers = (mean_accuracy_score_pitchers - largest_class_percent_of_total_pitchers) / (1 -
largest_class_percent_of_total_pitchers)
print kappa_pitchers

# calculating F1 score. which is the harmonic of specificity
f1_scores_batters = cross_val_score(NB_Classifier, df_batters_explanatory,
response_batters,cv = 10, scoring='f1')
print f1_scores_batters.mean()

f1_scores_pitchers = cross_val_score(NB_Classifier, df_pitchers_explanatory,
response_pitchers,cv = 10, scoring='f1')
print f1_scores_pitchers.mean()

# Calculating the ROC area under the curve score. 
roc_scores_batters = cross_val_score(NB_Classifier, df_batters_explanatory,
response_batters,cv = 10, scoring='roc_auc')
print roc_scores_batters.mean()

roc_scores_pitchers = cross_val_score(NB_Classifier, df_pitchers_explanatory,
response_pitchers,cv = 10, scoring='roc_auc')
print roc_scores_pitchers.mean()

# Here's the interpretability of AUC 
# .90 -1= excellent
# .8 - .9 = good
# .7 - .8 = fair
# .6 - .7 = poor 
# .5 - .6 = fail
# Our Batters model is a fail on this scale however our pitching model has a fair 
# performance

# pulling out a training and test slice from the data
from sklearn.cross_validation import train_test_split

xTrain_bat, xTest_bat, yTrain_bat, yTest_bat = train_test_split(df_batters_explanatory,
response_batters, test_size = 0.3)

xTrain_p, xTest_p, yTrain_p, yTest_p = train_test_split(df_batters_explanatory,
response_batters, test_size = 0.3)

# get predictions on the test slice of the classifier
y_predicted_bat = NB_Classifier.fit(xTrain_bat, yTrain_bat).predict(xTest_bat)
y_predicted_p = NB_Classifier.fit(xTrain_p, yTrain_p).predict(xTest_p)

# create a confusion matrix for the data
cm_bat = pandas.crosstab(yTest_bat, y_predicted_bat, rownames=['True Label'],
colnames = ['Predicted Label'],margins = True)

print cm_bat

cm_p = pandas.crosstab(yTest_p, y_predicted_p, rownames=['True Label'],
colnames = ['Predicted Label'],margins = True)

print cm_p

# extracting probabilities for the clasifier 
y_probabilities_bat=pandas.DataFrame(NB_Classifier.fit(xTrain_bat,yTrain_bat).predict_proba(xTest_bat))

y_probabilities_p=pandas.DataFrame(NB_Classifier.fit(xTrain_p,yTrain_p).predict_proba(xTest_p))

from sklearn import metrics
# remember to pass the ROC CURVE METHOD THE PROBABILITY OF A 'TRUE'
# CLLASS, OR COLUMN 1 IN THIS CASE
fpr_bat, tpr_bat, thresholds_bat = metrics.roc_curve(yTest_bat, y_probabilities_bat[1])
fpr_p, tpr_p, thresholds_p = metrics.roc_curve(yTest_p, y_probabilities_p[1])
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr_bat,tpr_bat)
plt.plot(fpr_p,tpr_p, lw = 2, color = 'red')
plt.xlabel('False Positives Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

# looking at the ROC curv, the f1 score
# ROC score, accuracy and Kappa how helpful is this estimator

##########
## CART ##
##########

# now let's create some classfication trees
from sklearn import tree
# create a decision tree classifier instance
decision_tree = tree.DecisionTreeClassifier(random_state=1)

# realize that the above code is the exact same as the code below, which shows the objects 
# default values. We can change these values to tune the tree. 
decision_tree = tree.DecisionTreeClassifier(criterion = 'gini', splitter = 'best',
                                            max_features = None, max_depth = None,
                                            min_samples_split = 2,
                                            min_samples_leaf = 2, max_leaf_nodes = None,
                                            random_state = 1)
                                            
# fit the tree classifier
decision_tree.fit(xTrain_bat,yTrain_bat)
decision_tree.fit(xTrain_p,yTrain_p)

## predict on the test data and look at confusion matrix
predicted_values_bat = decision_tree.predict(xTest_bat)
predicted_values_p = decision_tree.predict(xTest_p)

cm_bat = pandas.crosstab(yTest_bat,predicted_values_bat,rownames=['True Label'],
                     colnames = ['Predicted Label'],margins = True)
                     
print cm_bat

cm_p = pandas.crosstab(yTest_p,predicted_values_p,rownames=['True Label'],
                     colnames = ['Predicted Label'],margins = True)
                     
print cm_p
# extracting decision tree probabilities 
predicted_probs_bat = pandas.DataFrame(decision_tree.predict_proba(xTest_bat))
predicted_probs_p = pandas.DataFrame(decision_tree.predict_proba(xTest_p))

# now lets plot the ROC curve and compare it to the Naive Bayes
fpr_cart_bat, tpr_cart_bat, thresholds_cart_bat = metrics.roc_curve(yTest_bat,predicted_probs_bat[1])
plt.figure()
plt.plot(fpr_bat,tpr_bat,color = 'g')
plt.plot(fpr_cart_bat,tpr_cart_bat,color = 'b')
plt.xlabel('False Positives Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

fpr_cart_p, tpr_cart_p, thresholds_cart_p = metrics.roc_curve(yTest_p,predicted_probs_p[1])
plt.figure()
plt.plot(fpr_p,tpr_p,color = 'g')
plt.plot(fpr_cart_p,tpr_cart_p,color = 'b')
plt.xlabel('False Positives Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

## now let's do 10-fold CV, compute accuracy, f1, AUC, and Kappa
accuracy_scores_cart_bat = cross_val_score(decision_tree, df_batters_explanatory,
response_batters, cv=10, scoring='accuracy')

accuracy_scores_cart_p = cross_val_score(decision_tree, df_pitchers_explanatory,
response_pitchers, cv=10, scoring='accuracy')

print accuracy_scores_cart_bat.mean()
print accuracy_scores_cart_p.mean()
# mean accuracy of 90% or so. let's compare this to the naive Bayes mean accuracy.
print accuracy_scores_batters.mean()
print accuracy_scores_pitchers.mean()
# so, which model is more accurate? 

# let's calculate Cohen's Kappa
mean_accuracy_score_cart_bat = accuracy_scores_cart_bat.mean()
mean_accuracy_score_cart_p = accuracy_scores_cart_p.mean()
# recall we already calculated the largest_class_percent_of_total above.
kappa_cart_bat = (mean_accuracy_score_cart_bat - largest_class_percent_of_total_batters) / (1-largest_class_percent_of_total_batters)
kappa_cart_p = (mean_accuracy_score_cart_p - largest_class_percent_of_total_pitchers) / (1-largest_class_percent_of_total_pitchers)
print kappa_cart_bat
print kappa_cart_p
# so Kappa of 0.096.  What does this say in absolute terms of the 
# ability of the model to predict better than just random selection?

# let's compare to Naive Bayes. 
print kappa_batters
print kappa_pitchers
# which is better?

# calculating F1 score, which is the weighted average of specificity
# and sensitivity. 
f1_scores_cart_bat = cross_val_score(decision_tree, df_batters_explanatory, 
response_batters, cv=10, scoring='f1')

f1_scores_cart_p = cross_val_score(decision_tree, df_pitchers_explanatory, 
response_pitchers, cv=10, scoring='f1')

#compare F1 of decision tree and naive bayes
print f1_scores_cart_bat.mean()
print f1_scores_batters.mean()

print f1_scores_cart_p.mean()
print f1_scores_pitchers.mean()

## calculating the ROC area under the curve score. 
roc_scores_cart_bat = cross_val_score(decision_tree, df_batters_explanatory, 
response_batters, cv=10, scoring='roc_auc')

roc_scores_cart_p = cross_val_score(decision_tree, df_pitchers_explanatory, 
response_pitchers, cv=10, scoring='roc_auc')

# let's compare the decision tree with Naive Bayes.
print roc_scores_cart_bat.mean()
print roc_scores_batters.mean()

print roc_scores_cart_p.mean()
print roc_scores_pitchers.mean()


# now, let's fine-tune the tree model.
from sklearn.grid_search import  GridSearchCV

# Conduct a grid search for the best tree depth
decision_tree = tree.DecisionTreeClassifier(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)

depth_range = range(1, 20)
param_grid = dict(max_depth=depth_range)

grid_bat = GridSearchCV(decision_tree, param_grid, cv=10, scoring='roc_auc')
grid_bat.fit(df_batters_explanatory,response_batters)

grid_p = GridSearchCV(decision_tree, param_grid, cv=10, scoring='roc_auc')
grid_p.fit(df_pitchers_explanatory,response_pitchers)

# Check out the scores of the grid search
grid_mean_scores_bat = [result[1] for result in grid_bat.grid_scores_]
grid_mean_scores_p = [result[1] for result in grid_p.grid_scores_]

# Plot the results of the grid search
plt.figure()
plt.plot(depth_range, grid_mean_scores_bat)
plt.hold(True)
plt.plot(grid_bat.best_params_['max_depth'], grid_bat.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')
plt.grid(True)

# now let's calculate other accuracy metrics for the best estimator.
best_decision_tree_est_bat = grid_bat.best_estimator_
best_decision_tree_est_p = grid_p.best_estimator_

## lets compare on best accuracy
accuracy_scores_best_cart_bat = cross_val_score(best_decision_tree_est_bat,
df_batters_explanatory, response_batters, cv=10, scoring='accuracy')

accuracy_scores_best_cart_p = cross_val_score(best_decision_tree_est_p,
df_pitchers_explanatory, response_pitchers, cv=10, scoring='accuracy')

print accuracy_scores_best_cart_bat.mean()
print accuracy_scores_cart_bat.mean()

print accuracy_scores_best_cart_p.mean()
print accuracy_scores_cart_p.mean()

#accuracy scores look identical. So, Cohen's Kappa will be identical. 

# calculating F1 score, which is the weighted average of specificity
# and sensitivity. 
f1_scores_best_cart_bat = cross_val_score(best_decision_tree_est_bat, 
df_batters_explanatory, response_batters, cv=10, scoring='f1')

f1_scores_best_cart_p = cross_val_score(best_decision_tree_est_p, 
df_pitchers_explanatory, response_pitchers, cv=10, scoring='f1')

#compare F1 scores
print f1_scores_best_cart_bat.mean()
print f1_scores_cart_bat.mean()

print f1_scores_best_cart_p.mean()
print f1_scores_cart_p.mean()
## they're identical

## calculating the ROC area under the curve score. 
roc_scores_best_cart_bat = cross_val_score(best_decision_tree_est_bat,
df_batters_explanatory, response_batters, cv=10, scoring='roc_auc')

roc_scores_best_cart_p = cross_val_score(best_decision_tree_est_p,
df_pitchers_explanatory, response_pitchers, cv=10, scoring='roc_auc')

print roc_scores_best_cart_bat.mean()
print roc_scores_cart_bat.mean()

print roc_scores_best_cart_p.mean()
print roc_scores_cart_p.mean()
# Now let's plot the ROC curve of the  best grid estimator vs 
# our older decision tree classifier.
predicted_probs_cart_best_bat=pandas.DataFrame(best_decision_tree_est_bat.predict_proba(xTest_bat))

predicted_probs_cart_best_p=pandas.DataFrame(best_decision_tree_est_p.predict_proba(xTest_p))

fpr_cart_best_bat, tpr_cart_best_bat, thresholds_cart_best_bat = metrics.roc_curve(yTest_bat, predicted_probs_bat[1])
plt.figure()
plt.plot(fpr_bat, tpr_bat, color = 'g')
plt.plot(fpr_cart_bat, tpr_cart_bat, color = 'b')
plt.plot(fpr_cart_best_bat, tpr_cart_best_bat, color = 'r')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')


## 

print(__doc__)

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold

###############################################################################
# Data IO and generation

# import some data to play with
iris = df_batters
X = iris[explan_batters]
y = iris.inducted
X, y = X[y != 2], y[y != 2]
n_samples, n_features = X.shape

# Add noisy features
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

###############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(y, n_folds=6)
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
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
plt.show()