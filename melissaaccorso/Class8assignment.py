# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:27:06 2015

@author: melaccor
"""

from __future__ import division
import pandas
import sqlite3
conn=sqlite3.connect('C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite')
# open a cursor as we are executing a SQL statement that does not produce a pandas DataFrame
cur = conn.cursor()
# writing the query to simplify creating our response feature. 
sql = """
select a.playerID, a.inducted as inducted, batting.*, pitching.*, fielding.* from
(select playerID, case when avginducted = 0 then 0 else 1 end as inducted from 
(select playerID, avg(case when inducted = 'Y' then 1 else 0 end ) as avginducted from HallOfFame 
where yearid < 2000
group by playerID)) a 
left outer join
(select playerID,  sum(AB) as atbats, sum(H) as totalhits, sum(R) as totalruns, sum(HR) as totalhomeruns, sum(SB) as stolenbases, sum(RBI) as totalRBI, sum(SO) as strikeouts, sum(IBB) as intentionalwalks
from Batting
group by playerID) batting on batting.playerID = a.playerID
left outer join(select playerID, sum(G) as totalgames, sum(SO) as shutouts, sum(sv) as totalsaves, sum(H) as totalhits, sum(er) as earnedruns, sum(so) as strikeouts, sum(WP) as wildpitches, sum(R) as totalruns
from Pitching
group by playerID) pitching on pitching.playerID = a.playerID 
left outer join
(select playerID, sum(G) as games, sum(InnOuts) as timewithouts, sum(PO) as putouts, sum(E) as errors, sum(DP) as doubleplays
from Fielding
group by playerID) fielding on fielding.playerID = a.playerID;"""
df = pandas.read_sql(sql, conn)
# closing the connection.
conn.close()

df.dropna(inplace = True)

print df.head()

# dropping duplicate playerID columns
df.drop('playerID', 1, inplace = True)

#rename playerID column
df.rename(columns={'a.playerID': 'playerID'}, inplace=True)
df.head()

#get columns
colnames = list(df.columns.values)
print colnames
colnames.remove('playerID')
colnames.remove('inducted')
print colnames

# getting summary statistics on the data
df.describe()

#splitting out the explanatory features 
explanatory_features = colnames
explanatory_df = df[explanatory_features]

#dropping rows with no data.
explanatory_df.dropna(how='all', inplace = True)

#extracting column names
explanatory_colnames = explanatory_df.columns

#doing the same for response
response_series = df.inducted
response_series.dropna(how='all', inplace = True) 

#seeing which explanatory feature rows got removed.  Looks like none.
response_series.index[~response_series.index.isin(explanatory_df.index)]
# if there were any, we need to make sure that we only keep indices 
# that are the union of the explanatory and response features post-dropping.

# imputing NaNs with the mean value for that column.  We will 
# go over this in further detail in next week's class.
from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='mean', axis=0)
# fitting the object on our data -- we do this so that we can save the 
# fit for our new data.
imputer_object.fit(explanatory_df)
explanatory_df = imputer_object.transform(explanatory_df)

# create a naive Bayes classifier and get it cross-validated accuracy score. 
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score

# creating the naive bayes classifier object 
naive_bayes_classifier = MultinomialNB()

# running a cross-validates score on accuracy.  Notice I set 
# n_jobs to -1, which means I'm going to use all my computer's 
# cores to find the result.
accuracy_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='accuracy')

# let's see how accurate the model is, on average.
print accuracy_scores.mean()
#Accuracy has a mean of 0.39

# let's calculate Cohen's Kappa
mean_accuracy_score = accuracy_scores.mean()
largest_class_percent_of_total = response_series.value_counts(normalize = True)[0]
# largest_class_percent_total is around 86%.  
# So a completely naive model that predicts noone
# is inducted into the Hall of Fame is 86% correct on average.
kappa = (mean_accuracy_score - largest_class_percent_of_total) / (1-largest_class_percent_of_total)
print kappa
## kappa is highly negative (-3.27).  So, if we weigh a positive prediction to be as important as a negative one, our model isn't being that great at predicting at all. 

# calculating F1 score, which is the harmonic mean of specificity
# and sensitivity. 
f1_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='f1')

print f1_scores.mean()
# so combined two-class acccuracy doesn't look too good. F1 scores is 0.29

#calculating the ROC area under the curve score. 
roc_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='roc_auc')
print roc_scores.mean()
#ROC mean area around curve is 0.58--fail

## here's the interpretability of AUC
#.90-1 = excellent 
#.80-.90 = good 
#.70-.80 = fair 
#.60-.70 = poor
#.50-.60 = fail
# so, on AUC terms, this is a fail.


# now, let's create a confusion matrix and plot an ROC curve.
# ideally, we want these to incorporate fully cross-validated
# information, but for the sake of time we're only going to 
# look at one slice.  See http://scikit-learn.org/stable/auto_examples/plot_roc_crossval.html#example-plot-roc-crossval-py for more information on how to really do it. 

## pulling out a training and test slice from the data.
from sklearn.cross_validation import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(explanatory_df, response_series, test_size =  0.3)

# get predictions on the test slice of the classifier. 
y_predicted = naive_bayes_classifier.fit(xTrain, yTrain).predict(xTest)

# create confusion matrix for the data
cm = pandas.crosstab(yTest, y_predicted, rownames=['True Label'], colnames=['Predicted Label'], margins=True)

print cm
# what do the accuracy of predicting not inducted vs inducted look like (recall)
# what does the accuracy of each prediction look like (precision)

#Predicted Label  0   1  All
#True Label                 
#0                5  41   46
#1                1  10   11
#All              6  51   57

#Model is better at predicting true positives than true negatives


## let's plot ROC curve ##

## extracting probabilties for the clasifier
y_probabilities = pandas.DataFrame(naive_bayes_classifier.fit(xTrain, yTrain).predict_proba(xTest))

from sklearn import metrics
# remember to pass the ROC curve method the probability of a 'True' 
# class, or column 1 in this case.
fpr, tpr, thresholds = metrics.roc_curve(yTest, y_probabilities[1])
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
## looking at the ROC curve, the f1 score,
# ROC score, accuracy, and Kappa how helpful is this estimator? 
#Model is not a very good estimator: low accuracy of 0.39, kappa highly negative at -3.27, f1 scores 0.29, ROC curve of 0.58



## CART ###


# now, let's create some classification trees. 
from sklearn import tree
# Create a decision tree classifier instance
decision_tree = tree.DecisionTreeClassifier(random_state=1)

# realize that the above code is the exact same as the code below,
# which shows the object's default values.  We can change these values
# to tune the tree.
decision_tree = tree.DecisionTreeClassifier(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)


# Fit the decision tree classider
decision_tree.fit(xTrain, yTrain)

## predict on the test data, look at confusion matrix and the ROC curve
predicted_values = decision_tree.predict(xTest)

cm = pandas.crosstab(yTest, predicted_values, rownames=['True Label'], colnames=['Predicted Label'], margins=True)

print cm
# looks a little better, right?
#Predicted Label   0   1  All
#True Label                  
#0                45   1   46
#1                 7   4   11
#All              52   5   57

#Better at predicting true negatives than previous model

# getting variable importances
importances_df = pandas.DataFrame(explanatory_colnames)
importances_df['importances'] = decision_tree.feature_importances_

# extracting decision tree probabilities
predicted_probs = pandas.DataFrame(decision_tree.predict_proba(xTest))

# now, let's plot the ROC curve and compare it to Naive Bayes (which will be in green)
fpr_cart, tpr_cart, thresholds_cart = metrics.roc_curve(yTest, predicted_probs[1])
plt.figure()
plt.plot(fpr, tpr, color = 'g')
plt.plot(fpr_cart, tpr_cart, color = 'b')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
#Compared to the ROC model, we see that the decision tree model is much better

## now let's do 10-fold CV, compute accuracy, f1, AUC, and Kappa
accuracy_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='accuracy')

print accuracy_scores_cart.mean()
# mean accuracy of 86%. let's compare this to the naive Bayes mean accuracy.
print accuracy_scores.mean()
#naive bayes has accuracy of 0.39
#so, which model is more accurate? 
#the decision tree model is much more accurate

# let's calculate Cohen's Kappa
mean_accuracy_score_cart = accuracy_scores_cart.mean()
# recall we already calculated the largest_class_percent_of_total above.
kappa_cart = (mean_accuracy_score_cart - largest_class_percent_of_total) / (1-largest_class_percent_of_total)
print kappa_cart
# so Kappa of 0.0004.  What does this say in absolute terms of the ability of the model to predict better than just random selection?

# let's compare to Naive Bayes. 
print kappa
#Naive Bayes Kappa is -3.27
#which is better?
#The kappa for the decision tree model is much more reasonable

# calculating F1 score, which is the weighted average of specificity and sensitivity. 
f1_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='f1')

#compare F1 of decision tree and naive bayes
print f1_scores_cart.mean()
#F1 Scores are 0.41 for the decision tree
print f1_scores.mean()
#f1 scores are 0.29

# calculating the ROC area under the curve score. 
roc_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='roc_auc')

# let's compare the decision tree with Naive Bayes.
print roc_scores_cart.mean()
#ROC Score for the decision tree is 0.70
print roc_scores.mean()
#ROC Score for naive bayes is 0.58

# now, let's fine-tune the tree model.
from sklearn.grid_search import  GridSearchCV

# Conduct a grid search for the best tree depth
decision_tree = tree.DecisionTreeClassifier(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)

depth_range = range(1, 20)
param_grid = dict(max_depth=depth_range)

grid = GridSearchCV(decision_tree, param_grid, cv=10, scoring='roc_auc')
grid.fit(explanatory_df, response_series)

# Check out the scores of the grid search
grid_mean_scores = [result[1] for result in grid.grid_scores_]

# Plot the results of the grid search
plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.plot(grid.best_params_['max_depth'], grid.best_score_, 'ro', markersize=12, markeredgewidth=1.5, markerfacecolor='None', markeredgecolor='r')
plt.grid(True)

# now let's calculate other accuracy metrics for the best estimator.
best_decision_tree_est = grid.best_estimator_

## lets compare on best accuracy
accuracy_scores_best_cart = cross_val_score(best_decision_tree_est, explanatory_df, response_series, cv=10, scoring='accuracy')

print accuracy_scores_best_cart.mean()
#Best accuracy score for decision trees 0.87
print accuracy_scores_cart.mean()
#Accuracy score for first decision tree 0.86
#accuracy score for best decision tree is slightly larger by 0.01

# calculating F1 score, which is the weighted average of specificity and sensitivity. 
f1_scores_best_cart = cross_val_score(best_decision_tree_est, explanatory_df, response_series, cv=10, scoring='f1')

#compare F1 scores
print f1_scores_best_cart.mean()
#Best decision tree f1 scores is 0.56
print f1_scores_cart.mean()
#F1 score for first decision tree is 0.41

## f1 score for best decision tree is slightly higher

## calculating the ROC area under the curve score. 
roc_scores_best_cart = cross_val_score(best_decision_tree_est, explanatory_df, response_series, cv=10, scoring='roc_auc')

print roc_scores_best_cart.mean()
#ROC Scores for the best decision tree is 0.76
print roc_scores_cart.mean()
#ROC Scores for generic decision tree is 0.70

# Now let's plot the ROC curve of the  best grid estimator vs our older decision tree classifier.
predicted_probs_cart_best = pandas.DataFrame(best_decision_tree_est.predict_proba(xTest))

fpr_cart_best, tpr_cart_best, thresholds_cart_best = metrics.roc_curve(yTest, predicted_probs[1])
plt.figure()
plt.plot(fpr, tpr, color = 'g')
plt.plot(fpr_cart, tpr_cart, color = 'b')
plt.plot(fpr_cart_best, tpr_cart_best, color = 'r')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)') 

## KNN Model ##
from sklearn.neighbors import KNeighborsClassifier
import numpy

#Now use 10-fold cross validation to score model
from sklearn.cross_validation import cross_val_score

KNN_Classifier = KNeighborsClassifier(n_neighbors=3,p=2)

knn_scores = cross_val_score(KNN_Classifier,explanatory_df, response_series, cv=10,scoring='accuracy')

#get avg score
knn_accuracy = numpy.mean(knn_scores)
print knn_accuracy

#now tune model for optimal K
k_range = range(1,30,2)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k,p=2)
    scores.append(numpy.mean(cross_val_score(knn,explanatory_df,
                response_series,cv=10,scoring = 'accuracy')))

import matplotlib.pyplot as plt
plt.figure()
plt.plot(k_range,scores)

#good thing not necc have to write forloop there's a function
from sklearn.grid_search import GridSearchCV
knn = KNeighborsClassifier(p=2)
k_range = range(1,30,2)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn,param_grid,cv=5,scoring='accuracy')
grid.fit(explanatory_df,response_series)

grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(k_range,grid_mean_scores)

best_oob_score = grid.best_score_
grid.best_params_
Knn_optimal = grid.best_estimator_
print Knn_optimal

#KNN Optimal is
#KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_neighbors=7, p=2, weights='uniform')

#Thus use as default model to calculate scores
knn_accuracy = cross_val_score(Knn_optimal,explanatory_df, response_series, cv=10,scoring='accuracy')
print knn_accuracy.mean()
knn_f1_scores = cross_val_score(Knn_optimal, explanatory_df, response_series, cv=10, scoring='f1')
print knn_f1_scores.mean()
knn_roc_scores= cross_val_score(Knn_optimal, explanatory_df, response_series, cv=10, scoring='roc_auc')
print knn_roc_scores.mean()

#Comparisons
print 'Accuracy'
print 'NB '+str(accuracy_scores.mean())
print 'DT '+str(accuracy_scores_cart.mean())
print 'KNN '+str(knn_accuracy.mean())
print '\n'
print 'F1 Scores'
print 'NB '+str(f1_scores.mean())
print 'DT '+str(f1_scores_best_cart.mean())
print 'KNN '+str(knn_f1_scores.mean())
print '\n'
print 'ROC Scores'
print 'NB '+str(roc_scores.mean())
print 'DT '+str(roc_scores_best_cart.mean())
print 'KNN '+str(knn_roc_scores.mean())

#KNN is the best model so let's calculate the stratified ROC curve for that

# Stratified K Fold ROC
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold

cv = StratifiedKFold(response_series, n_folds=10,indices=False)
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
for i, (train, test) in enumerate(cv):
    probas_ = Knn_optimal.fit(explanatory_df[train], response_series[train]).predict_proba(explanatory_df[test])
    # Confusion Matrix
    predicted_values = Knn_optimal.predict(explanatory_df[test])
    cm = pandas.crosstab(response_series[test],predicted_values, rownames = ['True Label'], colnames = ['Predicted Label'], margins = True)
    print "K Nearest Neighbor Confusion Matrix: %d" % (i+1)
    print cm
    print '\n'
    # Compute ROC curve and area under the curve
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