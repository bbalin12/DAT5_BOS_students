# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 10:33:19 2015

@author: jeppley
"""


##########################
###  HOMEWORK: Class 8 ###
##########################


import pandas as pd
import sqlite3 as sql
import numpy as np
from sklearn.preprocessing import Imputer as imp
from sklearn.naive_bayes import MultinomialNB as nb
from sklearn.cross_validation import cross_val_score as cv
from sklearn.cross_validation import train_test_split as split
from sklearn import metrics
from sklearn import tree
from sklearn.grid_search import  GridSearchCV as grid
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt


##########################
###  Extracting Data   ###
##########################

con = sql.connect('C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite')

### Creating baseline table for data merging ###
cur = con.cursor()

table_creation_query = """
CREATE TABLE hall_of_fame_inductees_3 as 
select playerID, yearID, category, case when average_inducted = 0 then 0 else 1 end as inducted from (
select playerID, yearID, category, avg(case when inducted = 'Y' then 1 else 0 end ) as average_inducted from  HallOfFame hf
where yearid < 2000
group by playerID
) base;"""

cur.execute(table_creation_query)
cur.close()

### Merging data into baseline table ###

con = sql.connect('C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite')

data = '''select h.*, 
  b.b_atbat, b.b_hits, p.p_wins, f.f_puts, f.catcher, f.pitcher, f.dhitter
from 
  (select playerid, inducted
  from hall_of_fame_inductees_3 
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
  group by playerid 
  HAVING max(yearID) > 1950 and min(yearID) >1950 ) b
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
     max(case when pos = 'DH' then 1 else 0 end) as dhitter
  from fielding
  group by playerid) f
  on h.playerid = f.playerid
  where b.playerID is not null
;'''

new = pd.read_sql(data, con)
con.close()


##########################
###   Cleaning Data    ###
##########################


pd.set_option('display.max_columns', None)


new.describe()

### dropping duplicate playerID columns ###
new.drop('playerid', 1, inplace = True)

### splitting out the explanatory features ###
explanatory_features = [col for col in new.columns if col not in ['year', 'inducted']]
explanatory_features
explanatory_df = new[explanatory_features]

### dropping rows with no data ###
explanatory_df.dropna(how='all', inplace = True) 
response_series = new.inducted
response_series.dropna(how='all', inplace = True) 

response_series.index[~response_series.index.isin(explanatory_df.index)]

explanatory_df.describe()
response_series.describe()



### imputing missing cases ###

imputer_object = imp(missing_values='NaN', strategy='mean', axis=0)
# fitting the object on our data -- we do this so that we can save the 
# fit for our new data.
imputer_object.fit(explanatory_df)
explanatory_df = imputer_object.transform(explanatory_df)


##########################
### Naive Bayes Model  ###
##########################


### creating naive bayes classifier ###

naive_bayes_classifier = nb()

accuracy_scores = cv(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='accuracy')
print accuracy_scores.mean()
#looks like on average the model is 60% accurate, not very high

### calculating accuracy metrics for comparison ###

## ACCURACY METRIC 1: Cohen's Kappa ##

mean_accuracy_score = accuracy_scores.mean()
largest_class_percent_of_total = response_series.value_counts(normalize = True)[0]

largest_class_percent_of_total
#the largest class percent total is 90%, thus the model will correctly
#predict 90% of the time that someone WILL NOT be in the hall of fame

kappa = (mean_accuracy_score - largest_class_percent_of_total) / (1-largest_class_percent_of_total)
print kappa
#kappa is -2.92, not very good at all, information gain is less than if it were random

## ACCURACY METTRIC 2: F1

f1_scores = cv(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='f1')

print f1_scores.mean()
#Combined the accuracy score is 15%, not very good at all

roc_scores = cv(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='roc_auc')
print roc_scores.mean()
# the model fairs extremely poorly here, with .47, in other words a model failure

##########################
###   Decision Tree    ###
##########################

decision_tree2 = tree.DecisionTreeClassifier(random_state=1)

## pulling out a training and test slice from the data.
xTrain, xTest, yTrain, yTest = split(
                    explanatory_df, response_series, test_size =  0.3)

# Fit the decision tree classider
decision_tree2.fit(xTrain, yTrain)

## predict on the test data
predicted_values2 = decision_tree2.predict(xTest)

## computiner accuracy metrics for Decision Tree ##
accuracy_scores_cart = cv(decision_tree2, explanatory_df, response_series, cv=10, scoring='accuracy')

print accuracy_scores_cart.mean()
#The mode is 92% accurate

mean_accuracy_score_cart = accuracy_scores_cart.mean()
kappa_cart = (mean_accuracy_score_cart - largest_class_percent_of_total) / (1-largest_class_percent_of_total)
print kappa_cart
#kappa here is 0.20, which is much better than the naive bayes model

f1_scores_cart = cv(decision_tree2, explanatory_df, response_series, cv=10, scoring='f1')

#compare F1 of decision tree and naive bayes
print f1_scores_cart.mean()
# score for f1 is .63, much better than naive but still not amazing

roc_scores_cart = cv(decision_tree2, explanatory_df, response_series, cv=10, scoring='roc_auc')

# let's compare the decision tree with Naive Bayes.
print roc_scores_cart.mean()
# score of .82 here which is actually pretty good

##########################
###  Model Comparisons ###
##########################


## Accuracy Score Comparisons ##


##class 5 accuracy from knn was 87% at its best, looks like the decision tree does best here
print accuracy_scores.mean()
#looks like on average the model is 60% accurate
print accuracy_scores_cart.mean()
#The model is 92% accurate

## Kappa Comparisons ##
print kappa
#kappa is -2.92
print kappa_cart
#kappa here is 0.20


## F1 Comparisons ##

print f1_scores.mean()
#Combined the accuracy score is 15%
print f1_scores_cart.mean()
# score for f1 is .63

## ROC Comparisons ##

print roc_scores.mean()
# the model fairs extremely poorly here, with .47, in other words a model failure
print roc_scores_cart.mean()
# score of .82 here which is actually pretty good

### Conclusion: Based on comparisons of the accuracy levels and other information, 
### I can conclude that I will move forward with the decision tree model for further
### cross-validation and printing ROC for the various iterations

##########################
###  Multiple ROCs     ###
##########################

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(response_series, n_folds=6)
classifier = tree.DecisionTreeClassifier(random_state=1)

classifier
cv

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(explanatory_df[train], response_series[train]).predict_proba(explanatory_df[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(response_series[test], probas_[:, 1])
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

cm = pd.crosstab(yTest, predicted_values2, rownames=['True Label'], colnames=['Predicted Label'], margins=True)

print cm
