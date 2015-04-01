# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 15:03:22 2015

@author: MatthewCohen
"""

import pandas
import sqlite3
conn = sqlite3.connect('/Users/MatthewCohen/Documents/SQLite/lahman2013.sqlite')
cur = conn.cursor()

table_creation_query = """
CREATE TABLE hall_of_fame_inductees3 as  
select playerID, case when average_inducted = 0 then 0 else 1 end as inducted from (
select playerID, avg(case when inducted = 'Y' then 1 else 0 end ) as average_inducted from  HallOfFame hf
where yearid < 2000
group by playerID
) bb;"""

cur.execute(table_creation_query)
cur.close()

monster_query = """
select m.nameGiven, hfi.inducted, batting.*, pitching.*, fielding.* from hall_of_fame_inductees3 hfi 
left outer join master m on hfi.playerID = m.playerID
left outer join 
(
select playerID,  sum(AB) as total_at_bats, sum(H) as total_hits, sum(R) as total_runs, sum(HR) as total_home_runs, sum(SB) as total_stolen_bases,
sum(RBI) as total_RBI, sum(IBB) as total_intentional_walks
from Batting
group by playerID
HAVING max(yearID) > 1950 and min(yearID) >1950 
)
batting on batting.playerID = hfi.playerID
left outer join
(
 select playerID, sum(G) as total_games_pitched, sum(SO) as total_shutouts, sum(sv) as total_saves, sum(IPouts) as total_outs_pitched, sum(er) as total_pitching_earned_runs, sum(so) as total_strikeouts, 
avg(ERA) as average_ERA,
sum(R) as total_runs_allowed
from Pitching
group by playerID
) 
pitching on pitching.playerID = hfi.playerID 
LEFT OUTER JOIN
(
select playerID, sum(G) as total_games_fielded, sum(E) as total_errors, sum(DP) as total_double_plays
from Fielding
group by playerID
) 
fielding on fielding.playerID = hfi.playerID
where batting.playerID is not null
"""

df = pandas.read_sql(monster_query, conn)
conn.close()

pandas.set_option('display.max_columns', None)
df.head(10)
df.columns

df.drop('playerID',  1, inplace = True)

df.describe()

explanatory_features = [col for col in df.columns if col not in ['nameGiven', 'inducted']]
explanatory_df = df[explanatory_features]
explanatory_df.dropna(how='all', inplace = True)
explanatory_colnames = explanatory_df.columns
response_series = df.inducted
response_series.dropna(how='all', inplace = True)
response_series.index[~response_series.index.isin(explanatory_df.index)]

from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_object.fit(explanatory_df)
explanatory_df = imputer_object.transform(explanatory_df)

from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score

naive_bayes_classifier = MultinomialNB()

accuracy_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='accuracy', n_jobs = -1)
print accuracy_scores.mean()


import numpy as np
from sklearn import svm, datasets
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################

X = explanatory_df
y = response_series

X, y = X[y != 2], y[y != 2]
n_samples, n_features = X.shape

# Add noisy features
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

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


cv = cross_val_score(y, n_folds=6)




####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################



#################### Don't need this once above works ###############################################

from sklearn.cross_validation import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(
                    explanatory_df, response_series, test_size =  0.3)
y_predicted = naive_bayes_classifier.fit(xTrain, yTrain).predict(xTest)
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

########################### (SEE ABOVE) #############################################################

from sklearn import tree
decision_tree = tree.DecisionTreeClassifier(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)
decision_tree.fit(xTrain, yTrain)
predicted_values = decision_tree.predict(xTest)
cm = pandas.crosstab(yTest, predicted_values, rownames=['True Label'], colnames=['Predicted Label'], margins=True)
print cm

importances_df = pandas.DataFrame(explanatory_colnames)
importances_df['importances'] = decision_tree.feature_importances_

predicted_probs = pandas.DataFrame(decision_tree.predict_proba(xTest))

fpr_cart, tpr_cart, thresholds_cart = metrics.roc_curve(yTest, predicted_probs[1])
plt.figure()
plt.plot(fpr, tpr, color = 'g')
plt.plot(fpr_cart, tpr_cart, color = 'b')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')


accuracy_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='accuracy', n_jobs = -1)
print accuracy_scores_cart.mean()
print accuracy_scores.mean()

# KAPPA
mean_accuracy_score = accuracy_scores.mean()
largest_class_percent_of_total = response_series.value_counts(normalize = True)[0]
kappa = (mean_accuracy_score - largest_class_percent_of_total) / (1-largest_class_percent_of_total)

mean_accuracy_score_cart = accuracy_scores_cart.mean()
kappa_cart = (mean_accuracy_score_cart - largest_class_percent_of_total) / (1-largest_class_percent_of_total)

print kappa
print kappa_cart

# F1 SCORE
f1_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='f1', n_jobs = -1)

f1_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='f1', n_jobs = -1)

print f1_scores.mean()
print f1_scores_cart.mean()

# AUC
roc_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

roc_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

print roc_scores.mean()
print roc_scores_cart.mean()


