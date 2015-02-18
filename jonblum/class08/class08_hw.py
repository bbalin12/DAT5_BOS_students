'''
jonblum
2015-02-12
datbos05
class 8 hw
'''


# basics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# i/o
import sqlite3

# models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

# tools
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV
from scipy import interp


# grab same features as from HW5
conn = sqlite3.connect('/Users/jon/Documents/code/datbos05/data/lahman2013.sqlite')
sql = '''
SELECT hof.playerID, b.totalCareerHits, b.careerBattingAvg, p.avgCareerERA, f.careerFieldingPercentage, MAX(hof.inducted) AS inducted
FROM HallOfFame hof
LEFT JOIN
        (SELECT playerID, SUM(H) as totalCareerHits, (SUM(H)*1.0) / SUM(AB) as careerBattingAvg
        FROM Batting
        GROUP BY playerID) b
ON b.playerID = hof.playerID
LEFT JOIN
        (SELECT playerID, AVG(ERA) as avgCareerERA
        FROM Pitching
        GROUP BY playerID) p
ON  p.playerID = hof.playerID
LEFT JOIN
        (SELECT playerID, 1.0 * (SUM(PO) + SUM(A)) / (SUM(PO) + SUM(A) + SUM(E)) as careerFieldingPercentage
        FROM Fielding
        GROUP BY playerID) f
ON f.playerID = hof.playerID
WHERE hof.yearID < 2000 AND hof.category = 'Player'
GROUP BY hof.playerID;
'''

df = pd.read_sql(sql,conn)

conn.close()

# Y->1 N->0
df['inducted_boolean'] = 0
df.inducted_boolean[df.inducted == 'Y'] = 1
df.drop('inducted',1,inplace=True)

explanatory_features = [col for col in df.columns if col not in ['playerID', 'inducted_boolean']]
explanatory_df = df[explanatory_features]

# drop rows with no data at all (older players from other leauges)
explanatory_df.dropna(how='all', inplace = True)

# doing the same for response
response_series = df.inducted_boolean
response_series.dropna(how='all', inplace = True)

# Drop from response/explanatory any rows that were dropped from the other


response_series.drop(response_series.index[~response_series.index.isin(explanatory_df.index)],0,inplace=True)
explanatory_df.drop(explanatory_df.index[~explanatory_df.index.isin(response_series.index)],0,inplace=True)

len(response_series) == len(explanatory_df)
# True

# Impute NaNs
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit(explanatory_df)
explanatory_df = pd.DataFrame(imputer.transform(explanatory_df),columns=explanatory_df.columns, index=explanatory_df.index)


#################
# Create Models #
#################

# 1. Redo kNN Classifier from HW5

knn = KNeighborsClassifier(p = 2)
k_range = range(1,50,2)
knn_grid_params = dict(n_neighbors=k_range)
knn_grid = GridSearchCV(knn, knn_grid_params, cv=10, scoring='accuracy')
knn_grid.fit(explanatory_df,response_series)

print knn_grid.best_score_, knn_grid.best_params_
# 0.846 for k=29

knn_best_model = knn_grid.best_estimator_


# 2. Create Naive Bayes Classifier

nb = MultinomialNB()
alpha_range = [i/20.0 for i in range(0,21)]
nb_grid_params = dict(alpha=alpha_range)
nb_grid = GridSearchCV(nb,nb_grid_params, cv=10, scoring='accuracy')
nb_grid.fit(explanatory_df, response_series)

print nb_grid.best_score_, nb_grid.best_params_
# 0.766 for alpha=0 (no smoothing)

nb_best_model = nb_grid.best_estimator_

# 3. Create Decision Tree Classifier

dt = DecisionTreeClassifier()
depth_range = range(1,21)
min_split_range = range(2,11)
dt_grid_params = dict(max_depth=depth_range, min_samples_split=min_split_range)
dt_grid = GridSearchCV(dt,dt_grid_params, cv=10, scoring='accuracy')
dt_grid.fit(explanatory_df, response_series)

print dt_grid.best_score_, dt_grid.best_params_
# 0.868 for {'min_samples_split': 2, 'max_depth': 4}

dt_best_model = dt_grid.best_estimator_


###################
# Evaluate Models #
###################

# for kappa - 79% of our responses are 0, so our model has to beat naively setting all responses to 0
largest_class_percent_of_total = response_series.value_counts(normalize = True)[0]



# N-Fold Cross-Validated Accuracy & Cohen's Kappa

knn_cv_accuracy_scores = cross_val_score(knn_best_model, explanatory_df, response_series, cv=10, scoring='accuracy', n_jobs = -1)
knn_cv_accuracy_mean = knn_cv_accuracy_scores.mean()
print knn_cv_accuracy_mean
# 0.845861491817  (also saw this when creating the model with GridSearch scoring for accuracy)

knn_kappa = (knn_cv_accuracy_mean - largest_class_percent_of_total) / (1 - largest_class_percent_of_total)
print knn_kappa
# 0.24847062496500702  (not great, but at least it's positive)



nb_cv_accuracy_scores = cross_val_score(nb_best_model, explanatory_df, response_series, cv=10, scoring='accuracy', n_jobs = -1)
nb_cv_accuracy_mean = nb_cv_accuracy_scores.mean()
print nb_cv_accuracy_mean
# 0.76624891962 (also saw this when creating the model with GridSearch scoring for accuracy)

nb_kappa = (nb_cv_accuracy_mean - largest_class_percent_of_total) / (1 - largest_class_percent_of_total)
print nb_kappa
# -0.139694456773 (negative - worse than a model with all 0s)



dt_cv_accuracy_scores = cross_val_score(dt_best_model, explanatory_df, response_series, cv=10, scoring='accuracy', n_jobs = -1)
dt_cv_accuracy_mean = dt_cv_accuracy_scores.mean()
print dt_cv_accuracy_mean
# 0.868012374642 (also saw this when creating the model with GridSearch scoring for accuracy)

dt_kappa = (dt_cv_accuracy_mean - largest_class_percent_of_total) / (1 - largest_class_percent_of_total)
print dt_kappa
# 0.35647114555 (decent!)



# F1 Scores

knn_f1_scores = cross_val_score(knn_best_model, explanatory_df, response_series, cv=10, scoring='f1', n_jobs = -1)
print knn_f1_scores.mean()
# 0.495277622941


nb_f1_scores = cross_val_score(nb_best_model, explanatory_df, response_series, cv=10, scoring='f1', n_jobs = -1)
print nb_f1_scores.mean()
# 0.476190762977


dt_f1_scores = cross_val_score(dt_best_model, explanatory_df, response_series, cv=10, scoring='f1', n_jobs = -1)
print dt_f1_scores.mean()
# 0.640250233416



# ROC AUC

knn_roc_scores = cross_val_score(knn_best_model, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print knn_roc_scores.mean()
# 0.796909547767


nb_roc_scores = cross_val_score(nb_best_model, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print nb_roc_scores.mean()
# 0.718659912693


dt_roc_scores = cross_val_score(dt_best_model, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print dt_roc_scores.mean()
# 0.792233730518

####################################################
# CURVES AND MATRICES FOR ALL SLICES OF BEST MODEL #
####################################################


# Overall best model is decision tree

# ROC

X = explanatory_df.as_matrix() # as_matrix to turn into numpy arrays
y = response_series.as_matrix() # as_matrix to turn into numpy arrays

cv = StratifiedKFold(y, n_folds=10)
classifier = dt_best_model


mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])

    # create confusion matrix for the data
    yhats = classifier.fit(X[train], y[train]).predict(X[test])
    cm = pd.crosstab(y[test], yhats, rownames=['True Label'], colnames=['Predicted Label'], margins=True)
    print "Confusion Matrix for Fold ", i
    print cm,'\n\n'

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


# Confusion Matrices

'''
Confusion Matrix for Fold  0
Predicted Label   0   1  All
True Label
0                69   3   72
1                 8  11   19
All              77  14   91


Confusion Matrix for Fold  1
Predicted Label   0   1  All
True Label
0                69   3   72
1                 7  12   19
All              76  15   91


Confusion Matrix for Fold  2
Predicted Label   0   1  All
True Label
0                65   7   72
1                 7  12   19
All              72  19   91


Confusion Matrix for Fold  3
Predicted Label   0   1  All
True Label
0                71   1   72
1                 8  11   19
All              79  12   91


Confusion Matrix for Fold  4
Predicted Label   0   1  All
True Label
0                70   2   72
1                 9  10   19
All              79  12   91


Confusion Matrix for Fold  5
Predicted Label   0   1  All
True Label
0                68   4   72
1                12   6   18
All              80  10   90


Confusion Matrix for Fold  6
Predicted Label   0   1  All
True Label
0                65   7   72
1                 6  12   18
All              71  19   90


Confusion Matrix for Fold  7
Predicted Label   0   1  All
True Label
0                66   5   71
1                 9   9   18
All              75  14   89


Confusion Matrix for Fold  8
Predicted Label   0   1  All
True Label
0                66   5   71
1                 4  14   18
All              70  19   89


Confusion Matrix for Fold  9
Predicted Label   0   1  All
True Label
0                66   5   71
1                 7  11   18
All              73  16   89
'''




