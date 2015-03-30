# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 19:13:02 2015
@author: mmcgoldr

GA DATA SCIENCE HOMEWORK

Class 9:
1. Join your SQL query used in last class' homework (to predict Baseball Hall 
   of Fame indution) with the table we created in today's class (called 
   dominant_team_per_player). 
2. Pick at least one additional categorical feature to include in your data.
3. Bin and encode your categorical features. 
4. Scale your data and impute for your numeric NaNs.
5. Remove features with perfect correlation and/or no variation.
6. Perform recursive feature elimination on the data.
7. Decide whether to use grid search to find your 'optimal' model.
8. Bring in data after the year 2000, and preform the same transformations on the
   data you did with your training data.
9. Predict Hall of Fame induction after the year 2000.

Class 10:
1. Run a Random Forest (RF), Boosting Trees (GBM), and Neural Network (NN) 
   classifier on the data you assembled in the homework from class 9.
2. See which of the methods you've used so far (RF, GBM, NN, Decision Tree, 
   Logistic Regression, Naive Bayes) is the most accurate (measured by ROC AUC).
3. Use grid seach to optimize your NN's tuning parameters for learning_rate, 
   iteration_range, and compoents, as well as any others you'd like to test.
   
Class 11:
1. Using the data from the last homework, perform principal component analysis 
   on your data. Decide how many components to keep.
2. Run a Boosting Tree on the components and see if in-sample accuracy beats a 
   classifier with the raw data not trasnformed by PCA. 
3. Run a support vector machine on your data. Tune your SVM to optimize accuracy
4. Bring in data after 2000, and preform the same transformations on the data 
   you did with your training data. Compare Random Forest, Boosting Tree, and 
   SVM accuracy on the data after 2000. 
5. Find some sort of attribute in the Baseball dataset that sits on a two-dimenstional 
   plane and has discrete clusters. Perform K-Means and DBSCAN clustering.
   Determine which better represents your data and the intiution behind why the
   model was the best for your dataset. Plot a dendrogram of your dataset.

SUMMARY OF RESULTS

Classification:

For this modeling, my dependent variable is Hall of Fame induction status
(yes, no).  My indpendent variables are two categorical features (dominant team and 
dominant position)  and several numeric features, including: at bats, runs, hits,
home runs, stolen bases, strikes from batting; wins, losses, shutouts, saves,
earned runs and stolen outs from pitching; and puts, assists, double plays and
passes from fielding.  My data are aggregated to the playerID level.  I use all
players before 2000 as my training set and all players on/after 2000 as my 
holdout for future predictions.  Further train/test sampling on the pre-2000 
set is conducted via cross validation to measure model performance before 
producing future predictions.

Before modeling the data, I perform the following transformations: remove players 
where all features are missing; replace categorical missings with 'None'; bin and 
binarize categorical features; impute missing numeric features with 0; check for 
features with no variance (none) and perfect correlation(also none); and finally, 
scale all features.

See attached file, mmcgoldr_hwc9-11_model_results.xlsx with a summary table of 
results for Decision Tree (DT), Naive Bayes (NB), Random Forest (RF), 
Gradient Boosting (GB), Neural Net (NN) and Support Vector Machine (SVM).  
All models are based on 10-fold cross validation, and in some cases, recursive 
feature search and/or grid search have been conducted to tune the models.  
The summary table shows out-of-sample ROC_AUC through cross-validation, 
as well as true positive rate (TPR) and true negative rate (TNR) for future 
predictions on/after the year 2000 (total=248, inducted=30, not inducted=218).

In short, SVM with linear kernel after tuning produces the best ROC_AUC (93.7%)
after tuning and cross-validation and the best true positive rate (60%) on
future predictions. A tuned GB model is a close second, with out-of-sample 
ROC_AUC 93.1% and true positive rate of 50% on future predictions.  Interestingly,
Neural Net has the highest true negative rate and overall accuracy on future 
predictions, but the model performs very poorly on true positive rate (0%) and 
out-of-sample ROC_AUC.

PCA:

According to my scree plot, all features can be reduced to a single factor.
However, running both GB and SVM models on the factored data
reduces out-of-sample ROC_AUC substantially, even with grid search for tuning.  
ROC_AUC drops from to for GB and from to for SVM.

Clustering: 

For this modeling, I select average home runs and average salary
by player school.  There are 399 schools with at least 1 player with non-missing 
home run and salary information.  A scatter plot of average HR and average 
salary shows that the data are approximately "V" shaped.  I examine results for
4-cluster and 5-cluster solutions using both Kmeans and Hierarchical clustering,
as well as a 2-cluster solution from DBScan.  Hierarchical clustering appears
to best represent the data.  This is not surprising given that HC's use of 
absolute distance is likely to work better with data that are unevenly dispersed.

"""


#IMPORT PACKAGES--------------------------------------------------------------

import pandas as pd
import sqlite3 as sq
import numpy as np
from sklearn import preprocessing as pp
from sklearn.preprocessing import Imputer as imp
from sklearn.cross_validation import cross_val_score as cv
from sklearn.grid_search import GridSearchCV as gscv
from sklearn.feature_selection import RFECV as rfe
from sklearn import tree as tr
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn import ensemble as ens
from sklearn.neural_network import BernoulliRBM as rbm
from sklearn import linear_model as lm
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, fclusterdata
import matplotlib.pylab as plt
from pylab import rcParams



#LOAD USER-DEFINED FUNCTIONS--------------------------------------------------

#convert low-freq categorical feature values to 'Other'
def cleanup_data(df, cutoffPercent = .01):
    for col in df:
        sizes = df[col].value_counts(normalize = True)
        values_to_delete = sizes[sizes<cutoffPercent].index
        df[col].ix[df[col].isin(values_to_delete)] = "Other"
    return df

#binazrize catergoical feature values into individual variables
def get_binary_values(data_frame):
    """encodes categorical features in Pandas."""
    all_columns = pd.DataFrame(index = data_frame.index)
    for col in data_frame.columns:
        data = pd.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pd.concat([all_columns, data], axis=1)
    return all_columns

#find and remove variables with zero variance
def find_zero_var(df):
    """finds columns in the dataframe with zero variance -- ie those
        with the same value in every observation.
    """
    toKeep = []
    toDelete = []
    for col in df:
      if len(df[col].value_counts()) > 1:
         toKeep.append(col)
      else:
        toDelete.append(col)      
    return {'toKeep':toKeep, 'toDelete':toDelete}
    
#find and remove variables with perfect correlation
def find_perfect_corr(df):
    """finds columns that are eother positively or negatively perfectly 
        correlated (with correlations of +1 or -1), and creates a dict that 
        includes which columns to drop so that each remaining column is independent
    """
    corrMatrix = df.corr()
    corrMatrix.loc[:,:] = np.tril(corrMatrix.values, k = -1)
    already_in = set()
    result = []
    for col in corrMatrix:
        perfect_corr = corrMatrix[col][abs(np.round(corrMatrix[col],10)) == 1.00].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    toRemove = []
    for item in result:
        toRemove.append(item[1:(len(item)+1)])
        toRemove = sum(toRemove, [])
    return {'corrGroupings':result, 'toRemove':toRemove}

# create new class with a .coef_ attribute (a little fix for scikit-learn)
class TreeClassifierWithCoef(tr.DecisionTreeClassifier):
    def fit(self, *args, **kwargs):
        super(tr.DecisionTreeClassifier, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_



#SET PLOT SIZE

rcParams['figure.figsize'] = 10,5



#GET DATA FROM SQL DB INTO PANDAS DATA FRAME-----------------------------------

#connect to SQLite DB
conn = sq.connect('C:\Users\mmcgoldr\Dropbox\GA\DataScience\SQLite\lahman2013.sqlite')

#get position data for each player
query1 = """select playerid, pos from fielding;"""
position = pd.read_sql(query1, conn)

#close connection
conn.close()

#identify most frequent position per player
position_frequent = position.groupby(['playerID']).agg(lambda x:x.value_counts().idxmax())

#write position table back to DB
conn = sq.connect('C:\Users\mmcgoldr\Dropbox\GA\DataScience\SQLite\lahman2013.sqlite')
position_frequent.to_sql('dominant_position_per_player', conn, if_exists = 'replace')

#reconnect to SQLite DB
conn = sq.connect('C:\Users\mmcgoldr\Dropbox\GA\DataScience\SQLite\lahman2013.sqlite')

#get position, dominant team and performance stats
query2 = """select h.*, 
  b.b_atbat, b.b_runs, b.b_hits, b.b_hruns, b.b_stbas, b.b_strik,
  p.p_wins, p.p_loss, p.p_shout, p.p_saves, p.p_eruns, p.p_stout, 
  f.f_puts, f.f_assis, f.f_dplay, f.f_pass, o.pos, t.teamid
from 
  (select playerid, max(case when inducted = 'Y' then 1 else 0 end) as inducted, max(yearid) as year
   from halloffame 
   where category = 'Player'
   group by playerid) h
left outer join 
  (select playerid,
    sum(ab) as b_atbat, 
    sum(r) as b_runs, 
    sum(h) as b_hits, 
    sum(hr) as b_hruns, 
    sum(sb) as b_stbas,
    sum(so) as b_strik
  from batting
  group by playerid) b
  on h.playerid = b.playerid
left outer join
  (select playerid,
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
     sum(po) as f_puts,
     sum(a) as f_assis,
     sum(dp) as f_dplay,
     sum(pb) as f_pass
  from fielding
  group by playerid) f
  on h.playerid = f.playerid
left outer join
  (select * from dominant_position_per_player) o
  on h.playerid = o.playerid
left outer join
  (select * from dominant_team_per_player) t
  on h.playerid = t.playerid
;"""

df = pd.read_sql(query2, conn)

#close connection
conn.close()

#check data
df.head()
df.tail()
df.shape

#split data before and on/after year 2000 (training vs future predictions)
pre2000 = df[df.year < 2000.00]
post2000 = df[df.year >= 2000.00]

#check split
pre2000.year.max()
post2000.year.min()
pre2000.shape #895 players
post2000.shape #262 players


#TRANSFORM DATA FOR TRAINING (< 2000)------------------------------------------

#drop records where all explanatory variables are missing (13 dropped)
pre2000.isnull().sum()
pre2000.dropna(thresh=4,inplace=True)
pre2000.isnull().sum()
pre2000.shape #882 players

#split dataset into id, explanatory and response features
pre2000.columns

exp = [col for col in pre2000.columns if col not in ['playerid','inducted','year']]

pre2000_id = pre2000[['playerid','year']]
pre2000_exp = pre2000[exp]
pre2000_res = pre2000.inducted

#split explanatory features into categorical vs numeric
pre2000_exp.dtypes

catfeat = pre2000_exp.ix[:, pre2000_exp.dtypes == 'object']
numfeat= pre2000_exp.ix[:, pre2000_exp.dtypes != 'object']

#fill cat NaNs with 'None'
catfeat = catfeat.fillna('None')

#bin and binarize cat vars
catfeat.POS.value_counts(normalize = True)
catfeat.teamID.value_counts(normalize = True)

cleanup_data(catfeat, cutoffPercent = .01)
catfeat_bin = get_binary_values(catfeat)

pos_list = set(catfeat.POS)
team_list = set(catfeat.teamID)

#check binned/binarized cat vars
catfeat_bin.head()
catfeat_bin.shape
catfeat_bin.isnull().sum()

# fill NANs in nuemeric features with zero
numfeat.fillna(0, inplace=True)

#check imputed numeric data
numfeat.head()
numfeat.shape
numfeat.isnull().sum()

# Merge categorical and numeric dataframes
pre2000_exp = pd.concat([numfeat, catfeat_bin],axis = 1)

#Check merged data
pre2000_exp.head()
pre2000_exp.shape
pre2000_exp.isnull().sum()

# find features with no variance (NONE / KEEP ALL)
find_zero_var(pre2000_exp)

#find features with perfect correlation (NONE / KEEP ALL)
find_perfect_corr(pre2000_exp)

#scale features
scaler = pp.StandardScaler()
scaler.fit(pre2000_exp)
pre2000_exp_scaled = pd.DataFrame(scaler.transform(pre2000_exp), 
                                  index = pre2000_exp.index, 
                                  columns = pre2000_exp.columns)

#retain unscaled features
pre2000_exp = pd.concat([numfeat, catfeat_bin],axis = 1)
find_zero_var(pre2000_exp)
find_perfect_corr(pre2000_exp)


#TRANSFORM DATA FOR FUTURE PREDICTIONS (>= 2000)-------------------------------

#drop records where all explanatory variables are missing (14 dropped)
post2000.dropna(thresh=4,inplace=True)
post2000.shape

#split dataset into id, explanatory and response features
post2000_id = post2000[['playerid','year']]
post2000_exp = post2000[exp]
post2000_res = post2000.inducted

#split explanatory features into categorical vs numeric
catfeat_post = post2000_exp.ix[:, post2000_exp.dtypes == 'object']
numfeat_post= post2000_exp.ix[:, post2000_exp.dtypes != 'object']

#fill categorical missings with 'None'
catfeat_post.fillna('None', inplace=True)

#replace low-frequency categorical values with 'Other' using
#same breaks as training data
catfeat_post.POS[~catfeat_post.POS.isin(pos_list)] = 'Other'
catfeat_post.teamID[~catfeat_post.teamID.isin(team_list)] = 'Other'

#binarize categorical feature values into individual variables
catfeat_post_bin = get_binary_values(catfeat_post)

#identify binarized variables in training set that are not in future
#prediction set and add these columns (all set to 0) in future set
missing_col = catfeat_bin.columns[~catfeat_bin.columns.isin(catfeat_post_bin.columns)]
for c in missing_col:
    catfeat_post_bin[c] = 0
catfeat_post_bin = catfeat_post_bin[catfeat_bin.columns]

#impute numerical missings with 0
numfeat_post.fillna(0, inplace=True)

#merge categorical and numeric features
post2000_exp = pd.concat([numfeat_post, catfeat_post_bin],axis = 1)

#make sure final columns and their order are identical to training set
post2000_exp = post2000_exp[pre2000_exp.columns]

#scale data
scaler = pp.StandardScaler()
scaler.fit(post2000_exp)
post2000_exp_scaled = pd.DataFrame(scaler.transform(post2000_exp), 
                                   index = post2000_exp.index, 
                                   columns = post2000_exp.columns)
                                   
#retain unscaled features
post2000_exp = pd.concat([numfeat_post, catfeat_post_bin],axis = 1)
post2000_exp = post2000_exp[pre2000_exp.columns]



#BUILD DECISION TREE MODEL (SCALED DATA)---------------------------------------

#instantiate decision tree model with coeff
dt = TreeClassifierWithCoef(criterion = 'gini', splitter = 'best', max_features = None, 
                              max_depth = None, min_samples_split = 2, min_samples_leaf = 2, 
                              max_leaf_nodes = None, random_state = 1)

#conduct recursive feature search
dt_rfe_cv = rfe(estimator=dt, step=1, cv=10, scoring='roc_auc', verbose = 1)
dt_rfe_cv.fit(pre2000_exp_scaled, pre2000_res)

#identify and plot optimal number of features (d = 17), ROC_AUC = 0.8262
print dt_rfe_cv.n_features_
print dt_rfe_cv.grid_scores_.max()

plt.figure()
plt.xlabel("DT: Number of Features selected")
plt.ylabel("DT: Cross Validation Score (ROC_AUC)")
plt.plot(range(1, len(dt_rfe_cv.grid_scores_) + 1), dt_rfe_cv.grid_scores_)
plt.show()

#identify selected features
dt_features = pre2000_exp_scaled.columns[dt_rfe_cv.get_support()]
print dt_features

#identify best estimator
dt_estimator = dt_rfe_cv.estimator_

#predict classes
dt_predictions = pd.Series(dt_estimator.predict(post2000_exp_scaled[dt_features]))

#cross predicted vs actual
post2000_res.index = dt_predictions.index
dt_crosstab = pd.crosstab(post2000_res, dt_predictions, rownames=['Actual'], 
                          colnames=['Predicted'], margins=True)
print dt_crosstab

#combine RFE with grid search to find optimal tuning parameter and features
depth_range = range(2, 10)
param_grid = dict(estimator__max_depth=depth_range)
dt_rfe_gs = gscv(dt_rfe_cv, param_grid, cv=10, scoring='roc_auc')
dt_rfe_gs.fit(pre2000_exp_scaled, pre2000_res)

#show and plot results (optimal max depth is 5)
print dt_rfe_gs.best_params_
print dt_rfe_gs.grid_scores_ 

dt_grid_mean_scores = [score[1] for score in dt_rfe_gs.grid_scores_]
plt.figure()
plt.plot(depth_range, dt_grid_mean_scores)
plt.hold(True)
plt.plot(dt_rfe_gs.best_params_['estimator__max_depth'], dt_rfe_gs.best_score_, 'ro', 
         markersize=12, markeredgewidth=1.5,markerfacecolor='None', markeredgecolor='r')
plt.grid(True)

#identify best estimator
dt_estimator_tuned = dt_rfe_gs.best_estimator_

#identify selected features (2 features: batting runs, pitching wins)
dt_features_tuned = pre2000_exp_scaled.columns[dt_estimator_tuned.get_support()]

#predict classes
dt_predictions_tuned = pd.Series(dt_estimator_tuned.predict(post2000_exp))

#cross predicted vs actual
post2000_res.index = dt_predictions_tuned.index
dt_crosstab_tuned = pd.crosstab(post2000_res, dt_predictions_tuned, rownames=['Actual'], 
                          colnames=['Predicted'], margins=True)
print dt_crosstab_tuned



#BUILD NAIVE BAYES MODEL (UNSCALED DATA)------------------------------------------------

#run model
nb = mnb()

#conduct recursive feature search
nb_rfe_cv = rfe(estimator=nb, step=1, cv=10, scoring='roc_auc', verbose = 1)
nb_rfe_cv.fit(pre2000_exp, pre2000_res)

#identify and plot optimal number of features (d = 50). ROC_AUC=0.6391
print nb_rfe_cv.n_features_
print nb_rfe_cv.grid_scores_.max()

plt.figure()
plt.xlabel("NB: Number of Features selected")
plt.ylabel("NB: Cross Validation Score (ROC_AUC)")
plt.plot(range(1, len(nb_rfe_cv.grid_scores_) + 1), nb_rfe_cv.grid_scores_)
plt.show()

#identify selected features
nb_features = pre2000_exp.columns[nb_rfe_cv.get_support()]
print nb_features

#identify best estimator
nb_estimator = nb_rfe_cv.estimator_

#predict classes
nb_predictions = pd.Series(nb_estimator.predict(post2000_exp[nb_features]))

#cross predicted vs actual
post2000_res.index = nb_predictions.index
nb_crosstab = pd.crosstab(post2000_res, nb_predictions, rownames=['Actual'], 
                          colnames=['Predicted'], margins=True)
print nb_crosstab



#BUILD RANDOM FOREST MODEL (SCALED DATA)------------------------------------------------

#instantiate Random Forest model
rf = ens.RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None, 
                                min_samples_split=2, min_samples_leaf=1, max_features='auto', 
                                max_leaf_nodes=None, bootstrap=True, oob_score=False, 
                                random_state=None, verbose=0, min_density=None, 
                                compute_importances=None)

#fit model and check scores (mean = .9215, max = .9597)
rf_cv = cv(rf, pre2000_exp_scaled, pre2000_res, cv=10, scoring='roc_auc')
print rf_cv .mean()
print rf_cv.max()

#perform grid search to find the optimal number of trees
rftree_range = range(10, 550, 10)
param_grid = dict(n_estimators = rftree_range)
rf_grid = gscv(rf, param_grid, cv=10, scoring='roc_auc')
rf_grid.fit(pre2000_exp_scaled, pre2000_res)

#check results from grid search
rf_grid_mean_scores = [result[1] for result in rf_grid.grid_scores_]
plt.figure()
plt.plot(rftree_range, rf_grid_mean_scores)

#identify best estimator
rf_estimator_tuned = rf_grid.best_estimator_

#identify optimal number of trees (130 trees) and best roc_auc score (0.9274)
print rf_estimator_tuned.n_estimators
print rf_grid.best_score_

#predict classes
rf_tuned_predictions = pd.Series(rf_estimator_tuned.predict(post2000_exp_scaled))

#cross predicted vs actual
post2000_res.index = rf_tuned_predictions.index
rf_tuned_crosstab = pd.crosstab(post2000_res, rf_tuned_predictions, rownames=['Actual'], 
                          colnames=['Predicted'], margins=True)
print rf_tuned_crosstab



#BUILD GRADIENT BOOSTING MODEL (SCALED DATA)--------------------------------------------

#instantiate Gradient Boosting model
gb = ens.GradientBoostingClassifier()

#fit model and check scores (mean = .9205, max = .9440)
gb_cv = cv(gb, pre2000_exp_scaled, pre2000_res, cv=10, scoring='roc_auc')
print gb_cv.mean()
print gb_cv.max()

#perform grid search to find optimal learning rate, sampling rate and number of estimators
learnrate_range = np.arange(0.01, 0.4, 0.02)
subsample_range = np.arange(0.25, 1, 0.25)
n_estimators_range = range(25, 100, 25)
param_grid = dict(learning_rate = learnrate_range, subsample = subsample_range, n_estimators = n_estimators_range)
gb_grid = gscv(gb, param_grid, cv=10, scoring='roc_auc')
gb_grid.fit(pre2000_exp_scaled, pre2000_res)

#identify best parameters (learning_rate=.21, subsample=0.75, n_estimators=75)
print gb_grid.best_params_

#identify best score (0.9306)
print gb_grid.best_score_

#identify best estimator
gb_estimator_tuned = gb_grid.best_estimator_

#predict classes
gb_tuned_predictions = pd.Series(gb_estimator_tuned.predict(post2000_exp_scaled))

#cross predicted vs actual
post2000_res.index = gb_tuned_predictions.index
gb_tuned_crosstab = pd.crosstab(post2000_res, gb_tuned_predictions, rownames=['Actual'], 
                          colnames=['Predicted'], margins=True)
print gb_tuned_crosstab



#BUILD NEURAL NET MODEL (SCALED DATA)---------------------------------------------------

#instantiate logistic regression and neural net components
lr = lm.LogisticRegression()
nn = rbm(random_state=0, verbose=True) 

#create pipeline of neural net connected to logistic regression
nn_pipe = Pipeline(steps=[('neural_net', nn), ('logistic_classifier', lr)])

#fit model and check scores (mean = .5570, max = .6827)
nn_cv = cv(nn_pipe, pre2000_exp_scaled, pre2000_res, cv=10, scoring='roc_auc')
print nn_cv.mean()
print nn_cv.max()

#perform grid search to find optimal components, learning rate and iterations
learnrate_range = np.arange(0.01, 0.2, 0.05)
iteration_range = range(30, 50, 5)
components_range = range(250, 500, 50)
param_grid = dict(neural_net__n_components = components_range, 
                  neural_net__learning_rate = learnrate_range, neural_net__n_iter = iteration_range)
nn_grid = gscv(nn_pipe, param_grid, cv=5, scoring='roc_auc')
nn_grid.fit(pre2000_exp_scaled, pre2000_res)

#identify best parameters (components = 250, learning_rate=.01, iterations=30)
print nn_grid.best_params_

#identify best score (0.6670)
print nn_grid.best_score_

#identify best estimator
nn_estimator_tuned = nn_grid.best_estimator_

#predict classes
nn_tuned_predictions = pd.Series(nn_estimator_tuned.predict(post2000_exp_scaled))

#cross predicted vs actual
post2000_res.index = nn_tuned_predictions.index
nn_tuned_crosstab = pd.crosstab(post2000_res, nn_tuned_predictions, rownames=['Actual'], 
                          colnames=['Predicted'], margins=True)
print nn_tuned_crosstab



#BUILD SUPPORT VECTOR MACHINE MODEL (sCALED DATA)--------------------------------------

#instantiate SVM with quadratic kernel, no PCA
svm = SVC(kernel='poly')

#run SVM (mean = 0.8770, max = 0.9226)
svm_cv = cv(svm, pre2000_exp_scaled, pre2000_res, cv=10, scoring='roc_auc')
print svm_cv.mean()
print svm_cv.max()

#perform grid to identify optimal kernel
param_grid = dict(kernel = ['linear','poly','rbf','sigmoid'])
svm_grid = gscv(svm, param_grid, cv=10, scoring='roc_auc')
svm_grid.fit(pre2000_exp_scaled, pre2000_res)

#identify best estimator 
svm_estimator_tuned = svm_grid.best_estimator_

#identify optimal kernal (linear)
print svm_estimator_tuned.kernel

#identify best score (0.9371)
print svm_grid.best_score_

#predict classes
svm_tuned_predictions = pd.Series(svm_estimator_tuned.predict(post2000_exp_scaled))

#cross predicted vs actual
post2000_res.index = svm_tuned_predictions.index
svm_tuned_crosstab = pd.crosstab(post2000_res, svm_tuned_predictions, rownames=['Actual'], 
                          colnames=['Predicted'], margins=True)
print svm_tuned_crosstab



#RUN PCA---------------------------------------------------------------------------

#run PCA
pca = PCA(n_components=10)
pca.fit(pre2000_exp_scaled)

#extract components
pca_df = pd.DataFrame(pca.transform(pre2000_exp_scaled))

#make a scree plot
pca_var = pd.DataFrame({'variance': pca.explained_variance_, 'principal component': pca_df.columns.tolist()})
pca_var['principal component'] = pca_var['principal component'] + 1
pca_var.plot(x = 'principal component', y= 'variance')

pca_df1 = pca_df.ix[:,0:0]



#BUILD GRADIENT BOOSTING AND SV MODELS WITH PCA----------------------------------------

#instantiate Gradient Boosting model
gb = ens.GradientBoostingClassifier()

#perform grid search with 1 factor (best score: 0.8459)
learnrate_range = np.arange(0.01, 0.4, 0.02)
subsample_range = np.arange(0.25, 1, 0.25)
n_estimators_range = range(25, 100, 25)
param_grid = dict(learning_rate = learnrate_range, subsample = subsample_range, n_estimators = n_estimators_range)
gb_pca1_grid = gscv(gb, param_grid, cv=10, scoring='roc_auc')
gb_pca1_grid.fit(pca_df1, pre2000_res) 
print gb_pca1_grid.best_score_
print gb_pca1_grid.best_params_


#instantiate SVM
svm = SVC()

#perform grid search with with 1 factor (best score: )
param_grid = dict(kernel = ['linear','poly','rbf','sigmoid'])
svm_pca1_grid = gscv(svm, param_grid, cv=10, scoring='roc_auc')
svm_pca1_grid.fit(pca_df1, pre2000_res)
print svm_pca1_grid.best_score_
print svm_pca1_grid.best_estimator_.kernel



#BUILD SEGMENTATION MODELS FOR 2-D FEATURES------------------------------------

#connect to SQLite DB
conn = sq.connect('C:\Users\mmcgoldr\Dropbox\GA\DataScience\SQLite\lahman2013.sqlite')

#get average home runs and average salary by player school
query = """select a.schoolID as schoolID, avg(b.total_hruns) as average_hruns, 
avg(c.total_salary) as average_salary
from
  (select e.playerID, e.schoolID
   from (select playerID, max(yearMax) as yearMax from SchoolsPlayers group by playerID) d
   left outer join SchoolsPlayers e
   on d.playerID = e.playerID and d.yearMax = e.yearMax) a
inner join
  (select playerID, sum(hr) as total_hruns from batting where hr is NOT NULL group by playerID) b
  on a.playerID = b.playerID
inner join 
  (select playerID, sum(salary) as total_salary from salaries where salary is NOT NULL group by playerID) c
  on a.playerID = c.playerID
group by a.schoolID;"""

df = pd.read_sql(query, conn)

#close connection
conn.close()

#check data
df.shape
df.isnull().sum()
df.columns

#plot average salary vs average home runs
rcParams['figure.figsize'] = 10,5
plt = df.plot(x='average_hruns', y='average_salary', s=60, kind='scatter')
for i, txt in enumerate(df.schoolID):
    plt.annotate(txt, (df.average_hruns[i],df.average_salary[i]))
    
#scale numeric features
features = df[['average_hruns','average_salary']]
scaler = pp.StandardScaler()
scaler.fit(features)
features = pd.DataFrame(scaler.transform(features), columns = features.columns)

#run and plot KMeans with 4 clusters
km4 = KMeans(n_clusters=4)
km4.fit(features)
km4_labels = km4.labels_
plt.scatter(df.average_hruns, df.average_salary, s=60, c=km4_labels)

#run and plot KMeans with 5 clusters
km5 = KMeans(n_clusters=5)
km5.fit(features)
km5_labels = km5.labels_
plt.scatter(df.average_hruns, df.average_salary, s=60, c=km5_labels)

#calculate distance matrix for hierarchical clustering
distanceMatrix = pdist(features)

#run and plot Hierarchical clustering with 4 clusters
hc4 = fcluster(linkage(distanceMatrix, method='complete'),6,'distance')
hc4_output = pd.DataFrame({'school':df.schoolID.tolist() , 'cluster':hc4})
plt.scatter(df.average_hruns, df.average_salary, s=60, c=hc4_output.cluster)

#run and plot Hierarchical clustering with 5 clusters
hc5 = fcluster(linkage(distanceMatrix, method='complete'),5,'distance')
hc5_output = pd.DataFrame({'school':df.schoolID.tolist() , 'cluster':hc5})
plt.scatter(df.average_hruns, df.average_salary, s=60, c=hc5_output.cluster)

#plot dendrograms for 4 and 5
rcParams['figure.figsize'] = 70,8

dend4 = dendrogram(linkage(distanceMatrix, method='complete'), 
           color_threshold=6, 
           leaf_font_size=8,
           labels = df.schoolID.tolist())
           
dend5 = dendrogram(linkage(distanceMatrix, method='complete'), 
           color_threshold=5, 
           leaf_font_size=8,
           labels = df.schoolID.tolist())

#run and plot DBSCAN
rcParams['figure.figsize'] = 10,5
dbsc = DBSCAN().fit(np.array(features))
db_labels = dbsc.labels_
plt.scatter(df.average_hruns, df.average_salary, s=60, c=db_labels)