# -*- coding: utf-8 -*-
"""
Created on Tue Feb 03 21:08:39 2015

@author: mmcgoldr

Class 5 Homework

Build a preditive model that predicts whether a player was inducted to the 
Baseball Hall of Fame before 2000 using their batting, pitching, and fielding 
results - not the number of those they received. Please make sure to use K-fold 
cross validaiton and grid search to find your optimal mode.

"""

#import packages and functions

from __future__ import division
import numpy as np
import sqlite3 as sq
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.cross_validation import cross_val_score as cvs
from sklearn.grid_search import GridSearchCV as gscv

#get data from sqlite db into pandas dataframe and close connection
conn = sq.connect('C:\Users\mmcgoldr\Dropbox\GA\DataScience\SQLite\lahman2013.sqlite')

query = """
select h.*, 
  b.b_atbat, b.b_runs, b.b_hits, b.b_hruns, b.b_stbas, b.b_strik,
  p.p_years, p.p_wins, p.p_loss, p.p_shout, p.p_saves, p.p_eruns, p.p_stout, 
  f.f_years, f.f_puts, f.f_assis, f.f_dplay, f.f_pass, f.catcher, f.pitcher, f.dhitter
from 
  (select playerid, max(case when inducted = 'Y' then 1 else 0 end) as inducted, max(yearid) as year
   from halloffame 
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
  group by playerid) b
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
;"""

df = pd.read_sql(query, conn)

conn.close()

#create variable
df['p_ratio'] = (df.p_wins + df.p_saves)/(df.p_wins + df.p_saves + df.p_loss)

#examine data

df.shape #total players=1157

df.inducted.value_counts() # yes=241, no=916 = 20.8%
df.pitcher.value_counts() # yes=472
df.catcher.value_counts() # yes=143
df.dhitter.value_counts() # yes=301
df[(df.pitcher==0) & (df.catcher==0) & (df.dhitter==0)].index.value_counts().sum() #yes=300

df.groupby('pitcher').inducted.value_counts()  #90/472 = 21.1%
df.groupby('catcher').inducted.value_counts()  #21/143 = 14.7%
df.groupby('dhitter').inducted.value_counts()  #30/301 = 10.0%

df.columns
bcol = ['b_atbat','b_runs','b_hits','b_hruns','b_stbas','b_strik']
pcol = ['p_years','p_wins','p_loss','p_shout','p_saves','p_eruns','p_stout','p_ratio']
fcol = ['f_years','f_puts','f_assis','f_dplay','f_pass']
acol = ['b_atbat','b_runs','b_hits','b_hruns','b_stbas','b_strik',
       'p_years','p_wins','p_loss','p_shout','p_saves','p_eruns','p_stout','p_ratio',
       'f_years','f_puts','f_assis','f_dplay','f_pass']

df[bcol].head()
df[bcol].tail()
df[bcol].describe()

df[pcol].head()
df[pcol].tail()
df[pcol].describe()

df[fcol].head()
df[fcol].tail()
df[fcol].describe()

#identify missings

df.isnull().sum()
df[df.pitcher==1].isnull().sum()
df[df.catcher==1].isnull().sum()
df[df.dhitter==1].isnull().sum()
df[(df.pitcher==0) & (df.catcher==0) & (df.dhitter==0)].isnull().sum()
df[(df.pitcher!=1) & (df.pitcher!=0)].isnull().sum() #27 players have no stats

#drop 27 players where all B/P/F stats are missing
df.dropna(thresh=4, inplace=True) 

#fill missing stats with zeros
for a in acol:
    df[a].fillna(value=0, inplace=True)

#look at distributions of stats by induction status to identify potential model inputs
for p in pcol:
    df[df.pitcher==1].boxplot(column=p,by='inducted')  #try wins or ratio
    
for b in bcol:
    df[df.dhitter==1].boxplot(column=b,by='inducted')  #try home runs, runs, hits
    
for a in acol:
    df[df.catcher==1].boxplot(column=a,by='inducted')  #try puts, assists
    
for a in acol:
    df[(df.pitcher==0) & (df.catcher==0) & (df.dhitter==0)].boxplot(column=a,by='inducted')  #try hits, runs, puts
    
#scatter plot matrix of potential model inputs
pd.scatter_matrix(df[['b_hits','b_runs','b_hruns','p_wins','p_ratio','f_puts','f_assis']])

#split data into before year 2000 and on/after year 2000
df_train = df[df.year < 2000]
df_test = df[df.year >= 2000]

df_train.shape
df_test.shape

#select response variable in before 2000 set
response_train = df_train.inducted
response_test = df_test.inducted


#FIRST MODEL FIT: CATEGORICAL POSITION VARIABLES

#select explanatory variables
explanatory_train1 = df_train[['pitcher','catcher','dhitter']]
explanatory_test1 = df_test[['pitcher','catcher','dhitter']]

#run KNN
knn=knc(p = 2) #specify Euclidean distance
k_range = range(1,30, 2) #specify range of k to test
param_grid = dict(n_neighbors=k_range) #set up grid for results
grid=gscv(knn, param_grid, cv=10, scoring='accuracy') #instantiate model
grid.fit(explanatory_train1, response_train) #fit model

grid_mean_scores_1 = [result[1] for result in grid.grid_scores_]
best_score_1 = grid.best_score_
best_param_1 = grid.best_params_
knn_optimal_1 = grid.best_estimator_

knn_optimal_pred_1 = knn_optimal_1.predict(explanatory_test1)
accuracy_1 = len(response_test[response_test == knn_optimal_pred_1]) / len(response_test)


#SECOND MODEL FIT: CONTINUOUS STATS

#select explanatory variables
explanatory_train2 = df_train[['b_hruns','p_ratio','f_assis']]
explanatory_test2 = df_test[['b_hruns','p_ratio','f_assis']]

#run KNN
knn=knc(p = 2) #specify Euclidean distance
k_range = range(1,30, 2) #specify range of k to test
param_grid = dict(n_neighbors=k_range) #set up grid for results
grid=gscv(knn, param_grid, cv=10, scoring='accuracy') #instantiate model
grid.fit(explanatory_train2, response_train) #fit model

grid_mean_scores_2 = [result[1] for result in grid.grid_scores_]
best_score_2 = grid.best_score_
best_param_2 = grid.best_params_
knn_optimal_2 = grid.best_estimator_

#future predictions
knn_optimal_pred_2 = knn_optimal_2.predict(explanatory_test2)
accuracy_2 = len(response_test[response_test == knn_optimal_pred_2]) / len(response_test)


#THIRD MODEL FIT: CATEGORICAL + CONTINUOUS STATS

#select explanatory variables
explanatory_train3 = df_train[['pitcher','catcher','dhitter','b_hruns','p_ratio','f_assis']]
explanatory_test3 = df_test[['pitcher','catcher','dhitter','b_hruns','p_ratio','f_assis']]

#run KNN and 
knn=knc(p = 2) #specify Euclidean distance
k_range = range(1,30, 2) #specify range of k to test
param_grid = dict(n_neighbors=k_range) #set up grid for results
grid=gscv(knn, param_grid, cv=10, scoring='accuracy') #instantiate model
grid.fit(explanatory_train3, response_train) #fit model

#check results
grid_mean_scores_3 = [result[1] for result in grid.grid_scores_]
best_score_3 = grid.best_score_
best_param_3 = grid.best_params_
knn_optimal_3 = grid.best_estimator_

#future predictions
knn_optimal_pred_3 = knn_optimal_3.predict(explanatory_test3)
accuracy_3 = len(response_test[response_test == knn_optimal_pred_3]) / len(response_test)
print accuracy_3*100  #87.5%

#COMPARE
print best_score_1
print best_param_1
print accuracy_1*100

print best_score_2
print best_param_2
print accuracy_2*100

print best_score_3
print best_param_3
print accuracy_3*100


#CONCLUSION
# The third model with categorical + continuous predictors had the higest OOS
# accuracy (85.94%).  However, the first model with just 3 categorical predictors 
# based on position (pitcher, catcher, designated hitter) yielded the highest
# OOB accuracy (87.90%) and the lowest k (5).

