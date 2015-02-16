# -*- coding: utf-8 -*-
"""
Created on Fri Feb 06 13:48:01 2015

@author: jeppley
"""


#SETTING UP THE DATA EXERCISE 

# importing division from the 'future' release of Python (i.e. Python 3)
from __future__ import division

# importing numpy and the KNN content in scikit-learn 
# along with SQLite and pandas.
import numpy
import sqlite3
import pandas
from sklearn.neighbors import KNeighborsClassifier

# designating model constants at the top of the file per PEP8 
# see https://www.python.org/dev/peps/pep-0008/
# this is the percent we want to hold out for our cross-validation.
CROSS_VALIDATION_AMOUNT = .2


#WRANGLING DATA NEEDED FOR MODELING: LOOKING AT BATTING PREDICTIONS FIRST

#Work done in SQL before bringing in for python amendments


create table pitchingsub as
select playerID, yearID, ERA, HR, W, G from pitching
group by playerID, yearID

create table hallsub2 as
select playerID, yearID, inducted from halloffame
where yearID < 2000
group by playerID, yearID

create table pitchingmod as
select a.*, b.ERA, b.HR, b.W, b.G from hallsub2 a
left outer join
pitchingsub b on a.playerID=b.playerID and a.yearID=b.yearID




# connect to the baseball database. Notice I am passing the full path
# to the SQLite file.
conn = sqlite3.connect("C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite")
# creating an object contraining a string that has the SQL query. 
sql = '''select h.*, 
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
;'''
# passing the connection and the SQL string to pandas.read_sql.
sql = pandas.read_sql(sql, conn)
# closing the connection.
conn.close()


#checking what dataset looks like

sql.shape #total players=1157
sql.inducted.value_counts()
sql.pitcher.value_counts()
sql.columns


#looking at descriptive statistics

bcol = ['b_atbat','b_runs','b_hits','b_hruns','b_stbas','b_strik']
pcol = ['p_years','p_wins','p_loss','p_shout','p_saves','p_eruns','p_stout']
fcol = ['f_years','f_puts','f_assis','f_dplay','f_pass']
acol = ['b_atbat','b_runs','b_hits','b_hruns','b_stbas','b_strik',
       'p_years','p_wins','p_loss','p_shout','p_saves','p_eruns','p_stout', 'f_years','f_puts','f_assis','f_dplay','f_pass']
       
       
       
sql[bcol].head()
sql[bcol].tail()
sql[bcol].describe()

sql[pcol].head()
sql[pcol].tail()
sql[pcol].describe()

sql[fcol].head()
sql[fcol].tail()
sql[fcol].describe()



#identify missings

sql.isnull().sum()
sql[sql.pitcher==1].isnull().sum()
sql[sql.catcher==1].isnull().sum()
sql[sql.dhitter==1].isnull().sum()
sql[(sql.pitcher==0) & (sql.catcher==0) & (sql.dhitter==0)].isnull().sum()
sql[(sql.pitcher!=1) & (sql.pitcher!=0)].isnull().sum() #27 players have no stats

#drop 27 players where all B/P/F stats are missing
sql.dropna(thresh=4, inplace=True) 

#fill missing stats with zeros
for a in acol:
    sql[a].fillna(value=0, inplace=True)



#look at distributions of stats by induction status to identify potential model inputs
for p in pcol:
    sql[sql.pitcher==1].boxplot(column=p,by='inducted')  #try wins or ratio
    
for b in bcol:
    sql[sql.dhitter==1].boxplot(column=b,by='inducted')  #try home runs, runs, hits
    
for a in acol:
    sql[sql.catcher==1].boxplot(column=a,by='inducted')  #try puts, assists
    
for a in acol:
    sql[(sql.pitcher==0) & (sql.catcher==0) & (sql.dhitter==0)].boxplot(column=a,by='inducted')  #try hits, runs, puts
    
#scatter plot matrix of potential model inputs
pandas.scatter_matrix(sql[['b_hits','p_wins','f_puts', 'b_atbat', 'p_shout']])
#at bats and hits are very highly correlated, likely get rid of at bats


#split data into before year 2000 and on/after year 2000
sql_train = sql[sql.year < 2000]
sql_test = sql[sql.year >= 2000]

sql_train.shape
sql_test.shape

#select response variable in before 2000 set
response_train = sql_train.inducted
response_test = sql_test.inducted

#FIRST MODEL FIT: CATEGORICAL POSITION VARIABLES

#select explanatory variables
explanatory_train1 = sql_train[['pitcher','catcher','dhitter']]
explanatory_test1 = sql_test[['pitcher','catcher','dhitter']]

#run KNN

from sklearn.neighbors import KNeighborsClassifier as knc
knn=knc(p = 2) #specify Euclidean distance
k_range = range(1,30, 2) #specify range of k to test, every second number from 1 to 30, this is number of neighbors
param_grid = dict(n_neighbors=k_range) #set up grid for results

from sklearn.grid_search import GridSearchCV as gscv
grid=gscv(knn, param_grid, cv=10, scoring='accuracy') #instantiate model
grid.fit(explanatory_train1, response_train) #fit model

grid_mean_scores_1 = [result[1] for result in grid.grid_scores_]
best_score_1 = grid.best_score_
best_param_1 = grid.best_params_
knn_optimal_1 = grid.best_estimator_

best_score_1
best_param_1
knn_optimal_1

knn_optimal_pred_1 = knn_optimal_1.predict(explanatory_test1)
accuracy_1 = len(response_test[response_test == knn_optimal_pred_1]) / len(response_test)

accuracy_1
#accuracy is 88% using just categorical variables for position

import matplotlib.pyplot as plt
plt.figure()
plt.plot(k_range, grid_mean_scores_1)
#appears as if at around a k of 5, our accuracy score is highest

#SECOND MODEL FIT: CONTINUOUS STATS

#select explanatory variables
explanatory_train2 = sql_train[['b_hits','p_wins','f_puts', 'b_atbat']]
explanatory_test2 = sql_test[['b_hits','p_wins','f_puts', 'b_atbat']]

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

best_score_2
best_param_2
knn_optimal_2

#future predictions
knn_optimal_pred_2 = knn_optimal_2.predict(explanatory_test2)
accuracy_2 = len(response_test[response_test == knn_optimal_pred_2]) / len(response_test)

accuracy_2
#predict hall of fame with 83% accuracy using just hits, wins, and puts
#predict hall of fame with 87% accuracy using hits, wins, puts, at bats, and shouts

#THIRD MODEL FIT: CATEGORICAL + CONTINUOUS STATS

#select explanatory variables
explanatory_train3 = sql_train[['pitcher','catcher','dhitter','b_hits','p_wins','f_puts', 'b_atbat']]
explanatory_test3 = sql_test[['pitcher','catcher','dhitter','b_hits','p_wins','f_puts', 'b_atbat']]

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
print accuracy_3*100  #87%

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


















