# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 18:34:20 2015

@author: Haruo M
"""
from __future__ import division

import sqlite3
import pandas
import numpy
import statsmodels.formula.api as smf

CROSS_VALIDATION_AMOUNT = .2

conn = sqlite3.connect('C:\Users\mizutani\Documents\SQLite\lahman2013.sqlite')
sql = """SELECT playerID, yearID, R AS total_runs, G AS total_games, AB AS total_at_bats, 
H AS total_hits, HR as total_homeruns, RBI AS runs_batted_in, BB AS base_on_balls
FROM Batting 
WHERE yearID > 1954
ORDER BY yearID ASC;"""

df = pandas.read_sql(sql, conn)
conn.close()

df.dropna(inplace = True)      

holdout_num = round(len(df.index) * CROSS_VALIDATION_AMOUNT, 0)

# creating our training and text indices
test_indices = numpy.random.choice(df.index, holdout_num, replace = False )
train_indices = df.index[~df.index.isin(test_indices)]
# our training set
df_train = df.ix[train_indices,]
# our test set
df_test = df.ix[test_indices,]


est_hits = smf.ols(formula='total_runs ~ total_hits', data=df_train).fit()
est_AB = smf.ols(formula='total_runs ~ total_at_bats', data=df_train).fit()
est_HR = smf.ols(formula='total_runs ~ total_homeruns', data=df_train).fit()
est_BB = smf.ols(formula='total_runs ~ base_on_balls', data=df_train).fit()

print est_hits.summary()
print est_AB.summary()
print est_HR.summary()
print est_BB.summary()

print est_hits.rsquared
print est_AB.rsquared
print est_HR.rsquared
print est_BB.rsquared

df_train['yhat'] = est_hits.predict(df_train)
df_train['residuals'] = df_train.total_runs - df_train.yhat
plt = df_train.plot(x='yhat', y='residuals', kind='scatter')
plt = df_train.plot(x='total_hits', y='total_runs', kind='scatter')
plt.plot(df_train.total_hits, df_train.yhat, color='blue', linewidth=3)
df_train['residuals'] = df_train.total_runs - df_train.yhat
RMSE_hits = (((df_train.residuals) ** 2).mean() ** (1/2))
percent_avg_dev_hits = RMSE_hits / df_train.total_runs.mean()     
        
df_train['yhat'] = est_AB.predict(df_train)
plt = df_train.plot(x='yhat', y='residuals', kind='scatter')
plt = df_train.plot(x='total_at_bats', y='total_runs', kind='scatter')
plt.plot(df_train.total_at_bats, df_train.yhat, color='blue', linewidth=3)
df_train['residuals'] = df_train.total_runs - df_train.yhat
RMSE_AB = (((df_train.residuals) ** 2).mean() ** (1/2))
percent_avg_dev_AB = RMSE_AB / df_train.total_runs.mean()
         
df_train['yhat'] = est_HR.predict(df_train)
plt = df_train.plot(x='yhat', y='residuals', kind='scatter')
plt = df_train.plot(x='total_homeruns', y='total_runs', kind='scatter')
plt.plot(df_train.total_homeruns, df_train.yhat, color='blue', linewidth=3)
df_train['residuals'] = df_train.total_runs - df_train.yhat
RMSE_HR = (((df_train.residuals) ** 2).mean() ** (1/2))
percent_avg_dev_HR = RMSE_HR / df_train.total_runs.mean()
         
df_train['yhat'] = est_BB.predict(df_train)
plt = df_train.plot(x='yhat', y='residuals', kind='scatter')
plt = df_train.plot(x='base_on_balls', y='total_runs', kind='scatter')
plt.plot(df_train.base_on_balls, df_train.yhat, color='blue', linewidth=3)
df_train['residuals'] = df_train.total_runs - df_train.yhat
RMSE_BB = (((df_train.residuals) ** 2).mean() ** (1/2))
percent_avg_dev_BB = RMSE_BB / df_train.total_runs.mean()

# Average deviation
print 'average deviation: {0}%'.format(round(percent_avg_dev_hits*100, 1))
print 'average deviation: {0}%'.format(round(percent_avg_dev_AB*100, 1))
print 'average deviation: {0}%'.format(round(percent_avg_dev_HR*100, 1))
print 'average deviation: {0}%'.format(round(percent_avg_dev_BB*100, 1))

# Average total_games is dummy feature
df_train['more_sigma_total_games'] = 0
df_train.more_sigma_total_games[df_train.total_games > (df_train.total_games.mean() + df_train.total_games.std())] = 1

df_train['less_sigma_total_games'] = 0
df_train.less_sigma_total_games[df_train.total_games < (df_train.total_games.mean() + df_train.total_games.std())] = 1

# let's run the formula.
bin_est = smf.ols(formula='total_runs ~ more_sigma_total_games + less_sigma_total_games', data=df_train).fit()
print bin_est.summary()

# lets plot these predictions against actuals
df_train['binary_yhat'] = bin_est.predict(df_train)
plt = df_train.plot(x='total_games', y='total_runs', kind='scatter')
plt.plot(df_train.total_games, df_train.binary_yhat, color='blue', linewidth=3)

# let's combine all three factors together: total hits, stolen bases, and year.
large_est = smf.ols(formula='total_runs ~ total_hits + total_at_bats + more_sigma_total_games + less_sigma_total_games', data=df_train).fit()
print large_est.summary()
large_rsquared = large_est.rsquared
print large_rsquared


df_train['large_yhat'] = large_est.predict(df_train)
df_train['large_residuals'] = df_train.total_runs - df_train.large_yhat
RMSE_large = (((df_train.large_residuals) ** 2).mean() ** (1/2))
print 'average deviation for large equation: {0}'.format(round(RMSE_large, 4))

## RMSe looks better.  Is it really more predictive?
# let's plot the fit of just hits and the full equation.
plt = df_train.plot(x='total_games', y='total_runs', kind='scatter')
#plt.plot(df_train.total_games, df_train.yhat, color='blue', linewidth=3)
plt.plot(df_train.total_games, df_train.large_yhat, color='red', linewidth=3)



########################


df_test['more_sigma_total_games'] = 0
df_test.more_sigma_total_games[df_test.total_games > (df_test.total_games.mean() + df_test.total_games.std())] = 1

df_test['less_sigma_total_games'] = 0
df_test.less_sigma_total_games[df_test.total_games < (df_test.total_games.mean() + df_test.total_games.std())] = 1


# let's predict both modes on the post_2005 data.
#df_test['yhat'] = est.predict(df_test)
df_test['large_yhat'] = large_est.predict(df_test)

# creating the residuals
#df_test['hits_residuals'] = df_test.total_runs - df_test.yhat
df_test['large_residuals'] = df_test.total_runs - df_test.large_yhat

# calculating  RMSE
RMSE_large = (((df_test.large_residuals) ** 2).mean() ** (1/2))
#RMSE_hits =  (((df_test.hits_residuals) ** 2).mean() ** (1/2))

print 'average deviation for large equation: {0}'.format(round(RMSE_large, 4))
#print 'average deviation for just hits: {0}'.format(round(RMSE_hits, 4))
# what does this show you?  
# We were OVERFITTING our data with the large equaiton!
                                            
# lets plot how bad the overfit was.
plt = df_test.plot(x='total_games', y='total_runs', kind='scatter')
#plt.plot(df_test.total_games, df_test.yhat, color='blue', linewidth=3)
plt.plot(df_test.total_games, df_test.large_yhat, color='red', linewidth=3)





         