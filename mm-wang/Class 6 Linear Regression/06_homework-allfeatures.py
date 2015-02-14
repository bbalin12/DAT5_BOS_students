# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 17:03:39 2015

@author: Margaret
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 14:10:31 2015

@author: Margaret
"""

# importing division from the 'future' release of Python (i.e. Python 3)
from __future__ import division

import sqlite3
import pandas

# importing statsmodels to run the linear regression
# scikit-learn also has a linear model, but stats is more user friendly output
import statsmodels.formula.api as smf

# connect the baseball database
conn = conn = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')

sql = """ 
SELECT yearID, sum(R) as total_runs, sum('2B') as doubles, sum('3B') as triples, 
sum(G_batting) as batgames, 
sum(SO) as strikes, sum(CS) as caught,
sum(BB) as base_balls, sum(IBB) as walks
FROM Batting
WHERE yearID > 1901 and yearID < 2000
GROUP BY yearID
ORDER BY yearID ASC
"""

df = pandas.read_sql(sql, conn)
conn.close()

# drop all NaNs in dataset
df.dropna(inplace = True)



## DOUBLES
est = smf.ols(formula = 'total_runs ~ doubles + pow(doubles,2)', data = df).fit() ## ~ is equal to
print est.summary()


df['yhat'] = est.predict(df)

# plot how well the model fits the data
plt = df.plot(x='yearID',  y='total_runs', title = 'Doubles + Doubles^2', color = 'black', kind = 'scatter')
plt.plot(df.yearID, df.yhat, color='blue', linewidth=3)

# look at residuals to see if there's heteroskedasticity
df['residuals'] = df.total_runs - df.yhat

plt = df.plot(x = 'total_runs', y='residuals', kind = 'scatter')

# calculating RMSE, ** for exponent in python
RMSE = (((df.residuals)**2).mean()**(1/2))

# understand % by which model deviates from actuals on average
perc_avg_dev = RMSE / df.total_runs.mean()
# string formatting with python
print 'feature: doubles'
print 'average deviation: {0}%'.format(round(perc_avg_dev*100,1))




## TRIPLES

est = smf.ols(formula = 'total_runs ~ triples + pow(triples,2)', data = df).fit() ## ~ is equal to
print est.summary()


df['yhat'] = est.predict(df)

# plot how well the model fits the data
plt = df.plot(x='yearID',  y='total_runs',title = 'Triples + Triples^2', color = 'black', kind = 'scatter')
plt.plot(df.yearID, df.yhat, color='magenta', linewidth=3)

# look at residuals to see if there's heteroskedasticity
df['residuals'] = df.total_runs - df.yhat

plt = df.plot(x = 'total_runs', y='residuals', kind = 'scatter')

# calculating RMSE, ** for exponent in python
RMSE = (((df.residuals)**2).mean()**(1/2))

# understand % by which model deviates from actuals on average
perc_avg_dev = RMSE / df.total_runs.mean()
# string formatting with python
print 'feature: triples'
print 'average deviation: {0}%'.format(round(perc_avg_dev*100,1))



## BATTED GAMES

est = smf.ols(formula = 'total_runs ~ batgames', data = df).fit() ## ~ is equal to
print est.summary()


df['yhat'] = est.predict(df)

# plot how well the model fits the data
plt = df.plot(x='yearID',  y='total_runs', title = 'Batted Games', color = 'black', kind = 'scatter')
plt.plot(df.yearID, df.yhat, color='yellow', linewidth=3)

# look at residuals to see if there's heteroskedasticity
df['residuals'] = df.total_runs - df.yhat

plt = df.plot(x = 'total_runs', y='residuals', kind = 'scatter')

# calculating RMSE, ** for exponent in python
RMSE = (((df.residuals)**2).mean()**(1/2))

# understand % by which model deviates from actuals on average
perc_avg_dev = RMSE / df.total_runs.mean()
# string formatting with python
print 'feature: batted games'
print 'average deviation: {0}%'.format(round(perc_avg_dev*100,1))



## STRIKES

est = smf.ols(formula = 'total_runs ~ strikes + pow(strikes,2)', data = df).fit() ## ~ is equal to
print est.summary()


df['yhat'] = est.predict(df)

# plot how well the model fits the data
plt = df.plot(x='yearID',  y='total_runs', title = 'Strikes', color = 'black', kind = 'scatter')
plt.plot(df.yearID, df.yhat, color='green', linewidth=3)

# look at residuals to see if there's heteroskedasticity
df['residuals'] = df.total_runs - df.yhat

plt = df.plot(x = 'total_runs', y='residuals', kind = 'scatter')

# calculating RMSE, ** for exponent in python
RMSE = (((df.residuals)**2).mean()**(1/2))

# understand % by which model deviates from actuals on average
perc_avg_dev = RMSE / df.total_runs.mean()
# string formatting with python
print 'feature: strikes'
print 'average deviation: {0}%'.format(round(perc_avg_dev*100,1))



## CAUGHT

est = smf.ols(formula = 'total_runs ~ caught', data = df).fit() ## ~ is equal to
print est.summary()


df['yhat'] = est.predict(df)

# plot how well the model fits the data
plt = df.plot(x='yearID',  y='total_runs', title = 'Caught', color = 'black', kind = 'scatter')
plt.plot(df.yearID, df.yhat, color='purple', linewidth=3)

# look at residuals to see if there's heteroskedasticity
df['residuals'] = df.total_runs - df.yhat

plt = df.plot(x = 'total_runs', y='residuals', kind = 'scatter')

# calculating RMSE, ** for exponent in python
RMSE = (((df.residuals)**2).mean()**(1/2))

# understand % by which model deviates from actuals on average
perc_avg_dev = RMSE / df.total_runs.mean()
# string formatting with python
print 'feature: caught'
print 'average deviation: {0}%'.format(round(perc_avg_dev*100,1))



## BASE ON BALLS

est = smf.ols(formula = 'total_runs ~ base_balls', data = df).fit() ## ~ is equal to
print est.summary()


df['yhat'] = est.predict(df)

# plot how well the model fits the data
plt = df.plot(x='yearID',  y='total_runs', title = 'Base on Balls', color = 'black', kind = 'scatter')
plt.plot(df.yearID, df.yhat, color='orange', linewidth=3)

# look at residuals to see if there's heteroskedasticity
df['residuals'] = df.total_runs - df.yhat

plt = df.plot(x = 'total_runs', y='residuals', kind = 'scatter')

# calculating RMSE, ** for exponent in python
RMSE = (((df.residuals)**2).mean()**(1/2))

# understand % by which model deviates from actuals on average
perc_avg_dev = RMSE / df.total_runs.mean()
# string formatting with python
print 'feature: base_balls'
print 'average deviation: {0}%'.format(round(perc_avg_dev*100,1))



## INTENTIONAL WALKS

est = smf.ols(formula = 'total_runs ~ walks', data = df).fit() ## ~ is equal to
print est.summary()


df['yhat'] = est.predict(df)

# plot how well the model fits the data
plt = df.plot(x='yearID',  y='total_runs', title = 'Intentional Walks', color = 'black', kind = 'scatter')
plt.plot(df.yearID, df.yhat, color='red', linewidth=3)

# look at residuals to see if there's heteroskedasticity
df['residuals'] = df.total_runs - df.yhat

plt = df.plot(x = 'total_runs', y='residuals', kind = 'scatter')

# calculating RMSE, ** for exponent in python
RMSE = (((df.residuals)**2).mean()**(1/2))

# understand % by which model deviates from actuals on average
perc_avg_dev = RMSE / df.total_runs.mean()
# string formatting with python
print 'feature: walks'
print 'average deviation: {0}%'.format(round(perc_avg_dev*100,1))