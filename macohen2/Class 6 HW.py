# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 15:20:14 2015

@author: MatthewCohen
"""

from __future__ import division

import sqlite3
import pandas
import numpy

import statsmodels.formula.api as smf

conn = sqlite3.connect('/Users/MatthewCohen/Documents/SQLite/lahman2013.sqlite')

sql = """SELECT yearID, avg(R) as average_runs, avg(H) as average_hits, avg(SB) as average_stolen_bases, avg(BB) as average_walks
FROM Batting 
WHERE yearID > 1950 and yearID <2000
GROUP BY yearID
ORDER BY yearID ASC;"""

df = pandas.read_sql(sql, conn)
conn.close()

df.dropna(inplace = True)

est = smf.ols(formula='average_runs ~ average_hits + average_stolen_bases', data=df).fit()
print est.summary()
print est.rsquared

df['yhat'] = est.predict(df)

plt = df.plot(x='average_hits', y='average_runs', kind='scatter')
plt.plot(df.average_hits, df.yhat, color='blue',
         linewidth=3)
         
df['residuals'] = df.average_runs - df.yhat
plt = df.plot(x='yhat', y='residuals', kind='scatter')

RMSE = (((df.residuals) ** 2).mean() ** (1/2))
percent_avg_dev = RMSE / df.average_runs.mean()
print 'average deviation: {0}%'.format(round(percent_avg_dev*100, 1))


df['post_1975'] = 0
df.post_1975[df.yearID>1975] = 1



bin_est = smf.ols(formula='average_runs ~ post_1975', data=df).fit()
print bin_est.summary()

df['binary_yhat'] = bin_est.predict(df)
plt = df.plot(x='yearID', y='average_runs', kind='scatter')
plt.plot(df.yearID, df.binary_yhat, color='blue',
         linewidth=3)


est2 = smf.ols(formula='average_runs ~ average_hits + average_walks', data=df).fit()
print est2.summary()
print est2.rsquared

df['yhat2'] = est2.predict(df)
df['residuals2'] = df.average_runs - df.yhat2


totalest = smf.ols(formula='average_runs ~ average_hits + average_walks + average_stolen_bases', data=df).fit()
print totalest.summary()
print totalest.rsquared

df['total_yhat'] = totalest.predict(df)
df['total_residuals'] = df.average_runs - df.total_yhat

RMSE_total = (((df.total_residuals) ** 2).mean() ** (1/2))
percent_avg_dev = RMSE_total / df.average_runs.mean()
print 'average deviation: {0}%'.format(round(percent_avg_dev*100, 1))


plt = df.plot(x='yhat2', y='residuals2', kind='scatter')

# est 2 seems to be the best model b/c it predicts the data with about the same accuracy as "totalest" however
# uses one less variable

conn = sqlite3.connect('/Users/MatthewCohen/Documents/SQLite/lahman2013.sqlite')

sql = """SELECT yearID, avg(R) as average_runs, avg(H) as average_hits, avg(SB) as average_stolen_bases, avg(BB) as average_walks
FROM Batting 
WHERE yearID > 2000
GROUP BY yearID
ORDER BY yearID ASC;"""

df2 = pandas.read_sql(sql, conn)
conn.close()

df2.dropna(inplace = True)

est2 = smf.ols(formula='average_runs ~ average_hits + average_walks', data=df2).fit()
print est2.summary()
print est2.rsquared




