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
WHERE yearID > 1940 and yearID < 2000
GROUP BY yearID
ORDER BY yearID ASC
"""

df = pandas.read_sql(sql, conn)
conn.close()

# drop all NaNs in dataset
df.dropna(inplace = True)


### DUMMY BINARY FEATURES
# 1962 to 1980
df['s1962_e1980'] = 0
df.s1962_e1980[(df.yearID>1962)&(df.yearID<=1980)] = 1

# 1940 to 1962
df['s1940_e1962'] = 0
df.s1940_e1962[(df.yearID<=1962)] = 1


# let's run the formula
bin_est = smf.ols(formula='total_runs ~ s1940_e1962 + s1962_e1980',data=df).fit()
print bin_est.summary()
# intercept represents all years before 1985
# p value just tells us whether there is a relationship - there is most likely a relationship

df['binary_yhat']=bin_est.predict(df)

# see how stolen bases fits the data
plt = df.plot(x='yearID',y='total_runs',kind='scatter')
plt.plot(df.yearID,df.binary_yhat,color='blue',linewidth=3)
# steps represent the ranges of years


### DOUBLES + CAUGHT + 1962

est = smf.ols(formula = 'total_runs ~ doubles + caught + s1940_e1962 + s1962_e1980', data = df).fit() ## ~ is equal to
print est.summary()

df['yhat'] = est.predict(df)

# plot how well the model fits the data
plt = df.plot(x='yearID',  y='total_runs', kind = 'scatter')
plt.plot(df.yearID, df.yhat, color='blue', linewidth=3)

# look at residuals to see if there's heteroskedasticity
df['residuals'] = df.total_runs - df.yhat

plt = df.plot(x = 'total_runs', y='residuals', kind = 'scatter')

# calculating RMSE, ** for exponent in python
RMSE = (((df.residuals)**2).mean()**(1/2))

# understand % by which model deviates from actuals on average
perc_avg_dev = RMSE / df.total_runs.mean()
# string formatting with python
print 'average deviation: {0}%'.format(round(perc_avg_dev*100,1))



### COMBINING ALL FACTORS: doubles + base_balls + walks, triples + base_balls + walks, triples^2 + strikes^2 + base_balls,
# years of 1940-1960 and 1960-1980, 1950-1965 and 1965-1980
large_est = smf.ols(formula = 'total_runs ~ triples + pow(triples, 2) + base_balls + s1940_e1962 + s1962_e1980',
                    data=df).fit()
print large_est.summary()
# stolen_bases are no longer a meaningful count
# you only care about the total RMSE of the whole element
# even though relationship may be too small to be significant, there may be something there

large_rsquared = large_est.rsquared
print large_rsquared
print est.rsquared

# R squared of large_rsquared will necessarily be higher because more variables
# However, it is only an explanation of good fit, not necessarily accuracy
# Adjusted R squared is higher, adjusting for more variables, but still only talking about fit

# calculate residuals and RMSE
df['large_yhat'] = large_est.predict(df)
df['large_residuals'] = df.total_runs - df.large_yhat

RMSE_large = (((df.large_residuals)**2).mean()**(1/2))

print 'average deviation for second equation: {0}'.format(round(RMSE_large,4))
print 'average deviation for first equation: {0}'.format(round(RMSE,4))

# is it more predictive?
plt = df.plot(x='yearID', y='total_runs', kind='scatter')
plt.plot(df.yearID, df.yhat, color='cyan', linewidth=3)
plt.plot(df.yearID, df.large_yhat, color='yellow', linewidth=3)




### DATA AFTERWARD
conn = conn = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')

sql = """ 
SELECT yearID, sum(R) as total_runs, sum('2B') as doubles, sum('3B') as triples, 
sum(G_batting) as batgames, 
sum(SO) as strikes, sum(CS) as caught,
sum(BB) as base_balls, sum(IBB) as walks
FROM Batting
WHERE yearID >= 2000
GROUP BY yearID
ORDER BY yearID ASC
"""
# grouping everything by year

df_after = pandas.read_sql(sql, conn)
conn.close()

df_after['s1962_e1980'] = 0
df_after.s1962_e1980[(df.yearID>1962)&(df.yearID<=1980)] = 1

# 1940 to 1965
df_after['s1940_e1962'] = 0
df_after.s1940_e1962[(df.yearID<=1962)] = 1

# let's predict both modes on new data
df_after['yhat'] = est.predict(df_after)
df_after['large_yhat'] = large_est.predict(df_after)

# create the residuals
df_after['hits_residuals'] = df_after.total_runs - df_after.yhat
df_after['large_residuals'] = df_after.total_runs - df_after.large_yhat

# calculate RMSE
RMSE_large = (((df_after.large_residuals)**2).mean()**(1/2))
RMSE_hits = (((df_after.hits_residuals)**2).mean()**(1/2))


print 'average new data deviation for second equation: {0}'.format(round(RMSE_large,4))
print 'average new data deviation for first equation: {0}'.format(round(RMSE_hits,4))

# equation was OVERFIT using large equation

# let's plot how bad the overfit was
plt = df_after.plot(x='yearID',y='total_runs',kind='scatter')
plt.plot(df_after.yearID, df_after.yhat, color='blue', linewidth=3)
plt.plot(df_after.yearID, df_after.large_yhat, color='orange', linewidth=3)
