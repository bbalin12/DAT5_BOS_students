# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 21:25:08 2015

@author: megan
"""
# importing division from the 'future' release of Python (i.e. Python 3)
from __future__ import division

import sqlite3
import pandas
import numpy
import statsmodels.formula.api as smf

# Using the Baseball dataset, build a linear regression model that predicts how many runs a player will have in a given year.
# Begin with more than one possible model; each of which should have at least one categorical dummy feature and at least two continuous explanatory features.
# Make sure you check for heteroskedasticity in your models.
# Decide whether to include or take out the model's features depending on whether they may be collinear or insignificant.
# Interpret the model's coefficients and intercept.
# Calculate the models' R-squared and in-sample RMSE.
# Make sure to use a holdout group or k-fold CV to calculate out-of-sample RMSE for your model group.
# Decide on the best model to use, and justify why you made that choice.

##############
# Model 1 & 2
##############
conn = sqlite3.connect('/Users/megan/Documents/SQLite/lahman2013.sqlite')
sql = """select b.yearID,
sum(b.R) as total_runs, sum(b.H) as total_hits, 
sum(b.SB) as stolen_bases, sum(b.SO) as strikeouts, 
sum(b.IBB) as total_intentional_walks,
sum(b.'2B') as doubles, sum(b.'3B') as triples,
sum(b.HR) as homeruns, sum(b.AB) as atbats,
b.teamID, b.lgID
from Batting b
where b.yearID > 1954
and b.yearid < 2005
group by b.yearID, b.teamID
order by b.yearID ASC
"""

df = pandas.read_sql(sql, conn)
conn.close()
df.dropna(inplace = True)     

# Create dummy variables
df['national_league'] = 0
df.national_league[df.lgID == 'NL'] = 1 
df['redsox'] = 0
df.redsox[df.teamID == 'BOS'] = 1 
 
# Fit models
est_model1 = smf.ols(formula='total_runs ~ doubles + homeruns + national_league', data=df).fit()
print est_model1.summary()
print est_model1.rsquared

est_model2 = smf.ols(formula='total_runs ~ doubles + triples + homeruns + atbats + redsox', data=df).fit()
print est_model2.summary()
print est_model2.rsquared

# Plot data
df['yhat_model1'] = est_model1.predict(df)
plt = df.plot(x='national_league', y='total_runs', kind='scatter') 
plt = df.plot(x='doubles', y='total_runs', kind='scatter')
plt = df.plot(x='homeruns', y='total_runs', kind='scatter')
plt.plot(df.total_hits, df.yhat_model1, color='blue',
         linewidth=3)
         
df['yhat_model2'] = est_model2.predict(df)
plt = df.plot(x='redsox', y='total_runs', kind='scatter') 
plt = df.plot(x='doubles', y='total_runs', kind='scatter')
plt = df.plot(x='triples', y='total_runs', kind='scatter')
plt = df.plot(x='homeruns', y='total_runs', kind='scatter')
plt = df.plot(x='atbats', y='total_runs', kind='scatter')
plt.plot(df.total_hits, df.yhat_model2, color='blue',
         linewidth=3)

# Look for heteroskedasticity.. looks good.
df['residuals_model1'] = df.total_runs - df.yhat_model1
plt = df.plot(x='yhat_model1', y='residuals_model1', kind='scatter')
df['residuals_model2'] = df.total_runs - df.yhat_model2
plt = df.plot(x='yhat_model2', y='residuals_model2', kind='scatter')

# Calculate RMSE
RMSE_model1 = (((df.residuals_model1) ** 2).mean() ** (1/2))
percent_avg_dev_model1 = RMSE_model1 / df.total_runs.mean()
print 'average deviation for model 1: {0}%'.format(round(percent_avg_dev_model1*100, 1))

RMSE_model2 = (((df.residuals_model2) ** 2).mean() ** (1/2))
percent_avg_dev_model2 = RMSE_model2 / df.total_runs.mean()
print 'average deviation for model 2: {0}%'.format(round(percent_avg_dev_model2*100, 1))

# Now, let's plot how well the model fits the data. 
plt = df.plot(x='yearID', y='total_runs', kind='scatter')
plt.plot(df.total_hits, df.yhat_model1, color='blue',
         linewidth=3)

plt = df.plot(x='yearID', y='total_runs', kind='scatter')
plt.plot(df.total_hits, df.yhat_model2, color='blue',
         linewidth=3)

###########################
# PREFORMANCE ON TEST DATA
###########################

# Let's see how well it predicts data after 2005
conn = sqlite3.connect('/Users/megan/Documents/SQLite/lahman2013.sqlite')
sql = """select b.yearID, 
sum(b.R) as total_runs, sum(b.H) as total_hits, 
sum(b.SB) as stolen_bases, sum(b.SO) as strikeouts, 
sum(b.IBB) as total_intentional_walks,
sum(b.'2B') as doubles, sum(b.'3B') as triples,
sum(b.HR) as homeruns, sum(b.AB) as atbats,
b.teamID, b.lgID
from Batting b
where b.yearid > 2005
group by b.yearID, b.teamID
order by b.yearID ASC"""
df_post_2005 = pandas.read_sql(sql, conn)
conn.close()
df.dropna(inplace = True)     

# Create dummy variables
df_post_2005['national_league'] = 0
df_post_2005.national_league[df.lgID == 'NL'] = 1 
df_post_2005['redsox'] = 0
df_post_2005.redsox[df.teamID == 'BOS'] = 1 
 
# Predict
df_post_2005['yhat_model1'] = est_model1.predict(df_post_2005)
df_post_2005['yhat_model2'] = est_model2.predict(df_post_2005)
df_post_2005['residuals_model1'] = df_post_2005.total_runs - df_post_2005.yhat_model1
plt = df.plot(x='yhat_model1', y='residuals_model1', kind='scatter')
df_post_2005['residuals_model2'] = df_post_2005.total_runs - df_post_2005.yhat_model2
plt = df.plot(x='yhat_model2', y='residuals_model2', kind='scatter')

RMSE_model1 = (((df_post_2005.residuals_model1) ** 2).mean() ** (1/2))
percent_avg_dev_model1 = RMSE_model1 / df.total_runs.mean()
print 'average deviation: {0}%'.format(round(percent_avg_dev_model1*100, 1))

RMSE_model2 = (((df_post_2005.residuals_model2) ** 2).mean() ** (1/2))
percent_avg_de_model2v = RMSE_model2 / df.total_runs.mean()
print 'average deviation: {0}%'.format(round(percent_avg_dev_model2*100, 1))

plt = df_post_2005.plot(x='yearID', y='total_runs', kind='scatter')
plt.plot(df_post_2005.yearID, df_post_2005.yhat_model1, color='blue',
         linewidth=3)

plt = df_post_2005.plot(x='yearID', y='total_runs', kind='scatter')
plt.plot(df_post_2005.yearID, df_post_2005.yhat_model2, color='blue',
         linewidth=3)
         
# Model 2 is the better model choice