# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 18:51:06 2015

@author: jchen
"""

# importing division from Python 3
from __future__ import division
# import packages
import sqlite3
import pandas as pd
import statsmodels.formula.api as smf

# connect to the baseball database 
conn = sqlite3.connect('/Users/jchen/Documents/SQLite/lahman2013.sqlite')
# pull in player averages for each year before 2005
sql = """
select yearID, 
	 avg(G_batting) as games_as_batter,
 	 avg(AB) as at_bats,
	 avg(R) as runs, 
	 avg(H) as hits, 
       avg("2B") as doubles,
       avg("3B") as triples,
       avg(HR) as homeruns,
       avg(RBI) as rbi,
	 avg(SB) as stolen_bases,  
	 avg(BB) as walks,
	 avg(SO) as strikeouts,
       avg(IBB) as intentional_walks,
       avg(HBP) as hits_by_pitch,
       avg(SH) as sacrifice_hits,
       avg(SF) as sacrifice_flies
from Batting 
where yearID < 2005
group by yearID
order by yearID;
"""

df = pd.read_sql(sql, conn)
conn.close()

# count up null values
df.isnull().sum()

# drop nulls to make this simpler
df.dropna(inplace = True)  

# start off using all our variables
eq = 'runs ~ games_as_batter + at_bats + hits + doubles \
        + triples + homeruns + rbi + stolen_bases + walks \
        + strikeouts + intentional_walks + hits_by_pitch \
        + sacrifice_hits + sacrifice_flies'

est = smf.ols(formula=eq, data=df).fit()
print est.summary()
# we can see that rbi, hits_by_pitch, and triples have the lowest p-values  
# stolen_bases and sacrifice_hits also has a low p-value
# R-squared of 99.8% - most of variance is explained (makes sense - we used all the data)

# Try a model with rbi, stolen_bases, sacrifice_his, and triples
est_1 = smf.ols(formula='runs ~ rbi + stolen_bases + triples + sacrifice_hits', data=df).fit()
print est_1.summary() 
# This is a signficant intercept, as well...
print est_1.rsquared # 99.7% - quite high

# predict with model 1
df['yhat_1'] = est_1.predict(df)

# plot the data
plt = df.plot(x='yearID', y='runs', kind='scatter')
plt.plot(df.yearID, df.yhat_1, color='blue',
         linewidth=3)
# Almost an exact fit...

# look at residuals
df['residuals_1'] = df.runs - df.yhat_1

plt = df.plot(x='yhat_1', y='residuals_1', kind='scatter')
# seems to be a bit clustered at the lower end of yhat 

# calculate RMSE
RMSE = (((df.residuals_1)**2)**(1/2)).mean()
percent_avg_dev = RMSE / df.runs.mean()
print 'average deviation model 1: {0}%'.format(round(percent_avg_dev*100, 1))
# 0.8% deviation - this is fairly low
# Makes sense, as these features all indicate plays that score runs

# let's now add in years as a variable
# Create some categrical variables for year based on baseball's eras
df['pre_1900'] = 0 # olden days
df.pre_1900[df.yearID<=1900] = 1

df['years_1901_1919'] = 0 # something called the dead-ball era
df.years_1901_1919[(df.yearID>1900) & (df.yearID<=1919)] = 1 

df['years_1920_1941'] = 0 # babe ruth era
df.years_1920_1941[(df.yearID>1919) & (df.yearID<=1941)] = 1 

df['years_1942_1965'] = 0 # post-war
df.years_1942_1965[(df.yearID>1941) & (df.yearID<=1965)] = 1 

df['years_1966_1975'] = 0 
df.years_1966_1975[(df.yearID>1965) & (df.yearID<=1975)] = 1 

df['years_1976_1995'] = 0 # free-agents, steroids, and the like
df.years_1976_1995[(df.yearID>1975) & (df.yearID<=1995)] = 1 
# everything else will be considered more modern times

# augmented model
eq_years = 'runs ~ rbi + stolen_bases + triples + sacrifice_hits + \
                pre_1900 + years_1901_1919 + years_1920_1941 + \
                years_1942_1965 + years_1966_1975 + years_1976_1995'
est_2 = smf.ols(formula=eq_years, data=df).fit()
print est_2.summary()
print est_2.rsquared # 99.8% - even better
# Looks like some of the earlier year bins have poor p-values.
# Likely not a lot of data in these eras.

# predict
df['yhat_2'] = est_2.predict(df)
# Check out residuals
df['residuals_2'] = df.runs - df.yhat_2
plt = df.plot(x='yhat_2', y='residuals_2', kind='scatter')
# clustered on the lower end of yhat_2 but pretty even distribution of residuals
# could be that the model is poor at predicting outliers/very high number of avg runs

RMSE = (((df.residuals_2)**2)**(1/2)).mean()
percent_avg_dev = RMSE / df.runs.mean()
print 'average deviation model 2: {0}%'.format(round(percent_avg_dev*100, 1))
# 0.6% deviation - a little better

# Seems like we've gotten pretty close
# let's look at data after 2005.
conn = sqlite3.connect('/Users/jchen/Documents/SQLite/lahman2013.sqlite')
# creating an object contraining a string that has the SQL query. 
sql = """
select yearID, 
	 avg(G_batting) as games_as_batter,
 	 avg(AB) as at_bats,
	 avg(R) as runs, 
	 avg(H) as hits, 
       avg("2B") as doubles,
       avg("3B") as triples,
       avg(HR) as homeruns,
       avg(RBI) as rbi,
	 avg(SB) as stolen_bases,  
	 avg(BB) as walks,
	 avg(SO) as strikeouts,
       avg(IBB) as intentional_walks,
       avg(HBP) as hits_by_pitch,
       avg(SH) as sacrifice_hits,
       avg(SF) as sacrifice_flies
from Batting 
where yearID >= 2005
group by yearID
order by yearID;
"""

df_post_2005 = pd.read_sql(sql, conn)
conn.close()

# re-create the dummy features for the new data.
# all 0 because post-1995 is not itself a column
df_post_2005['pre_1900'] = 0 
df_post_2005['years_1901_1919'] = 0 
df_post_2005['years_1920_1941'] = 0 
df_post_2005['years_1942_1965'] = 0
df_post_2005['years_1966_1975'] = 0 
df_post_2005['years_1976_1995'] = 0 

# Use model 1 on out-of-sample data
df_post_2005['yhat_1'] = est_1.predict(df_post_2005)
df_post_2005['yhat_2'] = est_2.predict(df_post_2005)

# Residuals
df_post_2005['residuals_1'] = df_post_2005.runs - df_post_2005.yhat_1
df_post_2005['residuals_2'] = df_post_2005.runs - df_post_2005.yhat_2

# calculating  RMSE
RMSE_1 = (((df_post_2005.residuals_1) ** 2) ** (1/2)).mean()
RMSE_2 = (((df_post_2005.residuals_2) ** 2) ** (1/2)).mean()

print 'average deviation for model 1: {0}'.format(
                                            round(RMSE_1, 4))
print 'average deviation for model 2: {0}'.format(
                                            round(RMSE_1, 4))

# Both models produced essentially the same deviation
# This was expected as model_2 only improved r-squared and RMSE slighty.
# For all intents and purposes, no need to include year bins in the model along with original features
