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
# pull in data for each year before 2005
sql = """
select yearID, 
       playerID,
	 sum(G_batting) as games_as_batter,
 	 sum(AB) as at_bats,
	 sum(R) as runs, 
	 sum(H) as hits, 
       sum("2B") as doubles,
       sum("3B") as triples,
       sum(HR) as homeruns,
       sum(RBI) as rbi,
	 sum(SB) as stolen_bases,  
	 sum(BB) as walks,
	 sum(SO) as strikeouts,
       sum(IBB) as intentional_walks,
       sum(HBP) as hits_by_pitch,
       sum(SH) as sacrifice_hits,
       sum(SF) as sacrifice_flies
from Batting 
where yearID < 2005
group by yearID, playerID
order by yearID, playerID;
"""

df = pd.read_sql(sql, conn)
conn.close()

# count up null values
df.isnull().sum()
# lots of null values 

# Let's drop the columns with the most null values
df.drop(['intentional_walks', 'sacrifice_flies'], axis=1, inplace=True)

# drop rows with null values
df.dropna(how='any', inplace=True)

# Start off just using at bats and hits
est_0 = smf.ols(formula= 'runs ~ at_bats + hits', data=df).fit()
print est_0.summary()
print est_0.rsquared
# r-squared: .94 - not bad
# negative intercept means with no at bats, there are negative runs
# which, given the relationship between the two, makes sense.
# the negative intercept of at-bats, however, is odd

# predict with model 0
df['yhat_0'] = est_0.predict(df)

# look at residuals
df['residuals_0'] = df.runs - df.yhat_0

plt = df.plot(x='yhat_0', y='residuals_0', kind='scatter')
# range seems to grow wider at higher y-hat values
# less accuracy at higher values of runs

# calculate RMSE
RMSE_0 = (((df.residuals_0)**2)**(1/2)).mean()
percent_avg_dev_0 = RMSE_0 / df.runs.mean()
print RMSE_0
print 'average deviation model 0: {0}%'.format(round(percent_avg_dev_0*100, 1))
# 19% deviation - not very accurate

###################

# Look at other features
est_1 = smf.ols(formula='runs ~ doubles + triples + homeruns + rbi + stolen_bases + walks + strikeouts + hits_by_pitch + sacrifice_hits', data=df).fit()
print est_1.summary() 
print est_1.rsquared # 95.5%
# we see a negative coefficient here for strikeouts, which makes sense
# the more a player strikes out the fewer chances for runs

# predict with model 1
df['yhat_1'] = est_1.predict(df)

df['residuals_1'] = df.runs - df.yhat_1

plt = df.plot(x='yhat_1', y='residuals_1', kind='scatter')
# distribution skewed slightly down

# calculate RMSE
RMSE_1 = (((df.residuals_1)**2)**(1/2)).mean()
percent_avg_dev_1 = RMSE_1 / df.runs.mean()
print RMSE_1
print 'average deviation model 1: {0}%'.format(round(percent_avg_dev_1*100, 1))
# 17% deviation - not a very big improvement

###################

# Use a combination of intuitively relevant features
est_2 = smf.ols(formula='runs ~ hits + triples + homeruns + stolen_bases + rbi + strikeouts', data=df).fit()
print est_2.summary() 
print est_2.rsquared # 96%
# negative coefficient for rbi as well - which makes intuitive sense
# the more runs batted in for others, the less a player himself has

# predict with model 1
df['yhat_2'] = est_2.predict(df)

df['residuals_2'] = df.runs - df.yhat_2

plt = df.plot(x='yhat_2', y='residuals_2', kind='scatter')
# a similar plot to model 0

# calculate RMSE
RMSE_2 = (((df.residuals_2)**2)**(1/2)).mean()
percent_avg_dev_2 = RMSE_2 / df.runs.mean()
print RMSE_2
print 'average deviation model 2: {0}%'.format(round(percent_avg_dev_2*100, 1))
# slight improvement in average deviation: 15.7%

###################

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
eq_years = 'runs ~ hits + triples + homeruns + stolen_bases + rbi + strikeouts + \
                pre_1900 + years_1901_1919 + years_1920_1941 + years_1942_1965 + years_1966_1975 + years_1976_1995'
est_years = smf.ols(formula=eq_years, data=df).fit()
print est_years.summary()
print est_years.rsquared # 99.8% - even better
# Looks like some of the earlier year bins have poor p-values.
# Likely not a lot of data in these eras.

# predict
df['yhat_years'] = est_years.predict(df)
# Check out residuals
df['residuals_years'] = df.runs - df.yhat_years
plt = df.plot(x='yhat_years', y='residuals_years', kind='scatter')
# clustered on the lower end of yhat_2 but pretty even distribution of residuals
# could be that the model is poor at predicting outliers/very high number of avg runs

RMSE_years = (((df.residuals_years)**2)**(1/2)).mean()
percent_avg_dev_years = RMSE_years / df.runs.mean()
print RMSE_years
print 'average deviation model 2: {0}%'.format(round(percent_avg_dev_years*100, 1))
# 16.4% deviation - slightly higher


# Seems like we've gotten pretty close
# let's look at data after 2005.
conn = sqlite3.connect('/Users/jchen/Documents/SQLite/lahman2013.sqlite')
# creating an object contraining a string that has the SQL query. 
sql = """
select yearID, 
       playerID,
	 sum(G_batting) as games_as_batter,
 	 sum(AB) as at_bats,
	 sum(R) as runs, 
	 sum(H) as hits, 
       sum("2B") as doubles,
       sum("3B") as triples,
       sum(HR) as homeruns,
       sum(RBI) as rbi,
	 sum(SB) as stolen_bases,  
	 sum(BB) as walks,
	 sum(SO) as strikeouts,
       sum(HBP) as hits_by_pitch,
       sum(SH) as sacrifice_hits
from Batting 
where yearID >= 2005
group by yearID, playerID
order by yearID, playerID;
"""

df_post_2005 = pd.read_sql(sql, conn)
conn.close()

df_post_2005.dropna(how='any', inplace=True)

# re-create the dummy features for the new data.
# all 0 because post-1995 is not itself a column
df_post_2005['pre_1900'] = 0 
df_post_2005['years_1901_1919'] = 0 
df_post_2005['years_1920_1941'] = 0 
df_post_2005['years_1942_1965'] = 0
df_post_2005['years_1966_1975'] = 0 
df_post_2005['years_1976_1995'] = 0 

# Use last two model on out-of-sample data
df_post_2005['yhat_2'] = est_2.predict(df_post_2005)
df_post_2005['yhat_years'] = est_years.predict(df_post_2005)

# Residuals
df_post_2005['residuals_2'] = df_post_2005.runs - df_post_2005.yhat_2
df_post_2005['residuals_years'] = df_post_2005.runs - df_post_2005.yhat_years

# calculating  RMSE
RMSE_post_2 = (((df_post_2005.residuals_2) ** 2) ** (1/2)).mean()
RMSE_post_years = (((df_post_2005.residuals_years) ** 2) ** (1/2)).mean()

percent_avg_dev_post_2 = RMSE_post_2 / df.runs.mean()
percent_avg_dev_post_years = RMSE_post_years / df.runs.mean()


print 'average deviation for model 2: {0}'.format(
                                            round(RMSE_post_2, 4))
print 'average % deviation for model 2: {0}%'.format(
                                            round(percent_avg_dev_post_2*100, 1))
print 'average deviation for model 2 + years: {0}'.format(
                                            round(RMSE_post_years, 4))
print 'average % deviation for model 2 + years: {0}%'.format(
                                            round(percent_avg_dev_post_years*100, 1))

# Model without year actually has a lower % deviation
