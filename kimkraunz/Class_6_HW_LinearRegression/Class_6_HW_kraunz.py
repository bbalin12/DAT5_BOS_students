# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 11:40:17 2015

@author: jkraunz
"""

import pandas
import numpy as np
import sqlite3
import matplotlib.pyplot as plt

conn = sqlite3.connect('/Users/jkraunz/Documents/SQLite/lahman2013.sqlite.crdownload')

sql = '''
Select playerID, yearID, teamID as team, lgID as league, G_batting as games, AB as at_bats, 
R as runs, H as hits, `2B` as doubles, `3B` as triples, HR as home_runs, 
SB as stolen_bases, SO as strike_outs, BB as walks, HBP as hit_by_pitch, IBB as intentional_walks
From Batting
Where yearID <= 2005 and yearID > 1954 and at_bats > 10
'''

df = pandas.read_sql(sql, conn)

conn.close()

df.head(30)

df.dropna(inplace = True)

df.head(30)
df.count()

#################################################################
# create categorical data for years


# add the singles variable
df['singles'] = df.hits - df.doubles - df.triples - df.home_runs
df.hits.head(30)
df.singles.head(30)

df['from_1995_to_2005'] = 0
df.from_1995_to_2005[(df.yearID > 1994) & (df.yearID <= 2005)] = 1

df['post_2005'] =  0
df.post_2005[df.yearID > 2005] = 1

df.post_2005.head(30)
df.from_1995_to_2005.head(30)


#################################################################

# Exploratory statistics and graphs

# look for correlation between variables
df.corr('pearson')

df.describe()

plt = plt.hist(df.yearID, bins=np.arange(1950, 2004), color='#cccccc')
# Will only use data from 1955 on because numbers are small prior to that


common_params = dict(bins=20, range=(0, 250), normed=True)
plt.subplots_adjust(hspace=2)
plt.subplot(611)
plt.title('Games')
plt.hist(df.games, **common_params)
plt.subplot(612)
plt.title('At bats')
plt.hist(df.at_bats, **common_params)
plt.subplot(613)
plt.title('Runs')
plt.hist(df.runs, **common_params)
plt.subplot(614)
plt.title('Hits')
plt.hist(df.hits, **common_params)
plt.subplot(615)
plt.title('Strike Outs')
plt.hist(df.strike_outs, **common_params)
plt.subplot(616)
plt.title('Walks')
plt.hist(df.walks, **common_params)
plt.show()

common_params = dict(bins=20, range=(0, 50), normed=True)
plt.subplots_adjust(hspace=.01)
plt.subplot(411)
plt.title('Doubles')
plt.hist(df.doubles, **common_params)
plt.subplot(412)
plt.title('Triples')
plt.hist(df.triples, **common_params)
plt.subplot(413)
plt.title('Home Runs')
plt.hist(df.home_runs, **common_params)
plt.subplot(414)
plt.title('Stolen Bases')
plt.hist(df.stolen_bases, **common_params)
plt.show()


f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(5, 2, sharex='col', sharey='row')
ax1.scatter(df.games, df.runs)
ax1.set_ylabel('Runs')
ax1.set_title('Games')
ax2.scatter(df.at_bats, df.runs)
ax2.set_title('At bats')
ax3.scatter(df.singles, df.runs)
ax3.set_ylabel('Runs')
ax3.set_title('Singles')
ax4.scatter(df.hits, df.runs)
ax4.set_title('Hits')
ax5.scatter(df.doubles, df.runs)
ax5.set_ylabel('Runs')
ax5.set_title('Doubles')
ax6.scatter(df.triples, df.runs)
ax6.set_title('Triples')
ax7.scatter(df.home_runs, df.runs)
ax7.set_ylabel('Runs')
ax7.set_title('Home Runs')
ax8.scatter(df.stolen_bases, df.runs)
ax8.set_title('Stolen Bases')
ax9.scatter(df.strike_outs, df.runs)
ax9.set_ylabel('Runs')
ax9.set_title('Strike Outs')
ax10.scatter(df.walks, df.runs)
ax10.set_title('Walks')

##########################################################################

# linear regression to predict number of runs
import statsmodels.formula.api as smf
from __future__ import division

m1_est = smf.ols(formula = 'runs ~  yearID + at_bats + hits + doubles + triples + home_runs + stolen_bases + strike_outs + walks', data = df).fit()

print m1_est.summary()
df['m1_yhat'] = m1_est.predict(df)
df['m1_residuals']= df.runs - df.m1_yhat

plt = df.plot(x='m1_yhat', y='m1_residuals', kind='scatter')

m1_RMSE = (((df.m1_residuals) ** 2).mean() ** (1/2))

print m1_RMSE
print 'average deviation for model 1: {0}'.format(round(m1_RMSE, 4))

# heteroskedacity is OK but not great.  more variant residuals at higher yhats

# The model was changed to have include singles instead of hits


m2_est = smf.ols(formula = 'runs ~  yearID + at_bats + singles+ doubles + triples + home_runs + stolen_bases + strike_outs + walks', data = df).fit()

print m2_est.summary()
df['m2_yhat'] = m2_est.predict(df)
df['m2_residuals']= df.runs - df.m2_yhat

plt = df.plot(x='m2_yhat', y='m2_residuals', kind='scatter')

m2_RMSE = (((df.m2_residuals) ** 2).mean() ** (1/2))

print 'average deviation for model 1: {0}'.format(round(m1_RMSE, 4))
print 'average deviation for model 2: {0}'.format(round(m2_RMSE, 4))

# changing hits to singles did not change the model since singles is in direct correlation to hits
# heteroskedacity is OK but not great.  more variant residuals at higher yhats


# changed the model to include years as a categorical variable since year played will most likely have
# an effect on runs.  The categories were pre-1995 (pre-steroid era), from_1995_to_2005 (steroid era),
# and post 2005 (hopefully post steroid era)

m3_est = smf.ols(formula = 'runs ~ from_1995_to_2005 + post_2005 + at_bats + singles+ doubles + triples + home_runs + stolen_bases + strike_outs + walks', data = df).fit()

print m3_est.summary()
df['m3_yhat'] = m3_est.predict(df)
df['m3_residuals']= df.runs - df.m3_yhat

plt = df.plot(x='m3_yhat', y='m3_residuals', kind='scatter')

m3_RMSE = (((df.m3_residuals) ** 2).mean() ** (1/2))

print 'average deviation for model 1: {0}'.format(round(m1_RMSE, 4))
print 'average deviation for model 2: {0}'.format(round(m2_RMSE, 4))
print 'average deviation for model 3: {0}'.format(round(m3_RMSE, 4))

# there is a slightly lower RMSE by making the years categorical
# heteroskedacity is OK but not great.  more variant residuals at higher yhats

# changed the model to include intentional walks and hit by pitch, to see if these
# rarer events would have an effect on runs.  also removed at_bats since it is strongly
# collinear

m4_est = smf.ols(formula = 'runs ~ from_1995_to_2005 + post_2005 + singles+ doubles + triples + home_runs + stolen_bases + strike_outs + walks + hit_by_pitch + intentional_walks', data = df).fit()

print m4_est.summary()
df['m4_yhat'] = m4_est.predict(df)
df['m4_residuals']= df.runs - df.m4_yhat

plt = df.plot(x='m4_yhat', y='m4_residuals', kind='scatter')

m4_RMSE = (((df.m4_residuals) ** 2).mean() ** (1/2))

print 'average deviation for model 1: {0}'.format(round(m1_RMSE, 4))
print 'average deviation for model 2: {0}'.format(round(m2_RMSE, 4))
print 'average deviation for model 3: {0}'.format(round(m3_RMSE, 4))
print 'average deviation for model 4: {0}'.format(round(m4_RMSE, 4))

# adding hit by pitch and intentionally walked  and removing at_bats both 
# lowered the RMSE and raised the r squared
# heteroskedacity is OK but not great.  more variant residuals at higher yhats.


# checked to see what the effect of at_bats was on the model.  left out intentional 
# walks and hit by pitch as well (only looking for change in model due to at_bats)

m5_est = smf.ols(formula = 'runs ~ from_1995_to_2005 + post_2005 + singles + doubles + triples + home_runs + stolen_bases + strike_outs + walks', data = df).fit()

print m5_est.summary()
df['m5_yhat'] = m5_est.predict(df)
df['m5_residuals']= df.runs - df.m5_yhat

plt = df.plot(x='m5_yhat', y='m5_residuals', kind='scatter')

m5_RMSE = (((df.m5_residuals) ** 2).mean() ** (1/2))

print 'average deviation for model 1: {0}'.format(round(m1_RMSE, 4))
print 'average deviation for model 2: {0}'.format(round(m2_RMSE, 4))
print 'average deviation for model 3: {0}'.format(round(m3_RMSE, 4))
print 'average deviation for model 4: {0}'.format(round(m4_RMSE, 4))
print 'average deviation for model 5: {0}'.format(round(m5_RMSE, 4))

# Model 4 has the highest R squared and the lowest RMSE.  Will use going forward.
# The heteroskedacity was similar between the models.  It was not great but not bad enough 
# to manipulate the variables

conn = sqlite3.connect('/Users/jkraunz/Documents/SQLite/lahman2013.sqlite.crdownload')

sql = '''
Select playerID, yearID, teamID as team, lgID as league, G_batting as games, AB as at_bats, 
R as runs, H as hits, `2B` as doubles, `3B` as triples, HR as home_runs, 
SB as stolen_bases, SO as strike_outs, BB as walks, HBP as hit_by_pitch, IBB as intentional_walks
From Batting
Where yearID > 2005 and at_bats > 10
'''

df = pandas.read_sql(sql, conn)

conn.close()

df.head(30)
df.describe()

df.dropna(inplace = True)

df.head(30)
df.count()

df['from_1995_to_2005'] = 0
df['post_2005'] = 1
df['singles'] = df.hits - df.doubles - df.triples - df.home_runs

df['m4_yhat'] = m4_est.predict(df)

df['m4_residuals']= df.runs - df.m4_yhat

plt = df.plot(x='m4_yhat', y='m4_residuals', kind='scatter')

m4_RMSE = (((df.m4_residuals) ** 2).mean() ** (1/2))


print 'average deviation for M4 equation on post 2005 data: {0}'.format(
                                            round(m4_RMSE, 4))


