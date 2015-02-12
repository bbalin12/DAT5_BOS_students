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

plt.hist(df.yearID, bins=np.arange(1910, 2004), color='#cccccc')

plt.hist(df.yearID, bins=np.arange(1950, 2004), color='#cccccc')
# Will only use data from 1955 on because numbers are small prior to that

plt.hist(df.games, bins=20)
plt.xlabel('Games')

plt = df.at_bats.hist(bins=50)
plt.title('At Bats')

plt = df.at_bats.hist(bins=10, range = [0,10])
plt.title('At Bats')
# large number with 10 or less at_bats.  will eliminate from dataset (although they shouldn't)
# have an effect

plt = df.at_bats.hist(bins=100)
plt.title('At Bats')

plt = df.runs.hist(bins=15, range = [0, 150])
plt.title('Runs')

plt = df.hits.hist(bins=15, range = [0, 250])
plt.title('Hits')

plt = df.doubles.hist(bins=15, range = [0, 60])
plt.title('Doubles')

plt = df.triples.hist(bins=10, range = [0, 20])
plt.title('Triples')

plt = df.home_runs.hist(bins=15, range = [0, 50])
plt.title('Home Runs')

plt = df.stolen_bases.hist(bins=15, range = [0, 60])
plt.title('Solen Bases')

plt = df.strike_outs.hist(bins=20, range = [0, 200])
plt.title('Strike Outs')

plt = df.walks.hist(bins=20, range=[0, 140])
plt.title("Walks")

df.plot(kind = 'scatter', x = 'games', y = 'runs', alpha = 0.5)

df.plot(kind = 'scatter', x = 'at_bats', y = 'runs', alpha = 0.5)

df.plot(kind = 'scatter', x = 'singles', y = 'runs', alpha = 0.5)

df.plot(kind = 'scatter', x = 'hits', y = 'runs', alpha = 0.5)

df.plot(kind = 'scatter', x = 'doubles', y = 'runs', alpha = 0.5)

df.plot(kind = 'scatter', x = 'triples', y = 'runs', alpha = 0.5)

df.plot(kind = 'scatter', x = 'home_runs', y = 'runs', alpha = 0.5)

df.plot(kind = 'scatter', x = 'stolen_bases', y = 'runs', alpha = 0.5)

df.plot(kind = 'scatter', x = 'strike_outs', y = 'runs', alpha = 0.5)

df.plot(kind = 'scatter', x = 'walks', y = 'runs', alpha = 0.5)

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

df.dropna(inplace = True)

df.head(30)
df.count()

df['from_1995_to_2005'] = 0
df['post_2005'] = 1
df['singles'] = df.hits - df.doubles - df.triples - df.home_runs

post_2005_m4_est = smf.ols(formula = 'runs ~ from_1995_to_2005 + post_2005 + singles + doubles + triples + home_runs + stolen_bases + strike_outs + walks + hit_by_pitch + intentional_walks', data = df).fit()

print post_2005_m4_est.summary()
df['post_2005_m4_yhat'] = post_2005_m4_est.predict(df)
df['post_2005_m4_residuals']= df.runs - df.post_2005_m4_yhat

plt = df.plot(x='post_2005_m4_yhat', y='post_2005_m4_residuals', kind='scatter')

post_2005_m4_RMSE = (((df.post_2005_m4_residuals) ** 2).mean() ** (1/2))


print 'average deviation for model 4: {0}'.format(round(m4_RMSE, 4))

print 'average deviation for post 2005 model 4: {0}'.format(round(post_2005_m4_RMSE, 4))


######################################################################################

# Doing it again but post 1957 when eligibility for the batting title changed to 3.1 plate appearances per game.
# Will use a minimum of 477 at bats to account for the 154 game seasons

conn = sqlite3.connect('/Users/jkraunz/Documents/SQLite/lahman2013.sqlite.crdownload')

sql = '''
Select playerID, yearID, teamID as team, lgID as league, G_batting as games, AB as at_bats, 
R as runs, H as hits, `2B` as doubles, `3B` as triples, HR as home_runs, 
SB as stolen_bases, SO as strike_outs, BB as walks, HBP as hit_by_pitch, IBB as intentional_walks
From Batting
Where yearID <= 2005 and yearID >= 1957 and at_bats > 477
'''

df = pandas.read_sql(sql, conn)

conn.close()

df.head(30)

df.dropna(inplace = True)

df.head(30)
df.count()

df['singles'] = df.hits - df.doubles - df.triples - df.home_runs
df.hits.head(30)
df.singles.head(30)

df['from_1995_to_2005'] = 0
df.from_1995_to_2005[(df.yearID > 1994) & (df.yearID <= 2005)] = 1

df['post_2005'] =  0
df.post_2005[df.yearID > 2005] = 1

df.post_2005.head(30)
df.from_1995_to_2005.head(30)


df.corr('pearson')

df.describe()


plt.hist(df.yearID, bins=np.arange(1957, 2004), color='#cccccc')


plt.hist(df.games, bins=20)
plt.xlabel('Games')

plt = df.at_bats.hist(bins=50)
plt.title('At Bats')

plt = df.runs.hist(bins=15, range = [0, 150])
plt.title('Runs')

plt = df.hits.hist(bins=15, range = [0, 250])
plt.title('Hits')

plt = df.doubles.hist(bins=15, range = [0, 60])
plt.title('Doubles')

plt = df.triples.hist(bins=10, range = [0, 20])
plt.title('Triples')

plt = df.home_runs.hist(bins=15, range = [0, 50])
plt.title('Home Runs')

plt = df.stolen_bases.hist(bins=15, range = [0, 60])
plt.title('Solen Bases')

plt = df.strike_outs.hist(bins=20, range = [0, 200])
plt.title('Strike Outs')

plt = df.walks.hist(bins=20, range=[0, 140])
plt.title("Walks")

df.plot(kind = 'scatter', x = 'games', y = 'runs', alpha = 0.5)

df.plot(kind = 'scatter', x = 'at_bats', y = 'runs', alpha = 0.5)

df.plot(kind = 'scatter', x = 'singles', y = 'runs', alpha = 0.5)

df.plot(kind = 'scatter', x = 'hits', y = 'runs', alpha = 0.5)

df.plot(kind = 'scatter', x = 'doubles', y = 'runs', alpha = 0.5)

df.plot(kind = 'scatter', x = 'triples', y = 'runs', alpha = 0.5)

df.plot(kind = 'scatter', x = 'home_runs', y = 'runs', alpha = 0.5)

df.plot(kind = 'scatter', x = 'stolen_bases', y = 'runs', alpha = 0.5)

df.plot(kind = 'scatter', x = 'strike_outs', y = 'runs', alpha = 0.5)

df.plot(kind = 'scatter', x = 'walks', y = 'runs', alpha = 0.5)

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

# Model 1 is OK.  It has a higher RMSE than when I include 1955 and on and at_bats 
# less than 477
# heteroskedacity is much better with new parameters

# again changed model so that hits became single
m2_est = smf.ols(formula = 'runs ~  yearID + at_bats + singles+ doubles + triples + home_runs + stolen_bases + strike_outs + walks', data = df).fit()

print m2_est.summary()
df['m2_yhat'] = m2_est.predict(df)
df['m2_residuals']= df.runs - df.m2_yhat

plt = df.plot(x='m2_yhat', y='m2_residuals', kind='scatter')

m2_RMSE = (((df.m2_residuals) ** 2).mean() ** (1/2))

print 'average deviation for model 1: {0}'.format(round(m1_RMSE, 4))
print 'average deviation for model 2: {0}'.format(round(m2_RMSE, 4))

# again, models 1 and 2 are identical because singles is a function of hits


# made years categorical to represent steroid era
m3_est = smf.ols(formula = 'runs ~ from_1995_to_2005 + post_2005 + at_bats + singles+ doubles + triples + home_runs + stolen_bases + strike_outs + walks', data = df).fit()

print m3_est.summary()
df['m3_yhat'] = m3_est.predict(df)
df['m3_residuals']= df.runs - df.m3_yhat

plt = df.plot(x='m3_yhat', y='m3_residuals', kind='scatter')

m3_RMSE = (((df.m3_residuals) ** 2).mean() ** (1/2))

print 'average deviation for model 1: {0}'.format(round(m1_RMSE, 4))
print 'average deviation for model 2: {0}'.format(round(m2_RMSE, 4))
print 'average deviation for model 3: {0}'.format(round(m3_RMSE, 4))

# slightly lower RMSE with categorical years but much better r squared

# eliminated at_bats since it's a collinear variable and added intentional walks
# and hit by pitch even though relatively rare events
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

# The r squared increases and the RMSE decreases.
# heteroskedacity is still good

# remove intentional walk and hit by pitch to see if the rare events have an effect
# on the model

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

# in removing at_bats and including hit by pitch and intentional walks, we decrease
# the accuracy and goodness of fit.  I've found that model 4 is my best model based on 
# the lowest RMSE and highed r squared

# Now I want to test my model on an outside group.  I heldout the data for years 
# after 2005 to test on.

conn = sqlite3.connect('/Users/jkraunz/Documents/SQLite/lahman2013.sqlite.crdownload')

sql = '''
Select playerID, yearID, teamID as team, lgID as league, G_batting as games, AB as at_bats, 
R as runs, H as hits, `2B` as doubles, `3B` as triples, HR as home_runs, 
SB as stolen_bases, SO as strike_outs, BB as walks, HBP as hit_by_pitch, IBB as intentional_walks
From Batting
Where yearID > 2005 and at_bats > 477
'''

df = pandas.read_sql(sql, conn)

conn.close()

df.head(30)

df.dropna(inplace = True)

df.head(30)
df.count()

df['from_1995_to_2005'] = 0
df['post_2005'] = 1
df['singles'] = df.hits - df.doubles - df.triples - df.home_runs

post_2005_m4_est = smf.ols(formula = 'runs ~ from_1995_to_2005 + post_2005 + singles+ doubles + triples + home_runs + stolen_bases + strike_outs + walks + hit_by_pitch + intentional_walks', data = df).fit()

print post_2005_m4_est.summary()
df['post_2005_m4_yhat'] = post_2005_m4_est.predict(df)
df['post_2005_m4_residuals']= df.runs - df.post_2005_m4_yhat

plt = df.plot(x='post_2005_m4_yhat', y='post_2005_m4_residuals', kind='scatter')

post_2005_m4_RMSE = (((df.post_2005_m4_residuals) ** 2).mean() ** (1/2))

print 'average deviation for model 4: {0}'.format(round(m4_RMSE, 4))
print 'average deviation for post 2005 model 4: {0}'.format(round(post_2005_m4_RMSE, 4))


'''
Model selection: I tested 5 linear regression models to see the effect of different
variables on my models.  First, I used explatory statistics and graphs to check for collinearity
as well as interesting trends.  Because there were so many players with less than 10 at bats
I eleminated them because I didn't want random events to skew the data.  I also limited 
the analysis to 1955 and later because of the limited number of data prior to 1955.

I designated data from post 2005 as my holdout group.

To start, I tested the effect of year, hits, doubles, triples, home runs, stolen bases, 
and walkson runs.  I tested whether changing hits to singles had an effect on my model
and found that models 1 and 2 were identical because singles was a function of hits
and they are directly collinear.  I chose to use singles instead of hits moving forward.
I then tested whether changing years to a category representing pre-steroid era (pre-1995), 
the steroid era (1995-2005), and post-steroid era (after 2005).  I did see a modestly decreased 
RMSE and increased R squared.  I next tested whether removing at bats and adding hit by
pitch and intentional walks had an effect on predicting runs.  I removed at bats because it 
is strongly collinear and added hit by pitch and intentional walks because although they
are rare events, they could have an effect on runs.  I saw the largest decrease in RMSE in 
this model.  Finally I tested removed hit by pitch and intentional walks to see whether the change
between models 3 and 4 could be attributed to removing at bats or adding hit by pitch 
and intentional walks.  There was an increase in RMSE and decrease in r squared back to 
the approximate levels for models 3 and earlier.  I also plotted the residuals to understand 
the heteroskedacity of the models.  I found that while there was some heteroskedacity, it was modest
and I don't believe transforming variables will improve it enough to lose the interpretibility
of non-transformed variables.

I then reran the above models but limited the data to only 1957 or later and at bats greater 
than 477.  1957 was the year that the parameters for the batting title changed and I wanted to 
place a minimum on at bats that fit the parameters for being eligible for the batting title.  Again, 
I found that model 4 had the highest r squared and lowest RMSE.

My model:
runs = pre-1995 + 1995-2005 + singles + doubles + triple + home runs + stolen bases + strike outs
        + walks + hit by pitch + intentional walks

Model testing: 
I tested my model on a holdout group which was defined as the data later than 2005.  I tested the 
above model with both parameters (post 1955 and at bats greater than 10) and (post 1957 and at bats 
greater than 477).  

When I tested my model that included only those that played post 1957 and with greater than 477 
at bats on those that played after 2005, I found that my RMSE decreased from 8.4077 to 8.3377.  
I found that the R squared remained constant at .990.

I also found that my features were normally distributed when I limited at bats to greater than 477 
and only those that played after 1957.

Interpreting my model:
In the test model that included only those players with greater than 477 at bats and those that 
played after 1957, I found that year played had the largest effect on runs.  For every single, double, 
or triple, .33, .54, and 1.29 runs were scored.   For every home run 1.01 runs were scored.  
For every stolen base, walk, or hit by pitch, .31, .32, and .40 runs were scored, respectively.
Finally, both strike outs and intentional walks had a negative effect on runs scored.  For every
strike out, runs decreased by 0.026 and for every intentional walk, runs decreased by 0.55.  
Every feature was statistically significant but year played, triples, and home runs had the largest 
effect on runs scored while strike outs had the smallest effect. 

Notes:
It is impossible to score 1.29 runs for every triple or 1.01 runs for every home run
so the model can still be improved.
'''