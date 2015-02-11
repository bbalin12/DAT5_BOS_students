# -*- coding: utf-8 -*-
"""
Created on Sun Feb 08 16:01:30 2015

@author: jeppley
"""

# importing division from the 'future' release of Python (i.e. Python 3)
from __future__ import division

import sqlite3
import pandas
# importing statsmodels to run the linear regression
# scikit-learn also has a linear model method, but the statsmodels version
# has more user-friendly output.
import statsmodels.formula.api as smf

# connect to the baseball database. 
conn = sqlite3.connect('C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite')
# SQL
sql = """select playerID, yearID, teamID, sum(R) as total_runs, 
sum(AB) times_at_bat,
sum(G) as games, sum(BB) as walks, sum(HBP) as hit_by_pitch,
sum(H) as total_hits, sum(SB) as stolen_bases, sum(SO) as strikeouts, 
sum(IBB) as total_intentional_walks
from Batting 
where yearID > 1954
and yearid < 2005
group by yearID
order by yearID ASC"""

df = pandas.read_sql(sql, conn)
conn.close()

# dropping ALL NaNs in the dataset.
df.dropna(inplace = True)

#creating Yankees dummy variable

df['yankee'] = 0
df.yankee[df.teamID == 'NYA'] = 1

df['yankee']

#MODEL 1


# games, times at bat, walks, and being part of a winning team at the Yankees. 
est1 = smf.ols(formula='total_runs ~ games + times_at_bat + walks + yankee', data=df).fit()
# now, let's print out the results.
print est1.summary()
# notice the R-squared, coefficeints, and p-values. 
# how would you interpret the covariates?
#appears to be strong collinearity, especially since 'times_at_bat' variable
#has a negative coefficient which I would not expect
#being a Yankee is not a significant player in determining runs


print est1.rsquared

#Creating a yhat column
df['yhat'] = est1.predict(df)

plt = df.plot(x='games', y='total_runs', kind='scatter')
plt.plot(df.games, df.yhat, color='blue',
         linewidth=3)
#seems to be a strong positive relationship
    
plt = df.plot(x='times_at_bat', y='total_runs', kind='scatter')
plt.plot(df.times_at_bat, df.yhat, color='blue',
         linewidth=3)
#also strong positive relationship, this looks like source of collinearity
#this one doesn't look to predict as well
         
plt = df.plot(x='walks', y='total_runs', kind='scatter')
plt.plot(df.walks, df.yhat, color='blue',
         linewidth=3)

#this predicts surprisingly well


#Checking heteroskedasticity 
df['residuals'] = df.total_runs - df.yhat
plt = df.plot(x='total_runs', y='residuals', kind='scatter')
#there does appear to be some heteroskedasticity, particularly at top corner

RMSE1 = (((df.residuals) ** 2).mean() ** (1/2))
RMSE1
#on average, the model is off about 704 runs for each observation

percent_avg_dev1 = RMSE1 / df.total_runs.mean()
# notice I'm using string formatting in when I print the results.
print 'average deviation: {0}%'.format(round(percent_avg_dev1*100, 1))
# looks like in-sample deviation is 4.2% on average. 

#looking at model without 'times_at_bat'

# games, walks
est2 = smf.ols(formula='total_runs ~ games + walks', data=df).fit()
# now, let's print out the results.
print est2.summary()

#Creating a yhat column
df['yhat2'] = est2.predict(df)

#Checking heteroskedasticity 
df['residuals2'] = df.total_runs - df.yhat2
plt = df.plot(x='total_runs', y='residuals2', kind='scatter')
#there does appear to be some heteroskedasticity, particularly at top corner

RMSE2 = (((df.residuals2) ** 2).mean() ** (1/2))
RMSE2
#on average, the model is off about 778 runs for each observation

percent_avg_dev2 = RMSE2 / df.total_runs.mean()
# notice I'm using string formatting in when I print the results.
print 'average deviation: {0}%'.format(round(percent_avg_dev2*100, 1))
# looks like in-sample deviation is 4.6% on average. 

#MODEL 2


# connect to the baseball database. 
conn = sqlite3.connect('C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite')
# SQL
sql2 = """select playerID, yearID, teamID, lgID, sum(R) as total_runs, 
sum(AB) times_at_bat,
sum(G) as games, sum(BB) as walks, sum(HBP) as hit_by_pitch,
sum(H) as total_hits, sum(SB) as stolen_bases, sum(SO) as strikeouts, 
sum(IBB) as total_intentional_walks, sum(HR) as homeruns
from Batting 
where yearID > 1954
and yearid < 2005
group by yearID
order by yearID ASC"""

df3 = pandas.read_sql(sql2, conn)
conn.close()

# dropping ALL NaNs in the dataset.
df3.dropna(inplace = True)

df3['nationalleague'] = 0
df3.nationalleague[df3.lgID == 'NL'] = 1


#home runs, intentional walks, and what league 
est3 = smf.ols(formula='total_runs ~ homeruns + total_intentional_walks + nationalleague', data=df3).fit()
# now, let's print out the results.
print est3.summary()
# notice the R-squared, coefficeints, and p-values. 
# how would you interpret the covariates?
#Interesting, as National League dummy seems significant as well as homeruns, intent walks

print est3.rsquared

#Creating a yhat column
df3['yhat3'] = est3.predict(df3)

plt = df3.plot(x='homeruns', y='total_runs', kind='scatter')
plt.plot(df3.homeruns, df3.yhat3, color='blue',
         linewidth=3)
#seems to be a strong positive relationship
    
plt = df.plot(x='total_intentional_walks', y='total_runs', kind='scatter')
plt.plot(df3.total_intentional_walks, df3.yhat3, color='blue',
         linewidth=3)
#this one does not look to predict very well at all
         

#Checking heteroskedasticity 
df3['residuals3'] = df3.total_runs - df3.yhat3
plt = df3.plot(x='total_runs', y='residuals3', kind='scatter')
#the heteroskedasticity is not as prominent in this model

RMSE3 = (((df3.residuals3) ** 2).mean() ** (1/2))
RMSE3
#on average, the model is off about 893 runs for each observation

percent_avg_dev3 = RMSE3 / df.total_runs.mean()
# notice I'm using string formatting in when I print the results.
print 'average deviation: {0}%'.format(round(percent_avg_dev3*100, 1))
# looks like in-sample deviation is 5.3% on average. 

# I would argue to use the second of the two models  for two main reasons:
#it predicts almost as well with more confidence in the coefficients because
#there is less heteroskedasticity in model 2, and there are some interesting
#values in understanding a players home runs, and intentional base walks
#as they give some indication on player mentality


#Out of sample results

conn = sqlite3.connect('C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite')
# creating an object contraining a string that has the SQL query. 
sql = """select playerID, yearID, teamID, lgID, sum(R) as total_runs, 
sum(AB) times_at_bat,
sum(G) as games, sum(BB) as walks, sum(HBP) as hit_by_pitch,
sum(H) as total_hits, sum(SB) as stolen_bases, sum(SO) as strikeouts, 
sum(IBB) as total_intentional_walks, sum(HR) as homeruns
from Batting 
where yearID >= 2005
group by yearID
order by yearID ASC"""

# passing the connection and the SQL string to pandas.read_sql.
hw_post_2005 = pandas.read_sql(sql, conn)
# closing the connection.
conn.close()

# dropping ALL NaNs in the dataset.
hw_post_2005.dropna(inplace = True)

#recreating dummy variable in out of sample model
hw_post_2005['nationalleague'] = 0
hw_post_2005.nationalleague[hw_post_2005.lgID == 'NL'] = 1

# let's predict both modes on the post_2005 data.
hw_post_2005['yhat1'] = est2.predict(hw_post_2005)
hw_post_2005['yhat2'] = est3.predict(hw_post_2005)

# creating the residuals
hw_post_2005['mod1resids'] = hw_post_2005.total_runs - hw_post_2005.yhat1
hw_post_2005['mod2resids'] = hw_post_2005.total_runs - hw_post_2005.yhat2

# calculating  RMSE
RMSE_mod1 = (((hw_post_2005.mod1resids) ** 2).mean() ** (1/2))
RMSE_mod2 =  (((hw_post_2005.mod2resids) ** 2).mean() ** (1/2))

print 'average deviation for model1: {0}'.format(
                                            round(RMSE_mod1, 4))

print 'average deviation for model2: {0}'.format(
                                            round(RMSE_mod2, 4))
                                            
#model one average deviation is 726, model 2 is 501, so while model 1 seemed
#to do worse in sample, out of sample it stayed more steady whereas model 2
#deviated a bit much for comfort from in sample

plt = hw_post_2005.plot(x='yearID', y='total_runs', kind='scatter')
plt.plot(hw_post_2005.yearID, hw_post_2005.yhat1, color='blue',
         linewidth=3)
plt.plot(hw_post_2005.yearID, hw_post_2005.yhat2, color='red',
         linewidth=3)                                            







