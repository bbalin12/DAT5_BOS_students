# -*- coding: utf-8 -*-
"""
Created on Sat Feb 07 21:32:21 2015

@author: Teresa
"""

# importing division from the 'future' release of Python (i.e. Python 3)
from __future__ import division

import sqlite3
import pandas
import numpy
# importing statsmodels to run the linear regression
# scikit-learn also has a linear model method, but the statsmodels version
# has more user-friendly output.
import statsmodels.formula.api as smf

# connect to the baseball database. 
DATABASE = r'C:\Users\Teresa\Documents\GA Data Science\lahman2013.sqlite'
conn = sqlite3.connect(DATABASE)
# SQL
sql = """select yearID, 
avg(G) as games,
avg(G_batting) as games_as_batter,
avg(AB) as at_bats,
avg(R) as runs, 
avg(H) as hits, 
--avg(2B) as doubles,
--avg(3B) as triples,
avg(HR) as home_runs,
avg(RBI) as runs_batted_in,
avg(SB) as stolen_bases,
avg(CS) as caught_stealing,
avg(BB) as base_on_balls,
avg(SO) as strikeouts, 
avg(IBB) as intentional_walks,
avg(HBP) as hit_by_pitch,
avg(SH) as sacrifice_hits,
avg(SF) as sacrifice_flies,
avg(GIDP) as grounded_into_double_plays
from Batting 
where yearID > 1954
and yearid < 2005
group by yearID
order by yearID ASC"""

df = pandas.read_sql(sql, conn)
conn.close()

# dropping ALL NaNs in the dataset.
df.dropna(inplace = True)      

df.head()

#lets start with a model with EVERYTHING
EQUATION = ('runs ~ games + games_as_batter + at_bats + hits + \
            home_runs + runs_batted_in + stolen_bases + caught_stealing + \
            base_on_balls + strikeouts + intentional_walks + hit_by_pitch + \
            sacrifice_hits + sacrifice_flies + grounded_into_double_plays')


est = smf.ols(formula=EQUATION, data=df).fit()

print est.summary()

#We see that runs_batted_in is the best predictor among all variabes
#This makes sense as a runs batted in assumes hat a run has happened.
#Thus our first model is:
MODEL1 = 'runs ~ runs_batted_in'
m1 = smf.ols(formula=MODEL1, data=df).fit()
print m1.summary()
#We see that this has a variance of 0.997 and a p-value of less 0 so it explains alot of variance and it's effect is not random
#let's graph the equation
# let's create a y-hat column in our dataframe. 
df['yhat'] = m1.predict(df)

# now, let's plot how well the model fits the data. 
plt = df.plot(x='runs_batted_in', y='runs', kind='scatter')
plt.plot(df.runs_batted_in, df.yhat, color='blue',
         linewidth=3)
#we see that the fit is almost exact

# let's get a look at the residuals to see if there's heteroskedasticity
df['residuals'] = df.runs - df.yhat
plt = df.plot(x='yhat', y='residuals', kind='scatter')
## there doesn't seem to be any noticable clear heteroskedasticity. 

# let's calculate RMSE -- notice you use two multiply signs for exponenets
# in Python
RMSE = (((df.residuals) ** 2).mean() ** (1/2))
# so, on average, the model is off by 0.144 runs for each observation.

# lets understand the percent by which the model deviates from actuals on average
percent_avg_dev = RMSE / df.runs.mean()
# notice I'm using string formatting in when I print the results.
print 'average deviation: {0}%'.format(round(percent_avg_dev*100, 1))
# looks like in-sample deviation is 0.7% on average. 

#The next question is can we improve this model?
#let's first try to add another continuous variable
#take a look at it's correlations with other variables
#we see that it's strongly correlated with many other variables so there may be 
#alot of multicollinearity, thus we want the additional variable to add value
#but not be very correlatd to runs_batted in
df.corr()['runs_batted_in']

#We see that the least correlated positive variables are sacrifice_hits
M1_V2 = 'runs ~ runs_batted_in + games + games_as_batter + intentional_walks \
+ strikeouts + intentional_walks + hit_by_pitch + sacrifice_hits'
m1v2 = smf.ols(formula=M1_V2, data=df).fit()
print m1v2.summary()
# we see that runs_batted_in, strikeouts, hit_by_pitch and sacrifice_hits are significant
M1_V2 = 'runs ~ runs_batted_in + strikeouts + hit_by_pitch + sacrifice_hits'
m1v2 = smf.ols(formula=M1_V2, data=df).fit()
print m1v2.summary()
# let's get a look at the residuals to see if there's heteroskedasticity
df['m1v2_yhat'] = m1v2.predict(df)
df['m1v2_res'] = df.runs - df.m1v2_yhat
plt = df.plot(x='m1v2_yhat', y='m1v2_res', kind='scatter')
m1v2_RMSE = (((df.m1v2_res) ** 2).mean() ** (1/2))
# so, on average, the model is off by 0.06 runs for each observation which is an improvement over just runs_batted_in.

# lets understand the percent by which the model deviates from actuals on average
m1v2_percent_avg_dev = m1v2_RMSE / df.runs.mean()
# notice I'm using string formatting in when I print the results.
print 'average deviation: {0}%'.format(round(m1v2_percent_avg_dev*100, 1))
# looks like in-sample deviation is 0.3% on average.
#looks like deviation and RMSE improve but not by much



plt = df.plot(x='yearID', y='runs', kind='scatter')
plt.plot(df.yearID, df.yhat, color='blue',
         linewidth=1)
plt.plot(df.yearID, df.m1v2_yhat, color='red',
linewidth=1)

#we see in the graph that the deviation in prediction between m1 and m1v2 
#occurs in  the late years specifically post 2000

#note that we did not account for annual effects, let's try to add it
#we see that there's a general upward trend before 1980 of runs with years and a downward trend after 1980
#thus let' create a varible of before_1980, after_1980

# First, we encode a dummy feature for years after 1980. 
df['post_1980'] = 0
df.post_1980[df.yearID>1980] = 1
# do we need to create another dummy feaure for all the other years? 

BIN_EQ = 'runs ~ post_1980'
# let's run the formula.
bin_est = smf.ols(formula=BIN_EQ, data=df).fit()
print bin_est.summary()
# interpret the results for me, please.

# lets plot these predictions against actuals
df['bin_yhat'] = bin_est.predict(df)
plt = df.plot(x='yearID', y='runs', kind='scatter')
plt.plot(df.yearID, df.bin_yhat, color='blue',
         linewidth=3)


M1_V3 = 'runs ~ post_1980 + runs_batted_in + strikeouts + hit_by_pitch + sacrifice_hits'

# let's combine all three factors together: total hits, stolen bases, and year.
m1v3 = smf.ols(formula=M1_V3, data=df).fit()
print m1v3.summary()

#we see that the categorical variable is just slightly significant at 0.03, can we improve it?
#lets go by two decade intervals
df['post_1990'] = 0
df.post_1990[df.yearID>1990] = 1
df['from_1970_to_1990'] = 0
df.from_1970_to_1990[(df.yearID>1970) & (df.yearID<=1990)] = 1


BIN_EQ = 'runs ~ post_1990+from_1970_to_1990'
# let's run the formula.
bin_est = smf.ols(formula=BIN_EQ, data=df).fit()
print bin_est.summary()
# interpret the results for me, please.

# lets plot these predictions against actuals
df['bin_yhat'] = bin_est.predict(df)
plt = df.plot(x='yearID', y='runs', kind='scatter')
plt.plot(df.yearID, df.bin_yhat, color='blue',
         linewidth=3)


M1_V4 = 'runs ~ post_1990 + from_1970_to_1990 + runs_batted_in + strikeouts + hit_by_pitch + sacrifice_hits'

# let's combine all three factors together: total hits, stolen bases, and year.
m1v4 = smf.ols(formula=M1_V4, data=df).fit()
print m1v4.summary()
#we see that neither of these date binaries are significant so go back to the pre/post 1980 model

## let's caclulate residuals and RMSE. 
df['m1v3_yhat'] = m1v3.predict(df)
df['m1v3_residuals'] = df.runs - df.m1v3_yhat

m1v3_RMSE = (((df.m1v3_residuals) ** 2).mean() ** (1/2))

print 'average deviation for m1v2 equation: {0}'.format(
                                            round(m1v3_RMSE, 4))

# lets understand the percent by which the model deviates from actuals on average
m1v3_percent_avg_dev = m1v3_RMSE / df.runs.mean()
# notice I'm using string formatting in when I print the results.
print 'average deviation: {0}%'.format(round(m1v3_percent_avg_dev*100, 1))
# looks like in-sample deviation is 0.3% on average.
#looks like deviation and RMSE improve but not by much

#looks like predictive power didn't improve tha much

## RMSe looks better.  Is it really more predictive?
# let's plot the fit of just hits and the full equation.
plt = df.plot(x='yearID', y='runs', kind='scatter')
plt.plot(df.yearID, df.yhat, color='blue',
         linewidth=1)
plt.plot(df.yearID, df.m1v2_yhat, color='red',
         linewidth=1)
plt.plot(df.yearID, df.m1v3_yhat, color='black',
         linewidth=1)

#we see that adding the categorical variable for years didn't seem to affect in sample predictive power that much..
#we will compare the out of sample effect when we have other models

# let's look at data after 2005.
DATABASE = r'C:\Users\Teresa\Documents\GA Data Science\lahman2013.sqlite'
conn = sqlite3.connect(DATABASE)
# creating an object contraining a string that has the SQL query. 
sql = """select yearID, 
avg(G) as games,
avg(G_batting) as games_as_batter,
avg(AB) as at_bats,
avg(R) as runs, 
avg(H) as hits, 
--avg(2B) as doubles,
--avg(3B) as triples,
avg(HR) as home_runs,
avg(RBI) as runs_batted_in,
avg(SB) as stolen_bases,
avg(CS) as caught_stealing,
avg(BB) as base_on_balls,
avg(SO) as strikeouts, 
avg(IBB) as intentional_walks,
avg(HBP) as hit_by_pitch,
avg(SH) as sacrifice_hits,
avg(SF) as sacrifice_flies,
avg(GIDP) as grounded_into_double_plays
from Batting 
where yearID >=2005
group by yearID
order by yearID ASC"""

# passing the connection and the SQL string to pandas.read_sql.
df_post_2005 = pandas.read_sql(sql, conn)
# closing the connection.
conn.close()

# re-create the dummy features for the new data.
df_post_2005['post_1980'] = 1

# let's predict both modes on the post_2005 data.
df_post_2005['yhat'] = m1.predict(df_post_2005)
df_post_2005['m1v2_yhat'] = m1v2.predict(df_post_2005)
df_post_2005['m1v3_yhat'] = m1v3.predict(df_post_2005)

# creating the residuals
df_post_2005['m1_residuals'] = df_post_2005.runs - df_post_2005.yhat
df_post_2005['m1v2_residuals'] = df_post_2005.runs - df_post_2005.m1v2_yhat
df_post_2005['m1v3_residuals'] = df_post_2005.runs - df_post_2005.m1v3_yhat

# calculating  RMSE
RMSE_m1 = (((df_post_2005.m1_residuals) ** 2).mean() ** (1/2))
RMSE_m1v2 =  (((df_post_2005.m1v2_residuals) ** 2).mean() ** (1/2))
RMSE_m1v3 =  (((df_post_2005.m1v3_residuals) ** 2).mean() ** (1/2))
print 'average deviation for m1 equation: {0}'.format(
                                            round(RMSE_m1, 4))

print 'average deviation for m1v2 hits: {0}'.format(
                                            round(RMSE_m1v2, 4))

print 'average deviation for m1v3 hits: {0}'.format(
                                            round(RMSE_m1v3, 4))
# what does this show you?  
# We were OVERFITTING our data by adding the categories!
                                            
# lets plot how bad the overfit was.
plt = df_post_2005.plot(x='yearID', y='runs', kind='scatter')
plt.plot(df_post_2005.yearID, df_post_2005.yhat, color='blue',
         linewidth=1)
plt.plot(df_post_2005.yearID, df_post_2005.m1v2_yhat, color='red',
         linewidth=1)
plt.plot(df_post_2005.yearID, df_post_2005.m1v3_yhat, color='yellow',
         linewidth=1)

#It appears that deviaion begins to occur after 2005, still the model with the categories isn't too bad

#Now because runs_batted_in implies already knowing the number of runs it may not be a realistic predictor variable
#let's exclude it and run the total model again
#lets start with a model with EVERYTHING
EQUATION = ('runs ~ games + games_as_batter + at_bats + hits + \
            home_runs + stolen_bases + caught_stealing + \
            base_on_balls + strikeouts + intentional_walks + hit_by_pitch + \
            sacrifice_hits + sacrifice_flies + grounded_into_double_plays')


est = smf.ols(formula=EQUATION, data=df).fit()

print est.summary()

#We see that at_bats, hits, home_run, base_on_balls, and grounded_into_double_plays 
#are significant at the 5% level
#at_bats   coefficient:-0.09 p-value:0
#hits      coefficient:0.69 p-value:0
#home_runs coefficient:0.97 p-value:0
#caught_stealing coefficient:-0.57 p-value 0.028
#base_on_balls   coefficient:0.35 p-value:0
#grounded_into_double_plays   coefficient:-0.64 p-value:0




EQUATION = ('runs ~ at_bats + hits + \
            home_runs + caught_stealing + \
            base_on_balls + grounded_into_double_plays')
            
est = smf.ols(formula=EQUATION, data=df).fit()

print est.summary()

#caught_stealing is now insignificant so remove it

EQUATION = ('runs ~ at_bats + hits + \
            home_runs + \
            base_on_balls + grounded_into_double_plays')
            
est = smf.ols(formula=EQUATION, data=df).fit()

print est.summary()

#Despite the all variables being significant there is still high multicollinearity
# so i am going to examine the correlations of other variables with hits


df.corr()['hits']

EQUATION = ('runs ~ games_as_batter + hits + \
            home_runs + hit_by_pitch + sacrifice_hits')
            
est = smf.ols(formula=EQUATION, data=df).fit()

print est.summary()

#hit_by_pitch and sacrifice_hits are not significant

EQUATION = ('runs ~ games_as_batter + hits + \
            home_runs')
            
est = smf.ols(formula=EQUATION, data=df).fit()

print est.summary()

MODEL2 = EQUATION
m2 = smf.ols(formula=MODEL2, data=df).fit()
print m2.summary()
#We see that this has a variance of 0.986 and a p-value of less 0 so it explains alot of variance and it's effect is not random
#let's graph the equation
# let's create a y-hat column in our dataframe. 
df['m2_yhat'] = m2.predict(df)

# now, let's plot how well the model fits the data. 
plt = df.plot(x='hits', y='runs', kind='scatter')
plt.plot(df.hits, df.m2_yhat, color='blue',
         linewidth=3)
#its a bit of a mess but appears to fit well..instead maybe plot over time?

plt = df.plot(x='yearID', y='runs', kind='scatter')
plt.plot(df.yearID, df.m2_yhat, color='blue',
         linewidth=3)
#looking at the model over time it appears to fit quite well

# let's get a look at the residuals to see if there's heteroskedasticity
df['m2_residuals'] = df.runs - df.m2_yhat
plt = df.plot(x='m2_yhat', y='m2_residuals', kind='scatter')
## there doesn't seem to be any noticable clear heteroskedasticity. 

# let's calculate RMSE -- notice you use two multiply signs for exponenets
# in Python
RMSE_m2 = (((df.m2_residuals) ** 2).mean() ** (1/2))
# so, on average, the model is off by 0.303 runs for each observation.

# lets understand the percent by which the model deviates from actuals on average
percent_avg_dev_m2 = RMSE_m2 / df.runs.mean()
# notice I'm using string formatting in when I print the results.
print 'average deviation: {0}%'.format(round(percent_avg_dev_m2*100, 1))
#average deviation is 1.5%

#add date variable
df.head()

EQUATION