# -*- coding: utf-8 -*-
"""
Created on Mon Feb 09 12:03:57 2015

@author: garauste
"""

##########################################################################
##################### Hypotheses for this analysis #######################
##########################################################################
#
# Our hypotheses for this analysis is that Average batting statistics of
# all players on an annualized basis are more effective at predictng the 
# future number of runs for a player in a season than predicting one 
# players performance based on another
#
# Two models will be build in this analysis, one using the playing 
# statistics from the career of barry bonds. The second model will be
# built using the average annual statistics of all players since 1930. 
#
# Both of these models will then be tested out of sample to determine
# which model has the greater predictive power.
#
##########################################################################


# import division from the _future_ release of python
from __future__ import division

import sqlite3
import pandas

## import statsmodels to run the linear regression
# scikit - learn also has a linear model method but the statsmodels version has a more user friendly output

import statsmodels.formula.api as smf

#connect to the baseball database
conn = sqlite3.connect('C:\Users\garauste\Documents\SQLite\lahman2013.sqlite')

## Trying to estimate the number of runs for a player in a given year: Therefore dividing each metric by the number of players in that year to give - also beginning the dataset in 1930 to allow for more observations 
sql = """select yearID, sum(AB)/count(playerID) as average_AtBats, sum(R)/count(playerID) as avg_runs_per_player, sum(H)/count(playerID) as avg_hits_per_player, sum(HR)/count(playerID) as average_HomeRuns, sum(SB)/count(playerID) as avg_stolen_bases_per_player, sum(SO)/count(playerID) as avg_strikeouts, sum(IBB)/count(playerID) as avg_total_intentional_walks
from Batting 
where yearID > 1930
and yearid < 2005
group by yearID
order by yearID ASC"""

sql_individ = """select yearID,AB as At_Bats, R as annual_runs,H as hits, HR as homers,  G as Games, SO as StrikeOuts
from Batting 
where playerID = 'bondsba01'
group by yearID
order by yearID asc"""

df_avg = pandas.read_sql(sql,conn)
df_ind = pandas.read_sql(sql_individ,conn)
conn.close()

## dropping all NaNs from the dataset
df_avg.dropna(inplace = True)
df_ind.dropna(inplace = True)

## Inspecting Data ##
df_ind.tail()

################################################################################
############## Creating some dummy categorical Variables #######################
################################################################################

## According to Wikipedia: Conditioning and diets improved in the 1980s as teams and coaches begin to develop a more specialized approach. A dummy variable will be included to determine whether this supposed change had any statistical impact
df_avg['post_1980'] = 0
df_avg.post_1980[df_avg.yearID>=1980]=1

## Adding in a world war 2 dummy variable ##
df_avg['WW2'] = 0
df_avg.WW2[(df_avg.yearID>=1939)&(df_avg.yearID<=1945)] = 1

## Barry Bonds had multiple surgeries and injuries in 2005 - include a dummy for these
df_ind['BarryGoesToHospital']=0
df_ind.BarryGoesToHospital[df_ind.yearID>=2007] = 1

# starting out with the most obvisious connection -- more runs means more hits 
est_avg = smf.ols(formula='avg_runs_per_player ~ avg_hits_per_player + avg_stolen_bases_per_player+avg_strikeouts + WW2+post_1980 + average_HomeRuns + avg_total_intentional_walks + average_AtBats', data = df_avg).fit()
est_barry = smf.ols(formula = 'annual_runs ~ hits + homers + Games + StrikeOuts + BarryGoesToHospital + At_Bats',data = df_ind).fit()

# Create a y-hat colum in our data frame
df_avg['yhat_avg'] = est_avg.predict(df_avg)
df_ind['yhat_barry'] = est_barry.predict(df_ind)


###########################################################################
### Creating and Plotting the Residuals to check for Heteroskedasticity ###
###########################################################################

df_avg['residuals'] = df_avg.avg_runs_per_player - df_avg.yhat_avg
df_ind['residuals'] = df_ind.annual_runs - df_ind.yhat_barry

plt = df_avg.plot(x='avg_hits_per_player',y='residuals',kind='scatter')
plt = df_ind.plot(x='hits',y='residuals',kind='scatter')

plot = df_avg.plot(x='avg_runs_per_player',y='residuals',kind='scatter')
plot = df_ind.plot(x='annual_runs',y='residuals',kind='scatter')

plot = df_avg.plot(x='average_HomeRuns',y='residuals',kind='scatter')
plot = df_ind.plot(x='homers',y='residuals',kind='scatter')

# There appears to be some multicollinearity between the variables in the dataset. In the individual dataset. This is potentially caused by the limited number of variables in the dataset leading to a poor sample size particularly for the BarryBonds model. 

###########################################################################
################ Examining Model Co-variates and Results ##################
###########################################################################

## Print out Summaries for both models
print est_avg.summary()
## There are a number of interesting takeaways from examining the results of the avg model output.
# Firstly, stolen bases, at bats and intentional walks all have negative coefficients - this
# seems counter-intuitive. Secondly, strikeouts have a positive coefficient which is also 
# counter-intuitive. However we also have a high r-squared value which would suggest that we 
# are over-fitting the model. Both of our categorical variables are not statistically significant
# We will remove some of these variables and re-run the model
print est_barry.summary()
# In the individual model we are seeing far fewer significant variables. The main significant variable is HomeRuns per year. The rest of the coefficients signs are as expected but they are not statistically significant. We will remove strikeouts to see if this has any impact on the other model inputs.

# Re-run the models with StrikeOuts removed from both models #
est_avg = smf.ols(formula='avg_runs_per_player ~ avg_hits_per_player + post_1980 + average_HomeRuns', data = df_avg).fit()
est_barry = smf.ols(formula = 'annual_runs ~ hits + Games + BarryGoesToHospital+At_Bats',data = df_ind).fit()

# Re-examine the outputs to check for changes in coefficients #
print est_avg.summary()
print est_barry.summary()

RMSE_avg = ((((df_avg.residuals)**2).mean())**(1/2))
RMSE_ind = ((((df_ind.residuals)**2).mean())**(1/2))

## RMSE of the average model is reasonable however the RMSE of Barry's model is very high.
print RMSE_avg
print RMSE_ind

## lets understand the percent by which the model deviates from the actuals on average
percent_avg_dev = RMSE_avg/df_avg.avg_runs_per_player.mean()
percent_avg_dev_barry = RMSE_ind/df_ind.annual_runs.mean()
## using string formatting when printing the results
print 'Average Model: average deviation:{0}%'.format(round(percent_avg_dev*100,1 ))
print 'Barry Model: average deviation:{0}%'.format(round(percent_avg_dev_barry*100,1 ))

###########################################################################
########## Testing the Predictive Accuracy of the Model ###################
###########################################################################

#connect to the baseball database
conn = sqlite3.connect('C:\Users\garauste\Documents\SQLite\lahman2013.sqlite')

sql = """select yearID, sum(AB)/count(playerID) as average_AtBats, sum(R)/count(playerID) as avg_runs_per_player, sum(H)/count(playerID) as avg_hits_per_player, sum(HR)/count(playerID) as average_HomeRuns, sum(SB)/count(playerID) as avg_stolen_bases_per_player, sum(SO)/count(playerID) as avg_strikeouts, sum(IBB)/count(playerID) as avg_total_intentional_walks
from Batting 
where yearID > 2005
group by yearID
order by yearID ASC"""

sql_individ = """select yearID,AB as At_Bats, R as annual_runs,H as hits, HR as homers,  G as Games, SO as StrikeOuts
from Batting 
where playerID = 'jeterde01'
group by yearID
order by yearID asc"""

df_post_2005 = pandas.read_sql(sql,conn)
df_jeter = pandas.read_sql(sql_individ,conn)
conn.close()

## dropping all NaNs from the dataset
df_post_2005.dropna(inplace = True)
df_jeter.dropna(inplace = True)

# Reinstiate the dummys for both data sets #
df_post_2005['post_1980'] = 1
df_post_2005['WW2'] = 0

df_jeter['BarryGoesToHospital']=0
df_jeter.BarryGoesToHospital[df_ind.yearID>=2007] = 1

# Estimating the effectiveness of both models on the out of sample dataset
df_post_2005['yhat'] = est_avg.predict(df_post_2005)
df_jeter['yhat'] = est_barry.predict(df_jeter)

# create residuals 
df_post_2005['residuals'] = df_post_2005.avg_runs_per_player - df_post_2005.yhat
df_jeter['residuals'] = df_jeter.annual_runs- df_jeter.yhat

## Calcing RMSE
RMSE_post_2005 = (((df_post_2005.residuals)**2).mean()**(1/2))
RMSE_jeter = (((df_jeter.residuals)**2).mean()**(1/2))

## Percent of Deviation 
percent_avg_dev_post_2005 = RMSE_post_2005/df_post_2005.avg_runs_per_player.mean()
percent_avg_dev_jeter = RMSE_jeter/df_jeter.annual_runs.mean()

print 'average deviation for just hits: {0}%'.format(round(percent_avg_dev_post_2005 *100,1))
print 'average deviation for just hits: {0}%'.format(round(percent_avg_dev_jeter*100,1))
## from the average deviation we can see that our average model has performed well when
# predicting the average runs for players after 2005. However our Barry Bonds model has been
# ineffective when used to predict the number of runs hit by Derek Jeter. This is not a surprising
# result and is what we expected to find

# Let's plot how bad the overfit was #
plt = df_post_2005.plot(x='yearID',y='avg_runs_per_player',kind='scatter')
plt.plot(df_post_2005.yearID,df_post_2005.yhat, color ='blue', linewidth=3)

plt = df_jeter.plot(x='yearID',y='annual_runs',kind='scatter')
plt.plot(df_jeter.yearID,df_jeter.yhat, color ='blue', linewidth=3)

## Finally we can see how our post 2005 data fits the average model quite well
# Whereas the Derek Jeter dataset is a poor fit to the Barry Bonds model