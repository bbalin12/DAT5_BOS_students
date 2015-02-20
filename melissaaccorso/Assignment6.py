# -*- coding: utf-8 -*-
"""
Created on Mon Feb 09 19:26:40 2015

@author: melaccor
"""

#Using the Baseball dataset, build a linear regression model that predicts how many runs a player will have in a given year.
#Begin with more than one possible model; each of which should have at least one 
    #categorical dummy feature and at least two continuous explanatory features.
#Make sure you check for heteroskedasticity in your models.
#Decide whether to include or take out the model's features depending on whether they may be collinear or insignificant.
#Interpret the model's coefficients and intercept.
#Calculate the models' R-squared and in-sample RMSE.
#Make sure to use a holdout group or k-fold CV to calculate out-of-sample RMSE for your model group.
#Decide on the best model to use, and justify why you made that choice.

from __future__ import division
import sqlite3
import pandas
import statsmodels.formula.api as smf

#Model 1
conn=sqlite3.connect('C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite')
sql = """select yearID, sum(R) as total_runs, sum(H) as total_hits, sum(SB) as stolen_bases, lgID
from Batting 
where yearID > 1954
and yearid < 2005
group by yearID
order by yearID ASC"""

df = pandas.read_sql(sql, conn)
conn.close()
df.dropna(inplace = True)

#creating league dummy variable
df['NL'] = 0
df.NL[df.lgID == 'NL'] = 1
df['NL']

#explanatory features:hits, year, and being part of the National League
bin_est = smf.ols(formula='total_runs ~ total_hits + yearID + NL', data=df).fit()
print bin_est.summary()
#R-squared=.96, pretty strong
#Neg Intercept-can't have runs without hits
#coeffs positive which is expected
#Covariates NL and yearID have p-values above 0.05 which would be expected. They aren't significant variables in determining runs (NL is much more nonsignificant: p-val=.998) 
print bin_est.rsquared

# let's create a y-hat column in our dataframe. 
df['yhat'] = bin_est.predict(df)
# now, let's plot how well the model fits the data. 
plt = df.plot(x='total_hits', y='total_runs', kind='scatter')
plt.plot(df.total_hits, df.yhat, color='blue', linewidth=3)
# looks like a strong positive linear relationship.
# let's get a look at the residuals to see if there's heteroskedasticity
df['residuals'] = df.total_runs - df.yhat
plt = df.plot(x='total_hits', y='residuals', kind='scatter')
## there doesn't seem to be any noticable heteroskedasticity. 
# let's calculate RMSE
RMSE = (((df.residuals) ** 2).mean() ** (1/2))
RMSE
# so, on average, the model is off by about 770 runs for each observation.
# lets understand the percent by which the model deviates from actuals on average
percent_avg_dev = RMSE / df.total_runs.mean()
print 'average deviation: {0}%'.format(round(percent_avg_dev*100, 1))
# looks like in-sample deviation is 4.6% on average. 

#looking at model without covariates that are insignificant...NL and yearID

est2 = smf.ols(formula='total_runs ~ total_hits', data=df).fit()
# now, let's print out the results.
print est2.summary()
#Creating a yhat column
df['yhat2'] = est2.predict(df)
#Checking heteroskedasticity 
df['residuals2'] = df.total_runs - df.yhat2
plt = df.plot(x='total_runs', y='residuals2', kind='scatter')
#there doesn't seem to be any clear heteroskedasticity
RMSE2 = (((df.residuals2) ** 2).mean() ** (1/2))
RMSE2
#on average, the model is off about 798 runs for each observation
percent_avg_dev2 = RMSE2 / df.total_runs.mean()
print 'average deviation: {0}%'.format(round(percent_avg_dev2*100, 1))
# looks like in-sample deviation is 4.7% on average. 

##Therefore, taking out insignificant covariates didn't help model's predicitability. 



#Model 2
conn=sqlite3.connect('C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite')
sql = """select yearID, sum(R) as total_runs, sum(SB) as stolen_bases, sum(HR) as homeruns, teamID
from Batting 
where yearID > 1954
and yearid < 2005
group by yearID
order by yearID ASC"""

df3 = pandas.read_sql(sql, conn)
conn.close()
df3.dropna(inplace = True)

#creating Red Sox dummy variable
df3['BOS'] = 0
df3.BOS[df3.teamID == 'BOS'] = 1
df3['BOS']

#explanatory features:homeruns, stolen bases, year, and being on the Boston Red Sox
bin_est3 = smf.ols(formula='total_runs ~ homeruns + stolen_bases + yearID + BOS', data=df3).fit()
print bin_est3.summary()
#R-squared=.98, really strong--more explanatory variables though than last model
#Covariates BOS and yearID have p-values above 0.05 which would be expected. They aren't significant variables in determining runs 
print bin_est3.rsquared

# let's create a y-hat column in our dataframe. 
df3['yhat3'] = bin_est3.predict(df3)
# now, let's plot how well the model fits the data. 
plt = df3.plot(x='homeruns', y='total_runs', kind='scatter')
plt.plot(df3.homeruns, df3.yhat3, color='blue', linewidth=3)
# looks like a strong positive linear relationship.
# let's get a look at the residuals to see if there's heteroskedasticity
df3['residuals3'] = df3.total_runs - df3.yhat3
plt = df3.plot(x='total_runs', y='residuals3', kind='scatter')
#seems to be more heteroskedasticity in this model
# let's calculate RMSE
RMSE3 = (((df3.residuals3) ** 2).mean() ** (1/2))
RMSE3
#on average, the model is off about 565 runs for each observation
# lets understand the percent by which the model deviates from actuals on average
percent_avg_dev3 = RMSE3 / df3.total_runs.mean()
print 'average deviation: {0}%'.format(round(percent_avg_dev3*100, 1))
# looks like in-sample deviation is 3.4% on average. 

#looking at model without covariates that are insignificant...BOS and yearID

est4 = smf.ols(formula='total_runs ~ homeruns + stolen_bases', data=df3).fit()
# now, let's print out the results.
print est4.summary()
#Creating a yhat column
df3['yhat4'] = est4.predict(df3)
#Checking heteroskedasticity 
df3['residuals4'] = df3.total_runs - df3.yhat4
plt = df3.plot(x='total_runs', y='residuals4', kind='scatter')
#seems to have same amount of heteroskedasticity
RMSE4 = (((df3.residuals4) ** 2).mean() ** (1/2))
RMSE4
#on average, the model is off about 572 runs for each observation
percent_avg_dev4 = RMSE4 / df3.total_runs.mean()
print 'average deviation: {0}%'.format(round(percent_avg_dev4*100, 1))
# looks like in-sample deviation is 3.4% on average. 

##Therefore, taking out insignificant covariates didn't make the model's predictablity improve. 

#From the two models I created, Model 2 would be more useful to use in order to predict total runs
#More collinearity in Model 2
#The RMSE is lower for model 2 and in-sample deviation is also lower
#Therefore, this model is off by less run for each observation and the model deviates less from actuals on average


#Out of sample results

conn = sqlite3.connect('C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite')
sql = """select yearID, sum(R) as total_runs, sum(SB) as stolen_bases, sum(HR) as homeruns, teamID, sum(H) as total_hits, lgID
from Batting 
where yearID >= 2005
group by yearID
order by yearID ASC"""
df_post_2005 = pandas.read_sql(sql, conn)
conn.close()

df_post_2005.dropna(inplace = True)

#recreating dummy variable in out of sample model
df_post_2005['BOS'] = 0
df_post_2005.BOS[df_post_2005.teamID == 'BOS'] = 1
df_post_2005['NL'] = 0
df_post_2005.NL[df_post_2005.lgID == 'NL'] = 1

# let's predict both models on the post_2005 data.
df_post_2005['yhat1'] = bin_est.predict(df_post_2005)
df_post_2005['yhat2'] = bin_est3.predict(df_post_2005)

# creating the residuals
df_post_2005['mod1resids'] = df_post_2005.total_runs - df_post_2005.yhat1
df_post_2005['mod2resids'] = df_post_2005.total_runs - df_post_2005.yhat2

# calculating  RMSE
RMSE_mod1 = (((df_post_2005.mod1resids) ** 2).mean() ** (1/2))
RMSE_mod2 =  (((df_post_2005.mod2resids) ** 2).mean() ** (1/2))

print 'average deviation for model1: {0}'.format(round(RMSE_mod1, 4))
print 'average deviation for model2: {0}'.format(round(RMSE_mod2, 4))
                                            
#model one average deviation is 893 while model 2 is 819, so model 1 still does
#worse out of sample compared to Model 2. However, the average deviations are 
#closer in value in out of sample, in which Model 2 doesn't perform as well--still slightly better than Model 1

plt = df_post_2005.plot(x='yearID', y='total_runs', kind='scatter')
plt.plot(df_post_2005.yearID, df_post_2005.yhat1, color='blue', linewidth=3)
plt.plot(df_post_2005.yearID, df_post_2005.yhat2, color='red', linewidth=3)













