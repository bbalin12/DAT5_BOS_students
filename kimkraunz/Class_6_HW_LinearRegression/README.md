 __Kim Kraunz__
# Class 6 Homework - Linear Regression


## Introduction
I used the Lahman Baseball Database for all analysis. In this homework I used linear regression to predict the number of runs scored by a player in a year.

I used the following code to pull the data from the SQLite database.

```
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
```

I created categorical data for years.

```
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
```

####Exploratory Statistics

I looked for correlation between variables using Pearson's Coefficient and the mean of medians of the variables.

```
df.corr('pearson')

df.describe()
```

I also plotted histograms and scatter plots of the variables.

```
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
```
![Hist1](https://github.com/bbalin12/DAT5_BOS_students/blob/master/kimkraunz/Class_6_HW_LinearRegression/Hist1.png)

![Hist2](https://github.com/bbalin12/DAT5_BOS_students/blob/master/kimkraunz/Class_6_HW_LinearRegression/Hist2.png)

![Scatter](https://github.com/bbalin12/DAT5_BOS_students/blob/master/kimkraunz/Class_6_HW_LinearRegression/Scatter.png)

I noticed that the majority of variables were skewed to the left.

####Linear Regression

######Model 1
I then created my first model to predict runs.  My model included the following features: yearID, at_bats, hits, doubles, triples, home_runs, stolen_bases, strike_outs, walks

```
import statsmodels.formula.api as smf
from __future__ import division

m1_est = smf.ols(formula = 'runs ~  yearID + at_bats + hits + doubles + triples + home_runs + stolen_bases + strike_outs + walks', data = df).fit()
```
I printed out the summary.

```
print m1_est.summary()

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   runs   R-squared:                       0.967
Model:                            OLS   Adj. R-squared:                  0.967
Method:                 Least Squares   F-statistic:                 1.033e+05
Date:                Tue, 10 Mar 2015   Prob (F-statistic):               0.00
Time:                        11:46:20   Log-Likelihood:                -97324.
No. Observations:               31410   AIC:                         1.947e+05
Df Residuals:                   31400   BIC:                         1.948e+05
Df Model:                           9                                         
================================================================================
                   coef    std err          t      P>|t|      [95.0% Conf. Int.]
--------------------------------------------------------------------------------
Intercept      -45.0460      4.379    -10.288      0.000       -53.628   -36.464
yearID           0.0228      0.002     10.325      0.000         0.018     0.027
at_bats         -0.0148      0.001    -11.941      0.000        -0.017    -0.012
hits             0.3233      0.005     70.166      0.000         0.314     0.332
doubles          0.1916      0.009     21.417      0.000         0.174     0.209
triples          0.7509      0.021     35.793      0.000         0.710     0.792
home_runs        0.6632      0.007     90.940      0.000         0.649     0.678
stolen_bases     0.4187      0.005     86.893      0.000         0.409     0.428
strike_outs     -0.0094      0.002     -4.066      0.000        -0.014    -0.005
walks            0.2302      0.003     86.994      0.000         0.225     0.235
==============================================================================
Omnibus:                     3186.172   Durbin-Watson:                   1.541
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            18822.185
Skew:                           0.307   Prob(JB):                         0.00
Kurtosis:                       6.742   Cond. No.                     2.89e+05
==============================================================================
```

The model had a high R squared - .967

I then determined the residuals so I could plot them and check for heteroskedacity.

```
df['m1_yhat'] = m1_est.predict(df)
df['m1_residuals']= df.runs - df.m1_yhat

plt = df.plot(x='m1_yhat', y='m1_residuals', kind='scatter')
```
![Heteroskedacity](https://github.com/bbalin12/DAT5_BOS_students/blob/master/kimkraunz/Class_6_HW_LinearRegression/Heteroskedacity.png)

I then calculated the RMSE

```
m1_RMSE = (((df.m1_residuals) ** 2).mean() ** (1/2))

print m1_RMSE
print 'average deviation for model 1: {0}'.format(round(m1_RMSE, 4))

average deviation for model 1: 5.3633
```

It seems like a good starting point.  The heteroskedacity is OK but not great.  There are more variant residuals at higher yhats.

######Model 2

I changed the model to include singles instead of hits.  I won't include the summary because the model turned out to be the exact same because singles was derived from hits so was a perfect correlation to it.

######Model 3

I changed the model to include years as a categorical variable since year played will most likely have
an effect on runs.  The categories were pre-1995 (pre-steroid era), from_1995_to_2005 (steroid era),
and post 2005 (hopefully post steroid era).

```
m3_est = smf.ols(formula = 'runs ~ from_1995_to_2005 + post_2005 + at_bats + singles+ doubles + triples + home_runs + stolen_bases + strike_outs + walks', data = df).fit()

print m3_est.summary()

   OLS Regression Results                            
==============================================================================
Dep. Variable:                   runs   R-squared:                       0.983
Model:                            OLS   Adj. R-squared:                  0.983
Method:                 Least Squares   F-statistic:                 1.765e+05
Date:                Tue, 10 Mar 2015   Prob (F-statistic):               0.00
Time:                        11:52:14   Log-Likelihood:                -97187.
No. Observations:               31410   AIC:                         1.944e+05
Df Residuals:                   31400   BIC:                         1.945e+05
Df Model:                          10                                         
=====================================================================================
                        coef    std err          t      P>|t|      [95.0% Conf. Int.]
-------------------------------------------------------------------------------------
Intercept            -0.1939      0.053     -3.674      0.000        -0.297    -0.090
from_1995_to_2005     1.4233      0.073     19.598      0.000         1.281     1.566
post_2005          1.952e-15   1.86e-16     10.494      0.000      1.59e-15  2.32e-15
at_bats              -0.0129      0.001    -10.397      0.000        -0.015    -0.010
singles               0.3197      0.005     69.618      0.000         0.311     0.329
doubles               0.5003      0.008     62.769      0.000         0.485     0.516
triples               1.0746      0.020     53.127      0.000         1.035     1.114
home_runs             0.9814      0.007    139.825      0.000         0.968     0.995
stolen_bases          0.4209      0.005     88.227      0.000         0.412     0.430
strike_outs          -0.0128      0.002     -5.582      0.000        -0.017    -0.008
walks                 0.2310      0.003     87.747      0.000         0.226     0.236
==============================================================================
Omnibus:                     3111.743   Durbin-Watson:                   1.558
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            18030.359
Skew:                           0.299   Prob(JB):                         0.00
Kurtosis:                       6.663   Cond. No.                          nan
==============================================================================

df['m3_yhat'] = m3_est.predict(df)
df['m3_residuals']= df.runs - df.m3_yhat

plt = df.plot(x='m3_yhat', y='m3_residuals', kind='scatter')

m3_RMSE = (((df.m3_residuals) ** 2).mean() ** (1/2))

print 'average deviation for model 1: {0}'.format(round(m1_RMSE, 4))
print 'average deviation for model 2: {0}'.format(round(m2_RMSE, 4))
print 'average deviation for model 3: {0}'.format(round(m3_RMSE, 4))

average deviation for model 1: 5.3633
average deviation for model 2: 5.3633
average deviation for model 3: 5.3398
```
![Heteroskedacity3](https://github.com/bbalin12/DAT5_BOS_students/blob/master/kimkraunz/Class_6_HW_LinearRegression/Heteroskedacity3.png)

The RMSE decreases and the Rsqaured increases slightly by making the years categorical.
The heteroskedacity is still OK but not great.


######Model 4

I changed the model to include intentional walks and hit by pitch, to see if these
rarer events would have an effect on runs.  I also removed at_bats since it is strongly
collinear with other features.

```
m4_est = smf.ols(formula = 'runs ~ from_1995_to_2005 + post_2005 + singles+ doubles + triples + home_runs + stolen_bases + strike_outs + walks + hit_by_pitch + intentional_walks', data = df).fit()

print m4_est.summary()

OLS Regression Results                            
==============================================================================
Dep. Variable:                   runs   R-squared:                       0.984
Model:                            OLS   Adj. R-squared:                  0.984
Method:                 Least Squares   F-statistic:                 1.735e+05
Date:                Tue, 10 Mar 2015   Prob (F-statistic):               0.00
Time:                        11:55:24   Log-Likelihood:                -95971.
No. Observations:               31410   AIC:                         1.920e+05
Df Residuals:                   31399   BIC:                         1.921e+05
Df Model:                          11                                         
=====================================================================================
                        coef    std err          t      P>|t|      [95.0% Conf. Int.]
-------------------------------------------------------------------------------------
Intercept            -0.2089      0.049     -4.298      0.000        -0.304    -0.114
from_1995_to_2005     1.0308      0.070     14.727      0.000         0.894     1.168
post_2005         -2.748e-15    1.8e-16    -15.238      0.000      -3.1e-15 -2.39e-15
singles               0.2790      0.002    140.105      0.000         0.275     0.283
doubles               0.4663      0.007     65.377      0.000         0.452     0.480
triples               1.0727      0.019     55.566      0.000         1.035     1.111
home_runs             1.0298      0.007    151.308      0.000         1.017     1.043
stolen_bases          0.4022      0.005     87.603      0.000         0.393     0.411
strike_outs          -0.0422      0.002    -22.723      0.000        -0.046    -0.039
walks                 0.2674      0.003    100.255      0.000         0.262     0.273
hit_by_pitch          0.2912      0.015     19.884      0.000         0.263     0.320
intentional_walks    -0.5553      0.012    -46.118      0.000        -0.579    -0.532
==============================================================================
Omnibus:                     3088.090   Durbin-Watson:                   1.623
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            18364.108
Skew:                           0.281   Prob(JB):                         0.00
Kurtosis:                       6.703   Cond. No.                     5.41e+08
==============================================================================


df['m4_yhat'] = m4_est.predict(df)
df['m4_residuals']= df.runs - df.m4_yhat

plt = df.plot(x='m4_yhat', y='m4_residuals', kind='scatter')

m4_RMSE = (((df.m4_residuals) ** 2).mean() ** (1/2))

print 'average deviation for model 1: {0}'.format(round(m1_RMSE, 4))
print 'average deviation for model 2: {0}'.format(round(m2_RMSE, 4))
print 'average deviation for model 3: {0}'.format(round(m3_RMSE, 4))
print 'average deviation for model 4: {0}'.format(round(m4_RMSE, 4))

average deviation for model 1: 5.3633
average deviation for model 2: 5.3633
average deviation for model 3: 5.3398
average deviation for model 4: 5.1371
```
![Heteroskedacity4](https://github.com/bbalin12/DAT5_BOS_students/blob/master/kimkraunz/Class_6_HW_LinearRegression/Heteroskedacity4.png)

Adding hit by pitch and intentionally walked and removing at_bats both 
lowered the RMSE and raised the Rsquared.  This dataset is large enough that it can absorb rare events and improve the model.  Again, the heteroskedacity is OK but not great.

######Model 5

Lastly, I checked to see what the effect of at_bats was on the model.  I left out intentional 
walks and hit by pitch as well so I can look for change in the model due to at_bats.

```
m5_est = smf.ols(formula = 'runs ~ from_1995_to_2005 + post_2005 + singles + doubles + triples + home_runs + stolen_bases + strike_outs + walks', data = df).fit()

print m5_est.summary()

OLS Regression Results                            
==============================================================================
Dep. Variable:                   runs   R-squared:                       0.982
Model:                            OLS   Adj. R-squared:                  0.982
Method:                 Least Squares   F-statistic:                 1.954e+05
Date:                Tue, 10 Mar 2015   Prob (F-statistic):               0.00
Time:                        11:58:40   Log-Likelihood:                -97241.
No. Observations:               31410   AIC:                         1.945e+05
Df Residuals:                   31401   BIC:                         1.946e+05
Df Model:                           9                                         
=====================================================================================
                        coef    std err          t      P>|t|      [95.0% Conf. Int.]
-------------------------------------------------------------------------------------
Intercept            -0.3565      0.050     -7.061      0.000        -0.455    -0.258
from_1995_to_2005     1.5374      0.072     21.380      0.000         1.397     1.678
post_2005          2.208e-15   8.48e-17     26.037      0.000      2.04e-15  2.37e-15
singles               0.2770      0.002    134.183      0.000         0.273     0.281
doubles               0.4686      0.007     63.521      0.000         0.454     0.483
triples               1.0470      0.020     52.126      0.000         1.008     1.086
home_runs             0.9649      0.007    140.903      0.000         0.951     0.978
stolen_bases          0.4257      0.005     89.511      0.000         0.416     0.435
strike_outs          -0.0263      0.002    -13.813      0.000        -0.030    -0.023
walks                 0.2279      0.003     86.981      0.000         0.223     0.233
==============================================================================
Omnibus:                     3115.893   Durbin-Watson:                   1.549
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            18008.875
Skew:                           0.301   Prob(JB):                         0.00
Kurtosis:                       6.660   Cond. No.                          nan
==============================================================================

df['m5_yhat'] = m5_est.predict(df)
df['m5_residuals']= df.runs - df.m5_yhat

plt = df.plot(x='m5_yhat', y='m5_residuals', kind='scatter')

m5_RMSE = (((df.m5_residuals) ** 2).mean() ** (1/2))

print 'average deviation for model 1: {0}'.format(round(m1_RMSE, 4))
print 'average deviation for model 2: {0}'.format(round(m2_RMSE, 4))
print 'average deviation for model 3: {0}'.format(round(m3_RMSE, 4))
print 'average deviation for model 4: {0}'.format(round(m4_RMSE, 4))
print 'average deviation for model 5: {0}'.format(round(m5_RMSE, 4))

average deviation for model 1: 5.3633
average deviation for model 2: 5.3633
average deviation for model 3: 5.3398
average deviation for model 4: 5.1371
average deviation for model 5: 5.349
```
![Heteroskedacity5](https://github.com/bbalin12/DAT5_BOS_students/blob/master/kimkraunz/Class_6_HW_LinearRegression/Heteroskedacity5.png)


Model 4 has the highest R squared and the lowest RMSE.  The intentional walks and hit by pitch had a larger effect in model 4 than the at_bats.  I will use model 4 to test whether the linear regression model can accurately predict runs scored.

####Testing

I pulled in the data after 2005 and tested model 4.  I manipulated the data exactly as I had changed it in the training data.

```
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
```

I then used the model to predict the runs.

```
df['m4_yhat'] = m4_est.predict(df)

df['m4_residuals']= df.runs - df.m4_yhat

plt = df.plot(x='m4_yhat', y='m4_residuals', kind='scatter')

m4_RMSE = (((df.m4_residuals) ** 2).mean() ** (1/2))


print 'average deviation for M4 equation on post 2000 data: {0}'.format(
                                            round(m4_RMSE, 4))

average deviation for M4 equation on post 2005 data: 5.0724
```

![Heteroskedacity6](https://github.com/bbalin12/DAT5_BOS_students/blob/master/kimkraunz/Class_6_HW_LinearRegression/Heteroskedacity6.png)

####Conclusion
I still had a relatively low RMSE when I tested the linear regression model using data from 2006 and later.  By modifying the model's features I was able to create a stronger predictive model of runs scored in a year.
