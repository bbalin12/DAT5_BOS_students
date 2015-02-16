# Gareth Austen Linear Regression Homework for Class 06

### Hypotheses for this Analysis: 
Our Hypotheses is that average batting statistics of all players on an annulazied basis are more effective at predicting 
the number of future runs for a player in a season than attempting to predict one players performance based on another.

Two models will be built in this analysis. The first using the playing statistics from the career of Barry Bonds and the
second using the average annual statistics of all players since 1930

Both of these models will be tested out of sample to determine which model has the greater predictive accuracy.

### Reading in the data

Our two data frames were created using the following sql queries:

```
#connect to the baseball database
conn = sqlite3.connect('C:\Users\garauste\Documents\SQLite\lahman2013.sqlite')

## Trying to estimate the number of runs for a player in a given year: Therefore dividing each 
## metric by the number of players in that year to give - also beginning the dataset 
## in 1930 to allow for more observations 
sql = """select yearID, sum(AB)/count(playerID) as average_AtBats, sum(R)/count(playerID) as avg_runs_per_player, 
sum(H)/count(playerID) as avg_hits_per_player, sum(HR)/count(playerID) as average_HomeRuns, 
sum(SB)/count(playerID) as avg_stolen_bases_per_player, sum(SO)/count(playerID) as avg_strikeouts, 
sum(IBB)/count(playerID) as avg_total_intentional_walks
from Batting 
where yearID > 1930
and yearid < 2005
group by yearID
order by yearID ASC"""

sql_individ = """select yearID,AB as At_Bats, R as annual_runs,H as hits, HR as homers,  
G as Games, SO as StrikeOuts
from Batting 
where playerID = 'bondsba01'
group by yearID
order by yearID asc"""

df_avg = pandas.read_sql(sql,conn)
df_ind = pandas.read_sql(sql_individ,conn)

conn.close()
```

### Dummy Categorical Variable Creation
According to Wikipedia: Conditioning and diets improved in the 1980s as teams and coaches begin to 
develop a more specialized approach. A dummy variable will be included to determine whether this 
supposed change had any statistical impact. A dummy varaible has been included for Barry Bonds long 
injury layoff in 2005 and also for the years during World War 2. The following code was used to create
these variables: 

```
df_avg['post_1980'] = 0
df_avg.post_1980[df_avg.yearID>=1980]=1

## Adding in a world war 2 dummy variable ##
df_avg['WW2'] = 0
df_avg.WW2[(df_avg.yearID>=1939)&(df_avg.yearID<=1945)] = 1

## Barry Bonds had multiple surgeries and injuries in 2005 - include a dummy for these
df_ind['BarryGoesToHospital']=0
df_ind.BarryGoesToHospital[df_ind.yearID>=2007] = 1
```

### Creating two initial models
The following two models were created:

```
# starting out with the most obvisious connection -- more runs means more hits 
est_avg = smf.ols(formula='avg_runs_per_player ~ avg_hits_per_player + avg_stolen_bases_per_player+avg_strikeouts + WW2+post_1980 + average_HomeRuns + avg_total_intentional_walks + average_AtBats', data = df_avg).fit()
est_barry = smf.ols(formula = 'annual_runs ~ hits + homers + Games + StrikeOuts + BarryGoesToHospital + At_Bats',data = df_ind).fit()

# Create a y-hat colum in our data frame
df_avg['yhat_avg'] = est_avg.predict(df_avg)
df_ind['yhat_barry'] = est_barry.predict(df_ind)
```

### Checking for Heteroskedasticity in the Residuals

The first plot below plots the residuals on the y-axis and the average hits per player on the x-axis

![Avg_Model_hitsvsresiduals](https://github.com/GarAust89/DAT5_BOS_students/blob/master/GarAust89/Class_06%20HW/Avg_Model_hitsvsresiduals.png)

While there is quite a lot of variance in the error terms there does not appear to be any specific trends or obvious heteroskedasticity

The next plot shows the residuals on the y-axis and Barry's Bonds annual hits on the x-axis

![Barrys_model_hitsvsresiduals](https://github.com/GarAust89/DAT5_BOS_students/blob/master/GarAust89/Class_06%20HW/Barrys_model_hitsvsresiduals.png)

Examining the same data for Barry's Model we can see that there is a a lot of variance in the error terms for Barry's Data however there are too few data points to be able to identify heteroskedasticity in the residuals. 

### Examining the model output 

**Below are the regression results for the average model:** 

                             OLS Regression Results                            
===============================================================================
Dep. Variable:     avg_runs_per_player   R-squared:                       0.952
Model:                             OLS   Adj. R-squared:                  0.946
Method:                  Least Squares   F-statistic:                     160.0
Date:                 Mon, 16 Feb 2015   Prob (F-statistic):           1.02e-39
Time:                         12:45:23   Log-Likelihood:                -54.202
No. Observations:                   74   AIC:                             126.4
Df Residuals:                       65   BIC:                             147.1
Df Model:                            8                                         
===============================================================================================
                                  coef    std err          t      P>|t|      [95.0% Conf. Int.]
-----------------------------------------------------------------------------------------------
Intercept                      -0.2668      1.055     -0.253      0.801        -2.374     1.840
avg_hits_per_player             0.8226      0.077     10.637      0.000         0.668     0.977
avg_stolen_bases_per_player    -0.3686      0.123     -2.997      0.004        -0.614    -0.123
avg_strikeouts                  0.0370      0.036      1.039      0.303        -0.034     0.108
WW2                             0.0635      0.259      0.245      0.807        -0.453     0.580
post_1980                       0.3475      0.251      1.382      0.172        -0.155     0.850
average_HomeRuns                0.6256      0.136      4.606      0.000         0.354     0.897
avg_total_intentional_walks    -0.3877      0.217     -1.787      0.079        -0.821     0.046
average_AtBats                 -0.0955      0.025     -3.876      0.000        -0.145    -0.046
==============================================================================
Omnibus:                        0.781   Durbin-Watson:                   2.066
Prob(Omnibus):                  0.677   Jarque-Bera (JB):                0.883
Skew:                          -0.218   Prob(JB):                        0.643
Kurtosis:                       2.689   Cond. No.                     2.51e+03
==============================================================================

Interesting, stolen bases, at bats and intentional walks all have negative coefficients - this seems counter intuitive.
StrikeOuts also have a positive coefficient which is also counter-intuitive. R-sqaured and Adjusted R-Sqaured are both 
high however there is a possibility of overfitting in the model. 

**Below are the results for Barry's Model**

                            OLS Regression Results                            
==============================================================================
Dep. Variable:            annual_runs   R-squared:                       0.942
Model:                            OLS   Adj. R-squared:                  0.918
Method:                 Least Squares   F-statistic:                     40.35
Date:                Mon, 16 Feb 2015   Prob (F-statistic):           2.02e-08
Time:                        12:49:13   Log-Likelihood:                -72.535
No. Observations:                  22   AIC:                             159.1
Df Residuals:                      15   BIC:                             166.7
Df Model:                           6                                         
=======================================================================================
                          coef    std err          t      P>|t|      [95.0% Conf. Int.]
---------------------------------------------------------------------------------------
Intercept               0.4422      8.041      0.055      0.957       -16.698    17.582
hits                    0.3020      0.251      1.204      0.247        -0.233     0.837
homers                  0.7616      0.256      2.974      0.009         0.216     1.307
Games                   0.3666      0.212      1.726      0.105        -0.086     0.819
StrikeOuts              0.0114      0.219      0.052      0.959        -0.456     0.478
BarryGoesToHospital    -9.8147      9.609     -1.021      0.323       -30.296    10.667
At_Bats                -0.0357      0.107     -0.335      0.742        -0.263     0.191
==============================================================================
Omnibus:                        0.292   Durbin-Watson:                   1.806
Prob(Omnibus):                  0.864   Jarque-Bera (JB):                0.026
Skew:                          -0.079   Prob(JB):                        0.987
Kurtosis:                       2.941   Cond. No.                     2.89e+03
==============================================================================
