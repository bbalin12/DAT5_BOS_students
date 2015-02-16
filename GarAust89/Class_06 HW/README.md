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

![Avg_Model_hitsvsresiduals](
