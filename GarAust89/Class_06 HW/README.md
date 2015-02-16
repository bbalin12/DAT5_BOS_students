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
