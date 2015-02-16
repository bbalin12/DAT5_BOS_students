# Gareth Austen KNN Homework for Class 05

Step 1 was to pull data from the lahman 2013 baseball dataset. Initally tried to use a large query to pull data from all
four tables simultaneously using the following SQL query:

```
select a.playerID, a.inducted, f.Games, f.Hits, f.At_Bats, f.Homers, f.Pitcher_Ws, f.Pitcher_ShutOuts,
f.Pitcher_StrikeOuts, f.Pitcher_Earned_Run_Avg, f.Field_Position, f.Field_Errors from HallOfFame a 
left outer join 
(
select b.G as Games, b.H as Hits, b.AB as At_Bats, b.HR as Homers, b.playerID, e.Pitcher_Ws, e.Pitcher_ShutOuts,
e.Pitcher_StrikeOuts, e.Pitcher_Earned_Run_Avg,e.Field_Position, e.Field_Errors  from Batting b
left outer join 
(
select c.playerID, c.W as Pitcher_Ws, c.SHO as Pitcher_ShutOuts, c.SO as Pitcher_StrikeOuts, c.ERA as Pitcher_Earned_Run_Avg, 
d.Pos as Field_Position, d.E as Field_Errors from Pitching c left outer join Fielding d on c.playerID = d.playerID
) e 
on b.playerID = e.playerID) f
on a.playerID = f.playerID
where yearID<2000;
```

However, when I dropepd the NA rows from the pandas data frame we were left with a very small number of datapoints. 
I believe this is because of the crossover between Batters and Pitchers. Therefore I decided to create individual models
for pitchers and batters. 
