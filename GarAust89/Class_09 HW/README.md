# Gareth Austen Homework for Class 09 - Data Manipulation

#### Introduction
I pulled data from the SQLite database using the below query. I decided to pull in the 
state that each player was born in to determine if it had any impact on whether a player 
was inducted into the Hall of Fame. 

```
select coalesce(a.nameFirst ,"") || coalesce(a.nameLast,"") as PlayerName, hfi.inducted,
a.playerID, a.birthState, b.*, c.*,d.*
from hall_of_fame_inductees hfi
left outer join master a
on a.playerID = hfi.playerID
left outer join 
(
select sum(b.G) as Games, sum(b.H) as Hits, sum(b.AB) as At_Bats, sum(b.HR) as Homers,
b.playerID from Batting b group by b.playerID
) b
on hfi.playerID = b.playerID
left outer join 
(
select d.playerID, sum(d.A) as Fielder_Assists, sum(d.E) as Field_Errors, 
sum(d.DP) as Double_Plays from Fielding d group by d.playerID 
) c
on hfi.playerID = c.playerID
left outer join
(
select b.playerID, sum(b.ERA) as Pitcher_Earned_Run_Avg, sum(b.W) as Pitcher_Ws,
sum(b.SHO) as Pitcher_ShutOuts, sum(b.SO) as Pitcher_StrikeOuts,
sum(b.HR) as HR_Allowed, sum(b.CG) as Complete_Games 
from Pitching b 
group by b.playerID
) d 
on hfi.playerID = d.playerID;
```

