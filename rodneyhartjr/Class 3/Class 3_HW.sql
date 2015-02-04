SELECT * FROM MASTER;

SELECT * FROM BATTING;

SELECT playerID, nameGiven, birthYear FROM Master;

SELECT playerID, yearID, TeamID FROM Batting;

--Execerise
SELECT playerID, yearID, TeamID, lgID as league_id, G as games, g_batting as games_as_batter, AB as at_bats, R as Runs, H as Hits
FROM Batting
WHERE yearID > 2000 AND playerID IN ('aardsda01','abbotpa01');

--Execerise
SELECT playerID, yearID, TeamID, lgID as league_id, G as games, g_batting as games_as_batter, AB as at_bats, R as Runs, H as Hits
FROM Batting
WHERE yearID > 2000 AND playerID = 'aardsda01'
ORDER BY games_as_batter DESC;

--Execerise
SELECT playerID, yearID, TeamID, lgID as league_id, G as games, g_batting as games_as_batter, AB as at_bats, R as Runs, H as Hits
FROM Batting
WHERE yearID > 2000 AND playerID = 'aardsda01'
ORDER BY league_id ASC, games_as_batter DESC;

--Execerise
SELECT b.playerID, b.yearID, b.TeamID, b.lgID as league_id, b.G as games, b.G_batting as games_as_batter, 
b.AB as at_bats, b.R as Runs, b.H as Hits, m.nameGiven 
FROM Batting b
LEFT JOIN Master m on b.playerID = m.playerID
WHERE b.yearID > 2000 AND b.playerID = 'aardsda01'
ORDER BY games_as_batter DESC;

--Execerise
SELECT b.playerID, b.yearID,b.teamID, b.G_batting as games_batting, pp.* from Batting b
LEFT JOIN PitchingPost pp on b.playerID = pp.playerID
WHERE b.playerID in( 'aardsda01', 'abbotpa01')
and b.yearID > 2000
and b.yearID < 2010
order by b.yearID desc;

--Execerise
SELECT b.playerID, b.yearID,b.teamID, b.G_batting as games_batting, pp.* from Batting b
INNER JOIN PitchingPost pp on b.playerID = pp.playerID
WHERE b.playerID in( 'aardsda01', 'abbotpa01')
and b.yearID > 2000
and b.yearID < 2010
order by b.yearID desc;

--Execerise
SELECT b.teamID, count(b.playerID) as num_players
from Batting b
LEFT OUTER JOIN Teams t ON t.teamID = b.teamID and t.yearID = b.yearID
GROUP BY b.teamID
ORDER BY num_players desc;

--Exercise
SELECT b.teamID, count(b.playerID) as num_players
from Batting b
LEFT OUTER JOIN Teams t ON t.teamID = b.teamID and t.yearID = b.yearID
WHERE t.yearID >= 1950
GROUP BY b.teamID
ORDER BY num_players desc;

--Exercise
SELECT playerID, min(yearID) FROM batting
group by playerID
order by min(yearID) asc;

--Exercise
SELECT DISTINCT playerID FROM batting;

--Exercise
SELECT (CASE WHEN G_batting >= 20 THEN 1 ELSE 0 END) as many_games_at_bat, b.* FROM batting b;

--Exercise
select sq1.maxyear, b.* from Batting b
LEFT OUTER JOIN
(SELECT  playerID, max(yearID) as maxyear from Batting 
GROUP BY playerID) sq1 ON sq1.playerID = b.playerID AND b.yearID =sq1.maxyear;

--Exercise
select sq1.maxyear, b.* from Batting b
INNER JOIN
(SELECT  playerID, max(yearID) as maxyear from Batting 
GROUP BY playerID) sq1 ON sq1.playerID = b.playerID AND b.yearID =sq1.maxyear;

--Exercise
select sq1.maxyear, b.* from Batting b
LEFT OUTER JOIN
(SELECT  playerID, max(yearID) as maxyear from Batting
GROUP BY playerID) sq1 ON sq1.playerID = b.playerID AND b.yearID =sq1.maxyear
WHERE sq1.maxyear is not null;

--Exercise
CREATE TABLE mytable as
select sq1.maxyear, b.* from Batting b
LEFT OUTER JOIN
(SELECT  playerID, max(yearID) as maxyear from Batting
GROUP BY playerID) sq1 ON sq1.playerID = b.playerID AND b.yearID =sq1.maxyear
WHERE sq1.maxyear is not null

--Exercise
select sq1.maxyear, b.* from Batting b
LEFT OUTER JOIN 
(SELECT  playerID, max(yearID) as maxyear from Batting 
GROUP BY playerID) sq1
ON b.playerID = sq1.playerID
AND b.yearID = sq1.maxyear
WHERE  sq1.maxyear is not null

--HW 1
SELECT m.nameGiven, b.playerID, b.yearID, b.TeamID, b.lgID as league_id, b.G as games, b.G_batting as games_as_batter, 
b.AB as at_bats, b.R as Runs, b.H as Hits
FROM Batting b
LEFT JOIN Master m on b.playerID = m.playerID
ORDER BY AB DESC;

--HW 2
SELECT m.nameGiven, b.playerID, sum(b.AB) as Bats
from Batting b
LEFT OUTER JOIN Master m ON m.playerID = b.playerID
GROUP BY b.playerID
ORDER BY Bats desc;

--HW 3
Create table Rookies_AB_3 as
select ry1.playerID, b.AB as AB, ry1.rookie_year from Batting b
LEFT OUTER JOIN 
(SELECT  playerID, min(yearID) as rookie_year from Batting 
GROUP BY playerID) ry1
ON b.playerID = ry1.playerID
Where b.AB is not null
GROUP By b.playerID;
--HW 3b
select avg(AB) from Rookies_AB_3;


--HW 4
CREATE TABLE old_at_bats as
SELECT b.playerID, m.birthYear, max(b.yearID), b.ab as AB
FROM Batting b
LEFT OUTER JOIN Master m
ON b.playerID = m.playerID
WHERE m.birthYear > 1980 and AB is not null
GROUP BY b.playerID
ORDER BY min(yearID) DESC;
--HW 4a
SELECT avg(AB) from old_at_bats;

--HW 5
Create table NYY_AB_2S as
select ry1.playerID, ry1.teamID, t.name, ry1.AB as AB, ry1.rookie_year from Teams t
LEFT OUTER JOIN 
(SELECT  playerID, teamID, AB, min(yearID) as rookie_year from Batting 
GROUP BY playerID) ry1
ON t.teamID = ry1.teamID
Where ry1.AB is not null
AND (ry1.rookie_year + 1) >= 1980
AND t.name = 'New York Yankees'
--HW 5b
SELECT avg(AB) from NYY_AB_2S;

--HW 6
--saved as python file in class 3 folder






--ExerciseΩ