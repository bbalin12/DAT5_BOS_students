SELECT * FROM Master;
SELECT * FROM Batting;
SELECT * FROM Pitching;
SELECT * FROM Fielding;

SELECT playerID, nameGiven, birthYear FROM Master;
SELECT playerID, yearID, TeamID FROM Batting;


SELECT playerID, yearID, TeamID, lgID as league_id, G as games, G_batting as games_as_batter, AB as at_bats, R as Runs, H as Hits 
FROM Batting; 

SELECT playerID, yearID, TeamID, lgID as league_id, G as games, G_batting as games_as_batter, 
AB as at_bats, R as Runs, H as Hits 
FROM Batting
WHERE yearID > 2000; 

SELECT playerID, yearID, TeamID, lgID as league_id, G as games, G_batting as games_as_batter, 
AB as at_bats, R as Runs, H as Hits 
FROM Batting
WHERE yearID > 2000 AND playerID = 'aardsda01';


SELECT playerID, yearID, TeamID, lgID as league_id, G as games, G_batting as games_as_batter, 
AB as at_bats, R as Runs, H as Hits 
FROM Batting
WHERE yearID > 2000 AND playerID in( 'aardsda01', 'abbotpa01');

SELECT playerID, yearID, TeamID, lgID as league_id, G as games, G_batting as games_as_batter, 
AB as at_bats, R as Runs, H as Hits 
FROM Batting
WHERE yearID > 2000 AND playerID = 'aardsda01'
ORDER BY games_as_batter DESC; 

SELECT playerID, yearID, TeamID, lgID as league_id, G as games, G_batting as games_as_batter, 
AB as at_bats, R as Runs, H as Hits 
FROM Batting
WHERE yearID > 2000 AND playerID = 'aardsda01'
ORDER BY league_id ASC, games_as_batter DESC;

SELECT b.playerID, b.yearID, b.TeamID, b.lgID as league_id, b.G as games, b.G_batting as games_as_batter, 
b.AB as at_bats, b.R as Runs, b.H as Hits, m.nameGiven 
FROM Batting b
LEFT JOIN Master m on b.playerID = m.playerID
WHERE b.yearID > 2000 AND b.playerID = 'aardsda01'
ORDER BY games_as_batter DESC;

LEFT JOIN PitchingPost pp on b.playerID = pp.playerID
WHERE b.playerID in( 'aardsda01', 'abbotpa01')
and b.yearID > 2000
and b.yearID < 2010
order by b.yearID desc

SELECT b.playerID, b.yearID,b.teamID, b.G_batting as games_batting, pp.* from Batting b
INNER JOIN PitchingPost pp on b.playerID = pp.playerID
WHERE b.playerID in( 'aardsda01', 'abbotpa01')
and b.yearID > 2000
and b.yearID < 2010
order by b.yearID desc

SELECT b.teamID, b.playerID, t.teamID, t.name from Batting b
LEFT JOIN Teams t ON t.teamID = b.teamID and t.yearID = b.yearID;

SELECT t.name, COUNT(b.playerID)  from Batting b
LEFT OUTER JOIN Teams t ON t.teamID = b.teamID and t.yearID = b.yearID
GROUP BY t.name

SELECT t.name, COUNT(b.playerID) as num_players  from Batting b
LEFT OUTER JOIN Teams t ON t.teamID = b.teamID and t.yearID = b.yearID
GROUP BY t.name
ORDER BY num_players desc

SELECT t.name, COUNT(b.playerID) as num_players  from Batting b
LEFT OUTER JOIN Teams t ON t.teamID = b.teamID and t.yearID = b.yearID
WHERE t.yearID >=1950
GROUP BY t.name
ORDER BY num_players desc

SELECT b.teamID, COUNT(b.playerID) as num_players  from Batting b
LEFT OUTER JOIN Teams t ON t.teamID = b.teamID and t.yearID = b.yearID
WHERE t.yearID >=1950
GROUP BY t.name
ORDER BY num_players desc

SELECT b.playerID,  min(b.yearID) as rookie_year from Batting b
GROUP BY b.playerID;

SELECT b.playerID,  sum(b.G_batting) as total_games_at_bat from Batting b
GROUP BY b.playerID
order by sum(b.G_batting) desc


SELECT DISTINCT playerID FROM Batting;
SELECT COUNT(DISTINCT playerID) FROM Batting;
SELECT COUNT(playerID) FROM Batting;

SELECT CASE WHEN b.G_batting >=20 THEN 1 ELSE 0 END as many_games_batted , b.*
FROM Batting b;

SELECT  playerID, max(yearID) as maxyear from Batting 
GROUP BY playerID;


select sq1.maxyear, b.* from Batting b
LEFT OUTER JOIN 
(SELECT  playerID, max(yearID) as maxyear from Batting 
GROUP BY playerID) sq1
ON b.playerID = sq1.playerID
AND b.yearID = sq1.maxyear;


select sq1.maxyear, b.* from Batting b
LEFT OUTER JOIN 
(SELECT  playerID, max(yearID) as maxyear from Batting 
GROUP BY playerID) sq1
ON b.playerID = sq1.playerID
AND b.yearID = sq1.maxyear
WHERE  sq1.maxyear is not null;

SELECT * FROM
(
select sq1.maxyear, b.* from Batting b
LEFT OUTER JOIN 
(SELECT  playerID, max(yearID) as maxyear from Batting 
GROUP BY playerID) sq1
ON b.playerID = sq1.playerID
AND b.yearID = sq1.maxyear) db
WHERE maxyear is not null;

CREATE TABLE mytable as 
select 
* from
(
select sq1.maxyear, b.* from Batting b
LEFT OUTER JOIN 
(SELECT  playerID, max(yearID) as maxyear from Batting 
GROUP BY playerID) sq1
ON b.playerID = sq1.playerID
AND b.yearID = sq1.maxyear) db
WHERE  maxyear is not null;

select * from mytable
-- once this is done, I can query from the new table as you do with any other:
SELECT * from lastyear;

--Find the player with the most at-bats in a single season.
select 
playerID, max(AB)
from Batting;

--Find the name of the the player with the most at-bats in baseball history.
SELECT
b.nameGiven, max(AB)
FROM
(
select 
playerID, sum(AB) AS AB
from Batting
group by 1
) a
join Master b on a.playerID = b.playerID
;

--Find the average number of at_bats of players in their rookie season
select
avg(AB)
from
(
SELECT b.playerID,  min(b.yearID) as rookie_year, b.AB  from Batting b
GROUP BY b.playerID);


--Find the average number of at_bats of players in their final season for all players born after 1980.
select
avg(AB)
from
(
SELECT b.playerID,  MAX(b.yearID) as rookie_year, m.birthYear, b.AB  from Batting b
join Master m on b.playerID=m.playerID
WHERE m.birthYear >1980
GROUP BY b.playerID)


--Find the average number of at_bats of Yankees players who began their second season at or after 1980.

select
avg(t.AB)
FROM
(select
b1.playerID, min(b1.yearID) as sec_min, b1.AB
from
Batting b1
join 
(
SELECT b.playerID,  min(b.yearID) as rookie_year from Batting b
GROUP BY b.playerID) b2
on b1.playerID=b2.playerID
join Teams t on b1.teamID=t.teamID and t.name='New York Yankees'
where b1.yearID > b2.rookie_year
group by 1
) t
where t.sec_min >=1980

--Among players from universities, Which school has generated the most players
SELECT 
sch.schoolName,
count(distinct m.playerID)
FROM master m
left join SchoolsPlayers s on m.playerID=s.playerID
left join Schools sch on s.schoolID=sch.schoolID
group by 1
order by 2 desc;

--Player that has won the most award
SELECT
m.nameFirst,
m.nameLast,
count(a.awardID),
count(distinct a.awardID)
FROM master m
join AwardsPlayers a on m.playerID=a.playerID
where a.awardID is not null
group by 1,2
order by 3 desc;


--Each team's salary in 2013. New York Highlanders has higest salary
select
t.name,
avg(t1.salary)
FROM 
(
SELECT 
s.teamID,
s.playerID,
s.salary
FROM Salaries s
where s.yearID=2013
) t1
join Teams t on t1.teamID=t.teamID
group by 1
order by 2 desc;

--How many players were born in New England

select
case when m.birthState in ('NH', 'VT', 'MA', 'RI', 'CT', 'ME') then 'New England'
else 'Other' end,
count(distinct m.playerID)
from master m 
group by 1 ;



