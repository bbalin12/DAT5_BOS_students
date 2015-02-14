---Class Work---
select * from Batting;
select * from Master;

--Find the player with the most at-bats in a single season
SELECT b.yearID, b.playerID, max(AB) as maxab, m.nameGiven
FROM Batting b
LEFT JOIN Master m on b.playerID = m.playerID
WHERE b.yearID 
ORDER BY maxab DESC;
--James Calvin had 716 bats in 2007

--Find the name of the the player with the most at-bats in baseball history
SELECT b.playerID, sum(AB) as sumab, m.nameGiven
FROM Batting b
LEFT JOIN Master m on b.playerID = m.playerID
group BY b.playerID
order by sumab DESC;
--Peter Edward has the most at-bats in baseball history (14053 bats)

--Find the average number of at_bats of players in their rookie season
create table avgtable as 
select b.playerID, min(yearID) as minyear, AB, m.nameGiven
from Batting b
left join master m on b.playerID=m.playerID
group by b.playerID;
select avg(AB) as avg_ab from avgtable;
--58 bats 

--Find the average number of at_bats of players in their final season for all players born after 1980
create table newtable as 
select b.playerID, max(yearID) as maxyear, AB, m.nameGiven, m.birthYear
from Batting b
left join master m on b.playerID=m.playerID
where m.birthYear>1980
group by b.playerID;
select avg(AB) as avg_ab from newtable;
--92 bats 

--Find the average number of at_bats of Yankees players who began their second season at or after 1980
SELECT AVG(second_at_bats) from 
(SELECT y.playerID,y.yearID,second_at_bats
FROM (select playerID,yearID, AB as second_at_bats from Batting B 
WHERE teamID='NYA'
GROUP BY playerID,yearID) y 
INNER JOIN 
(SELECT * FROM (
SELECT playerID,MIN(yearID) AS second_smallest FROM (SELECT playerID, yearID
FROM Batting
EXCEPT
SELECT playerID, MIN(yearID)
FROM Batting
GROUP BY playerID)
GROUP BY playerID)
WHERE second_smallest>=1980) s ON y.playerID=s.playerID and y.yearID=s.second_smallest);
--103 bats

--Novel Questions
--What are the names of the players that played more than 160 games in the most recent year in the National League?
select b.playerID, max(yearID) as maxyear, b.lgID, m.nameGiven, b.G from batting b
left join master m on b.playerID=m.playerID 
where b.lgID='NL' and b.G>160
group by b.playerID
order by max(yearID) desc;
--Starlin DeJesus, Daniel Thomas, Hunter Andrew, and Joseph Daniel

--How many distinct teams are in the batting data set?
select count (DISTINCT teamID) from batting; 
--149

--How many people have had greater than 100 runs in batting data set?
create table runs as
select CASE WHEN R> 100 THEN 1 ELSE 0 END as many_runs, b.* from batting b
where many_runs is not null and ;
select count (DISTINCT playerID) from runs
where many_runs=1;
--748 people 

--Find the average number of runs of NL players in their rookie season using subqueries 
create table runtable as
select * from
(select sq1.minyear, b.* from Batting b
INNER JOIN
(SELECT  playerID, min(yearID) as minyear from Batting 
GROUP BY playerID) sq1 
ON sq1.playerID = b.playerID AND b.yearID =sq1.minyear) dd
WHERE lgID='NL';
select avg(R) from runtable;
--6.56~7 runs