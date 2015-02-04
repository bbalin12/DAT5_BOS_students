--1. Find the player with the most at-bats in a single season.
select yearID, playerID,sum(AB) as max_at_bats FROM Batting
group by yearID,playerID
order by max_at_bats DESC
--rollijii01 in 2007 with 716 max_at_bats

--2. Find the name of the the player with the most at-bats in baseball history.
select m.nameGiven, m.nameLast,b.playerID, b.total_at_bats FROM Master m LEFT JOIN (select playerID,sum(AB) as total_at_bats FROM Batting
group by playerID
order by total_at_bats DESC LIMIT 1) b where m.playerID = b.playerID
--roseoe01 or Peter Edward Rose at 14,053 total at bats

--3.Find the average number of at_bats of players in their rookie season.
select avg(rookie_at_bats) from (
	select playerID,min(yearID),AB as rookie_at_bats 
	from Batting B
group by playerID);

--Total average at bats for rookies in their rookie season is 58

--4. Find the average number of at_bats of players in their final season for all players 
--born after 1980.
select avg(final_at_bats) from (
	select playerID, max(yearID) AS final_season,AB as final_at_bats 
	from Batting 
	where playerID in (
		select distinct playerID from Master where birthYear>1980
		)
group by playerID);
--Titak average for players in their final season bor after 1980 is 92

--5. Find the average number of at_bats of Yankees players who began their second season
--at or after 1980.
----second season is not the second smallest year after their rookie,
----thus first identify rookie year and remove it
SELECT AVG(second_at_bats) from 
	(SELECT y.playerID,y.yearID,second_at_bats FROM 
		(SELECT playerID,yearID,AB as second_at_bats from Batting B where teamID in (
				SELECT DISTINCT teamID from Teams
				where name = 'New York Yankees')
		GROUP BY playerID,yearID) y INNER JOIN 
	(SELECT * FROM (
		SELECT playerID,MIN(yearID) AS second_smallest FROM (SELECT playerID, yearID
		      FROM Batting
		      EXCEPT
		      SELECT playerID, MIN(yearID)
		      FROM Batting
		      GROUP BY playerID)
		GROUP BY playerID)
	WHERE second_smallest>=1980) s ON y.playerID=s.playerID and y.yearID=s.second_smallest);
--Total average at bats for yankee players in teir second season is 103

--4 Novel Questions

--Create full, working queries to answer at least four novel questions you have about the dataset using the following concepts:

--USED - The WHERE clause
--USED - ORDER BY
--USED - LEFT JOIN
--USED - GROUP BY
--USED - SELECT DISTINCT
--USED - Subqueries 

--USED - IS NOT NULL
--USED - INNER JOIN
--USED - CASE statements

--What are the Top 5 teams for at bats in 2013?
SELECT teamABs.*,teamNames.name FROM 
(SELECT teamID,SUM(AB) AS team_at_bats FROM Batting B
WHERE yearID = 2013
GROUP BY teamID
) teamABs
LEFT JOIN
(SELECT DISTINCT teamID,name FROM Teams
WHERE yearID=2013) teamNames on teamABs.teamID = teamNames.teamID
ORDER BY team_at_bats DESC;

--The Top 5 teams for at bats in 2013 are:
----Detroit Tigers (DET): 5,735
----Arizona Diamondbacks (ARI): 5,676
----Boston Red Sox (BOS): 5,651
----Baltimore Orioles (BAL): 5,620
----Colorado Rockies (COL): 5,599

--Who are the Boston Red Sox players with G_batting more than 100 in 2013?
SELECT p.playerID,m.nameGiven,m.nameLast,p.G_batting FROM 
(SELECT playerID, G_batting FROM (SELECT playerID,teamID, yearID, G_batting,CASE WHEN G_batting>=100 THEN 1 ELSE 0 END 
	as many_games_at_bat
	FROM Batting B 
	WHERE yearID=2013 and teamID='BOS')
WHERE many_games_at_bat = 1) p
INNER JOIN (SELECT DISTINCT playerID,nameGiven,nameLast FROM Master) m
ON p.playerID = m.playerID
ORDER BY p.G_batting DESC;

--The players are:
----pedrodu01	Dustin Luis	Pedroia	160
----napolmi01	Michael Anthony	Napoli	139
----ortizda01	David Americo	Ortiz	137
----ellsbja01	Jacoby McCabe	Ellsbury	134
----navada01	Daniel James	Nava	134
----drewst01	Stephen Oris	Drew	124
----victosh01	Shane Patrick	Victorino	122
----saltaja01	Jarrod Scott	Saltalamacchia	121
----gomesjo01	Jonny Johnson	Gomes	116

--Where are all the IS NOT NULLs for g_batting coming from?
SELECT playerID,yearID,G,AB,G_batting FROM Batting 
WHERE AB is not NULL and G_batting is NULL 
----They appear to come from players in the year 2012

--Look at how baseball has grown over the years by looking at the number of teams each year
select yearID,count(teamID) from Teams
group by yearID
order by yearID

--The number of teams have increased from 9 in 1871 to 30 teams in 2013