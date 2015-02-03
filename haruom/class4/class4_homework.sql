-- 1. Find the player with the most at-bats in a single season.

SELECT b.yearID, max(b.AB) as at_bats, m.nameGiven 
FROM Batting b
LEFT JOIN Master m on b.playerID = m.playerID
GROUP BY yearID
ORDER BY yearID ASC, at_bats DESC;

-- 2. Find the name of the the player with the most at-bats in baseball history.

SELECT b.playerID, m.nameGiven, SUM(b.AB) as at_bats
FROM Batting b
LEFT JOIN Master m on b.playerID = m.playerID
GROUP BY b.playerID
ORDER BY at_bats DESC
LIMIT 1;

-- 3. Find the average number of at_bats of players in their rookie season.

SELECT ROUND(AVG(sq1.at_bats), 2) AS AVE_at_bats_Rookie FROM Batting b
LEFT OUTER JOIN 
(SELECT b.playerID,  b.AB as at_bats, min(b.yearID) as rookie_year from Batting b
GROUP BY b.playerID) sq1
ON b.playerID = sq1.playerID;

-- 4. Find the average number of at_bats of players in their final season for all players born after 1980.

SELECT ROUND(AVG(sq1.at_bats), 2) AS Last_at_bats_born_after_1980 FROM Batting b
LEFT OUTER JOIN 
(SELECT b.playerID,  b.AB as at_bats, max(b.yearID) as rookie_year, m.birthYear from Batting b
LEFT JOIN Master m ON b.playerID = m.playerID WHERE m.birthYear >1980 GROUP BY b.playerID) sq1
ON b.playerID = sq1.playerID;

-- 5. Find the average number of at_bats of Yankees players who began their second season at or after 1980.

SELECT sq1.first_year, ROUND(AVG(b.AB),2) FROM Batting b
LEFT OUTER JOIN 
(SELECT  playerID, MIN(yearID) as first_year from Batting 
WHERE teamID = 'NYA' GROUP BY playerID) sq1
ON b.playerID = sq1.playerID
AND b.yearID = sq1.first_year
WHERE b.yearID > 1980 AND teamID = 'NYA';

-- Below are original queries
-- Who are the players who has the most homerun in each season?

SELECT b.yearID, max(b.HR) as homerun, m.nameGiven 
FROM Batting b
LEFT JOIN Master m on b.playerID = m.playerID
GROUP BY yearID
ORDER BY yearID ASC, homerun DESC;

--Who is the play who has the biggest number of homerun in the baseball history?

SELECT b.playerID, m.nameGiven, SUM(b.HR) as HomeRun
FROM Batting b
LEFT JOIN Master m on b.playerID = m.playerID
GROUP BY b.playerID
ORDER BY HomeRun DESC
LIMIT 1;

--Who is the player who has the longest season in the baseball history?

SELECT b.playerID, COUNT(b.yearID) AS Seasons, m.nameGiven FROM Batting b
LEFT JOIN Master m ON b.playerID = m.playerID
GROUP BY b.playerID
ORDER BY Seasons DESC;

-- Who is the best runner in the baseball history?

SELECT SUM(R) AS total_run, m.nameGiven FROM Batting b
LEFT JOIN Master m ON b.playerID = m.playerID
GROUP BY b.playerID
ORDER BY total_run DESC
LIMIT 1;

