-- jonblum
-- 2015-01-29
-- datbos05
-- class 4 hw
-- (sqlite)


-- 1. Find the player with the most at-bats in a single season.
SELECT m.nameLast, m.nameFirst, SUM(b.AB) as at_bats, b.yearID AS year
FROM Batting b
INNER JOIN Master m ON b.playerID = m.playerID
GROUP BY b.playerID, b.yearID -- account for multiple stints in one year: SUM(AB) by year
ORDER BY at_bats DESC
LIMIT 1;
-- Jimmy Rollins: 716 at-bats in 2007


-- 2. Find the name of the player with the most at-bats in baseball history.
SELECT m.nameLast, m.nameFirst, SUM(b.AB) as total_at_bats
FROM Batting b
INNER JOIN Master m ON b.playerID = m.playerID
GROUP BY b.playerID
ORDER BY total_at_bats DESC
LIMIT 1;
-- Pete Rose: 14,053 total at-bats


-- 3. Find the average number of at-bats of playesr in their rookie season.
SELECT AVG(rookie_year_at_bats) FROM
        (SELECT b.playerID, b.yearID, ry.rookie_year, SUM(b.AB) as rookie_year_at_bats
        FROM Batting b
        INNER JOIN
                (SELECT b.playerID, MIN(b.yearID) as rookie_year
                FROM Batting b
                GROUP BY b.playerID) ry
        ON b.playerID = ry.playerID AND b.yearID = ry.rookie_year
        GROUP BY b.playerID, b.yearID); --  account for multiple stints in one year: SUM(AB) by year
-- 59.747 at-bats


-- 4. Find the average number of at_bats of players in their final season for all players born after 1980.
-- (Assuming final == most recent, since most are still playing)
SELECT AVG(last_year_at_bats) FROM
	(SELECT b.playerID,ly.nameFirst, ly.nameLast,ly.birthYear, b.yearID, SUM(b.AB) as last_year_at_bats
	FROM Batting b
	INNER JOIN
		(SELECT b.playerID, m.nameFirst, m.nameLast, MAX(b.yearID) as last_year, m.birthYear
		FROM Batting b
		INNER JOIN Master m ON b.playerID = m.playerID
		WHERE m.birthYear > 1980
		GROUP BY b.playerID) ly
	ON b.playerID = ly.playerID AND b.yearID = ly.last_year
	GROUP BY b.playerID, b.yearID); --  account for multiple stints in one year: SUM(AB) by year
-- 94.435 at-bats


-- 5. Find the average number of at_bats of Yankees players who began their second season at or after 1980.
-- Ambiguous - assuming 'second season with the Yankees'.  Assuming 'average yankee career at-bats'.
-- Second season is the min of what's left when you exclude the min
SELECT AVG(yankeeCareerAtBats) FROM
        (SELECT b.playerID, SUM(b.AB) as yankeeCareerAtBats
        FROM Batting b
        INNER JOIN
                (SELECT b.playerID, b.teamID, MIN(b.yearID) as secondYankeeYear
                FROM Batting b
                INNER JOIN
                        (SELECT b.playerID, b.teamID, MIN(b.yearID) as firstYankeeYear
                        FROM Batting b
                        WHERE b.teamID = 'NYA'
                        GROUP BY b.playerID) fyy
                ON b.playerID = fyy.playerID
                WHERE b.teamID = 'NYA' AND b.yearID != fyy.firstYankeeYear
                GROUP BY b.playerID) syy
        ON b.playerID = syy.playerID
        WHERE syy.secondYankeeYear >= 1980 AND b.teamID = 'NYA'
        GROUP BY b.playerID);
-- 644.576 at-bats


-- Extra Queries

-- 1. Who were the top 3 highest-paid players in 2010?
SELECT m.nameFirst, m.nameLast, SUM(s.salary) as total_salary
FROM Salaries s
INNER JOIN Master m
ON m.playerID = s.playerID
WHERE s.yearID = 2010
GROUP BY s.playerID, s.yearID
ORDER BY total_salary DESC;
-- Alex Rodriguez - $33,000,000
-- CC Sabathia - $24,285,714
-- Derek Jeter - $22,600,000


-- 2. What was the mean player salary in 2010 and which which players were 'highly paid' (over this)
SELECT m.nameFirst, m.nameLast, SUM(s.salary) as total_player_salary, CASE WHEN SUM(s.salary) > avg.overall_avg_salary THEN 1 ELSE 0 END AS highly_paid
FROM Salaries s
INNER JOIN Master m
ON s.playerID = m.playerID
LEFT JOIN (
        SELECT AVG(s.salary) as overall_avg_salary
        FROM Salaries s
        WHERE s.yearID = 2010
        GROUP BY yearID) avg
WHERE s.yearID = 2010
GROUP BY s.playerID, s.yearID
ORDER BY total_player_salary DESC;
-- Avg salary is $3,278,746.83
-- 254 players paid over this


-- 3. Which school has sent the most players to the majors?
-- (Since some players played at multiple schools, technically
-- "Which school has had the most future MLB players play for it?")
SELECT s.schoolName, COUNT(sp.playerID) as num_players_from
FROM SchoolsPlayers sp
LEFT JOIN Schools s
ON sp.schoolID = s.schoolID
WHERE sp.schoolID IS NOT NULL
GROUP BY sp.schoolID
ORDER by num_players_from DESC;
-- University of Southern California with 102


-- 4. Which postwar manager has won the most World Series?
SELECT m.nameFirst, m.nameLast, COUNT(mg.playerID) as ws_wins  FROM SeriesPost sp
LEFT JOIN Managers mg
ON sp.yearID = mg.yearID AND sp.teamIDwinner = mg.teamID
LEFT JOIN Master m
ON mg.playerID = m.playerID
WHERE sp.yearID >= 1945 AND sp.round = 'WS'
GROUP BY mg.playerID
ORDER BY ws_wins DESC
-- Casey Stengel, with 7 WS wins