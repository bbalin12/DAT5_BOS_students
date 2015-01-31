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
