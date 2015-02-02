--1. Find the player with the most at-bats in a single season.
SELECT m.nameFirst as firstname, m.nameLast as lastname, b.playerID, 
max(b.AB) as at_bats
FROM Batting b
LEFT JOIN Master m on b.playerID = m.playerID
GROUP BY b.yearID
ORDER BY at_bats DESC
--Jimmy Rollins


--2. Find the name of the the player with the most at-bats in baseball history.
SELECT m.nameFirst as firstname, m.nameLast as lastname, b.playerID, b.yearID, 
sum(b.AB) as total_at_bats
FROM Batting b
LEFT JOIN Master m on b.playerID = m.playerID
GROUP BY b.playerID
ORDER BY total_at_bats DESC
--Pete Rose


--3. Find the average number of at_bats of players in their rookie season.
SELECT avg(b.AB) as avg_at_bats FROM Batting b
INNER JOIN
(SELECT b.playerID, min(b.yearID) as rookie_year FROM Batting b
GROUP BY b.playerID) rookie on b.playerID = rookie.playerID and b.yearID = rookie_year;
--58.1


--4. Find the average number of at_bats of players in their final season for all players born after 1980.
SELECT avg(b.AB) as avg_at_bats FROM Batting b
INNER JOIN
(SELECT b.playerID, max(b.yearID) as final_year, m.birthYear as birthYear FROM Batting b
LEFT JOIN Master m on b.playerID = m.playerID
WHERE birthYear > 1980
GROUP BY b.playerID) vet on b.playerID = vet.playerID and b.yearID = final_year;
-- 88.9


--5. Find the average number of at_bats of Yankees players who began their second season at or after 1980.
SELECT t.teamID, t.name, avg(at_bats) FROM Teams t
LEFT JOIN
(SELECT b.playerID, b.yearID, b.teamID, b.AB as at_bats from Batting b
INNER JOIN
(SELECT b.playerID, min(b.yearID) as second_year FROM Batting b
LEFT JOIN
(SELECT b.playerID, min(b.yearID) as rookie_year FROM Batting b 
GROUP BY b.playerID) rookie on rookie.playerID = b.playerID AND rookie_year = b.yearID
WHERE rookie_year is null
GROUP BY b.playerID) secyear on b.playerID = secyear.playerID AND second_year = b.yearID) players
ON players.teamID = t.teamID
WHERE t.name = 'New York Yankees' AND players.yearID >= 1980
--103.2


--6. Pass #5 into a pandas DataFrame and write it back to SQLite.
--04_homework-pandas_io.py



-- 4 Distinct Queries

--Which players who have played on the Boston Red Sox were inducted to be in the Hall of Fame, and which years were they inducted?
SELECT firstname, lastname, h.playerID, h.yearID as inducted_year FROM HallOfFame h
INNER JOIN
(SELECT m.nameFirst as firstname, m.nameLast as lastname, b.playerID, b.yearID FROM Batting b
LEFT JOIN Master m on b.playerID = m.playerID
WHERE b.teamID = 'BOS') redsox on redsox.playerID = h.playerID
WHERE h.inducted = 'Y'
GROUP BY redsox.playerID;
--34, Luis Aparicio to start

--Which teams in the league have had the most wins in a single season?
SELECT t.name, sum(t.W) as wins, t.yearID as year FROM Teams t
GROUP BY t.name, t.yearID
ORDER BY wins DESC;
--New York Giants, Philadelphia Athletics, etc

--How many players have been in both the National and American Leagues?
SELECT COUNT(DISTINCT al.playerID) as Players_Both, al.lgID as American_League, National_League FROM Batting al
LEFT JOIN
(SELECT DISTINCT b.playerID, b.lgID as National_League FROM Batting b
WHERE National_League = 'NL') nl ON nl.playerID = al.playerID
WHERE American_League = 'AL' AND American_League != National_League
--5575

--What is the greatest number of runs by a player who has passed away?
SELECT m.nameFirst as firstname, m.nameLast as lastname, runs FROM Master m
LEFT JOIN
(SELECT b.playerID, max(b.R) as runs FROM Batting b
GROUP BY b.playerID) player_runs ON m.playerID = player_runs.playerID
WHERE m.deathYear IS NOT NULL AND runs IS NOT NULL
ORDER BY runs DESC
--Bill Hamilton, 192 runs has the most


-- pandas
--(1) query the Baseball dataset, (2) transform the data in some way, and (3) write a new table back to the databse
--Take National and American League Players and create a new column "Both Leagues"
--04_homework-pandas_transform.py
