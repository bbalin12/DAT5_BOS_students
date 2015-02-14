# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 21:02:01 2015

@author: megan
"""

import sqlite3
import pandas

#######################
# IN CLASS EXCERCISES #
#######################

db = '/Users/megan/Documents/GeneralAssembly/012915_SQLIntro/lahman2013.sqlite'

# 1. Find the player with the most at-bats in a single season
conn = sqlite3.connect(db)
sql = """
SELECT playerID, yearID, SUM(AB) as sumAB
FROM Batting
GROUP BY playerID, yearID
ORDER BY sumAB DESC;
"""
df = pandas.read_sql(sql, conn)
print df.playerID[0]
conn.close()

# 2. Find the name of the player with the most at-bats in baseball history
conn = sqlite3.connect(db)
sql = """
SELECT playerID, SUM(AB) as sumAB
FROM Batting
GROUP BY playerID
ORDER BY sumAB DESC;
"""
df = pandas.read_sql(sql, conn)
print df.playerID[0]
conn.close()

# 3. Find the average number of at-bats of players in their rookie season
conn = sqlite3.connect(db)
sql = """
SELECT playerID, AB, min(yearID) as minyear 
FROM Batting
GROUP BY playerID;
"""
df = pandas.read_sql(sql, conn)
df.AB.mean()
conn.close()

# 4. Find the average number of at_bats of players in their final season 
#    for all players born after 1980
conn = sqlite3.connect(db)
sql = """
SELECT b.playerID, b.AB, max(b.yearID) as maxyear, m.birthYear FROM Batting b
LEFT OUTER JOIN Master m ON m.playerID = b.playerID
WHERE m.birthYear > 1980
GROUP BY b.playerID; 
"""
df = pandas.read_sql(sql, conn)
df.AB.mean()
conn.close()

# 5. Find the average number of at_bats of Yankees players who began their
#    their second season at or after 1980
conn = sqlite3.connect(db)
sql = """
SELECT b.playerId, b.teamID, b.yearID, b.AB, sq1.minYear
FROM Batting b
LEFT OUTER JOIN
(SELECT playerID, min(yearID) as minYear from Batting
group by playerID) sq1 
ON sq1.playerID = b.playerID
WHERE b.yearID > sq1.minYear and b.yearID >= 1980
and b.teamID = 'NYA';
"""
df = pandas.read_sql(sql, conn)
df.AB.mean()
conn.close()

# 6. Pass #5 into a pandas DataFrame and write it back to SQLite
conn = sqlite3.connect(db)
df.to_sql('ex6_table', conn, if_exists = 'replace')
conn.close()

############
# HOMEWORK #
############

# Create full, working queries to answer at least four novel questions you have about the dataset using the following concepts...

# Which player earned the highest salary of all time? In what year?
conn = sqlite3.connect(db)
sql = """
SELECT m.nameFirst, m.nameLast, s.yearID, s.teamID, s.salary FROM Salaries s
LEFT OUTER JOIN Master m ON m.playerID = s.playerID
ORDER BY salary Desc;
"""
df = pandas.read_sql(sql, conn)
print "{0} {1} earned {2} in {3} while playing for {4}".format(df.nameFirst[0], df.nameLast[0], df.salary[0], df.yearID[0], df.teamID[0])
conn.close()

# How many player managers have managed over the years?
conn = sqlite3.connect(db)
sql = """
SELECT COUNT(DISTINCT playerID) as numManagers FROM Managers
WHERE plyrMgr = 'Y';
"""
df = pandas.read_sql(sql, conn)
print "{0} managers have managed over the years".format(df.numManagers[0])
conn.close()

# How many pitchers had many hits? (Many is considered > 100)
conn = sqlite3.connect(db)
sql = """
SELECT COUNT(DISTINCT playerID) as numPlayers FROM
(SELECT CASE WHEN H >= 100 THEN 1 ELSE 0 END as many_hits, 
p.* 
FROM Pitching p)
WHERE many_hits = 1;
"""
df = pandas.read_sql(sql, conn)
print "{0} pitchers have had greater than 100 hits".format(df.numPlayers[0])
conn.close()

# What is the earliest year with recorded batting values?
conn = sqlite3.connect(db)
sql = """
SELECT min(yearID) as minYear from Batting
group by yearID
order by minYear asc;
"""
df = pandas.read_sql(sql, conn)
print "{0} is the earliest year with recorded batting values ".format(df.minYear[0])
conn.close()

# Using Pandas, (1) query the Baseball dataset, (2) transform the data in some way, and (3) write a new table back to the databse.
conn = sqlite3.connect(db)
sql = """
SELECT * FROM Master;
"""
df = pandas.read_sql(sql, conn)
df = df.dropna(subset=['deathYear'])
df.to_sql('DeadPlayersMaster', conn, if_exists = 'replace')
conn.close()


