# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 22:05:50 2015

@author: Margaret
"""

#importing SQLite and pandas
import sqlite3
import pandas

#connect to the baseball database
conn = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')

#create an object containing a string that has the SQL query. """ to allow query to exist on multiple lines
sql5 = """SELECT t.teamID, t.name, avg(at_bats) FROM Teams t
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
WHERE t.name = 'New York Yankees' AND players.yearID >= 1980"""

#reading SQL into pandas
df = pandas.read_sql(sql5,conn)

conn.close()

df.head()

conn = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')
#write table back to database, replace in place
df.to_sql('yankees_atbats', conn, if_exists = 'replace')

#close connection
conn.close()