# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 21:03:43 2015

@author: melaccor
"""

import sqlite3
import pandas

conn=sqlite3.connect('C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite')
sql="""
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
WHERE second_smallest>=1980) s ON y.playerID=s.playerID and y.yearID=s.second_smallest);"""
df=pandas.read_sql(sql,conn);

conn=sqlite3.connect('C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite')
df.to_sql('avg_AB', conn, if_exists = 'replace')
conn.close()


conn=sqlite3.connect('C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite')
sql="""
select yearID, teamID from Batting;"""
df=pandas.read_sql(sql,conn);
df=df[(df.yearID>2010) & (df.teamID=='NYA')];
conn=sqlite3.connect('C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite')
df.to_sql('year_team', conn, if_exists = 'replace')
conn.close()
















