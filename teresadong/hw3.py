# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 21:03:42 2015

@author: tdong1
"""

import sqlite3
import pandas



conn = sqlite3.connect('D:\\Training\\DataScience\\Class3\\lahman2013.sqlite')

sql = """
SELECT AVG(second_at_bats) from (SELECT y.playerID,y.yearID,second_at_bats
FROM (select playerID,yearID,sum(AB) as second_at_bats from Batting B where teamID in (
	select distinct teamID from Teams
	where name = 'New York Yankees')
group by playerID,yearID) y INNER JOIN (SELECT * FROM (
SELECT playerID,MIN(yearID) AS second_smallest FROM (SELECT playerID, yearID
      FROM Batting
      EXCEPT
      SELECT playerID, MIN(yearID)
      FROM Batting
      GROUP BY playerID)
GROUP BY playerID)
WHERE second_smallest>1980) s ON y.playerID=s.playerID and y.yearID=s.second_smallest)"""

hw4_b6 = pandas.read_sql(sql,conn)

#close connection with database
conn.close()
print 'closed connection'

#just check out first few rows of dataframe
hw4_b6.head()

conn = sqlite3.connect('D:\\Training\\DataScience\\Class3\\lahman2013.sqlite')
#
hw4_b6.to_sql('avg_NYY_2small',conn,if_exists='replace')
conn.close()

#test to make sure table uploaded
sql = 'select * from avg_NYY_2small'
conn = sqlite3.connect('D:\\Training\\DataScience\\Class3\\lahman2013.sqlite')
check_hw4b6 = pandas.read_sql(sql,conn)
check_hw4b6.head()
conn.close()

sql = """SELECT teamABs.*,teamNames.name FROM 
(SELECT teamID,SUM(AB) AS team_at_bats FROM Batting B
WHERE yearID = 2013
GROUP BY teamID
) teamABs
LEFT JOIN
(SELECT DISTINCT teamID,name FROM Teams
WHERE yearID=2013) teamNames on teamABs.teamID = teamNames.teamID
ORDER BY team_at_bats DESC"""

conn = sqlite3.connect('D:\\Training\\DataScience\\Class3\\lahman2013.sqlite')
teamranking_2013 = pandas.read_sql(sql,conn)
teamranking_2013.head()

#keep top 10 only fro the team rankings table
top10rankings = teamranking_2013[:10]
print top10rankings

#upload top 10 at bats to SQL
top10rankings.to_sql('top10_teams_AB',conn,if_exists='replace')
conn.close()

#test to make sure table uploaded
sql = 'select * from top10_teams_AB'
conn = sqlite3.connect('D:\\Training\\DataScience\\Class3\\lahman2013.sqlite')
check_top10_team_AB = pandas.read_sql(sql,conn)
check_top10_team_AB.head()
conn.close()


