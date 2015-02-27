# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 14:29:41 2015

@author: jkraunz
"""

import sqlite3
import pandas

# connect to the baseball database
conn = sqlite3.connect('/Users/jkraunz/Documents/sqlite/lahman2013.sqlite.crdownload')

sql = '''
Select avg(at_bats) From
(Select playerID, teamID, second_year, at_bats, Rookie_year FROM
(Select playerID, teamID, min(yearID) as second_year, at_bats, Rookie_year FROM
(Select b.playerID, b.teamID, b.yearID, b.AB as at_bats, a.Rookie_year FROM batting b
Left Join
(select playerID, min(yearID) as Rookie_year from batting
group by playerID) a
on b.playerID = a.playerID
Where yearID != Rookie_year)
Group by playerID)
where Rookie_year > 1979 and teamID = 'NYA' and at_bats is not null)
'''

df = pandas.read_sql(sql, conn)

conn.close()

conn = sqlite3.connect('/Users/jkraunz/Documents/sqlite/lahman2013.sqlite.crdownload')

df.to_sql('yankees_ABs', conn, if_exists = 'replace')
conn.close()