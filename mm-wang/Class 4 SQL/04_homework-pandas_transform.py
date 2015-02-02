# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 19:18:36 2015

@author: Margaret
"""

#importing SQLite and pandas
import sqlite3
import pandas

#connect to the baseball database
conn = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')

#create an object containing a string that has the SQL query. """ to allow query to exist on multiple lines
nl_al = """
SELECT al.playerID as Players_Both, al.lgID as American_League, National_League FROM Batting al
LEFT JOIN
(SELECT b.playerID, b.lgID as National_League FROM Batting b
WHERE National_League = 'NL') nl ON nl.playerID = al.playerID
WHERE American_League = 'AL' AND American_League != National_League
"""

#reading SQL into pandas
df = pandas.read_sql(nl_al,conn)

conn.close()

df['Both_Leagues'] = False
df.Both_Leagues[(df.American_League == 'AL') & (df.National_League == 'NL')] = True
df.drop_duplicates('Players_Both')

#conn = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')
#write table back to database, replace in place
#df.to_sql('both_leagues', conn, if_exists = 'replace')

#close connection
#conn.close()