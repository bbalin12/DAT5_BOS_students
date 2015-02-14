# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 21:02:53 2015

@author: rodneyhartjr
"""

import sqlite3
import pandas

conn = sqlite3.connect('/Users/admin/Documents/SQLite/lahman2013.sqlite')

sql = """select sq1.maxyear, b.* from Batting b
LEFT OUTER JOIN 
(SELECT  playerID, max(yearID) as maxyear from Batting 
GROUP BY playerID) sq1
ON b.playerID = sq1.playerID
AND b.yearID = sq1.maxyear
WHERE  sq1.maxyear is not null"""

df = pandas.read_sql(sql, conn)

conn.close()

df.fillna(0, inplace = true)

conn = sqlite3.connect('/Users/admin/Documents/SQLite/lahman2013.sqlite')

df.to_sql('pandas_table', conn, if_exists = 'replace')

conn.close()