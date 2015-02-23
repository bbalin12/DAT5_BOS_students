# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 10:54:59 2015

@author: jkraunz
"""

import sqlite3
import pandas

conn = sqlite3.connect('/Users/jkraunz/Documents/sqlite/lahman2013.sqlite.crdownload')

sql = """
Select p.* from pitching p
Where ERA < 3.0
"""


df = pandas.read_sql(sql, conn)

conn.close()

df['low_era'] = 0

conn = sqlite3.connect('/Users/jkraunz/Documents/sqlite/lahman2013.sqlite.crdownload')

df.to_sql('p_table', conn, if_exists = 'replace')

conn.close()