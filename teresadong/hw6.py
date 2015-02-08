# -*- coding: utf-8 -*-
"""
Created on Sat Feb 07 21:32:21 2015

@author: Teresa
"""

# importing division from the 'future' release of Python (i.e. Python 3)
from __future__ import division

import sqlite3
import pandas
import numpy
# importing statsmodels to run the linear regression
# scikit-learn also has a linear model method, but the statsmodels version
# has more user-friendly output.
import statsmodels.formula.api as smf

# connect to the baseball database. 
DATABASE = r'C:\Users\Teresa\Documents\GA Data Science\lahman2013.sqlite'
conn = sqlite3.connect(DATABASE)
# SQL
sql = """select yearID, 
avg(G) as games
avg(G_batting) as games_as_batter,
avg(AB) as as_bats,
avg(R) as runs, 
avg(H) as hits, 
avg(2B) as doubles,
avg(3B) as triples,
avg(HR) as home_runs,
avg(RunsBattedIn) as runs_batted_in,
avg(SB) as stolen_bases,
avg(CS) as caught_stealing,
avg(BB) as base_on_balls,
avg(SO) as strikeouts, 
avg(IBB) as intentional_walks,
avg(HBP) as hit_by_pitch,
avg(SH) as sacrifice_hits,
avg(SF) as sacrifice_flies,
avg(GIDP) as grounded_into_double_plays
from Batting 
where yearID > 1954
and yearid < 2005
group by yearID
order by yearID ASC"""

df = pandas.read_sql(sql, conn)
conn.close()

# dropping ALL NaNs in the dataset.
df.dropna(inplace = True)      