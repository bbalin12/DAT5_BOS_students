# -*- coding: utf-8 -*-
"""
Created on Sun Feb 01 17:05:03 2015

@author: mmcgoldr
"""

#import packages: pandas, sqlite3
import pandas as pd
import sqlite3 as sq


#PART 1 --------------------------------------------------------------------
#Average at bats among Yankees players who began their second season on/after 1980
#Answer: 197.7

#connect to sql db
conn = sq.connect('C:\Users\mmcgoldr\Dropbox\GA\DataScience\SQLite\lahman2013.sqlite')

#store sql query
query = """select avg(d.ab) as NYA_AB_season2_after1979
from
  (select b.playerid, a.first_year_NYA, min(b.yearid) as second_year_NYA
   from
     (select playerid, min(yearid) as first_year_NYA from batting where teamid = 'NYA' group by playerid) a
   inner join
     (select playerid, yearid from batting where teamid = 'NYA') b
   on a.playerid = b.playerid and a.first_year_NYA != b.yearid
   group by b.playerid
   having second_year_NYA >= 1980) c
inner join
  (select playerid, ab from batting where teamid = 'NYA') d
on c.playerid = d.playerid
"""

#create pandas data frame with result from sql query on db
result = pd.read_sql(query, conn)

print result

#write table (1 row) to db
result.to_sql('pandas_table', conn, if_exists = 'replace')

#close connection
conn.close()


#PART 2---------------------------------------------------------------------
#connect to sql db
conn = sq.connect('C:\Users\mmcgoldr\Dropbox\GA\DataScience\SQLite\lahman2013.sqlite')

#store sql query
query = """select a.teamid, sum(a.winner) as total_ws_wins
from (select teamid, case when wswin = 'Y' then 1 else 0 end as winner from teams where yearid > 1983) a
group by a.teamid
order by total_ws_wins desc"""

#create pandas data frame with result from sql query on db
df = pd.read_sql(query, conn)

#write data frame to db
df.to_sql('wswin_by_team_since1930', conn, if_exists = 'replace')

#close connection
conn.close()




