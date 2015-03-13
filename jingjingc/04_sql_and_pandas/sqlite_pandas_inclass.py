# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 14:50:34 2015

@author: jchen
"""

# importing SQLite and pandas packages
import sqlite3 
import pandas as pd

# connect to the baseball database
conn = sqlite3.connect('/Users/jchen/Documents/SQLite/lahman2013.sqlite')

# create an object that holds the SQL query.
# Average AB for all Yankees players who began their second season at or after 1980
sql = """
with yankee_rookies as 
	(select playerID, min(b.yearID) as rookie_year 
	from batting b 
	inner join teams t on b.teamID=t.teamID and b.yearID=t.yearID
	where t.name='New York Yankees' 
	group by 1),
	yankee_years as
	(select b.playerID, b.yearID
	from batting b 
	inner join teams t on b.teamID=t.teamID and b.yearID=t.yearID
	where t.name='New York Yankees'
	group by 1,2),
	yankee_seconds as
	(select playerID, min(yearID) as second_year
	from (select * from yankee_years except select * from yankee_rookies) as a
	group by 1)
select b.playerID,
    avg(b.AB) as avg_at_bats
from batting b
inner join teams t on b.teamID=t.teamID
where b.playerID in (select playerID from yankee_seconds where second_year>=1980)
	and t.name='New York Yankees'
group by 1
"""

# pass the connection and the SQL string to pandas
df = pd.read_sql(sql, conn)

# replace NaNs with 0s
df.fillna(0, inplace = True)

# write the data frame back to SQLite
df.to_sql('avg_yankees_second_year_ab', conn, if_exists = 'replace')

# close the connection
conn.close()


