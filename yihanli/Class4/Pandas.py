# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 21:03:08 2015

@author: YihanLi
"""

import sqlite3
import pandas
conn = sqlite3.connect('/Users/YihanLi/Documents/SQLite/lahman2013.sqlite')

sql = """
select
avg(t.AB)
FROM
(select
b1.playerID, min(b1.yearID) as sec_min, b1.AB
from
Batting b1
join 
(
SELECT b.playerID,  min(b.yearID) as rookie_year from Batting b
GROUP BY b.playerID) b2
on b1.playerID=b2.playerID
join Teams t on b1.teamID=t.teamID and t.name='New York Yankees'
where b1.yearID > b2.rookie_year
group by 1
) t
where t.sec_min >=1980
"""

df=pandas.read_sql(sql, conn)

#conn.close()

df.fillna(0, inplace = True)

#conn = sqlite3.connect('/Users/YihanLi/Documents/SQLite/lahman2013.sqlite')

df.to_sql('In_class', conn, if_exists='replace', index=False)



sql = """
SELECT 
sch.schoolName,
count(distinct m.playerID)
FROM master m
left join SchoolsPlayers s on m.playerID=s.playerID
left join Schools sch on s.schoolID=sch.schoolID
group by 1
order by 2 desc;
"""

df=pandas.read_sql(sql, conn)

#conn.close()

df.fillna(0, inplace = True)

#conn = sqlite3.connect('/Users/YihanLi/Documents/SQLite/lahman2013.sqlite')

df.to_sql('Question1', conn, if_exists='replace', index=False)

#conn = sqlite3.connect('/Users/YihanLi/Documents/SQLite/lahman2013.sqlite')

sql = """
SELECT
m.nameFirst,
m.nameLast,
count(a.awardID),
count(distinct a.awardID)
FROM master m
join AwardsPlayers a on m.playerID=a.playerID
where a.awardID is not null
group by 1,2
order by 3 desc;
"""

df=pandas.read_sql(sql, conn)

#conn.close()

df.fillna(0, inplace = True)

#conn = sqlite3.connect('/Users/YihanLi/Documents/SQLite/lahman2013.sqlite')

df.to_sql('Question2', conn, if_exists='replace', index=False)

sql = """
select
t.name,
avg(t1.salary)
FROM 
(
SELECT 
s.teamID,
s.playerID,
s.salary
FROM Salaries s
where s.yearID=2013
) t1
join Teams t on t1.teamID=t.teamID
group by 1
order by 2 desc;
"""

df=pandas.read_sql(sql, conn)

#conn.close()

df.fillna(0, inplace = True)

#conn = sqlite3.connect('/Users/YihanLi/Documents/SQLite/lahman2013.sqlite')

df.to_sql('Question3', conn, if_exists='replace', index=False)

sql = """
select
case when m.birthState in ('NH', 'VT', 'MA', 'RI', 'CT', 'ME') then 'New England'
else 'Other' end,
count(distinct m.playerID)
from master m 
group by 1 ;

"""

df=pandas.read_sql(sql, conn)

#conn.close()

df.fillna(0, inplace = True)

#conn = sqlite3.connect('/Users/YihanLi/Documents/SQLite/lahman2013.sqlite')

df.to_sql('Question4', conn, if_exists='replace', index=False)



conn.close()
