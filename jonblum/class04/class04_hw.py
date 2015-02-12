'''
jonblum
2015-01-29
datbos05
class 4 hw
'''

import sqlite3
import pandas as pd

conn = sqlite3.connect('/Users/jon/Documents/Code/datbos05/data/lahman2013.sqlite')

sql = """
SELECT AVG(yankeeCareerAtBats) FROM
        (SELECT b.playerID, SUM(b.AB) as yankeeCareerAtBats
        FROM Batting b
        INNER JOIN
                (SELECT b.playerID, b.teamID, MIN(b.yearID) as secondYankeeYear
                FROM Batting b
                INNER JOIN
                        (SELECT b.playerID, b.teamID, MIN(b.yearID) as firstYankeeYear
                        FROM Batting b
                        WHERE b.teamID = 'NYA'
                        GROUP BY b.playerID) fyy
                ON b.playerID = fyy.playerID
                WHERE b.teamID = 'NYA' AND b.yearID != fyy.firstYankeeYear
                GROUP BY b.playerID) syy
        ON b.playerID = syy.playerID
        WHERE syy.secondYankeeYear >= 1980 AND b.teamID = 'NYA'
        GROUP BY b.playerID);
"""

df = pd.read_sql(sql,conn)

df.to_sql('RecentYankeeCareerAtBats',conn,if_exists = 'replace')

conn.close()

#############

conn = sqlite3.connect('/Users/jon/Documents/Code/datbos05/data/lahman2013.sqlite')

sql = """
SELECT *
FROM Master m
"""

df2 = pd.read_sql(sql,conn)

df2 = df2[df2.height < 65]

df2.to_sql('ShortPlayers',conn,if_exists = 'replace')

conn.close()
