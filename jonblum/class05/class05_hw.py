'''
jonblum
2015-02-03
datbos05
class 5 hw
'''


from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sqlite3

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV


CROSS_VALIDATION_AMOUNT = .2

conn = sqlite3.connect('/Users/jon/Documents/code/datbos05/data/lahman2013.sqlite')
sql = '''
SELECT hof.playerID, b.totalCareerHits, b.careerBattingAvg, p.avgCareerERA, f.careerFieldingPercentage, MAX(hof.inducted) AS inducted
FROM HallOfFame hof
LEFT JOIN
        (SELECT playerID, SUM(H) as totalCareerHits, (SUM(H)*1.0) / SUM(AB) as careerBattingAvg
        FROM Batting
        GROUP BY playerID) b
ON b.playerID = hof.playerID
LEFT JOIN
        (SELECT playerID, AVG(ERA) as avgCareerERA
        FROM Pitching
        GROUP BY playerID) p
ON  p.playerID = hof.playerID
LEFT JOIN
        (SELECT playerID, 1.0 * (SUM(PO) + SUM(A)) / (SUM(PO) + SUM(A) + SUM(E)) as careerFieldingPercentage
        FROM Fielding
        GROUP BY playerID) f
ON f.playerID = hof.playerID
WHERE hof.yearID < 2000 AND hof.category = 'Player'
GROUP BY hof.playerID;
'''

df = pd.read_sql(sql,conn)

conn.close()

df.fillna(value=0,inplace=True)

response_series = df['inducted']
explanatory_variables = df[['totalCareerHits','careerBattingAvg','avgCareerERA','careerFieldingPercentage']]

knn = KNeighborsClassifier(p = 2)
k_range = range(1,30,2)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(explanatory_variables,response_series)


grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(k_range,grid_mean_scores)
best_oob_score = grid.best_score_

# k=15 is optimal
knn_optimal = grid.best_estimator_


conn = sqlite3.connect('/Users/jon/Documents/code/datbos05/data/lahman2013.sqlite')
sql = '''
SELECT hof.playerID, b.totalCareerHits, b.careerBattingAvg, p.avgCareerERA, f.careerFieldingPercentage, MAX(hof.inducted) AS inducted
FROM HallOfFame hof
LEFT JOIN
        (SELECT playerID, SUM(H) as totalCareerHits, (SUM(H)*1.0) / SUM(AB) as careerBattingAvg
        FROM Batting
        GROUP BY playerID) b
ON b.playerID = hof.playerID
LEFT JOIN
        (SELECT playerID, AVG(ERA) as avgCareerERA
        FROM Pitching
        GROUP BY playerID) p
ON  p.playerID = hof.playerID
LEFT JOIN
        (SELECT playerID, 1.0 * (SUM(PO) + SUM(A)) / (SUM(PO) + SUM(A) + SUM(E)) as careerFieldingPercentage
        FROM Fielding
        GROUP BY playerID) f
ON f.playerID = hof.playerID
WHERE hof.yearID >= 2000 AND hof.category = 'Player'
GROUP BY hof.playerID;
'''


df = pd.read_sql(sql,conn)

conn.close()

df.fillna(value=0,inplace=True)

response_series = df['inducted']
explanatory_variables = df[['totalCareerHits','careerBattingAvg','avgCareerERA','careerFieldingPercentage']]

optimal_knn_preds = knn_optimal.predict(explanatory_variables)

number_correct = len(response_series[response_series == optimal_knn_preds])
total_in_test_set = len(response_series)
accuracy = number_correct / total_in_test_set

print accuracy
# 84.7% accuracy on post-2000 data
