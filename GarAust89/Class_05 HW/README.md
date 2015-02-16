# Gareth Austen KNN Homework for Class 05

Step 1 was to pull data from the lahman 2013 baseball dataset. Initally tried to use a large query to pull data from all
four tables simultaneously using the following SQL query:

```
select a.playerID, a.inducted, f.Games, f.Hits, f.At_Bats, f.Homers, f.Pitcher_Ws, f.Pitcher_ShutOuts,
f.Pitcher_StrikeOuts, f.Pitcher_Earned_Run_Avg, f.Field_Position, f.Field_Errors from HallOfFame a 
left outer join 
(
select b.G as Games, b.H as Hits, b.AB as At_Bats, b.HR as Homers, b.playerID, e.Pitcher_Ws, e.Pitcher_ShutOuts,
e.Pitcher_StrikeOuts, e.Pitcher_Earned_Run_Avg,e.Field_Position, e.Field_Errors  from Batting b
left outer join 
(
select c.playerID, c.W as Pitcher_Ws, c.SHO as Pitcher_ShutOuts, c.SO as Pitcher_StrikeOuts, c.ERA as Pitcher_Earned_Run_Avg, 
d.Pos as Field_Position, d.E as Field_Errors from Pitching c left outer join Fielding d on c.playerID = d.playerID
) e 
on b.playerID = e.playerID) f
on a.playerID = f.playerID
where yearID<2000;
```

However, when I dropepd the NA rows from the pandas data frame we were left with a very small number of datapoints. 
I believe this is because of the crossover between Batters and Pitchers. Therefore I decided to create individual models
for pitchers and batters. 

After creating two separate dataframes, I followed a similar process to the in-class exercise:
1. Split both datasets into response and explanatory variables
2. Created separate holdouts for both Batters and Pitchers
3. Created training and test sets
4. Instatiated two KNN models and fit them to the dataset
5. Predicted whether or not players were inducted into the Hall of Fame
6. Compared the accuracy of the two models to the in-class exercise model

All of the above was completed using the following code segement:

```
# separate your response variable from your explanatory variable for both the batters and pitchers datasets
response_series_batters = df_batters.inducted
response_series_pitchers = df_pitchers.inducted
explanatory_vars_batters = df_batters[['Games','Hits','At_Bats','Homers','Double_Plays','Fielder_Assists','Field_Errors']]
explanatory_vars_pitchers= df_pitchers[['Pitcher_Ws','Pitcher_ShutOuts','Pitcher_StrikeOuts','HR_Allowed','Complete_Games']]

## Designate Separate holdouts for both the batters and pitchers 
holdout_num_batters = round(len(df_batters.index)*CROSS_VALIDATION_AMOUNT,0)
holdout_num_pitchers = round(len(df_pitchers.index)*CROSS_VALIDATION_AMOUNT,0)

# creating our training and test indices for the batter and pitcher datasets #
test_indices_batters = numpy.random.choice(df_batters.index, holdout_num_batters, replace = False)
train_indices_batters = df_batters.index[~df_batters.index.isin(test_indices_batters)]
test_indices_pitchers = numpy.random.choice(df_pitchers.index, holdout_num_pitchers, replace = False)
train_indices_pitchers = df_pitchers.index[~df_pitchers.index.isin(test_indices_pitchers)] 

# create our training set for both datasets
response_train_batters = response_series_batters.ix[train_indices_batters,]
explanatory_train_batters = explanatory_vars_batters.ix[train_indices_batters,]
response_train_pitchers = response_series_pitchers.ix[train_indices_pitchers,]
explanatory_train_pitchers = explanatory_vars_pitchers.ix[train_indices_pitchers,]

# create our test set for both datasets
response_test_batters = response_series_batters.ix[test_indices_batters,]
explanatory_test_batters = explanatory_vars_batters.ix[test_indices_batters,]
response_test_pitchers= response_series_pitchers.ix[test_indices_pitchers,]
explanatory_test_pitchers = explanatory_vars_pitchers.ix[test_indices_pitchers,]

## Instantiating the KNN Classifier, with p = 2 for Euclidian distance
KNN_Classifier_batters = KNeighborsClassifier(n_neighbors=3,p=2)
KNN_Classifier_pitchers = KNeighborsClassifier(n_neighbors=3,p=2)
# fitting the data to the training set #
KNN_Classifier_batters.fit(explanatory_train_batters,response_train_batters)
KNN_Classifier_pitchers.fit(explanatory_train_pitchers,response_train_pitchers)

# predicting the data on the test set
predicted_response_batters = KNN_Classifier_batters.predict(explanatory_test_batters)
predicted_response_pitchers = KNN_Classifier_pitchers.predict(explanatory_test_pitchers)

# calculating accuracy
number_correct_batters = len(response_test_batters[response_test_batters == predicted_response_batters])
total_in_test_set_batters = len(response_test_batters)
accuracy_batters = number_correct_batters/total_in_test_set_batters
print accuracy_batters*100

number_correct_pitchers = len(response_test_pitchers[response_test_pitchers == predicted_response_pitchers])
total_in_test_set_pitchers = len(response_test_pitchers)
accuracy_pitchers = number_correct_pitchers/total_in_test_set_pitchers
print accuracy_pitchers*100
```

#### Initial Results 
It appears from examining the output of these initial models that the Batters Model is more accurate than 
the pitchers model. The Batters model has an accuracy of 79.6% while the pitchers model has an accuracy of
69.7% however both of these models are significantly less accurate than the inclass model which has an 
accuracy of 98%

## Perform K Fold Cross Validation

The next step in the analysis was to perform K-Fold Cross Validation of our two models. For this exercise 
I decided to use 10-Fold Cross Validation. This was completed using the following code segment: 

```
# LET'S USE 10-FOLD CROSS-VALIDATION TO SCORE OUR MODEL
from sklearn.cross_validation import cross_val_score
# we need to re-instantiate the model
KNN_Classifier_batters = KNeighborsClassifier(n_neighbors=3,p=2)
KNN_Classifier_pitchers = KNeighborsClassifier(n_neighbors=3,p=2)

# Notice that instead of passing in the train and test sets we are passing 
# the entire dataset as method will auto split
scores_batters = cross_val_score(KNN_Classifier_batters, explanatory_vars_batters,response_series_batters,cv=10,scoring='accuracy')
scores_pitchers = cross_val_score(KNN_Classifier_pitchers, explanatory_vars_pitchers,response_series_pitchers,cv=10,scoring='accuracy')
                         
# print out scores object
print scores_batters 
print scores_pitchers
        
# now let's get the average accuary score 
mean_accuracy_batters = numpy.mean(scores_batters)
print mean_accuracy_batters*100
mean_accuracy_pitchers = numpy.mean(scores_pitchers)
print mean_accuracy_pitchers*100
# look at how this differes from the previous two accuarcies 
print accuracy_batters*100
print accuracy_pitchers*100
``` 

#### Cross-Validation Results

Interestingly using Cross Validation has not significantly improved our Batting model but has led to significant improvements in the pitching model. Next we will search for the optimal value of K. 

## Finding the optimal value of K

The optimal Value of K was determined by using a range of K values and then plotting the outcome. This was completed using the following code segment: 

```
# Tune the model for the optimal number of K
k_range = range(1,30,2)
scores_batters =[]
for k in k_range:
    knn_batters = KNeighborsClassifier(n_neighbors=k,p=2)
    scores_batters.append(numpy.mean(cross_val_score(knn_batters,explanatory_vars_batters,response_series_batters,cv=5,scoring='accuracy')))
    
k_range = range(1,30,2)
scores_pitchers =[]
for k in k_range:
    knn_pitchers = KNeighborsClassifier(n_neighbors=k,p=2)
    scores_pitchers.append(numpy.mean(cross_val_score(knn_pitchers,explanatory_vars_pitchers,response_series_pitchers,cv=5,scoring='accuracy')))

    
# Plot the K values (x-axis) versus the 5 fold CV Score
import matplotlib.pyplot as plt
plt.figure()
plt.plot(k_range,scores_batters)

plt.figure()
plt.plot(k_range,scores_pitchers)
# optimal value of K appears to be 3
```

#### Optimal K Results
Below are two plots for the Batting and Pitching datasets. 


