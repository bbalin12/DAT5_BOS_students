# Gareth Austen Homework for Class 09 - Data Manipulation

#### Introduction
I pulled data from the SQLite database using the below query. I decided to pull in the 
state that each player was born in to determine if it had any impact on whether a player 
was inducted into the Hall of Fame. 

```
select coalesce(a.nameFirst ,"") || coalesce(a.nameLast,"") as PlayerName, hfi.inducted,
a.playerID, a.birthState, b.*, c.*,d.*
from hall_of_fame_inductees hfi
left outer join master a
on a.playerID = hfi.playerID
left outer join 
(
select sum(b.G) as Games, sum(b.H) as Hits, sum(b.AB) as At_Bats, sum(b.HR) as Homers,
b.playerID from Batting b group by b.playerID
) b
on hfi.playerID = b.playerID
left outer join 
(
select d.playerID, sum(d.A) as Fielder_Assists, sum(d.E) as Field_Errors, 
sum(d.DP) as Double_Plays from Fielding d group by d.playerID 
) c
on hfi.playerID = c.playerID
left outer join
(
select b.playerID, sum(b.ERA) as Pitcher_Earned_Run_Avg, sum(b.W) as Pitcher_Ws,
sum(b.SHO) as Pitcher_ShutOuts, sum(b.SO) as Pitcher_StrikeOuts,
sum(b.HR) as HR_Allowed, sum(b.CG) as Complete_Games 
from Pitching b 
group by b.playerID
) d 
on hfi.playerID = d.playerID;
```

In my previous homeworks I had split the data out by Pitchers and Batters. However this is very
time consuming and therefore I have to re-combine the datasets using the above sql query. 

#### Importing the Functions create in class

The next step was to reinstiate the functions that we had created during class to clean and 
manipulate the data. The following code segment was used:

```
###################################################################
######## Importing Functions created in inclass exercise ##########
###################################################################

# Function to cut off data that has less than 1% of all volume
def cleanup_data(df, cutoffPercent = .01):
    for col in df:    
        sizes = df[col].value_counts(normalize=True) # normalize = True gives percentages
        # get the names of the levels that make up less than 1% of the dataset
        values_to_delete = sizes[sizes<cutoffPercent].index
        df[col].ix[df[col].isin(values_to_delete)] = 'Other'
    return df

# Function to create binary dummies for catergorical data    
def get_binary_values(data_frame):
    """encodes the categorical features in Pandas
    """
    all_columns = pandas.DataFrame(index=data_frame.index)
    for col in data_frame.columns:
        data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii','replace'))
        all_columns = pandas.concat([all_columns,data],axis=1)
    return all_columns

# Function to find variables with no variance at all - Need to Impute before this step
def find_zero_var(df):
    """ find the columns in the dataframe with zero variance -- ie those 
        with the same value in every observation
    """
    toKeep = []
    toDelete = []
    for col in df:
        if len(df[col].value_counts())>1:
            toKeep.append(col)
        else:
            toDelete.append(col)
    #
    return {'toKeep':toKeep, 'toDelete':toDelete}
    
# Function to find the variables with perfect correlation
def find_perfect_corr(df):
    """finds columns that are eother positively or negatively 
    perfectly correlated (with correlations of +1 or -1), and creates a dict 
        that includes which columns to drop so that each remaining column
        is independent
    """  
    corrMatrix = df.corr()
    corrMatrix.loc[:,:] =  numpy.tril(corrMatrix.values, k = -1)
    already_in = set()
    result = []
    for col in corrMatrix:
        perfect_corr = corrMatrix[col][abs(numpy.round(corrMatrix[col],10)) == 1.00].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    toRemove = []
    for item in result:
        toRemove.append(item[1:(len(item)+1)])
    toRemove = sum(toRemove, [])
    return {'corrGroupings':result, 'toRemove':toRemove}
```

