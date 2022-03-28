# atBatModel.ipynb

This Jupyter notebook is meant as an example of how to use atBatModel.py. Required packages:
* numpy
* pandas
* pybaseball
* sklearn


```python
import atBatModel as abm
import numpy as np
```

Initialize player classes using a name. The players' names and various IDs are saved as class variables.


```python
pitcher = abm.player(name=["Gerrit", "Cole"])
batter = abm.player(name=["Mike", "Trout"])

print("Player name: ", batter.playerName)
print(batter.playerID)
```

    Player name:  Mike Trout
    name_last               trout
    name_first               mike
    key_mlbam              545361
    key_retro            troum001
    key_bbref           troutmi01
    key_fangraphs           10155
    mlb_played_first       2011.0
    mlb_played_last        2021.0
    Name: 0, dtype: object
    

Download player Statcast data. Per pybaseball, it is represented as a Pandas dataFrame. It is returned as a function output and saved to the player object's namespace. By default, the date range is from today to January 1 of the previous year.


```python
pitcher.getStatcastData(playerType="pitcher", verbose=True)
batter.getStatcastData(playerType="batter", verbose=True, dateRange=["2020-01-01","2022-01-01"])
```

    Gathering Player Data
    Pitcher: Gerrit Cole
    Found 3322 observations from 2021-01-01 to 2022-03-27.
    
    Gathering Player Data
    Batter: Mike Trout
    Found 1859 observations from 2020-01-01 to 2022-01-01.
    
    

To construct the Markov model, we initialize an instance of the markovModel class using pitcher and batter classes as parameters. The construction of the Markov matrix and calculation of the logistic model takes place automatically. The sanitized data used to calculate the model is also saved in the object namespace.


```python
model = abm.markovModel(pitcher = pitcher, batter = batter)

np.round(model.markovMatrix, 3)
```




    array([[0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
            0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
           [0.431, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
            0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
           [0.   , 0.425, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
            0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
           [0.   , 0.   , 0.419, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
            0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
           [0.45 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
            0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
           [0.   , 0.445, 0.   , 0.   , 0.433, 0.   , 0.   , 0.   , 0.   ,
            0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
           [0.   , 0.   , 0.447, 0.   , 0.   , 0.43 , 0.   , 0.   , 0.   ,
            0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
           [0.   , 0.   , 0.   , 0.443, 0.   , 0.   , 0.425, 0.   , 0.   ,
            0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
           [0.   , 0.   , 0.   , 0.   , 0.449, 0.   , 0.   , 0.   , 0.138,
            0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
           [0.   , 0.   , 0.   , 0.   , 0.   , 0.447, 0.   , 0.   , 0.446,
            0.142, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
           [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.448, 0.   , 0.   ,
            0.442, 0.15 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
           [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.446, 0.   ,
            0.   , 0.437, 0.166, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
           [0.   , 0.   , 0.   , 0.427, 0.   , 0.   , 0.   , 0.422, 0.   ,
            0.   , 0.   , 0.428, 1.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
           [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
            0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   ],
           [0.024, 0.028, 0.032, 0.032, 0.024, 0.026, 0.027, 0.03 , 0.023,
            0.024, 0.025, 0.028, 0.   , 0.   , 1.   , 0.   , 0.   , 0.   ],
           [0.009, 0.011, 0.011, 0.009, 0.009, 0.01 , 0.011, 0.01 , 0.01 ,
            0.01 , 0.01 , 0.01 , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   ],
           [0.017, 0.016, 0.017, 0.016, 0.017, 0.016, 0.016, 0.016, 0.016,
            0.016, 0.016, 0.016, 0.   , 0.   , 0.   , 0.   , 1.   , 0.   ],
           [0.067, 0.074, 0.076, 0.072, 0.068, 0.071, 0.074, 0.075, 0.368,
            0.366, 0.362, 0.352, 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ]])



To obtain the raw outcome vector, we use the method simulatePitches. We may specify the starting count, as well as a number of iterations of the matrix equation x_{n+1} = Ax_n. Every iteration is like one pitch in an at-bat.


```python
outcomeVector = model.simulatePitches(100, (0,0))

np.round(outcomeVector, 3)
```




    array([0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
           0.   , 0.   , 0.   , 0.191, 0.   , 0.103, 0.04 , 0.067, 0.6  ])



We may also compute stats using the method outcomeStats.


```python
model.outcomeStats()
```




    Pitcher    Gerrit Cole
    Batter      Mike Trout
    pWalk            0.191
    pHBP               0.0
    p1B              0.103
    p2B               0.04
    pHR              0.067
    pOut               0.6
    AVG               0.21
    OBP                0.4
    SLG              0.451
    OPS              0.851
    wOBA             0.413
    dtype: object


