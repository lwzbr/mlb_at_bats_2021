import numpy as np
import pandas as pd
import pybaseball as pyb
from sklearn import linear_model

def classifyOutcome(series):
    # Types of events we classify as strikes, fouls, balls, and field outs
    strikeEvents = np.array(["foul_tip", "called_strike", "swinging_strike", "swinging_strike_blocked", "missed_bunt"])
    foulEvents = np.array(["foul", "foul_bunt"])
    ballEvents = np.array(["ball", "blocked_ball"])
    outEvents = np.array(["force_out","field_error","field_out","fielders_choice","fielders_choice_out","grounded_into_double_play","double_play","sac_fly"])

    events = series["events"]
    description = series["description"]

    if np.isin(description, strikeEvents):
        return "K"
    elif np.isin(description, ballEvents):
        return "B"
    elif np.isin(description, foulEvents):
        return "F"
    elif np.isin(events, outEvents):
        return "FO"
    elif events == "hit_by_pitch":
        return "HBP"
    elif events == "single":
        return "1B"
    elif events == "double":
        return "2B"
    # elif events == "triple":
    #     return "3B"
    elif events == "home_run":
        return "HR"

def processData(df) -> pd.DataFrame:
    pd.options.mode.chained_assignment = None  # default='warn'
    df = df[["pitch_type","events","description","release_speed","release_spin_rate","balls","strikes","pfx_x","pfx_z","plate_x","plate_z","vx0","vy0","vz0"]]

    outcomes = df.apply(classifyOutcome, axis=1)

    df["Outcome"] = outcomes
    df = df.drop(["events", "description"], axis=1)

    df = df.dropna()
    
    return df

def getCountData(df, count):
    return df[(df.balls == count[0]) & (df.strikes == count[1])]

def trainLogisticModel(pitchData, dataFeatures):

    X = pitchData[dataFeatures].values
    y = pitchData["Outcome"].values

    #X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2, random_state = 1)

    lm = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
    return lm.fit(X, y)

def vectorFromCount(count=(0,0)):
    vector = np.zeros(18, dtype=float)
    index = count[0] + 4*(count[1])
    
    vector[index] = 1

    return vector

class player:
    def __init__(self, **lookup)-> None:
        pyb.cache.enable()

        playerID = pd.Series(dtype=object)

        # Maybe we have a key?
        for key in ("key_mlbam","key_retro","key_bbref","key_fangraphs"):
            l = lookup.get(key)

            if not (l == None):
                playerID = pyb.playerid_reverse_lookup([l], key_type=key[4:])
        
        # If we still haven't found anyone, see if a name has been given
        if not ((lookup.get("name_first")) == None or (lookup.get("name_last") == None)) and (playerID.shape[0] == 0):
            playerID = pyb.playerid_lookup(first = lookup.get("name_first"), last = lookup.get("name_last"))

        if (playerID.shape[0] == 0):
            # PlayerID still empty, throw exception

            raise AssertionError(str("Failed to find player with these inputs or insufficient information given:\n" + str(lookup)))
        elif (playerID.shape[0] > 1):
            # Found several players, warn that we will be using the first one
            print("Warning: Found", playerID.shape[0], "players with input\n", str(lookup), "\nDefaulting to the first player!")

        self.playerID = playerID.iloc[0]
        self.playerName = (self.playerID["name_first"] + " " + self.playerID["name_last"]).title()

    def getStatcastData(self, playerType = str, dateRange = [], verbose = False) -> pd.DataFrame:
        # By default we will get pitches from games between today and 01/01/(last year)
        if dateRange == []:
            from datetime import date
            today = date.today()

            jan1LastYear = str(today.year - 1) + "-01-01"

            dateRange.append(jan1LastYear)
            dateRange.append(today.strftime("%Y-%m-%d"))

        if playerType.lower() == "batter":
            getSC = pyb.statcast_batter
        else:
            getSC = pyb.statcast_pitcher

        statcastData = getSC(start_dt = dateRange[0], end_dt = dateRange[1], player_id = self.playerID["key_mlbam"])

        if verbose:
            print(playerType.title() + ": " + self.playerName)
            print("Found " + str(statcastData.shape[0]) + " observations from " + dateRange[0] + " to " + dateRange[1] + ".\n")

        self.statcastData = statcastData

        return statcastData

class markovModel:
    def __init__(self, pitcher = player, batter = player) -> None:
        dataFeatures = np.array(["release_speed","release_spin_rate","pfx_x","pfx_z","plate_x","plate_z","vx0","vy0","vz0"])

        self.modelData = dict()

        try:
            data = batter.statcastData
        except AttributeError:
            data = batter.getStatcastData(playerType="batter")
        finally:
            self.modelData["batter"] = processData(data)

        try:
            data = pitcher.statcastData
        except AttributeError:
            data = pitcher.getStatcastData(playerType="pitcher")
        finally:
            self.modelData["pitcher"] = processData(data)

        # try:
        #     data = batter.statcastData
        # except AttributeError:
        #     data = batter.getStatcastData(playerType="batter")

        # for x in ["batter", "pitcher"]:
        #     try:
        #         data =  eval(x + ".statcastData")
        #     except AttributeError:
        #         print(type(x))
        #         data = eval(x + ".getStatcastData(playerType=" + x + ")")
            
        #     self.modelData[x] = processData(data)

        self.pitcher = pitcher
        self.batter = batter

        # Train the batter model
        X = (self.modelData["batter"])[dataFeatures].values
        y = (self.modelData["batter"])["Outcome"].values

        lm = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')

        self.batterModel = lm.fit(X, y)

        # Function to compute outcomes based on batter model
        def predictOutcomes(pitch, df):
            lm = self.batterModel
            pred = pd.DataFrame()
            
            pred.index = lm.classes_

            xProbs = np.round(lm.predict_proba(np.array(df.loc[pitch].drop("Freq")).reshape(1, -1)), 3)

            pred[pitch] = xProbs.transpose()

            return pred.transpose()

        # Function to compute outcome probabilities by count
        def countProbabilities(count):
            # Grab the pitcher's data
            countData = getCountData(self.modelData["pitcher"], count)

            # We select the mean parameters for every pitch type
            pitchStats = countData.groupby("pitch_type")[dataFeatures].mean()
            pitchStats = pitchStats[dataFeatures]

            pitchStats["Freq"] = countData["pitch_type"].value_counts()
            pitchStats["Freq"] = pitchStats["Freq"]/pitchStats["Freq"].sum()

            countOutcomes = pd.DataFrame([],columns=['1B', '2B', 'B', 'F', 'FO', 'HBP', 'HR', 'K'])

            for p in pitchStats.index:
                countOutcomes = pd.concat([countOutcomes, predictOutcomes(p, pitchStats)], axis=0)

            countOutcomes = pd.concat([countOutcomes, pitchStats["Freq"]], axis=1).fillna(0)

            return countOutcomes

        vectorsA = [
            [(0,0), {1: ["B"], 4: ["K","F"], 13: ["HBP"], 14: ["1B"], 15: ["2B"], 16: ["HR"], 17: ["FO"]}],   #0
            [(1,0), {2: ["B"], 5: ["K","F"], 13: ["HBP"], 14: ["1B"], 15: ["2B"], 16: ["HR"], 17: ["FO"]}],   #1
            [(2,0), {3: ["B"], 6: ["K","F"], 13: ["HBP"], 14: ["1B"], 15: ["2B"], 16: ["HR"], 17: ["FO"]}],   #2
            [(3,0), {7: ["K","F"], 12: ["B"], 13: ["HBP"], 14: ["1B"], 15: ["2B"], 16: ["HR"], 17: ["FO"]}],  #3
            [(0,1), {5: ["B"], 8: ["K","F"], 13: ["HBP"], 14: ["1B"], 15: ["2B"], 16: ["HR"], 17: ["FO"]}],   #4
            [(1,1), {6: ["B"], 9: ["K","F"], 13: ["HBP"], 14: ["1B"], 15: ["2B"], 16: ["HR"], 17: ["FO"]}],   #5
            [(2,1), {7: ["B"], 10: ["K","F"], 13: ["HBP"], 14: ["1B"], 15: ["2B"], 16: ["HR"], 17: ["FO"]}],  #6
            [(3,1), {11: ["K","F"], 12: ["B"], 13: ["HBP"], 14: ["1B"], 15: ["2B"], 16: ["HR"], 17: ["FO"]}], #7
            [(0,2), {8: ["F"], 9: ["B"], 13: ["HBP"], 14: ["1B"], 15: ["2B"], 16: ["HR"], 17: ["FO","K"]}],   #8
            [(1,2), {9: ["F"], 10: ["B"], 13: ["HBP"], 14: ["1B"], 15: ["2B"], 16: ["HR"], 17: ["FO","K"]}],  #9
            [(2,2), {10: ["F"], 11: ["B"], 13: ["HBP"], 14: ["1B"], 15: ["2B"], 16: ["HR"], 17: ["FO","K"]}], #10
            [(3,2), {11: ["F"], 12: ["B"], 13: ["HBP"], 14: ["1B"], 15: ["2B"], 16: ["HR"], 17: ["FO","K"]}]  #11
        ]

        matrixA = np.zeros((18,18), dtype=np.float64)

        matrixA[12,12] = 1
        matrixA[13,13] = 1
        matrixA[14,14] = 1
        matrixA[15,15] = 1
        matrixA[16,16] = 1
        matrixA[17,17] = 1

        for i in range(len(vectorsA)):
            count, code = vectorsA[i]
            vector = np.zeros(18)

            outcomeMatrix = countProbabilities(count)
            outcomeMatrix = outcomeMatrix.multiply(outcomeMatrix["Freq"], axis=0).drop("Freq", axis=1)

            outcomes = outcomeMatrix.sum()

            for j in code:
                vector[j] = sum(outcomes[code.get(j)])

            matrixA[i] = vector

        self.markovMatrix = matrixA.transpose()
    
    def simulatePitches(self, pitches=100, startCount=(0,0)):
        startVector = vectorFromCount(count=startCount)
        
        self.outcomeVector = np.matmul(startVector, np.linalg.matrix_power(self.markovMatrix.transpose(), pitches))

        return self.outcomeVector

    def outcomeStats(self):
        try:
            inVector = self.outcomeVector
        except AttributeError:
            inVector = self.simulatePitches()
        
        # stats = pd.Series(
        #     [
        #         self.pitcher.playerName, 
        #         self.batter.playerName], 
        #     index = [
        #         "pitcher_name", "batter_name"
        #     ]
        # )
        stats = pd.Series(
            {
                # "pitcher": self.pitcher,
                # "batter": self.batter
                # "pitcher_key_mlbam": self.pitcher.playerID["key_mlbam"],
                # "batter_key_mlbam": self.batter.playerID["key_mlbam"],
                "pitcher_name": self.pitcher.playerName,
                "batter_name": self.batter.playerName
            }            
        )
        statsVector = pd.Series(np.round(inVector[12:18], 3), index = ["x_pBB", "x_pHBP", "x_p1B", "x_p2B", "x_pHR", "x_pOut"])

        stats = pd.concat([stats, statsVector])

        stats["x_AVG"] = np.round(sum(stats[["x_p1B", "x_p2B", "x_pHR"]]), 3)

        stats["x_OBP"] = np.round((1 - statsVector["x_pOut"]), 3)

        stats["x_SLG"] = np.round(stats["x_p1B"] + 2*stats["x_p2B"] + 4*stats["x_pHR"], 3)
        
        stats["x_OPS"] = np.round(sum(stats[["x_OBP", "x_SLG"]]), 3)

        wOBAWeights = np.array([0.69, 0.72, 0.89, 1.27, 2.1])
        stats["x_wOBA"] = np.round(np.dot(wOBAWeights, inVector[12:17]), 3)

        return stats.transpose()

if __name__ == "__main__":
    print("Welcome to the baseball at-bat modeler.")
