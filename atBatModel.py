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
    def __init__(self, name = list)-> None:
        pyb.cache.enable()

        self.playerName = (name[0] + " " + name[1]).title()

        playerID = pyb.playerid_lookup(name[1], name[0])

        if playerID.shape[0] == 0:
            raise Exception("No players found by the name " + self.playerName)

        self.playerID = playerID.iloc[0]

    def getStatcastData(self, playerType = str, dateRange = ["2021-01-01", "2021-10-03"], verbose = False) -> pd.DataFrame:
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

        for x in ["batter", "pitcher"]:
            try:
                data =  eval(x + ".statcastData")
            except AttributeError:
                data = eval(x + ".getStatcastData(playerType=" + x + ")")
            
            self.modelData[x] = processData(data)

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
            pitchStats = countData.groupby("pitch_type").mean()
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

            outcomes = np.round(outcomeMatrix.sum(), decimals=3)

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
        
        stats = pd.Series([self.pitcher.playerName, self.batter.playerName], index = ["Pitcher", "Batter"])
        statsVector = pd.Series(np.round(inVector[12:18], 3), index = ["pWalk", "pHBP", "p1B", "p2B", "pHR", "pOut"])

        stats = stats.append(statsVector)

        stats["AVG"] = np.round(sum(stats[["p1B", "p2B", "pHR"]]), 3)

        stats["OBP"] = np.round((1 - statsVector["pOut"]), 3)

        stats["SLG"] = np.round(stats["p1B"] + 2*stats["p2B"] + 4*stats["pHR"], 3)
        
        stats["OPS"] = np.round(sum(stats[["OBP", "SLG"]]), 3)

        wOBAWeights = np.array([0.69, 0.72, 0.89, 1.27, 2.1])
        stats["wOBA"] = np.round(np.dot(wOBAWeights, inVector[12:17]), 3)

        return stats.transpose()

if __name__ == "__main__":
    print("Welcome to the baseball at-bat modeler.")