


import numpy as np


def forecast_600(pls):
    PADict = dict()
    for pl in pls:
        PADict[pl] = 600.
    return PADict

def get_plate_appearances(pls):
    # to complete, we need PA predictions.
    #ST = pd.read_csv('../batting-order/data/2023/estimated_batting_stats_2023.csv')
    ST = pd.read_csv('data/estimated-batting-stats-2024.csv')


    namelist = np.array([x for  x in ST['player'].values])

    print(namelist)

    PADict = dict()
    for name in namelist:
        PADict[name] = ST['PAprediction'][namelist==name]

    #f = open('data/estimated_batting_stats_2023.csv','w')
    #print('player,G22,G21,G20,PA,modelPAs162,modelPAsG',file=f)

    print(PADict)
    # reset PAs to be from last year only

    PADict = dict()
    for name in namelist:
        try:
            PADict[name] = PADict[name]
        except:
            try:
                PADict[name] = lastyeardf['PA'][lastyeardf['Name']==name].values
            except:
                PADict[name] = [200.]

    print(PADict)

    return PADict

