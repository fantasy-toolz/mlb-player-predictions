
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import scipy
import scipy.stats as ss


savedate = '022121'
savedate = '030622'


# obtain the data
import src.predictiondata as predictiondata

minyear,maxyear = 2021,2022

years = range(minyear,maxyear)



try: # is the relevant file already constructed?
    df = pd.read_csv('predictions/AllPitching_{}_{}.csv'.format(minyear,maxyear-1))
except:
    df = predictiondata.grab_fangraphs_pitching_data(years)
    df.to_csv('predictions/AllPitching_{}_{}.csv'.format(minyear,maxyear-1))
    df = pd.read_csv('predictions/AllPitching_{}_{}.csv'.format(minyear,maxyear-1))

lastyear = maxyear-1
try:
    lastyeardf = pd.read_csv('predictions/AllPitching_{}.csv'.format(lastyear))
except:
    lastyeardf = predictiondata.grab_fangraphs_pitching_data([lastyear])
    lastyeardf.to_csv('predictions/AllPitching_{}.csv'.format(lastyear))
    lastyeardf = pd.read_csv('predictions/AllPitching_{}.csv'.format(lastyear))


# pick some players of interest based on draft location,
names = ['Joe Ryan','Jose Urquidy','Alex Wood','Clayton Kershaw','Shane McClanahan','Luis Castillo','Justin Verlander']

for name in names:
    try:
        print(name,lastyeardf['SO'][lastyeardf['Name']==name].values[0],lastyeardf['HR'][lastyeardf['Name']==name].values[0],lastyeardf['TBF'][lastyeardf['Name']==name].values[0])
        print(name,(lastyeardf['SO'][lastyeardf['Name']==name].values[0])/lastyeardf['TBF'][lastyeardf['Name']==name].values[0])
    except:
        print(name)
#lastyeardf['PA'][lastyeardf['Name']==name].values
