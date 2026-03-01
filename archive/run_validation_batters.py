
import numpy as np
import pandas as pd


savedate = '030622'

# obtain the data
import src.predictiondata as predictiondata

minyear,maxyear = 2021,2022
minyear,maxyear = 2022,2023

remake=False

years = range(minyear,maxyear)

if remake==True: # force remaking
    df = predictiondata.grab_fangraphs_data(years)
    df.to_csv('predictions/AllHitting_{}_{}.csv'.format(minyear,maxyear-1))

try: # is the relevant file already constructed?
    df = pd.read_csv('predictions/AllHitting_{}_{}.csv'.format(minyear,maxyear-1))
except:
    df = predictiondata.grab_fangraphs_data(years)
    df.to_csv('predictions/AllHitting_{}_{}.csv'.format(minyear,maxyear-1))

lastyear = maxyear-1
try:
    lastyeardf = pd.read_csv('predictions/AllHitting_{}.csv'.format(lastyear))
except:
    lastyeardf = predictiondata.grab_fangraphs_data([lastyear])
    lastyeardf.to_csv('predictions/AllHitting_{}.csv'.format(lastyear))
    lastyeardf = pd.read_csv('predictions/AllHitting_{}.csv'.format(lastyear))


# pick some players of interest based on draft location,
names = ['Mitch Haniger','Jesse Winker','Kyle Schwarber','Frank Schwindel','Adam Duvall','Randal Grichuk']

for name in names:
    try:
        print(name,lastyeardf['R'][lastyeardf['Name']==name].values[0],lastyeardf['RBI'][lastyeardf['Name']==name].values[0],lastyeardf['PA'][lastyeardf['Name']==name].values[0])
        print(name,(lastyeardf['R'][lastyeardf['Name']==name].values[0]+lastyeardf['RBI'][lastyeardf['Name']==name].values[0])/lastyeardf['PA'][lastyeardf['Name']==name].values[0])
    except:
        print(name)
#lastyeardf['PA'][lastyeardf['Name']==name].values

"""
printfile = 'predictions/predictit_b'+savedate+'.dat'

A = np.genfromtxt(printfile,\

                  dtype={'names':       ("Name", "HR", "eHR","pHR",\
                                         "H","eH","pH",\
                                         "AB", "eAB","pAB",\
                                         "SB","eSB","pSB",\
                                         "RBI","eRBI","pRBI",\
                                         "R","eR","pR",\
                                        "PA","Afac"),\
                             'formats': ('S20',  'f4',   'f4','f4',\
                                         'f4',  'f4',  'f4','f4','f4','f4',\
                                         'f4',  'f4',  'f4',  'f4','f4','f4',\
                                         'f4',  'f4',  'f4', 'f4','f4')},\
                 delimiter=',')
"""
