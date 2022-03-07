
import numpy as np
import pandas as pd


savedate = '022121'
savedate = '030622'
nclusters=12

# new weights for January 4th 2021
year_weights = {}
year_weights[2017.0] = 0.07
year_weights[2018.0] = 0.13
year_weights[2019.0] = 0.3
year_weights[2020.0] = 0.5

year_weights = {}
year_weights[2018.0] = 0.07
year_weights[2019.0] = 0.13
year_weights[2020.0] = 0.3
year_weights[2021.0] = 0.5
print(year_weights)


# obtain the data
import src.predictiondata as predictiondata

minyear,maxyear = 2018,2022
years = range(minyear,maxyear)
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


# use 12 clusters
import src.makeclusters as makeclusters
year_df,stereotype_df,dfnew,hitter_cluster_centroid_df = makeclusters.create_clusters(df,0,nclusters,years)




# penalty if missing: disabled for now
year_weights_penalty = {}
year_weights_penalty[2017.0] = 0.00
year_weights_penalty[2018.0] = 0.0#5
year_weights_penalty[2019.0] = 0.0#1
year_weights_penalty[2020.0] = 0.0#5
year_weights_penalty[2021.0] = 0.0#5

# set regression factors
regression_factor     = 0.8
err_regression_factor = 0.8

# bring in PA guesses
ST = np.genfromtxt('data/Stolen_PAs_0216.csv',delimiter=',',\
                  dtype=[('uid','i4'),('pa','f4'),('name','S20')],skip_header=1)

namelist = np.array([x.decode() for  x in ST['name']])
PADict = dict()
for name in namelist:
    PADict[name] = ST['pa'][namelist==name]


PADict = dict()
for name in namelist:
    try:
        PADict[name] = lastyeardf['PA'][lastyeardf['Name']==name].values
    except:
        PADict[name] = [200.]


#consider age factors
# for hitters, call the falloff at age 33:
# de-weight anything after 33 with a penalty increasing with age
age_penalty_slope = 0.07 # I think 0.1 is AGGRESSIVE
age_pivot         = 64.0


# set up the projections
import src.projectplayers as projectplayers

pls = np.unique(np.array(list(df['Name'])))

printfile = 'predictions/hitter_predictions'+savedate+'.dat'

ShouldProject = projectplayers.predict_players(pls,years,printfile,dict(),PADict,dfnew,hitter_cluster_centroid_df,year_weights,year_weights_penalty,regression_factor,err_regression_factor,age_penalty_slope,age_pivot)


print(ShouldProject)

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


import src.rankandprint as rprint

# rank players
totrank = rprint.roto_rank(A)

# create error bands
fantasy_stats=['HR', 'H', 'AB', 'SB', 'RBI','R']
xvals = np.linspace(0.,120.,1000)
LDict,MDict,HDict = rprint.make_mid_min_max(A,totrank,fantasy_stats,xvals)

# make html table
printfile = 'predictions/batter_predictions'+savedate+'.tbl'
rprint.print_html_ranks(printfile,A,totrank,LDict,MDict,HDict)

# make easier to read csv
printfile = 'predictions/batter_predictions'+savedate+'.csv'
rprint.print_csv_ranks(printfile,A,totrank,LDict,MDict,HDict)
