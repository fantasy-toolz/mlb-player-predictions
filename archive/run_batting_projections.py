"""

may need to install
pip install html5lib

"""


import numpy as np
import pandas as pd

# set up the parameters
analysis_year         = 2025 # what year are we projecting?
savedate              = '022325' # arbitrary tag for saving files
nclusters             = 12 # how many archetypes are there?
weight_distribution   = [0.5, 0.3, 0.13, 0.07] # how much do the past four years contribute?
regression_factor     = 0.8 # how much regression to the mean?
err_regression_factor = 0.8 # how much uncertainty in the regression to the mean?

year_weights = dict()
for year in range(analysis_year-1,analysis_year-5,-1):
    year_weights[year] = weight_distribution[analysis_year-year-1]


# obtain the data: this should be changed to be statscraping
import mlbstatscraping as mss

years = range(analysis_year-4,analysis_year)
for year in years:
    HittingDF = mss.get_fangraphs_data('hitting',[year])
    HittingDF.to_csv('data/AllHitting_{}.csv'.format(year),index=False)
    lastyeardf = HittingDF
    if year == analysis_year-4:
        df = HittingDF
    else:
        df = pd.concat([df,HittingDF])


# use 12 clusters
import src.makeclusters as makeclusters
year_df,stereotype_df,dfnew,hitter_cluster_centroid_df = makeclusters.create_hitting_clusters(df,nclusters,years)
pls = np.unique(np.array(list(df['Name'])))


# get plate appearances
import src.plateappearances as plateappearances
#PADict = plateappearances.get_plate_appearances(pls)
PADict = plateappearances.forecast_600(pls)


PADF = mss.get_fantasypros_projections('hitters',preseason=True)
PADF['PA'] = PADF['AB'] + PADF['BB']
for plr in PADict.keys():
    try:
        paentry = PADF['PA'][PADF['Player']==plr+' '].values[0]
        PADict[plr] = int(paentry)
        print(paentry)
    except:
        PADict[plr] = 10

# add optional age adjustments
import src.ageregression as ageregression
year_weights_penalty, age_penalty_slope, age_pivot = ageregression.return_age_factors()

# set up the projections
import src.projectplayers as projectplayers


printfile = 'predictions/hitter_predictions_{}_'.format(analysis_year)+savedate+'.dat'

ShouldProject = projectplayers.predict_players(pls,years,printfile,dict(),PADict,dfnew,hitter_cluster_centroid_df,year_weights,year_weights_penalty,regression_factor,err_regression_factor,age_penalty_slope,age_pivot)



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
totrank,sumrank = rprint.roto_rank(A)

# create error bands
fantasy_stats=['HR', 'H', 'AB', 'SB', 'RBI','R']
xvals = np.linspace(0.,120.,1000)
LDict,MDict,HDict = rprint.make_mid_min_max(A,totrank,fantasy_stats,xvals)

# make html table
printfile = 'predictions/{}/batter_predictions_{}.tbl'.format(analysis_year,savedate)
rprint.print_html_ranks(printfile,A,totrank,LDict,MDict,HDict)

# make easier to read csv
printfile = 'predictions/{}/batter_predictions_{}.csv'.format(analysis_year,savedate)
rprint.print_csv_ranks(printfile,A,totrank,sumrank,LDict,MDict,HDict)
