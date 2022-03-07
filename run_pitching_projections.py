
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import scipy
import scipy.stats as ss


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

# penalty if missing
year_weights_penalty = {}
year_weights_penalty[2017.0] = 0.00
year_weights_penalty[2018.0] = 0.00
year_weights_penalty[2019.0] = 0.00
year_weights_penalty[2020.0] = 0.00
year_weights_penalty[2021.0] = 0.00

regression_factor = 0.8
err_regression_factor = 1.5

# set up saves predictions 2021
#http://www.espn.com/fantasy/baseball/flb/story?page=REcloserorgchart

closers = [b'Hunter Harvey',b'Matt Barnes',b'Aroldis Chapman',b'Diego Castillo',b'Kirby Yates',\
          b'Liam Hendriks',b'James Karinchak',b'Bryan Garcia',b'Greg Holland',b'Taylor Rogers',\
          b'Ryan Pressly',b'Raisel Iglesias',b'Jake Diekman',b'Rafael Montero',b'Jose Leclerc',\
          b'Will Smith',b'Anthony Bass',b'Edwin Diaz',b'Hector Neris',b'Brad Hand',\
          b'Craig Kimbrel',b'Lucas Sims',b'Josh Hader',b'Richard Rodriguez',b'Alex Reyes',\
          b'Stefan Crichton',b'Scott Oberg',b'Kenley Jansen',b'Drew Poneranz',b'Reyes Moronta']

next_up = [b'Dillon Tate',b'Darwinzon Hernandez',b'Zach Britton',b'Nick Anderson',b'Jordan Romano',\
          b'Aaron Bummer',b'Nick Wittgren',b'Buck Farmer',b'Scott Barlow',b'Tyler Duffey',\
          b'Enoli Paredes',b'Mike Mayers',b'Jordan Weems',b'Yohan Ramirez',b'Johnathan Hernandez',\
          b'Chris Martin',b'Yimi Garcia',b'Trevor May',b'Archie Bradley',b'Daniel Hudson',\
          b'Rowan Wick',b'Amir Garrett',b'Devin Williams',b'Nick Burdi',b'Giovanny Gallegos',\
          b'Kevin Ginkel',b'Daniel Bard',b'Blake Treinen',b'Emilio Pagan',b'Tyler Rogers',\
          ]

tweaks = [b'Josh Hader',b'Dellin Betances',b'Zach Britton']


# obtain the data
import src.predictiondata as predictiondata

minyear,maxyear = 2017,2021
minyear,maxyear = 2018,2022

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


# use 12 clusters
import src.makeclusters as makeclusters
year_df,stereotype_df,dfnew,hitter_cluster_centroid_df = makeclusters.create_pitching_clusters(df,0,nclusters,years)


#consider age factors
# for hitters, call the falloff at age 33:
# de-weight anything after 33 with a penalty increasing with age

age_penalty_slope = 0.025 # I think 0.1 is AGGRESSIVE
age_pivot = 33.0


ST = np.genfromtxt('data/Stolen_IPs_0216.csv',delimiter=',',\
                  dtype=[('uid','i4'),('ip','f4'),('name','S20')],skip_header=1)

namelist = np.array([x.decode() for  x in ST['name']])
IPDict = dict()
for name in namelist:
    IPDict[name] = ST['ip'][namelist==name]

print(lastyeardf.keys())

IPDict = dict()
for name in namelist:
    try:
        IPDict[name] = lastyeardf['IP'][lastyeardf['Name']==name].values
    except:
        IPDict[name] = 25.


import src.projectplayers as projectplayers

pls = np.unique(np.array(list(df['Name'])))

printfile = 'predictions/pitcher_predictions'+savedate+'.dat'

ShouldProject = projectplayers.predict_pitchers(pls,years,printfile,dict(),IPDict,dfnew,hitter_cluster_centroid_df,year_weights,year_weights_penalty,regression_factor,err_regression_factor,age_penalty_slope,age_pivot)
print(ShouldProject)





arr1 = (np.array(list(dfnew['ER'][dfnew['Value Cluster']==4.])).astype('float32'))
arr2 = (np.array(list(dfnew['W'][dfnew['Value Cluster']==4.])).astype('float32'))
arr3 = (np.array(list(dfnew['IP'][dfnew['Value Cluster']==4.])).astype('float32'))

arr1tmp = (np.array(list(dfnew['ER'])))
arr2tmp = (np.array(list(dfnew['W'])))
arr3tmp = (np.array(list(dfnew['IP'])))

ipmin=100. # set the transition from starters to relievers
iplim = np.where(arr3tmp<ipmin)
arr1 = arr1tmp[iplim]
arr2 = arr2tmp[iplim]
arr3 = arr3tmp[iplim]


xval = arr1/(arr3*arr3)

plt.figure()
plt.scatter(xval,arr2,color='black',s=2.)


a = np.polyfit(xval[xval<0.015].astype('float32'),arr2[xval<0.015].astype('float32'),2)
WFIT_RP = np.poly1d(a)
plt.plot(np.linspace(0,0.02,100),WFIT_RP(np.linspace(0,0.02,100)))

plt.xlabel('ER/IP$^2$')
plt.ylabel('W')
plt.ylim(0.,20.)
plt.title('RP Relationship')
plt.tight_layout()
plt.savefig('figures/RPrelationship.png')

iplim = np.where(arr3tmp>ipmin)
arr1 = arr1tmp[iplim]
arr2 = arr2tmp[iplim]
arr3 = arr3tmp[iplim]


xval = arr1/(arr3*arr3)

plt.figure()
plt.scatter(xval,arr2,color='black',s=2.)


a = np.polyfit(xval[xval<0.015].astype('float32'),arr2[xval<0.015].astype('float32'),3)
WFIT_SP = np.poly1d(a)
plt.plot(np.linspace(0,0.02,100),WFIT_SP(np.linspace(0,0.02,100)))

plt.xlabel('ER/IP$^2$')
plt.ylabel('W')
plt.ylim(0.,20.)
plt.title('SP Relationship')
plt.tight_layout()
plt.savefig('figures/SPrelationship.png')



printfile = 'predictions/pitcher_predictions'+savedate+'.dat'

# bring the data back in
A = np.genfromtxt(printfile,\
                  dtype={'names':       ("Name", "HR", "eHR","sHR",\
                                         "ER","eER","sER",\
                                         "BB", "eBB","sBB",\
                                         "H","eH","sH",\
                                         "SO","eSO","sSO",\
                                        "TBF","IPc","IP","Afac"),\
                             'formats': ('S20',  'i4',   'i4', 'f8',\
                                         'i4',  'i4', 'f8',\
                                         'i4','i4',  'f8', \
                                         'i4',  'i4', 'f8',\
                                         'i4',  'i4', 'f8',\
                                         'i4',  'i4',  'i4','f4')},\
                 delimiter=',')

print(np.nanmax(A['SO']))
print(len(A['Name']))

import src.pitchingmodels as pmodels
era,eera,whip,ewhip,ww,eww = pmodels.make_wins_model(A)


import src.rankandprint as rprint

# very first ranking
# create blank svals for first sorting
svals = np.zeros(len(ww))
totrank = rprint.make_totrank_pitching(A,era,eera,whip,ewhip,ww,svals)


fantasy_stats = ['HR', 'ER', 'BB', 'H', 'SO']
xvals = np.linspace(0.,350.,2000)
LDict,MDict,HDict = rprint.make_mid_min_max(A,totrank,fantasy_stats,xvals,simple=True)

svals,esvals = pmodels.make_saves(A,totrank,closers,next_up,tweaks)

# rerank with Saves
totrank = rprint.make_totrank_pitching(A,era,eera,whip,ewhip,ww,svals)


printfile = 'predictions/pitcher_predictions'+savedate+'.tbl'
rprint.print_html_ranks_pitching(printfile,A,totrank,LDict,MDict,HDict,era,eera,whip,ewhip,ww,eww,svals,esvals)

printfile = 'predictions/pitcher_predictions_'+savedate+'.csv'
rprint.print_csv_ranks_pitching(printfile,A,totrank,LDict,MDict,HDict,era,eera,whip,ewhip,ww,eww,svals,esvals)
