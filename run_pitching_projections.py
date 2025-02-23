
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import scipy
import scipy.stats as ss

# set up the parameters
analysis_year         = 2025 # what year are we projecting?
savedate              = '022325' # arbitrary tag for saving files
nclusters             = 12 # how many archetypes are there?
weight_distribution   = [0.5, 0.3, 0.13, 0.07] # how much do the past four years contribute?
regression_factor     = 0.8 # how much regression to the mean?
err_regression_factor = 1.5 # how much uncertainty in the regression to the mean?

year_weights = dict()
for year in range(analysis_year-1,analysis_year-5,-1):
    year_weights[year] = weight_distribution[analysis_year-year-1]


# obtain the data: this should be changed to be statscraping
import mlbstatscraping as mss

years = range(analysis_year-4,analysis_year)
for year in years:
    PitchingDF = mss.get_fangraphs_data('pitching',[year])
    PitchingDF.to_csv('data/AllPitching_{}.csv'.format(year),index=False)
    lastyeardf = PitchingDF
    if year == analysis_year-4:
        df = PitchingDF
    else:
        df = pd.concat([df,PitchingDF])



# add optional age adjustments
import src.ageregression as ageregression
year_weights_penalty, age_penalty_slope, age_pivot = ageregression.return_age_factors()


# set up saves predictions 2023
#http://www.espn.com/fantasy/baseball/flb/story?page=REcloserorgchart

closers = [b'Felix Bautista',b'Kenley Jansen',b'Clay Holmes',b'Pete Fairbanks',b'Jordan Romano',\
          b'Kendall Graveman',b'Emmanuel Clase',b'Alex Lange',b'Scott Barlow',b'Jorge Lopez',\
          b'Ryan Pressly',b'Carlos Estevez',b'Trevor May',b'Paul Sewald',b'Jose Leclerc',\
          b'Raisel Iglesias',b'Dylan Floro',b'Edwin Diaz',b'Seranthony Dominguez',b'Kyle Finnegan',\
          b'Brad Boxberger',b'Alexis Diaz',b'Devin Williams',b'David Bednar',b'Ryan Helsley',\
          b'Mark Melancon',b'Daniel Bard',b'Evan Phillips',b'Josh Hader',b'Camilo Doval']

next_up = [b'Cionel Perez',b'Chris Martin',b'Jonathan Loaisiga',b'Jason Adam',b'Erik Swanson',\
          b'Aaron Bummer',b'James Karinchak',b'Jose Cisnero',b'Aroldis Chapman',b'Jhoan Duran',\
          b'Rafael Montero',b'Jimmy Herget',b'Zach Jackson',b'Andres Munoz',b'Johnathan Hernandez',\
          b'Joe Jimenez',b'Tanner Scott',b'David Robertson',b'Craig Kimbrel',b'Carl Edwards Jr.',\
          b'Brandon Hughes',b'Lucas Sims',b'Matt Bush',b'Wil Crowe',b'Giovanny Gallegos',\
          b'Kevin Ginkel',b'Pierce Johnson',b'Daniel Hudson',b'Robert Suarez',b'Taylor Rogers',\
          ]

tweaks = [b'Josh Hader',b'Dellin Betances',b'Zach Britton']

closers = [b'Paul Sewald',b'Raisel Iglesias',b'Craig Kimbrel',b'Kenley Jansen',b'Adbert Alzolay',b'John Brebbia',b'Alexis Diaz',b'Emmanuel Clase',b'Justin Lawrence',b'Alex Lange',b'Josh Hader',b'Will Smith',b'James McArthur',b'John McMillon',b'John Schreiber',b'Carlos Estevez',b'Robert Stephenson',b'Evan Phillips',b'Tanner Scott',b'Devin Williams',b'Jhoan Duran',b'Edwin Diaz',b'Clay Holmes',b'Mason Miller',b'Lucas Erceg',b'Trevor Gott',b'Dany Jimenez',b'Jose Alvarado',b'Jeff Hoffman',b'David Bednar',b'Ryan Helsley',b'Robert Suarez',b'Camilo Doval',b'Andres Munoz',b'Pete Fairbanks',b'Jose Leclerc',b'David Robertson',b'Jordan Romano',b'Kyle Finnegan',b'Hunter Harvey']
next_up = [b'Kevin Ginkel',b'Miguel Castro',b'A.J. Minter',b'Pierce Johnson',b'Yennier Cano',b'Danny Coulombe',b'Chris Martin',b'Josh Winckowski',b'Hector Neris',b'Julian Merryweather',b'Jesse Chavez',b'Bryan Shaw',b'Emilio Pagan',b'Lucas Sims',b'Scott Barlow',b'Trevor Stephan',b'Tyler Kinley',b'Daniel Bard',b'Jason Foley',b'Andrew Chafin',b'Ryan Pressly',b'Bryan Abreu',b'Matt Moore',b'Brusdar Graterol',b'Joe Kelly',b'Andrew Nardi',b'AJ Puk',b'Joel Payamps',b'Trevor Megill',b'Griffin Jax',b'Brock Stewart',b'Addam Ottavino',b'Drew Smith',b'Tommy Kahnle',b'Jonathan Loaisiga',b'Gregory Soto',b'Aroldis Chapman',b'Colin Holderman',b'Giovanny Gallegos',b'JoJo Romero',b'Yuki Matsui',b'Enyel De Los Santos',b'Tyler Rogers',b'Taylor Rogers',b'Matt Brash',b'Gregory Santos',b'Jason Adam',b'Colin Poche',b'Josh Sborz',b'Erik Swanson',b'Tim Mayza',b'Tanner Rainey']
tweaks = []



# use 12 clusters
import src.makeclusters as makeclusters
year_df,stereotype_df,dfnew,hitter_cluster_centroid_df = makeclusters.create_pitching_clusters(df,nclusters,years)


# now we need a pitching projection
import src.totalbattersfaced as totalbattersfaced
IPDict1 = totalbattersfaced.get_ip_predictions(np.array(dfnew['Name'].values),lastyeardf)

# make a threshold that only gives starters
import src.projectplayers as projectplayers

pls = np.unique(np.array(list(df['Name'])))

minG = 5
pls1 = np.unique(np.array(list(df['Name'].loc[((df['GS'].astype('float')>minG)&(df['Year']==2024))])))
pls2 = np.unique(np.array(list(df['Name'].loc[((df['GS'].astype('float')>minG)&(df['Year']==2023))])))
pls3 = np.unique(np.array(list(df['Name'].loc[((df['GS'].astype('float')>minG)&(df['Year']==2022))])))
pls4 = np.unique(np.array(list(df['Name'].loc[((df['GS'].astype('float')>minG)&(df['Year']==2021))])))

# select closers
minG = 1
pls1 = np.unique(np.array(list(df['Name'].loc[((df['GS'].astype('float')<minG)&(df['Year']==2023))])))
pls2 = np.unique(np.array(list(df['Name'].loc[((df['GS'].astype('float')<minG)&(df['Year']==2022))])))
pls3 = np.unique(np.array(list(df['Name'].loc[((df['GS'].astype('float')<minG)&(df['Year']==2021))])))
pls4 = np.unique(np.array(list(df['Name'].loc[((df['GS'].astype('float')<minG)&(df['Year']==2020))])))


pls = np.unique(np.concatenate([pls1,pls2,pls3,pls4]))

IPDict = dict()
for pl in pls:
    try:
        IPDict[pl] = float(IPDict1[pl][0])
    except:
        IPDict[pl] = 25.


IPDict = dict()
IPDF = mss.get_fantasypros_projections('pitchers',preseason=True)
for pl in pls:
    try:
        IPDict[pl] = float(IPDF['IP'][IPDF['Player']==(pl+' ')][0])
    except:
        print('Failed for {}'.format(pl))
        IPDict[pl] = 25.


#pls = np.unique(np.array(list(df['Name'])))

print('Projecting {} players'.format(pls.size))

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

totrank,valrank = rprint.make_totrank_pitching(A,era,eera,whip,ewhip,ww,svals)


fantasy_stats = ['HR', 'ER', 'BB', 'H', 'SO']
xvals = np.linspace(0.,350.,2000)
LDict,MDict,HDict = rprint.make_mid_min_max(A,totrank,fantasy_stats,xvals,simple=True)

svals,esvals = pmodels.make_saves(A,totrank,closers,next_up,tweaks)

# rerank with Saves
# and weight categories again
# [ip,so,era,whip,w,svals]
weights = [1.0,2.0,1.0,1.0,0.5,1.0]
weights = [1.0,2.0,0.5,1.0,0.5,1.0] # a
weights = [1.0,2.0,1.0,1.0,0.0,1.0] # b
weights = [0.5,2.0,1.0,1.0,0.0,1.0] # c

totrank,valrank = rprint.make_totrank_pitching(A,era,eera,whip,ewhip,ww,svals,weights=weights)


printfile = 'predictions/pitcher_predictions'+savedate+'.tbl'
rprint.print_html_ranks_pitching(printfile,A,totrank,LDict,MDict,HDict,era,eera,whip,ewhip,ww,eww,svals,esvals)

printfile = 'predictions/pitcher_predictions_'+savedate+'.csv'
#rprint.print_csv_ranks_pitching(printfile,A,totrank,valrank,LDict,MDict,HDict,era,eera,whip,ewhip,ww,eww,svals,esvals)
rprint.print_csv_ranks_pitching(printfile,A,totrank,LDict,MDict,HDict,era,eera,whip,ewhip,ww,eww,svals,esvals)
