
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import scipy
import scipy.stats as ss


savedate = '022121'
savedate = '030622'
savedate = '010223'
savedate = '020923'
savedate = '111923'
savedate = '112723'
savedate = '122623c'
savedate = '012824'
savedate = '022624'

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

year_weights = {}
year_weights[2019.0] = 0.07
year_weights[2020.0] = 0.13
year_weights[2021.0] = 0.3
year_weights[2022.0] = 0.5

year_weights = {}
year_weights[2020.0] = 0.07
year_weights[2021.0] = 0.13
year_weights[2022.0] = 0.3
year_weights[2023.0] = 0.5

# penalty if missing
year_weights_penalty = {}
year_weights_penalty[2017.0] = 0.00
year_weights_penalty[2018.0] = 0.00
year_weights_penalty[2019.0] = 0.00
year_weights_penalty[2020.0] = 0.00
year_weights_penalty[2021.0] = 0.00
year_weights_penalty[2022.0] = 0.00
year_weights_penalty[2023.0] = 0.00

regression_factor = 0.8
err_regression_factor = 1.5

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

# obtain the data
import src.predictiondata as predictiondata

minyear,maxyear = 2017,2021
minyear,maxyear = 2018,2022
minyear,maxyear = 2019,2023
minyear,maxyear = 2020,2024
minyear,maxyear = 2023,2024

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
year_df,stereotype_df,dfnew,hitter_cluster_centroid_df = makeclusters.create_pitching_clusters(df,nclusters,years)


#consider age factors
# for hitters, call the falloff at age 33:
# de-weight anything after 33 with a penalty increasing with age

age_penalty_slope = 0.025 # I think 0.1 is AGGRESSIVE
age_pivot = 33.0

# now we need a pitching projection
"""
ST = np.genfromtxt('data/Stolen_IPs_0216.csv',delimiter=',',dtype=[('uid','i4'),('ip','f4'),('name','S20')],skip_header=1)
"""
ST = pd.read_csv('data/consolidated-ip-estimate-2024.csv')

print(ST['Player'])

namelist = np.array(ST['Player'].values)
IPDict = dict()
for name in namelist:
    try:
        IPDict[name] = ST['regression'][namelist==name].values[0]
    except:
        print("trouble for {}".format(name))

print(IPDict)
#IPDict = dict()

for name in lastyeardf['Name']:
    try:
        if IPDict[name] > 0.0:
            print('Good IP for {}'.format(name))
            continue
    except:
        print("going on for {}".format(name))
    try:
        IPDict[name] = lastyeardf['IP'][lastyeardf['Name']==name].values[0]
    except:
        IPDict[name] = 60. # set the closer default


print(IPDict['Zack Greinke'])
print(IPDict['Jose Berrios'])

#print(IPDict.keys())
"""
print(lastyeardf.keys())

IPDict = dict()
for name in namelist:
    try:
        IPDict[name] = lastyeardf['IP'][lastyeardf['Name']==name].values
    except:
        IPDict[name] = 25.
"""

# make a threshold that only gives starters
import src.projectplayers as projectplayers

pls = np.unique(np.array(list(df['Name'])))

minG = 5
pls1 = np.unique(np.array(list(df['Name'].loc[((df['GS']>minG)&(df['Year']==2023))])))
pls2 = np.unique(np.array(list(df['Name'].loc[((df['GS']>minG)&(df['Year']==2022))])))
pls3 = np.unique(np.array(list(df['Name'].loc[((df['GS']>minG)&(df['Year']==2021))])))
pls4 = np.unique(np.array(list(df['Name'].loc[((df['GS']>minG)&(df['Year']==2020))])))

# select closers
minG = 1
pls1 = np.unique(np.array(list(df['Name'].loc[((df['GS']<minG)&(df['Year']==2023))])))
pls2 = np.unique(np.array(list(df['Name'].loc[((df['GS']<minG)&(df['Year']==2022))])))
pls3 = np.unique(np.array(list(df['Name'].loc[((df['GS']<minG)&(df['Year']==2021))])))
pls4 = np.unique(np.array(list(df['Name'].loc[((df['GS']<minG)&(df['Year']==2020))])))


pls = np.unique(np.concatenate([pls1,pls2,pls3,pls4]))

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
