
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# obtain the data
import src.predictiondata as predictiondata

trainingyear,testyear = 2022,2023

trainingdf = pd.read_csv('predictions/AllPitching_{}.csv'.format(trainingyear))
testdf     = pd.read_csv('predictions/AllPitching_{}.csv'.format(testyear))



# match pitchers between the two years
trainingip = []
testip = []
minG = 5

# reprocess to include only SPs
trainingdfSP = trainingdf.loc[trainingdf['GS']>minG]
trainingdfSP.to_csv('data/StartingPitchers_{}.csv'.format(trainingyear))

testdfSP = testdf.loc[testdf['GS']>minG]
testdfSP.to_csv('data/StartingPitchers_{}.csv'.format(testyear))


for n in range(0,len(trainingdf['Name'])):
    name = trainingdf['Name'].values[n]
    testdfP = testdf.loc[testdf['Name']==name]
    try:
        if (((trainingdf['GS'].values[n]>minG))&(testdfP['GS'].values[0]>minG)):
            print(name,trainingdf['IP'].values[n],testdfP['IP'].values[0])
            trainingip.append(trainingdf['IP'].values[n])
            testip.append(testdfP['IP'].values[0])
        else:
            print(name,' missing min G')
    except:
        print(name)


trainingip,testip = np.array(trainingip),np.array(testip)
r = np.corrcoef(trainingip,testip)[0, 1]
print("Pearson correlation coefficient from previous year:", r)

# this fit is completely noise dominated (and formally fails owing to being unable to calculate std of a fixed-value array)
r = np.corrcoef(165*np.ones(testip.size)+np.random.normal(0,0.1,size=testip.size),testip)[0, 1]
print("Pearson correlation coefficient from 165:", r)

"""
for ip in np.linspace(10,200,100):
    r = np.corrcoef(ip*np.ones(testip.size)+np.random.normal(0,0.1,size=testip.size),testip)[0, 1]
    print("Pearson correlation coefficient from {}".format(ip), r)
"""
