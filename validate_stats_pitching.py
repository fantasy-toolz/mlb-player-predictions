import src.predictiondata as predictiondata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

years = [2024]
df1 = predictiondata.grab_fangraphs_pitching_data(years)
df1.to_csv('data/2024pitching.csv', index=False)

ST = pd.read_csv('data/consolidated-ip-estimate-2024.csv')
df1['Player'] = df1['Name']


FP = pd.read_csv('data/fantasypros-pitchingprojections-2024.csv')
FP = FP.head(200)
FP['Player'] = FP['Player'].apply(lambda x: x.split('\xa0(')[0])

df = pd.merge(df1, FP, on='Player')

df = pd.merge(df1, ST, on='Player')

df['IPdiff'] = (df['IP'].astype('float') - df['Model24'].astype('float'))/(df['IP'].astype('float')+0.3)
print(np.nanmean(df['IPdiff']),np.nanmedian(df['IPdiff']),np.nanstd(df['IPdiff']))

df['IPdiff'] = (df['IP'].astype('float') - df['Model24'].astype('float'))
print(np.nanmean(df['IPdiff']),np.nanmedian(df['IPdiff']),np.nanstd(df['IPdiff']))

df['IPdiff'] = (df['IP'].astype('float') - df['regression'].astype('float'))
print(np.nanmean(df['IPdiff']),np.nanmedian(df['IPdiff']),np.nanstd(df['IPdiff']))


plt.plot(np.linspace(0,1.,df['IPdiff'].size),np.sort(df['IPdiff'].values),color='red',linestyle='--')
plt.xlabel('fraction')
plt.ylabel('(IP - IPprediction)/IP')


top_10_IPdiff = df.nlargest(10, 'IPdiff')[['Player', 'IPdiff','IP','regression','Model24']]
print(top_10_IPdiff)

top_10_IPdiff = df.nsmallest(10, 'IPdiff')[['Player', 'IPdiff','IP','Model24']]
print(top_10_IPdiff)

df['IPdiff'] = np.abs(df['IP'].astype('float') - df['regression'].astype('float'))
top_10_IPdiff = df.nsmallest(10, 'IPdiff')[['Player', 'IPdiff','IP','regression','Model24']]
print(top_10_IPdiff)

PAdiff = df['PA'].astype('int') - df['PA23'].astype('int')
print(np.nanmean(PAdiff),np.nanmedian(PAdiff),np.nanstd(PAdiff))

PAdiff = df['PA'].astype('int') - df['threeyearavgPA'].astype('int')
print(np.nanmean(PAdiff),np.nanmedian(PAdiff),np.nanstd(PAdiff))


savedate = '022624'
Preds = pd.read_csv('predictions/pitcher_predictions_{}.csv'.format(savedate))

# set up all predictions to have a 'p' in front of them
for key in Preds.keys():
    Preds['p'+key] = Preds[key]
    del(Preds[key])

Preds['Name'] = Preds['pName'] # undo name change
Preds['Name'] = Preds['Name'].str.rstrip() # pesky trailing space

df = pd.merge(df1, Preds, on='Name')

df['WHIP'] = (df['BB'].astype('float') + df['H'].astype('float'))/(df['IP'].astype('float')+0.1)

Rdiff = df['W'].astype('int') - df['pW'].astype('int')
print(np.nanmean(Rdiff),np.nanmedian(Rdiff),np.nanstd(Rdiff),np.nanmean(df['peW']))

Rdiff = df['ERA'].astype('float') - df['pERA'].astype('float')
print(np.nanmean(Rdiff),np.nanmedian(Rdiff),np.nanstd(Rdiff),np.nanmean(df['peERA']))

Rdiff = df['WHIP'].astype('float') - df['pWHIP'].astype('float')
print(np.nanmean(Rdiff),np.nanmedian(Rdiff),np.nanstd(Rdiff),np.nanmean(df['peWHIP']))

Rdiff = df['SO'].astype('float') - df['pSO'].astype('float')
print(np.nanmean(Rdiff),np.nanmedian(Rdiff),np.nanstd(Rdiff),np.nanmean(df['peSO']))

Rdiff = df['SV'].astype('float') - df['pS'].astype('float')
print(np.nanmean(Rdiff),np.nanmedian(Rdiff),np.nanstd(Rdiff),np.nanmean(df['peS']))
