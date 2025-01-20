import src.predictiondata as predictiondata
import matplotlib.pyplot as plt
import numpy as np

years = [2024]
df1 = predictiondata.grab_fangraphs_data(years)
df1.to_csv('data/2024hitting.csv', index=False)

ST = pd.read_csv('data/estimated-batting-stats-2024.csv')
df1['player'] = df1['Name']

df = pd.merge(df1, ST, on='player')

df['PAdiff'] = (df['PA'].astype('int') - df['PAprediction'].astype('int'))/(df['PA'].astype('int')+1)
print(np.nanmean(df['PAdiff']),np.nanmedian(df['PAdiff']),np.nanstd(df['PAdiff']))

df['PAdiff'] = df['PA'].astype('int') - df['PAprediction'].astype('int')
print(np.nanmean(df['PAdiff']),np.nanmedian(df['PAdiff']),np.nanstd(df['PAdiff']))

plt.plot(np.linspace(0,1.,df['PAdiff'].size),np.sort(df['PAdiff'].values),color='black')
plt.xlabel('fraction')
plt.ylabel('(PA - PAprediction)/PA')


top_10_PAdiff = df.nlargest(10, 'PAdiff')[['player', 'PAdiff','PA','PAprediction']]
print(top_10_PAdiff)

top_10_PAdiff = df.nsmallest(10, 'PAdiff')[['player', 'PAdiff','PA','PAprediction']]
print(top_10_PAdiff)

df['PAdiff'] = np.abs(df['PA'].astype('int') - df['PAprediction'].astype('int'))
top_10_PAdiff = df.nsmallest(10, 'PAdiff')[['player', 'PAdiff','PA','PAprediction']]
print(top_10_PAdiff)

PAdiff = df['PA'].astype('int') - df['PA23'].astype('int')
print(np.nanmean(PAdiff),np.nanmedian(PAdiff),np.nanstd(PAdiff))

PAdiff = df['PA'].astype('int') - df['threeyearavgPA'].astype('int')
print(np.nanmean(PAdiff),np.nanmedian(PAdiff),np.nanstd(PAdiff))


savedate = '021124'
Preds = pd.read_csv('predictions/batter_predictions{}.csv'.format(savedate))

# set up all predictions to have a 'p' in front of them
for key in Preds.keys():
    Preds['p'+key] = Preds[key]
    del(Preds[key])

Preds['Name'] = Preds['pName'] # undo name change
Preds['Name'] = Preds['Name'].str.rstrip() # pesky trailing space

df = pd.merge(df1, Preds, on='Name')

Rdiff = df['R'].astype('int') - df['pR'].astype('int')
print(np.nanmean(Rdiff),np.nanmedian(Rdiff),np.nanstd(Rdiff),np.nanmean(df['peR']))

Rdiff = df['AVG'].astype('float') - df['pAVG'].astype('float')
print(np.nanmean(Rdiff),np.nanmedian(Rdiff),np.nanstd(Rdiff),np.nanmean(df['peAVG']))

Rdiff = df['RBI'].astype('float') - df['pRBI'].astype('float')
print(np.nanmean(Rdiff),np.nanmedian(Rdiff),np.nanstd(Rdiff),np.nanmean(df['peRBI']))

Rdiff = df['SB'].astype('float') - df['pSB'].astype('float')
print(np.nanmean(Rdiff),np.nanmedian(Rdiff),np.nanstd(Rdiff),np.nanmean(df['peSB']))

Rdiff = df['HR'].astype('float') - df['pHR'].astype('float')
print(np.nanmean(Rdiff),np.nanmedian(Rdiff),np.nanstd(Rdiff),np.nanmean(df['peHR']))
