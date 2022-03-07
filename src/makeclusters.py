
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def create_clusters(df,indx,ccen,\
                    years,
                    min_pas=100,\
                    fantasy_stats=['HR', 'H', 'AB', 'SB', 'RBI','R'],\
                    denominator='PA',
                    savedir='predictions/'):

    nclustersmax = ccen

    yr = str(int(years[-1])+1)

    StarterCenters = {}
    StereotypeKeys = {}

    #df = df.loc[df['Name']==df['Name'] ]#) & (hitter_eoy_df['wRC+'] != '&nbsp;')]

    # strip some columns
    for column in df.columns[4:]:
        if column != 'wRC+':
            try:
                df[column] = df[column].astype(float)
            except:
                df[column] = df[column].str[:-1]
                df[column] = df[column].astype(float)


    # predictions are only valid for certain numbers of PAs
    df = df.loc[(df['PA']> min_pas)]


    for stat in fantasy_stats:
        if stat=='H':
            df['{0}.Normalize'.format(stat)] = 100.*(df[stat]-df['HR'])/df[denominator]
        else:
            df['{0}.Normalize'.format(stat)] = 100.*(df[stat])/df[denominator]


    # this needs to be made adaptive
    Y = df[['HR.Normalize','H.Normalize',\
        'AB.Normalize','SB.Normalize', 'RBI.Normalize', 'R.Normalize']].values


    # run the actual K-means
    # use the following features:
    # HR, H, AB, SB, RBI, R
    kmeans = KMeans(n_clusters=ccen, random_state=3425)
    kmeans.fit(Y)
    predict = kmeans.predict(Y)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    StarterCenters[ccen] = centroids
    StereotypeKeys[ccen] = {}


    hitter_cluster_centroid_df = pd.DataFrame(centroids, \
                                              columns=['HR.Centroid',\
                                                       'H.Centroid',\
                                                       'AB.Centroid',\
                                                       'SB.Centroid',\
                                                       'RBI.Centroid',\
                                                       'R.Centroid'])#,\
                                                       #'BB.Centroid'])

    hitter_cluster_centroid_df['Tot.Rank']  = 0
    for column in hitter_cluster_centroid_df.columns[:-1]:
        stat = column.split(".")[0]
        # this formula scales relative to overall value
        meanval = np.nanmedian(df['{0}.Normalize'.format(stat)])
        stdval = np.nanstd(df['{0}.Normalize'.format(stat)])

        hitter_cluster_centroid_df['{0}.Rank'.format(stat)]  = (hitter_cluster_centroid_df['{0}.Centroid'.format(stat)] - meanval)/stdval
        hitter_cluster_centroid_df['Tot.Rank'] = hitter_cluster_centroid_df['Tot.Rank'] +hitter_cluster_centroid_df['{0}.Rank'.format(stat)]

    # Now Let's Predict our Clusters
    df['Clusters'] = pd.Series(predict, index=df.index)

    # Let's continue this party by reducing the field of potential stereotypes to players that are going to play.
    short_hitter_df = df.loc[df['PA']>0]

    # Awesome, time to simplify this DataFrame.
    short_hitter_df = short_hitter_df[['Name', 'Year', 'HR.Normalize','H.Normalize', 'AB.Normalize','SB.Normalize', 'RBI.Normalize', 'R.Normalize',\
                                       'Clusters', 'HR','H', 'AB','SB', 'RBI', 'R']]

    # Merge in the centroids for comparisons and find the sum of each stat's deviation for each player.
    short_hitter_df = short_hitter_df.merge(hitter_cluster_centroid_df, right_index = True, left_on='Clusters')
    short_hitter_df['Centroid Diff'] = 0
    hit_fields = ['HR.{0}','H.{0}', 'AB.{0}','SB.{0}', 'RBI.{0}', 'R.{0}']
    #hit_fields = ['HR.{0}','H.{0}', 'AB.{0}','SB.{0}', 'RBI.{0}', 'R.{0}', 'BB.{0}']
    for field in hit_fields:
        short_hitter_df[field.format('Diff')] = abs(short_hitter_df[field.format('Centroid')] - short_hitter_df[field.format('Normalize')])
        short_hitter_df[field.format('Diff')] = short_hitter_df[field.format('Diff')].rank(pct=True)
        short_hitter_df['Centroid Diff'] = short_hitter_df['Centroid Diff'] + short_hitter_df[field.format('Diff')]

    # Now we can use the deviation sums to find the player closest to each centroid and create a DataFrame of stereotype players.
    idx = short_hitter_df.groupby(['Clusters'])['Centroid Diff'].transform(min) == short_hitter_df['Centroid Diff']
    stereotype_df = short_hitter_df[idx].copy()

    # Clean up the DataFrame...
    for stat in fantasy_stats:
        stereotype_df['{0}.Normalize'.format(stat)] = stereotype_df['{0}.Normalize'.format(stat)] - stereotype_df['{0}.Normalize'.format(stat)] % 1
        stereotype_df['{0}'.format(stat)] = stereotype_df['{0}'.format(stat)]

    stereotype_df = stereotype_df[['Clusters', 'Name', 'Year', 'HR', 'H', 'AB', 'SB', 'RBI', 'R']]
    #stereotype_df = stereotype_df[['Clusters', 'Name', 'Year', 'HR', 'H', 'AB', 'SB', 'RBI', 'R','BB']]
    stereotype_df = stereotype_df.sort_values(['Clusters'], ascending = True)


    # Update cluster values
    hitter_cluster_centroid_df['Value Cluster'] = hitter_cluster_centroid_df['Tot.Rank'].rank(ascending      =1, method = 'first')
    hitter_cluster_centroid_df['Clusters'] = hitter_cluster_centroid_df.index
    cluster_equiv = hitter_cluster_centroid_df[['Clusters', 'Value Cluster']]
    stereotype_df = stereotype_df.merge(cluster_equiv, on = 'Clusters', how = 'left')

    df = df.merge(cluster_equiv, on = 'Clusters', how = 'left')

    year_dfs2 = []
    for year in years:
        year_df = df.loc[df['Year'] == year]
        year_df = year_df[['Name', 'Value Cluster']]
        year_df.columns = ['Name', 'Value Cluster {0}'.format(year)]
        year_dfs2.append(year_df)

    year_df = year_dfs2[0]

    for i in year_dfs2[1:]:
        year_df = year_df.merge(i, on = 'Name', how = 'outer')
        # Clean DataFrame by filling nulls with zeros

    for column in year_df.columns[1:]:
        year_df[column] = year_df[column].fillna(0)

    if ccen == nclustersmax:
        year_df.to_csv(savedir+yr+'Clusters_By_Year_starters{}.csv'.format(ccen), index = False)
        df.to_csv(savedir+yr+'All_Player_Data_starters{}.csv'.format(ccen), index = False)
        stereotype_df.to_csv(savedir+yr+'Stereotype_Players_starters{}.csv'.format(ccen), index = False)

    # add the value clusters
    for ii in range(len(stereotype_df['Value Cluster'])):
        StereotypeKeys[ccen][ii] = stereotype_df['Value Cluster'][ii]

    return year_df,stereotype_df,df,hitter_cluster_centroid_df




def create_pitching_clusters(df,indx,ccen,\
                    years,
                    min_pas=100,\
                    savedir='predictions/'):
    StarterCenters = {}
    StereotypeKeys = {}

    n_clusters = ccen
    df = df.loc[df['Name']==df['Name'] ]#) & (hitter_eoy_df['wRC+'] != '&nbsp;')]

    for column in df.columns[4:]:
        if column != 'wRC+':
            try:
                #df[column] = df[column].astype(float)
                df.loc[column] = df[column].astype(float)
            except:
                df.loc[column] = df[column].str[:-1]
                df.loc[column] = df[column].astype(float)

    # if selecting for starters
    #df = df.loc[(df['TBF']> 150) & (df['GS'] > 10) ]

    # if selecting for relievers
    #df = df.loc[(df['TBF']> 100) & (df['GS'] < 10) ]

    # if selecting for starters
    df = df.loc[(df['TBF']> 100)]



    fantasy_stats = ['HR', 'ER', 'BB', 'H', 'SO']

    denominator = 'TBF'
    denominator = 'IP'


    for stat in fantasy_stats:
        if stat=='SO':
            df['{0}.Normalize'.format(stat)] = 100.*df[stat]/df[denominator]
        elif stat=='H':
            df['{0}.Normalize'.format(stat)] = -100.*(df[stat]-df['HR'])/df[denominator]
        elif (stat != 'W') & (stat != 'SV'):
            #df['{0}.Normalize'.format(stat)] = df[stat]*200.0/df['IP']
            df['{0}.Normalize'.format(stat)] = -100.*df[stat]/df[denominator]
        else:
            df['{0}.Normalize'.format(stat)] = 10.*df[stat]/df['G']

    Y = df[['HR.Normalize','ER.Normalize','BB.Normalize','H.Normalize', 'SO.Normalize']].values

    # Set up the clusters
    kmeans = KMeans(n_clusters=ccen, random_state=3425)
    kmeans.fit(Y)
    predict = kmeans.predict(Y)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    StarterCenters[ccen] = centroids
    StereotypeKeys[ccen] = {}

    hitter_cluster_centroid_df = pd.DataFrame(centroids, columns=['HR.Centroid', 'ER.Centroid', 'BB.Centroid', 'H.Centroid', 'SO.Centroid'])

    hitter_cluster_centroid_df['Tot.Rank']  = 0
    for column in hitter_cluster_centroid_df.columns[:-1]:
        stat = column.split(".")[0]
        # this formula scales relative to overall value
        meanval = np.nanmedian(df['{0}.Normalize'.format(stat)])
        stdval = np.nanstd(df['{0}.Normalize'.format(stat)])

        hitter_cluster_centroid_df['{0}.Rank'.format(stat)]  = (hitter_cluster_centroid_df['{0}.Centroid'.format(stat)] - meanval)/stdval
        hitter_cluster_centroid_df['Tot.Rank'] = hitter_cluster_centroid_df['Tot.Rank'] +hitter_cluster_centroid_df['{0}.Rank'.format(stat)]

    # Now Let's Predict our Clusters
    df['Clusters'] = pd.Series(predict, index=df.index)

    # Let's continue this party by reducing the field of potential stereotypes to players that are going to play.
    short_hitter_df = df.loc[df['IP']>0]

    # Awesome, time to simplify this DataFrame.
    short_hitter_df = short_hitter_df[['Name', 'Year', 'HR.Normalize','ER.Normalize', 'BB.Normalize','H.Normalize', 'SO.Normalize','Clusters']]

    # Merge in the centroids for comparisons and find the sum of each stat's deviation for each player.
    short_hitter_df = short_hitter_df.merge(hitter_cluster_centroid_df, right_index = True, left_on='Clusters')
    short_hitter_df['Centroid Diff'] = 0
    hit_fields = ['HR.{0}','ER.{0}', 'BB.{0}','H.{0}', 'SO.{0}']
    for field in hit_fields:
        short_hitter_df[field.format('Diff')] = abs(short_hitter_df[field.format('Centroid')] - short_hitter_df[field.format('Normalize')])
        short_hitter_df[field.format('Diff')] = short_hitter_df[field.format('Diff')].rank(pct=True)
        short_hitter_df['Centroid Diff'] = short_hitter_df['Centroid Diff'] + short_hitter_df[field.format('Diff')]

    # Now we can use the deviation sums to find the player closest to each centroid and create a DataFrame of stereotype players.
    idx = short_hitter_df.groupby(['Clusters'])['Centroid Diff'].transform(min) == short_hitter_df['Centroid Diff']
    stereotype_df = short_hitter_df[idx].copy()

    # Clean up the DataFrame...
    for stat in fantasy_stats:
        stereotype_df['{0}.Normalize'.format(stat)] = stereotype_df['{0}.Normalize'.format(stat)] - stereotype_df['{0}.Normalize'.format(stat)] % 1
    #stereotype_df['BA.Normalize'] = stereotype_df['H.Normalize'] / 600
    stereotype_df = stereotype_df[['Clusters', 'Name', 'Year', 'HR.Normalize', 'ER.Normalize', 'BB.Normalize', 'H.Normalize', 'SO.Normalize']]
    stereotype_df = stereotype_df.sort_values(['Clusters'], ascending = True)


    # Update cluster values
    hitter_cluster_centroid_df['Value Cluster'] = hitter_cluster_centroid_df['Tot.Rank'].rank(ascending      =1, method = 'first')
    hitter_cluster_centroid_df['Clusters'] = hitter_cluster_centroid_df.index
    cluster_equiv = hitter_cluster_centroid_df[['Clusters', 'Value Cluster']]
    stereotype_df = stereotype_df.merge(cluster_equiv, on = 'Clusters', how = 'left')

    df = df.merge(cluster_equiv, on = 'Clusters', how = 'left')

    year_dfs2 = []
    for year in years:
        year_df = df.loc[df['Year'] == year]
        year_df = year_df[['Name', 'Value Cluster']]
        year_df.columns = ['Name', 'Value Cluster {0}'.format(year)]
        year_dfs2.append(year_df)

    year_df = year_dfs2[0]

    for i in year_dfs2[1:]:
        year_df = year_df.merge(i, on = 'Name', how = 'outer')
        # Clean DataFrame by filling nulls with zeros

    for column in year_df.columns[1:]:
        year_df[column] = year_df[column].fillna(0)


    year_df.to_csv(savedir+'Clusters_By_Year_starters{}.csv'.format(ccen), index = False)
    df.to_csv(savedir+'All_Player_Data_starters{}.csv'.format(ccen), index = False)
    stereotype_df.to_csv(savedir+'Stereotype_Players_starters{}.csv'.format(ccen), index = False)

    # add the value clusters
    for ii in range(len(stereotype_df['Value Cluster'])):
        StereotypeKeys[ccen][ii] = stereotype_df['Value Cluster'][ii]

    return year_df,stereotype_df,df,hitter_cluster_centroid_df
