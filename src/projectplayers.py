
import pandas as pd
import numpy as np

def predict_players(pls,years,printfile,AgeDict,PADict,df,hitter_cluster_centroid_df,year_weights,year_weights_penalty,regression_factor,err_regression_factor,age_penalty_slope,age_pivot):

    ShouldProject = []
    f = open(printfile,'w')

    fantasy_stats=['HR', 'H', 'AB', 'SB', 'RBI','R']

    for pl in pls:
        #print(pl)
        pstats = np.zeros(6)
        perr = np.zeros(6)
        pskew = np.zeros(6)
        pa = 0.
        yrsum = 0.
        yrsum_denom = 0.
        nyrs = 0.
        ip_s = 0.
        ab = -1.


        # rip these guys
        skiplist = ['Jose Fernandez','Yordano Ventura', 'Jung Ho Kang']

        if pl in skiplist:
            continue

        # match to stolen IPs
        try:
            #pa = ST['pa'][namelist==pl][0]
            pa = PADict[pl][0]
            #pa = df['PA'][(df['Name']==pl) & (df['Year']==2021.0) ].values[0]
            #print(pa)
        except:
            pass
            #print(pl)
        #print(ip_s)

        # try to get ages
        #print(pl)
        try:
            age = float(AgeDict[pl])
        except:
            pass
            #print('No age for', pl)
            # default to no penalties
            age = 25.0

        #for year in [2017.0,2018.0,2019.0,2020.0]:
        for yrin in years:
        #for year in [2018.0,2019.0,2020.0,2021.0]:

            year = int(yrin)

            try:
                v = list(df['Value Cluster'][(df['Name']==pl) & (df['Year']==year) ])[0]

                for indx,stat in enumerate(fantasy_stats):
                    statcen = list(hitter_cluster_centroid_df['{0}.Centroid'.format(stat)]\
                          [hitter_cluster_centroid_df['Value Cluster']==v])[0]
                    statnorm = list(df['{0}.Normalize'.format(stat)][(df['Name']==pl) & (df['Year']==year) ])[0]

                    # 0.5 is the regression factor for the difference
                    pstats[indx] += year_weights[year] * ( regression_factor*(statnorm-statcen) + statcen)
                    perr[indx] += year_weights[year] * ( err_regression_factor*(statnorm-statcen))
                    pskew[indx] += year_weights[year] * np.sqrt(err_regression_factor)*((statnorm-statcen)*(statnorm-statcen))

                yrsum += year_weights[year]
                yrsum_denom += year_weights[year]

                # apply an age penalty
                #                  0.1          * (33 - 33)        + (-1)
                age_penalty = age_penalty_slope * ((age-age_pivot) + (year-2018.))
                yrsum -= np.max([age_penalty,0.0])

                nyrs += 1.
                #stereotype_df['{0}.Normalize'.format(stat)] = stereotype_df['{0}.Normalize'.format(stat)] - stereotype_df['{0}.Normalize'.format(stat)] % 1
            except:
                #yrsum += year_weights_penalty[year]
                yrsum_denom -= year_weights_penalty[year]
            #print(yrsum)

        if nyrs < 0.5: nyrs=1000.

        adj_fac = 1.#ip/ip_s
        #print(adj_fac)
        #print(yrsum_denom,pa)
        if (yrsum_denom <= 0.) & (pa > 100.):
            #print(pl,pa)
            ShouldProject.append(pl)
            continue

        #print(pl,yrsum,yrsum_denom)
        #print(pskew-(perr*perr))

        #print(ab)
        if pa > 100:
            print(pl,end=', ',file=f)
            for indx,p in enumerate(pstats):
                print(p,end=', ',file=f)

                # put in a Poisson floor for counting stats
                errval = np.nanmax([np.abs(perr[indx]),np.sqrt(p),1.])
                #print(perr[indx],end=', ',file=f)
                print(errval,end=', ',file=f)
                print(pskew[indx]-(perr[indx]*perr[indx]),end=', ',file=f)

            print(int(pa),end=', ',file=f)
            print(yrsum/yrsum_denom,end=', ',file=f)

            print('',file=f)


    f.close()
    return ShouldProject



def predict_pitchers(pls,years,printfile,AgeDict,IPDict,df,hitter_cluster_centroid_df,year_weights,year_weights_penalty,regression_factor,err_regression_factor,age_penalty_slope,age_pivot):
    ShouldProject = []


    # rip these guys
    skiplist = ['Jose Fernandez','Yordano Ventura']

    fantasy_stats = ['HR', 'ER', 'BB', 'H', 'SO']

    f = open(printfile,'w')

    for pl in pls:
        #print(pl)
        pstats = np.zeros(5)
        perr = np.zeros(5)
        pskew = np.zeros(5)
        tbf = 0.
        yrsum = 0.
        yrsum_denom = 0.
        nyrs = 0.
        ip_s = 0.

        if pl in skiplist:
            continue

        # match to IP predictions
        try:
            #ip = ST['ip'][namelist==pl][0]
            ip = IPDict[pl]#df['IP'][(df['Year']==2021.0) & (df['Name']==pl)].values[0]
        except:
            pass
            #print(pl)
            ip = 5.
        #print(ip_s)
        #print('Found {}IP for {}'.format(ip,pl))

        try:
            age = float(AgeDict[pl])
        except:
            #pass
            #print('No age for', pl)
            # default to no penalties
            age = 25.0

        #for year in [2017.0,2018.0,2019.0,2020.0]:

        for yrin in years:

            year = int(yrin)
            #print(year)
            #print(df['Value Cluster'][(df['Name']==pl) & (df['Year']==year) ])
            try:
                v = list(df['Value Cluster'][(df['Name']==pl) & (df['Year']==year) ])[0]
                tbf += list(df['TBF'][(df['Name']==pl) & (df['Year']==year) ])[0]
                ip_s += list(df['IP'][(df['Name']==pl) & (df['Year']==year) ])[0]
                #print(tbf,ip)
                #print(v)
                #print(v.values())
                for indx,stat in enumerate(fantasy_stats):

                    statcen = list(hitter_cluster_centroid_df['{0}.Centroid'.format(stat)]\
                          [hitter_cluster_centroid_df['Value Cluster']==v])[0]
                    statnorm = list(df['{0}.Normalize'.format(stat)][(df['Name']==pl) & (df['Year']==year) ])[0]
                    #print(statnorm-statcen)


                    # 0.5 is the regression factor for the difference
                    pstats[indx] += year_weights[year] * ( regression_factor*(statnorm-statcen) + statcen)
                    perr[indx] += year_weights[year] * ( err_regression_factor*(statnorm-statcen))
                    pskew[indx] += year_weights[year] * np.sqrt(err_regression_factor)*((statnorm-statcen)*(statnorm-statcen))
                yrsum += year_weights[year]
                yrsum_denom += year_weights[year]

                # apply an age penalty
                #                  0.1          * (33 - 33)        + (-1)
                age_penalty = age_penalty_slope * ((age-age_pivot) + (year-2018.))


                yrsum -= np.max([age_penalty,0.0])
                nyrs += 1.
                #stereotype_df['{0}.Normalize'.format(stat)] = stereotype_df['{0}.Normalize'.format(stat)] - stereotype_df['{0}.Normalize'.format(stat)] % 1
            except:
                #yrsum += year_weights_penalty[year]
                #print('failed year for {}: {}'.format(pl,year))
                yrsum_denom += year_weights_penalty[year]
                age_penalty = age_penalty_slope * ((age-age_pivot) + (year-2018.))
                yrsum -= np.max([age_penalty,0.0])
            #print('Running yrsum for {}: {}, {}'.format(pl,year,yrsum))

        if nyrs < 0.5: nyrs=1000.

        adj_fac = 1.#ip/ip_s
        #print(adj_fac)

        #print(pl,nyrs,yrsum)

        if yrsum <0.1:
            #print(pl)
            yrsum = 0.1
            yrsum_denom = 0.15

        if (yrsum_denom <0.) & (ip>25):
            ShouldProject.append(pl)
            continue

        if ip > 25:

            print(pl,end=', ',file=f)
            for indx,p in enumerate(pstats):
                print(int(adj_fac*np.abs(np.round((p*ip/(100.))/yrsum,2))),end=', ',file=f)
                print(int(adj_fac*np.abs(np.round((perr[indx]*ip/(100.))/yrsum,2))),end=', ',file=f)
                print(pskew[indx] - (perr[indx]*perr[indx]),end=', ',file=f)

            print(int(adj_fac*np.abs(np.round((tbf/(nyrs))*yrsum_denom,2))),end=', ',file=f)
            print(int(np.abs(np.round((ip_s/(nyrs))*yrsum_denom,2))),end=', ',file=f)
            print(int(ip),end=', ',file=f)
            print(yrsum/yrsum_denom,end=', ',file=f)



            #print(int(np.abs(np.round((tbf/(nyrs))*yrsum,2))),end=', ')
            #print(int(np.abs(np.round((ip/(nyrs))*yrsum,2))),end=', ')

            print('',file=f)

        else:
            pass
        #print('Skipping projections for ',pl)


    f.close()

    return ShouldProject
