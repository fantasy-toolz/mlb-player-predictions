

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm


mpl.rcParams['xtick.minor.visible'] = False
mpl.rcParams['ytick.minor.visible'] = False


savedate = '030622'
season = '2022'

TeamRosterDF = pd.read_csv('../batting-order/data/team-batting-order-{}.csv'.format(season))

team = 'ARI'
nlineups = len(TeamRosterDF[TeamRosterDF['team']==team])
print(nlineups)

Stats = pd.read_csv('predictions/AllHitting_{}_{}.csv'.format(season,season))
Preds = pd.read_csv('predictions/batter_predictions{}.csv'.format(savedate))

nplayers = Stats['Name'].size

CheckPAS = dict()

for plr in range(0,nplayers):
    team = Stats['Team'][plr]
    if team=='TBR': team='TB'
    if team=='SDP': team='SD'
    if team=='WSN': team='WSH'
    if team=='CHW': team='CWS'
    if team=='SFG': team='SF'
    if team=='KCR': team='KC'
    if team=='- - -':
        print('NO TEAM FOR',Stats['Name'][plr])
        continue
    nlineups = len(TeamRosterDF[TeamRosterDF['team']==team])
    #print(Stats['Name'][plr],team,Stats['G'][plr],nlineups,Stats['G'][plr]/float(nlineups))
    try:
        plrname = Preds['Name'][Preds['Name']==Stats['Name'][plr]].values[0]
        #print(plrname,end=': ')
        #print(Stats['Name'][plr],team,Stats['G'][plr],nlineups,Stats['G'][plr]/float(nlineups))
        pa_per_game = (Stats['PA'][plr]/Stats['G'][plr])
        fraction_of_games = Stats['G'][plr]/float(nlineups)
        sclpas = int(pa_per_game * fraction_of_games * 162.)
        CheckPAS[plrname] = [Preds['PA'][Preds['Name']==Stats['Name'][plr]].values[0],sclpas,Preds['PA'][Preds['Name']==Stats['Name'][plr]].values[0]/float(sclpas)]
        #print(Preds['PA'][Preds['Name']==Stats['Name'][plr]].values[0],sclpas)
    except:
        pass
    #print(Preds['Name'][Preds['Name']==Stats['Name'][plr]])


# reprocess by prediction
sclvals = np.linspace(-1.,1.,40)
sclnum = np.zeros(sclvals.size)

for plr in CheckPAS.keys():
    #print(plr,np.log10(CheckPAS[plr][2]))
    fracshift = (CheckPAS[plr][1]-CheckPAS[plr][0])/CheckPAS[plr][0]
    fracshift = np.log10(1./CheckPAS[plr][2])
    try:
        sclnum[int((fracshift-sclvals[0])/(sclvals[1]-sclvals[0]))] += 1
        if fracshift>0.25:
            print(plr,CheckPAS[plr][0],CheckPAS[plr][1])
    except:
        if fracshift > 1:
            print('Offscale positive!',plr)
        else:
            print('Offscale negative!',plr)
    #sclnum[]

plt.figure()
plt.plot(sclvals,sclnum,drawstyle='steps-mid',color='black')
plt.xlabel('log(Actual/Predicted)')
plt.ylabel('Number')
plt.tight_layout()
plt.savefig('/Users/mpetersen/Downloads/fracshift.png')



"""
Taylor Ward 179 490
Owen Miller 155 530
Ben Gamel 252 589
Harold Ramirez 158 361
Cedric Mullins II 282 722
Daniel Vogelbach 317 564
Alejandro Kirk 267 495
Garrett Cooper 302 586
Tyler Naquin 267 509
Tyler Wade 184 363
Jorge Mateo 169 550
Pavin Smith 238 527
Chas McCormick 194 481
Darin Ruf 225 628
Tyrone Taylor 196 361
Adolis Garcia 178 657
Kyle Farmer 296 583
Joey Bart 151 381
Austin Hedges 226 422
Abraham Toro 173 532
"""
