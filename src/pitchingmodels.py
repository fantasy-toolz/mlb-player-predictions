
import numpy as np


def make_saves(A,totrank,closers,next_up,tweaks):
    totrank_sort = totrank.argsort()

    save_distribution = np.array([int(np.round(x)) for x in np.linspace(38,17,len(closers))])
    save_uncertainty = 2*np.sqrt(save_distribution)

    next_up_distribution = np.array([int(np.round(x)) for x in np.linspace(17,5,len(next_up))])
    next_up_uncertainty = 2*np.sqrt(next_up_distribution)


    svals = np.zeros(totrank_sort.size,dtype='i4')
    esvals = np.zeros(totrank_sort.size,dtype='i4')

    cl = 0
    for i in range(0,totrank_sort.size):
        pl = A['Name'][totrank_sort[i]]
        if pl in closers:
            print(cl,pl,save_distribution[cl])
            svals[totrank_sort[i]] = int(save_distribution[cl])
            esvals[totrank_sort[i]] = int(save_uncertainty[cl])
            cl+=1
        #print(A['Name'][totrank_sort[i]].decode())



    nu = 0
    for i in range(0,totrank_sort.size):
        pl = A['Name'][totrank_sort[i]]
        if pl in next_up:
            if pl in tweaks:
                svals[totrank_sort[i]] = int(0.5*next_up_distribution[nu])
                esvals[totrank_sort[i]] = int(0.5*next_up_uncertainty[nu])
            else:
                svals[totrank_sort[i]] = int(next_up_distribution[nu])
                esvals[totrank_sort[i]] = int(next_up_uncertainty[nu])
            print(pl,svals[totrank_sort[i]])
            nu+=1
        #print(A['Name'][totrank_sort[i]].decode())


    return svals,esvals


def make_wins_model(A):
    ipmin = 100.
    a = [ 1.29725376e+04, -5.16521652e+02,  6.41559801e+00]
    WFIT_RP = np.poly1d(a)
    a = [-2.61854783e+07 , 7.22277918e+05 ,-6.69910419e+03 , 2.40221020e+01]
    WFIT_SP = np.poly1d(a)


    # adjust categories for the afac
    for cat in ['SO']:
        A[cat] = np.round(A[cat].astype('float') * (1. - np.abs(1. - A['Afac'])),0).astype('int')

    for cat in ['HR','ER','BB','H']:
        A[cat] = np.round(A[cat].astype('float') / (1. - np.abs(1. - A['Afac'])),0).astype('int')


    adj_fac_tmp = A['IP']/A['IPc']

    #np.max(adj_fac)

    adj_fac = 0.*adj_fac_tmp + 1.#np.array([np.min([x,2.]) for x in adj_fac_tmp])


    rearr = (-1.*adj_fac*A['SO']).argsort()
    #for x in rearr:
    #    print('{0:20s}'.format(A['Name'][x].decode()),(adj_fac*A['SO'])[x],'+/-',A['eSO'][x])


    # WHIP
    whip = (A['HR'] + A['H'] + A['BB'])/A['IP']

    # adjust the minima
    ewhip = (np.max([A['eHR'],np.sqrt(A['HR'])],axis=0) + \
             np.max([A['eH'],np.sqrt(A['H'])],axis=0) + \
             np.max([A['eBB'],np.sqrt(A['BB'])],axis=0)\
            )/A['IP']


    rearr = (whip).argsort()
    #for x in rearr:
    #    print('{0:20s}'.format(A['Name'][x].decode()),np.round(whip[x],2),'+/-',np.round(ewhip[x],2))


    # ERA
    era = 9.*(A['ER'])/A['IP']
    eera = 9.*(np.max([A['eER'],np.sqrt(A['ER'])],axis=0))/A['IP']


    rearr = (era).argsort()
    #for x in rearr:
    #    print('{0:20s}'.format(A['Name'][x].decode()),np.round(era[x],2),'+/-',np.round(eera[x],2))


    # for starters
    ww = (np.min([np.zeros(A['IP'].size)+150.,A['IP']],axis=0)/150.)*WFIT_SP(A['ER']/(A['IP']*A['IP']))
    eww = WFIT_SP(15*(np.max([A['eER'],np.sqrt(A['ER'])],axis=0))/(A['IP']*A['IP']))

    print(len(ww))
    # overwrite the relievers
    ww[A['IP']<ipmin] = ((np.min([np.zeros(A['IP'].size)+150.,A['IP']],axis=0)/150.)*WFIT_RP(A['ER']/(A['IP']*A['IP'])))[A['IP']<ipmin]
    eww[A['IP']<ipmin] = (WFIT_RP(15*(np.max([A['eER'],np.sqrt(A['ER'])],axis=0))/(A['IP']*A['IP'])))[A['IP']<ipmin]



    rearr = (-1.*ww).argsort()
    printwins = False
    if printwins:
        for x in rearr:
            print(x,'{0:20s}'.format(A['Name'][x].decode()),int(np.nanmax([ww[x],0])),'+/-',int(np.nanmin([np.nanmax([np.abs(eww[x]),4.]),3.*np.sqrt(np.nanmax([np.abs(ww[x]),4.]))])))

    return era,eera,whip,ewhip,ww,eww
