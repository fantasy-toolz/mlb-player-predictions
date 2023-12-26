
import pandas as pd
import numpy as np


# automatic rankings
import scipy.stats as ss
import scipy

def roto_rank(A):

    sumrank = np.zeros(A['Name'].size)

    for ivec,vec in enumerate([A['HR'],(A['H']+A['HR']),A['RBI'],A['R'],A['SB']]):

        if ivec==0:
            sumrank += 1.4*ss.rankdata(-1.*A['PA']*vec/100.)
        elif ivec==4:
            sumrank += 0.9*ss.rankdata(-1.*A['PA']*vec/100.)
        else:
            sumrank += ss.rankdata(-1.*A['PA']*vec/100.)


    totrank = ss.rankdata(sumrank)
    return totrank



def make_totrank_pitching(A,era,eera,whip,ewhip,ww,svals,weights=[1.0,1.0,1.0,1.0,1.0,1.0]):
    # weights is [ip,so,era,whip,w,svals]
    sumrank = np.zeros(A['Name'].size)

    #for vec in [A['IP'],A['SO'],era,whip,ww]:
    tSO = np.max([A['eSO'],np.sqrt(A['SO'])],axis=0)

    # old strategy: double up on IP to get rid of the low IP guys
    # removed on 26 December 2023
    for ivec,vec in enumerate([A['IP'],A['SO'],era,whip,ww,svals]):
        if (vec[0]==era[0]) | (vec[0]==whip[0]) | (vec[0]==eera[0]) | (vec[0]==ewhip[0]):
            sumrank += weights[ivec]*ss.rankdata(vec)
        else:
            sumrank += weights[ivec]*ss.rankdata(-1.*vec)

    return ss.rankdata(sumrank),sumrank



def normal_pdf(x,mu,sigma):
    """return the probability distribution of a normal curve"""

    prefac = 1./(sigma*np.sqrt(2.*np.pi))
    return prefac * np.exp(-0.5*((x-mu)/sigma)**2.)

def normal_cdf(x,mu,sigma):
    """return the cumulative distribution of a normal curve"""

    return 0.5 * scipy.special.erf((x-mu)/(sigma*np.sqrt(2))) + 0.5

def skew_normal_cdf(x,mu,sigma,alpha):
    """return the cumulative distribution of a skew normal curve

    follow here: https://en.wikipedia.org/wiki/Skew_normal_distribution
    """

    term1 = normal_cdf(x,mu,sigma)

    term2 = scipy.special.owens_t((x-mu)/(sigma),alpha)

    return term1 - 2*term2



def make_mid_min_max(A,totrank,fantasy_stats,xvals,simple=False):


    LDict = {}
    HDict = {}
    MDict = {}

    for st in fantasy_stats:
        LDict[st] = np.zeros(totrank.size)
        HDict[st] = np.zeros(totrank.size)
        MDict[st] = np.zeros(totrank.size)


    for pl,indx in enumerate((totrank).argsort()):

        for st in fantasy_stats:

            if simple:
                LDict[st][indx] = xvals[np.abs(skew_normal_cdf(xvals,A[st][indx],\
                                                               np.nanmax([A['e'+st][indx],np.sqrt(A[st][indx])]),\
                                                               A['s'+st][indx]-1.)-0.2).argmin()]
                MDict[st][indx] = xvals[np.abs(skew_normal_cdf(xvals,A[st][indx],\
                                                               np.nanmax([A['e'+st][indx],np.sqrt(A[st][indx])]),\
                                                               A['s'+st][indx]-1.)-0.5).argmin()]
                HDict[st][indx] = xvals[np.abs(skew_normal_cdf(xvals,A[st][indx],\
                                                               np.nanmax([A['e'+st][indx],np.sqrt(A[st][indx])]),\
                                                               A['s'+st][indx]-1.)-0.8).argmin()]
            else:
                LDict[st][indx] = xvals[np.abs(skew_normal_cdf(xvals,A[st][indx],\
                                                               np.nanmax([A['e'+st][indx],np.sqrt(A[st][indx])]),\
                                                               A['p'+st][indx]-1.)-0.2).argmin()]
                MDict[st][indx] = xvals[np.abs(skew_normal_cdf(xvals,A[st][indx],\
                                                               np.nanmax([A['e'+st][indx],np.sqrt(A[st][indx])]),\
                                                               A['p'+st][indx]-1.)-0.5).argmin()]
                HDict[st][indx] = xvals[np.abs(skew_normal_cdf(xvals,A[st][indx],\
                                                               np.nanmax([A['e'+st][indx],np.sqrt(A[st][indx])]),\
                                                               A['p'+st][indx]-1.)-0.8).argmin()]

    return LDict,MDict,HDict


def print_html_ranks(printfile,A,totrank,LDict,MDict,HDict):

    f = open(printfile,'w')





    print('<table>',file=f)
    print('<tr><th>Name</th><th>\
        PA</th><th>AVG</th><th>lAVG</th><th>hAVG</th><th>\
        HR</th><th>lHR</th><th>hHR</th><th>R</th><th>lR</th><th>hR</th><th>\
        RBI</th><th>lRBI</th><th>hRBI</th><th>SB</th><th>lSB</th><th>hSB</th><th>Rank</th></tr>',file=f)

    for pl,indx in enumerate((totrank).argsort()):
        #print(A['Name'][indx])

        if ((A['PA'][indx] > 0) & (MDict['H'][indx] > 0.05)):

            print('<tr><td><a href=\"'+'batters/player{}.html'.format(indx)+'\">',A['Name'][indx].decode(),'</a></td><td>',int(A['PA'][indx]),'</td><td>',\
             np.round(((0.93*MDict['H']+MDict['HR'])/(1.0*A['AB'])),3)[indx],'</td><td>',\
                  np.round((((0.93*MDict['H']+LDict['HR']))/(1.0*A['AB'])),3)[indx],'</td><td>',\
                  np.round((((0.93*MDict['H']+HDict['HR']))/(1.0*A['AB'])),3)[indx],'</td><td>',\
            int((A['PA']*MDict['HR']/100.)[indx]),'</td><td>',\
                  int((A['PA']*LDict['HR']/100.)[indx]),'</td><td>',\
                  int((A['PA']*HDict['HR']/100.)[indx]),'</td><td>',\
            int((A['PA']*MDict['R']/100.)[indx]),'</td><td>',\
                  int((A['PA']*LDict['R']/100.)[indx]),'</td><td>',\
                  int((A['PA']*HDict['R']/100.)[indx]),'</td><td>',\
            int((A['PA']*MDict['RBI']/100.)[indx]),'</td><td>',\
                  int((A['PA']*LDict['RBI']/100.)[indx]),'</td><td>',\
                  int((A['PA']*HDict['RBI']/100.)[indx]),'</td><td>',\
            int((A['PA']*MDict['SB']/100.)[indx]),'</td><td>',\
                  int((A['PA']*LDict['SB']/100.)[indx]),'</td><td>',\
                  int((A['PA']*HDict['SB']/100.)[indx]),'</td><td>',\
            int(totrank[indx]),'</td></tr>',file=f)

    print('</table>',file=f)
    f.close()


def print_csv_ranks(printfile,A,totrank,LDict,MDict,HDict):

    f = open(printfile,'w')

    print('Name,PA,AVG,eAVG,HR,eHR,R,eR,RBI,eRBI,SB,eSB,Rank,',file=f)

    for pl,indx in enumerate((totrank).argsort()):
        #print(A['Name'][indx])

        if A['PA'][indx] > 0:

            try:
                print(A['Name'][indx].decode(),',',int(A['PA'][indx]),',',\
                 np.round(((A['H']+A['HR'])/A['AB']),3)[indx],',',\
                  0.5*np.round(np.max([((A['eH']+A['eHR'])/A['AB']),(((A['H']+A['HR'])/A['AB']))**2.],axis=0)[indx],3),',',\
                 int((A['PA']*A['HR']/100.)[indx]),',',int(np.max([(A['PA']*A['eHR']/100.)[indx],(np.sqrt(A['PA']*A['HR']/100.))[indx]])),',',\
                 int((A['PA']*A['R']/100.)[indx]),',',int(np.max([(A['PA']*A['eR']/100.)[indx],(np.sqrt(A['PA']*A['R']/100.))[indx]])),',',\
                 int((A['PA']*A['RBI']/100.)[indx]),',',int(np.max([(A['PA']*A['eRBI']/100.)[indx],(np.sqrt(A['PA']*A['RBI']/100.))[indx]])),',',\
                 int((A['PA']*A['SB']/100.)[indx]),',',int(np.max([(A['PA']*A['eSB']/100.)[indx],(np.sqrt(A['PA']*A['SB']/100.))[indx]])),',',\
                int(totrank[indx]),',',file=f)
            except:
                pass

    f.close()




def print_html_ranks_pitching(printfile,A,totrank,LDict,MDict,HDict,era,eera,whip,ewhip,ww,eww,svals,esvals):

    f = open(printfile,'w')



    print('<table>',file=f)
    print('<tr><th>Name</th><th>\
        IP</th><th>SO</th><th>lSO</th><th>hSO</th><th>\
        ERA</th><th>lERA</th><th>hERA</th><th>\
        WHIP</th><th>lWHIP</th><th>hSO</th><th>\
        W</th><th>lW</th><th>hW</th><th>S</th><th>lS</th><th>hS</th><th>TBF</th><th>Rank</th></tr>',file=f)

    for pl,indx in enumerate((totrank).argsort()):
        #print(A['Name'][indx])
        wmodel = int(np.nanmax([ww[indx],0.]))
        wmodello = int(np.nanmax([ww[indx]-0.5*eww[indx],ww[indx]-4.]))
        wmodelhi = int(np.nanmax([ww[indx]+0.5*eww[indx],ww[indx]+4.]))

        if wmodel <= 3.:
            wmodello = int(0.)
            wmodelhi = int(wmodel*2)

        if wmodello > wmodel:
            wmodello = int(wmodel/2.)

        print('<tr><td>',A['Name'][indx].decode(),'</td><td>',A['IP'][indx],'</td><td>',\
             int(MDict['SO'][indx]),'</td><td>',\
              int(LDict['SO'][indx]),'</td><td>',\
              int(HDict['SO'][indx]),'</td><td>',\
             np.round(era[indx],2),'</td><td>',\
              np.round(era[indx]-eera[indx],2),'</td><td>',\
              np.round(era[indx]+eera[indx],2),'</td><td>',\
             np.round(whip[indx],2),'</td><td>',\
              np.round(whip[indx]-ewhip[indx],2),'</td><td>',\
              np.round(whip[indx]+ewhip[indx],2),'</td><td>',\
             wmodel,'</td><td>',\
              wmodello,'</td><td>',\
              wmodelhi,'</td><td>',\
             svals[indx],'</td><td>',\
              svals[indx]-esvals[indx],'</td><td>',\
              svals[indx]+esvals[indx],'</td><td>',\
        A['TBF'][indx],'</td><td>',int(totrank[indx]),'</td></tr>',file=f)

    print('</table>',file=f)
    f.close()



def print_csv_ranks_pitching(printfile,A,totrank,LDict,MDict,HDict,era,eera,whip,ewhip,ww,eww,svals,esvals):
    f = open(printfile,'w')

    print('Name,IP,SO,eSO,ERA,eERA,WHIP,eWHIP,W,eW,S,eS,Rank,',file=f)

    for pl,indx in enumerate((totrank).argsort()):
        #print(A['Name'][indx])

        print(A['Name'][indx].decode(),',',A['IP'][indx],',',\
             A['SO'][indx],',',int(np.max([A['eSO'],np.sqrt(A['SO'])],axis=0)[indx]),',',\
             np.round(era[indx],2),',',np.round(eera[indx],2),',',\
             np.round(whip[indx],2),',',np.round(ewhip[indx],2),',',\
             int(np.nanmax([ww[indx],0.])),',',int(np.nanmax([eww[indx],4.])),',',\
             svals[indx],',',esvals[indx],',',\
             int(totrank[indx]),',',file=f)

    f.close()
