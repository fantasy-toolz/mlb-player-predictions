

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ST = pd.read_csv('/Users/mpetersen/Downloads/tgfbi_2024.csv')

namelist = np.array([x for  x in ST['Name'].values])
uninames = np.unique(namelist)

f = open('/Users/mpetersen/Downloads/tgfbi_2024_averages.csv','w')
print('Name,AvgDraft,SigmaDraft',file=f)

for name in uninames:
    draftpick = np.nanmean(ST.loc[ST['Name']==name]['Overall Pick'].values)
    draftdispersion = np.nanstd(ST.loc[ST['Name']==name]['Overall Pick'].values)
    print('{0},{1},{2}'.format(name,np.round(draftpick,1),np.round(draftdispersion,1)),file=f)


f.close()

COMP = pd.read_csv('/Users/mpetersen/Downloads/ToolzTop100.csv')

plt.figure()
plt.scatter(COMP['Overall'],COMP['TGFBI mean'],color='black',s=10)
for n in range(0,len(COMP['Overall'].values)):
    plt.plot([COMP['Overall'].values[n],COMP['Overall'].values[n]],[COMP['TGFBI mean'].values[n]-COMP['TGFBI dispersion'].values[n],COMP['TGFBI mean'].values[n]+COMP['TGFBI dispersion'].values[n]],color='black',lw=1.)

plt.plot([0,150],[0,150],color='grey',linestyle='dashed')
plt.xlabel('Toolz')
plt.ylabel('Experts')
plt.axis([0.,120.,0.,170.])
plt.tight_layout()
plt.savefig('/Users/mpetersen/Downloads/draftcomp2024.png',dpi=300)


