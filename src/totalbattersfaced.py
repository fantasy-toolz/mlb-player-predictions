

"""

ST = np.genfromtxt('data/Stolen_IPs_0216.csv',delimiter=',',dtype=[('uid','i4'),('ip','f4'),('name','S20')],skip_header=1)

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

print(lastyeardf.keys())

IPDict = dict()
for name in namelist:
    try:
        IPDict[name] = lastyeardf['IP'][lastyeardf['Name']==name].values
    except:
        IPDict[name] = 25.

"""

def get_ip_predictions(namelist,lastyeardf):
    IPDict = dict()
    for name in namelist:
        try:
            IPDict[name] = float(lastyeardf['IP'][lastyeardf['Name']==name].values)
        except:
            IPDict[name] = 25.
    return IPDict