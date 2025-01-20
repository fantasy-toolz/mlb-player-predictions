

import matplotlib.pyplot as plt
import pandas as pd

P = pd.read_csv('eval2024.txt')
P = pd.read_csv('validation2022.txt')

# true,predicted,sigma,name

plt.scatter(P['true'],P['predicted'],s=4.,color='black')
plt.plot([0.,10.],[0.,10.],color='red',linestyle='dashed')
plt.xlabel('True W')
plt.ylabel('Predicted W')
plt.tight_layout()
plt.savefig('/Users/mpetersen/Downloads/valid.png',dpi=300)