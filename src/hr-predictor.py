
import pandas as pd
"""
Calculates the weights of the columns in the featurevec that provide the best approximation of the target variable SO using 
linear regression.

Parameters:
- featurevec (list): List of column names to be used as features.
- targetvec (list): List of column names to be used as the target variable.

Returns:
- weights (ndarray): Array of coefficients representing the weights of the features in the linear regression model.

Raises:
- None

Note:
- The LinearRegression class is imported from the sklearn.linear_model module.
"""
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
#

"""
# this bit gets the data, but I gave you a csv already
import src.predictiondata as predictiondata
years = [2024]
df1 = predictiondata.grab_fangraphs_pitching_data(years, type=1)
df0 = predictiondata.grab_fangraphs_pitching_data(years)

df = pd.merge(df1, df0, on='Name')

df['ERA'] = 9*df['ER'].astype('float')/df['IP'].astype('float')
df.to_csv('data/advanced_pitching_2024.csv', index=False)
"""

df = pd.read_csv('/Users/mpetersen/Downloads/stats-2.csv')
"""
Index(['last_name, first_name', 'player_id', 'year', 'pa', 'home_run',
       'k_percent', 'bb_percent', 'avg_swing_speed', 'fast_swing_rate',
       'launch_angle_avg', 'sweet_spot_percent', 'barrel_batted_rate',
       'solidcontact_percent', 'hard_hit_percent', 'whiff_percent',
       'swing_percent', 'pull_percent', 'straightaway_percent',
       'opposite_percent', 'flyballs_percent'],
"""

featurevec = ['k_percent', 'bb_percent', 'avg_swing_speed', 'fast_swing_rate',
       'launch_angle_avg', 'sweet_spot_percent', 'barrel_batted_rate',
       'solidcontact_percent', 'hard_hit_percent', 'whiff_percent',
       'swing_percent', 'pull_percent', 'straightaway_percent',
       'opposite_percent', 'flyballs_percent']
targetvec = ['home_run']

for feature in featurevec:
    df[feature] = df[feature].astype('float')

validdf = df.loc[df['IP'].astype('float') > 50]

X = df[featurevec]
y = df[targetvec]

# normalize the data per feature
for feature in featurevec:
    X[feature] = (X[feature] - X[feature].mean()) / X[feature].std()

model = LinearRegression()
model.fit(X, y)

weights = model.coef_
print(weights)

for k,feature in enumerate(featurevec): 
    print(np.round(weights[0][k],3),feature)

# Create an instance of PCA
pca = PCA()

# Fit the PCA model to the normalized feature vector
pca.fit(X)

# Get the principal components
principal_components = pca.components_

# Print the principal components
print(principal_components)

# Calculate the absolute values of the principal components
abs_principal_components = abs(principal_components[0])

# Find the index of the feature with the maximum absolute principal component value
most_predictive_feature_index = abs_principal_components.argmax()

# Get the name of the most predictive feature
most_predictive_feature = featurevec[most_predictive_feature_index]

print("Most predictive feature:", most_predictive_feature)


for k,feature in enumerate(featurevec): 
    print(np.round(abs_principal_components[k],3),feature)
