# split training data into various splits
# show variation between different test splits from the same dataset

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from utils.composition import generate_features

plt.rcParams.update({'font.size': 16})
plt.style.use('seaborn-colorblind')


print('Prepping Data')
datas = pd.read_csv('data/material_properties/ael_shear_modulus_vrh/train.csv')
datas['cif_id'] = datas['cif_id'].str.split('_').str[0]
datas.rename({'cif_id': 'formula'}, axis='columns', inplace=True)
X, y, _ = generate_features(datas, 'onehot')
X = X.values
y = y.values

scores = []
for i in range(30):
    # Split and normalize data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    # Train and score model
    print('Running Model', i+1)
    reg = Ridge()
    reg.fit(X_train, y_train)
    metric = reg.score(X_test, y_test)
    print('Score', metric)
    scores.append(metric)

scores = np.sort(np.array(scores))

# Plot train-test split results
plt.figure(figsize=(8, 8))
plt.bar(range(len(scores)), scores)
plt.ylabel('r$^2$')
plt.xlabel('Sorted Train-Test Splits')
plt.savefig('figures/test_splits/30_splits.png', dpi=300, transparent=True)
