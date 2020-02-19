# train on properties with different descriptors
# visualize their performance on element when an element is left out

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import Ridge

from utils.composition import generate_features

pal = sns.color_palette("Set1", n_colors=7, desat=.5)
sns.set_palette(pal)

plt.rcParams.update({'font.size': 16})

# Load in material property data
print('Prepping Data')
datas = pd.read_csv('data/material_properties/ael_bulk_modulus_vrh/train.csv')
datas['cif_id'] = datas['cif_id'].str.split('_').str[0]
datas.rename({'cif_id': 'formula'}, axis='columns', inplace=True)

# Generate holdout element dataframe
heldout_element = 'Mg'
train = datas[~datas.formula.str.contains(heldout_element)]
test = datas[datas.formula.str.contains(heldout_element)]
print('Test points', test.shape[0], 'Train points', train.shape[0])
print(test)

elem_props = ['onehot',
              'random_200',
              'magpie',
              'atom2vec',
              'mat2vec',
              'jarvis',
              'oliynyk']

pretty_feats = {'onehot': 'onehot',
                'random_200': 'random',
                'magpie': 'Magpie',
                'jarvis': 'Jarvis',
                'atom2vec': 'Atom2Vec',
                'mat2vec': 'mat2vec',
                'oliynyk': 'Oliynyk'}

r2s = []
for featurizer in elem_props:
    # Featurize training data
    X_train, y_train, _ = generate_features(train, featurizer)
    X_train = X_train.values
    y_train = y_train.values

    # Featurize test data
    X_test, y_test, _ = generate_features(test, featurizer)
    X_test = X_test.values
    y_test = y_test.values

    # Scale and normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    # Train and score model
    print('Running Model with', featurizer)
    reg = Ridge()
    reg.fit(X_train, y_train)
    metric = reg.score(X_test, y_test)
    print('Score', metric)
    r2s.append(metric)
# %%
# Plot holdout results
plt.figure(figsize=(6, 6))

plt.bar([pretty_feats[feat] for feat in elem_props], r2s)

for feat, r2 in zip(elem_props, r2s):

    if pretty_feats[feat] == 'Atom2Vec':
        plt.text(pretty_feats[feat], 0.05, f'{r2:0.2f}',
                 horizontalalignment='center')
    else:
        plt.text(pretty_feats[feat], r2+0.03, f'{r2:.2f}',
                 horizontalalignment='center')

plt.xlabel('Featurizer')
plt.xticks(rotation=45)
plt.ylabel('r$^2$ ($B$)')
plt.tick_params(right=True, top=True, direction='in', length=7)
plt.tick_params(which='minor', right=True, top=True, direction='in', length=4)
minor_locator = AutoMinorLocator(2)
plt.axes().yaxis.set_minor_locator(minor_locator)
plt.ylim(0, 1)
plt.savefig('figures/holdouts/holdouts.png', dpi=300, transparent=True, bbox_inches='tight')






