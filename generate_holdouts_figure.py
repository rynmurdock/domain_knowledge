# train on properties with different descriptors
# visualize their performance on element when an element is left out

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
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


all_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
               'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
               'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
               'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
               'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
               'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
               'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
               'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
               'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
               'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
               'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
               'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']


results_df = pd.DataFrame(columns=['symbol', 'elem_prop', 'r2'])

for heldout_element in all_symbols:
    # Generate holdout element dataframe
    train = datas[~datas.formula.str.contains(heldout_element)]
    test = datas[datas.formula.str.contains(heldout_element)]
    print('Test points with', heldout_element + ':', test.shape[0])
    print('Train points:', train.shape[0])
    print(test)
    
    
    # Set r2 scores of elements rarely in the data to -100; we'll exclude these
    if test.shape[0] <= 10:
        for featurizer in elem_props:
            row = [heldout_element, featurizer, -100]
            results_df.loc[len(results_df)] = row
        continue
    
    
    for featurizer in elem_props:
        # Featurize training data
        X_train, y_train, _ = generate_features(train, featurizer)
        X_train = X_train.values
        y_train = y_train.values
    
        # Featurize test data
        X_test, y_test, _ = generate_features(test, featurizer)
        X_test = X_test.values
        y_test = y_test.values
        
        if X_test.shape[0] <= 10: # some rare elements are not featurizable
            row = [heldout_element, featurizer, -100]
            results_df.loc[len(results_df)] = row
            continue
    
        # Scale and normalize data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = normalize(X_train)
        X_test = normalize(X_test)
    
        # Train and score model
        reg = Ridge(random_state=100)
        reg.fit(X_train, y_train)
        metric = reg.score(X_test, y_test)
        row = [heldout_element, featurizer, metric]
        results_df.loc[len(results_df)] = row
        
results_df.to_csv('figures/holdouts/holdout_data.csv')
        
        
        
# %%
# Plot holdout results

results_df = pd.read_csv('figures/holdouts/holdout_data.csv')

# get all average r2 values and plot them
r2s = []

# make sure all featurizers only use r2 values that're not -100 in any of them
not_used = results_df[results_df['r2'].isin([-100])]['symbol'].drop_duplicates()
results_df = results_df[~results_df['symbol'].isin(not_used)]


# I'll probably want to remove outliers or somehow show all elements


# collect the average r2
for feat in elem_props:
    indies = results_df[results_df['elem_prop'] == feat]
    print(indies, len(indies))
    sc = indies['r2'].mean()
    print(sc)
    r2s.append(sc)
    
    

plt.figure(figsize=(6, 6))

plt.bar([pretty_feats[feat] for feat in elem_props], r2s)

for feat, r2 in zip(elem_props, r2s):
    if pretty_feats[feat] == 'Atom2Vec':
        plt.text(pretty_feats[feat], 0.05, f'{r2:.2f}',
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






