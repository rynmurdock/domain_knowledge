# train on properties with different descriptors
# visualize their performance on element when an element is left out

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import Normalize
import seaborn as sns
import matplotlib.cm as cm

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

def r2_heatmap(df, elem_prop, save_dir='figures/holdouts/'):
    name = elem_prop
    ptable = pd.read_csv('data/element_properties/ptable.csv')
    ptable.index = ptable['symbol'].values
    n_row = ptable['row'].max()
    n_column = ptable['column'].max()
        
    this_feat = df[df['elem_prop'] == elem_prop]
    
    
    for idx, _, element, elem_prop, r2 in this_feat.itertuples():
        ptable.loc[ptable['symbol'] == element, 'count'] += r2

    elem_tracker = ptable['count']
    print(elem_tracker)

    fig, ax = plt.subplots(figsize=(n_column, n_row))
    rows = ptable['row']
    columns = ptable['column']
    symbols = ptable['symbol']
    rw = 0.9  # rectangle width (rw)
    rh = rw  # rectangle height (rh)
    for row, column, symbol in zip(rows, columns, symbols):
        row = ptable['row'].max() - row
        cmap = cm.YlGn
        count_min = -1
        count_max = 1
        norm = Normalize(vmin=count_min, vmax=count_max)
        count = elem_tracker[symbol]
        color = cmap(norm(count))
        if count == -100:
            color = 'silver'
        if row < 3:
            row += 0.5
        rect = patches.Rectangle((column, row), rw, rh,
                                 linewidth=1.5,
                                 edgecolor='gray',
                                 facecolor=color,
                                 alpha=1)

        plt.text(column+rw/2, row+rw/2, symbol,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=20,
                 fontweight='semibold', color='k')

        ax.add_patch(rect)

    granularity = 20
    for i in range(-granularity, granularity + 1, 2):
        value = (i) * count_max/(granularity)
        color = cmap(norm(value))
        length = 4
        x_offset = 3.9
        y_offset = 7.8
        x_loc = i/(granularity * 2) * length + x_offset
        width = length / granularity * 2
        height = 0.35
        rect = patches.Rectangle((x_loc * 2, y_offset), width, height,
                                 linewidth=1.5,
                                 edgecolor='gray',
                                 facecolor=color,
                                 alpha=1)

        if value in [-1, -.5, 0, .5, 1]:
            text = str(round(value,2))
            if value == -1:
                text = '<' + str(round(value,2))
            plt.text(x_loc * 2 + width - x_loc / granularity, y_offset-0.4, text,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontweight='semibold',
                     fontsize=20, color='k')

        ax.add_patch(rect)

    plt.text(x_offset+length, y_offset+0.7,
             'Element Count',
             horizontalalignment='center',
             verticalalignment='center',
             fontweight='semibold',
             fontsize=20, color='k')

    ax.set_ylim(-0.15, n_row+.1)
    ax.set_xlim(0.85, n_column+1.1)

    # fig.patch.set_visible(False)
    ax.axis('off')

    if save_dir is not None:
        fig_name = f'{save_dir}/{name}_ptable.png'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.draw()
    plt.pause(0.001)
    plt.close()

    






results_df = pd.read_csv('figures/holdouts/holdout_data.csv')

# you could drop the -100s
# not_used = results_df[results_df['r2'].isin([-100])]['symbol'].drop_duplicates()
# results_df = results_df[~results_df['symbol'].isin(not_used)]

for elem_prop in elem_props:
    r2_heatmap(results_df, elem_prop)