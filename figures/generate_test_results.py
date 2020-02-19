import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator


pal = sns.color_palette("Set1", n_colors=7, desat=.5)
sns.set_palette(pal)

markers = ['o', 'v', 'h', 'd', 'X', '8', 'P']

nets = [[512, 512], [32, 32]]

for units in nets:
    units = str(units)[1:-1]

    data = os.listdir('figures/data')

    elem_props = ['random_200',
                  'magpie',
                  'atom2vec',
                  'mat2vec',
                  'oliynyk',
                  'jarvis',
                  'onehot']

    # Only the four viable descriptors are used
    # elem_props = ['onehot', 'magpie',
    #               'mat2vec', 'jarvis']
    pretty_descs = {'onehot': 'onehot',
                    'random_200': 'random',
                    'magpie': 'Magpie',
                    'atom2vec': 'Atom2Vec',
                    'mat2vec': 'mat2vec',
                    'jarvis': 'Jarvis',
                    'oliynyk': 'Oliynyk',
                    'jarvis_shuffled': 'Jarvis Shuffled Along Elements'}

    to_symbols = {'energy_atom': '$E_{atom}$',
                  'ael_shear_modulus_vrh': '$G$',
                  'ael_bulk_modulus_vrh': '$B$',
                  'ael_debye_temperature': '$\\theta$',
                  'Egap': '$E_g$',
                  'agl_thermal_conductivity_300K': '$\\kappa$',
                  'agl_log10_thermal_expansion_300K': '$\\alpha$'}

    # material_props = ['ael_shear_modulus_vrh', 'Egap',
    #                         'agl_log10_thermal_expansion_300K',
    #                         'ael_bulk_modulus_vrh']
    pretty_feats = {'onehot': 'onehot',
                    'random_200': 'random',
                    'magpie': 'Magpie',
                    'jarvis': 'Jarvis',
                    'atom2vec': 'Atom2Vec',
                    'mat2vec': 'mat2vec'
                    }
    material_props = ['Egap', 'ael_shear_modulus_vrh',
                      'ael_bulk_modulus_vrh', 'ael_debye_temperature',
                      'energy_atom', 'agl_thermal_conductivity_300K',
                      'agl_log10_thermal_expansion_300K']

    to_pretty = {'energy_atom': 'ab initio Energy',
                 'ael_shear_modulus_vrh': 'Shear Modulus',
                 'ael_bulk_modulus_vrh': 'Bulk Modulus',
                 'ael_debye_temperature': 'Debye Temperature',
                 'Egap': 'Band Gap',
                 'agl_thermal_conductivity_300K': 'Thermal Conductivity',
                 'agl_log10_thermal_expansion_300K': 'Thermal Expansion'}

    def by_descriptor(title=''):
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(8, 8))
        for sub, material_prop in enumerate(material_props):
            y = []
            x = []
            for elem_prop in elem_props:
                location = 'figures/test_results/' + elem_prop + \
                    ' -- ' + material_prop + ' -- ' + str(units) + '.csv'
                df = pd.read_csv(location)
                x.append(pretty_descs[elem_prop])
                y.append(float(df.iloc[0, 3][1:-1]))
                #  grab the 0th row and 3rd column;
                #  slice that string for only the value
            plt.plot(x, y, markers[sub] + '--', linewidth=2,
                     markersize=10, label=to_symbols[material_prop], alpha=1)
            if sub == 4:
                plt.plot(x, y, markers[sub] + '--', linewidth=2,
                         label=None, alpha=1, markersize=10, color='gold')
        plt.tick_params(right=True, top=True, direction='in')
        plt.yticks(np.arange(.1, 1.1, 0.1))
        plt.xlabel('Descriptor')
        plt.ylabel('r$^2$')
        minor_locator = AutoMinorLocator(2)
        plt.axes().yaxis.set_minor_locator(minor_locator)
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig('figures/test_results/' + units
                    + title + '_test_results_r2_by_descriptor.png',
                    dpi=300, transparent=True, bbox_inches='tight')

    def by_material_prop(title=''):
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(8, 8))

        for mr, elem_prop in enumerate(elem_props):
            y = []
            x = []
            for sub, material_prop in enumerate(material_props):
                location = 'figures/test_results/' + elem_prop + ' -- ' + \
                    material_prop + ' -- ' + str(units) + '.csv'
                df = pd.read_csv(location)
                x.append(to_symbols[material_prop])
                y.append(float(df.iloc[0, 3][1:-1]))
            plt.plot(x, y, markers[mr] + '--', linewidth=2, markersize=10,
                     label=pretty_descs[elem_prop], alpha=1)
        plt.tick_params(right=True, top=True, direction='in')
        plt.yticks(np.arange(.1, 1.1, 0.1))

        plt.ylabel('r$^2$')
        minor_locator = AutoMinorLocator(2)
        plt.axes().yaxis.set_minor_locator(minor_locator)
        plt.ylim(0, 1)
        plt.legend()
        # plt.title('Test results with ' + units + ' Hidden Units')
        plt.savefig('figures/test_results/' + units + title +
                    '_test_results_r2.png',
                    dpi=300, transparent=True, bbox_inches='tight')



    by_descriptor('_all')
    by_material_prop('_all')

    def best_of_all(models, elem_props, material_props):
        best_vals = []
        best_x = []
        prt = []
        for material_prop in material_props:
            y = []
            x = []
            for elem_prop in elem_props:
                for model in models:
                    location = 'figures/test_results/' + elem_prop + ' -- ' +\
                        material_prop + ' -- ' + str(model)[1:-1] + '.csv'
                    df = pd.read_csv(location)
                    y.append(float(df.iloc[0, -1][1:-1]))  # r2
                    x.append(str(pretty_descs[elem_prop] +
                                 ' in ' + str(model)[1:-1].replace(', ', 'x') +
                                 ' model'))
            best_vals.append(max(y))
            best_x.append(x[y.index(max(y))])
            prt.append(to_symbols[material_prop])
        all_df = pd.DataFrame(np.array([best_x, best_vals]), columns=prt)
        all_df.to_csv('figures/test_results/top_scores.csv')

    best_of_all(nets, elem_props, material_props)

