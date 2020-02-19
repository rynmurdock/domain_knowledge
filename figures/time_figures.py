import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-colorblind')

markers = ['o', 'v', 'h', 'd', 'X', 'S', 'P']

nets = [[512, 512]]

# elem_props = ['onehot',
#               'random_200',
#               'magpie',
#               'atom2vec',
#               'mat2vec',
#               'jarvis_shuffled',
#               'jarvis']
# Only the four viable descriptors are used
elem_props = ['magpie', 'onehot',
              'mat2vec', 'jarvis']


pretty_descs = {'onehot': 'onehot',
                'random_200': 'random',
                'magpie': 'Magpie',
                'atom2vec': 'Atom2Vec',
                'mat2vec': 'mat2vec',
                'jarvis': 'Jarvis',
                'jarvis_shuffled': 'Jarvis Shuffled Along Elements'}

material_props = ['ael_shear_modulus_vrh', 'Egap',
                  'agl_log10_thermal_expansion_300K',
                  'ael_bulk_modulus_vrh']
# material_props = ['ael_shear_modulus_vrh', 'Egap',
#                   'agl_log10_thermal_expansion_300K',
#                   'ael_bulk_modulus_vrh', 'ael_debye_temperature',
#                   'energy_atom', 'agl_thermal_conductivity_300K']

to_symbols = {'energy_atom': '$\\Delta H$',
              'ael_shear_modulus_vrh': '$G$',
              'ael_bulk_modulus_vrh': '$B$',
              'ael_debye_temperature': '$\\theta$',
              'Egap': '$E_g$',
              'agl_thermal_conductivity_300K': '$\\kappa$',
              'agl_log10_thermal_expansion_300K': '$\\alpha$'}


def plot_train(df, title):
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(8, 8))
    for mr, material_prop in enumerate(material_props):
        y = []
        x = []
        for elem_prop in elem_props:
            x.append(pretty_descs[elem_prop])
            y.append(df[elem_prop + ' -- ' + material_prop]/60)
        plt.plot(x, y, markers[mr] + '--', linewidth=2,
                 label=to_symbols[material_prop], alpha=1, markersize=10)
    plt.xlabel('Descriptor')
    plt.ylabel('Time in Minutes')
    plt.legend()
    plt.savefig('figures/time/' + units + 'time_for_' + title + '.png',
                dpi=300, transparent=True)


def table_inference(df, title):
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(27, 3))
    # we're just using the first material property; shouldn't vary anyways
    predictions = pd.read_csv('data/material_properties/' +
                              material_props[0] + '/test.csv')
    predictions_tested = predictions.values.shape[0]
    y = []
    for elem_prop in elem_props:
        y.append(np.round(predictions_tested / df[elem_prop + ' -- ' +
                                                  material_props[0]][0], 2))
    colbls = [pretty_descs[elem_prop] for elem_prop in elem_props]
    the_table = plt.table([y], colLabels=colbls,
                          rowLabels=['Predictions/Second'], loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(24)
    the_table.scale(.3, 3)

    # Removing ticks and spines enables you to get the figure only with table
    plt.tick_params(axis='x', which='both',
                    bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both',
                    right=False, left=False, labelleft=False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)

    plt.savefig('figures/time/' + units + 'time_for_' + title + '.png',
                dpi=300, transparent=True)


for units in nets:
    units = str(units)[1:-1]

    train_times = pd.read_csv('figures/time/time_for_training' +
                              units + '.csv')
    test_times = pd.read_csv('figures/time/time_for_testing' +
                             units + '.csv')
    plot_train(train_times, 'training')
    table_inference(test_times, 'testing', )
