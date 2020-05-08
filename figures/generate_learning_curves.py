import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

pal = sns.color_palette("Set1", n_colors=7, desat=.5)
sns.set_palette(pal)
markers = ['o', 'v', 'h', 'd', 'X', '8', 'P']


plt.rcParams.update({'font.size': 16})


nets = [[512, 512], [32, 32]]


elem_props = ['onehot',
              'random_200',
              'magpie',
              'oliynyk',
              'mat2vec',
              'jarvis',
              'atom2vec']

pretty_descs = {'onehot': 'onehot',
                'random_200': 'random',
                'magpie': 'Magpie',
                'atom2vec': 'Atom2Vec',
                'mat2vec': 'mat2vec',
                'jarvis': 'Jarvis',
                'oliynyk': 'Oliynyk'}

material_props = ['ael_shear_modulus_vrh', 'energy_atom',
                  'agl_log10_thermal_expansion_300K',
                  'agl_thermal_conductivity_300K',
                  'Egap',
                  'ael_debye_temperature',
                  'ael_bulk_modulus_vrh']

to_symbols = {'energy_atom': '$\\Delta H$',
              'ael_shear_modulus_vrh': '$G$',
              'ael_bulk_modulus_vrh': '$B$',
              'ael_debye_temperature': '$\\theta$',
              'Egap': '$E_g$',
              'agl_thermal_conductivity_300K': '$\\kappa$',
              'agl_log10_thermal_expansion_300K': '$\\alpha$'}

markers = {prop: marker for prop, marker in zip(elem_props, markers)}


def avg_and_r2_learning_curs(std=False, elem_props=elem_props):
    pal = sns.color_palette("Set1", n_colors=7, desat=.5)
    sns.set_palette(pal)
    avg_improvement = {elem_prop: np.zeros(6) for elem_prop in elem_props}
    for sub, material_prop in enumerate(material_props):
        r2 = [0, 9, 12]
        style = r2
        plt.figure(figsize=(6, 6))

        for mr, elem_prop in enumerate(elem_props):
            location = 'figures/data/' + elem_prop+' -- ' + \
                material_prop+' -- ' + str(units)[1:-1] + '.csv'
            df = pd.read_csv(location)
            x = df.iloc[:, style[0]].copy()
            y = df.iloc[:, style[1]].copy()

            if material_prop == 'Egap':
                x_all = x.copy()

            if std:
                y_std = df.iloc[:, style[2]]
                y_std_minus = y - 1.96*(y_std/np.sqrt(5))
                y_std_plus = y + 1.96*(y_std/np.sqrt(5))
                plt.fill_between(x, y_std_minus, y_std_plus, alpha=0.5)

            if elem_prop == 'onehot':
                plt.plot(x, y, '-', linewidth=6, color='grey',
                label=pretty_descs[elem_prop], alpha=0.9, markersize=8)
                y_oh = y.copy()

            else:
                plt.plot(x, y, markers[elem_prop] + '-', linewidth=3,
                         dashes=(3, 0.6), label=pretty_descs[elem_prop],
                         alpha=1, markersize=10, mfc='w', mew=1.5)

            y[y < 0] = 0
            y_oh[y_oh < 0] = 0
            avg_improvement[elem_prop][:len(y)] += y - y_oh
            plt.xlabel('Number of Training Data')
            plt.ylabel(f'r$^2$ ({to_symbols[material_prop]})')
            plt.xticks(x.values)

            plt.tick_params(right=True, top=True, direction='in', length=7)
            plt.tick_params(which='minor', right=True, top=True,
                            direction='in', length=4)
            plt.yticks(np.arange(0, 1.1, 0.2))

            minor_locator = AutoMinorLocator(2)
            plt.axes().yaxis.set_minor_locator(minor_locator)
            plt.ylim(0, 1)
            plt.xscale('log')
        plt.legend(handletextpad=0.5, handlelength=1.5)
        if std:
            plt.savefig('figures/learning_curves/' + material_prop +
                        '_' + str(units)[1:-1] + '_1std_curve.png',
                        dpi=300, transparent=True, bbox_inches='tight')
        else:
            plt.savefig('figures/learning_curves/' + material_prop +
                        '_' + str(units)[1:-1] + '_learning_curve_r2.png',
                        dpi=300, transparent=True, bbox_inches='tight')

    avg_improvement = {key: value/len(material_props) for
                                      key, value in avg_improvement.items()}
    test = pd.DataFrame(avg_improvement)
    test = test[elem_props]

    plt.figure(figsize=(6, 6))
    for prop in elem_props:
        if prop == 'onehot':
            plt.plot(x_all[0:], test[prop].values[0:], '-', linewidth=6,
                     color='grey', label=pretty_descs[prop], alpha=0.9,
                     markersize=8)
        else:
            plt.plot(x_all[0:], test[prop].values[0:], markers[prop]+'-',
                     linewidth=3, dashes=(3, 0.6), label=pretty_descs[prop],
                     alpha=1, markersize=10, mfc='w', mew=1.5)
    # plt.yticks(np.arange(0, 1.1, 0.1))
    # plt.ylim(-0.025, 0.025)
    plt.tick_params(right=True, top=True, direction='in', length=7)
    plt.tick_params(which='minor', right=True, top=True, direction='in',
                    length=4)
    plt.xlabel('Number of Training Data')
    plt.ylabel(f'Average improvement over onehot')
    plt.yticks(np.arange(-.2, .3, 0.1))
    minor_locator = AutoMinorLocator(2)
    plt.axes().yaxis.set_minor_locator(minor_locator)
    plt.xticks(x_all)
    plt.xscale('log')
    plt.legend(ncol=2, handletextpad=0.5, handlelength=1.5, columnspacing=0.3)
    plt.savefig('figures/learning_curves/' + 
                '_' + str(units)[1:-1] + '_comparison_r2.png',
                dpi=300, transparent=True, bbox_inches='tight')


def to_p(x):
    return to_symbols[x]


def just_one():
    one_desc = 'onehot'
    mse = [0, 8, 11]
    style = mse
    plt.figure(figsize=(6, 6))
    for mr, material_prop in enumerate(material_props):
        location = 'figures/data/' + one_desc + ' -- ' + material_prop + \
            ' -- '+str(units)[1:-1] + '.csv'
        df = pd.read_csv(location)
        x = df.iloc[:, style[0]]
        y = np.sqrt(df.iloc[:, style[1]])

        if mr == 1:
            plt.plot(x, y, '' + '-', color='grey',
                     linewidth=4, markersize=10, label='onehot', alpha=1,
                     mfc='w', mew=1.5)
        else:
            plt.plot(x, y, '' + '-', color='grey',
                 linewidth=4, markersize=10, alpha=1, mfc='w', mew=1.5)

    # one_desc = 'oliynyk'
    one_desc = 'jarvis'
    mse = [0, 8, 11]
    style = mse
    for mr, material_prop in enumerate(material_props):
        if material_prop == 'Egap':
            x_max = x
        location = 'figures/data/' + one_desc + ' -- ' + material_prop + \
            ' -- '+str(units)[1:-1] + '.csv'

        df = pd.read_csv(location)
        x = df.iloc[:, style[0]]
        y = np.sqrt(df.iloc[:, style[1]])

        plt.tick_params(right=True, top=True, direction='in', length=7)
        if to_symbols[material_prop] == '$\\Delta H$':
            marker_str = '$H$'
        elif to_symbols[material_prop] == '$E_g$':
            marker_str = '$E$'
        else:
            marker_str = f'{to_symbols[material_prop]}'
        plt.plot(x, y, '-',
                 marker=marker_str,
                 dashes=(3, 0.6),
                 linewidth=3, markersize=12,
                 alpha=1, mfc='w', mew=0.6, mec='k')




    plt.tick_params(right=True, top=True, direction='in', length=7)
    plt.tick_params(which='minor', right=True, top=True, direction='in',
                    length=4)
    minor_locator = AutoMinorLocator(2)
    plt.axes().yaxis.set_minor_locator(minor_locator)


    plt.xlabel('Number of Training Datapoints')
    plt.ylabel('RMSE')
    plt.xticks(x_max)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('figures/learning_curves/just_' + one_desc +
                str(units)[1:-1] + '_learning_curve_rmse.png',
                dpi=300, transparent=True, bbox_inches='tight')

def curve_rates():
    markers = ['o', 'v', 'h', 'd', 'X', '8', 'P']
    mse = [0, 8, 11]
    style = mse
    plt.figure(figsize=(6, 6))
    for mr, elem_prop in enumerate(elem_props):
        line = []
        lb = []
        for material_prop in material_props:
            location = 'figures/data/' + elem_prop + ' -- ' +\
                material_prop + ' -- ' + str(units)[1:-1] + '.csv'
            df = pd.read_csv(location)
            x = np.log(df.iloc[:, style[0]].values)
            y = np.log(np.sqrt(df.iloc[:, style[1]].values))
            line.append((y[0]-y[-1])/x[-1])
            lb.append(to_symbols[material_prop])

        plt.tick_params(right=True, top=True, direction='in', length=7)
        plt.plot(lb, line, markers[mr] + '--', linewidth=2,
                 markersize=10, label=pretty_descs[elem_prop])
    plt.ylabel('RMSE Rate of Decline')
    plt.legend(loc='upper left')
    plt.savefig('figures/learning_curves/' + str(units)[1:-1] +
                '_rates_rmse.png',
                dpi=300, transparent=True, bbox_inches='tight')


def multi_figure(mat_props, std=False):
    markers = ['o', 'v', 'h', 'd', 'X', '8', 'P']
    fig = plt.figure(figsize=(6, 6))
    for sub, material_prop in enumerate(mat_props):
        r2 = [0, 9, 12]
        style = r2

        fig.text(0.5, 0.04, 'Number of Training Datapoints',
                 ha='center', va='center')
        fig.text(0.06, 0.5, 'r$^2$', ha='center',
                 va='center', rotation='vertical')

        for mr, elem_prop in enumerate(elem_props):
            location = 'figures/data/' + elem_prop + ' -- ' + \
                material_prop + ' -- ' + str(units)[1:-1] + '.csv'
            df = pd.read_csv(location)
            x = df.iloc[:, style[0]]
            y = df.iloc[:, style[1]]

            if std:
                y_std = df.iloc[:, style[2]]
                y_std_minus = y - 1.96*(y_std/np.sqrt(5))
                y_std_plus = y + 1.96*(y_std/np.sqrt(5))
                plt.fill_between(x, y_std_minus, y_std_plus, alpha=0.5)

            plt.tick_params(right=True, top=True, direction='in', length=7)
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.plot(x, y, markers[mr] + '--', linewidth=2,
                     markersize=10, label=pretty_descs[elem_prop], alpha=1)
            plt.ylim(0, 1)
            plt.title(to_symbols[material_prop])
    plt.legend()
    if std:
        plt.savefig('figures/learning_curves/' + '_' +
                    str(units)[1:-1] + '_1std_many_curve.png',
                    dpi=300, transparent=True, bbox_inches='tight')
    else:
        plt.savefig('figures/learning_curves/' + '_' + str(units)[1:-1] +
                    '_many_curves.png',
                    dpi=300, transparent=True, bbox_inches='tight')

def basic_lr_curve(metric, mat_p):
    pal = sns.color_palette("Set1", n_colors=7, desat=.5)
    sns.set_palette(pal)
    if metric == 'MAE':
        style = [0, 7, 10]
    if metric == 'MSE':
        style = [0, 8, 11]
    if metric == 'r2':
        style = [0, 9, 12]
    plt.figure(figsize=(6, 6))
    for mr, elem_p in enumerate(elem_props):
        location = 'figures/data/' + elem_p + ' -- ' + mat_p + \
            ' -- '+str(units)[1:-1] + '.csv'

        df = pd.read_csv(location)
        x = df.iloc[:, style[0]]
        y = df.iloc[:, style[1]]
        if elem_p == 'onehot':
            plt.plot(x, y, '-', linewidth=6, color='grey',
            label=pretty_descs[elem_p], alpha=0.9, markersize=8)
        else:
            plt.plot(x, y, '-', marker=markers[elem_p], 
                     label=pretty_descs[elem_p], linewidth=4, markersize=10, 
                     alpha=1, mfc='w', mew=1.5,)

    
    plt.tick_params(right=True, top=True, direction='in', length=7)
    plt.tick_params(which='minor', right=True, top=True, direction='in',
                    length=4)
    minor_locator = AutoMinorLocator(2)
    plt.axes().yaxis.set_minor_locator(minor_locator)

    plt.legend()
    plt.xlabel('Number of Training Datapoints')
    plt.ylabel(f'{metric} ({to_symbols[mat_p]})')
    plt.savefig('figures/learning_curves/' + mat_p + '_learning_curve_' 
                + metric + str(units)[1:-1] + '.png',
                dpi=300, transparent=True, bbox_inches='tight')

for units in nets:
    for mat in material_props:
        basic_lr_curve(metric='MAE', mat_p=mat)
        plt.show()
    
    avg_and_r2_learning_curs()
    plt.show()

    just_one()
    plt.show()
    
    
    
