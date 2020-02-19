import numpy as np
import pandas as pd
import os
# Suppress info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import train_nn


net_archs = [[32, 32], [512, 512]]

# mat_props = ['ael_shear_modulus_vrh',
#              'energy_atom',
#              'agl_log10_thermal_expansion_300K',
#              'agl_thermal_conductivity_300K',
#              'Egap',
#              'ael_debye_temperature',
#              'ael_bulk_modulus_vrh']
mat_props = ['energy_atom']

features = ['oliynyk.csv', 'onehot.csv', 'random_200.csv', 'magpie.csv',
            'atom2vec.csv', 'mat2vec.csv', 'jarvis_shuffled.csv', 'jarvis.csv']

for units in net_archs:
    for material_property in mat_props:
        # Load data
        df_t = pd.read_csv('data/material_properties/' +
                           material_property+'/train.csv')
        df_v = pd.read_csv('data/material_properties/' +
                           material_property+'/val.csv')

        # Extract formula from CIF ID
        df_t['cif_id'] = df_t['cif_id'].str.split('_').str[0]
        df_t.rename({'cif_id': 'formula'}, axis='columns', inplace=True)
        df_v['cif_id'] = df_v['cif_id'].str.split('_').str[0]
        df_v.rename({'cif_id': 'formula'}, axis='columns', inplace=True)

        # Combine the train and validation data for reshuffling
        combined = df_t.append(df_v, ignore_index=True)

        # Define the number of training data inputs using a base 3
        # logarithmic series starting at base to the first_pow.
        max_samples = df_t.shape[0]
        base = 3
        first_pow = 4
        max_pow = np.ceil(np.log(max_samples) / np.log(base)).astype(np.int32)
        n_data = [base**log for log in range(first_pow, max_pow)]
        n_data.append(max_samples)

        for feature in features:
            feature = feature.split('.csv')[0]
            data_l = []
            seeds = [1, 2, 3, 4, 5]

            for training_samples in n_data:
                t_mae = []
                t_mse = []
                t_r2 = []
                v_mae = []
                v_mse = []
                v_r2 = []

                for seed in seeds:
                    print('\n' + 75*'-')
                    print('Starting Validation for:')
                    print(f'property: {material_property}')
                    print(f'feature: {feature}')
                    print(f'training size: {training_samples}/{n_data}')
                    print(f'seed: {seed}/{seeds[-1]}')
                    print(75*'-' + '\n')
                    df_t_samp = combined.sample(n=training_samples,
                                                random_state=seed)

                    df_v = combined[~combined.isin(df_t_samp)]
                    df_v_samp = df_v.dropna().sample(n=512, random_state=seed)

                    model = train_nn.Model(df_t_samp,
                                           df_v_samp,
                                           model='net',
                                           units=units,
                                           representation=feature,
                                           batch_size=16)

                    # Store model performance metrics
                    out = model.train(80000, training_samples//model.batch_size)
                    tmae, tmse, tr2, vmae, vmse, vr2 = out
                    t_mae.append(tmae)
                    t_mse.append(tmse)
                    t_r2.append(tr2)
                    v_mae.append(vmae)
                    v_mse.append(vmse)
                    v_r2.append(vr2)

                data_l.append([np.mean(t_mae),
                               np.mean(t_mse),
                               np.mean(t_r2),
                               np.std(t_mae),
                               np.std(t_mse),
                               np.std(t_r2),
                               np.mean(v_mae),
                               np.mean(v_mse),
                               np.mean(v_r2),
                               np.std(v_mae),
                               np.std(v_mse),
                               np.std(v_r2)])

            # Save model metrics to csv
            csver = pd.DataFrame(data_l,
                                 index=n_data,
                                 columns=['t_Mean_MAE',
                                          't_Mean_MSE',
                                          't_Mean_R2',
                                          't_Std_MAE',
                                          't_Std_MSE',
                                          't_Std_R2',
                                          'v_Mean_MAE',
                                          'v_Mean_MSE',
                                          'v_Mean_R2',
                                          'v_Std_MAE',
                                          'v_Std_MSE',
                                          'v_Std_R2'])

            print(csver)
            csver.to_csv('figures/data/' + feature +
                         ' -- ' + material_property +
                         ' -- ' + str(units)[1:-1] + '.csv')
