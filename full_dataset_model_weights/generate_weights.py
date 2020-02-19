import pandas as pd
import time
import sys
sys.path.append('./')
import train_nn

units = [512, 512]

secs_to_train = []
combos = []

mat_props = ['ael_shear_modulus_vrh',
             'energy_atom',
             'agl_log10_thermal_expansion_300K',
             'agl_thermal_conductivity_300K',
             'Egap',
             'ael_debye_temperature',
             'ael_bulk_modulus_vrh']

features = ['onehot.csv', 'random_200.csv', 'magpie.csv', 'atom2vec.csv',
            'mat2vec.csv', 'jarvis_shuffled.csv', 'jarvis.csv', 'oliynyk.csv']

for material_property in mat_props:
    df_t = pd.read_csv('data/material_properties/' +
                       material_property+'/train.csv')
    df_v = pd.read_csv('data/material_properties/' +
                       material_property+'/val.csv')

    # Extract formula from CIF ID
    df_t['cif_id'] = df_t['cif_id'].str.split('_').str[0]
    df_t.rename({'cif_id': 'formula'}, axis='columns', inplace=True)
    df_v['cif_id'] = df_v['cif_id'].str.split('_').str[0]
    df_v.rename({'cif_id': 'formula'}, axis='columns', inplace=True)

    for feature in features:
        # Track the time required for training
        start_time = time.time()

        # Print progress update
        feature = feature.split('.csv')[0]
        print('Generating weights for', material_property, 'using', feature)

        # Define the model we want to train
        model = train_nn.Model(df_t,
                               df_v,
                               batch_size=16,
                               units=units,
                               model='net',
                               representation=feature)

        # Train the model
        save_string = 'full_dataset_model_weights/' + feature + ' -- ' + \
            material_property + ' -- ' + str(units)[1:-1] + '.ckpt'
        model.train(100000, checkin=1800, save_weights_to_path=save_string)
        combos.append(feature + ' -- ' + material_property)

        # Get model training time
        end_time = time.time()
        duration = end_time - start_time
        print('Duration', duration)
        secs_to_train.append(duration)

time_df = pd.DataFrame([secs_to_train], columns=combos)
time_df.to_csv('figures/time/time_for_training' + str(units)[1:-1] + '.csv')
