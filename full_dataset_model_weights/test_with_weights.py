import pandas as pd
import time
import sys
sys.path.append('./')
import train_nn

archs = [[512, 512], [32, 32]]


mat_props = ['ael_shear_modulus_vrh',
             'energy_atom',
             'agl_log10_thermal_expansion_300K',
             'agl_thermal_conductivity_300K',
             'Egap',
             'ael_debye_temperature',
             'ael_bulk_modulus_vrh']

features = ['onehot.csv', 'random_200.csv', 'magpie.csv', 'atom2vec.csv',
            'mat2vec.csv', 'jarvis_shuffled.csv', 'jarvis.csv', 'oliynyk.csv']
for units in archs:
    secs_to_train = []
    combos = []
	for material_property in mat_props:
	    test_df = pd.read_csv('data/material_properties/' +
	                          material_property+'/test.csv')
	    df_t = pd.read_csv('data/material_properties/' +
	                       material_property+'/train.csv')

	    # Extract formula from CIF ID
	    test_df['cif_id'] = test_df['cif_id'].str.split('_').str[0]
	    test_df.rename({'cif_id': 'formula'}, axis='columns', inplace=True)
	    df_t['cif_id'] = df_t['cif_id'].str.split('_').str[0]
	    df_t.rename({'cif_id': 'formula'}, axis='columns', inplace=True)

	    for feature in features:
	        # Track the time required for training
	        start_time = time.time()

	        # Print progress update
	        feature = feature.split('.csv')[0]
	        print('Predicting', material_property,
	              'using full dataset weights of', feature)

	        # Define the model we want to train
	        model = train_nn.Model(df_t,  # full training data is fed for scaling
	                               test_df,
	                               batch_size=16,
	                               units=units,
	                               model='net',
	                               representation=feature)

	        # Define the path to which the trained model weights were saved
	        restore_path = './full_dataset_model_weights/' + feature + ' -- ' + \
	                       material_property + ' -- ' + str(units)[1:-1] + '.ckpt'

	        # Generate predictions using stored model weights then save to CSV
	        model.predict(restore_path)

	        # Get model training time
	        combos.append(feature+' -- '+material_property)
	        end_time = time.time()
	        duration = end_time-start_time
	        print('Duration', duration)
	        secs_to_train.append(duration)

	time_df = pd.DataFrame([secs_to_train], columns=combos)
	time_df.to_csv('figures/time/time_for_testing'+str(units)[1:-1]+'.csv')
