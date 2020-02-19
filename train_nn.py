import models.net as net
import models.nn_utils as nn_utils

import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, normalize
from utils.composition import generate_features

# Suppress info messages
tf.get_logger().setLevel('ERROR')


class Model:
    """
        Creates the model given a material property to learn, a representation,
        an architecture, a list of the number of units at each layer,
        and batch size.
    """

    def __init__(self,
                 df_t,
                 df_v,
                 representation='magpie',
                 model='net',
                 units=[512],
                 batch_size=1):

        self.batch_size = batch_size
        self.val_size = batch_size*(df_v.values.shape[0]//batch_size)

        df_v = df_v.iloc[:self.val_size, :]
        tf.reset_default_graph()

        print('Truncating test or validation to', df_v.shape[0],
              'to be divisible by batch_size')

        # Featurize train and validation data
        print('Prepping Data')
        train_feats, train_labels, _ = generate_features(df_t, representation)
        val_feats, val_labels, _ = generate_features(df_v, representation)

        # Scale and normalize data
        scaler = StandardScaler()
        scaler.fit(train_feats)
        train_feats = normalize(scaler.transform(train_feats))
        train_feats = pd.DataFrame(train_feats)
        val_feats = normalize(scaler.transform(val_feats))
        val_feats = pd.DataFrame(val_feats)

        # Get dataset
        train_dataset = nn_utils.get_tf_dataset_pipeline(train_feats,
                                                         train_labels,
                                                         batch_size)
        val_dataset = nn_utils.get_tf_dataset_pipeline(val_feats,
                                                       val_labels,
                                                       batch_size,
                                                       shuffle=False)

        # Define labels
        self.train_labels = train_dataset[:, 0]
        self.val_labels = val_dataset[:, 0]
        t_feats = train_dataset[:, 1:]
        v_feats = val_dataset[:, 1:]

        # Create model
        print('Creating Model')
        self.train_predictions = net.network(t_feats,
                                             layers=units,
                                             training=True)
        self.val_predictions = net.network(v_feats,
                                           layers=units,
                                           training=False)

        # Defining loss criterion and optimizer
        loss = tf.losses.mean_squared_error(
                        labels=tf.reshape(self.train_labels, (-1,)),
                        predictions=tf.reshape(self.train_predictions, (-1,)))
        self.opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        # Start session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        print('Model Created')

    def train(self, batches, checkin, save_weights_to_path=None):
        """
        Returns the max validation and loss given a number of batches and a
        number of batches to fetch train and validation metrics.
        """
        saver = tf.train.Saver()

        t_mse = []
        t_mae = []
        t_r2 = []
        v_mse = []
        v_mae = []
        v_r2 = []

        # Train and get metrics
        for iteration in range(batches):
            self.sess.run(self.opt)
            if iteration % checkin == 0:
                t_pred = []
                t_lab = []
                v_pred = []
                v_lab = []
                for _ in range(self.val_size//self.batch_size):
                    output = self.sess.run([self.train_labels,
                                            self.train_predictions,
                                            self.val_labels,
                                            self.val_predictions])
                    t_lab += output[0].flatten().tolist()
                    t_pred += output[1].flatten().tolist()
                    v_lab += output[2].flatten().tolist()
                    v_pred += output[3].flatten().tolist()

                t_r2.append(r2_score(t_lab, t_pred))
                t_mse.append(mean_squared_error(t_lab, t_pred))
                t_mae.append(mean_absolute_error(t_lab, t_pred))
                v_r2.append(r2_score(v_lab, v_pred))
                v_mse.append(mean_squared_error(v_lab, v_pred))
                v_mae.append(mean_absolute_error(v_lab, v_pred))

                is_minimum = v_mae[-1] == min(v_mae)
                if save_weights_to_path is not None and is_minimum:
                    print('Saving current best model')
                    saver.save(self.sess, save_weights_to_path)

                if len(v_mae) > 20:  # checkin at least 20 times
                    num_worse_than_prev = np.array(v_mae[-12:-1]) < \
                                          np.array(v_mae[-11:])
                    criterion = np.count_nonzero(num_worse_than_prev)
                    # If majority of validation results are worse compared to
                    # the previous time steps, then stop
                    if criterion > 6:
                        print('\n' + 57*'-')
                        print('Early stopping triggered due to ' +
                              'validation not improving.')
                        print(57*'-' + '\n')
                        break

        best_mae_idx = v_mae.index(min(v_mae))
        return (t_mae[best_mae_idx],
                t_mse[best_mae_idx],
                t_r2[best_mae_idx],
                v_mae[best_mae_idx],
                v_mse[best_mae_idx],
                v_r2[best_mae_idx])

    def predict(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

        v_mse = []
        v_mae = []
        v_r2 = []
        v_lab = []
        v_pred = []

        for _ in range(self.val_size // self.batch_size):
            output = self.sess.run([self.val_labels,
                                    self.val_predictions])
            v_lab += output[0].flatten().tolist()
            v_pred += output[1].flatten().tolist()

        v_r2.append(r2_score(v_lab, v_pred))
        v_mse.append(mean_squared_error(v_lab, v_pred))
        v_mae.append(mean_absolute_error(v_lab, v_pred))

        data_l = []
        data_l.append([v_mae, v_mse, v_r2])
        csver = pd.DataFrame(data_l,
                             index=[self.val_size],
                             columns=['Test_MAE',
                                      'Test_MSE',
                                      'Test_R2'])
        print(csver)
        csv_path = path.split('/')[2].split('.ckpt')[0]
        print('Saving to:', csv_path, '\n')
        csver.to_csv('figures/test_results/' + csv_path + '.csv')
