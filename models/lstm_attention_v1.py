import numpy as np
import sys
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

from models.data_cleaning import generate_cleaned_filtered_data
from models.attention_decoder import AttentionDecoder
from models.data_partitioning import validate_df
from models.data_partitioning import split_fixed_origin
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

test_frac = 0.1  # fraction of the whole data
train_frac = 0.8  # fraction of the remaining data

cleaned_data_path = './data/processed/df_merged.csv'

ASSETS = ['INTC.O', 'WFC.N', 'AMZN.O', 'A.N', 'BHE.N']


def top_down_acc(y_true, y_pred):
    return K.abs(K.sign(y_true) + K.sign(y_pred)) / 2


def time_lag_data(X, y, n_in=1, n_out=1):
    n_features = X.shape[1]
    feature_names = X.columns

    # Define column names
    names = list()
    for i in range(n_in):
        names += [('%s(t-%d)' % (feature_names[j], -(i+1-n_in))) for j in range(n_features)]

    x_list = []
    # input sequence (t-n, ... t-1)
    for i in range(X.shape[0]-n_in-n_out+2):
        rows_x = []
        for _, row in X[i:i+n_in].iterrows():
            rows_x += row.tolist()
        x_list.append(rows_x)

    X_time = pd.DataFrame(x_list, columns=names)
    # forecast sequence (t, t+1, ... t+n)
    cols = list()
    for i in range(0, n_out):
        if i == 0:
            cols += [('%s(t)' % ('returnsOpenNextMktres10'))]
        else:
            cols += [('%s(t+%d)' % ('returnsOpenNextMktres10', i))]
    # put it all together

    y_list = []
    # input sequence (t-n, ... t-1)
    for i in range(n_in-1, X.shape[0]-n_out+1):
        y_list.append(y[i:i+n_out].tolist())

    y_time = pd.DataFrame(y_list, columns=cols)

    return X_time, y_time


df = pd.read_csv(cleaned_data_path)
df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'time'], inplace=True, axis=1)

# For loop for assets
asset = 'INTC.O'
df = df[df['assetCode'] == 'INTC.O']
df.drop(['assetCode'], axis=1, inplace=True)

split = len(df) - round(test_frac*len(df))
df_test = df[split:]
df_tv = df[:split]

# For loop for different splitting techniques
df_train, df_val = train_test_split(df_tv,
                                    train_size=train_frac,
                                    shuffle=False)

y_train = df_train['returnsOpenNextMktres10']
y_train_init = y_train.reset_index(drop=True)
X_train = df_train.drop(['returnsOpenNextMktres10'], axis=1)
X_train_init = X_train.reset_index(drop=True)
print('The train data size is : ', X_train.shape, y_train.shape)

y_val = df_val['returnsOpenNextMktres10']
y_val_init = y_val.reset_index(drop=True)
X_val = df_val.drop(['returnsOpenNextMktres10'], axis=1)
X_val_init = X_val.reset_index(drop=True)
print('The validation data size is : ', X_val.shape, y_val.shape)

y_test = df_test['returnsOpenNextMktres10']
y_test_init = y_test.reset_index(drop=True)
X_test = df_test.drop(['returnsOpenNextMktres10'], axis=1)
X_test_init = X_test.reset_index(drop=True)
print('The test data size is : ', X_test.shape, y_test.shape)

# TODO : test data
# TODO : IMAGES

# Hyperparameter tuning
# lag (1, 5, 15, 30, 60, 90), dropout (LSTM) (0, 0.05, 0.4), cells (16, 32, 64)
n_features = 40
n_timesteps_out = 1
n_epochs = 100

for n_timesteps_in in [1, 5, 15, 30, 60, 90]:
    for dropout in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        for cells in [16, 32, 64]:

            X_train, y_train = time_lag_data(X_train_init, y_train_init,
                                             n_in=n_timesteps_in,
                                             n_out=n_timesteps_out)
            print('The train data size is : ', X_train.shape, y_train.shape)

            X_val, y_val = time_lag_data(X_val_init, y_val_init,
                                         n_in=n_timesteps_in,
                                         n_out=n_timesteps_out)
            print('The val data size is : ', X_val.shape, y_val.shape)

            scaler = MinMaxScaler((-1, 1), False)
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            # Reshape the datasets
            X_train = X_train.reshape((len(X_train), n_timesteps_in, n_features))
            y_train = y_train.values.reshape((len(y_train), n_timesteps_out, 1))

            X_val = X_val.reshape((len(X_val), n_timesteps_in, n_features))
            y_val = y_val.values.reshape((len(y_val), n_timesteps_out, 1))


            # Model with Encoder/Decoder
            model = Sequential()
            model.add(LSTM(cells, dropout=dropout,
                           input_shape=(n_timesteps_in, n_features)))
            model.add(RepeatVector(n_timesteps_out))
            model.add(LSTM(cells, dropout=dropout, return_sequences=True))
            model.add(TimeDistributed(Dense(1, activation='tanh')))
            model.compile(loss='mean_squared_error', optimizer='adam',
                          metrics=[top_down_acc])
            model.summary()
            history = model.fit(X_train,
                                y_train,
                                epochs=n_epochs,
                                validation_data=(X_val, y_val),
                                shuffle=False)

            with open('history_ed_v0_ts_{}_drop_{}_cells_{}'.format(str(n_timesteps_in),
                                                                    str(dropout),
                                                                    str(cells)), 'wb') as file_hs:
                pickle.dump(history.history, file_hs)

            # plot training history
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(history.history['loss'])
            ax.plot(history.history['val_loss'])
            ax.set_xlim([0, 125])
            ax.set_ylim([0, 0.01])
            # plt.plot(history.history['top_down_acc'])

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Mean Absolute Error Loss')
            ax.set_title('Loss Over Time')
            ax.legend(['Train','Val'])
            # plt.legend(['Train','Val', 'Top Down Accuracy'])
            fig.savefig('lstm_ed_v0_ts_{}_drop_{}_cells_{}.png'.format(str(n_timesteps_in),
                                                                    str(dropout),
                                                                    str(cells)))
prediction = model_at.predict(X_test)
prediction[:,0].shape
y_test[:,0].shape
r = sum(top_down_acc(p[0], np.float32(t[0])) for p, t in zip(prediction[:,0], y_test[:,0]))/len(y_test)
with tf.Session() as sess:
    a = sess.run(r)
a

for n_timesteps_in in [5]:  # [1, 5, 15, 30, 60, 90]:
    for dropout in [0]:  # [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        for cells in [64]:  # [16, 32, 64]:

            n_timesteps_out = n_timesteps_in

            X_train, y_train = time_lag_data(X_train_init, y_train_init,
                                             n_in=n_timesteps_in,
                                             n_out=n_timesteps_out)
            print('The train data size is : ', X_train.shape, y_train.shape)

            X_val, y_val = time_lag_data(X_val_init, y_val_init,
                                         n_in=n_timesteps_in,
                                         n_out=n_timesteps_out)
            print('The val data size is : ', X_val.shape, y_val.shape)

            X_test, y_test = time_lag_data(X_test_init, y_test_init,
                                         n_in=n_timesteps_in,
                                         n_out=n_timesteps_out)
            print('The test data size is : ', X_test.shape, y_test.shape)

            scaler = MinMaxScaler((-1, 1), False)
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            # Reshape the datasets
            X_train = X_train.reshape((len(X_train), n_timesteps_in, n_features))
            y_train = y_train.values.reshape((len(y_train), n_timesteps_out, 1))

            X_val = X_val.reshape((len(X_val), n_timesteps_in, n_features))
            y_val = y_val.values.reshape((len(y_val), n_timesteps_out, 1))

            X_test = X_test.reshape((len(X_test), n_timesteps_in, n_features))
            y_test = y_test.values.reshape((len(y_test), n_timesteps_out, 1))


            # Model with Encoder/Decoder
            model_at = Sequential()
            model_at.add(LSTM(cells, input_shape=(n_timesteps_in, n_features),
                              return_sequences=True))
            model_at.add(AttentionDecoder(cells, n_features))
            model_at.add(Dense(1, activation='tanh'))
            model_at.compile(loss='mean_squared_error', optimizer='adam',
                             metrics=[top_down_acc])
            model_at.summary()
            history = model_at.fit(X_train,
                                y_train,
                                epochs=n_epochs,
                                validation_data=(X_val, y_val),
                                shuffle=False)

            with open('results/history_att_v0_ts_{}_drop_{}_cells_{}'.format(str(n_timesteps_in),
                                                                    str(dropout),
                                                                    str(cells)), 'wb') as file_hs:
                pickle.dump(history.history, file_hs)

            prediction = model_at.predict(X_test)
            top_down_accuracy = sum(top_down_acc(p[0], np.float32(t[0])) for p, t in zip(prediction[:,0], y_test[:,0]))/len(y_test)

            with tf.Session() as sess:
                top_down_accuracy = sess.run(top_down_accuracy)
            # plot training history
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(history.history['loss'])
            ax.plot(history.history['val_loss'])
            ax.set_xlim([0, 100])
            ax.set_ylim([0, 0.01])
            # plt.plot(history.history['top_down_acc'])

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Mean Absolute Error Loss')
            ax.set_title('Loss Over Time - Predicted Top-Down Accuracy : {}'.format(str(top_down_accuracy)))
            ax.legend(['Train','Val'])
            # plt.legend(['Train','Val', 'Top Down Accuracy'])
            fig.savefig('results/lstm_att_v0_ts_{}_drop_{}_cells_{}.png'.format(str(n_timesteps_in),
                                                                    str(dropout),
                                                                    str(cells)))
