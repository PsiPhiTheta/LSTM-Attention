''' In this version, I train the model on INTC.O using the hyper parameters 
found with the version 02 but using the rolling window splitting strategy.
'''

from data_partitioning import split_rolling_window
from data_cleaning import get_cleaned_filtered_data, extract_asset

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


DRY_RUN = False
DUMP_HISTORY = True

DATA_PATH = './data/processed/cleaned_filtered_data.csv'
HISTORY_TOP_PATH = './data/history/'

test_frac = 0.1 # fraction of the whole data used for test set
n_epochs = 1 # Number of pass over the data when training

# Params for rolling window (fraction of the remaining data)
train_frac = 0.2
val_frac = 0.1
shift = 15

asset = 'INTC.O'
lstm_size = 64
lag = 15
dropout = 0.10


def add_lag(df, lag=1):
	cols = [df]
	for i in range(lag, 0, -1):
		cols.append(df.shift(i))
	return pd.concat(cols, axis=1).dropna()


def top_down_acc(y_true, y_pred):
	return K.abs(K.sign(y_true) + K.sign(y_pred)) / 2


def create_model(lstm_size, dropout, lag, n_features):
	model = Sequential()
	model.add(LSTM(lstm_size, dropout=dropout, 
				   input_shape=(lag+1, n_features)))
	model.add(Dense(1, activation='tanh'))
	model.compile(loss='mse', optimizer='adam', 
				  metrics=[top_down_acc])
	return model


def get_df(test_frac, asset):

	# Fetch the data from the saved csv
	X_clean, y_clean = get_cleaned_filtered_data(DATA_PATH)

	# Extract the asset and perform some cleaning
	df, y = extract_asset(X_clean, y_clean, asset)
	cols = ['Unnamed: 0', 'assetCode', 'time']
	df.drop(cols, axis=1, inplace=True)
	df.fillna(-1, inplace=True) # Making sure unknown values are obvious
	n_features = df.shape[1]
	
	# Merge the labels and the features into one dataset
	df['y'] = y

	# Add the lag features
	df_lag = add_lag(df.drop(['y'], axis=1), lag)
	df_lag = df_lag.assign(y=df['y'])
	total_len = len(df_lag)

	# Isolating the test set
	split = len(df_lag) - round(test_frac*len(df_lag))
	df_lag_test = df_lag[split:]
	df_lag = df_lag[:split] 

	# Scale the data
	scaler = MinMaxScaler((-1, 1), False)

	temp_y = df_lag['y']
	df_lag.drop('y', axis=1, inplace=True)
	scaler.fit_transform(df_lag)
	df_lag['y'] = temp_y

	temp_y = df_lag_test['y']
	df_lag_test.drop('y', axis=1, inplace=True)
	scaler.transform(df_lag_test)
	df_lag_test['y'] = temp_y

	assert total_len == len(df_lag) + len(df_lag_test)

	return df_lag, df_lag_test, n_features


if __name__ == '__main__':

	df_lag, _, n_features = get_df(test_frac, asset)


	# Create the model
	# Input shape expected (timesteps, input_dim)
	model = create_model(lstm_size, dropout, lag, n_features)

	# Train and evaluate using rolling window
	train_size = round(train_frac * len(df_lag))
	val_size = round(val_frac * len(df_lag))
	count = -1
	for df_train, df_val in split_rolling_window(df_lag, train_size, 
			val_size, shift):
		count += 1
		y_train = df_train['y']
		X_train = df_train.drop(['y'], axis=1)
		y_val = df_val['y']
		X_val = df_val.drop(['y'], axis=1)

		# Reshape input data according to Keras documentation
		# (batch_size, timesteps, input_dim)
		X_train = X_train.values.reshape((-1, lag+1, n_features))
		X_val = X_val.values.reshape((-1, lag+1, n_features))

		# Fit the model
		checkpoint_name = ('best-lstm-{:03d}-{}-{}-{}-{}.hdf5').format(
			count, asset, lstm_size, lag, int(dropout*100))
		checkpoint = ModelCheckpoint(
			'./data/models/rollingwindow/' + checkpoint_name,
			monitor='val_loss',
			save_best_only=True)
		history = model.fit(X_train, 
							y_train, 
							epochs=n_epochs, 
							validation_data=(X_val, y_val),
							shuffle=False,
							callbacks=[checkpoint])

		# Dumpm the history to a pickle file
		if DUMP_HISTORY:
			path = (HISTORY_TOP_PATH + 'rollingwindow/lstm.{:03d}-{}-{}-{}-{}'
				'.pickle'.format(count, asset, lstm_size, lag, int(dropout*100)))
			with open(path, 'wb') as f:
				pickle.dump(history.history, f)

		if DRY_RUN:
			break