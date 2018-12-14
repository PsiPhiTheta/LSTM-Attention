from data_partitioning import validate_df
from data_partitioning import split_fixed_origin
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

ASSETS = ['INTC.O', 'WFC.N', 'AMZN.O', 'A.N', 'BHE.N']
DATA_PATH = './data/processed/cleaned_filtered_data.csv'
HISTORY_TOP_PATH = './data/history/'

lag = 20 # size of look back
test_frac = 0.1 # fraction of the whole data
train_frac = 0.8 # fraction of the remaining data
lstm_size = 64
n_epochs = 200
dropout = 0.1

lstm_sizes = [16, 32, 64]
lags = [1, 5, 15, 30, 60, 90]
dropouts = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def add_lag(df, lag=1):
	cols = [df]
	for i in range(lag, 0, -1):
		cols.append(df.shift(i))
	return pd.concat(cols, axis=1).dropna()


def plot_train_loss(history, ylim=(0, 0.03)):
	''' Plot the training and validation loss.

	Parameters
	----------
	history : dict
		Dictionary as loaded from the saved pickle files.

	'''
	plt.ylim(ylim)

	plt.plot(history['loss'])
	plt.plot(history['val_loss'])

	plt.xlabel('Epoch')
	plt.ylabel('Mean Absolute Error Loss')
	plt.title('Training Loss')
	plt.legend(['Train','Val'])
	plt.show()
	

def top_down_acc(y_true, y_pred):
	return K.abs(K.sign(y_true) + K.sign(y_pred)) / 2

if __name__ == '__main__':
	
	# Fetch the data from the saved csv
	X_clean, y_clean = get_cleaned_filtered_data(DATA_PATH)

	for asset, lstm_size, lag, dropout in product(
			ASSETS, lstm_sizes, lags, dropouts):

		# Extract the asset and perform some cleaning
		X, y = extract_asset(X_clean, y_clean, asset)
		cols = ['Unnamed: 0', 'assetCode', 'time']
		X.drop(cols, axis=1, inplace=True)
		X.fillna(-1, inplace=True) # Making sure unknown values are obvious
		n_features = X.shape[1]
		
		# Merge the labels and the features into one dataset
		df = X
		df['y'] = y

		# Isolating the test set
		split = len(df) - round(test_frac*len(df))
		df_test = df[split:]
		df = df[:split] 

		# Some user feedback
		print('\nTraining with\n\tlstm size: {}\n\tlag: {}\n\tdropout: {}\n'
			.format(lstm_size, lag, dropout))
	
		# Add the lag features
		df_lag = add_lag(df.drop(['y'], axis=1), lag)
		df_lag['y'] = df['y']

		# Train and evaluate using fixed origin
		train_size = round(train_frac * len(df_lag))
		for df_train, df_val in split_fixed_origin(df_lag, train_size):
			y_train = df_train['y']
			X_train = df_train.drop(['y'], axis=1)
			y_val = df_val['y']
			X_val = df_val.drop(['y'], axis=1)

			# Scale the data
			scaler = MinMaxScaler((-1, 1), False)
			scaler.fit_transform(X_train)
			scaler.transform(X_val)

			# Reshape input data according to Keras documentation
			# (batch_size, timesteps, input_dim)
			X_train = X_train.values.reshape((-1, lag+1, n_features))
			X_val = X_val.values.reshape((-1, lag+1, n_features))

			# Create the model
			# Input shape expected (timesteps, input_dim)
			model = Sequential()
			model.add(LSTM(lstm_size, dropout=dropout, 
						   input_shape=(lag+1, n_features)))
			model.add(Dense(1, activation='tanh'))
			model.compile(loss='mse', optimizer='adam', 
						  metrics=[top_down_acc])

			# Fit the model
			checkpoint_name = ('best-lstm-{{epoch:03d}}-{{val_loss:.4f}}-{}-{}-'
				'{}-{}.hdf5').format(asset, lstm_size, lag, int(dropout*100))
			checkpoint = ModelCheckpoint(
				'./data/models/' + checkpoint_name,
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
				path = HISTORY_TOP_PATH + 'lstm-{}-{}-{}-{}.pickle'.format(
					asset, lstm_size, lag, int(dropout*100))
				with open(path, 'wb') as f:
					pickle.dump(history.history, f)

		if DRY_RUN:
			break