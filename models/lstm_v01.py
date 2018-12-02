from data_partitioning import validate_df
from data_partitioning import split_fixed_origin
from data_cleaning import get_cleaned_filtered_data, extract_asset

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import plot_model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DRY_RUN = True # if True, will only run for one asset with fixed origin strategy

ASSETS = ['INTC.O', 'WFC.N', 'AMZN.O', 'A.N', 'BHE.N']
DATA_PATH = './data/processed/cleaned_filtered_data.csv'


test_frac = 0.1 # fraction of the whole data
train_frac = 0.8 # fraction of the remaining data
latent_dim = 50 # LSTM hidden units
batch_size = 1
look_back = 30


def create_dataset(X, look_back=1):
	cols = list()
	for i in range(look_back, 0, -1):
		cols.append(X.shift(i))

	return pd.concat(cols, axis=1)

if __name__ == '__main__':
	X, y = get_cleaned_filtered_data(DATA_PATH)


	for asset in ASSETS:
		X, y = extract_asset(X, y, asset)
		X['y'] = y

		# Isolating the test set
		split = len(X) - round(test_frac*len(X))
		X_test = X[split:]
		y_test = X_test['y']
		X_test = X_test.drop(['y'], axis=1)
		X = X[:split]

		# Training and validating the model using fixed origin
		train_size = round(train_frac * len(X))

		for X_train, X_val in split_fixed_origin(X, train_size):
			y_train = X_train['y']
			X_train = X_train.drop(['y'], axis=1)
			y_val = X_val['y']
			X_val = X_val.drop(['y'], axis=1)

			# fill nan ad drop the asset code and time
			drop_col = ['Unnamed: 0', 'assetCode', 'time']
			X_train.fillna(0, inplace=True)
			X_val.fillna(0, inplace=True)
			X_train.drop(drop_col, axis=1, inplace=True)
			X_val.drop(drop_col, axis=1, inplace=True)

			# Create the sets according to the look_back range
			X_train = create_dataset(X_train, look_back)

			# input dimensionality
			data_dim = X_train.shape[-1]

			# Reshape input to 3 dimensions (batch_size, timesteps, data_dim)
			X_train = X_train.reshape((batch_size, X_train.shape[0], data_dim))
			X_val = X_val.reshape((batch_size, X_val.shape[0], data_dim))
			y_train = y_train.reshape((batch_size, -1, 1))
			y_val = y_val.reshape((batch_size, -1, 1))

			# Expected input shape: (batch_size, timesteps, data_dim)
			model = Sequential()
			model.add(LSTM(latent_dim, input_dim=data_dim, 
				return_sequences=True))
			model.add(Dense(1))
			model.compile(loss='mse', optimizer='adam')
			history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
					  			epochs=60, batch_size=batch_size)
		
			# plot training history
			plt.plot(history.history['loss'])
			plt.plot(history.history['val_loss'])

			plt.xlabel('Epoch')
			plt.ylabel('Mean Absolute Error Loss')
			plt.title('Loss Over Time')
			plt.legend(['Train','Val'])

		if DRY_RUN:
			break;

