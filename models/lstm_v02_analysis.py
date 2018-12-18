from os import listdir
from os.path import isfile, join
import pandas as pd
import pickle
import re

from sklearn.preprocessing import MinMaxScaler

from lstm_v02 import create_model, add_lag
from data_partitioning import split_fixed_origin
from data_cleaning import get_cleaned_filtered_data, extract_asset


DATA_PATH = './data/processed/cleaned_filtered_data.csv'

test_frac = 0.1
train_frac = 0.8


asset = 'INTC.O'
val_loss = 0.0016
epoch = 7
lstm_size = 32
lag = 60
dropout = 40


def get_saved_model_path(root, val_loss, epoch, asset, lstm_size, lag, dropout):

	# Generate file path from parameters
	return root + 'best-lstm-{:03}-{}-{}-{}-{}-{}.hdf5'.format(
		epoch, val_loss, asset, lstm_size, lag, dropout)


if __name__ == '__main__':

	# Fetch the data
	X_clean, y_clean = get_cleaned_filtered_data(DATA_PATH)
	X, y = extract_asset(X_clean, y_clean, asset)
	cols = ['Unnamed: 0', 'assetCode', 'time']
	X.drop(cols, axis=1, inplace=True)
	X.fillna(-1, inplace=True)
	n_features = X.shape[1]

	# Split the data 
	df = X
	df['y'] = y
	split = len(df) - round(test_frac*len(df))
	df_test = df[split:]
	df = df[:split] 

	print(len(df_test))

	# Add the lag features
	df_lag = add_lag(df.drop(['y'], axis=1), lag)
	df_lag['y'] = df['y']
	df_test_lag = add_lag(df_test.drop(['y'], axis=1), lag)
	df_test_lag['y'] = df_test['y']

	X_test = df_test_lag.drop(['y'], axis=1)
	y_test = df_test_lag['y']

	train_size = round(train_frac * len(df_lag))
	for df_train, df_val in split_fixed_origin(df_lag, train_size):
		X_train = df_train.drop(['y'], axis=1)

		# Scale the data
		scaler = MinMaxScaler((-1, 1), False)
		scaler.fit_transform(X_train)
		scaler.transform(X_test)

	# Reshape to keras input shape
	X_test = X_test.values.reshape((-1, lag+1, n_features))

	# Create the model from saved weights
	weights_path = get_saved_model_path(
		'./data/models/', val_loss, epoch, asset, lstm_size, lag, dropout)
	model = create_model(lstm_size, dropout, lag, n_features)
	model.load_weights(weights_path)

	# Test and print the results
	scores = model.evaluate(X_test, y_test, verbose=0)
	print('\n{} : {}\n{} : {}'.format(
		model.metrics_names[0], scores[0], model.metrics_names[1], scores[1]))