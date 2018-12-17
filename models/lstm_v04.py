''' In this version, I train the model on INTC.O using the hyper parameters 
found withthe version 02 but using the rolling origin recalibration splitting 
strategy.
'''

import pickle
from keras.callbacks import ModelCheckpoint

from data_partitioning import split_rolling_origin_recal
from data_cleaning import get_cleaned_filtered_data, extract_asset

from lstm_v03 import add_lag, top_down_acc, create_model, get_df


DATA_PATH = './data/processed/cleaned_filtered_data.csv'
HISTORY_PATH = './data/history/lstm_recal/'
CHECKPOINT_PATH = './data/models/lstm_recal/'

test_frac = 0.1
n_epochs = 1

init_train_frac = 0.1
rolling_size = 10

asset = 'INTC.O'
lstm_size = 64
lag = 15
dropout = 0.10


if __name__ == '__main__':

	# Get the cleaned and processed data
	df_lag, _, n_features = get_df(test_frac, asset)

	# Instantiate the model
	model = create_model(lstm_size, dropout, lag, n_features)

	# Train and evaluate using the rolling origin recalibration strategy
	init_train_size = round(init_train_frac * len(df_lag))
	count = -1
	for df_train, df_val in split_rolling_origin_recal(df_lag, 
			init_train_size, rolling_size):
		count += 1
		y_train = df_train['y']
		X_train = df_train.drop(['y'], axis=1)
		y_val = df_val['y']
		X_val = df_val.drop(['y'], axis=1)

		# Reshape to match Keras input shape (batch_size, timsteps, input_dim)
		X_train = X_train.values.reshape((-1, lag+1, n_features))
		X_val = X_val.values.reshape((-1, lag+1, n_features))

		# Fit the model
		checkpoint_name = ('best-lstm-{:03d}-{}-{}-{}-{}.hdf5').format(
			count, asset, lstm_size, lag, int(dropout*100))
		checkpoint = ModelCheckpoint(
			CHECKPOINT_PATH + checkpoint_name,
			monitor='val_loss',
			save_best_only=True)

		history = model.fit(X_train, 
							y_train, 
							epochs=n_epochs, 
							validation_data=(X_val, y_val),
							shuffle=False,
							callbacks=[checkpoint])

		# Dumpm the history to a pickle file
		path = (HISTORY_PATH + 'lstm.{:03d}-{}-{}-{}-{}.pickle'.format(
			count, asset, lstm_size, lag, int(dropout*100)))
		with open(path, 'wb') as f:
			pickle.dump(history.history, f)


