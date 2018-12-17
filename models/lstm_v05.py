''' In this version, I train the model on INTC.O using the hyper parameters 
found withthe version 02 but using the rolling origin update splitting strategy.
'''

import pickle
from keras.callbacks import ModelCheckpoint

from data_partitioning import split_fixed_origin
from data_cleaning import get_cleaned_filtered_data, extract_asset

from lstm_v03 import add_lag, top_down_acc, create_model, get_df


DATA_PATH = './data/processed/cleaned_filtered_data.csv'
HISTORY_PATH = './data/history/{}/'
CHECKPOINT_PATH = './data/models/{}/'

ASSETS = ['WFC.N', 'AMZN.O', 'A.N', 'BHE.N']

test_frac = 0.1
train_frac = 0.8
n_epochs = 50

lstm_size = 64
lag = 15
dropout = 0.1


if __name__ == '__main__':

	for asset in ASSETS :

		# Get the cleaned and processed data
		df_lag, _, n_features = get_df(test_frac, asset)

		# Instantiate the model
		model = create_model(lstm_size, dropout, lag, n_features)

		# Train and evaluate the model
		train_size = round(train_frac * len(df_lag))
		for df_train, df_val in split_fixed_origin(df_lag, train_size):
			y_train = df_train['y']
			X_train = df_train.drop(['y'], axis=1)
			y_val = df_val['y']
			X_val = df_val.drop(['y'], axis=1)

			# Reshape to match Keras input shape (batch_size, timsteps, input_dim)
			X_train = X_train.values.reshape((-1, lag+1, n_features))
			X_val = X_val.values.reshape((-1, lag+1, n_features))

			# Some user feedback
			print('\nFitting model for {}\n'.format(asset))

			# Fit the model
			checkpoint_name = ('best-lstm-{{epoch:03d}}-{{val_loss:.4f}}-{}-'
				'{}-{}-{}.hdf5').format(asset, lstm_size, lag, int(dropout*100))
			checkpoint = ModelCheckpoint(
				CHECKPOINT_PATH.format(asset) + checkpoint_name,
				monitor='val_loss',
				save_best_only=True)

			history = model.fit(X_train, 
								y_train, 
								epochs=n_epochs, 
								validation_data=(X_val, y_val),
								shuffle=False,
								callbacks=[checkpoint])

			# Dumpm the history to a pickle file
			path = (HISTORY_PATH.format(asset) + 'lstm-{}-{}-{}-{}.pickle'
				.format(asset, lstm_size, lag, int(dropout*100)))
			with open(path, 'wb') as f:
				pickle.dump(history.history, f)


