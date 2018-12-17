from os import listdir
import pandas as pd

from lstm_v03 import create_model, get_df


HISTORY_PATH = './data/history/{}/'
CHECKPOINT_PATH = './data/models/{}/'
ASSETS = ['WFC.N', 'AMZN.O', 'A.N', 'BHE.N']

def models_to_csv():
	for asset in ASSETS:
		models = [f for f in listdir(CHECKPOINT_PATH.format(asset))]
		models = pd.DataFrame(models)
		models = models[0].str[:-5]
		models = models.str.split('-', expand=True)
		models = models.drop([0, 1], axis=1)
		models.columns = [
			'epoch', 'val_loss', 'asset', 'lstm_size', 'lag', 'dropout']

		# Cast to numeric
		models['epoch'] = pd.to_numeric(models['epoch'])
		models['val_loss'] = pd.to_numeric(models['val_loss'])
		models['lstm_size'] = pd.to_numeric(models['lstm_size'])
		models['lag'] = pd.to_numeric(models['lag'])
		models['dropout'] = pd.to_numeric(models['dropout'])

	    # Write to csv file
		models.to_csv('./data/lstm-{}-results.csv'.format(asset))


def perform_test_best_model():
	test_frac = 0.1

	lstm_size = 64
	lag = 15
	dropout = 0.1

	for asset in ASSETS:

		print(asset)

		df_lag, df_lag_test, n_features = get_df(test_frac, asset)
		X_test = df_lag_test.drop('y', axis=1)
		y_test = df_lag_test['y']

		X_test = X_test.values.reshape((-1, lag+1, n_features))

		w = [f for f in listdir(CHECKPOINT_PATH.format(asset))][-1]
		model = create_model(lstm_size, dropout, lag, n_features)
		model.load_weights(CHECKPOINT_PATH.format(asset) + w)
		scores = model.evaluate(X_test, y_test, verbose=0)
		print(scores[0], scores[1])

