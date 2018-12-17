from os import listdir
import matplotlib.pyplot as plt
import pickle

from lstm_v03 import create_model, get_df

def concat_history():
	path = './data/history/lstm_recal/'
	keys = ['val_loss', 'val_top_down_acc', 'loss', 'top_down_acc']
	
	hist_list = listdir(path)
	history = {key: [] for key in keys}
	
	for hist_name in hist_list:
		with open(path + hist_name, 'rb') as f:
			hist = pickle.load(f)
		
		for key in keys:
			history[key] += hist[key]
	
	return history


def plot_train_loss(history, ylim=(0, 0.03)):
    plt.ylim(ylim)

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])

    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error Loss')
    plt.title('Training Loss')
    plt.legend(['Train','Val'])
    plt.show()


def perform_tests():

	test_frac = 0.1

	asset = 'INTC.O'
	lstm_size = 64
	lag = 15
	dropout = 10

	path = './data/models/rollingwindow/'
	models = listdir(path)

	df_lag, df_lag_test, n_features = get_df(test_frac)
	X_test = df_lag_test.drop('y', axis=1)
	y_test = df_lag_test['y']

	# Reshape input data according to Keras documentation
	# (batch_size, timesteps, input_dim)
	X_test = X_test.values.reshape((-1, lag+1, n_features))

	model = create_model(lstm_size, dropout, lag, n_features)

	f = open('data/lstm_recalibration.csv', 'w+')
	f.write(model.metrics_names[0] + ',' + model.metrics_names[1] + '\n')
	
	for model_name in models:
		model.load_weights(path + model_name)
		scores = model.evaluate(X_test, y_test, verbose=0)
		f.write('{},{}\n'.format(scores[0], scores[1]))

	f.close()


if __name__ == '__main__':
	history = concat_history()
	# plot_train_loss(history)
	
	# perform_tests()