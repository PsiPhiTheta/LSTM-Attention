import numpy as np
import pandas as pd



def split_data_ordered(X, y, strategy='lastBlockEval', train_ratio=0.7, 
	                   val_ratio=0.2, sort_column='time'):
	'''split the given data into training, validation and test sets in an 
	ordered fashion according to the desired strategy. The proportion accorded 
	to the test set is infered from the training and the validation sets' 
	ratios.

	Parameters
	----------
	X : dataframe
		The data to be split.
	y : dataframe
		The labels to be split.
	strategy : {'lastBlockEval'}
		The strategy to be used for spliting the data. 
	train_ratio : float
		The ratio of data points that should be included in the training set.
	val_ratio : float
		The ratio of data points that should be included in the validation set.
	sort_column : String
		Column on which the data should be sorted. Defaults to 'time'.

	Returns
	-------
	X_train : list of dataframe
		A list containing the training sets.
	y_train : list of dataframe
		A list containing the training labels sets.
	X_val : list of dataframe
		A list containing the validation sets.
	y_val : list of dataframe
		A list containing the validation labels sets.
	X_test : dataframe
		The test set
	y_test : dataframe
		The test set
	
	'''

	if len(X) != len(y):
		raise Exception('X and y should have the same length: len(X) is {}, \
			            len(y) is {}'.format(len(X), len(y)))

	if sort_column not in X.columns:
		raise Exception('X should have a column named {}'.format(sort_column))

	if sort_column not in y.columns:
		raise Exception('y should have a column named {}'.format(sort_column))

	X.sort_values(by=[sort_column])
	y.sort_values(by=[sort_column])

	if (strategy is 'lastBlockEval'):
		X_train, X_val, X_test = _last_block_eval(X, train_ratio, val_ratio)
		y_train, y_val, y_test = _last_block_eval(y, train_ratio, val_ratio)
		return (X_train, y_train, X_val, y_val, X_test, y_test)


	raise Exception('Unknown strategy: {}'.format(strategy))



def _last_block_eval(X, train_ratio, val_ratio):
	'''split the given data into train, validation and test sets in an ordered 
	fashion.

	Parameters
	----------
	X : dataframe
		The data to be split.
	train_ratio : float
		The ratio of data points that should be included in the training set.
	val_ratio : float
		The ratio of data points that should be included in the validation set.
	
	Returns
	-------
	list of dataframe
		List containing all training dataframes.
	list of dataframe
		List containing all the validation dataframes.
	dataframe
		The test set.

	'''
	if X is None:
		return None

	# Split and return the data
	n = len(X)
	splits = [int(train_ratio*n), int(train_ratio*n + val_ratio*n)]
	train, val, test = np.split(X, splits)
	return ([train], [val], test)


if __name__ == '__main__':
	

	# Unit test for last block eval
	X = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), 
					 columns=list('ABCD'))
	a, b, c = _last_block_eval(X, 0.7, 0.2)
	assert len(a[0]) == 70
	assert len(b[0]) == 20
	assert len(c) == 10

	# Test split_data_ordered
	X = pd.DataFrame(np.random.randint(0, 100, size=(101, 2)), 
					 columns=list('AB'))
	y = pd.DataFrame(np.random.randint(0, 2, size=(101, 1)), 
					 columns=['target'])
	time = range(0, 101)
	X['time'] = time
	y['time'] = time
	xt, yt, xv, yv, xtt, ytt = split_data_ordered(X, y, 'lastBlockEval')
