import numpy as np
import pandas as pd



def validate_df(X, y, sort_column='time'):
	''' Validate the dataset

	Parameters
	----------
	X : dataframe
		The data.
	y : dataframe
		The labels.
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

	return X.sort_values(by=[sort_column]), y.sort_values(by=[sort_column])


def split_fixed_origin(X, train_ratio, val_ratio):
	''' Fixed-origin evaluation is typically applied during forecasting 
	competitions. A forecast for each value present in the test set is computed 
	using only the training set. The forecast origin is fixed to the last point 
	in the training set. So, for each horizon only one forecast can be computed. 
	Obvious drawbacks of this type of evaluation are, that characteristics of 
	the forecast origin might heavily influence evaluation results, and, as only
	one forecast per horizon is present, averaging is not possible within one 
	series and one horizon (Bergmeir & Benitez, 2012). 

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
	n = len(X)
	splits = [int(train_ratio*n), int(train_ratio*n + val_ratio*n)]
	train, val, test = np.split(X, splits)
	return ([train], [val], test)


def split_rolling_origin_recal(X, initial_train_set_size, rolling_size, 
							   test_ratio=0.1):
	'''Within rolling-origin-recalibration evaluation, forecasts for a fixed 
	horizon are performed by sequentially moving values from the test set to the 
	training set, and changing the forecast origin accordingly. For each 
	forecast, the model is recalibrated using all available data in the training 
	set, which often means a complete retraining of the model 
	(Bergmeir & Benitez, 2012).

	Parameters
	----------
	X : dataframe
		The data to be split.
	initial_train_set_size : int
		The initial size of the training set.
	rolling_size : int
		The number of data points moved from the validation set to the train set 
		at each iteration.
	test_ratio : float
		The ratio of data points that should be hold out for the test set.
	
	Returns
	-------
	list of dataframe
		List containing all training dataframes.
	list of dataframe
		List containing all the validation dataframes.
	dataframe
		The test set.

	'''
	split = int(len(X) * test_ratio)
	X = X[:split]
	X_test = X[split:]
	



def split_rolling_origin_update(X, test_ratio=0.1):
	'''Rolling-origin-update evaluation is probably the normal use case of most 
	applications. Forecasts are computed in analogy to rolling-origin-
	recalibration evaluation, but values from the test set are not moved to the 
	training set, and no model recalibration is performed. Instead, past values 
	from the test set are used merely to update the input information of the 
	model. Both types of rolling-origin evaluation are often referred to as 
	n-step-ahead evaluation, with n being the forecast horizon used during the 
	evaluation. Tashman [47] argues that model recalibration probably yields 
	better results than updating. But recalibration may be computationally 
	expensive, and within a real-world application, the model typically will be 
	built once by experts, and later it will be used with updated information as 
	new values are available, but it will certainly not be rebuilt. 
	(Bergmeir & Benitez, 2012).

	Parameters
	----------
	X : dataframe
		The data to be split.
	test_ratio : float
		The ratio of data points that should be hold out for the test set.
	
	Returns
	-------
	list of dataframe
		List containing all training dataframes.
	list of dataframe
		List containing all the validation dataframes.
	dataframe
		The test set.

	'''
	return  None


def split_rolling_window(X, test_ratio=0.1):
	'''Rolling-window evaluation is similar to rolling-origin evaluation, but
	the amount of data used for training is kept constant, so that as new data
	is available, old data from the beginning of the series is discarded. 
	Rolling-window evaluation is only applicable if the model is rebuilt in
	every window, and has merely theoretical statistical advantages, that might
	be noted in practice only if old values tend to disturb model generation 
	(Bergmeir & Benitez, 2012).

	Parameters
	----------
	X : dataframe
		The data to be split.
	test_ratio : float
		The ratio of data points that should be hold out for the test set.
	
	Returns
	-------
	list of dataframe
		List containing all training dataframes.
	list of dataframe
		List containing all the validation dataframes.
	dataframe
		The test set.

	'''
	return None


if __name__ == '__main__':

	# Unit test for last block eval
	X = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), 
					 columns=list('ABCD'))
	a, b, c = _fixed_origin(X, 0.7, 0.2)
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
	xt, yt, xv, yv, xtt, ytt = split_data_ordered(X, y, 'fixedOrigin')
