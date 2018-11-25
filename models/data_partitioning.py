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


def split_fixed_origin(X, train_size):
    ''' Generator that yields training and validation sets according to the 
    fixed-origin evaluation strategy.

    Fixed-origin evaluation is typically applied during forecasting 
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
    train_ratio : int
        The size of the training set.
    
    Returns
    -------
    dataframe
        The training set.
    dataframe
        The validation set.

    '''
    yield np.split(X, [train_size])


def split_rolling_origin_recal(X, initial_train_size, rolling_size):
    ''' Generator that yields training and validation sets according to the
    rolling-origin-recalibration evaluation strategy.

    Within rolling-origin-recalibration evaluation, forecasts for a fixed 
    horizon are performed by sequentially moving values from the test set to the 
    training set, and changing the forecast origin accordingly. For each 
    forecast, the model is recalibrated using all available data in the training 
    set, which often means a complete retraining of the model 
    (Bergmeir & Benitez, 2012).

    Parameters
    ----------
    X : dataframe
        The data to be split.
    initial_train_size : int
        The initial size of the training set.
    rolling_size : int
        The number of elements that are moved from the validation set to the
        training set at each iteration.
    
    Returns
    -------
    dataframe
        The test set.
    dataframe
        The validation set.

    '''
    pointer = initial_train_size
    while pointer < len(X):
        yield X[:pointer], X[pointer:]
        pointer += rolling_size


def split_rolling_origin_update(X, train_size, val_size):
    ''' Generator that yields a training and a validation sets according to the 
    rolling_origin_update strategy. Essentially, this is the same as 
    split_rolling but the model should not be recalibrated but simply updated 
    after each subsequent iteration.

    After the first iteration which 

    Rolling-origin-update evaluation is probably the normal use case of most 
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
    train_window_size : int
        The number of data points to be included in the training set window.
    val_window_size : int
        The number of data points to be included in the validation set window.
    
    Returns
    -------
    dataframe
        The test set followed by one new observation at a time.
    dataframe
        The validation set followed by None after the first iteration.

    '''   
    yield (X[:train_size], 
           X[train_size:train_size+val_size])
    pointer = train_size+val_size

    while pointer <= len(X):
        yield X[pointer]
        pointer += 1


def split_rolling_window(X, train_size, val_size, shift):
    ''' Generator that yields training and validation sets according to the
    rolling-window evaluation strategy.

    Rolling-window evaluation is similar to rolling-origin evaluation, but
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
    train_window_size : int
        The number of data points to be included in the training set window.
    val_window_size : int
        The number of data points to be included in the validation set window.
    shift : int
        By how many data points do the windows shift after each iteration.
    
    Returns
    -------
    dataframe
        The test set.s
    dataframe
        The validation set.

    '''
    
    pointer = 0
    while pointer + train_size + val_size <= len(X):
        yield (X[pointer:pointer+train_size], 
               X[pointer+train_size:pointer+train_size+val_size])
        pointer += shift


if __name__ == '__main__':


    # Test split_data_ordered
    X = pd.DataFrame(np.random.randint(0, 100, size=(101, 2)), 
                     columns=list('AB'))
    y = pd.DataFrame(np.random.randint(0, 2, size=(101, 1)), 
                     columns=['target'])
    time = range(0, 101)
    X['time'] = time
    y['time'] = time

    # Unit tests setup
    df = pd.DataFrame({'A':range(10)})
    
    # Unit tests for split_rolling_origin_recal
    len_i = 4
    len_j = 6
    for i, j in split_rolling_origin_recal(df, 4, 2):
        assert len(i) == len_i and len(j) == len_j
        assert len(i) != 0 and len(j) != 0
        len_i += 2
        len_j -= 2


    # Unit tests for split_rolling_origin_update

    # Unit tests for split_rolling_window
    for i, j in split_rolling_window(df, 4, 2, 2):
        print(i.values.reshape(1,-1))
        print(j.values.reshape(1,-1))
        print()

