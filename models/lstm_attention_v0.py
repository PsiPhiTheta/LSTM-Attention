import numpy as np
import sys
import os
import pandas as pd
import glob

sys.path.append('../')
from models.data_cleaning import clean_market_data, clean_news_data

# Import libraries used for lstm
from keras.models import Sequential
from keras.layers import Input, Dense, multiply, Dot, Concatenate
from keras.layers.core import *
from keras.layers import LSTM
from keras.models import *

INPUT_DIM = 43
TIME_STEPS = 1
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False
assetcode_list = ["AMZN.O"]

MARKET_CLEAN_PATH = 'data/processed/market_cleaned_df.csv'
NEWS_CLEAN_PATH = 'data/processed/news_cleaned_df.csv'


def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


def model_attention_applied_after_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 50
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def model_attention_applied_before_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 32
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def extract_stock(df, assetCode, split=False):
    '''Extracts the training data for a particular asset

    Parameters
    ----------
    X_train : pandas dataframe containing all the assets' training data
    y_train : pandas dataframe containing all the assets' labels
    assetCode: asset code of asset to be extracted, in a list

    Returns
    -------
    X_train_asset : pandas dataframe containing data for only the chosen assetCode
    y_train_asset : pandas dataframe containing label for only the chosen assetCode
    '''
    df_asset = df[df['assetCode'].isin(assetCode)]
    if split:
        y = df_asset['returnsOpenNextMktres10']
        X = df_asset.drop(['returnsOpenNextMktres10'], axis=1)
        return X, y

    return df_asset


if __name__ == '__main__':

    df_market = pd.read_csv(MARKET_CLEAN_PATH)
    df_news = pd.read_csv(NEWS_CLEAN_PATH)

    df_merged = df_market.merge(df_news, 'left', ['time', 'assetCode'])
    df_merged = df_merged.sort_values(['time', 'assetCode'], ascending=[True, True])

    df_merged = extract_stock(df_merged, assetcode_list)
    # taking 80%, 10%, 10% for train, val, test sets
    df_train = df_merged[:522*1990]
    df_val = df_merged[522*1990:522*(1990+249)]
    df_test = df_merged[522*(1990+249):]

    # create the different data sets
    y_train = df_train['returnsOpenNextMktres10']
    X_train = df_train.drop(['returnsOpenNextMktres10'], axis=1)

    y_val = df_val['returnsOpenNextMktres10']
    X_val = df_val.drop(['returnsOpenNextMktres10'], axis=1)

    y_test = df_test['returnsOpenNextMktres10']
    X_test = df_test.drop(['returnsOpenNextMktres10'], axis=1)

    X_train_ar = X_train.drop(['assetCode', "time"], axis=1).as_matrix()
    X_train_ar = X_train_ar.reshape(X_train_ar.shape[0], 1, X_train_ar.shape[1])

    X_val_ar = X_val.drop(['assetCode', "time"], axis=1).as_matrix()
    X_val_ar = X_val_ar.reshape(X_val_ar.shape[0], 1, X_val_ar.shape[1])

    X_test_ar = X_test.drop(['assetCode', "time"], axis=1).as_matrix()
    X_test_ar = X_test_ar.reshape(X_val_ar.shape[0], 1, X_test_ar.shape[1])

    #y_train_ar = y_train.values.reshape((1990, 522))
    #y_val_ar = y_val.values.reshape((int(len(y_val)/522), 522))
    #y_test_ar = y_test.values.reshape((int(len(y_test)/522), 522))

    # 4. Build model from Keras
    N = 300000

    if APPLY_ATTENTION_BEFORE_LSTM:
        m = model_attention_applied_before_lstm()
    else:
        m = model_attention_applied_after_lstm()

    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(m.summary())

    m.fit(X_train_ar, y_train, epochs=3, batch_size=64, validation_data=(X_val_ar, y_val), verbose=1)

    attention_vectors = []
    for i in range(300):
        X_test_ar, y_test = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
        attention_vector = np.mean(get_activations(m,
                                                   X_test_ar,
                                                   print_shape_only=True,
                                                   layer_name='attention_vec')[0], axis=2).squeeze()
        #print('attention =', attention_vector)
        assert (np.sum(attention_vector) - 1.0) < 1e-5
        attention_vectors.append(attention_vector)

    attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
    # plot part.
    import matplotlib.pyplot as plt
    import pandas as pd

    pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                         title='Attention Mechanism as '
                                                                               'a function of input'
                                                                               ' dimensions.')
    plt.show()
