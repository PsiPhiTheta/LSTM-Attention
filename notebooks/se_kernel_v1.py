import sys
import os
import pandas as pd
import glob

sys.path.append('../')
from models.data_cleaning import clean_market_data, clean_news_data

# Import libraries used for lstm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Define some global variables
MARKET_DATA_PATH = '../data/raw/market_train_df.csv'
NEWS_DATA_PATH = '../data/raw/news_train_df.csv'
MERGED_PATH = '../data/processed/df_merged.csv'

MARKET_CLEAN_PATH = '../data/processed/market_cleaned_df.csv'
NEWS_CLEAN_CHUNK_PATH = '../data/processed/news_cleaned_df_'
NEWS_CLEAN_PATH = '../data/processed/news_cleaned_df.csv'

MARKET_CONTINOUS_PATH = '../data/processed/market_continuous_df.csv'
NEWS_CONTINUOUS_PATH = '../data/processed/news_continuous_df.csv'


def get_continuous_df(market_data_path, news_data_path, merged_path,
                      market_clean_path=MARKET_CLEAN_PATH,
                      news_clean_chunk_path=NEWS_CLEAN_CHUNK_PATH,
                      news_clean_path=NEWS_CLEAN_PATH,
                      market_continuous_path=MARKET_CONTINOUS_PATH,
                      news_continuous_path=NEWS_CONTINUOUS_PATH):
    """
    Cleans and filters the datasets to only select assets with
    continuous information
    """
    market_train_df = pd.read_csv(market_data_path)
    cleaned_market_df = clean_market_data(market_train_df)
    print('market data was cleaned')
    cleaned_market_df.to_csv(market_clean_path)
    print('cleaned market data was saved')
    # save memory usage
    del market_train_df

    series_len = cleaned_market_df.time.nunique()
    cleaned_market_df = cleaned_market_df.groupby('assetCode').filter(lambda x: len(x) == series_len)
    cleaned_market_df = cleaned_market_df.reset_index(drop=True)
    print('market data was filtered')
    cleaned_market_df.to_csv(market_continuous_path)
    print('filtered market data was saved')

    c = 0
    for news_chunk in pd.read_csv(news_data_path, chunksize=100000):
        print('news chunk_number ' + str(c))
        news_cleaned = clean_news_data(news_chunk)
        news_cleaned.to_csv(news_clean_chunk_path + str(c) + '.csv')
        print('news chunk number ' + str(c) + ' saved')
        c += 1

    news_files = glob.glob(news_clean_chunk_path + "*")
    cleaned_news_df = pd.concat((pd.read_csv(f, header=0) for f in news_files))
    print('cleaned news data concatenated')
    cleaned_news_df.to_csv(news_clean_path)
    print('cleaned news data was saved')

    assetcodes = cleaned_market_df['assetCode'].tolist()
    news_continuous_df = cleaned_news_df[cleaned_news_df['assetCode'].isin(assetcodes)]
    news_continuous_df.loc[:, 'time'] = pd.to_datetime(news_continuous_df.time).dt.normalize()
    news_continuous_df.to_csv(news_continuous_path)
    print('filtered news data was saved')
    df_merged = cleaned_market_df.merge(news_continuous_df.drop_duplicates(subset=['time', 'assetCode']), 'left', ['time', 'assetCode'])

    print('filling missing values and saving the merged dataset')
    df_merged = df_merged.fillna(-1)
    df_merged.to_csv(merged_path)

    # return the final merged dataset
    return df_merged


if __name__ == '__main__':

    if os.path.exists(MERGED_PATH):
        df_merged = pd.read_csv(MERGED_PATH)
    else:
        df_merged = get_continuous_df(MARKET_DATA_PATH,
                                      NEWS_DATA_PATH,
                                      MERGED_PATH)

    df_merged = df_merged.sort_values(['time', 'assetCode'], ascending=[True, True])

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

    X_train_ar = X_train.drop(['Unnamed: 0', 'assetCode', "time"], axis=1).as_matrix()
    X_train_ar = X_train_ar.reshape(int(X_train_ar.shape[0]/522), 1, 522*X_train_ar.shape[1])

    X_val_ar = X_val.drop(['Unnamed: 0', 'assetCode', "time"], axis=1).as_matrix()
    X_val_ar = X_val_ar.reshape(int(X_val_ar.shape[0]/522), 1, 522*X_val_ar.shape[1])

    X_test_ar = X_test.drop(['Unnamed: 0', 'assetCode', "time"], axis=1).as_matrix()
    X_test_ar = X_test_ar.reshape(int(X_test_ar.shape[0]/522), 1, 522*X_test_ar.shape[1])

    y_train_ar = y_train.values.reshape((1990, 522))
    y_val_ar = y_val.values.reshape((int(len(y_val)/522), 522))
    y_test_ar = y_test.values.reshape((int(len(y_test)/522), 522))

    # 4. Build Keras model
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, 41*522)))  # adds LSTM layer
    model.add(Dense(522))  # adds a dense layer
    model.compile(loss='mae', optimizer='adam')  # TODO: change the loss

    # 5. Fit RNN
    model.fit(X_train_ar, y_train_ar, epochs=3, batch_size=1,
              validation_data=(X_val_ar, y_val_ar), verbose=1, shuffle=False)

    model.save('vanilla_lstm_20181117.hdf5')
    print('model saved.')
