import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain


MARKET_DATA_PATH = './data/raw/market_train_df.csv'
NEWS_DATA_PATH = './data/raw/news_train_df.csv'


def clean_market_data(market_df, train=True):
    '''Clean and preprocess the market data for training or testing.
    
    Parameters
    ----------
    market_df : dataframe
        See https://www.kaggle.com/c/two-sigma-financial-news/data for full 
        description of the dataframe.
    train : bool
        When true, adds the target variable to the dataframe.
    
    Returns
    -------
    dataframe 
        Cleaned market data.
    
    '''
    # Select wanted columns
    if train:
        cols = ['assetCode', 'time', 'volume', 'open', 'returnsOpenPrevMktres1',
                'returnsOpenPrevMktres10', 'returnsOpenNextMktres10']
    else:
        cols = ['assetCode', 'time', 'volume', 'open', 'returnsOpenPrevMktres1',
                'returnsOpenPrevMktres10']
    market_df = market_df.loc[:,cols]

    # Drop NA
    market_df.dropna(inplace=True)

    # Filter out stocks that cover the full time series
    series_len = market_df.time.nunique()
    market_df = market_df.groupby('assetCode')                          .filter(lambda x: len(x) == series_len)
    assert (market_df.groupby('assetCode').size() == series_len).all()
    
    # Normalize time
    market_df.loc[:, 'time'] = pd.to_datetime(market_df.time).dt.normalize()
    
    return market_df



def clean_news_data(news_df):
    '''Clean and preprocess the news data for training or testing.
    
    Parameters
    ----------
    news_df : dataframe
        See https://www.kaggle.com/c/two-sigma-financial-news/data for full 
        description of the dataframe.
    
    Returns
    -------
    dataframe 
        Cleaned news data.
    
    '''
    # Select columns and drop NA
    cols = ['time','assetCodes', 'sentimentNegative', 'sentimentNeutral', 
            'sentimentPositive', 'urgency', 'provider', 'bodySize', 'relevance']
    news_df = news_df.loc[:,cols]
    news_df.dropna(inplace=True)
    
    # Normalize time
    news_df.loc[:, 'time'] = pd.to_datetime(news_df.time).dt.normalize()
    
    # assetCodes from String to List
    news_df['assetCodes'] = news_df['assetCodes'].str.findall(f"'([\w\./]+)'")
    
    # Explode news on assetCodes
    assetCodes_expanded = list(chain(*news_df['assetCodes']))
    assetCodes_index = news_df.index.repeat(news_df['assetCodes'].apply(len))
    assert len(assetCodes_expanded) == len(assetCodes_index)
    
    assetCodes_df =  pd.DataFrame({'index': assetCodes_index, 'assetCode': assetCodes_expanded})
    news_df_exploded = news_df.merge(assetCodes_df, 'right', right_on='index', left_index=True, validate='1:m')
    news_df_exploded.drop(['assetCodes', 'index'], 1, inplace=True)

    # Compute means for same date and assetCode
    news_agg_dict = {
        'sentimentNegative':'mean',
        'sentimentNeutral':'mean',
        'sentimentPositive':'mean',
        'urgency':'mean',
        'bodySize':'mean',
        'relevance':'mean'
        }
    news_df_agg = news_df_exploded.groupby(['time', 'assetCode'], as_index=False).agg(news_agg_dict)
    
    # Add provider information
    idx = news_df_exploded.groupby(['time', 'assetCode'])['urgency'].transform(max) == news_df_exploded['urgency']
    news_df_exploded_2 = news_df_exploded[idx][['time', 'assetCode', 'provider']].drop_duplicates(['time', 'assetCode'])
    news_df_agg = news_df_agg.merge(news_df_exploded_2, 'left', ['time', 'assetCode'])
    
    # One-hot encoding provider
    ohe_provider = pd.get_dummies(news_df_agg['provider'])
    news_df_agg = pd.concat([news_df_agg, ohe_provider], axis=1).drop(['provider'], axis=1)

    return news_df_agg



def clean_data(market_df, news_df, train=True):
    '''Clean and preprocess the news and market data for training then merge 
    them, to create a train set or test set.
    
    Parameters
    ----------
    market_df : dataframe
        See https://www.kaggle.com/c/two-sigma-financial-news/data for full 
        description of the dataframe.
    news_df : dataframe
        See https://www.kaggle.com/c/two-sigma-financial-news/data for full 
        description of the dataframe.
    train : bool
        When true, creates both the input features and the target dataframes.

    Returns
    -------
    dataframe 
        Cleaned data ready to be fed to the model. Returns both the input and
        the target dataframes when train=True.
    
    '''
    cleaned_market_df = clean_market_data(market_df, train)
    cleaned_news_df = clean_news_data(news_df)
    
    # Merge on market data
    df_merged = cleaned_market_df.merge(cleaned_news_df, 'left', ['time', 'assetCode'])
    
    if train:
        y = df_merged['returnsOpenNextMktres10']
        X = df_merged.drop(['returnsOpenNextMktres10'], axis=1)
        return X, y
    else:
        return df_merged


def extract_asset(X_train, y_train, assetCode):
    '''Extracts the training data for a particular asset
    
    Parameters
    ----------
    X_train : dataframe 
        Dataframe containing all the assets' training data.
    y_train : dataframe 
        Dataframe containing all the assets' labels.
    assetCode : String.
        Asset code of asset to be extracted.

    Returns
    -------
    dataframe 
        Dataframe containing data for only the chosen assetCode.
    dataframe 
        Dataframe containing label for only the chosen assetCode
    
    '''
    X_train_asset = X_train[X_train['assetCode']==assetCode]
    y_train_asset = X_train.join(y_train)
    y_train_asset = y_train_asset[y_train_asset['assetCode']==assetCode]
    y_train_asset = y_train_asset.T.tail(1).T
    
    return X_train_asset, y_train_asset


def generate_cleaned_filtered_data(market_data_path, news_data_path, 
                                   save_path, assetCodes):
    ''' Imports the raw data, cleans and filters it and then saves it.

    Parameters
    ----------
    market_data_path : String
        The path to the raw market data.
    news_data_path : String
        The path to the raw news data.
    save_path : String
        The path where to save the cleaned and filtered data.
    asset_Codes : List of Strings
        The asset codes to filter out of the dataset.

    '''
    print('Reading CSV files...')
    market_train_df = pd.read_csv(MARKET_DATA_PATH)
    news_train_df = pd.read_csv(NEWS_DATA_PATH)

    print('Cleaining data...')
    X_train, y_train = clean_data(market_train_df, news_train_df)

    assets = ['INTC.O', 'WFC.N', 'AMZN.O', 'A.N', 'BHE.N']
    print('Extracting assets {}...'.format(asset))
    X_train_asset = X_train[X_train['assetCode'].isin(assetCodes)]
    cleaned_filtered_data = X_train_asset.join(y_train)

    print('Saving cleaned and filtered data to {}.'.format(path))
    cleaned_filtered_data.to_csv(path)
    print('It can now be retrieved using get_cleaned_filtered_data()')


def get_cleaned_filtered_data(path):
    ''' Fetches the data from the CSV file generated by 
    generate_cleaned_filterd_data.
    
    Parameters
    ----------
    path : String
        The path to the cleaned and filtered data.

    Returns
    -------
    dataframe
        Dataframe containing the features (X).
    dataframe
        Dataframe containing the label (y).
    '''

    df = pd.read_csv(path)
    y = df['returnsOpenNextMktres10']
    X = df.drop(['returnsOpenNextMktres10'], axis=1)
    return X, y


if __name__ == '__main__':
    pass