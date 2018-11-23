import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def EDA():
    '''Prints a brief overview of the data.
    
    Parameters
    ----------
    none

    Returns
    -------
    none
    
    '''
    print(X_train.shape, y_train.shape)
    print(y_train.head())
    print(X_train.head())
    print(X_train.info())
    print(X_train.describe())


def plot_asset(market, assetCode):
    '''Plots an asset's price, volatility and voume.
    
    Parameters
    ----------
    market_df : dataframe
        See https://www.kaggle.com/c/two-sigma-financial-news/data for full description of the dataframe.
    assetCode : string
        The asset code of the instrument you want to plot

    Returns
    -------
    none
    
    '''
    # Set plot style
    plt.style.use('seaborn')
    
    # Fetch the asset from data
    ass_market = market[market['assetCode'] == assetCode]
    ass_market.index = ass_market.time

    # Setup 3 subplots
    f, axs = plt.subplots(2,1, sharex=True, figsize=(12,8))
    
    # Subplot 1. Close price 
    ass_market.close.plot(ax=axs[0], color='black')
    axs[0].set_ylabel("Price")

    # Subplot 2. Volatility 
    volat_df = (ass_market.close - ass_market.open)
    (ass_market.close - ass_market.open).plot(color='darkred', ax = axs[1])
    axs[1].set_ylabel("Volatility")

    # Show all subplots with label
    f.suptitle("Asset: %s" % assetCode, fontsize=22)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()


def plot_chosen_assets():
    '''Prints a group of select stocks, their price and their volatility.
    
    Parameters
    ----------
    none

    Returns
    -------
    none
    
    '''
    # Huge stocks (market cap 200BN - 1000BN)
    #plot_asset(market_train_df, "GOOGL.O") #nonsense data?
    #plot_asset(market_train_df, "AAPL.O") #randomly crashes from 2013-2015?
    #plot_asset(market_train_df, "FB.O") #Facebook: correct, verified, unpredictable volatility
    plot_asset(market_train_df, "INTC.O") #Intel: correct, verified, fair constant volatility
    plot_asset(market_train_df, "WFC.N") #Wells Fargo: correct, verified, crash volatility
    plot_asset(market_train_df, "AMZN.O") #Amazon: correct, verified, increasing volatility
    
    # SMEs (5-20Bn MC)
    #plot_asset(market_train_df, "ADI.N") #Analogue Devices (32Bn MC): kinda correct (one weird correction), verified
    #plot_asset(market_train_df, "NATI.O") #NI (6Bn MC): kinda correct (one weird correction in middle), verified
    plot_asset(market_train_df, "A.N") #Agilent Tech (20Bn MC): kinda correct (one weird correction toward end), verified
        
    # Small stocks (MC < 1Bn)
    #plot_asset(market_train_df, "ANDE.O") #Andersons (900M MC): unverified, high vol
    #plot_asset(market_train_df, "ALO.N") #Alio Gold (90M MC): unverified, low vol
    plot_asset(market_train_df, "BHE.N") #Benchmark Electronics (1Bn MC): verified, low vol

