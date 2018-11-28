# Antoine Viscardi's Activity Log

## 27 Nov 2018
- Generated figures for the four different data split strategies (basically learned how to use Inkscpape on the spot)
- Setup dev environment with Tensorflow and Keras
- Started watching Hvass' intro tutorials on Tensorflow and Keras on Youtube.
- Read [good blog post][3] on RNNs and why LSTMs (special form of RNNs) are so popular.
- Implemented functions to save the cleaned and filtered data localy and to retrieve it.



## 26 Nov 2018 (Meeting)
Antoine:
- Figures for data split
- Integrate everything with LSTM
- Figureout error rates (Read the paper, **Up/Down accuracy**)

- First step: use fixed origin and run all the stocks through it.


## 25 Nov 2018
- Refactored the methods to generators. 
- Finished implementing the other 3 strategies. 
- Started briefly investigating other more complicated methods presented by [Bergmeir & Benitez][2].

TODO:
- Determine how to evaluate models (methods for computing error). Should be able to find this information in [Bergmeir & Benitez' paper][2].
- Further investigate other methods of spliting the data.
- Integrate everythin so that we can at least test the LSTM and start generating results and graphs.
- Generate figures explaining each strategy.


## 23 Nov 2018
- Merge all branches on master
- found a [good paper][1] on current state of economic forcasting methods.
- found a [paper][2] dating from 2010 which compares 3 methods. I will implement these first.
- Cleaned data_cleaning.py module
- Finished implementing first spliting strategy: **Last Block Evaluation**.

TODO: 
- Refactor to iterator objects
- Finish implementing 3 other iterators


## 20 Nov 2018 - Meeting with Jonathan Lauraine
- Splitting data is well studied, many way to do.
	- One critical choice is choice of time frame. (Maybe you don't want to train on all the data.)
	- Try bunch of different ones.
	- Look into ways of splitting Train-validate sets. (see picture)
- Dealing with NANs, multiple ways to do this. (average over larger timeframe, fill with what you *think* it could be.)
- Many stocks: One model per stock or on model for all the stocks (hyperparameter again). One asset would be perfectly fine given the scope of the project.
- Comparing to baseline. 
- Be wary of look ahead bias

### Follow up meating with team
- Train model on one stock. (Can swap the stock at the end)
- Decide on stocks we will train/test on (Thomas)
- Implement different functions for splitting training/testing sets and for dealing with NAs. (Antoine)
- Implement Attention + LSTM model (Seung Eun)

- Leave last 10% for testing. Do training/validation on remaining 90%.


## 18 Nov 2018 - Short meeting about Prof. Grosse's email.
- Our main concern is time: already put lots of effort on data processing for financial time series **and literature review**. 
- In his opinion, would switching to waveforms represent an important overhead, or would the data be readily accessible and easy to implement.

- Thomas will do the talking.


## 15 Nov 2018
- Filtered market data to keep only the stocks continuous over the entire time series.
- Small bug where when importing the csv, thi time is a string, easy fix.
- 522 stocks from market data are continuous
- installed matplotlib
- looking at distribution of news data presence to understand better understand the sparsity


## 13 Nov 2018 
- Merged all branches to master, everyone should branch off from there and avoid deleting files
- Created python script for data cleaning (as opposed to doing it inside notebooks).


## 12 Nov 2018 - Meeting
- How do we deal with missing data?
	- Filter stock with only continuous data (market and news)
	- Make sure we only have continuous data (news + market), that is, filter out stock with missing data.
	- If data is very sparse, 
		- Average over a week / months
		- Could engineer a feature with the number of headlines included / sum of the relevances / etc.
		- Don't spend more than 2-3 hours on this.


## 05 Nov 2018
- Look into other kernels exploring the data. Things to consider for futur implementations:
	- Look for data errors: ridiculous spikes. Maybe replace those points with means.
- Current implementation: `dropna(inplace=True)` on both dataframe after slicing desired columns and before merging.
- Still struggling to merge market and news data. 


## 04 Nov 2018
- Started thinking about how to preprocess and clean the data for the Vanilla Net.
- Started implementing the cleaning code.


## 03 Nov 2018
- Familiarized myself with the competition rules, goals, data and framework.


[1](https://pubs.aeaweb.org/doi/pdfplus/10.1257/jep.28.2.3)
[2](https://www.sciencedirect.com/science/article/pii/S0020025511006773)
[3](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

