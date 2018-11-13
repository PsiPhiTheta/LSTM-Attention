# Antoine Viscardi's Activity Log

## 03 Nov 2018
- Familiarized myself with the competition rules, goals, data and framework.

## 04 Nov 2018
- Started thinking about how to preprocess and clean the data for the Vanilla Net.
- Started implementing the cleaning code.

## 05 Nov 2018
- Look into other kernels exploring the data. Things to consider for futur implementations:
	- Look for data errors: ridiculous spikes. Maybe replace those points with means.
- Current implementation: `dropna(inplace=True)` on both dataframe after slicing desired columns and before merging.
- Still struggling to merge market and news data. 

## 12 Nov 2018 - Meeting
- How do we deal with missing data?
	- Filter stock with only continuous data (market and news)
	- Make sure we only have continuous data (news + market), that is, filter out stock with missing data.
	- If data is very sparse, 
		- Average over a week / months
		- Could engineer a feature with the number of headlines included / sum of the relevances / etc.
		- Don't spend more than 2-3 hours on this.

## 13 Nov 2018 
- Merged all branches to master, everyone should branch off from there and avoid deleting files
- Created python script for data cleaning (as opposed to doing it inside notebooks).
