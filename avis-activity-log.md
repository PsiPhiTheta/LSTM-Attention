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