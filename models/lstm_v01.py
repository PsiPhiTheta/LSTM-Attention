from data_partitioning import validate_df
from data_partitioning import split_fixed_origin
from data_cleaning import get_cleaned_filtered_data, extract_asset

DRY_RUN = True # if True, will only run for one asset with fixed origin strategy

ASSETS = ['INTC.O', 'WFC.N', 'AMZN.O', 'A.N', 'BHE.N']
DATA_PATH = './data/processed/cleaned_filtered_data.csv'

X_train, y_train = get_cleaned_filtered_data(DATA_PATH)


for asset in ASSETS:

	X_train , y_train = extract_asset(X_train, y_train, asset)

	if DRY_RUN:
		break;

