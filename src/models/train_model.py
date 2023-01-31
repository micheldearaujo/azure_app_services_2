# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

from src.utils import *
# make the dataset
PERIOD = '800d'
INTERVAL = '1d'

STOCK_NAME = str(input("Which stock do you want to track? "))

logger.info("Starting the training pipeline..")

# download the dataset
make_dataset(STOCK_NAME, PERIOD, INTERVAL)

# load the raw dataset
stock_df = pd.read_csv("./data/raw/raw_stock_prices.csv", parse_dates=['Date'])
stock_df['Date'] = pd.to_datetime(stock_df['Date'])

# perform featurization
stock_df_feat = build_features(stock_df, features_list)

# train test split
X_train, X_test, y_train, y_test = ts_train_test_split(stock_df_feat, model_config["TARGET_NAME"], model_config["FORECAST_HORIZON"])


# Execute the whole pipeline
if __name__ == "__main__":

    predictions_df = validate_model(
        X=pd.concat([X_train, X_test], axis=0),
        y=pd.concat([y_train, y_test], axis=0),
        forecast_horizon=model_config["FORECAST_HORIZON"]
    )

    logger.info("Training Pipeline was sucessful!")
