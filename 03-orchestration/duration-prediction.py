#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    print(f"Original Shape {df.shape}")

    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    print(f"After Transform Shape {df.shape}")

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    # df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df


def create_X(df, dv=None):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_xgb(X_train, y_train, X_val, y_val, dv):
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    best_params = {
        'learning_rate': 0.09585355369315604,
        'max_depth': 30,
        'min_child_weight': 1.060597050922164,
        'objective': 'reg:squarederror',
        'reg_alpha': 0.018060244040060163,
        'reg_lambda': 0.011658731377413597,
        'seed': 42
    }

    booster = xgb.train(
        params=best_params,
        dtrain=train,
        num_boost_round=30,
        evals=[(valid, 'validation')],
        early_stopping_rounds=50
    )

    mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

    return booster, dv, best_params, valid

def train_linear_regression(X_train, y_train, dv):

    model = LinearRegression()
    model.fit(X_train, y_train)

    # print(f"Linear Regression Coefficients: {model.coef_}")
    print(f"Linear Regression Intercept: {model.intercept_}")

    mlflow.sklearn.log_model(model, artifact_path="models_mlflow")

    return model, dv

def train_model(X_train, y_train, X_val, y_val, dv, model_type='linear_regression'):
    with mlflow.start_run() as run:
        if model_type == 'xgboost':
            model, dv, best_params, X_val = train_xgb(X_train, y_train, X_val, y_val, dv)
            mlflow.log_params(best_params)
        elif model_type == 'linear_regression':
            model, dv = train_linear_regression(X_train, y_train, dv)

        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        return run.info.run_id

def run(year, month, model_type):
    df_train = read_dataframe(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv, model_type=model_type)
    print(f"MLflow run_id: {run_id}")
    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    parser.add_argument('--model_type', type=str, required=False, help='Type of model to train (xgboost or linear_regression)', default='xgboost')
    args = parser.parse_args()
    # If model type is not specified, default to linear regression
    if args.model_type is None:
        args.model_type = 'linear_regression'

    run_id = run(year=args.year, month=args.month, model_type=args.model_type)

    with open("run_id.txt", "w") as f:
        f.write(run_id)