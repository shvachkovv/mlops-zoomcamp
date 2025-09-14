#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import pyarrow
import pickle
import pandas as pd


#get_ipython().system('pip freeze | grep scikit-learn')
#get_ipython().system('python -V')


def main(year: int, month: int):

    print(f"Running pipeline for {year:04d}-{month:02d}")
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    def read_data(filename):
        df = pd.read_parquet(filename)

        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

        return df

    df = read_data(f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet")


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print("Standard deviation of predicted durations:", np.std(y_pred))
    print("Mean predicted duration:", np.mean(y_pred))

    year = 2023
    month = 3
    output_file = f"predictions_{year:04d}_{month:02d}.parquet"

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_predictions = pd.DataFrame({
        "ride_id": df['ride_id'],
        "predicted_duration": y_pred
    })

    df_predictions.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Taxi duration prediction")
    parser.add_argument("--year", type=int, required=True, help="Year of dataset (e.g. 2023)")
    parser.add_argument("--month", type=int, required=True, help="Month of dataset (e.g. 3)")
    args = parser.parse_args()

    main(args.year, args.month)