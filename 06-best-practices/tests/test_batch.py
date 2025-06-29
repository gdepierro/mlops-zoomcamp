# Unit test for batch
import os
import pytest
import pandas as pd
import lambda_function

from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

@pytest.fixture
def set_test_env():
    os.environ["TEST_ENV"] = "1"
    yield
    del os.environ["TEST_ENV"]

def test_batch_processing(set_test_env):

    from batch import read_data, main

    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    prepared_data = lambda_function.prepare_data(df, ['PULocationID', 'DOLocationID'])

    expected_data = [
        ('-1', '-1', 9.0),
        ('1', '1', 8.0),
        ('1', '-1', 0.9833333333333333),
        ('3', '4', 60.016666666666666)
    ]

    assert prepared_data == expected_data

