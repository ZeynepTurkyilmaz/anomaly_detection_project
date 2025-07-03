import numpy as np
import pandas as pd
import pytest
import os

# Assume your data path is relative to the project root
# For tests, it's often easier to define absolute paths or mock data.
# Here, we'll assume tests are run from the project root.

DATA_PATH = "data/simulated_sensor_data.csv"

def test_data_file_exists():
    """Verify that the simulated sensor data file exists."""
    assert os.path.exists(DATA_PATH), f"Data file not found at {DATA_PATH}"

def test_data_loading_and_columns():
    """Verify data loads correctly and has expected columns."""
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        pytest.fail(f"Could not load data: {e}")

    expected_columns = ['temperature', 'pressure', 'anomaly', 'timestamp']
    assert not df.empty, "DataFrame should not be empty"
    for col in expected_columns:
        assert col in df.columns, f"Missing expected column: {col}"

def test_data_types():
    """Verify data types of key columns."""
    df = pd.read_csv(DATA_PATH)
    assert pd.api.types.is_numeric_dtype(df['temperature']), "Temperature column should be numeric"
    assert pd.api.types.is_numeric_dtype(df['pressure']), "Pressure column should be numeric"
    assert pd.api.types.is_numeric_dtype(df['anomaly']), "Anomaly column should be numeric"
    # Note: timestamp is loaded as object by default unless parsed during read_csv
    # If you explicitly convert it in your pipeline, you might test for datetime.
    # assert pd.api.types.is_datetime64_any_dtype(df['timestamp']), "Timestamp column should be datetime"

def test_anomaly_values():
    """Verify anomaly column contains only 0s and 1s."""
    df = pd.read_csv(DATA_PATH)
    unique_anomaly_values = df['anomaly'].unique()
    assert np.all(np.isin(unique_anomaly_values, [0.0, 1.0])), "Anomaly column should only contain 0.0 or 1.0"

# You could add tests for data range, absence of NaNs, etc.