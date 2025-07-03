import pytest
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np

def test_scaler_instantiation():
    """Test that StandardScaler can be instantiated."""
    scaler = StandardScaler()
    assert isinstance(scaler, StandardScaler)

def test_isolation_forest_instantiation():
    """Test that IsolationForest can be instantiated."""
    model = IsolationForest(random_state=42)
    assert isinstance(model, IsolationForest)

def test_one_class_svm_instantiation():
    """Test that OneClassSVM can be instantiated."""
    model = OneClassSVM(nu=0.1)
    assert isinstance(model, OneClassSVM)

def test_model_fit_predict_basic():
    """Test that models can be fit and make predictions on dummy data."""
    X_dummy = np.array([[1, 2], [1.1, 2.2], [10, 20], [1.3, 2.1]])

    # Isolation Forest
    iso_forest = IsolationForest(random_state=42)
    iso_forest.fit(X_dummy)
    predictions_iso = iso_forest.predict(X_dummy)
    assert predictions_iso.shape == (X_dummy.shape[0],), "Isolation Forest predictions shape mismatch"
    assert set(predictions_iso).issubset({-1, 1}), "Isolation Forest predictions should be -1 or 1"

    # One-Class SVM
    oc_svm = OneClassSVM(nu=0.2) # nu must be > 0 and <= 1
    oc_svm.fit(X_dummy)
    predictions_ocsvm = oc_svm.predict(X_dummy)
    assert predictions_ocsvm.shape == (X_dummy.shape[0],), "One-Class SVM predictions shape mismatch"
    assert set(predictions_ocsvm).issubset({-1, 1}), "One-Class SVM predictions should be -1 or 1"