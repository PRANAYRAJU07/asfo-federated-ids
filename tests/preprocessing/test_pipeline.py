import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings
from app.preprocessing.pipeline import MissingValueCleaner, DuplicateRemover, Normalizer, Pipeline, OutlierCleaner, Encoder

def test_missing_value_cleaner():
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, np.nan]})
    cleaner = MissingValueCleaner()
    cleaned = cleaner.fit_transform(df)
    assert len(cleaned) == 1

def test_duplicate_remover():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    remover = DuplicateRemover()
    cleaned = remover.fit_transform(df)
    assert len(cleaned) == 2

@settings(deadline=None)
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=5))
def test_normalizer_hypothesis(data):
    df = pd.DataFrame({"num": data, "other": [1]*len(data)})
    # add some variance to avoid division by zero
    if df["num"].std() == 0:
        df.loc[0, "num"] = df["num"].iloc[0] + 1.0

    normalizer = Normalizer(numerical_cols=["num"])
    normalized = normalizer.fit_transform(df)
    
    assert np.isclose(normalized["num"].mean(), 0, atol=1e-5)
    assert np.isclose(normalized["num"].std(ddof=0), 1, atol=1e-5)

def test_pipeline():
    df = pd.DataFrame({"a": [1, np.nan, 1, 2], "b": [3, 4, 3, 5]})
    pipeline = Pipeline([
        MissingValueCleaner(),
        DuplicateRemover()
    ])
    result = pipeline.run(df)
    assert len(result) == 2

def test_outlier_cleaner():
    df = pd.DataFrame({"a": [1, 100, 2]})
    cleaner = OutlierCleaner()
    cleaned = cleaner.fit_transform(df)
    assert len(cleaned) == 3

def test_encoder():
    df = pd.DataFrame({"cat": ["apple", "banana", "apple"]})
    encoder = Encoder(categorical_cols=["cat"])
    encoded = encoder.fit_transform(df)
    assert list(encoded["cat"]) == [0, 1, 0]
    assert "cat" in encoder.encoders
