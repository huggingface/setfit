import pandas as pd
import pytest
from setfit.data import create_samples


@pytest.mark.parametrize("sample_size", [8, 16, 32])
def test_subset_is_larger_than_sample_size(sample_size):
    data = {"label": [0] * 50 + [1] * 50}
    df = pd.DataFrame(data)
    sample_df = create_samples(df, sample_size=sample_size, seed=0)
    assert len(sample_df) == (sample_size * 2)


@pytest.mark.parametrize("sample_size", [8, 16, 32])
def test_subset_is_smaller_than_sample_size(sample_size):
    data = {"label": [0] * 3 + [1] * 3}
    df = pd.DataFrame(data)
    sample_df = create_samples(df, sample_size=sample_size, seed=0)
    assert len(sample_df) == len(df)
