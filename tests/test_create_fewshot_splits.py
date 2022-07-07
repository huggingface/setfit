import pandas as pd
from datasets import Dataset

from setfit.data import SAMPLE_SIZES, SEEDS, create_fewshot_splits


def test_expected_number_of_splits():
    dataset = Dataset.from_pandas(pd.DataFrame({"label": [0] * 50 + [1] * 50}))
    splits_ds = create_fewshot_splits(dataset, SAMPLE_SIZES)
    assert len(splits_ds) == len(SAMPLE_SIZES) * len(SEEDS)
