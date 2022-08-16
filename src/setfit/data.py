from typing import List

import pandas as pd
from datasets import Dataset, DatasetDict


SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SAMPLE_SIZES = [2, 4, 8, 16, 32, 64]


def create_samples(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Samples a DataFrame to create an equal number of samples per class (when possible)."""
    examples = []
    for label in df["label"].unique():
        subset = df.query(f"label == {label}")
        if len(subset) > sample_size:
            examples.append(subset.sample(sample_size, random_state=seed, replace=False))
        else:
            examples.append(subset)
    return pd.concat(examples)


def create_fewshot_splits(dataset: Dataset, sample_sizes: List[int]) -> DatasetDict:
    """Creates training splits from the dataset with an equal number of samples per class (when possible)."""
    splits_ds = DatasetDict()
    df = dataset.to_pandas()
    for sample_size in sample_sizes:
        for idx, seed in enumerate(SEEDS):
            split_df = create_samples(df, sample_size, seed)
            splits_ds[f"train-{sample_size}-{idx}"] = Dataset.from_pandas(split_df, preserve_index=False)
    
    split_ex = list(splits_ds.values())[0]
    print(f"Few shot train split example:")
    print(f"{list(zip(split_ex['text'], split_ex['label']))}")
    return splits_ds
