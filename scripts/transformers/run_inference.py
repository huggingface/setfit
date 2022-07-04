from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import typer
from datasets import Dataset, load_dataset
from transformers import pipeline


RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

device_id = 0 if torch.cuda.is_available() else -1


def time_pipeline(pipe: pipeline, dataset: Dataset):
    latencies = []
    # Warm up
    for _ in range(10):
        _ = pipe("Warming up the pipeline :)")
    # Timed run
    total_start_time = perf_counter()
    for row in dataset:
        start_time = perf_counter()
        _ = pipe(row["text"])
        latency = perf_counter() - start_time
        latencies.append(latency)
    total_time_ms = (perf_counter() - total_start_time) * 1_000
    # Compute run statistics
    time_avg_ms = 1_000 * np.mean(latencies)
    time_std_ms = 1_000 * np.std(latencies)
    time_p95_ms = 1_000 * np.percentile(latencies, 95)
    print(
        f"P95 latency (ms) - {time_p95_ms}; Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f};",  # noqa
        time_p95_ms,
        f"Total time (ms) - {total_time_ms:.2f}",
    )


def main(
    model_id: str = "distilbert-base-uncased__sst2__train-16-4", dataset_id: str = "sst2", num_samples: int = None
):
    # Load dataset
    dataset = load_dataset(f"SetFit/{dataset_id}", split="test")
    if num_samples is not None:
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    # Load pipeline
    pipe = pipeline("text-classification", model=f"SetFit/{model_id}", device=device_id)
    # Time it!
    time_pipeline(pipe, dataset)


if __name__ == "__main__":
    typer.run(main)
