import gc
from pathlib import Path
from typing import List

import torch
import typer
from datasets import load_dataset
from evaluate import load
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from setfit.data import SAMPLE_SIZES, create_fewshot_splits
from setfit.utils import DEV_DATASET_TO_METRIC, TEST_DATASET_TO_METRIC
from utils import get_label_mappings, save_metrics


app = typer.Typer()

RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


@app.command()
def train_single_dataset(
    model_id: str = "distilbert-base-uncased",
    dataset_id: str = "sst2",
    metric: str = "accuracy",
    sample_sizes: List[int] = SAMPLE_SIZES,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    num_train_epochs_min: int = 25,
    num_train_epochs_max: int = 75,
    push_to_hub: bool = False,
    debug: bool = False,
):
    """Fine-tunes a pretrained checkpoint on the fewshot training sets"""
    # Load dataset
    dataset = load_dataset(f"SetFit/{dataset_id}")
    model_name = model_id.split("/")[-1]

    # Create metrics directory
    metrics_dir = RESULTS_PATH / Path(f"{model_name}-lr-{learning_rate}/{dataset_id}")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    # Load tokenizer and preprocess
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize_dataset(example):
        return tokenizer(example["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_dataset, batched=True)
    # Create fewshot samples
    fewshot_dset = create_fewshot_splits(tokenized_dataset["train"], sample_sizes)
    # Load model - we use a `model_init()` function here to load a fresh model with each fewshot training run
    num_labels, label2id, id2label = get_label_mappings(dataset["train"])

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=num_labels, id2label=id2label, label2id=label2id
        )

    # Define metrics
    metric_fn = load(metric)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return metric_fn.compute(predictions=preds, references=labels)

    for idx, (split, dset) in enumerate(fewshot_dset.items()):
        typer.echo(f"🍌🍌🍌 Fine-tuning on {dataset_id} with split: {split} 🍌🍌🍌")
        # Create split directory
        metrics_split_dir = metrics_dir / split
        metrics_split_dir.mkdir(parents=True, exist_ok=True)
        metrics_filepath = metrics_split_dir / "results.json"
        # Skip previously evaluated split
        if metrics_filepath.is_file():
            typer.echo(f"INFO -- split {split} already trained, skipping ...")
            continue

        if debug:
            if split.split("-")[1] in ["4", "8", "16", "32", "64"]:
                break
            if idx > 0:
                break

        # Create training and validation splits
        dset = dset.train_test_split(seed=42, test_size=0.2)

        # Define hyperparameters
        ckpt_name = f"{model_name}-finetuned-{dataset_id}-{split}"
        training_args = TrainingArguments(
            output_dir="checkpoints/fewshot/",
            overwrite_output_dir=True,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            eval_strategy="epoch",
            logging_steps=100,
            save_strategy="no",
            fp16=True,
            report_to="none",
        )

        if push_to_hub:
            training_args.push_to_hub = True
            training_args.hub_strategy = ("end",)
            training_args.hub_model_id = f"SetFit/{ckpt_name}"

        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=dset["train"],
            eval_dataset=dset["test"],
            tokenizer=tokenizer,
        )

        def hp_space(trial):
            return {
                "num_train_epochs": trial.suggest_int("num_train_epochs", num_train_epochs_min, num_train_epochs_max)
            }

        best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize", hp_space=hp_space)

        for k, v in best_run.hyperparameters.items():
            setattr(trainer.args, k, v)

        trainer.train()

        # Compute final metrics on full test set
        metrics = trainer.evaluate(tokenized_dataset["test"])
        eval_metrics = {}
        eval_metrics["score"] = metrics[f"eval_{metric}"] * 100.0
        eval_metrics["measure"] = metric

        # Save metrics
        save_metrics(eval_metrics, metrics_filepath)

        if push_to_hub:
            trainer.push_to_hub("Checkpoint upload", blocking=False)

        # Flush CUDA cache
        del trainer
        gc.collect()
        torch.cuda.empty_cache()


@app.command()
def train_all_datasets(
    model_id: str = "distilbert-base-uncased",
    sample_sizes: List[int] = SAMPLE_SIZES,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    push_to_hub: bool = False,
    num_train_epochs_min: int = 25,
    num_train_epochs_max: int = 75,
    is_dev_set: bool = False,
):
    """Fine-tunes a pretrained checkpoint on all of the SetFit development/test datasets."""
    if is_dev_set:
        DATASET_TO_METRIC = DEV_DATASET_TO_METRIC
    else:
        DATASET_TO_METRIC = TEST_DATASET_TO_METRIC

    for dataset_id, metric in DATASET_TO_METRIC.items():
        typer.echo(f"🏋️🏋️🏋️  Fine-tuning on dataset {dataset_id} 🏋️🏋️🏋️")
        train_single_dataset(
            model_id=model_id,
            dataset_id=dataset_id,
            metric=metric,
            sample_sizes=sample_sizes,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_train_epochs_min=num_train_epochs_min,
            num_train_epochs_max=num_train_epochs_max,
            push_to_hub=push_to_hub,
        )
    typer.echo("Training complete!")


if __name__ == "__main__":
    app()
