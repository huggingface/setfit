import json
from pathlib import Path
from typing import Dict, List, Tuple

import typer
from datasets import Dataset, load_dataset
from evaluate import load
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from setfit.data import create_fewshot_splits


app = typer.Typer()

DEV_DATASET_TO_METRIC = {
    "sst2": "accuracy",
    "imdb": "accuracy",
    "subj": "accuracy",
    "ag_news": "accuracy",
    "bbc-news": "accuracy",
    "enron_spam": "accuracy",
    "student-question-categories": "accuracy",
    "TREC-QC": "accuracy",
    "toxic_conversations": "matthews_correlation",
}

TEST_DATASET_TO_METRIC = {
    "emotion": "accuracy",
    "SentEval-CR": "accuracy",
    "sst5": "accuracy",
    "amazon_counterfactual_en": "matthews_correlation",
}

RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def get_label_mappings(dataset: Dataset) -> Tuple[int, dict, dict]:
    """Returns the label mappings of the dataset."""
    label_ids = dataset.unique("label")
    label_names = dataset.unique("label_text")
    label2id = {label: idx for label, idx in zip(label_names, label_ids)}
    id2label = {idx: label for label, idx in label2id.items()}
    num_labels = len(label_ids)
    return num_labels, label2id, id2label


def save_metrics(metrics: dict, metrics_filepath):
    with open(metrics_filepath, "w") as f:
        json.dump(metrics, f)


@app.command()
def train_single_dataset(
    dataset_name: str,
    metric_name: str,
    model_ckpt: str = "distilbert-base-uncased",
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    num_train_epochs_min: int = 25,
    num_train_epochs_max: int = 75,
    push_to_hub: bool = False,
    debug: bool = False,
) -> List[Dict[str, float]]:
    """Fine-tunes a pretrained checkpoint on the fewshot training sets"""
    # Load dataset
    dataset_id = "SetFit/" + dataset_name
    dataset = load_dataset(dataset_id)
    model_name = model_ckpt.split("/")[-1]

    # Create metrics directory
    metrics_dir = RESULTS_PATH / Path(f"{model_name}-lr-{learning_rate}/{dataset_name}")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    # Load tokenizer and preprocess
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def tokenize_dataset(example):
        return tokenizer(example["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_dataset, batched=True)
    # Create fewshot samples
    fewshot_dset = create_fewshot_splits(tokenized_dataset["train"])
    # Load model - we use a `model_init()` function here to load a fresh model with each fewshot training run
    num_labels, label2id, id2label = get_label_mappings(dataset["train"])

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            model_ckpt, num_labels=num_labels, id2label=id2label, label2id=label2id
        )

    # Define metrics
    metric_fn = load(metric_name)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return metric_fn.compute(predictions=preds, references=labels)

    for idx, (split, dset) in enumerate(fewshot_dset.items()):
        typer.echo(f"🍌🍌🍌 Fine-tuning on {dataset_name} with split: {split} 🍌🍌🍌")
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
        ckpt_name = f"{model_name}__{dataset_name}__{split}"
        training_args = TrainingArguments(
            output_dir="checkpoints/fewshot/",
            overwrite_output_dir=True,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            evaluation_strategy="epoch",
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
            setattr(training_args, k, v)

        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=dset["train"],
            eval_dataset=dset["test"],
            tokenizer=tokenizer,
        )

        trainer.train()
        # Compute final metrics on full test set
        metrics = trainer.evaluate(tokenized_dataset["test"])
        eval_metrics = {}
        eval_metrics["score"] = metrics[f"eval_{metric_name}"]
        eval_metrics["measure"] = metric_name
        print(metrics)

        # Save metrics
        save_metrics(eval_metrics, metrics_filepath)

        if push_to_hub:
            trainer.push_to_hub("Checkpoint upload", blocking=False)


@app.command()
def train_all_datasets(
    model_ckpt: str = "distilbert-base-uncased",
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    push_to_hub: bool = False,
    num_train_epochs_min: int = 25,
    num_train_epochs_max: int = 75,
    train_mode: str = "dev",
):
    """Fine-tunes a pretrained checkpoint on all of the SetFit development/test datasets."""
    if train_mode == "dev":
        DATASET_TO_METRIC = DEV_DATASET_TO_METRIC
    else:
        DATASET_TO_METRIC = TEST_DATASET_TO_METRIC

    for dataset_name, metric_name in DATASET_TO_METRIC.items():
        typer.echo(f"🏋️🏋️🏋️  Fine-tuning on dataset {dataset_name} 🏋️🏋️🏋️")
        train_single_dataset(
            dataset_name=dataset_name,
            metric_name=metric_name,
            model_ckpt=model_ckpt,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_train_epochs_min=num_train_epochs_min,
            num_train_epochs_max=num_train_epochs_max,
            push_to_hub=push_to_hub,
        )
    typer.echo("Training complete!")


if __name__ == "__main__":
    app()
