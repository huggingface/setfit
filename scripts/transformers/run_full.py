import gc
from pathlib import Path

import torch
import typer
from datasets import load_dataset
from evaluate import load
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

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
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    num_train_epochs: int = 20,
    push_to_hub: bool = False,
):
    """Fine-tunes a pretrained checkpoint on the fewshot training sets"""
    # Load dataset
    dataset = load_dataset(f"SetFit/{dataset_id}")
    model_name = model_id.split("/")[-1]

    # Create metrics directory
    metrics_dir = RESULTS_PATH / Path(f"{model_name}-lr-{learning_rate}/{dataset_id}")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    # Create split directory
    metrics_split_dir = metrics_dir / "train-full"
    metrics_split_dir.mkdir(parents=True, exist_ok=True)
    metrics_filepath = metrics_split_dir / "results.json"
    # Skip previously evaluated model
    if metrics_filepath.is_file():
        typer.echo("INFO -- model already trained, skipping ...")
        return

    # Load tokenizer and preprocess
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize_dataset(example):
        return tokenizer(example["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_dataset, batched=True)
    # Create training and validation splits
    train_eval_dataset = tokenized_dataset["train"].train_test_split(seed=42, test_size=0.2)
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

    # Define hyperparameters
    training_args = TrainingArguments(
        output_dir="checkpoints/full/",
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.001,
        evaluation_strategy="epoch",
        logging_steps=100,
        metric_for_best_model=metric,
        load_best_model_at_end=True,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        report_to="none",
    )

    if push_to_hub:
        ckpt_name = f"{model_name}-finetuned-{dataset_id}-train-full"
        training_args.push_to_hub = True
        training_args.hub_strategy = ("end",)
        training_args.hub_model_id = f"SetFit/{ckpt_name}"

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_eval_dataset["train"],
        eval_dataset=train_eval_dataset["test"],
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
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
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    num_train_epochs: int = 20,
    push_to_hub: bool = False,
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
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            push_to_hub=push_to_hub,
        )
    typer.echo("Training complete!")


if __name__ == "__main__":
    app()
