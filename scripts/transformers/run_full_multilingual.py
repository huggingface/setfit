import gc
from pathlib import Path

import torch
import typer
from datasets import concatenate_datasets, load_dataset
from evaluate import load
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from setfit.utils import MULTILINGUAL_DATASET_TO_METRIC
from utils import get_label_mappings, save_metrics


app = typer.Typer()


RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


@app.command()
def train_single_dataset(
    model_id: str = "xlm-roberta-base",
    dataset_id: str = "amazon_reviews_en",
    metric: str = "mae",
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    push_to_hub: bool = False,
    multilinguality: str = "each",
):
    """Fine-tunes a pretrained checkpoint on the fewshot training sets"""
    # Load tokenizer and preprocess
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize_dataset(example):
        return tokenizer(example["text"], truncation=True, max_length=512)

    # Load dataset
    if multilinguality == "each":
        train_dataset = load_dataset(f"SetFit/{dataset_id}", split="train")
        tokenized_dataset = train_dataset.map(tokenize_dataset, batched=True)
    elif multilinguality == "en":
        # Load English dataset
        english_dataset = [dset for dset in MULTILINGUAL_DATASET_TO_METRIC.keys() if dset.endswith("_en")][0]
        train_dataset = load_dataset(f"SetFit/{english_dataset}", split="train")
        tokenized_dataset = train_dataset.map(tokenize_dataset, batched=True)
    elif multilinguality == "all":
        # Concatenate all languages
        dsets = []
        for dataset in MULTILINGUAL_DATASET_TO_METRIC.keys():
            ds = load_dataset(f"SetFit/{dataset}", split="train")
            dsets.append(ds)
        # Create training set and sample for fewshot splits
        train_dataset = concatenate_datasets(dsets).shuffle(seed=42)
        tokenized_dataset = train_dataset.map(tokenize_dataset, batched=True)

    # Create training and validation splits
    train_eval_dataset = tokenized_dataset.train_test_split(seed=42, test_size=0.2)
    test_dataset = load_dataset(f"SetFit/{dataset_id}", split="test")
    tokenized_test_dataset = test_dataset.map(tokenize_dataset, batched=True)

    model_name = model_id.split("/")[-1]

    # Create metrics directory
    metrics_dir = RESULTS_PATH / Path(f"{model_name}-lr-{learning_rate}/{dataset_id}/{multilinguality}")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    # Create split directory
    metrics_split_dir = metrics_dir / "train-full"
    metrics_split_dir.mkdir(parents=True, exist_ok=True)
    metrics_filepath = metrics_split_dir / "results.json"
    # Skip previously evaluated model
    if metrics_filepath.is_file():
        typer.echo("INFO -- model already trained, skipping ...")
        return

    # Load model - we use a `model_init()` function here to load a fresh model with each fewshot training run
    num_labels, label2id, id2label = get_label_mappings(train_dataset)

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
        output_dir=f"checkpoints/full/{multilinguality}",
        overwrite_output_dir=True,
        num_train_epochs=20,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_steps=100,
        metric_for_best_model="eval_loss",
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
    metrics = trainer.evaluate(tokenized_test_dataset)
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
    model_id: str = "xlm-roberta-base",
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    push_to_hub: bool = False,
    multilinguality: str = "each",
):
    """Fine-tunes a pretrained checkpoint on all of the SetFit development/test datasets."""
    for dataset_id, metric in MULTILINGUAL_DATASET_TO_METRIC.items():
        typer.echo(f"🏋️🏋️🏋️  Fine-tuning on dataset {dataset_id} 🏋️🏋️🏋️")
        train_single_dataset(
            model_id=model_id,
            dataset_id=dataset_id,
            metric=metric,
            learning_rate=learning_rate,
            batch_size=batch_size,
            push_to_hub=push_to_hub,
            multilinguality=multilinguality,
        )
    typer.echo("Training complete!")


if __name__ == "__main__":
    app()
