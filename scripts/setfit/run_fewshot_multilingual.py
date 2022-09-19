import argparse
import copy
import json
import math
import os
import sys
from shutil import copyfile
from warnings import simplefilter

import numpy as np
from datasets import concatenate_datasets, load_dataset
from evaluate import load
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from setfit.data import SAMPLE_SIZES, create_fewshot_splits
from setfit.modeling import LOSS_NAME_TO_CLASS, SKLearnWrapper, SupConLoss, sentence_pairs_generation
from setfit.utils import MULTILINGUAL_DATASET_TO_METRIC


# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
parser.add_argument(
    "--datasets",
    nargs="+",
    default=None,
)
parser.add_argument("--sample_sizes", type=int, nargs="+", default=SAMPLE_SIZES)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_seq_length", type=int, default=256)
parser.add_argument(
    "--classifier",
    default="logistic_regression",
    choices=[
        "logistic_regression",
        "svc-rbf",
        "svc-rbf-norm",
        "knn",
        "pytorch",
        "pytorch_complex",
    ],
)
parser.add_argument("--loss", default="CosineSimilarityLoss")
parser.add_argument("--exp_name", default="")
parser.add_argument("--add_normalization_layer", default=False, action="store_true")
parser.add_argument("--optimizer_name", default="AdamW")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--multilinguality", type=str, default="each", choices=["each", "en", "all"])
args = parser.parse_args()

# Prepare directory for results
output_path = f"results/{args.model.replace('/', '-')}-{args.loss}-{args.classifier}-epochs_{args.num_epochs}-batch_{args.batch_size}-{args.exp_name}".rstrip(
    "-"
)
os.makedirs(output_path, exist_ok=True)

train_script_path = os.path.join(output_path, "train_script.py")

copyfile(__file__, train_script_path)

with open(train_script_path, "a") as f_out:
    f_out.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))


def compute_metrics(x_train, y_train, x_test, y_test, metric):
    """Computes the metrics for a given classifier."""
    # Define metrics
    metric_fn = load(metric)

    clf = get_classifier(model)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    metrics = metric_fn.compute(predictions=y_pred, references=y_test)
    print(f"{metric} -- {metrics[metric]:.4f}")

    return metrics


def get_classifier(sbert_model):
    if args.classifier == "logistic_regression":
        return SKLearnWrapper(sbert_model, LogisticRegression())


def eval_setfit(train_data, test_data, model, loss_class, num_epochs, metric):
    x_train = train_data["text"]
    y_train = train_data["label"]

    x_test = test_data["text"]
    y_test = test_data["label"]

    if loss_class is None:
        return compute_metrics(x_train, y_train, x_test, y_test, metric)

    # sentence-transformers adaptation
    batch_size = args.batch_size
    if loss_class in [
        losses.BatchAllTripletLoss,
        losses.BatchHardTripletLoss,
        losses.BatchSemiHardTripletLoss,
        losses.BatchHardSoftMarginTripletLoss,
        SupConLoss,
    ]:

        train_examples = [InputExample(texts=[text], label=label) for text, label in zip(x_train, y_train)]
        train_data_sampler = SentenceLabelDataset(train_examples)

        batch_size = min(args.batch_size, len(train_data_sampler))
        train_dataloader = DataLoader(train_data_sampler, batch_size=batch_size, drop_last=True)

        if loss_class is losses.BatchHardSoftMarginTripletLoss:
            train_loss = loss_class(
                model=model,
                distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance,
            )
        elif loss_class is SupConLoss:
            train_loss = loss_class(model=model)
        else:
            train_loss = loss_class(
                model=model,
                distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance,
                margin=0.25,
            )

        train_steps = len(train_dataloader) * num_epochs
    else:
        train_examples = []
        for _ in range(num_epochs):
            train_examples = sentence_pairs_generation(np.array(x_train), np.array(y_train), train_examples)

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = loss_class(model)
        train_steps = len(train_dataloader)

    print(f"{len(x_train)} train samples in total, {train_steps} train steps with batch size {batch_size}")

    warmup_steps = math.ceil(train_steps * 0.1)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        steps_per_epoch=train_steps,
        warmup_steps=warmup_steps,
        show_progress_bar=False,
    )

    return compute_metrics(x_train, y_train, x_test, y_test, metric)


# Configure dataset <> metric mapping. Defaults to accuracy
if args.datasets is not None:
    DATASET_TO_METRIC = {dataset: "accuracy" for dataset in args.datasets}
else:
    DATASET_TO_METRIC = MULTILINGUAL_DATASET_TO_METRIC

# Configure loss function
loss_class = LOSS_NAME_TO_CLASS[args.loss]

# Load model
model = SentenceTransformer(args.model)
model_original_state = copy.deepcopy(model.state_dict())
model.max_seq_length = args.max_seq_length

if args.add_normalization_layer:
    model._modules["2"] = models.Normalize()

# Train on language X and evaluate on language X
if args.multilinguality == "each":
    for dataset, metric in DATASET_TO_METRIC.items():
        print(f"\n\n\n============== {dataset} (each) ============")
        train_ds = load_dataset(f"SetFit/{dataset}", split="train")
        fewshot_ds = create_fewshot_splits(train_ds, args.sample_sizes)
        test_dataset = load_dataset(f"SetFit/{dataset}", split="test")

        for name in fewshot_ds:
            results_path = os.path.join(output_path, dataset, "each", name, "results.json")
            print(f"\n\n======== {os.path.dirname(results_path)} =======")
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            if os.path.exists(results_path):
                continue

            model.load_state_dict(copy.deepcopy(model_original_state))
            metrics = eval_setfit(
                fewshot_ds[name],
                test_dataset,
                model,
                loss_class,
                args.num_epochs,
                metric,
            )

            with open(results_path, "w") as f_out:
                json.dump(
                    {"score": metrics[metric] * 100, "measure": metric},
                    f_out,
                    sort_keys=True,
                )

# Train on English and evaluate on language X
if args.multilinguality == "en":
    for dataset, metric in DATASET_TO_METRIC.items():
        print(f"\n\n\n============== {dataset} (en) ============")
        english_dataset = [dset for dset in DATASET_TO_METRIC.keys() if dset.endswith("_en")][0]
        train_ds = load_dataset(f"SetFit/{english_dataset}", split="train")
        fewshot_ds = create_fewshot_splits(train_ds, args.sample_sizes)
        test_dataset = load_dataset(f"SetFit/{dataset}", split="test")

        for name in fewshot_ds:
            results_path = os.path.join(output_path, dataset, "en", name, "results.json")
            print(f"\n\n======== {os.path.dirname(results_path)} =======")
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            if os.path.exists(results_path):
                continue

            model.load_state_dict(copy.deepcopy(model_original_state))
            metrics = eval_setfit(
                fewshot_ds[name],
                test_dataset,
                model,
                loss_class,
                args.num_epochs,
                metric,
            )

            with open(results_path, "w") as f_out:
                json.dump(
                    {"score": metrics[metric] * 100, "measure": metric},
                    f_out,
                    sort_keys=True,
                )

# Train on all languages and evaluate on language X
if args.multilinguality == "all":
    # Concatenate all languages
    dsets = []
    for dataset in DATASET_TO_METRIC.keys():
        ds = load_dataset(f"SetFit/{dataset}", split="train")
        dsets.append(ds)
    # Create training set and sample for fewshot splits
    train_ds = concatenate_datasets(dsets).shuffle(seed=42)
    fewshot_ds = create_fewshot_splits(train_ds, args.sample_sizes)

    for dataset, metric in DATASET_TO_METRIC.items():
        test_dataset = load_dataset(f"SetFit/{dataset}", split="test")
        for name in fewshot_ds:
            results_path = os.path.join(output_path, dataset, "all", name, "results.json")
            print(f"\n\n======== {os.path.dirname(results_path)} =======")
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            if os.path.exists(results_path):
                continue

            model.load_state_dict(copy.deepcopy(model_original_state))
            metrics = eval_setfit(
                fewshot_ds[name],
                test_dataset,
                model,
                loss_class,
                args.num_epochs,
                metric,
            )

            with open(results_path, "w") as f_out:
                json.dump(
                    {"score": metrics[metric] * 100, "measure": metric},
                    f_out,
                    sort_keys=True,
                )
