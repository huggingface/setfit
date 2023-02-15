import argparse
import json
import os
import pathlib
import sys
import warnings
from collections import Counter
from shutil import copyfile
from warnings import simplefilter

from datasets import load_dataset
from sentence_transformers import models
from typing_extensions import LiteralString

from setfit import SetFitModel, SetFitTrainer
from setfit.data import get_templated_dataset
from setfit.utils import DEV_DATASET_TO_METRIC, LOSS_NAME_TO_CLASS, TEST_DATASET_TO_METRIC


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="paraphrase-mpnet-base-v2")
    parser.add_argument("--eval_dataset", default="SetFit/emotion")
    parser.add_argument("--candidate_labels", nargs="+")
    parser.add_argument("--reference_dataset", default=None)
    parser.add_argument("--label_names_column", default="label_text")
    parser.add_argument("--aug_sample_size", type=int, default=8)
    parser.add_argument("--num_iterations", type=int, default=20)
    parser.add_argument("--num_epochs", type=int, default=1)
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
    parser.add_argument("--override_results", default=False, action="store_true")
    parser.add_argument("--keep_body_frozen", default=False, action="store_true")

    args = parser.parse_args()

    return args


def create_results_path(dataset: str, split_name: str, output_path: str) -> LiteralString:
    results_path = os.path.join(output_path, dataset, split_name, "results.json")
    print(f"\n\n======== {os.path.dirname(results_path)} =======")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    return results_path


def main():
    args = parse_args()

    parent_directory = pathlib.Path(__file__).parent.absolute()
    output_path = (
        parent_directory
        / "results"
        / f"{args.model.replace('/', '-')}-{args.loss}-{args.classifier}-iterations_{args.num_iterations}-batch_{args.batch_size}-{args.exp_name}".rstrip(
            "-"
        )
    )
    os.makedirs(output_path, exist_ok=True)

    # Save a copy of this training script and the run command in results directory
    train_script_path = os.path.join(output_path, "train_script.py")
    copyfile(__file__, train_script_path)
    with open(train_script_path, "a") as f_out:
        f_out.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

    # Configure loss function
    loss_class = LOSS_NAME_TO_CLASS[args.loss]

    metric = DEV_DATASET_TO_METRIC.get(args.eval_dataset, TEST_DATASET_TO_METRIC.get(args.eval_dataset, "accuracy"))

    if args.reference_dataset is None and args.candidate_labels is None:
        args.reference_dataset = args.eval_dataset

    train_data = get_templated_dataset(
        reference_dataset=args.reference_dataset,
        candidate_labels=args.candidate_labels,
        sample_size=args.aug_sample_size,
        label_names_column=args.label_names_column,
    )
    test_data = load_dataset(args.eval_dataset, split="test")

    print(f"Evaluating {args.eval_dataset} using {metric!r}.")

    # Report on an imbalanced test set
    counter = Counter(test_data["label"])
    label_samples = sorted(counter.items(), key=lambda label_samples: label_samples[1])
    smallest_n_samples = label_samples[0][1]
    largest_n_samples = label_samples[-1][1]
    # If the largest class is more than 50% larger than the smallest
    if largest_n_samples > smallest_n_samples * 1.5:
        warnings.warn(
            "The test set has a class imbalance "
            f"({', '.join(f'label {label} w. {n_samples} samples' for label, n_samples in label_samples)})."
        )

    results_path = create_results_path(args.eval_dataset, "zeroshot", output_path)
    if os.path.exists(results_path) and not args.override_results:
        print(f"Skipping finished experiment: {results_path}")
        exit()

    # Load model
    if args.classifier == "pytorch":
        model = SetFitModel.from_pretrained(
            args.model,
            use_differentiable_head=True,
            head_params={"out_features": len(set(train_data["label"]))},
        )
    else:
        model = SetFitModel.from_pretrained(args.model)
    model.model_body.max_seq_length = args.max_seq_length
    if args.add_normalization_layer:
        model.model_body._modules["2"] = models.Normalize()

    # Train on current split
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=test_data,
        metric=metric,
        loss_class=loss_class,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_iterations=args.num_iterations,
    )
    if args.classifier == "pytorch":
        trainer.freeze()
        trainer.train()
        trainer.unfreeze(keep_body_frozen=args.keep_body_frozen)
        trainer.train(
            num_epochs=25,
            body_learning_rate=1e-5,
            learning_rate=args.lr,  # recommend: 1e-2
            l2_weight=0.0,
            batch_size=args.batch_size,
        )
    else:
        trainer.train()

    # Evaluate the model on the test data
    metrics = trainer.evaluate()
    print(f"Metrics: {metrics}")

    with open(results_path, "w") as f_out:
        json.dump(
            {"score": metrics[metric] * 100, "measure": metric},
            f_out,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
