import argparse
import json
import os
import pathlib
import sys
from shutil import copyfile
from warnings import simplefilter

from sentence_transformers import models
from typing_extensions import LiteralString

from setfit import SetFitModel, SetFitTrainer
from setfit.utils import DEV_DATASET_TO_METRIC, LOSS_NAME_TO_CLASS, TEST_DATASET_TO_METRIC, load_data_splits, load_pseudolabeled_examples


# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)

PSEUDOLABELS_DIR = "data/pseudolabeled/emotion_pairs/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="paraphrase-mpnet-base-v2")
    parser.add_argument("--dataset")
    parser.add_argument("--sample_size", type=int)
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
    parser.add_argument("--add_data_augmentation", default=False, action="store_true")
    parser.add_argument("--pseudolabels_path", default=None)
    parser.add_argument("--train_split", type=int)
    args = parser.parse_args()
    return args


def create_results_path(dataset: str, split_name: str, output_path: str) -> LiteralString:
    results_path = os.path.join(output_path, dataset, split_name, "results.json")
    print(f"\n\n======== {os.path.dirname(results_path)} =======")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    return results_path

def write_metrics(metrics, results_path, metric) -> None:
    with open(results_path, "w") as f_out:
        json.dump(
            {"score": metrics[metric] * 100, "measure": metric},
            f_out,
            sort_keys=True,
        )

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

    metric = TEST_DATASET_TO_METRIC.get(dataset, DEV_DATASET_TO_METRIC.get(dataset, 'accuracy'))

    few_shot_train_splits, test_data, _ = \
        load_data_splits(dataset, [args.sample_size], add_data_augmentation=args.sample_size == 0)
    
    split_name = f"train-{args.sample_size}-{args.train_split}"
    train_data = few_shot_train_splits[split_name]

    # Load model
    model = SetFitModel.from_pretrained(args.model)
    model.model_body.max_seq_length = args.max_seq_length
    if args.add_normalization_layer:
        model.model_body._modules["2"] = models.Normalize()

    ############ Zero-Shot with Data Augmentation ############
    if args.add_data_augmentation:
        few_shot_train_splits, test_data, _ = load_data_splits(dataset, [0], args.add_data_augmentation)
        zeroshot_train_data = few_shot_train_splits[split_name]

        results_path = create_results_path(dataset, "train-0-0", output_path)
        if os.path.exists(results_path) and not args.override_results:
            print(f"Skipping finished experiment: {results_path}")
            continue

        # Train on current split
        trainer = SetFitTrainer(
            model=model,
            train_dataset=zeroshot_train_data,
            eval_dataset=test_data,
            metric=metric,
            loss_class=loss_class,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            num_iterations=args.num_iterations,
        )
        trainer.train()

        # Evaluate the model on the test data
        metrics = trainer.evaluate()
        write_metrics(metrics, results_path, metric)

    ############ Train on pseudo-labeled data (Few-Shot / Zero-Shot) ############
    if args.pseudolabels_path is not None:
        pseudolabeled_examples = load_pseudolabeled_examples(PSEUDOLABELS_DIR + args.pseudolabels_path)

        results_path = create_results_path(dataset, split_name, output_path)
        if os.path.exists(results_path) and not args.override_results:
            print(f"Skipping finished experiment: {results_path}")
            continue

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
            pseudolabeled_examples=pseudolabeled_examples
        )
        
        trainer.train()

    # Evaluate the model on the test data
    metrics = trainer.evaluate()
    print(f"Metrics: {metrics}")
    print_and_write_metrics(metrics, results_path, metric)


if __name__ == "__main__":
    main()
