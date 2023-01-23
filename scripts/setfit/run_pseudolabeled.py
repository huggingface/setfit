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

PSEUDOLABELS_DIR = "data/pseudolabeled/"

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
    parser.add_argument("--allow_skip_exp", default=False, action="store_true")
    parser.add_argument("--keep_body_frozen", default=False, action="store_true")
    parser.add_argument("--add_data_augmentation", default=False, action="store_true")
    parser.add_argument("--pseudolabels_path", default=None)
    parser.add_argument("--train_split", type=int)
    parser.add_argument("--top_n", type=int)
    args = parser.parse_args()
    return args


def create_results_path(dataset: str, split_name: str, output_path: str) -> LiteralString:
    results_path = pathlib.Path(output_path) / dataset / split_name / "results.json"
    os.makedirs(results_path.parent, exist_ok=True)
    return results_path


def write_metrics(metrics, results_path, metric, pseudolabels_accuracy: float=None) -> None:
    metrics_dict = {"score": metrics[metric] * 100, "measure": metric}
    if pseudolabels_accuracy is not None:
        metrics_dict["pseudolabels_accuracy"] = str(pseudolabels_accuracy)

    metrics_json = json.dumps(metrics_dict, indent=2, sort_keys=True)
    print(f"Evaluation results: \n {metrics_json}")
    with open(results_path, "w") as f_out:
        f_out.write(metrics_json)


def main():
    args = parse_args()
    parent_directory = pathlib.Path(__file__).parent.resolve()
    output_path = parent_directory / "results" / args.exp_name
    os.makedirs(output_path, exist_ok=True)

    # Save a copy of this training script and the run command in results directory
    train_script_path = os.path.join(output_path, "train_script.py")
    copyfile(__file__, train_script_path)
    with open(train_script_path, "a") as f_out:
        f_out.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

    # Configure loss function
    loss_class = LOSS_NAME_TO_CLASS[args.loss]
    dataset = args.dataset

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

        results_path = create_results_path(dataset, "train-0-data_aug", output_path)
        if os.path.exists(results_path) and args.allow_skip_exp:
            print(f"Skipping finished experiment: {results_path}")
        else:
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
        pseudolabeled_examples, pl_accuracy = load_pseudolabeled_examples(PSEUDOLABELS_DIR + args.pseudolabels_path, args.top_n)

        results_path = create_results_path(dataset, split_name, output_path)
        experiment_name = str(results_path).split('results')[1].strip("/")

        if os.path.exists(results_path) and args.allow_skip_exp:
            print(f"Skipping finished experiment: {results_path}")
        else:
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
            print(f"Experiment: \n{experiment_name}\n")
            write_metrics(metrics, results_path, metric, pseudolabels_accuracy=pl_accuracy)


if __name__ == "__main__":
    main()
