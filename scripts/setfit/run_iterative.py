import argparse
import json
import os
import pathlib
import sys
from shutil import copyfile
from warnings import simplefilter

import numpy as np
from sentence_transformers import models, InputExample
from typing_extensions import LiteralString
from datasets import load_dataset

from setfit import SetFitModel, SetFitTrainer
from setfit.data import SAMPLE_SIZES
from setfit.utils import DEV_DATASET_TO_METRIC, LOSS_NAME_TO_CLASS, TEST_DATASET_TO_METRIC, load_data_splits


# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="paraphrase-mpnet-base-v2")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["emotion"],
    )
    parser.add_argument("--sample_sizes", type=int, nargs="+", default=SAMPLE_SIZES)
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
    parser.add_argument("--is_dev_set", type=bool, default=False)
    parser.add_argument("--is_test_set", type=bool, default=False)
    parser.add_argument("--override_results", default=False, action="store_true")
    parser.add_argument("--keep_body_frozen", default=False, action="store_true")
    parser.add_argument("--add_data_augmentation", default=False)
    parser.add_argument("--num_training_iterations", default=2)
    parser.add_argument("--unlabeled_bs", default=32)
    
    args = parser.parse_args()

    return args


def create_results_path(dataset: str, split_name: str, output_path: str) -> LiteralString:
    results_path = os.path.join(output_path, dataset, split_name, "results.json")
    print(f"\n\n======== {os.path.dirname(results_path)} =======")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    return results_path


# def get_quality_pairs(model: SetFitModel, examples):
#     pseudolabeled_examples = model.predict(examples["text"])
#     print(pseudolabeled_examples)
#     exit()


def generate_quality_pairs(model, train_sentences, train_labels, unlabeled_sentences):
    # Initialize two empty lists to hold the (sentence, sentence) pairs and
    # labels to indicate if a pair is positive or negative

    predictions = model.predict(unlabeled_sentences["text"])

    num_classes = np.unique(labels)
    idx = [np.where(labels == i)[0] for i in num_classes]

    for first_idx in range(len(sentences)):
        current_sentence = sentences[first_idx]
        label = labels[first_idx]
        second_idx = np.random.choice(idx[np.where(num_classes == label)[0][0]])
        positive_sentence = sentences[second_idx]
        # Prepare a positive pair and update the sentences and labels
        # lists, respectively
        pairs.append(InputExample(texts=[current_sentence, positive_sentence], label=1.0))

        negative_idx = np.where(labels != label)[0]
        negative_sentence = sentences[np.random.choice(negative_idx)]
        # Prepare a negative pair of sentences and update our lists
        pairs.append(InputExample(texts=[current_sentence, negative_sentence], label=0.0))
    # Return a 2-tuple of our sentence pairs and labels
    return pairs
    

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

    # Configure dataset <> metric mapping. Defaults to accuracy
    if args.is_dev_set:
        dataset_to_metric = DEV_DATASET_TO_METRIC
    elif args.is_test_set:
        dataset_to_metric = TEST_DATASET_TO_METRIC
    else:
        dataset_to_metric = {dataset: "accuracy" for dataset in args.datasets}

    # Configure loss function
    loss_class = LOSS_NAME_TO_CLASS[args.loss]

    for dataset, metric in dataset_to_metric.items():
        few_shot_train_splits, test_data, _ = load_data_splits(dataset, args.sample_sizes, args.add_data_augmentation)

        for split_name, train_data in few_shot_train_splits.items():
            results_path = create_results_path(dataset, split_name, output_path)
            if os.path.exists(results_path) and not args.override_results:
                print(f"Skipping finished experiment: {results_path}")
                continue
            
            quality_pairs = None
            unlabeled_dataset = load_dataset(dataset, split="validation").shuffle(seed=42)

            for i in range(args.num_training_iterations):
                # Load model
                model = SetFitModel.from_pretrained(args.model)
                model.model_body.max_seq_length = args.max_seq_length
                if args.add_normalization_layer:
                    model.model_body._modules["2"] = models.Normalize()


                # Train on current split
                trainer = SetFitTrainer(
                    model=model,
                    train_dataset=train_data,
                    extra_train_pairs=quality_pairs,
                    eval_dataset=test_data,
                    metric=metric,
                    loss_class=loss_class,
                    batch_size=args.batch_size,
                    num_epochs=args.num_epochs,
                    num_iterations=args.num_iterations,
                )
                trainer.train()

                unlabeled_batch = unlabeled_dataset.select(range(i * args.unlabeled_bs, (i + 1) * args.unlabeled_bs))
                quality_pairs = generate_quality_pairs(model, unlabeled_batch)

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
