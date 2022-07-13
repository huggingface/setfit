import argparse
import copy
import json
import math
import os
import pathlib
import sys
from shutil import copyfile
from warnings import simplefilter

import numpy as np
from datasets import load_dataset
from evaluate import load
from sentence_transformers import InputExample, losses
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
from setfit_wrapper import SetFit
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from utils import DEV_DATASET_TO_METRIC, TEST_DATASET_TO_METRIC

from setfit.data import SAMPLE_SIZES, create_fewshot_splits
from setfit.modeling import LOSS_NAME_TO_CLASS, SKLearnWrapper, SupConLoss, sentence_pairs_generation


# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="paraphrase-mpnet-base-v2")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sst2"],
    )
    parser.add_argument("--sample_sizes", type=int, nargs="+", default=SAMPLE_SIZES)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument(
        "--classifier",
        default="logistic_regression",
        choices=["logistic_regression", "svc-rbf", "svc-rbf-norm", "knn", "pytorch", "pytorch_complex"],
    )
    parser.add_argument("--loss", default="CosineSimilarityLoss")
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--add_normalization_layer", default=False, action="store_true")
    parser.add_argument("--optimizer_name", default="AdamW")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--is_dev_set", type=bool, default=False)
    parser.add_argument("--is_test_set", type=bool, default=False)
    args = parser.parse_args()

    return args


class RunFewShot:
    def __init__(self, args) -> None:
        # Prepare directory for results
        self.args = args

        parent_directory = pathlib.Path(__file__).parent.absolute()
        self.output_path = (
            parent_directory
            / "results"
            / f"{args.model.replace('/', '-')}-{args.loss}-{args.classifier}-epochs_{args.num_epochs}-batch_{args.batch_size}-{args.exp_name}".rstrip(
                "-"
            )
        )
        os.makedirs(self.output_path, exist_ok=True)

        # Save a copy of this training script and the run command in results directory
        train_script_path = os.path.join(self.output_path, "train_script.py")
        copyfile(__file__, train_script_path)
        with open(train_script_path, "a") as f_out:
            f_out.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

        # Configure dataset <> metric mapping. Defaults to accuracy
        if args.is_dev_set:
            self.dataset_to_metric = DEV_DATASET_TO_METRIC
        elif args.is_test_set:
            self.dataset_to_metric = TEST_DATASET_TO_METRIC
        else:
            self.dataset_to_metric = {dataset: "accuracy" for dataset in args.datasets}

        # Configure loss function
        self.loss_class = LOSS_NAME_TO_CLASS[args.loss]

        # Load SetFit Model
        self.model_wrapper = SetFit(
            self.args.model, max_seq_length=args.max_seq_length, add_normalization_layer=args.add_normalization_layer
        )
        self.model = self.model_wrapper.model

    def eval(self, x_train, y_train, x_test, y_test, metric):
        """Computes the metrics for a given classifier."""
        # Define metrics
        metric_fn = load(metric)

        clf = self.get_classifier(self.model)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        metrics = metric_fn.compute(predictions=y_pred, references=y_test)
        print(f"{metric} -- {metrics[metric]:.4f}")

        return metrics

    def get_classifier(self, sbert_model):
        if self.args.classifier == "logistic_regression":
            return SKLearnWrapper(sbert_model, LogisticRegression())

    def load_data_splits(self, dataset):
        """Loads a dataset from the Hugging Face Hub and returns the test split and few-shot training splits."""
        print(f"\n\n\n============== {dataset} ============")
        # Load one of the SetFit training sets from the Hugging Face Hub
        train_split = load_dataset(f"SetFit/{dataset}", split="train")
        few_shot_train_splits = create_fewshot_splits(train_split, self.args.sample_sizes)
        test_split = load_dataset(f"SetFit/{dataset}", split="test")
        print(f"Test set: {len(test_split)}")
        return few_shot_train_splits, test_split

    def train_eval_all_datasets(self):
        for dataset, metric in self.dataset_to_metric.items():
            few_shot_train_splits, test_split = self.load_data_splits(dataset)

            for split_name, split_data in few_shot_train_splits.items():
                results_path = os.path.join(self.output_path, dataset, split_name, "results.json")
                print(f"\n\n======== {os.path.dirname(results_path)} =======")
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                if os.path.exists(results_path):
                    print(f"Skipping finished experiment: {results_path}")
                    continue
                
                x_train = split_data["text"]
                y_train = split_data["label"]

                self.train(x_train, y_train, self.loss_class, self.args.num_epochs)

                x_test = test_split["text"]
                y_test = test_split["label"]

                metrics = self.eval(x_train, y_train, x_test, y_test, metric)

                with open(results_path, "w") as f_out:
                    json.dump({"score": metrics[metric] * 100, "measure": metric}, f_out, sort_keys=True)

    def train(self, x_train, y_train, loss_class, num_epochs):
        self.model.load_state_dict(copy.deepcopy(self.model_wrapper.model_original_state))

        if loss_class is None:
            return

        # sentence-transformers adaptation
        batch_size = self.args.batch_size
        if loss_class in [
            losses.BatchAllTripletLoss,
            losses.BatchHardTripletLoss,
            losses.BatchSemiHardTripletLoss,
            losses.BatchHardSoftMarginTripletLoss,
            SupConLoss,
        ]:

            train_examples = [InputExample(texts=[text], label=label) for text, label in zip(x_train, y_train)]
            train_data_sampler = SentenceLabelDataset(train_examples)

            batch_size = min(self.args.batch_size, len(train_data_sampler))
            train_dataloader = DataLoader(train_data_sampler, batch_size=batch_size, drop_last=True)

            if loss_class is losses.BatchHardSoftMarginTripletLoss:
                train_loss = loss_class(
                    model=self.model, distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance
                )
            elif loss_class is SupConLoss:
                train_loss = loss_class(model=self.model)
            else:
                train_loss = loss_class(
                    model=self.model, distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance, margin=0.25
                )

            train_steps = len(train_dataloader) * num_epochs
        else:
            train_examples = []
            for _ in range(num_epochs):
                train_examples = sentence_pairs_generation(np.array(x_train), np.array(y_train), train_examples)

            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
            train_loss = loss_class(self.model)
            train_steps = len(train_dataloader)

        print(f"{len(x_train)} train samples in total, {train_steps} train steps with batch size {batch_size}")

        warmup_steps = math.ceil(train_steps * 0.1)
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            steps_per_epoch=train_steps,
            warmup_steps=warmup_steps,
            show_progress_bar=False,
        )


def main():
    args = parse_args()

    run_fewshot = RunFewShot(args)
    run_fewshot.train_eval_all_datasets()


if __name__ == "__main__":
    main()
