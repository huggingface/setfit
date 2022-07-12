import argparse
import copy
import json
import math
import os
import pathlib
import sys
from shutil import copyfile
from warnings import simplefilter
from typing import List
from xmlrpc.client import Boolean
import pandas as pd


import numpy as np
from datasets import load_dataset,Dataset, DatasetDict
from evaluate import load
from sentence_transformers import InputExample, losses
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
from setfit_wrapper import SetFit
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from utils import DEV_DATASET_TO_METRIC, TEST_DATASET_TO_METRIC

from setfit.data import SAMPLE_SIZES
from setfit.modeling import LOSS_NAME_TO_CLASS, SKLearnWrapper, SupConLoss, sentence_pairs_generation

TEACHER=0
TEACHER_SEED= [0]
STUDENT=1
STUDENT_SEEDS= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)

def create_samples(df: pd.DataFrame, sample_size: int, seed: int, mode) -> pd.DataFrame:
    """Samples a DataFrame to create an equal number of samples per class (when possible)."""
    examples = []
    if mode==TEACHER:
        for label in df["label"].unique():
            subset = df.query(f"label == {label}")
            if len(subset) > sample_size:
                examples.append(subset.sample(sample_size, random_state=seed, replace=False))
            else:
                examples.append(subset)
                
        examples = pd.concat(examples)   
                
    if mode==STUDENT:  
        examples = df.sample(sample_size, random_state=seed, replace=False)
               
    return examples


def create_fewshot_splits(dataset: Dataset, sample_sizes: List[int], seeds, mode:Boolean) -> DatasetDict:
    """Creates training splits from the dataset with an equal number of samples per class (when possible)."""
    splits_ds = DatasetDict()
    df = dataset.to_pandas()
    for sample_size in sample_sizes:
        for idx, seed in enumerate(seeds):
            split_df = create_samples(df, sample_size, seed, mode)
            splits_ds[f"train-{sample_size}-{idx}"] = Dataset.from_pandas(split_df, preserve_index=False)
    return splits_ds

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", default="paraphrase-mpnet-base-v2")
    parser.add_argument("--student_model", default="paraphrase-MiniLM-L3-v2")
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
    def __init__(self, args, mode) -> None:
        # Prepare directory for results
        self.args = args

        if mode==TEACHER:
            model = args.teacher_model
            path_prefix = f"distil_teacher_{args.teacher_model.replace('/', '-')}"
         
        if mode==STUDENT:
            model = args.student_model
            path_prefix = f"distil_student_{args.student_model.replace('/', '-')}"
                
        parent_directory = pathlib.Path(__file__).parent.absolute()
        self.output_path = (
            parent_directory
            / "results"
            / f"{path_prefix}-{args.loss}-{args.classifier}-epochs_{args.num_epochs}-batch_{args.batch_size}-{args.exp_name}".rstrip(
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
            #self.args.model, max_seq_length=args.max_seq_length, add_normalization_layer=args.add_normalization_layer
            model, max_seq_length=args.max_seq_length, add_normalization_layer=args.add_normalization_layer
        )
        self.model = self.model_wrapper.model

    def compute_metrics(self, x_train, y_train, x_test, y_test, metric):
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

    def train(self, mode):
        for dataset, metric in self.dataset_to_metric.items():
            print(f"\n\n\n============== {dataset} ============")
            # Load one of the SetFit training sets from the Hugging Face Hub
            train_ds = load_dataset(f"SetFit/{dataset}", split="train")
            test_dataset = load_dataset(f"SetFit/{dataset}", split="test")
            print(f"Test set: {len(test_dataset)}")
            
            # if teacher training use only 1 split (send only 1 seed. seed= 0)
            if mode==TEACHER:
                fewshot_ds = create_fewshot_splits(train_ds, self.args.sample_sizes, seeds=TEACHER_SEED, mode=TEACHER)
            
            if mode==STUDENT: 
                fewshot_ds = create_fewshot_splits(train_ds, self.args.sample_sizes, seeds=STUDENT_SEEDS, mode=STUDENT)
                              
            for name in fewshot_ds:
                results_path = os.path.join(self.output_path, dataset, name, "results.json")
                print(f"\n\n======== {os.path.dirname(results_path)} =======")
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                # if os.path.exists(results_path):
                #     continue

                self.model.load_state_dict(copy.deepcopy(self.model_wrapper.model_original_state))
                metrics = self.eval_setfit(
                    fewshot_ds[name], test_dataset, self.loss_class, self.args.num_epochs, metric
                )

                with open(results_path, "w") as f_out:
                    json.dump({"score": metrics[metric] * 100, "measure": metric}, f_out, sort_keys=True)
                    

    def eval_setfit(self, train_data, test_data, loss_class, num_epochs, metric):
        x_train = train_data["text"]
        y_train = train_data["label"]

        x_test = test_data["text"]
        y_test = test_data["label"]

        if loss_class is None:
            return self.compute_metrics(x_train, y_train, x_test, y_test, metric)

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

        return self.compute_metrics(x_train, y_train, x_test, y_test, metric)


def main():
    args = parse_args()
    
    # Train few-shot teacher 
    fewshot_teacher = RunFewShot(args,mode=TEACHER)
    fewshot_teacher.train(mode=TEACHER)
    
    # Train few-shot student 
    fewshot_teacher = RunFewShot(args,mode=STUDENT)
    fewshot_teacher.train(mode=STUDENT)
    
    # generate student data
    
    X_train_embd_student = fewshot_teacher.model.encode(x_train_student)

if __name__ == "__main__":
    main()
