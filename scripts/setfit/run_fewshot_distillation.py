import argparse
import copy
import json
import math
import os
import pathlib
import sys
from shutil import copyfile
from typing import List
from warnings import simplefilter
from xmlrpc.client import Boolean

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from distillation_baseline import BaselineDistillation
from evaluate import load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

from setfit.modeling import (
    LOSS_NAME_TO_CLASS,
    SetFitBaseModel,
    SKLearnWrapper,
    sentence_pairs_generation,
    sentence_pairs_generation_cos_sim,
)
from setfit.utils import DEV_DATASET_TO_METRIC, TEST_DATASET_TO_METRIC


TEACHER_SEED = [0]
STUDENT_SEEDS = [1]
# STUDENT_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", default="paraphrase-mpnet-base-v2")
    parser.add_argument("--student_model", default="paraphrase-MiniLM-L3-v2")
    parser.add_argument("--baseline_student_model", default="nreimers/MiniLM-L3-H384-uncased")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sst2"],
    )
    parser.add_argument("--teacher_sample_sizes", type=int, nargs="+", default=16)
    parser.add_argument(
        "--student_sample_sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 100, 200, 1000],
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--baseline_model_epochs", type=int, default=10)
    parser.add_argument("--baseline_model_batch_size", type=int, default=16)

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
    args = parser.parse_args()

    return args


class RunFewShotDistill:
    def __init__(self, args, mode, trained_teacher_model, x_train_teacher, student_train_ds) -> None:
        # Prepare directory for results
        self.args = args

        # these attributes refer to the different modes to run the training
        self.TEACHER = 0
        self.SETFIT_STUDENT = 1
        self.BASELINE_STUDENT = 2

        if mode == self.TEACHER:
            model = args.teacher_model
            path_prefix = f"setfit_teacher_{args.teacher_model.replace('/', '-')}"
            self.mode = self.TEACHER

        if mode == self.SETFIT_STUDENT:
            model = args.student_model
            path_prefix = f"setfit_student_{args.student_model.replace('/', '-')}"
            self.trained_teacher_model = trained_teacher_model
            self.x_train_teacher = x_train_teacher
            self.mode = self.SETFIT_STUDENT

        if mode == self.BASELINE_STUDENT:
            model = args.baseline_student_model
            path_prefix = f"baseline_student_{args.student_model.replace('/', '-')}"
            self.trained_teacher_model = trained_teacher_model
            self.x_train_teacher = x_train_teacher
            self.student_train_ds = student_train_ds
            self.mode = self.BASELINE_STUDENT
            self.bl_stdnt_distill = BaselineDistillation(
                args.baseline_student_model,
                args.baseline_model_epochs,
                args.baseline_model_batch_size,
            )

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
        self.model_wrapper = SetFitBaseModel(
            # self.args.model, max_seq_length=args.max_seq_length, add_normalization_layer=args.add_normalization_layer
            model,
            max_seq_length=args.max_seq_length,
            add_normalization_layer=args.add_normalization_layer,
        )
        self.model = self.model_wrapper.model

    def create_samples(self, df: pd.DataFrame, sample_size: int, seed: int, mode) -> pd.DataFrame:
        """Samples a DataFrame to create an equal number of samples per class (when possible)."""
        examples = []
        if mode == self.TEACHER:
            for label in df["label"].unique():
                subset = df.query(f"label == {label}")
                if len(subset) > sample_size:
                    examples.append(subset.sample(sample_size, random_state=seed, replace=False))
                else:
                    examples.append(subset)

            examples = pd.concat(examples)

        if mode == self.SETFIT_STUDENT:
            examples = df.sample(sample_size, random_state=seed, replace=False)

        return examples

    def create_fewshot_splits(self, dataset: Dataset, sample_sizes: List[int], seeds, mode: Boolean) -> DatasetDict:
        """Creates training splits from the dataset with an equal number of samples per class (when possible)."""
        splits_ds = DatasetDict()
        df = dataset.to_pandas()
        for sample_size in sample_sizes:
            for idx, seed in enumerate(seeds):
                split_df = self.create_samples(df, sample_size, seed, mode)
                splits_ds[f"train-{sample_size}-{idx}"] = Dataset.from_pandas(split_df, preserve_index=False)
        return splits_ds

    def compute_metrics(self, x_train, y_train, x_test, y_test, y_pred, metric):
        """Computes the metrics for a given classifier."""
        # Define metrics
        metric_fn = load(metric)

        # clf = self.get_classifier(self.model)
        # clf.fit(x_train, y_train)
        # y_pred = clf.predict(x_test)

        metrics = metric_fn.compute(predictions=y_pred, references=y_test)
        print(f"{metric} -- {metrics[metric]:.4f}")

        return metrics

    def get_classifier(self, sbert_model):
        if self.args.classifier == "logistic_regression":
            return SKLearnWrapper(sbert_model, LogisticRegression())

    def train(self):
        for dataset, metric in self.dataset_to_metric.items():
            if self.mode == self.TEACHER:
                print("\n\n\n=========== Training Teacher =========")
            if self.mode == self.SETFIT_STUDENT:
                print("\n\n\n======== Training SetFit Student ======")
            if self.mode == self.BASELINE_STUDENT:
                print("\n\n\n======== Training Baseline Student ======")
            print(f"\n\n\n============== {dataset} ============")
            # Load one of the SetFit training sets from the Hugging Face Hub
            train_ds = load_dataset(f"SetFit/{dataset}", split="train")
            test_dataset = load_dataset(f"SetFit/{dataset}", split="test")
            print(f"Test set: {len(test_dataset)}")

            # if teacher training use only 1 split (send only 1 seed. seed= 0)
            if self.mode == self.TEACHER:
                fewshot_ds = self.create_fewshot_splits(
                    train_ds,
                    self.args.teacher_sample_sizes,
                    seeds=TEACHER_SEED,
                    mode=self.TEACHER,
                )

            if self.mode == self.SETFIT_STUDENT:
                fewshot_ds = self.create_fewshot_splits(
                    train_ds,
                    self.args.student_sample_sizes,
                    seeds=STUDENT_SEEDS,
                    mode=self.SETFIT_STUDENT,
                )
                self.student_train_ds = fewshot_ds

                # for training baseline student use the same data that was used for training setfit student
            if self.mode == self.BASELINE_STUDENT:
                fewshot_ds = self.student_train_ds
                num_classes = len(train_ds.unique("label"))
                self.bl_stdnt_distill.update_metric(metric)

            for name in fewshot_ds:
                results_path = os.path.join(self.output_path, dataset, name, "results.json")
                print(f"\n\n======== {os.path.dirname(results_path)} =======")
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                # if os.path.exists(results_path):
                #     continue

                if (self.mode == self.TEACHER) or (self.mode == self.SETFIT_STUDENT):
                    self.model.load_state_dict(copy.deepcopy(self.model_wrapper.model_original_state))
                    metrics = self.train_eval_setfit(
                        fewshot_ds[name],
                        test_dataset,
                        self.loss_class,
                        self.args.num_epochs,
                        metric,
                    )

                if self.mode == self.BASELINE_STUDENT:
                    metrics = self.train_baseline_student(fewshot_ds[name], test_dataset, num_classes)
                    print("Baseline model score: ", round(metrics[metric] * 100, 3))

                with open(results_path, "w") as f_out:
                    json.dump(
                        {"score": round(metrics[metric] * 100, 3), "measure": metric},
                        f_out,
                        sort_keys=True,
                    )

    def train_baseline_student(self, train_data, test_data, num_classes):
        x_train = train_data["text"]
        x_test = test_data["text"]
        y_test = test_data["label"]

        x_train = self.x_train_teacher + x_train
        x_train_embd_student = self.trained_teacher_model.sbert_model.encode(x_train)
        # baseline student uses teacher probabilities (converted to logits) for training
        y_train_teacher_pred_prob = self.trained_teacher_model.clf.predict_proba(x_train_embd_student)
        train_raw_student_prob = Dataset.from_dict({"text": x_train, "score": list(y_train_teacher_pred_prob)})

        metric = self.bl_stdnt_distill.standard_model_distillation(train_raw_student_prob, x_test, y_test, num_classes)

        return metric

    def train_eval_setfit(self, train_data, test_data, loss_class, num_epochs, metric):
        x_train = train_data["text"]
        y_train = train_data["label"]

        x_test = test_data["text"]
        y_test = test_data["label"]

        if loss_class is None:
            return self.compute_metrics(x_train, y_train, x_test, y_test, metric)

        # sentence-transformers adaptation
        batch_size = self.args.batch_size

        if self.mode == self.TEACHER:
            # save teacher train data for student training
            self.x_train_teacher = x_train
            train_examples = []
            for _ in range(num_epochs):
                train_examples = sentence_pairs_generation(np.array(x_train), np.array(y_train), train_examples)

        if self.mode == self.SETFIT_STUDENT:
            # generate student data
            # student train data = teacher train data + unlabeled data
            x_train = self.x_train_teacher + x_train
            x_train_embd_student = self.trained_teacher_model.sbert_model.encode(x_train)
            y_train = self.trained_teacher_model.clf.predict(x_train_embd_student)

            # setfit student uses cosine similarity between pairs for training
            cos_sim_matrix = [[0 for j in range(len(x_train))] for i in range(len(x_train))]
            for first_idx in range(len(x_train)):
                for second_idx in range(len(x_train)):
                    cos_sim_matrix[first_idx][second_idx] = float(
                        cosine_similarity(
                            x_train_embd_student[first_idx].reshape(1, -1),
                            x_train_embd_student[second_idx].reshape(1, -1),
                        )
                    )

            train_examples = []
            for x in range(num_epochs):
                train_examples = sentence_pairs_generation_cos_sim(np.array(x_train), train_examples, cos_sim_matrix)

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = loss_class(self.model)
        train_steps = len(train_dataloader)

        print(f"{len(x_train)} train samples in total, {train_steps} train steps with batch size {batch_size}")

        warmup_steps = math.ceil(train_steps * 0.1)
        # train sentence transformer
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            steps_per_epoch=train_steps,
            warmup_steps=warmup_steps,
            show_progress_bar=False,
        )
        # train classification head
        clf = self.get_classifier(self.model)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        if self.mode == self.TEACHER:
            self.full_setfit_model = clf

        return self.compute_metrics(x_train, y_train, x_test, y_test, y_pred, metric)


def main():
    args = parse_args()
    TEACHER = 0
    SETFIT_STUDENT = 1
    BASELINE_STUDENT = 2
    # 1. Train few-shot teacher
    fewshot_teacher = RunFewShotDistill(
        args,
        mode=TEACHER,
        trained_teacher_model=None,
        x_train_teacher=None,
        student_train_ds=None,
    )
    fewshot_teacher.train()

    # 2. Train few-shot setfit student
    setfit_student = RunFewShotDistill(
        args,
        mode=SETFIT_STUDENT,
        trained_teacher_model=fewshot_teacher.full_setfit_model,
        x_train_teacher=fewshot_teacher.x_train_teacher,
        student_train_ds=None,
    )
    setfit_student.train()

    # 3. Train few-shot baseline student
    baseline_student = RunFewShotDistill(
        args,
        mode=BASELINE_STUDENT,
        trained_teacher_model=fewshot_teacher.full_setfit_model,
        x_train_teacher=fewshot_teacher.x_train_teacher,
        student_train_ds=setfit_student.student_train_ds,
    )
    baseline_student.train()


if __name__ == "__main__":
    main()
