import argparse
import json
import os
import pathlib
import sys
from shutil import copyfile
from typing import List
from warnings import simplefilter
from xmlrpc.client import Boolean

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from distillation_baseline import BaselineDistillation
from sentence_transformers import losses

from setfit import DistillationSetFitTrainer, SetFitModel, SetFitTrainer
from setfit.modeling import SetFitBaseModel
from setfit.utils import DEV_DATASET_TO_METRIC, TEST_DATASET_TO_METRIC


TEACHER_SEED = [0]
STUDENT_SEEDS = [1, 2, 3, 4, 5]
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
    parser.add_argument("--teacher_sample_sizes", type=int, nargs="+", default=[16])
    parser.add_argument(
        "--student_sample_sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 100, 200, 1000],
    )
    parser.add_argument("--num_iterations_teacher", type=int, default=20)
    parser.add_argument("--num_iterations_student", type=int, default=20)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size_teacher", type=int, default=16)
    parser.add_argument("--batch_size_student", type=int, default=16)
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
    def __init__(self, args, mode, trained_teacher_model, teacher_train_dataset, student_train_dataset) -> None:
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
            self.teacher_train_dataset = teacher_train_dataset
            self.mode = self.SETFIT_STUDENT

        if mode == self.BASELINE_STUDENT:
            model = args.baseline_student_model
            path_prefix = f"baseline_student_{args.student_model.replace('/', '-')}"
            self.trained_teacher_model = trained_teacher_model
            self.teacher_train_dataset = teacher_train_dataset
            self.student_train_dataset = student_train_dataset
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
            / f"{path_prefix}-{args.loss}-{args.classifier}-student_iters_{args.num_iterations_student}-batch_{args.batch_size_student}-{args.exp_name}".rstrip(
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
        self.loss_class = losses.CosineSimilarityLoss

        self.model_name = model
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
            eval_dataset = load_dataset(f"SetFit/{dataset}", split="test")
            print(f"Test set: {len(eval_dataset)}")

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
                self.student_train_dataset = fewshot_ds

            # for training baseline student use the same data that was used for training setfit student
            if self.mode == self.BASELINE_STUDENT:
                fewshot_ds = self.student_train_dataset
                num_classes = len(train_ds.unique("label"))
                self.bl_stdnt_distill.update_metric(metric)

            for name in fewshot_ds:
                results_path = os.path.join(self.output_path, dataset, name, "results.json")
                print(f"\n\n======== {os.path.dirname(results_path)} =======")
                os.makedirs(os.path.dirname(results_path), exist_ok=True)

                if self.mode == self.TEACHER:
                    teacher_model = SetFitModel.from_pretrained(self.model_name)
                    teacher_trainer = SetFitTrainer(
                        model=teacher_model,
                        train_dataset=fewshot_ds[name],
                        eval_dataset=eval_dataset,
                        loss_class=losses.CosineSimilarityLoss,
                        metric=metric,
                        batch_size=self.args.batch_size_teacher,
                        num_iterations=self.args.num_iterations_teacher,  # The number of text pairs to generate for contrastive learning
                        num_epochs=1,  # The number of epochs to use for constrastive learning
                    )
                    teacher_trainer.train()

                    # Evaluate the model on the test data
                    metrics = teacher_trainer.evaluate()
                    print("Teacher metrics: ", metrics)

                    self.teacher_train_dataset = fewshot_ds[name]  # save teacher training data
                    self.trained_teacher_model = teacher_trainer.model

                if self.mode == self.SETFIT_STUDENT:

                    # student train data = teacher train data + unlabeled data
                    student_train_dataset = concatenate_datasets([self.teacher_train_dataset, fewshot_ds[name]])

                    student_model = SetFitModel.from_pretrained(self.model_name)
                    student_trainer = DistillationSetFitTrainer(
                        teacher_model=self.trained_teacher_model,
                        train_dataset=student_train_dataset,
                        student_model=student_model,
                        eval_dataset=eval_dataset,
                        loss_class=losses.CosineSimilarityLoss,
                        metric="accuracy",
                        batch_size=self.args.batch_size_student,
                        num_iterations=self.args.num_iterations_student,  # The number of text pairs to generate for contrastive learning
                        # column_mapping={"sentence": "text", "label": "label"} # Map dataset columns to text/label expected by trainer
                    )
                    # Student Train and evaluate
                    student_trainer.train()
                    metrics = student_trainer.evaluate()

                    print("Student metrics: ", metrics)

                if self.mode == self.BASELINE_STUDENT:
                    student_train_dataset = concatenate_datasets([self.teacher_train_dataset, fewshot_ds[name]])
                    metrics = self.train_baseline_student(student_train_dataset, eval_dataset, num_classes)
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

        x_train_embd_student = self.trained_teacher_model.model_body.encode(x_train)

        # baseline student uses teacher probabilities (converted to logits) for training
        y_train_teacher_pred_prob = self.trained_teacher_model.model_head.predict_proba(x_train_embd_student)

        train_raw_student_prob = Dataset.from_dict({"text": x_train, "score": list(y_train_teacher_pred_prob)})
        metric = self.bl_stdnt_distill.standard_model_distillation(train_raw_student_prob, x_test, y_test, num_classes)

        return metric


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
        teacher_train_dataset=None,
        student_train_dataset=None,
    )
    fewshot_teacher.train()

    # 2. Train few-shot setfit student
    setfit_student = RunFewShotDistill(
        args,
        mode=SETFIT_STUDENT,
        trained_teacher_model=fewshot_teacher.trained_teacher_model,
        teacher_train_dataset=fewshot_teacher.teacher_train_dataset,
        student_train_dataset=None,
    )
    setfit_student.train()

    # 3. Train few-shot baseline student
    baseline_student = RunFewShotDistill(
        args,
        mode=BASELINE_STUDENT,
        trained_teacher_model=fewshot_teacher.trained_teacher_model,
        teacher_train_dataset=fewshot_teacher.teacher_train_dataset,
        student_train_dataset=setfit_student.student_train_dataset,
    )
    baseline_student.train()


if __name__ == "__main__":
    main()
