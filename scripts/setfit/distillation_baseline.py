import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from evaluate import load
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


class BaselineDistillation:
    def __init__(self, student_model_name, num_epochs, batch_size) -> None:
        self.student_model_name = student_model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        self.seq_len = 64
        self.learning_rate = 6e-5

    def update_metric(self, metric):
        self.metric = load(metric)
        self.metric_name = metric

    def bl_student_preprocess(self, examples):
        label = examples["score"]
        examples = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.seq_len,
        )
        # Change this to real number
        examples["label"] = [float(i) for i in label]
        return examples

    def compute_metrics_for_regression(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        hot_labels = np.argmax(labels, axis=-1)
        return self.metric.compute(predictions=predictions, references=hot_labels)

    # ----------------------------------------------------------------#
    # ------------------------ Student training ----------------------#
    # ----------------------------------------------------------------#
    def standard_model_distillation(self, train_raw_student, x_test, y_test, num_classes):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        value2hot = {}
        for i in range(num_classes):
            a = [0] * num_classes
            a[i] = 1
            value2hot.update({i: a})

        test_dict = {"text": x_test, "score": [value2hot[i] for i in y_test]}
        raw_test_ds = Dataset.from_dict(test_dict)

        # validation and test sets are the same
        ds = {
            "train": train_raw_student,
            "validation": raw_test_ds,
            "test": raw_test_ds,
        }
        for split in ds:
            ds[split] = ds[split].map(self.bl_student_preprocess, remove_columns=["text", "score"])

        training_args = TrainingArguments(
            output_dir="baseline_distil_model",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            evaluation_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,
            weight_decay=0.01,
            push_to_hub=False,
        )

        # define data_collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # define student model
        student_model = AutoModelForSequenceClassification.from_pretrained(
            self.student_model_name, num_labels=num_classes
        ).to(device)

        trainer = RegressionTrainer(
            student_model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics_for_regression,
        )

        trainer.train()

        trainer.eval_dataset = ds["test"]
        # acc = round(trainer.evaluate()["eval_accuracy"], 3)

        score = trainer.evaluate()[f"eval_{self.metric_name}"]
        return {self.metric_name: score}
