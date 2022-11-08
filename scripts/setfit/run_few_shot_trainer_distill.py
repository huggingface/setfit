from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, SetFitTrainer 
from setfit.trainer_distill import DistilSetFitTrainer


def train_teacher():
    # Load a dataset from the Hugging Face Hub
    dataset = load_dataset("sst2")

    # Simulate the few-shot regime by sampling 8 examples per class
    num_classes = 2
    train_dataset_teacher = dataset["train"].shuffle(seed=42).select(range(16 * num_classes))
    train_dataset_student = dataset["train"].shuffle(seed=0).select(range(100))
    eval_dataset = dataset["validation"]

    # Load a SetFit model from Hub
    teacher_model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

    # Create trainer
    teacher_trainer = SetFitTrainer(
        model=teacher_model,
        train_dataset=train_dataset_teacher,
        eval_dataset=eval_dataset,
        loss_class=CosineSimilarityLoss,
        metric="accuracy",
        batch_size=16,
        num_iterations=20, # The number of text pairs to generate for contrastive learning
        num_epochs=1, # The number of epochs to use for constrastive learning
        column_mapping={"sentence": "text", "label": "label"} # Map dataset columns to text/label expected by trainer
    )

    # Teacher Train and evaluate
    teacher_trainer.train()
    metrics = teacher_trainer.evaluate()
    teacher_model = teacher_trainer.model
    print("Teacher results: ", metrics)


    #********************** Student training*********************************#

    student_model = SetFitModel.from_pretrained("nreimers/MiniLM-L3-H384-uncased")
    student_trainer = DistilSetFitTrainer(
        teacher_model = teacher_model,
        model=student_model,
        train_dataset=train_dataset_student,
        eval_dataset=eval_dataset,
        loss_class=CosineSimilarityLoss,
        metric="accuracy",
        batch_size=16,
        num_iterations=20, # The number of text pairs to generate for contrastive learning
        num_epochs=1, # The number of epochs to use for constrastive learning
        column_mapping={"sentence": "text", "label": "label"} # Map dataset columns to text/label expected by trainer
    )

    # Student Train and evaluate
    student_trainer.train()
    metrics = student_trainer.evaluate()
    print("Student results: ", metrics)
    


def main():
    train_teacher()


if __name__ == "__main__":
    main()