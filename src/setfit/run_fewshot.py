from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
from torch.utils.data import DataLoader
import math
from datasets import load_dataset
import numpy as np
import argparse
import copy
from warnings import simplefilter
import os
import json
from data import create_fewshot_splits
import sys
from shutil import copyfile
from modeling import LOSS_NAME_TO_CLASS, SupConLoss, SKLearnWrapper, sentence_pairs_generation

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="paraphrase-mpnet-base-v2")
parser.add_argument("--datasets", nargs="+", default=["sst2", "sst5",  "subj", "ag_news","bbc-news",
    "enron_spam","student-question-categories","TREC-QC","toxic_conversations","amazon_counterfactual_en", "imdb"])
parser.add_argument("--sample_sizes", type=int, nargs="+", default=None)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_seq_length", type=int, default=256)
parser.add_argument("--classifier", default="logistic_regression", choices=["logistic_regression", "svc-rbf", "svc-rbf-norm", "knn",
    "pytorch", "pytorch_complex"])
parser.add_argument("--loss", default="CosineSimilarityLoss")
parser.add_argument("--exp_name", default="")
parser.add_argument("--add_normalization_layer", default=False, action='store_true')
# parser.add_argument("--optimizer_name", default="AdamW")
# parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args()

output_path = f"results/stefit/{args.model.replace('/', '-')}-{args.loss}-{args.classifier}-epochs_{args.num_epochs}-batch_{args.batch_size}-{args.exp_name}".rstrip("-")
os.makedirs(output_path, exist_ok=True)

train_script_path = os.path.join(output_path, 'train_script.py')
copyfile(__file__, train_script_path)
with open(train_script_path, 'a') as f_out:
    f_out.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))


def test_classifier(x_train, y_train, x_test, y_test):
    """Computes the Accuracy/Average Precision for a given classifier."""

    clf = get_classifier(model)
    clf.fit(x_train, y_train)
   
    if perf_measure == "average_precision":
        y_pred = clf.predict_proba(x_test)
        if len(y_pred.shape) == 2:
            y_pred = y_pred[:, 1]

        ap = average_precision_score(y_test, y_pred) * 100
        print(f'Average Precision: {ap:.2f}')
        return ap
    else:
        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred) * 100
        print(f'Accuracy: {acc:.2f}')
        return acc


def get_classifier(sbert_model):
    if args.classifier == "logistic_regression":
        return SKLearnWrapper(sbert_model, LogisticRegression())


def eval_setfit(train_data, test_data, model, loss_class, num_epochs):
    x_train = train_data['text']
    y_train = train_data['label']

    x_test = test_data['text']
    y_test = test_data['label']

    if loss_class is None:
        return test_classifier(x_train, y_train, x_test, y_test)

    # S-BERT adaptation 
    batch_size = args.batch_size
    if loss_class in [losses.BatchAllTripletLoss, 
                    losses.BatchHardTripletLoss, 
                    losses.BatchSemiHardTripletLoss,
                    losses.BatchHardSoftMarginTripletLoss,
                    SupConLoss]:

        train_examples = [InputExample(texts=[text], label=label) for text, label in zip(x_train, y_train)] 
        train_data_sampler = SentenceLabelDataset(train_examples)

        batch_size = min(args.batch_size, len(train_data_sampler))
        train_dataloader = DataLoader(train_data_sampler, 
                                        batch_size=batch_size, 
                                        drop_last=True)


        if loss_class is losses.BatchHardSoftMarginTripletLoss:
            train_loss = loss_class(model=model, 
                                    distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance)
        elif loss_class is SupConLoss:
            train_loss = loss_class(model=model)
        else:
            train_loss = loss_class(model=model, 
                                    distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance, 
                                    margin=0.25)

        train_steps = len(train_dataloader) * num_epochs
    else:
        train_examples = [] 
        for _ in range(num_epochs):
            train_examples = sentence_pairs_generation(np.array(x_train), np.array(y_train), train_examples)

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = loss_class(model)
        train_steps = len(train_dataloader)

    print(f"{len(x_train)} train samples in total, {train_steps} train steps with batch size {batch_size}")
   

    warmup_steps = math.ceil(train_steps*0.1)
    print("Call model.fit")
    model.fit(train_objectives=[(train_dataloader, train_loss)], 
                epochs=1, 
                steps_per_epoch=train_steps, 
                warmup_steps=warmup_steps, 
                show_progress_bar=False)


    return test_classifier(x_train, y_train, x_test, y_test)


loss_class = LOSS_NAME_TO_CLASS[args.loss]

##################

model = SentenceTransformer(args.model)
model_original_state = copy.deepcopy(model.state_dict())
model.max_seq_length = args.max_seq_length

if args.add_normalization_layer:
    ##Add normalization layer
    model._modules['2'] = models.Normalize()


for dataset in args.datasets:
    print(f"\n\n\n============== {dataset} ============")
    dataset = dataset

    if dataset in ["amazon_counterfactual_en", "toxic_conversations"]:
        perf_measure = "average_precision"
    else:
        perf_measure = "accuracy"

    # Load one of the SetFit training sets from the Hugging Face Hub
    train_ds = load_dataset("SetFit/" + dataset, split="train")
    fewshot_ds = create_fewshot_splits(train_ds, args.sample_sizes)
    test_dataset = load_dataset("SetFit/" + dataset, split='test')

    print(f"Test set: {len(test_dataset)}")

    for name in fewshot_ds:
        results_path = os.path.join(output_path, dataset, name, "results.json")
        print(f"\n\n======== {os.path.dirname(results_path)} =======")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        if os.path.exists(results_path):
            continue

        model.load_state_dict(copy.deepcopy(model_original_state))
        score = eval_setfit(fewshot_ds[name], test_dataset, model, loss_class, args.num_epochs)

        with open(results_path, "w") as f_out:
            json.dump({'score': score, 'measure': perf_measure}, f_out, indent=4, sort_keys=True)
