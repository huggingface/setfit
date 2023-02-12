from datasets import load_dataset, concatenate_datasets,load_metric, Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from transformers import (HfArgumentParser, TrainingArguments)
import argparse
    
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, evaluation
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator, CECorrelationEvaluator
from torch.utils.data import DataLoader

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import torch
import random

from os import makedirs
from pathlib import Path
import pandas as pd

from collections import defaultdict
from itertools import chain, groupby
from typing import List

from setfit import SetFitModel, SetFitTrainer

from scripts.setfit.run_two_sents_test_quad_loss import setfit_two_sents_quad_loss

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

def sentence_pairs_generation(sentences, labels, pairs):
	# initialize two empty lists to hold the (sentence, sentence) pairs and
	# labels to indicate if a pair is positive or negative

  numClassesList = np.unique(labels)
  idx = [np.where(labels == i)[0] for i in numClassesList]

  for idxA in range(len(sentences)):      
    currentSentence = sentences[idxA]
    label = labels[idxA]
    idxB = np.random.choice(idx[np.where(numClassesList==label)[0][0]])
    posSentence = sentences[idxB]
		  # prepare a positive pair and update the sentences and labels
		  # lists, respectively
    pairs.append(InputExample(texts=[currentSentence, posSentence], label=1.0))

    negIdx = np.where(labels != label)[0]
    negSentence = sentences[np.random.choice(negIdx)]
		  # prepare a negative pair of images and update our lists
    pairs.append(InputExample(texts=[currentSentence, negSentence], label=0.0))
  
	# return a 2-tuple of our image pairs and labels
  return (pairs)

def generate_unified_format_dataset(in_data, task, sentence1, sentence2, label):
    data = []
    for row in in_data:
        data.append(
            InputExample(
                texts=[row[sentence1]] if task == "sst2" or task == "cola" else [row[sentence1], row[sentence2]],
                label= float(row[label]/5) if task == "stsb" else row[label]
            )
        )
    return data    

def generate_dataset(num_train_samples, full_train_data, full_eval_data, task_to_keys, seed):
    sentence1, sentence2 = task_to_keys[0]
    
    num_train_samples = min(num_train_samples, len(full_train_data))
    pos_ex = full_train_data.filter(lambda example: example['label']==1).select(range(num_train_samples)).shuffle(seed)
    neg_ex = full_train_data.filter(lambda example: example['label']==0).select(range(num_train_samples)).shuffle(seed)
    few_shot_train_data = concatenate_datasets([pos_ex, neg_ex])
    few_shot_train_data = few_shot_train_data.shuffle(seed)
    actual_train_data = few_shot_train_data
    
    train_data = pd.DataFrame({
    'sentence1': actual_train_data[sentence1],
    'sentence2': actual_train_data[sentence2],
    'label': actual_train_data['label']
        })

    eval_data = pd.DataFrame({
        'sentence1': full_eval_data[sentence1],
        'sentence2': full_eval_data[sentence2],
        'label': full_eval_data['label']
        })

    return train_data, eval_data    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="paraphrase-mpnet-base-v2")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sst2"],
    )
    parser.add_argument(
        "--sample_sizes",
        type=int,
        nargs="+",
        #default=[200, 500, 1000],
        default=[8, 16, 32, 64, 100, 200, 500, 1000],
    )
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
    parser.add_argument("--run_setfit_concat", default=True)
    parser.add_argument("--run_setfit_quad_loss", default=True)
    parser.add_argument("--run_e2e_test", default=False)
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

    args = parser.parse_args()

    return args    


def setfit_concat(args, x_train, y_train, x_eval, y_eval, metric):
    #train_examples = generate_multiple_sentence_pairs(x_train, y_train, num_itr)
    train_examples = [] 
    for x in range(args.num_iterations):
        train_examples = sentence_pairs_generation(np.array(x_train), np.array(y_train), train_examples)

    orig_model = SentenceTransformer(args.model)
    model = SentenceTransformer(args.model)

    # S-BERT adaptation 
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.CosineSimilarityLoss(model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=args.num_epochs, warmup_steps=10, show_progress_bar=True)

    # No Fit
    X_train_noFT = orig_model.encode(x_train)
    X_eval_noFT = orig_model.encode(x_eval)

    sgd =  LogisticRegression()
    sgd.fit(X_train_noFT, y_train)
    y_pred_eval_sgd = sgd.predict(X_eval_noFT)

    no_fit_result = metric.compute(predictions=y_pred_eval_sgd, references=y_eval)
    
    # With Fit (SetFit)
    X_train = model.encode(x_train)
    X_eval = model.encode(x_eval)

    sgd =  LogisticRegression()
    sgd.fit(X_train, y_train)
    y_pred_eval_sgd = sgd.predict(X_eval)

    setfit_result = metric.compute(predictions=y_pred_eval_sgd, references=y_eval)

    return round(setfit_result["accuracy"],3), round(no_fit_result["accuracy"],3)


def main():
    args = parse_args()
    seed = 0
    task = args.datasets[0]
    results_dir = "scripts/results"
    # parser = HfArgumentParser((Arguments, TrainingArguments))
    # args, training_args = parser.parse_args_into_dataclasses()
    # seed = training_args.seed
    # task = args.dataset_name
    # num_epochs = int(training_args.num_train_epochs)
    # batch_size = training_args.per_device_train_batch_size
    # few_shot = args.few_shot
    # num_train_steps = args.num_train_steps
    # results_dir = args.results_dir
    # num_iterations = args.args.num_iterations
    # st_model = args.st_model
    
    print('-------------')
    print('Starting eval')
    print('-------------')
    print('dataset_name = ', args.datasets[0])
   
    print('st_model = ', args.model)
    print('num_iterations = ', args.num_iterations)
    print('batch_size = ', args.batch_size)
    
    print('seed = ', seed)

    #args = sys.argv[1:]
    GLUE_TASKS_2SENT = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

    task_to_keys = {
        "cola": [("sentence", None),'validation',2],
        "mnli": [("premise", "hypothesis"),'validation_matched',3],
        "mnli-mm": [("premise", "hypothesis"),'validation_mismatched',3],
        "mrpc": [("sentence1", "sentence2"),'validation',2],
        "qnli": [("question", "sentence"),'validation',2],
        "qqp": [("question1", "question2"),'validation',2],
        "rte": [("sentence1", "sentence2"),'validation',2],
        "sst2": [("sentence", None),'validation',2],
        "stsb": [("sentence1", "sentence2"),'validation',1],
        "wnli": [("sentence1", "sentence2"),'validation',2]
    }

    actual_task = "mnli" if task == "mnli-mm" else task       # mnli-mm and mnli differ by the validation set 

    full_train_data = load_dataset('glue', actual_task, split='train')
    full_eval_data = load_dataset('glue', actual_task, split=task_to_keys[actual_task][1])
    metric = load_metric('glue', actual_task)

    #************ SetFit Concat ************

    set_seed(seed)

    # Generate few-shot train data: pick few positive & negative samples
    for num_train_samples in args.sample_sizes:
        print('************************************')
        print('num_train_samples = ',  num_train_samples)
        print('************************************')
        train_data, eval_data =  generate_dataset(num_train_samples, full_train_data, full_eval_data, task_to_keys[actual_task], seed)

        if args.run_setfit_concat=='1':
            train_data['concat_sen'] = train_data['sentence1'] +  train_data['sentence2']
            eval_data['concat_sen'] = eval_data['sentence1'] +  eval_data['sentence2']
            # Equal samples per class training
            train_df_sample = pd.concat([train_data[train_data.label==0], train_data[train_data.label==1]])
            
            x_train = train_df_sample.concat_sen.values.tolist()
            y_train = train_df_sample.label.values.tolist()

            x_eval = eval_data.concat_sen.values.tolist()
            y_eval = eval_data.label.values.tolist()

            setfit_result, no_fit_result = setfit_concat(args, x_train, y_train, x_eval, y_eval, metric)
        
            print("/nSetfit concat")
            print('Acc. No Fit HF', no_fit_result)
            print('Acc. SetFit HF', setfit_result)
            write_dir = Path(results_dir) / f"{args.datasets[0]}" / "setfit_concat" / f"num_ex={num_train_samples}_seed={seed}_res={setfit_result}"
            makedirs(write_dir, exist_ok=True)
            write_dir = Path(results_dir) / f"{args.datasets[0]}" / "setfit_concat_nofit" / f"num_ex={num_train_samples}_seed={seed}_res={no_fit_result}"
            makedirs(write_dir, exist_ok=True)
        #***********************************************************************************************
        #***************************   two_sents_quad_loss     *****************************************
        #***********************************************************************************************

        #actual_train_data = Dataset.from_pandas(actual_train_data, preserve_index=False)
        if args.run_setfit_quad_loss=='1':
            setfit_quad_result, no_fit_quad_result = setfit_two_sents_quad_loss(args, train_data, eval_data, metric)
            print('/nSetfit Quad loss ')
            print('Acc. No Fit HF', no_fit_quad_result)
            print('Acc. SetFit HF', setfit_quad_result)

            write_dir = Path(results_dir) / f"{args.datasets[0]}" / "setfit_quad_loss" / f"num_ex={num_train_samples}_seed={seed}_res={setfit_quad_result}"
            makedirs(write_dir, exist_ok=True)
            write_dir = Path(results_dir) / f"{args.datasets[0]}" / "setfit_quad_loss_nofit" / f"num_ex={num_train_samples}_seed={seed}_res={no_fit_quad_result}"
            makedirs(write_dir, exist_ok=True)

        #********************************* test *****************************************************
    
        if args.run_e2e_test=='1':
            
            num_classes=2
            model = SetFitModel.from_pretrained(
            args.model,
            use_differentiable_head=True,
            head_params={"out_features": num_classes},
            )

            trainer = SetFitTrainer(
                model=model,
                train_dataset=few_shot_train_data,
                eval_dataset=full_eval_data,
                loss_class=losses.CosineSimilarityLoss,
                metric="accuracy",
                batch_size=16,
                num_iterations=20, # The number of text pairs to generate for contrastive learning
                num_epochs=1, # The number of epochs to use for constrastive learning
                column_mapping={"sentence": "text", "label": "label"} # Map dataset columns to text/label expected by trainer
                )

            # Freeze head
            trainer.freeze()

            # Do contrastive training
            trainer.train(num_epochs=1)

            # Unfreeze head
            trainer.unfreeze()

            # Unfreeze head and freeze body
            # trainer.unfreeze(keep_body_frozen=True)

            # Train end-to-end
            trainer.train(num_epochs=1)


    print('Run Completed.')

if __name__ == "__main__":
    main()