
from sentence_transformers import  InputExample

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from os import makedirs
from pathlib import Path
import pandas as pd
import argparse
from typing import List

from datasets import Dataset, DatasetDict, load_dataset

#from scripts.setfit.cross_enc import train_and_eval_cross_enc, generate_unified_format_dataset
from cross_enc import train_cross_enc, evaluate_cross_encoder

from setfit.utils import DEV_DATASET_TO_METRIC, LOSS_NAME_TO_CLASS, TEST_DATASET_TO_METRIC, load_data_splits

SEEDS = [1, 2]
LABELED = 0
UNLABELED = 1
LABELED_DATA_SIZE=1000

def create_samples(df: pd.DataFrame, sample_size: int, seed: int, mode) -> pd.DataFrame:
        """Samples a DataFrame to create an equal number of samples per class (when possible)."""
        examples = []

        if mode == LABELED:
            for label in df["label"].unique():
                subset = df.query(f"label == {label}")
                try:
                    examples.append(subset.sample(sample_size, random_state=seed, replace=False))
                except ValueError as err:
                    print('there are not enough labeld samples - try to increase labeled_split_size')

            examples = pd.concat(examples)

        if mode == UNLABELED:
            examples = df.sample(sample_size, random_state=seed, replace=False)


        return examples

def create_fewshot_splits(dataset: Dataset, sample_sizes: List[int], seeds, mode) -> DatasetDict:
    """Creates training splits from the dataset with an equal number of samples per class (when possible)."""
    splits_ds = DatasetDict()
    df = dataset.to_pandas()
    for sample_size in sample_sizes:
        for idx, seed in enumerate(seeds):
            split_df = create_samples(df, sample_size, seed, mode)
            splits_ds[f"train-{sample_size}-{idx}"] = Dataset.from_pandas(split_df, preserve_index=False)
    return splits_ds

def sentence_pairs_generation(sentences, labels, pairs):
    # Initialize two empty lists to hold the (sentence, sentence) pairs and
    # labels to indicate if a pair is positive or negative

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

def unlabeled_sentence_pairs_generation(sentences, labels, pairs):
    # Initialize two empty lists to hold the (sentence, sentence) pairs and
    # labels to indicate if a pair is positive or negative

    idx = list(range(0, len(sentences)))

    for first_idx in range(len(sentences)):
        current_sentence = sentences[first_idx]   
        second_idx = np.random.choice(idx)
        sencond_sentence = sentences[second_idx]
        label=1.0 if labels[first_idx]==labels[second_idx] else 0.0

        pairs.append(InputExample(texts=[current_sentence, sencond_sentence], label=label))

    # Return a 2-tuple of our sentence pairs and labels
    return pairs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="paraphrase-mpnet-base-v2")
    parser.add_argument("--cross_enc_model", default="bert-base-uncased")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sst2"],
    )
    parser.add_argument(
        "--sample_sizes",
        type=int,
        nargs="+",
        #default=[8, 16, 32, 64, 100],
        default=[8, 16],
    )

    parser.add_argument(
        "--num_top_pairs",
        type=int,
        nargs="+",
        default=[500],
    )

    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[1],
    )
    parser.add_argument("--num_iterations", type=int, default=5)
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
    parser.add_argument("--labeled_split_size", type=int, default=1000)
    parser.add_argument("--unlabeled_split_size", type=int, default=2000)
    parser.add_argument("--run_setfit_quad_loss", default=True)
    parser.add_argument("--run_e2e_test", default=False)
    parser.add_argument("--loss", default="CosineSimilarityLoss")
    parser.add_argument("--is_dev_set", type=bool, default=False)
    parser.add_argument("--is_test_set", type=bool, default=False)

    args = parser.parse_args()

    return args    

def main():

    args = parse_args()
    task = args.datasets
    sample_sizes = args.sample_sizes
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    results_dir = "scripts/results"
    num_classes=2
    
    print('-------------')
    print('Starting...  ')
    print('-------------')
    print('dataset_name = ', task)
    print('sample_sizes = ', sample_sizes)
    print('num iterations = ', args.num_iterations)
    print('num epochs = ', args.num_epochs)
    print('num top pairs = ', args.num_top_pairs)
    print('seed = ', args.seed)
    print('st_model = ', args.model)
    print('cross_enc_model = ', args.cross_enc_model)
    print('labeled_split_size = ', args.labeled_split_size)
    print('unlabeled_split_size = ', args.unlabeled_split_size)

   # Configure dataset <> metric mapping. Defaults to accuracy
    if args.is_dev_set:
        dataset_to_metric = DEV_DATASET_TO_METRIC
    elif args.is_test_set:
        dataset_to_metric = TEST_DATASET_TO_METRIC
    else:
        dataset_to_metric = {dataset: "accuracy" for dataset in args.datasets}

    for dataset, metric in dataset_to_metric.items(): 
        # use first


        all_train_ds = load_dataset(f"SetFit/{dataset}", split="train")
        all_train_ds = all_train_ds.shuffle()
        
        # 1st train split = few-shot labeled, 2nd split -> unlabeled pairs (labels exist but only used for evaluation not for training)
        train_labeled_ds = all_train_ds.select(list(range(0, args.labeled_split_size)))
        train_unlabeled_ds = all_train_ds.select(list(range(args.labeled_split_size, args.labeled_split_size+args.unlabeled_split_size)))
        

        #test_ds  = load_dataset(f"SetFit/{dataset}", split="test")

        few_shot_train_splits = create_fewshot_splits(
                    dataset=train_labeled_ds,
                    sample_sizes=sample_sizes,
                    seeds=SEEDS,
                    mode = LABELED
                )
         
        #few_shot_train_splits, test_data = load_data_splits(dataset, args.sample_sizes, add_data_augmentation=False)

        # generate unlabeled pairs
        x_unlabeled = train_unlabeled_ds["text"]
        y_unlabeled = train_unlabeled_ds["label"]
        
        unlabeled_examples = []
        unlabeled_pairs = unlabeled_sentence_pairs_generation(np.array(x_unlabeled), np.array(y_unlabeled), unlabeled_examples)
        
        # test_examples = []
        # test_examples = sentence_pairs_generation(
        #                     np.array(x_test), np.array(y_test), test_examples
        #                     )
    
        for split_name, train_data in few_shot_train_splits.items():
            print('--------------------------')
            print('Starting split ', split_name)
            print('--------------------------')
           
            x_train = train_data["text"]
            y_train = train_data["label"]
            train_examples = []
            for _ in range(args.num_iterations):
                train_examples = sentence_pairs_generation(
                            np.array(x_train), np.array(y_train), train_examples
                            )

            # the dev data goes into an evaluator
            unlabeled_sentence_pairs=[]
            unlabeled_gold_labels=[]
            for example in unlabeled_pairs:
                unlabeled_sentence_pairs.append(example.texts)
                unlabeled_gold_labels.append(example.label) 

            unlabeled_test_data = {'sentence_pairs':unlabeled_sentence_pairs, 'gold_labels':unlabeled_gold_labels}       
            #dev_data = generate_unified_format_dataset(test_data, sentence1, sentence2, label)            
            cross_encoder, test_data_dict = train_cross_enc(args.cross_enc_model,
                                                        train_examples, 
                                                        batch_size, 
                                                        num_epochs, 
                                                        num_classes, 
                                                        unlabeled_test_data)

            pred_result = evaluate_cross_encoder(cross_encoder, unlabeled_sentence_pairs, unlabeled_gold_labels)
            
            for top_num in args.num_top_pairs:
            
                pred_result_top=[]
                unlabeled_sentence_pairs = []
                unlabeled_gold_labels=[]
                for i in range(0, top_num):
                    unlabeled_sentence_pairs.append(test_data_dict['unlabeled_sentence_pairs_sorted'][i])
                    unlabeled_gold_labels.append(test_data_dict['unlabeled_gold_labels_sorted'][i])
                    #test_data.append(InputExample(texts=unlabeled_sentence_pairs_sorted[i], label=unlabeled_gold_labels_sorted[i]))
                
                pred_result_top.append(evaluate_cross_encoder(cross_encoder, unlabeled_sentence_pairs, unlabeled_gold_labels))
                

            print(f'pred_result: {pred_result}  pred_result_top ({args.num_top_pairs}): {pred_result_top}')
            write_dir = Path(results_dir) / f"{args.datasets[0]}" / "cross_enc_pseudo_pairs" / f"cross_enc_{split_name}_res={pred_result}_top_res({args.num_top_pairs})={pred_result_top}"
            makedirs(write_dir, exist_ok=True)


    

if __name__ == "__main__":
    main()  