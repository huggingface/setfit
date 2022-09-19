import glob
import json
import os
import argparse
from functools import reduce
from transformers import AutoTokenizer

from cli import call_adapet
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from pandas.core.groupby import DataFrameGroupBy
from utilcode import fix_amzn, fix_train_amzn, write_seed_output, multiling_verb_pattern, str2bool

from src.test import do_test
from sklearn.metrics import average_precision_score, accuracy_score, matthews_corrcoef, mean_absolute_error
from utilcode import SINGLE_SENT_DATASETS, GLUE_DATASETS, AMZ_MULTI_LING



SEEDS = range(10)
SAMPLE_SIZES = [8, 64]          



def create_samples(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Samples a DataFrame to create an equal number of samples per class (when possible)."""
    examples = []
    for label in df["label"].unique():
        subset = df.query(f"label == {label}")
        if len(subset) > sample_size:
            examples.append(subset.sample(sample_size, random_state=seed, replace=False))
        else:
            examples.append(subset)
    return pd.concat(examples)

def create_fewshot_splits(sample_size, dataset: Dataset) -> DatasetDict:
    """Creates training splits from the dataset with an equal number of samples per class (when possible)."""
    splits_ds = DatasetDict()
    df = dataset.to_pandas()
    if sample_size == 500:
        #grab one
        for idx, seed in enumerate(range(0,1)):
            split_df = create_samples(df, sample_size, seed)
            splits_ds[f"train-{sample_size}-{idx}"] = Dataset.from_pandas(split_df, preserve_index=False)
    
    else:
        for idx, seed in enumerate(SEEDS):
            split_df = create_samples(df, sample_size, seed)
            splits_ds[f"train-{sample_size}-{idx}"] = Dataset.from_pandas(split_df, preserve_index=False)
    return splits_ds

def jsonl_from_dataset(dataset, task_name, updated_args, split="train"):
    """writes jsonl files from Dataset object in ADAPET format"""
    if task_name in GLUE_DATASETS:
        text1 = dataset["text1"]
        text2 = dataset["text2"]
    else:
        text = dataset["text"]
    label = dataset["label"]
    data_dir = updated_args.data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    writefile = data_dir + "/" + split + ".jsonl"
    try:
        os.remove(writefile)
        print("removing old {} file".format(split))

    except OSError:
        print("no old {} files found".format(split))
        pass
    print("writing new {} file".format(split))
    with open(writefile, "a") as f:
        if task_name in GLUE_DATASETS:
            for idx, txt1 in enumerate(text1):
                txt2 = text2[idx]
                lab = label[idx]
                json_dict = {"TEXT1": txt1, "TEXT2": txt2, "LBL": str(lab)}
                f.write(json.dumps(json_dict, ensure_ascii=False) + "\n", )
        elif task_name in AMZ_MULTI_LING: 
            for idx, txt in enumerate(text):
                lab = label[idx]
                json_dict = {"TEXT1": txt, "LBL": str(lab)}
                f.write(json.dumps(json_dict, ensure_ascii=False) + "\n")
        elif task_name in SINGLE_SENT_DATASETS:
            for idx, txt in enumerate(text):
                lab = label[idx]
                json_dict = {"TEXT1": txt, "LBL": str(lab)}
                f.write(json.dumps(json_dict) + "\n")
    print("{} split written".format(split))


def create_verbalizer(dataset):
    """creates verbalizer mapping from label to text label"""
    verbalizer = dict()
    label_text = dataset["label_text"]
    label = dataset["label"]
    for idx, lab in enumerate(label):
        if str(lab) not in verbalizer:
            verbalizer[str(lab)] = label_text[idx]
    
    return verbalizer

def get_max_num_lbl_tok(task_name, train_ds, pretrained_weight, lang_star_dict=None):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_weight)
    if task_name in AMZ_MULTI_LING and lang_star_dict:
        labels = set(lang_star_dict.values())
    else:
        labels = set(train_ds['label_text'])
    tokens = [tokenizer.encode(lab) for lab in labels]
    
    max_tokens = 0
    for t in tokens:
        num_tokens = len(t) - 2 #remove cls and sep token
        if num_tokens > max_tokens:
            max_tokens = num_tokens

    return max_tokens

def make_pattern(task_name, parser, english, prompt, lang_pattern):
    if task_name in GLUE_DATASETS:
        if task_name == 'SetFit/stsb':
            pattern = "[TEXT1] and [TEXT2] are [LBL]"
        
        else:
            pattern = "[TEXT1] and [TEXT2] imply [LBL]"
    
    elif task_name in AMZ_MULTI_LING: 
        pattern = lang_pattern
    
    else:
        pattern = "[TEXT1] this is [LBL]"
    
    return pattern

def update_parser(task_name, parser, max_tokens, verbalizer, lang_pattern):
    
    data_dir = "data/{}".format(task_name)
    args = parser.parse_args()
    english= args.english
    prompt = args.prompt
    pattern = make_pattern(task_name, parser, english, prompt, lang_pattern)
    
    
    args.data_dir = data_dir
    args.pattern = pattern
    args.dict_verbalizer = json.dumps(verbalizer, ensure_ascii=False)
    args.max_num_lbl_tok = int(max_tokens)
    
    return args

def write_generic_json(task_name, lang_pattern, verbalizer, updated_args, write_config="config/Generic.json"):    
    
    verbalizer = updated_args.dict_verbalizer
    max_tokens = updated_args.max_num_lbl_tok
    data_dir = updated_args.data_dir
        
    adapet_seed = updated_args.seed
    pattern = updated_args.pattern
    pretrained_weight = updated_args.pretrained_weight
    
    print('this is the pattern: {}'.format(pattern))
    print('this is the verbalizer: {}'.format(verbalizer))
    configs = {
        "pretrained_weight": pretrained_weight,
        "dataset": "generic",
        "generic_data_dir": data_dir,
        "pattern": pattern,
        "pattern_idx": 1,
        "dict_verbalizer": verbalizer,
        "idx_txt_trim": 1,
        "max_text_length": 256,
        "batch_size": 1,  # default is 1
        "eval_batch_size": 1, #default is 1. will crash on larger test sets if > 1
        "num_batches": updated_args.num_batches,  # default is 1000 MUST BE THE SAME AS eval_every or ADAPET will checkpoint on test set  
        "max_num_lbl_tok": int(max_tokens), #default is 1. gets automatically updated based on the tokenizer and dataset
        "eval_every": updated_args.eval_every, # default is 250 MUST BE THE SAME AS num_batches or ADAPET will checkpoint on test set  
        "eval_train": True,
        "warmup_ratio": 0.06,
        "mask_alpha": 0.105,
        "grad_accumulation_factor": 16,
        "seed": adapet_seed,
        "lr": 1e-5,
        "weight_decay": 1e-2,
    }
    if configs['num_batches'] != configs['eval_every']:
        raise ValueError("The number of batches and eval_every must be the same value to avoid checkpointing on test set")
        
    generic_json = json.dumps(configs, ensure_ascii=False)
    if not os.path.exists("config"):
        os.makedirs("config")
    try:
        os.remove(write_config)
        print("old config file deleted")
    except OSError:
        print("no config file found... writing new file")
        pass
    with open(write_config, "w") as f:
        f.write(generic_json)
    print("Generic json file written")
    return updated_args


def json_file_setup(task_name, train_ds, lang_pattern, max_tokens, parser):
    verbalizer = create_verbalizer(train_ds)
    updated_args = update_parser(task_name, parser, max_tokens, verbalizer, lang_pattern)
    #Pass ADAPET the training dataset 3 times, train, val, test
    jsonl_from_dataset(train_ds, task_name, updated_args, "train")
    
    jsonl_from_dataset(train_ds, task_name, updated_args, "val")
    
    jsonl_from_dataset(train_ds, task_name, updated_args, "test")
    

    write_generic_json(task_name, lang_pattern, verbalizer, updated_args)

    return updated_args

def multilingual_en(exp_dir, updated_args, pretrained_weight, sample_size, ds_seed):
    english = True
    prompt = True
    for task_name in AMZ_MULTI_LING:
        dataset = load_dataset(task_name)
        test_ds = dataset['test']
        y_true = test_ds["label"]
        jsonl_from_dataset(test_ds, task_name, updated_args, "test")
        pred_labels, pred_logits = do_test(exp_dir)
        mae = mean_absolute_error(y_true, pred_labels)*100
        write_seed_output(pretrained_weight+'_en', task_name, sample_size, ds_seed, mae, english, prompt)

def main(parser):
    args = parser.parse_args()
    english = args.english
    prompt = args.prompt  
    task_name = args.task_name
    pretrained_weight = args.pretrained_weight
    adapet_seed = args.seed
    multilingual = args.multilingual
    print("starting work on {}".format(task_name))        
    
    if task_name in AMZ_MULTI_LING:
        print('loading multilingual dataset')
        dataset = load_dataset(task_name)
        lang_star_dict, lang_pattern = multiling_verb_pattern(task_name, english, prompt)
        dataset = fix_amzn(dataset, lang_star_dict)
        if multilingual == 'all':
            dsets = []
            for task in AMZ_MULTI_LING:
                ds = load_dataset(task, split="train")
                dsets.append(ds)
            # Create training set and sample for fewshot splits
            train_ds = concatenate_datasets(dsets).shuffle(seed=42)
            train_ds = fix_train_amzn(train_ds, lang_star_dict)
        else:
            train_ds = dataset['train']
        
        test_ds = dataset['test']
    
    else:
        print('loading single sentence dataset')      
        train_ds = load_dataset(task_name, split="train")
        test_ds = load_dataset(task_name, split="test")  
    
    #determine the maximum number of tokens in the label text
    if task_name in AMZ_MULTI_LING:
        max_tokens = get_max_num_lbl_tok(task_name, train_ds, pretrained_weight, lang_star_dict)
    else: 
        max_tokens = get_max_num_lbl_tok(task_name, train_ds, pretrained_weight, lang_star_dict=None)

    num_labs = len(set(train_ds["label"]))
    
    if task_name not in AMZ_MULTI_LING:
        lang_pattern = None
    
    for sample_size in SAMPLE_SIZES:
        print("begun work on {} sample size : {}".format(task_name, sample_size))
        fewshot_ds = create_fewshot_splits(sample_size, train_ds)
        for ds_seed, ds in enumerate(fewshot_ds):                  
            current_split_ds = fewshot_ds[ds]                
        
            updated_args = json_file_setup(task_name, current_split_ds, lang_pattern, max_tokens, parser)

            # call ADAPET
            exp_dir = call_adapet(updated_args)
            
            #rewrite the existing "test" dataset with the true test data 
            jsonl_from_dataset(test_ds, task_name, updated_args, "test")
            
            pred_labels, pred_logits = do_test(exp_dir)
            
            y_true = test_ds["label"]

            if task_name in ['SetFit/toxic_conversations']:
                if len(pred_logits.shape) == 2:
                    y_pred = pred_logits[:, 1]
                    logit_ap = average_precision_score(y_true, y_pred)*100
                    write_seed_output(pretrained_weight, task_name, sample_size, ds_seed, logit_ap, english, prompt)
            
            elif task_name in ['SetFit/amazon_counterfactual_en']:
                mcc = matthews_corrcoef(y_true, pred_labels)*100
                write_seed_output(pretrained_weight, task_name, sample_size, ds_seed, mcc, english, prompt)
            
            elif task_name in AMZ_MULTI_LING and multilingual in ['each', 'all']:
                mae = mean_absolute_error(y_true, pred_labels)*100
                write_seed_output(pretrained_weight+'_'+multilingual, task_name, sample_size, ds_seed, mae, english, prompt)
            
            elif task_name == 'SetFit/amazon_reviews_multi_en' and multilingual == 'en':
                multilingual_en(exp_dir, updated_args, pretrained_weight, sample_size, ds_seed)
            
            else:
                acc = accuracy_score(y_true, pred_labels)*100
                write_seed_output(pretrained_weight, task_name, sample_size, ds_seed, acc, english, prompt)
                    
        print("no more work on {} sample size : {}".format(task_name, sample_size))
    print("finished work on {}".format(task_name))
    print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Arguments for running any datasets
    parser.add_argument('-d', "--data_dir", default=None, 
                            help="Data directory containing train/val/test jsonl files")
    parser.add_argument('-p', "--pattern", default=None, 
                            help="Pattern to be used for this dataset")
    parser.add_argument('-v', "--dict_verbalizer", type=json.loads, default=None, 
                        help="Dictionary mapping label name (in dataset) to the verbalizer to use, e.g. '{\"0\": \"Yes\", \"1\": \"No\"}'")

    # Model and training hyperparams
    parser.add_argument('-w', '--pretrained_weight', type=str, default='albert-xxlarge-v2', 
                            help='Pretrained model weights from huggingface')
    parser.add_argument('-bs', '--batch_size', type=int, default=1, help='batch size during training')
    parser.add_argument('--eval_batch_size', type=int, default=1, 
                            help='batch size during evaluation')
    parser.add_argument('--grad_accumulation_factor', type=int, default=16, help='number of gradient accumulation steps')
    parser.add_argument('--num_batches', type=int, default=1000, 
                            help='number of batches for experiment; 1 batch = grad_accumulation_factor x batch_size')

    parser.add_argument('--eval_every', type=int, default=1000, 
                            help='number of training batches per evaluation')
    parser.add_argument('--max_text_length', type=int, default=256, help='maximum total input sequence length after tokenization for ADAPET')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for the model')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay for the optmizer')
    parser.add_argument('--grad_clip_norm', type=float, default=1, help='gradient clipping norm')
    parser.add_argument('--warmup_ratio', type=float, default=0.06, help='linear warmup over warmup_steps for num_batches')

    # ADAPET hyperparameters
    parser.add_argument('--pattern_idx', default=1, help="Pattern index among all patterns available; For SuperGLUE, can use numbers >1 depending on dataset. For a new dataset, please set this to 1.")
    parser.add_argument('--mask_alpha', type=float, default=0.105, help='masking ratio for the label conditioning loss')
    parser.add_argument('--idx_txt_trim', type=int, default=1, help="TXT_ID of the text that can be trimmed (usually the longer text). Eg. if TXT1 needs to be trimmed, set this to 1.")
    parser.add_argument('--max_num_lbl_tok', type=int, default=1, help="The maximum number of tokens per label for the verbalizer. It will raise an error if the tokenizer tokenizes a label into more than 'max_num_lbl_tok' tokens.")
    parser.add_argument('--seed', type=int, default=0, action='store', help='adapet seed')
    # Replicating SuperGLUE results
    parser.add_argument('-c', '--config', type=str, default=None, help='Use this for replicating SuperGLUE results.')

    # SetFit data args
    parser.add_argument('--task_name', type=str, default='Setfit/sst2', action='store', help='Specifiy SetFit datasets')
    parser.add_argument('--english', type=str2bool, default='True', action='store', help='Prompt and verblize in English or not')
    parser.add_argument('--prompt', type=str2bool, default='True', action='store',help='keep the prompt or not')
    parser.add_argument('--multilingual', type=str, default='each', action='store',help='each, english, or all prompting')              
    
    main(parser)
    print("Job's done!")