#import argparse
import os
import torch
import numpy as np
from transformers import *

from src.data.Batcher import Batcher
from src.utils.Config import Config
#altered
from src.utils.util import device
from src.adapet import adapet
from src.eval.eval_model import test_eval


def do_test(exp_dir):
   #device = torch.device("cpu")
    config_file = os.path.join(exp_dir, "config.json")
    config = Config(config_file, mkdir=False)

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_weight)
    batcher = Batcher(config, tokenizer, config.dataset)
    dataset_reader = batcher.get_dataset_reader()

    model = adapet(config, tokenizer, dataset_reader).to(device)
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pt")))
    #altered
    pred_labels, pred_logits = test_eval(config, model, batcher)

    return pred_labels, pred_logits

#if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-e', "--exp_dir", required=True)

    #args = parser.parse_args()

'''
    config_file = os.path.join(args.exp_dir, "config.json")
    config = Config(config_file, mkdir=False)

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_weight)
    batcher = Batcher(config, tokenizer, config.dataset)
    dataset_reader = batcher.get_dataset_reader()

    model = adapet(config, tokenizer, dataset_reader).to(device)
    model.load_state_dict(torch.load(os.path.join(args.exp_dir, "best_model.pt")))
    test_eval(config, model, batcher)
'''

