import argparse
import os
import torch
import numpy as np
from transformers import *

from src.data.Batcher import Batcher
from src.adapet import adapet
from src.utils.Config import Config
from src.eval.eval_model import dev_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--exp_dir", required=True)
    args = parser.parse_args()

    config_file = os.path.join(args.exp_dir, "config.json")
    config = Config(config_file, mkdir=False)
    config.eval_dev = True

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_weight)
    batcher = Batcher(config, tokenizer, config.dataset)
    dataset_reader = batcher.get_dataset_reader()
    
    model = adapet(config, tokenizer, dataset_reader).to(device)
    model.load_state_dict(torch.load(os.path.join(args.exp_dir, "best_model.pt")))
    dev_acc, dev_logits = dev_eval(config, model, batcher, 0)

    with open(os.path.join(config.exp_dir, "dev_logits.npy"), 'wb') as f:
        np.save(f, dev_logits)
