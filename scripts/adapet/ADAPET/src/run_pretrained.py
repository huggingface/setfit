import argparse
import os
import torch
import numpy as np
from transformers import *

from src.data.Batcher import Batcher
from src.utils.Config import Config
from src.utils.util import device, ParseKwargs
from src.adapet import adapet
from src.eval.eval_model import dev_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model_dir", required=True)
    parser.add_argument('-c', "--config_file", required=True)
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_file, args.kwargs, mkdir=True)

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_weight)
    batcher = Batcher(config, tokenizer, config.dataset)
    dataset_reader = batcher.get_dataset_reader()

    model = adapet(config, tokenizer, dataset_reader).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "best_model.pt")))
    dev_eval(config, model, batcher, 0)

