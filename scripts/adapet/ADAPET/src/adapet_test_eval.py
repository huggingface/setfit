import os

import numpy as np
import torch
from src.adapet import adapet
from src.data.Batcher import Batcher
from src.eval.eval_model import test_eval
from src.utils.Config import Config
from src.utils.util import device
from transformers import *


os.environ["WANDB_DISABLED"] = "true"

def test_evaluation(exp_dir):
    config_file = os.path.join(exp_dir, "config.json")
    os.path.join(exp_dir, "config.json")
    config = Config(config_file, mkdir=False)

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_weight)
    batcher = Batcher(config, tokenizer, config.dataset)
    dataset_reader = batcher.get_dataset_reader()

    model = adapet(config, tokenizer, dataset_reader).to(device)
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pt")))
    #model.load_state_dict(torch.load(os.path.join(args.exp_dir, "best_model.pt")))
    test_eval(config, model, batcher)