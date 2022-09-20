import json
import os
import ast

from src.utils.util import make_exp_dir

class Config(object):
    def __init__(self, filename=None, kwargs=None, mkdir=True):
        # Dataset parameters
        self.dataset = "fewglue/BoolQ"
        self.num_lbl = 2
        self.max_num_lbl_tok = 10
        self.max_num_lbl = 10

        # Model and pattern parameters
        self.pretrained_weight = "bert-base-uncased"
        self.pattern_idx = "random"

        # Duration of training parameters
        self.batch_size = 8
        self.eval_batch_size = 64
        self.num_batches = 1000
        self.eval_every = 1
        self.grad_accumulation_factor = 1
        self.max_text_length = 64

        self.mask_alpha = 0.5

        self.eval_train = False
        self.eval_dev = True

        # Where experiments will be located
        self.exp_dir = None
        self.seed = 42
        self.exp_name = ""

        # Training Hyperparameters
        self.lr = 1e-3
        self.weight_decay = 0
        self.grad_clip_norm = 1
        self.warmup_ratio = 0

        # Generic dataset hyperparameters
        self.pattern = "[TEXT1] and [TEXT2] "
        self.idx_txt_trim = -1 # Indexed from 1
        self.dict_verbalizer = {"True": "Yes", "False": "No"}
        self.data_dir = "data/fewglue/BoolQ"
        #Added
        self.task_name = 'SetFit/sst2'

        if filename:
            self.__dict__.update(json.load(open(filename)))
        if kwargs:
            self.update_kwargs(kwargs)

        if filename or kwargs:
            self.update_exp_config(mkdir)

    def update_kwargs(self, kwargs):
        for (k, v) in kwargs.items():
            try:
                v = ast.literal_eval(v)
            except:
                v = v
            setattr(self, k, v)

    def update_exp_config(self, mkdir=True):
        '''
        Updates the config default values based on parameters passed in from config file
        '''


        self.base_dir = os.path.join("exp_out", self.dataset, self.pretrained_weight, self.task_name)
        if self.exp_name != "":
            self.base_dir = os.path.join(self.base_dir, self.exp_name)

        if mkdir:
            self.exp_dir = make_exp_dir(self.base_dir)

        if self.exp_dir is not None:
            self.dev_pred_file = os.path.join(self.exp_dir, "dev_pred.txt")
            self.dev_score_file = os.path.join(self.exp_dir, "dev_scores.json")
            self.test_score_file = os.path.join(self.exp_dir, "test_scores.json")
            self.save_config(os.path.join(self.exp_dir, os.path.join("config.json")))

    def to_json(self):
        '''
        Converts parameter values in config to json
        :return: json
        '''
        #altered -- ensure ascii now == False
        return json.dumps(self.__dict__, indent=4, sort_keys=True, ensure_ascii=False)

    def save_config(self, filename):
        '''
        Saves the config
        '''
        with open(filename, 'w+') as fout:
            fout.write(self.to_json())
            fout.write('\n')
