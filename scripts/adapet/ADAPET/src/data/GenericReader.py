import os
import json
import random
import itertools
import numpy as np
import torch
from collections import defaultdict

from src.data.tokenize import tokenize_pet_txt, tokenize_pet_mlm_txt
from src.utils.util import device

class GenericReader(object):
    '''
    GenericReader reads generic dataset
    '''

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer


        self.num_lbl = len(self.config.dict_verbalizer)

        self.config.pattern += f" {self.tokenizer.sep_token}"
        self.check_pattern(self.config.pattern)

        txt_idx_trim = "[TEXT%d]" % self.config.idx_txt_trim

        self.pattern = self.config.pattern.split(txt_idx_trim)
        self.pattern.insert(1, txt_idx_trim)

        self.label = list(self.config.dict_verbalizer.values())

        self.list_true_lbl = []

    def check_pattern(self, pattern):

        # Get maximum number of text
        self.text_ctr = 1
        while True:
            text_str = "[TEXT%d]" % self.text_ctr
            if text_str in pattern:
                self.text_ctr +=1
            else:
                break

        if self.text_ctr == 1:
            raise ValueError("Need at least one text ")

        if self.config.idx_txt_trim > self.text_ctr:
            raise ValueError("Text idx to trim %d is larger than number of text inputs %d" % (self.config.idx_txt_trim, self.text_ctr))

        num_mask_tok = pattern.count("[LBL]")
        if num_mask_tok != 1:
            raise ValueError("[LBL] must be in pattern 1 time, but is in pattern %d times" % pattern)


    def _get_file(self, split):
        '''
        Get filename of split

        :param split:
        :return:
        '''
        if split.lower() == "train":
            file = os.path.join(self.config.data_dir, "train.jsonl")
        elif split.lower() == "dev":
            file = os.path.join(self.config.data_dir, "val.jsonl")
        elif split.lower() == "test":
            file = os.path.join(self.config.data_dir, "test.jsonl")
        return file

    def get_num_lbl_tok(self):
        return self.config.max_num_lbl_tok

    def read_dataset(self, split=None, is_eval=None):
        '''
        Read the dataset
        :param split: partition of the dataset
        :param is_eval:

        :return:
        '''
        file = self._get_file(split)

        data = []

        with open(file, 'r') as f_in:
            for i, line in enumerate(f_in.readlines()):
                json_string = json.loads(line)

                dict_input = {}
                dict_input["idx"] = i
                for j in range(1, self.text_ctr):
                    dict_input["TEXT%d" % j] = json_string["TEXT%d" % j]

                dict_output = {}
                if "LBL" not in json_string:
                    raise ValueError("LBL not in json")

                if json_string["LBL"] not in self.config.dict_verbalizer:
                    raise ValueError("Label %s not in dictionary verbalizer" % json_string["LBL"])
                dict_output["lbl"] = list(self.config.dict_verbalizer.keys()).index(json_string["LBL"])
                dict_input_output = {"input": dict_input, "output": dict_output}
                data.append(dict_input_output)
        return data

    @property
    def pets(self):
        return ["PET1"]

    def prepare_pet_batch(self, batch, mode="PET1"):
        '''
        Prepare for train

        :param batch:
        :return:
        '''

        list_list_txt = [] # [num_text, bs]
        for i in range(1, self.text_ctr):
            list_list_txt.append(batch["input"]["TEXT%d" % i])

        if self.config.max_num_lbl_tok > 1:
            return self.prepare_pet_batch_multi_token_label(batch, list_list_txt)
        else:
            return self.prepare_pet_batch_single_token_label(batch, list_list_txt)

    def get_lbl_num_lbl_tok(self, lbl):
        num_lbl_tok = len(self.tokenizer(lbl, add_special_tokens=False)["input_ids"])
        return min(num_lbl_tok, self.get_num_lbl_tok())

    def prepare_pet_batch_single_token_label(self, batch, list_list_txt):
        '''
        Prepare pet batch when the labels only consist of 1 token

        '''

        bs = len(batch["input"]["TEXT1"])


        list_input_ids = []
        list_mask_idx = np.ones((bs, self.get_num_lbl_tok())) * self.config.max_text_length

        txt_trim = 1

        for b_idx in range(bs):
            mask_txt_split_tuple = []

            for idx, txt_split in enumerate(self.pattern):
                for i in range(1, self.text_ctr):
                    txt_split = txt_split.replace("[TEXT%d]" % i, list_list_txt[i-1][b_idx])
                txt_split = txt_split.replace("[LBL]", self.tokenizer.mask_token)
                mask_txt_split_tuple.append(txt_split)

            input_ids, mask_idx = tokenize_pet_txt(self.tokenizer, self.config, mask_txt_split_tuple[0], mask_txt_split_tuple[1], mask_txt_split_tuple[2], mask_txt_split_tuple[0], mask_txt_split_tuple[1], mask_txt_split_tuple[2], txt_trim)
            list_input_ids.append(input_ids)
            list_mask_idx[b_idx,:self.get_num_lbl_tok()] = range(mask_idx, mask_idx+self.get_num_lbl_tok())

        return torch.tensor(list_input_ids).to(device), torch.tensor(list_mask_idx).to(device), self.label

    def prepare_pet_batch_multi_token_label(self, batch, list_list_txt):
        '''
        Prepare pet batch when the labels only consist of 1 token

        '''

        bs = len(batch["input"]["TEXT1"])

        list_input_ids = []
        list_mask_idx = np.ones((bs, self.num_lbl, self.get_num_lbl_tok())) * self.config.max_text_length - 1
        txt_trim = 1

        for b_idx in range(bs):
            mask_txt_split_tuple = []

            for idx, txt_split in enumerate(self.pattern):
                for i in range(1, self.text_ctr):
                    txt_split = txt_split.replace("[TEXT%d]" % i, list_list_txt[i - 1][b_idx])
                txt_split = txt_split.replace("[LBL]", self.tokenizer.mask_token * self.get_num_lbl_tok())
                mask_txt_split_tuple.append(txt_split)

            input_ids, mask_idx = tokenize_pet_txt(self.tokenizer, self.config, mask_txt_split_tuple[0],
                                                   mask_txt_split_tuple[1], mask_txt_split_tuple[2],
                                                   mask_txt_split_tuple[0], mask_txt_split_tuple[1],
                                                   mask_txt_split_tuple[2], txt_trim)
            list_input_ids.append(input_ids)

            max_num_lbl_tok = 0
            for idx, lbl in enumerate(self.label):
                num_lbl_tok = self.get_lbl_num_lbl_tok(lbl)
                if num_lbl_tok > max_num_lbl_tok:
                    max_num_lbl_tok = num_lbl_tok

            for i in range(self.num_lbl):
                list_mask_idx[b_idx, i, :max_num_lbl_tok] = range(mask_idx, mask_idx + max_num_lbl_tok)

        list_label = []
        for i in range(bs):
            list_label.append(self.label)

        return torch.tensor(list_input_ids).to(device), torch.tensor(list_mask_idx).to(device).long(), list_label

    def prepare_pet_mlm_batch(self, batch, mode="PET1"):

        '''
        Prepare for train

        :param batch:
        :return:
        '''

        list_list_txt = [] # [num_text, bs]
        for i in range(1, self.text_ctr):
            list_list_txt.append(batch["input"]["TEXT%d" % i])

        bs = len(batch["input"]["TEXT1"])

        prep_lbl = np.random.randint(self.num_lbl, size=bs)
        tgt = torch.from_numpy(prep_lbl).long() == batch["output"]["lbl"]

        list_orig_input_ids = []
        list_masked_input_ids = []

        txt_trim = 1

        for b_idx, lbl in enumerate(prep_lbl):
            txt_split_tuple = []

            for idx, txt_split in enumerate(self.pattern):
                for i in range(1, self.text_ctr):
                    txt_split = txt_split.replace("[TEXT%d]" % i, list_list_txt[i-1][b_idx])
                txt_split_inp = txt_split.replace("[LBL]", self.label[lbl])
                txt_split_tuple.append(txt_split_inp)

            orig_input_ids, masked_input_ids, mask_idx = tokenize_pet_mlm_txt(self.tokenizer, self.config, txt_split_tuple[0], txt_split_tuple[1], txt_split_tuple[2], txt_trim)
            list_orig_input_ids.append(orig_input_ids)
            list_masked_input_ids.append(masked_input_ids)

        return torch.tensor(list_orig_input_ids).to(device),  torch.tensor(list_masked_input_ids).to(device), prep_lbl, tgt.to(device)

    def prepare_eval_pet_batch(self, batch, mode="PET1"):
        return self.prepare_pet_batch(batch, mode)

    def store_test_lbl(self, list_idx, pred_lbl, true_lbl, logits):
        self.list_true_lbl.append(pred_lbl)

    def flush_file(self, write_file):
        self.list_true_lbl = torch.cat(self.list_true_lbl, dim=0).cpu().int().numpy().tolist()

        read_file = self._get_file("test")

        reverse_dict = {idx: lbl for idx, lbl in enumerate(self.config.dict_verbalizer.keys())}

        with open(read_file, 'r') as f_in:
            for ctr, line in enumerate(f_in.readlines()):
                answer_dict = {}
                answer_dict["idx"] = ctr
                pred_lbl = self.list_true_lbl[ctr]

                answer = reverse_dict[pred_lbl]
                answer_dict["label"] = answer

                write_file.write(json.dumps(answer_dict) + "\n")
