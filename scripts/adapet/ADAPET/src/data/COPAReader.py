import os
import json

import numpy as np
import torch
import random
from collections import defaultdict

from src.data.tokenize import tokenize_pet_txt, tokenize_pet_mlm_txt
from src.utils.util import device


class COPAReader(object):
    '''
    COPAReader reads COPA dataset
    '''

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.num_lbl = 2

        self.list_true_lbl = []

        self.pet_patterns_effect = [["\" [CHOICE1] \" or \" [CHOICE2] \"?", " [PREMISE] ", ", so [MASK]. [SEP]"],
                                    ["[CHOICE1] or [CHOICE2] ?", " [PREMISE] ", ", so [MASK]. [SEP]"],
                                    ["Because [PREMISE] ,",  " [MASK]. [SEP]", ""]]

        self.pet_patterns_cause = [["\" [CHOICE1] \" or \" [CHOICE2] \"?", " [PREMISE] ", "because [MASK]. [SEP]"],
                                   ["[CHOICE1] or [CHOICE2] ?", " [PREMISE] ", "because [MASK]. [SEP]"],
                                   ["Because [MASK] ,", "  [PREMISE]. [SEP]", ""]]

        # Different lbl for each pattern
        self.pet_pvps = self.pet_patterns_effect
        self._num_pets = len(self.pet_pvps)
        self._pet_names = ["PET{}".format(i+1) for i in range(self._num_pets)]

    def _get_file(self, split):
        '''
        Get filename of split

        :param split:
        :return:
        '''
        if split.lower() == "train":
            file = os.path.join("data", "fewglue", "COPA", "train.jsonl")
        elif split.lower() == "dev":
            file = os.path.join("data", "superglue", "COPA", "val.jsonl")
        elif split.lower() == "test":
            file = os.path.join("data", "superglue", "COPA", "test.jsonl")
        else:
            raise ValueError("Invalid split: %s" % split)
        return file

    def get_num_lbl_tok(self):
        return 20


    def read_dataset(self, split=None, is_eval=False):
        '''
        Read the original dataset

        :param split: partition of the dataset
        :param is_eval:
        '''

        file = self._get_file(split)
        data = []

        with open(file, 'r') as f_in:
            for line in f_in.readlines():
                json_string = json.loads(line)

                premise = json_string["premise"]
                choice1 = json_string["choice1"]
                choice2 = json_string["choice2"]
                question = json_string["question"]
                idx = json_string["idx"]

                if "label" in json_string:
                    lbl = json_string["label"]
                else:
                    lbl = -1

                dict_input = {"premise": premise, "choice1": choice1,
                              "idx": idx, "choice2": choice2, "question": question}
                dict_output = {"lbl": lbl}

                dict_input_output = {"input": dict_input, "output": dict_output}
                data.append(dict_input_output)

        if split == 'train' or split == 'unlabeled':
            mirror_data = []
            for dict_input_output in data:
                dict_input, dict_output = dict_input_output["input"], \
                                          dict_input_output["output"]
                mirror_dict_input = {
                                        "premise": dict_input["premise"],
                                        "choice1": dict_input["choice2"],
                                        "choice2": dict_input["choice1"],
                                        "idx": dict_input["idx"],
                                        "question": dict_input["question"]
                }
                mirror_dict_output = {"lbl": 1 if dict_output["lbl"] == 0 else 0}
                mirror_dict_input_output = {
                        "input": mirror_dict_input,
                        "output": mirror_dict_output
                }
                mirror_data.append(mirror_dict_input_output)

            data.extend(mirror_data)

        data = np.asarray(data)

        return data

    @property
    def pets(self):
        return self._pet_names

    def get_lbl_num_lbl_tok(self, lbl):
        num_lbl_tok = len(self.tokenizer(lbl, add_special_tokens=False)["input_ids"])
        #
        return min(num_lbl_tok, self.get_num_lbl_tok())

    def prepare_pet_batch(self, batch, mode="PET1"):
        '''
        Prepare for train

        :param batch:
        :return:
        '''
        list_premise = batch["input"]["premise"]
        list_choice1 = batch["input"]["choice1"]
        list_choice2 = batch["input"]["choice2"]
        list_question = batch["input"]["question"]
        list_lbl = batch["output"]["lbl"]

        list_input_ids = []
        bs = len(batch["input"]["choice2"])
        list_mask_idx = np.ones((bs, self.num_lbl, self.config.max_num_lbl_tok)) * self.config.max_text_length - 1
        list_lbl_choices = []

        for b_idx, (p, c1, c2, ques, lbl) in enumerate(zip(list_premise, list_choice1, list_choice2, list_question, list_lbl)):
            mask_txt_split_tuple = []

            trimmed_c1 = c1[:-1]
            trimmed_c2 = c2[:-1]

            c1_num_lbl_tok = self.get_lbl_num_lbl_tok(trimmed_c1)
            c2_num_lbl_tok = self.get_lbl_num_lbl_tok(trimmed_c2)

            if c1_num_lbl_tok < c2_num_lbl_tok:
                trimmed_c1 = " ".join(trimmed_c1.split(" ") + [self.tokenizer.pad_token] * (c2_num_lbl_tok - c1_num_lbl_tok))
            if c2_num_lbl_tok < c1_num_lbl_tok:
                trimmed_c2 = " ".join(trimmed_c2.split(" ") + [self.tokenizer.pad_token] * (c1_num_lbl_tok - c2_num_lbl_tok))

            max_num_c_lbl_tok = max(c1_num_lbl_tok, c2_num_lbl_tok)

            txt_trim = -1
            if ques == "cause":
                pet_pvps = self.pet_patterns_cause
            elif ques == "effect":
                pet_pvps = self.pet_patterns_effect
            pattern = pet_pvps[self._pet_names.index(mode)]

            for idx, txt_split in enumerate(pattern):
                mask_txt_split_inp = txt_split.replace("[PREMISE]", p[:-1]).replace("[CHOICE1]", c1[:-1]).replace("[CHOICE2]", c2[:-1]).replace("[MASK]",
                                                                                                    "[MASK] " * max_num_c_lbl_tok)
                mask_txt_split_tuple.append(mask_txt_split_inp)

                # Trim the paragraph
                if "[PREMISE]" in txt_split:
                    txt_trim = idx

            input_ids, mask_idx = tokenize_pet_txt(self.tokenizer, self.config, mask_txt_split_tuple[0],
                                                   mask_txt_split_tuple[1], mask_txt_split_tuple[2],
                                                   mask_txt_split_tuple[0], mask_txt_split_tuple[1],
                                                   mask_txt_split_tuple[2], txt_trim)
            list_input_ids.append(input_ids)
            list_mask_idx[b_idx, 0, :max_num_c_lbl_tok] = range(mask_idx, mask_idx + max_num_c_lbl_tok)
            list_mask_idx[b_idx, 1, :max_num_c_lbl_tok] = range(mask_idx, mask_idx + max_num_c_lbl_tok)

            list_lbl_choices.append([trimmed_c1, trimmed_c2])


        return torch.tensor(list_input_ids).to(device), torch.tensor(list_mask_idx).to(device), list_lbl_choices

    def prepare_pet_mlm_batch(self, batch, mode="PET1"):

        '''
        Prepare for train

        :param batch:
        :return:
        '''
        # Always use pattern 3 for COPA
        mode = "PET3"

        list_premise = batch["input"]["premise"]
        list_choice1 = batch["input"]["choice1"]
        list_choice2 = batch["input"]["choice2"]
        list_question = batch["input"]["question"]
        list_lbl = batch["output"]["lbl"]

        bs = len(batch["input"]["question"])

        prep_lbl = np.random.randint(self.num_lbl, size=bs)
        tgt = torch.from_numpy(prep_lbl).long() == batch["output"]["lbl"]

        list_orig_input_ids = []
        list_masked_input_ids = []

        for b_idx, (p, c1, c2, ques, lbl) in enumerate(zip(list_premise, list_choice1, list_choice2, list_question, list_lbl)):
            txt_split_tuple = []

            txt_trim = -1
            if ques == "cause":
                pet_pvps = self.pet_patterns_cause
            elif ques == "effect":
                pet_pvps = self.pet_patterns_effect
            pattern = pet_pvps[self._pet_names.index(mode)]

            if lbl.item() == 0:
                lbl_choice = c1[:-1]
            elif lbl.item() == 1:
                lbl_choice = c2[:-1]
            else:
                raise ValueError("Invalid Lbl")

            for idx, txt_split in enumerate(pattern):
                txt_split_inp = txt_split.replace("[PREMISE]", p[:-1]).replace("[CHOICE1]", c1[:-1]).replace("[CHOICE2]", c2[:-1]).replace("[MASK]",
                                                                                                     lbl_choice)
                txt_split_tuple.append(txt_split_inp)

                if lbl.item() == 0:
                    # Trim the paragraph
                    if "[PREMISE]" in txt_split:
                        txt_trim = idx
                elif lbl.item() == 1:
                    # Trim the paragraph
                    if "[PREMISE]" in txt_split:
                        txt_trim = idx
                else:
                    raise ValueError("Invalid Lbl")

            orig_input_ids, masked_input_ids, mask_idx = tokenize_pet_mlm_txt(self.tokenizer, self.config, txt_split_tuple[0], txt_split_tuple[1], txt_split_tuple[2], txt_trim)
            list_orig_input_ids.append(orig_input_ids)
            list_masked_input_ids.append(masked_input_ids)

        return torch.tensor(list_orig_input_ids).to(device),  torch.tensor(list_masked_input_ids).to(device), prep_lbl, tgt.to(device)

    def prepare_eval_pet_batch(self, batch, mode="PET1"):
        '''
        Prepare for train

        :param batch:
        :return:
        '''
        list_premise = batch["input"]["premise"]
        list_choice1 = batch["input"]["choice1"]
        list_choice2 = batch["input"]["choice2"]
        list_question = batch["input"]["question"]
        list_lbl = batch["output"]["lbl"]

        list_input_ids = []
        bs = len(batch["input"]["choice2"])
        assert bs == 1, "Evaluation is done only for batch size 1 for COPA"
        # list_mask_idx = np.ones((bs, self.num_lbl, self.config.max_num_lbl_tok)) * self.config.max_text_length - 1
        list_lbl_choices = []

        list_mask_idx = []
        for b_idx, (p, c1, c2, ques, lbl) in enumerate(zip(list_premise, list_choice1, list_choice2, list_question, list_lbl)):
            c1_num_lbl_tok = len(self.tokenizer(c1[:-1], add_special_tokens=False)["input_ids"])
            c2_num_lbl_tok = len(self.tokenizer(c2[:-1], add_special_tokens=False)["input_ids"])
            # import ipdb; ipdb.set_trace()
            num_lbl_toks = [c1_num_lbl_tok, c2_num_lbl_tok]
            list_mask_idx_lbls = []
            if ques == "cause":
                pet_pvps = self.pet_patterns_cause
            elif ques == "effect":
                pet_pvps = self.pet_patterns_effect
            pattern = pet_pvps[self._pet_names.index(mode)]

            for lbl_idx, num_lbl_tok in enumerate(num_lbl_toks):
                mask_txt_split_tuple = []
                txt_trim = -1
                for idx, txt_split in enumerate(pattern):
                    mask_txt_split_inp = txt_split.replace("[PREMISE]", p[:-1]).replace("[CHOICE1]", c1[:-1]).replace("[CHOICE2]", c2[:-1]).replace("[MASK]",
                                                                                                        "[MASK] " * num_lbl_tok)
                    mask_txt_split_tuple.append(mask_txt_split_inp)

                    # Trim the paragraph
                    if "[CHOICE1]" in txt_split:
                        txt_trim = idx

                input_ids, mask_idx = tokenize_pet_txt(self.tokenizer, self.config, mask_txt_split_tuple[0],
                                                       mask_txt_split_tuple[1], mask_txt_split_tuple[2],
                                                       mask_txt_split_tuple[0], mask_txt_split_tuple[1],
                                                       mask_txt_split_tuple[2], txt_trim)
                list_input_ids.append(input_ids)
                list_mask_idx_lbl = list(range(mask_idx, mask_idx + num_lbl_tok))
                list_mask_idx_lbls.append(list_mask_idx_lbl)

            list_mask_idx.append(list_mask_idx_lbls)
            list_lbl_choices.append([c1[:-1], c2[:-1]])

        return torch.tensor(list_input_ids).to(device), list_mask_idx, list_lbl_choices


    def store_test_lbl(self, list_idx, pred_lbl, true_lbl, logits):
        self.list_true_lbl.append(pred_lbl)

    def flush_file(self, write_file):
        self.list_true_lbl = torch.cat(self.list_true_lbl, dim=0).cpu().int().numpy().tolist()

        read_file = self._get_file("test")

        with open(read_file, 'r') as f_in:
            for ctr, line in enumerate(f_in.readlines()):
                answer_dict = {}
                answer_dict["idx"] = ctr
                pred_lbl = self.list_true_lbl[ctr]

                answer_dict["label"] = pred_lbl

                write_file.write(json.dumps(answer_dict) + "\n")
