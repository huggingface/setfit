
import os
import json
import random
import torch
import numpy as np
from collections import defaultdict

from src.utils.util import device
from src.data.tokenize import tokenize_pet_txt, tokenize_pet_mlm_txt

class WiCReader(object):
    '''
    WiCReaer reads WiC dataset
    '''

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.num_lbl = 2
        self.list_true_lbl = []

        self.pet_labels = [["yes", "no"], ["yes", "no"], ["b", "2"], ["similar", "different"]]
        self.pet_patterns = [["\"[SENTENCE1]\" / \"[SENTENCE2]\".", "Similar sense of \"[WORD]\"? {}.".format(self.tokenizer.mask_token), ""],
                             ["[SENTENCE1] [SENTENCE2]", "Does [WORD] have the same meaing in both sentences? {}.".format(self.tokenizer.mask_token), ""],
                             ["[WORD].", "Sense (1) (a) '[SENTENCE1]' ({}) '[SENTENCE2]'".format(self.tokenizer.mask_token), ""]]

        self.pet_pvps = list(zip(self.pet_patterns, self.pet_labels))
        self._num_pets = len(self.pet_pvps)
        self._pet_names = ["PET{}".format(i+1) for i in range(self._num_pets)]

        self.list_true_lbl = []

        self.dict_inv_freq = defaultdict(int)
        self.tot_doc = 0

        self.dict_lbl_2_idx = {True: 0, False: 1}


    def _get_file(self, split):
        '''
        Get filename of split
        :param split:
        :return:
        '''
        if split.lower() == "train":
            file = os.path.join("data", "fewglue", "WiC", "train.jsonl")
        elif split.lower() == "dev":
            file = os.path.join("data", "superglue", "WiC", "val.jsonl")
        elif split.lower() == "test":
            file = os.path.join("data", "superglue", "WiC", "test.jsonl")
        elif split.lower() == "unlabeled":
            file = os.path.join("data", "fewglue", "WiC", "unlabeled.jsonl")
        elif split.lower() == "val":
            file = os.path.join("data", "fewglue", "WiC", "val.jsonl")
        else:
            raise ValueError("Invalid split: %s" % split)
        return file

    def get_num_lbl_tok(self):
        return 1

    def read_dataset(self, split=None, is_eval=False):
        '''
        Read the dataset
        :param split: partition of the
        '''

        file = self._get_file(split)
        data = []

        with open(file, 'r') as f_in:
            for line in f_in.readlines():
                json_string = json.loads(line)

                idx = json_string["idx"]
                if "word" not in json_string:
                    import ipdb; ipdb.set_trace()

                word = json_string["word"]
                sentence1 = json_string["sentence1"]
                sentence2 = json_string["sentence2"]

                if "label" in json_string:
                    lbl = self.dict_lbl_2_idx[json_string["label"]]
                else:
                    lbl = -1

                dict_input = {"idx": idx, "word": word, "sentence1": sentence1, "sentence2": sentence2}
                dict_output = {"lbl": lbl}
                dict_input_output = {"input": dict_input, "output": dict_output}
                data.append(dict_input_output)

        data = np.asarray(data)
        return data

    @property
    def pets(self):
        return self._pet_names

    def prepare_pet_batch(self, batch, mode="PET1"):
        '''
        Prepare for train

        :param batch:
        :return:
        '''
        list_sentence1 = batch["input"]["sentence1"]
        list_sentence2 = batch["input"]["sentence2"]
        list_word = batch["input"]["word"]

        list_input_ids = []
        bs = len(batch["input"]["sentence1"])
        list_mask_idx = np.ones((bs, self.get_num_lbl_tok())) * self.config.max_text_length

        pattern, label = self.pet_pvps[self._pet_names.index(mode)]

        for b_idx, (s1, s2, w) in enumerate(zip(list_sentence1, list_sentence2, list_word)):
            mask_txt_split_tuple = []
            txt_trim = -1

            for idx, txt_split in enumerate(pattern):
                mask_txt_split_inp = txt_split.replace("[SENTENCE1]", s1).replace("[SENTENCE2]", s2).replace("[WORD]", w)
                mask_txt_split_tuple.append(mask_txt_split_inp)

                # Trim the paragraph
                if "[SENTENCE1]" in txt_split:
                    txt_trim = idx

            input_ids, mask_idx = tokenize_pet_txt(self.tokenizer, self.config, mask_txt_split_tuple[0],
                                                   mask_txt_split_tuple[1], mask_txt_split_tuple[2],
                                                   mask_txt_split_tuple[0], mask_txt_split_tuple[1],
                                                   mask_txt_split_tuple[2], txt_trim)
            list_input_ids.append(input_ids)
            list_mask_idx[b_idx, :self.get_num_lbl_tok()] = range(mask_idx, mask_idx + self.get_num_lbl_tok())

        return torch.tensor(list_input_ids).to(device), torch.tensor(list_mask_idx).to(device), label

    def prepare_pet_mlm_batch(self, batch, mode="PET1"):
        '''
        Prepare for train

        :param batch:
        :return:
        '''

        list_sentence1 = batch["input"]["sentence1"]
        list_sentence2 = batch["input"]["sentence2"]
        list_word = batch["input"]["word"]

        bs = len(batch["input"]["sentence1"])

        prep_lbl = np.random.randint(self.num_lbl, size=bs)
        tgt = torch.from_numpy(prep_lbl).long() == batch["output"]["lbl"]

        pattern, label = self.pet_pvps[self._pet_names.index(mode)]

        list_orig_input_ids = []
        list_masked_input_ids = []

        for b_idx, (s1, s2, w, lbl) in enumerate(zip(list_sentence1, list_sentence2, list_word, prep_lbl)):
            txt_split_tuple = []

            txt_trim = -1

            for idx, txt_split in enumerate(pattern):
                txt_split_inp = txt_split.replace("[SENTENCE1]", s1).replace("[SENTENCE2]", s2).replace("[WORD]", w).replace("[MASK]",
                                                                                                     label[lbl])
                txt_split_tuple.append(txt_split_inp)

                # Trim the paragraph
                if "[SENTENCE1]" in txt_split:
                    txt_trim = idx

            orig_input_ids, masked_input_ids, mask_idx = tokenize_pet_mlm_txt(self.tokenizer, self.config, txt_split_tuple[0], txt_split_tuple[1], txt_split_tuple[2], txt_trim)
            list_orig_input_ids.append(orig_input_ids)
            list_masked_input_ids.append(masked_input_ids)

        return torch.tensor(list_orig_input_ids).to(device),  torch.tensor(list_masked_input_ids).to(device), prep_lbl, tgt.to(device)

    def prepare_eval_pet_batch(self, batch, mode="PET1"):
        return self.prepare_pet_batch(batch, mode)

    def store_test_lbl(self, list_idx, pred_lbl, true_lbl, logits):
        self.list_true_lbl.append(pred_lbl > 0.5)

    def flush_file(self, write_file):
        self.list_true_lbl = torch.cat(self.list_true_lbl, dim=0).cpu().int().numpy().tolist()

        read_file = self._get_file("test")

        with open(read_file, 'r') as f_in:
            for ctr, line in enumerate(f_in.readlines()):
                answer_dict = {}
                answer_dict["idx"] = ctr
                pred_lbl = self.list_true_lbl[ctr]

                if pred_lbl == 0:
                    answer = "true"
                else:
                    answer = "false"
                answer_dict["label"] = answer

                write_file.write(json.dumps(answer_dict) + "\n")
