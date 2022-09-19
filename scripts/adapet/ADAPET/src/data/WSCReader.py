import os
import json
import random
import warnings

import numpy as np
import torch
from collections import defaultdict

from src.data.tokenize import tokenize_pet_txt, tokenize_pet_mlm_txt
from src.utils.util import device


class WSCReader(object):
    '''
    WSC reads WSC dataset
    '''

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.num_lbl = 1
        self.list_true_lbl = []

        self.pet_patterns = [["[TEXT]", " The pronoun '*[NNP]*' refers to [MASK]. [SEP]", ""],
                             ["[TEXT]", " In the previous sentence, the pronoun '*[NNP]*' refers to [MASK]. [SEP]", ""],
                             ["[TEXT]", " Question: In the passage above, what does the pronoun '*[NNP]*' refer to? Answer:  [MASK]. [SEP]", ""]]

        # Different lbl for each pattern
        self.pet_pvps = self.pet_patterns
        self._num_pets = len(self.pet_pvps)
        self._pet_names = ["PET{}".format(i+1) for i in range(self._num_pets)]


    def _get_file(self, split):
        '''
        Get filename of split

        :param split:
        :return:
        '''
        if split.lower() == "train":
            file = os.path.join("data", "fewglue", "WSC", "train.jsonl")
        elif split.lower() == "dev":
            file = os.path.join("data", "superglue", "WSC", "val.jsonl")
        elif split.lower() == "test":
            file = os.path.join("data", "superglue", "WSC", "test.jsonl")
        else:
            raise ValueError("Invalid split: %s" % split)
        return file

    def get_num_lbl_tok(self):
        return 20

    def read_dataset(self, split=None, is_eval=False):
        '''
        Read the original dataset

        :param split: partition of the
        '''

        file = self._get_file(split)
        data = []

        with open(file, 'r') as f_in:
            for line in f_in.readlines():
                json_string = json.loads(line)

                text = json_string["text"]
                pronoun, pronoun_idx = json_string["target"]["span2_text"], \
                                       json_string["target"]["span2_index"]
                noun, noun_idx = json_string["target"]["span1_text"], \
                                 json_string["target"]["span1_index"]
                idx = json_string["idx"]

                if "label" in json_string:
                    lbl = json_string["label"]
                else:
                    lbl = -1

                words_text = text.split()
                words_lower = text.lower().split()
                words_noun = noun.lower().split()
                noun_len = len(words_noun)

                if words_lower[noun_idx:noun_idx + noun_len] != words_noun:
                    for offset in [-1, +1]:
                        if words_lower[noun_idx + offset:noun_idx + noun_len + offset] == words_noun:
                            noun_idx += offset


                if words_lower[noun_idx:noun_idx + noun_len] != words_noun:
                    warnings.warn(f"Got '{words_lower[noun_idx:noun_idx + noun_len]}' but expected "
                                   f"'{words_noun}' at index {noun_idx} for '{words_text}'")

                if words_text[pronoun_idx] != pronoun:
                    for offset in [-1, +1]:
                        if words_text[pronoun_idx + offset] == pronoun:
                            pronoun_idx += offset

                    if words_text[pronoun_idx] != pronoun and words_text[pronoun_idx].startswith(pronoun):
                        words_text = words_text[:pronoun_idx] \
                                  + [words_text[pronoun_idx][:len(pronoun)], words_text[pronoun_idx][len(pronoun):]] \
                                  + words_text[pronoun_idx + 1:]

                assert words_text[pronoun_idx] == pronoun, \
                    f"Got '{words_text[pronoun_idx]}' but expected '{pronoun}' at index {pronoun_idx} for '{words_text}'"

                orig_text = ' '.join(words_text)
                words_text[pronoun_idx] = '*' + words_text[pronoun_idx] + '*'
                text = ' '.join(words_text)

                len_noun = max(len(self.tokenizer(words_text[noun_idx], add_special_tokens=False)["input_ids"]), 1)
                len_pronoun = max(len(self.tokenizer(orig_text[pronoun_idx], add_special_tokens=False)["input_ids"]), 1)

                dict_input = {"text": text, "pronoun": pronoun, "orig_text": orig_text,
                              "idx": idx, "noun": noun, "pronoun_idx_first": pronoun_idx < noun_idx, "len_noun": len_noun, "len_pronoun": len_pronoun}

                dict_output = {"lbl": lbl}
                dict_input_output = {"input": dict_input, "output": dict_output}

                if split == 'train' and lbl != True:
                    continue
                data.append(dict_input_output)

        data = np.asarray(data)
        return data

    @property
    def pets(self):
        return self._pet_names

    def get_lbl_num_lbl_tok(self, lbl):
        num_lbl_tok = len(self.tokenizer(lbl, add_special_tokens=False)["input_ids"])
        return min(num_lbl_tok, self.get_num_lbl_tok())

    def prepare_pet_batch(self, batch, mode="PET1"):
        '''
        Prepare for train

        :param batch:
        :return:
        '''

        list_text = batch["input"]["text"]

        list_pronoun = batch["input"]["pronoun"]
        list_noun = batch["input"]["noun"]
        list_lbl = batch["output"]["lbl"]

        list_input_ids = []
        bs = len(batch["input"]["text"])
        list_mask_idx = np.ones((bs, self.num_lbl, self.config.max_num_lbl_tok)) * self.config.max_text_length - 1
        list_lbl_choices = []

        for b_idx, (t, p, n, lbl) in enumerate(zip(list_text, list_pronoun, list_noun, list_lbl)):
            mask_txt_split_tuple = []
            noun_num_lbl_tok = self.get_lbl_num_lbl_tok(n)
            num_lbl_tok = min(noun_num_lbl_tok + random.randint(0,3), self.config.max_num_lbl_tok) # random.randint(0,3)
            txt_trim = -1
            pattern = self.pet_patterns[self._pet_names.index(mode)]

            for idx, txt_split in enumerate(pattern):
                mask_txt_split_inp = txt_split.replace("[TEXT]", t).replace("[NNP]", p).replace("[MASK]", "[MASK] " * num_lbl_tok)
                mask_txt_split_tuple.append(mask_txt_split_inp)

                # Trim the paragraph
                if "[TEXT]" in txt_split:
                    txt_trim = idx

            input_ids, mask_idx = tokenize_pet_txt(self.tokenizer, self.config, mask_txt_split_tuple[0],
                                                   mask_txt_split_tuple[1], mask_txt_split_tuple[2],
                                                   mask_txt_split_tuple[0], mask_txt_split_tuple[1],
                                                   mask_txt_split_tuple[2], txt_trim)
            list_input_ids.append(input_ids)
            list_mask_idx[b_idx, 0, :num_lbl_tok] = range(mask_idx, mask_idx + num_lbl_tok)

            lbl_mask = n.split() + [self.tokenizer.pad_token] * (num_lbl_tok - noun_num_lbl_tok)
            list_lbl_choices.append([' '.join(lbl_mask)])


        return torch.tensor(list_input_ids).to(device), torch.tensor(list_mask_idx).to(device), list_lbl_choices

    
    def prepare_pet_mlm_batch(self, batch, mode="PET1"):
        '''
        Prepare for train

        :param batch:
        :return:
        '''
        list_text = batch["input"]["text"]
        list_pronoun = batch["input"]["pronoun"]
        list_noun = batch["input"]["noun"]
        list_lbl = batch["output"]["lbl"]

        list_orig_input_ids = []
        list_masked_input_ids = []

        tgt = torch.tensor([1.]).long()

        for b_idx, (t, p, n, lbl) in enumerate(zip(list_text, list_pronoun, list_noun, list_lbl)):
            txt_trim = -1
            pattern = self.pet_patterns[self._pet_names.index(mode)]
            txt_split_tuple = []

            for idx, txt_split in enumerate(pattern):

                txt_split_inp = txt_split.replace("[TEXT]", t).replace("[NNP]", p).replace("[MASK]", n)
                txt_split_tuple.append(txt_split_inp)
                
                if "[TEXT]" in txt_split:
                    txt_trim = idx

            orig_input_ids, masked_input_ids, mask_idx = tokenize_pet_mlm_txt(self.tokenizer, self.config, txt_split_tuple[0], txt_split_tuple[1], txt_split_tuple[2], txt_trim)
            list_orig_input_ids.append(orig_input_ids)
            list_masked_input_ids.append(masked_input_ids)

        return torch.tensor(list_orig_input_ids).to(device), torch.tensor(list_masked_input_ids).to(device), None, tgt.to(device)
   
    def prepare_eval_pet_batch(self, batch, mode="PET1"):
        '''
        Prepare for train
        :param batch:
        :return:
        '''
        list_text = batch["input"]["text"]

        list_pronoun = batch["input"]["pronoun"]
        list_noun = batch["input"]["noun"]
        list_lbl = batch["output"]["lbl"]

        list_input_ids = []
        list_mask_idx = []
        list_lbl_choices = []

        for b_idx, (t, p, n, lbl) in enumerate(zip(list_text, list_pronoun, list_noun, list_lbl)):
            
            mask_txt_split_tuple = []
            noun_num_lbl_tok = self.get_lbl_num_lbl_tok(n)
            num_lbl_tok = min(noun_num_lbl_tok + 1, self.config.max_num_lbl_tok) # random.randint(0,3)
            txt_trim = -1
            pattern = self.pet_patterns[self._pet_names.index(mode)]

            for idx, txt_split in enumerate(pattern):
                mask_txt_split_inp = txt_split.replace("[TEXT]", t).replace("[NNP]", p).replace("[MASK]", "[MASK] " * num_lbl_tok)
                mask_txt_split_tuple.append(mask_txt_split_inp)

                # Trim the paragraph
                if "[TEXT]" in txt_split:
                    txt_trim = idx

            input_ids, mask_idx = tokenize_pet_txt(self.tokenizer, self.config, mask_txt_split_tuple[0],
                                                   mask_txt_split_tuple[1], mask_txt_split_tuple[2],
                                                   mask_txt_split_tuple[0], mask_txt_split_tuple[1],
                                                   mask_txt_split_tuple[2], txt_trim)
            list_input_ids.append(input_ids)
            list_mask_idx.append(list(range(mask_idx, mask_idx + num_lbl_tok)))
            list_lbl_choices.append([n])

        return torch.tensor(list_input_ids).to(device), torch.tensor(list_mask_idx).to(device), list_lbl_choices

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

                if pred_lbl == 1:
                    answer = "true"
                else:
                    answer = "false"
                answer_dict["label"] = answer

                write_file.write(json.dumps(answer_dict) + "\n")
