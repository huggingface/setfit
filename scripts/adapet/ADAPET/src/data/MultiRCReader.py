import os
import random
import torch
import json
import itertools
from collections import defaultdict

import numpy as np

from src.utils.util import device
from src.data.tokenize import tokenize_pet_txt, tokenize_pet_mlm_txt

class MultiRCReader(object):
    '''
    MultiRC reads MultiRC dataset
    '''

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.num_lbl = 2

        self.list_idx = []
        self.list_lbl = []

        self.pet_labels = [["no", "yes"], ["false", "true"]]
        self.pet_patterns = [["[PARAGRAPH]", ". Question: [QUESTION] ?  Is it [ANSWER] ? {}. [SEP]".format(self.tokenizer.mask_token), ""],
                             ["[PARAGRAPH]", ". Question: [QUESTION] ? Is the correct answer \" [ANSWER] \" ? {}. [SEP]".format(self.tokenizer.mask_token), ""],
                             ["[PARAGRAPH]", ". Based on the previous passage, [QUESTION] ? Is \" [ANSWER] \" a correct answer ? {}. [SEP]".format(self.tokenizer.mask_token), ""]]
        self.pet_pvps = list(itertools.product(self.pet_patterns, self.pet_labels))
        self._num_pets = len(self.pet_pvps)
        self._pet_names = ["PET{}".format(i+1) for i in range(self._num_pets)]

    def _get_file(self, split):
        '''
        Get filename of split

        :param split:
        :return:
        '''
        if split.lower() == "train":
            file = os.path.join("data", "fewglue", "MultiRC", "train.jsonl")
        elif split.lower() == "dev":
            file = os.path.join("data", "superglue", "MultiRC", "val.jsonl")
        elif split.lower() == "test":
            file = os.path.join("data", "superglue", "MultiRC", "test.jsonl")
        elif split.lower() == "unlabeled":
            file = os.path.join("data", "fewglue", "MultiRC", "unlabeled.jsonl")
        elif split.lower() == "val":
            file = os.path.join("data", "fewglue", "MultiRC", "val.jsonl")
        else:
            raise ValueError("Invalid split: %s" % split)
        return file


    def get_num_lbl_tok(self):
        return 1

    def read_dataset(self, split=None, is_eval=False):
        '''
        Read the dataset

        :param split: partition of the dataset
        :param is_eval:
        '''

        file = self._get_file(split)
        data = []

        with open(file, 'r') as f_in:
            for line in f_in.readlines():
                json_string = json.loads(line)
                json_string_passage = json_string["passage"]

                idx = json_string["idx"]
                passage = json_string_passage["text"]

                for qas in json_string_passage["questions"]:
                    question = qas["question"]
                    qas_idx = qas["idx"]
                    list_answers = qas["answers"]

                    for json_answers in list_answers:
                        answer = json_answers["text"]
                        dict_input = {"idx": qas_idx, "passage": passage, "question": question, "answer": str(answer)}

                        if "label" in json_answers:
                            lbl = json_answers["label"]
                        else:
                            lbl = -1
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
        list_question = batch["input"]["question"]
        list_passage = batch["input"]["passage"]
        list_answer = batch["input"]["answer"]

        list_input_ids = []
        bs = len(batch["input"]["question"])
        list_mask_idx = np.ones((bs, self.get_num_lbl_tok())) * self.config.max_text_length

        pattern, label = self.pet_pvps[self._pet_names.index(mode)]

        for b_idx, (p, q, a) in enumerate(zip(list_passage, list_question, list_answer)):
            mask_txt_split_tuple = []
            txt_trim = -1

            for idx, txt_split in enumerate(pattern):
                mask_txt_split_inp = txt_split.replace("[PARAGRAPH]", p).replace("[QUESTION]", q).replace("[ANSWER]", a)
                mask_txt_split_tuple.append(mask_txt_split_inp)

                # Trim the paragraph
                if "[PARAGRAPH]" in txt_split:
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

        list_question = batch["input"]["question"]
        list_passage = batch["input"]["passage"]
        list_answer = batch["input"]["answer"]

        bs = len(batch["input"]["answer"])

        prep_lbl = np.random.randint(self.num_lbl, size=bs)
        tgt = torch.from_numpy(prep_lbl).long() == batch["output"]["lbl"]

        pattern, label = self.pet_pvps[self._pet_names.index(mode)]

        list_orig_input_ids = []
        list_masked_input_ids = []

        for b_idx, (p, q, a, lbl) in enumerate(zip(list_passage, list_question, list_answer, prep_lbl)):
            txt_split_tuple = []

            txt_trim = -1

            for idx, txt_split in enumerate(pattern):
                txt_split_inp = txt_split.replace("[PARAGRAPH]", p).replace("[QUESTION]", q).replace("[ANSWER]", a).replace("[MASK]",
                                                                                                     label[lbl])
                txt_split_tuple.append(txt_split_inp)

                # Trim the paragraph
                if "[PARAGRAPH]" in txt_split:
                    txt_trim = idx

            orig_input_ids, masked_input_ids, mask_idx = tokenize_pet_mlm_txt(self.tokenizer, self.config, txt_split_tuple[0], txt_split_tuple[1], txt_split_tuple[2], txt_trim)
            list_orig_input_ids.append(orig_input_ids)
            list_masked_input_ids.append(masked_input_ids)

        return torch.tensor(list_orig_input_ids).to(device),  torch.tensor(list_masked_input_ids).to(device), prep_lbl, tgt.to(device)

    def prepare_eval_pet_batch(self, batch, mode="PET1"):
        return self.prepare_pet_batch(batch, mode)

    def store_test_lbl(self, list_idx, pred_lbl, true_lbl, logits):
        self.list_idx.append(list_idx)
        self.list_lbl.append(pred_lbl)

    def flush_file(self, write_file):
        read_file = self._get_file("test")

        self.list_idx = [item for sublist in self.list_idx for item in sublist]
        self.list_lbl = torch.cat(self.list_lbl, dim=0).cpu().numpy().tolist()

        with open(read_file, 'r') as f_in:
            qas_ctr = 0
            ans_ctr = 0

            for i, line in enumerate(f_in.readlines()):
                json_string = json.loads(line)
                json_string_passage = json_string["passage"]

                pas_ctr = json_string["idx"]

                list_questions = []

                for qas in json_string_passage["questions"]:
                    list_answers = qas["answers"]

                    list_pred_answers = []

                    for answer in list_answers:
                        pred_lbl = self.list_lbl[ans_ctr]
                        list_pred_answers.append({"idx": ans_ctr, "label": pred_lbl})

                        ans_ctr += 1

                    list_questions.append({"idx": qas_ctr, "answers": list_pred_answers})

                    qas_ctr += 1
                line_dict = {"idx": pas_ctr, "passage": {"questions": list_questions}}
                write_file.write(json.dumps(line_dict) + "\n")
