import os
import json

import numpy as np
import torch
from collections import defaultdict

from src.data.tokenize import tokenize_pet_txt, tokenize_pet_mlm_txt
from src.utils.util import device
import random

class RecordReader(object):
    '''
    RecordReader reads Record dataset
    '''

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.num_lbl = self.config.max_num_lbl

        self.pet_patterns = [["[PASSAGE]", "[QUESTION]", ""]]
        self.dict_qas_idx2entity = {}

        # Different lbl for each pattern
        self.pet_pvps = self.pet_patterns
        self._num_pets = len(self.pet_pvps)
        self._pet_names = ["PET{}".format(i+1) for i in range(self._num_pets)]

    def _get_file(self, split):
        '''
        Get filename of split

        :param split:
        '''
        if split.lower() == "train":
            file = os.path.join("data", "fewglue", "ReCoRD", "train.jsonl")
        elif split.lower() == "dev":
            file = os.path.join("data", "superglue", "ReCoRD", "val.jsonl")
        elif split.lower() == "test":
            file = os.path.join("data", "superglue", "ReCoRD", "test.jsonl")
        return file

    def get_num_lbl_tok(self):
        return 20

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

                # Get dictionary mapping entity idx to entity
                dict_entity_idx_2_name = {}
                for entity in json_string_passage["entities"]:
                    start = entity["start"]
                    end = entity["end"]
                    word = passage[start:end + 1]
                    dict_entity_idx_2_name[(entity["start"], entity["end"])] = word

                for qas in json_string["qas"]:
                    question = qas["query"]
                    qas_idx = qas["idx"]

                    # If data has solution
                    if "answers" in qas:
                        list_answers = qas["answers"]
                        # Get dictionary of entities in answers
                        dict_entity_idx_2_sol = {}
                        for answer in list_answers:
                            start = answer["start"]
                            end = answer["end"]
                            text = answer["text"]
                            dict_entity_idx_2_sol[(start, end)] = text

                            # Get all unique false entities for margin
                            set_false_entities = set()
                            for (enty_idx, enty) in dict_entity_idx_2_name.items():
                                if enty_idx not in dict_entity_idx_2_sol.keys():
                                    set_false_entities.add(enty)
                            list_false_entities = list(set_false_entities)

                        # PET ensures each data gets exactly 1 true and the rest false during training
                        if split == "train" and not is_eval:
                            set_seen_enty = set()

                            for enty_idx, enty in dict_entity_idx_2_name.items():
                                # Create datapoints with each unique correct entity
                                if enty_idx in dict_entity_idx_2_sol:
                                    # Only see each entity once
                                    if enty not in set_seen_enty:
                                        list_sample_false_entities = random.sample(list_false_entities, min(len(list_false_entities), self.config.max_num_lbl-1))
                                        # Replace entity with [MASK]
                                        masked_question = question.replace("@placeholder", "[MASK]")
                                        set_seen_enty.add(enty)

                                        dict_input = {"idx": idx, "passage": passage, "question": masked_question, "true_entity": enty, "false_entities": list_sample_false_entities}
                                        dict_output = {"lbl": 0}
                                        dict_input_output = {"input": dict_input, "output": dict_output}
                                        data.append(dict_input_output)
                        # Construct evaluation sets with the lbl
                        else:
                            set_seen_enty = set()

                            for (enty_idx, enty) in dict_entity_idx_2_name.items():
                                if enty not in set_seen_enty:
                                    set_seen_enty.add(enty)

                            # Replace entity with [MASK]
                            masked_question = question.replace("@placeholder", "[MASK]")
                            # Compute label for evaluation
                            label = [0 if enty not in list(dict_entity_idx_2_sol.values()) else 1 for enty in set_seen_enty]

                            dict_input = {"idx": idx, "passage": passage, "question": masked_question,
                                            "candidate_entity": list(set_seen_enty)}
                            dict_output = {"lbl": label}
                            dict_input_output = {"input": dict_input, "output": dict_output}
                            data.append(dict_input_output)
                    else:
                        # Test set without labels
                        set_seen_enty = set()

                        for (enty_idx, enty) in dict_entity_idx_2_name.items():
                            if enty not in set_seen_enty:
                                set_seen_enty.add(enty)

                        # Replace entity with [MASK]
                        masked_question = question.replace("@placeholder", "[MASK]")

                        dict_input = {"idx": idx, "passage": passage, "question": masked_question,
                                        "candidate_entity": list(set_seen_enty), "qas_idx": qas_idx}
                        dict_output = {"lbl": -1}
                        dict_input_output = {"input": dict_input, "output": dict_output}
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
        list_passage = batch["input"]["passage"]
        list_question = batch["input"]["question"]
        list_true_entity = batch["input"]["true_entity"]
        list_false_entities = batch["input"]["false_entities"]
        list_lbl = batch["output"]["lbl"]

        bs = len(list_passage)

        assert(bs == 1)

        list_input_ids = []
        list_mask_idx = np.ones((bs, self.config.max_num_lbl, self.get_num_lbl_tok())) * self.config.max_text_length - 1
        list_lbl_choices = []

        for b_idx, (p, q, te, fe, lbl) in enumerate(zip(list_passage, list_question, list_true_entity, list_false_entities, list_lbl)):
            mask_txt_split_tuple = []

            true_num_lbl_tok = self.get_lbl_num_lbl_tok(te)
            max_num_lbl_tok = true_num_lbl_tok
            for idx, wrong_enty in enumerate(fe):
                num_lbl_tok = self.get_lbl_num_lbl_tok(wrong_enty)
                if num_lbl_tok > max_num_lbl_tok:
                    max_num_lbl_tok = num_lbl_tok

            txt_trim = -1
            pattern = self.pet_patterns[self._pet_names.index(mode)]

            for idx, txt_split in enumerate(pattern):
                mask_txt_split_inp = txt_split.replace("[PASSAGE]", p).replace("[QUESTION]", q + " [SEP]").replace("[MASK] ", "[MASK] " * max_num_lbl_tok).replace("@highlight", "-")
                mask_txt_split_tuple.append(mask_txt_split_inp)

                # Trim the paragraph
                if "[PASSAGE]" in txt_split:
                    txt_trim = idx

            input_ids, mask_idx = tokenize_pet_txt(self.tokenizer, self.config, mask_txt_split_tuple[0],
                                                   mask_txt_split_tuple[1], mask_txt_split_tuple[2],
                                                   mask_txt_split_tuple[0], mask_txt_split_tuple[1],
                                                   mask_txt_split_tuple[2], txt_trim)

            list_mask_idx[b_idx, 0, :true_num_lbl_tok] = range(mask_idx, mask_idx + true_num_lbl_tok)


            for idx, wrong_enty in enumerate(fe):
                num_lbl_tok = self.get_lbl_num_lbl_tok(wrong_enty)
                list_mask_idx[b_idx, (idx+1), :num_lbl_tok] = range(mask_idx, mask_idx + num_lbl_tok)


            list_input_ids.append(input_ids)
            candidates = [te]
            candidates.extend(fe)
            list_lbl_choices.append(candidates)

        return torch.tensor(list_input_ids).to(device), torch.tensor(list_mask_idx).to(device), list_lbl_choices


    def prepare_pet_mlm_batch(self, batch, mode="PET1"):
        '''
        Prepare for train
        :param batch:
        :return:
        '''

        list_passage = batch["input"]["passage"]
        list_question = batch["input"]["question"]
        list_true_entity = batch["input"]["true_entity"]
        list_false_entities = batch["input"]["false_entities"]
        list_lbl = batch["output"]["lbl"]

        bs = len(batch["input"]["passage"])

        prep_lbl = np.random.randint(self.num_lbl, size=bs)
        tgt = torch.from_numpy(prep_lbl).long() == batch["output"]["lbl"]

        list_orig_input_ids = []
        list_masked_input_ids = []

        for b_idx, (p, q, te, fe, lbl) in enumerate(zip(list_passage, list_question, list_true_entity, list_false_entities, list_lbl)):
            txt_split_tuple = []

            true_num_lbl_tok = self.get_lbl_num_lbl_tok(te)
            max_num_lbl_tok = true_num_lbl_tok
            for idx, wrong_enty in enumerate(fe):
                num_lbl_tok = self.get_lbl_num_lbl_tok(wrong_enty)
                if num_lbl_tok > max_num_lbl_tok:
                    max_num_lbl_tok = num_lbl_tok

            txt_trim = -1
            pattern = self.pet_patterns[self._pet_names.index(mode)]

            for idx, txt_split in enumerate(pattern):
                txt_split_inp = txt_split.replace("[PASSAGE]", p).replace("[QUESTION]", q + " [SEP]").replace("@highlight", "-")
                txt_split_tuple.append(txt_split_inp)

                # Trim the paragraph
                if "[PASSAGE]" in txt_split:
                    txt_trim = idx

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

        list_passage = batch["input"]["passage"]
        list_question = batch["input"]["question"]
        list_candidates = batch["input"]["candidate_entity"]

        bs = len(list_passage)

        assert(bs == 1)

        list_input_ids = []
        list_mask_idx = []

        for b_idx, (p, q, cands) in enumerate(zip(list_passage, list_question, list_candidates)):
            pattern = self.pet_patterns[self._pet_names.index(mode)]
            list_mask_idx_lbls = []

            for cand in cands:
                num_cnd_tok = len(self.tokenizer(cand, add_special_tokens=False)["input_ids"])
                mask_txt_split_tuple = []
                txt_trim = -1
                for idx, txt_split in enumerate(pattern):
                    mask_txt_split_inp = txt_split.replace("[PASSAGE]", p).replace("[QUESTION]", q + " [SEP]").replace("[MASK]", "[MASK] " * num_cnd_tok).replace("@highlight", "-")
                    mask_txt_split_tuple.append(mask_txt_split_inp)

                    # Trim the paragraph
                    if "[PASSAGE]" in txt_split:
                        txt_trim = idx

                input_ids, mask_idx = tokenize_pet_txt(self.tokenizer, self.config, mask_txt_split_tuple[0],
                                                       mask_txt_split_tuple[1], mask_txt_split_tuple[2],
                                                       mask_txt_split_tuple[0], mask_txt_split_tuple[1],
                                                       mask_txt_split_tuple[2], txt_trim)
                list_input_ids.append(input_ids)
                list_mask_idx_lbl = list(range(mask_idx, mask_idx + num_cnd_tok))
                list_mask_idx_lbls.append(list_mask_idx_lbl)

            list_mask_idx.append(list_mask_idx_lbls)

        return torch.tensor(list_input_ids).to(device), list_mask_idx, list_candidates

    def store_test_lbl(self, list_idx, pred_lbl, true_lbl, logits):
        self.dict_qas_idx2entity[list_idx[0].item()] =  true_lbl[0][pred_lbl[0].item()]

    def flush_file(self, write_file):
        for (idx, entity) in self.dict_qas_idx2entity.items():
            write_file.write(json.dumps({"idx": idx, "label": entity}) + "\n")
