import os
import math
import random
from functools import reduce
import torch
import torch.nn as nn
import re
import numpy as np
from transformers import *
from src.utils.util import device
os.environ["WANDB_DISABLED"] = "true"


class adapet(torch.nn.Module):
    def __init__(self, config, tokenizer, dataset_reader):
        '''
        ADAPET model

        :param config
        '''
        super(adapet, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_reader = dataset_reader

        pretrained_file = os.path.join("pretrained_models", self.config.pretrained_weight)
        if not os.path.exists(pretrained_file):
            pretrained_file = self.config.pretrained_weight

        if "albert" in pretrained_file:
            albert_config = AlbertConfig.from_pretrained(pretrained_file)
            self.model = AlbertForMaskedLM.from_pretrained(pretrained_file, config=albert_config)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(pretrained_file)


        self.num_lbl = self.dataset_reader.get_num_lbl()

        # Mask Idx Lkup hack to compute the loss at mask positions
        init_mask_idx_lkup = torch.cat([torch.eye(self.config.max_text_length), torch.zeros((1, self.config.max_text_length))], dim=0)
        self.mask_idx_lkup = nn.Embedding.from_pretrained(init_mask_idx_lkup) # [max_text_length+1, max_text_length]
        self.num_lbl = self.dataset_reader.get_num_lbl()

        self.lbl_idx_lkup = nn.Embedding.from_pretrained(torch.eye(self.num_lbl)) # [num_lbl, num_lbl]

        self.loss = nn.BCELoss(reduction="none")

        # Setup patterns depending on if random or not
        self.pattern_list = self.dataset_reader.dataset_reader.pets
        if config.pattern_idx == "random":
            self.pattern = lambda: random.choice(self.pattern_list)
        else:
            assert config.pattern_idx > 0 and config.pattern_idx <= len(self.pattern_list), "This dataset has {} patterns".format(len(self.pattern_list))
            self.pattern = self.pattern_list[config.pattern_idx-1]

    def get_single_logits(self,  pet_mask_ids, mask_idx, list_lbl):
        '''
        Get decoupled label logits at mask positions for single mask tokens

        :param pet_mask_ids:
        :param mask_idx:
        :param list_lbl: [num_lbl]
        :return:
        '''
        bs = pet_mask_ids.shape[0]

        # Get ids for lbls
        lbl_ids = np.ones((self.num_lbl, self.config.max_num_lbl_tok)) * self.tokenizer.pad_token_id
        for i, lbl in enumerate(list_lbl):
            i_lbl_ids = self.tokenizer(lbl, add_special_tokens=False)["input_ids"]
            lbl_ids[i, :len(i_lbl_ids)] = i_lbl_ids
        lbl_ids = torch.tensor(lbl_ids).to(device).long()  # [num_lbl, max_num_lbl_tok]

        # Trick to compute lbl at mask positions
        with torch.no_grad():
            mask_idx_emb = self.mask_idx_lkup(mask_idx.long())  # [bs, max_num_lbl_tok, max_seq_len]

        # Get probability for each vocab token at the mask position
        pet_logits = self.model(pet_mask_ids, (pet_mask_ids > 0).long())[0]  # [bs, max_seq_len, vocab_size]
        pet_mask_logits = torch.matmul(mask_idx_emb[:, :, None, :], pet_logits[:, None, :, :]).squeeze(
            2)  # [bs, max_num_lbl_tok, vocab_size]
        pet_mask_rep_vocab_prob = pet_mask_logits.softmax(dim=2)  # [bs, max_num_lbl_tok, vocab_size]

        bs_by_max_num_lbl_tok = list(pet_mask_logits.shape[:2])
        # Get probability for correct label at mask position
        mask_prob = torch.gather(pet_mask_rep_vocab_prob, 2, lbl_ids.view(1, -1).unsqueeze(1).repeat(
            bs_by_max_num_lbl_tok + [1]))  # [bs, max_num_lbl_tok, max_num_lbl_tok*num_lbl]
        mask_prob = mask_prob.transpose(1, 2)  # [bs,  max_num_lbl_tok*num_lbl, num_lbl_tok]
        mask_prob = mask_prob.reshape(bs, self.num_lbl, self.config.max_num_lbl_tok, self.config.max_num_lbl_tok)
        mask_diag_prob = torch.diagonal(mask_prob, dim1=2, dim2=3)  # [bs, num_lbl, num_lbl_tok]

        # Sum label probabilities across multiple positions to get label probability (will always be 1)
        lbl_prob = torch.sum(mask_diag_prob, dim=2)  # [bs, num_lbl]

        return lbl_prob

    def get_multilbl_logits(self, pet_mask_ids, mask_idx, batch_list_lbl ):
        '''
        Get decoupled label logits at mask positions for multiple mask tokens

        :param batch:
        :return:
        '''
        bs = pet_mask_ids.shape[0]
        num_lbl, max_num_lbl_tok = mask_idx.shape[1:]
        lbl_ids = np.zeros((bs, self.num_lbl, self.config.max_num_lbl_tok)) # [bs, num_lbl, max_num_lbl_tok]

        # Get lbl ids for multi token labels
        for i, list_lbl in enumerate(batch_list_lbl):
            for j, lbl in enumerate(list_lbl):
                i_j_lbl_ids = self.tokenizer(lbl, add_special_tokens=False)["input_ids"]
                lbl_ids[i, j, :len(i_j_lbl_ids)] = i_j_lbl_ids[:min(self.config.max_num_lbl_tok, len(i_j_lbl_ids))]
        lbl_ids = torch.from_numpy(lbl_ids).to(device)

        # Get probability for each vocab token at the mask position
        pet_logits = self.model(pet_mask_ids, (pet_mask_ids>0).long())[0] # [bs, max_seq_len, vocab_size]
        vs = pet_logits.shape[-1]
        mask_idx = mask_idx.reshape(bs, num_lbl*self.config.max_num_lbl_tok)
        pet_rep_mask_ids_logit = torch.gather(pet_logits, 1, mask_idx[:, :, None].repeat(1, 1, vs).long()) # [bs, num_lbl * max_num_lbl_tok, vs]
        pet_rep_mask_ids_logit = pet_rep_mask_ids_logit.reshape(bs, num_lbl, self.config.max_num_lbl_tok, vs) # [bs, num_lbl, max_num_lbl_tok, vs]
        pet_rep_mask_ids_prob = pet_rep_mask_ids_logit.softmax(dim=-1)

        # Compute logit for the lbl tokens at the mask position
        lbl_ids_expd = lbl_ids[...,None] # [bs, num_lbl, max_num_lbl_tok, 1]
        pet_rep_mask_ids_lbl_logit = torch.gather(pet_rep_mask_ids_prob, 3, lbl_ids_expd.long()).squeeze(3)  # [bs, num_lbl, max_num_lbl_tok]

        if self.config.dataset.lower() == 'fewglue/wsc':
            masked_pet_rep_mask_ids_lbl_logit = pet_rep_mask_ids_lbl_logit * (mask_idx!=(pet_mask_ids.shape[-1] - 1)).unsqueeze(1).long()
        else:
            masked_pet_rep_mask_ids_lbl_logit = pet_rep_mask_ids_lbl_logit * (lbl_ids>0).long()

        return masked_pet_rep_mask_ids_lbl_logit, lbl_ids, None


    def get_decoupled_label_loss(self, batch):
        '''
        Get decoupled label loss

        :param batch:
        :return:
        '''

        pet_mask_ids, mask_idx, list_lbl = self.dataset_reader.prepare_batch(batch, self.get_pattern())
        lbl = batch["output"]["lbl"].to(device)

        # Datasets where the label has more than 1 token
        if isinstance(list_lbl[0], list):
            lbl_logits, lbl_ids, _ = self.get_multilbl_logits(pet_mask_ids, mask_idx,
                                                              list_lbl)  # [bs, num_lbl, max_num_lbl_tok]
            if "wsc" in self.config.dataset.lower():
                reshape_lbl_logits = lbl_logits.reshape(-1)  # [bs * num_lbl * max_num_lbl_tok]
                reshape_lbl = torch.ones_like(reshape_lbl_logits)
                real_mask = lbl_logits > 0

            else:
                # Removing tokens that are common across choices
                same_words_ids = torch.stack([reduce(lambda x, y: (x == y) * y, lbl_logit) for lbl_logit in lbl_logits],
                                             dim=0)
                mask_same_words = (1 - (same_words_ids > 0).long()).repeat(1, lbl_logits.shape[1],
                                                                           1)  # [bs, num_lbl, max_num_lbl_tok]
                real_mask = mask_same_words * (lbl_ids > 0)

                # Applying the mask to the lbl_logits
                lbl_logits = lbl_logits * mask_same_words  # [bs, num_lbl, max_num_lbl_tok]
                reshape_lbl_logits = lbl_logits.reshape(-1)  # [bs * num_lbl * max_num_lbl_tok]

                with torch.no_grad():
                    lkup_lbl = self.lbl_idx_lkup(lbl.long())  # [bs, num_lbl]
                reshape_lbl = lkup_lbl[:, :, None].repeat(1, 1, self.config.max_num_lbl_tok).reshape(-1)

            full_sup_loss = self.loss(reshape_lbl_logits, reshape_lbl)  # [bs * num_lbl * max_num_lbl_tok]
            full_sup_loss = full_sup_loss.reshape(lbl_logits.shape)

            pet_disc_loss = torch.sum(full_sup_loss * real_mask) / torch.sum(real_mask)

        # Datasets where the label is 1 token
        else:
            # Get lbl logits
            lbl_logits = self.get_single_logits(pet_mask_ids, mask_idx, list_lbl) # [bs, num_lbl]
            reshape_lbl_logits = lbl_logits.reshape(-1) # [bs*num_lbl, ]

            # lbl is 1 at true_lbl idx, and 0 otherwise
            with torch.no_grad():
                lkup_lbl = self.lbl_idx_lkup(lbl)  # [bs, num_lbl]
            reshape_lbl = lkup_lbl.reshape(-1) # [bs*num_lbl]

            pet_disc_loss = torch.mean(self.loss(reshape_lbl_logits, reshape_lbl))

        return pet_disc_loss


    def get_pet_mlm_logits(self, input_ids, masked_input_ids):
        '''
        Get logits for PET MLM objective

        :param input_ids: [bs, max_seq_len]
        :param masked_input_ids: [bs, max_seq_len]
        :return:
        '''
        pet_mask_rep = self.model(masked_input_ids, (masked_input_ids > 0).long())[0]  # [bs, max_seq_len, vocab_size]
        pet_mask_rep_vocab_prob = pet_mask_rep.softmax(dim=-1)  # [bs, max_num_lbl_tok, vocab_size]

        pet_mask_rep_correct_vocab_prob = torch.gather(pet_mask_rep_vocab_prob, 2, input_ids[:,:,None]).squeeze(2) # [bs, max_seq_len]

        return pet_mask_rep_correct_vocab_prob


    def forward(self, batch):
        '''
        :param batch:
        :return:
        '''
        # Decoupled label loss
        pet_disc_loss = self.get_decoupled_label_loss(batch)
        # We perform a backward on the pet_disc_loss here in order to accomodate all tasks in a single GPU.
        pet_disc_loss.backward()

        # PET MLM loss
        input_ids, mask_input_ids, prep_lbl, tgt = self.dataset_reader.prepare_batch(batch, "PET_MLM_{}".format(
            self.get_pattern()))
        correct_vocab_prob = self.get_pet_mlm_logits(input_ids, mask_input_ids)

        max_seq_len = correct_vocab_prob.shape[1]

        full_loss = self.loss(correct_vocab_prob,
                                  tgt[:, None].repeat(1, max_seq_len).float())  # [bs, max_seq_len]
        mask_loss = input_ids != mask_input_ids  # [bs, max_seq_len]
        pet_mlm_loss = torch.sum(full_loss * mask_loss.float()) / torch.max(torch.sum(mask_loss),
                                                                            torch.tensor(1).to(device))

        loss = pet_disc_loss.clone().detach() + pet_mlm_loss

        dict_val = {"loss": loss, "pet_disc_loss": pet_disc_loss, "pet_mlm_loss": pet_mlm_loss}

        return loss, dict_val

    def get_eval_multilbl_logits(self, pet_mask_ids, batch_mask_idx, batch_list_lbl):
        '''
        Evaluate for labels with multiple tokens

        :param pet_mask_ids: [bs, max_seq_len ]
        :param batch_mask_idx: [bs, num_lbl, max_num_lbl_tok]
        :param list_lbl: [bs, num_lbl]
        :return:
        '''
        log_probs = []
        # Assume batch size 0
        list_lbl = batch_list_lbl[0]
        mask_idx = batch_mask_idx[0]

        if self.config.dataset.lower() == "generic": pet_mask_ids = pet_mask_ids.repeat(len(list_lbl), 1)

        for idx, lbl in enumerate(list_lbl):
            lbl_ids = self.tokenizer(lbl, add_special_tokens=False)["input_ids"]
            log_probabilities = []

            while True:
                masks = [(idx, tok_id) for idx, tok_id in zip(mask_idx[idx], lbl_ids) if tok_id != -100]
                if not masks:
                    break

                pet_rep = self.model(pet_mask_ids[idx:idx + 1], (pet_mask_ids[idx:idx + 1] > 0).long())[
                    0]  # [bs, max_seq_len]
                next_token_logits = pet_rep.softmax(dim=-1)[
                    0]  # The last indexing operation gets rid of batch dimension

                # Only implementing the 'default' non-autoregressive strategy for now
                mask_pos, masked_id = None, None
                max_prob = None
                for m_pos, m_id in masks:
                    m_prob = next_token_logits[m_pos][m_id].item()
                    if max_prob is None or m_prob > max_prob:
                        max_prob = m_prob
                        mask_pos, masked_id = m_pos, m_id

                log_probabilities.append(math.log(max_prob))
                pet_mask_ids[idx][mask_pos] = masked_id
                if isinstance(mask_pos, list):
                    tok_pos = mask_idx[idx].index(mask_pos)
                else:
                    tok_pos = torch.min(torch.nonzero(mask_idx[idx] == mask_pos)[0])
                lbl_ids[tok_pos] = -100

            log_probs.append(sum(log_probabilities))

        return torch.tensor([log_probs])

    def get_eval_wsc_logits(self, pet_mask_ids, batch_mask_idx, batch_list_lbl):
        '''
        Get logits using from generated probs
        Code adapted from: https://github.com/timoschick/pet/blob/271910ebd4c30a4e0f8aaba39a153ae3d5822e22/pet/task_helpers.py#L453-L519

        :param batch:
        :param batch_mask_idx: [bs,][num_lbl][num_lbl_tok]
        :return:
        '''

        # Assume batch size 0
        list_lbl = batch_list_lbl[0]
        mask_idx = batch_mask_idx[0]

        while True:
            mask_positions = [
                idx for idx, input_id in enumerate(pet_mask_ids[0]) if input_id == self.tokenizer.mask_token_id
            ]
            if not mask_positions:  # there are no masks left to process, we are doneå
                input_ids = pet_mask_ids[0].detach().cpu().tolist()
                output_actual = self.tokenizer.decode([
                    input_id for idx, input_id in enumerate(input_ids)
                    if idx in mask_idx and input_id not in self.tokenizer.all_special_ids
                ])

                output_expected = list_lbl[0]

                # transform both outputs as described in the T5 paper
                output_actual = output_actual.lower().strip()
                output_actual = [w for w in re.split('[^a-zA-Z]', output_actual) if w]
                output_expected = output_expected.lower().strip()
                output_expected = [w for w in re.split('[^a-zA-Z]', output_expected) if w]

                # compare outputs
                if all(x in output_expected for x in output_actual) or all(
                        x in output_actual for x in output_expected):
                    return torch.tensor([[0, 1]])
                return torch.tensor([[1, 0]])

            outputs = self.model(pet_mask_ids, (pet_mask_ids > 0).long())
            next_token_logits = outputs[0]
            next_token_logits = next_token_logits.softmax(dim=2)
            next_token_logits = next_token_logits[0].detach().cpu().numpy()

            most_confident = ()
            most_confident_score = -1

            for mask_position in mask_positions:
                ntl = next_token_logits[mask_position]
                top_token_id = np.argmax(ntl)
                top_score = ntl[top_token_id]
                if top_score > most_confident_score:
                    most_confident_score = top_score
                    most_confident = (mask_position, top_token_id)

            pet_mask_ids[0][most_confident[0]] = most_confident[1]

    def predict_helper(self, batch, pattern):
        '''
        Predict the lbl for particular pet

        :param batch:
        :param pet:
        :return:
        '''

        pattern = "EVAL_{}".format(pattern)

        pet_mask_ids, mask_idx, list_lbl = self.dataset_reader.prepare_batch(batch, pattern)

        if self.config.dataset.lower() == 'fewglue/wsc':
            lbl_logits = self.get_eval_wsc_logits(pet_mask_ids, mask_idx, list_lbl)
        elif isinstance(list_lbl[0], list):
            pattern = "EVAL_{}".format(pattern)
            pet_mask_ids, mask_idx, list_lbl = self.dataset_reader.prepare_batch(batch, pattern)
            lbl_logits = self.get_eval_multilbl_logits(pet_mask_ids, mask_idx, list_lbl)
        else:
            lbl_logits = self.get_single_logits(pet_mask_ids, mask_idx, list_lbl)

        return torch.argmax(lbl_logits, dim=1), lbl_logits

    def predict(self, batch):
        '''
        Predict lbl

        :param batch:
        :return:
        '''

        if self.config.pattern_idx == "random":
            list_lbl_logits = []
            for pattern in self.pattern_list:
                lbl_pred, lbl_logits = self.predict_helper(batch, pattern)
                list_lbl_logits.append(lbl_logits)
            pattern_lbl_logits = torch.stack(list_lbl_logits, dim=0) # [num_pattern, bs, num_lbl]

            pattern_lbl_prob = pattern_lbl_logits.softmax(dim=-1)

            lbl_logits = torch.mean(pattern_lbl_prob, dim=0)
            lbl_pred = torch.argmax(lbl_logits, dim=1)

            return lbl_pred, lbl_logits

        else:
            return self.predict_helper(batch, self.get_pattern())

    def get_pattern(self):
        '''
        Get pattern to use

        :return:
        '''
        try:
            pattern = self.pattern()
        except:
            pattern = self.pattern
        return pattern
