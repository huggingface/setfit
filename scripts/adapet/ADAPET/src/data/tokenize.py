import numpy as np
import math
import random
from collections import defaultdict


def tokenize_pet_mlm_txt(tokenizer, config, txt1, txt2, txt3, txt_trim, mask_idx=None):
    '''
    Tokenizes the text by trimming the appropriate txt

    :param tokenizer:
    param config:
    :param txt1:
    :param txt2:
    :param txt3:
    :param mask_txt1:
    :param mask_txt2:
    :param mask_txt3:
    :param txt_trim: idx of text to trim will never contain label
    :return mask_idx: list of list of idx of mask token in trunc_input_ids (in case lbl is more than 1 token)
    '''

    txt1_input_ids = tokenizer(txt1, add_special_tokens=False)["input_ids"]
    txt2_input_ids = tokenizer(txt2, add_special_tokens=False)["input_ids"]
    txt3_input_ids = tokenizer(txt3, add_special_tokens=False)["input_ids"]

    # Add 1 to account for CLS rep
    tot_length = len(txt1_input_ids) + len(txt2_input_ids) + len(txt3_input_ids) + 1

    # Don't need to trim text
    if tot_length <= config.max_text_length:
        trunc_input_ids = [tokenizer.pad_token_id] * config.max_text_length
        trunc_input_ids[:tot_length] = txt1_input_ids + txt2_input_ids + txt3_input_ids

    # Trim text
    else:
        num_trim = tot_length - config.max_text_length

        if txt_trim == 0:
            new_txt1_input_ids = txt1_input_ids[:-num_trim]
            trunc_input_ids = new_txt1_input_ids + txt2_input_ids + txt3_input_ids
        elif txt_trim == 1:
            new_txt2_input_ids = txt2_input_ids[:-num_trim]
            trunc_input_ids = txt1_input_ids + new_txt2_input_ids + txt3_input_ids
        elif txt_trim == 2:
            new_txt_3_input_ids = txt3_input_ids[:-num_trim]
            trunc_input_ids = txt1_input_ids + txt2_input_ids + new_txt_3_input_ids
        else:
            raise ValueError("Invalid Txt Trim")

    trunc_input_ids = [tokenizer.cls_token_id] + trunc_input_ids

    if mask_idx is None:
        sample_length = min(tot_length, config.max_text_length)
        upto_ratio_mask = np.random.rand()
        num_sample = max(int(upto_ratio_mask * config.mask_alpha * sample_length), 2) - 1
        mask_idx = random.sample(range(0, sample_length), k=num_sample)
        mask_idx = np.asarray(mask_idx)

    # Copy adds mask idx at random positions
    unsup_masked_ids = np.copy(trunc_input_ids)

    unsup_masked_ids[mask_idx] = tokenizer.mask_token_id

    return trunc_input_ids, unsup_masked_ids, mask_idx

def tokenize_pet_txt(tokenizer, config, txt1, txt2, txt3, mask_txt1, mask_txt2, mask_txt3, txt_trim):
    '''
    Tokenizes the text by trimming the appropriate txt

    :param txt1:
    :param txt2:
    :param txt3:
    :param mask_txt1:
    :param mask_txt2:
    :param mask_txt3:
    :param txt_trim: text to trim will never contain label
    :return trunc_input_ids: list of input ids (each exactly max_config_length)
    :return mask_idx: list of list of idx of mask token in trunc_input_ids (in case lbl is more than 1 token)
    '''
    txt1_input_ids = tokenizer(txt1, add_special_tokens=False)["input_ids"]
    txt2_input_ids = tokenizer(txt2, add_special_tokens=False)["input_ids"]
    txt3_input_ids = tokenizer(txt3, add_special_tokens=False)["input_ids"]

    mask_txt1_input_ids = tokenizer(mask_txt1, add_special_tokens=False)["input_ids"]
    mask_txt2_input_ids = tokenizer(mask_txt2, add_special_tokens=False)["input_ids"]
    mask_txt3_input_ids = tokenizer(mask_txt3, add_special_tokens=False)["input_ids"]

    # Add 1 to account for CLS rep
    tot_length = len(txt1_input_ids) + len(txt2_input_ids) + len(txt3_input_ids) + 1
    tot_mask_length = len(mask_txt1_input_ids) + len(mask_txt2_input_ids) + len(mask_txt3_input_ids) + 1

    # Don't need to trim text
    if tot_length <= config.max_text_length:
        trunc_input_ids = [tokenizer.pad_token_id] * config.max_text_length
        trunc_input_ids[:tot_length] = txt1_input_ids + txt2_input_ids + txt3_input_ids

        trunc_mask_input_ids = [tokenizer.pad_token_id] * config.max_text_length
        trunc_mask_input_ids[:tot_mask_length] = mask_txt1_input_ids + mask_txt2_input_ids + mask_txt3_input_ids

    # Trim text
    else:
        num_trim = tot_length - config.max_text_length

        if txt_trim == 0:
            new_txt1_input_ids = txt1_input_ids[:-num_trim]
            new_mask_txt1_input_ids = mask_txt1_input_ids[:-num_trim]
            trunc_input_ids = new_txt1_input_ids + txt2_input_ids + txt3_input_ids
            trunc_mask_input_ids = new_mask_txt1_input_ids + mask_txt2_input_ids + mask_txt3_input_ids
        elif txt_trim == 1:
            new_txt2_input_ids = txt2_input_ids[:-num_trim]
            new_mask_txt2_input_ids = mask_txt2_input_ids[:-num_trim]
            trunc_input_ids = txt1_input_ids + new_txt2_input_ids + txt3_input_ids
            trunc_mask_input_ids = mask_txt1_input_ids + new_mask_txt2_input_ids + mask_txt3_input_ids
        elif txt_trim == 2:
            new_txt_3_input_ids = txt3_input_ids[:-num_trim]
            new_mask_txt3_input_ids = mask_txt3_input_ids[:-num_trim]
            trunc_input_ids = txt1_input_ids + txt2_input_ids + new_txt_3_input_ids
            trunc_mask_input_ids = mask_txt1_input_ids + mask_txt2_input_ids + new_mask_txt3_input_ids
        else:
            raise ValueError("Invalid Txt Trim")


    trunc_input_ids = [tokenizer.cls_token_id] + trunc_input_ids
    trunc_mask_input_ids = [tokenizer.cls_token_id] + trunc_mask_input_ids

    mask_idx = trunc_mask_input_ids.index(tokenizer.mask_token_id)

    return trunc_input_ids, mask_idx
