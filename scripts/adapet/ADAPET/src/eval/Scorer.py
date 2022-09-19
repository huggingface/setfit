import torch
import re
import string
import numpy as np
import collections

from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score



class Scorer(object):

    def __init__(self, config, dataset):
        self.config = config

        # Metrics to compute
        self.compute_acc = False
        self.compute_f1 = False

        # Store which dataset
        self.is_boolq = False
        self.is_cb = False
        self.is_copa = False
        self.is_multirc = False
        self.is_record = False
        self.is_rte = False
        self.is_wic = False

        self.list_logits = []


        # Compute the dataset
        if dataset.lower() == "fewglue/boolq":
            self.compute_acc = True
            self.is_boolq = True
        elif dataset.lower() == "fewglue/cb":
            self.compute_acc = True
            self.compute_f1 = True
            self.is_cb = True
        elif dataset.lower() == "fewglue/copa":
            self.compute_acc = True
            self.is_copa = True
        elif dataset.lower() == "fewglue/multirc":
            self.compute_acc = True
            self.compute_f1 = True
            self.is_multirc = True
        elif dataset.lower() == "fewglue/record":
            self.compute_acc = True
            self.compute_f1 = True
            self.is_record = True
        elif dataset.lower() == "fewglue/rte":
            self.compute_acc = True
            self.is_rte = True
        elif dataset.lower() == "fewglue/wic":
            self.compute_acc = True
            self.is_wic = True
        elif dataset.lower() == "fewglue/wsc":
            self.compute_acc = True
            self.is_wsc = True
        elif dataset.lower() == "generic":
            self.compute_acc = True
            #self.compute_f1 = True
        else:
            raise ValueError("Invalid Dataset name")

        self.dict_idx2logits_lbl = {}

    def _compute_acc(self):
        '''
        :return:
        '''

        acc_cor_cnt = 0
        acc_ttl_cnt = 0

        if self.is_multirc:
            for (idx, pred_true_lbl) in self.dict_idx2logits_lbl.items():

                exact_match = True
                for (pred_lbl, true_lbl, _) in pred_true_lbl:
                    if pred_lbl != true_lbl:
                        exact_match = False
                if exact_match:
                    acc_cor_cnt += 1

                acc_ttl_cnt += 1

        else:
            for (idx, pred_true_lbl) in self.dict_idx2logits_lbl.items():
                pred_lbl = pred_true_lbl[0][0]
                true_lbl = pred_true_lbl[0][1]

                acc_ttl_cnt += 1
                if pred_lbl == true_lbl:
                    acc_cor_cnt += 1

        round_tot_acc = float(round(acc_cor_cnt / acc_ttl_cnt, 3))
        return round_tot_acc

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace.
        From official ReCoRD eval script
        """

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def string_f1_score(self, prediction, ground_truth):
        """Compute normalized token level F1
        From official ReCoRD eval script
        """
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def _compute_f1(self):

        if self.is_multirc:
            f1_pred_lbl = []
            f1_true_lbl = []

            for (idx, pred_true_lbl) in self.dict_idx2logits_lbl.items():
                for (pred_lbl, true_lbl, _) in pred_true_lbl:
                    f1_pred_lbl.append(pred_lbl)
                    f1_true_lbl.append(true_lbl)

        else:
            f1_pred_lbl = []
            f1_true_lbl = []

            for (idx, pred_true_lbl) in self.dict_idx2logits_lbl.items():
                f1_pred_lbl.append(pred_true_lbl[0][0])
                f1_true_lbl.append(pred_true_lbl[0][1])

        if self.is_record:
            f1 = f1_score(f1_true_lbl, f1_pred_lbl)
            avg_f1 = np.mean(f1)
        else:
            #f1 = f1_score(f1_true_lbl, f1_pred_lbl, average=None)
            #avg_f1 = np.mean(f1)
            # print('not record')
            mic_f1 = f1_score(f1_true_lbl, f1_pred_lbl, average="micro")
            mac_f1 = f1_score(f1_true_lbl, f1_pred_lbl, average="macro")
            if len(set(list(f1_true_lbl))) == 2:
                avg_pre = average_precision_score(f1_true_lbl, f1_pred_lbl, average="macro")
            else:
                avg_pre = 0
            mic_avg_f1 = np.mean(mic_f1)
            mac_avg_f1 = np.mean(mac_f1)
            mean_avg_pre = np.mean(avg_pre)
            mic_avg_f1 = round(mic_avg_f1, 4)
            mac_avg_f1 = round(mac_avg_f1, 4)
            mean_avg_pre = round(mean_avg_pre, 4)                

        return "f1_mic {} f1_mac {} avg_pre {}".format(mic_avg_f1, mac_avg_f1, mean_avg_pre)

        #return round(avg_f1, 3)


    def add_batch(self, list_idx, list_pred_lbl, list_true_lbl, lbl_logits, list_candidates=None):
        '''
        Keeps track of the accuracy of current batch
        :param logits:
        :param true_lbl:
        :return:
        '''

        self.list_logits.append(lbl_logits)

        lbl_logits = lbl_logits.tolist()

        if torch.is_tensor(list_true_lbl):
            list_true_lbl =  list_true_lbl.cpu().detach().numpy()

        if list_candidates is not None:
            for idx, pred_lbl, true_lbl, logit, cnd in zip(list_idx, list_pred_lbl.cpu().detach().numpy(), list_true_lbl, lbl_logits, list_candidates):
                if idx in self.dict_idx2logits_lbl:
                    self.dict_idx2logits_lbl[idx].append((pred_lbl, true_lbl, logit, cnd))
                else:
                    self.dict_idx2logits_lbl[idx] = [(pred_lbl, true_lbl, logit, cnd)]
        else:
            for idx, pred_lbl, true_lbl, logit in zip(list_idx, list_pred_lbl.cpu().detach().numpy(), list_true_lbl, lbl_logits):
                if idx in self.dict_idx2logits_lbl:
                    self.dict_idx2logits_lbl[idx].append((pred_lbl, true_lbl, logit))
                else:
                    self.dict_idx2logits_lbl[idx] = [(pred_lbl, true_lbl, logit)]

    def get_score(self, split):
        '''
        Gets the accuracy
        :return: rounded accuracy to 3 decimal places
        '''

        dict_scores = {}
        score_eval = 0

        if self.compute_acc:
            round_tot_acc = self._compute_acc()
            type = "%s_acc" % split
            dict_scores[type] = round_tot_acc
            score_eval = round_tot_acc
        if self.compute_f1:
            avg_f1 = self._compute_f1()
            type = "%s_f1" % split
            dict_scores[type] = avg_f1
            score_eval = avg_f1

        return score_eval, dict_scores

    def get_logits(self):
        #TODO: Hack to deal with multi label token logits
        if self.is_record:
            return np.zeros((10, 100))
        else:
            return np.concatenate(self.list_logits, axis=0)
