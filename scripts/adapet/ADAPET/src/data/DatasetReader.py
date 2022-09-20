import numpy as np


from src.data.BoolQReader import BoolQReader
from src.data.CBReader import CBReader
from src.data.RTEReader import RTEReader
from src.data.MultiRCReader import MultiRCReader
from src.data.WiCReader import WiCReader
from src.data.COPAReader import COPAReader
from src.data.RecordReader import RecordReader
from src.data.WSCReader import WSCReader
from src.data.GenericReader import GenericReader

class DatasetReader(object):
    '''
    DatasetReader is responsible for reading dataset
    '''
    def __init__(self, config, tokenizer, dataset):
        '''
        :param config:
        :param tokenizer:
        :param dataset:
        '''
        self.config = config
        self.dataset = dataset

        if self.dataset.lower() == "fewglue/boolq":
            self.dataset_reader = BoolQReader(self.config, tokenizer)
        elif self.dataset.lower() == "fewglue/cb":
            self.dataset_reader = CBReader(self.config, tokenizer)
        elif self.dataset.lower() == "fewglue/rte":
            self.dataset_reader = RTEReader(self.config, tokenizer)
        elif self.dataset.lower() == "fewglue/multirc":
            self.dataset_reader = MultiRCReader(self.config, tokenizer)
        elif self.dataset.lower() == "fewglue/wic":
            self.dataset_reader = WiCReader(self.config, tokenizer)
        elif self.dataset.lower() == "fewglue/copa":
            self.dataset_reader = COPAReader(self.config, tokenizer)
        elif self.dataset.lower() == "fewglue/record":
            self.dataset_reader = RecordReader(self.config, tokenizer)
        elif self.dataset.lower() == "fewglue/wsc":
            self.dataset_reader = WSCReader(self.config, tokenizer)
        elif self.dataset.lower() == "generic":
            self.dataset_reader = GenericReader(self.config, tokenizer)
        else:
            raise ValueError("Invalid Dataset name")

    def get_num_lbl_tok(self):
        '''
        Get number of token in labels for dataset

        :return:
        '''
        return self.dataset_reader.get_num_lbl_tok()

    def read_dataset(self, split, is_eval=False):
        '''
        Read dataset

        :param split:
        :param is_eval:
        :return:
        '''
        return np.asarray(self.dataset_reader.read_dataset(split, is_eval))

    def prepare_batch(self, batch, type):
        '''
        Prepare batch of data for model

        :param batch:
        :param type: pattern to prepare batch with and which mode to use (ex: PET_MLM_PET1)
        :return:
        '''
        # Prepare for PET MLM objective
        if "PET_MLM" in type:
            return self.dataset_reader.prepare_pet_mlm_batch(batch, mode=type.replace("PET_MLM_", ""))
        # Prepare for evaluation objective
        elif "EVAL" in type:
            return self.dataset_reader.prepare_eval_pet_batch(batch, mode=type.replace("EVAL_", ""))
        # Default is preparing for PET/Decoupled Label objective
        else:
            return self.dataset_reader.prepare_pet_batch(batch, mode=type)

    def store_test_lbl(self, list_idx, pred_lbl, true_lbl, logits):
        '''
        Store test outputs for SuperGLUE to submit to leaderboard

        :param list_idx:
        :param pred_lbl:
        :param true_lbl:
        :param logits:
        :return:
        '''
        self.dataset_reader.store_test_lbl(list_idx, pred_lbl, true_lbl, logits)

    def flush_file(self, write_file):
        '''
        Write out contents of test predictions to file

        :param write_file:
        :return:
        '''
        self.dataset_reader.flush_file(write_file)

    def get_num_lbl(self):
        '''
        Get number of lbls in dataset

        :return:
        '''
        return self.dataset_reader.num_lbl
