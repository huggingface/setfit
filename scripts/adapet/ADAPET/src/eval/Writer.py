
class Writer(object):

    def __init__(self, file, dataset_reader):
        self.write_file = open(file, 'w+')
        self.dataset_reader = dataset_reader

    def add_batch(self, list_idx, list_pred_lbl, list_true_lbl, lbl_logits):
        self.dataset_reader.store_test_lbl(list_idx, list_pred_lbl, list_true_lbl, lbl_logits)

    def flush_file(self):
        self.dataset_reader.flush_file(self.write_file)