import copy

from sentence_transformers import SentenceTransformer, models


class SetFit:
    def __init__(self, model, max_seq_length: int, add_normalization_layer: bool) -> None:
        self.model = SentenceTransformer(model)
        self.model_original_state = copy.deepcopy(self.model.state_dict())
        self.model.max_seq_length = max_seq_length

        if add_normalization_layer:
            self.model._modules["2"] = models.Normalize()
