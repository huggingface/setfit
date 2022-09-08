from .modeling import SKLearnWrapper


class SetFitPipeline:
    def __init__(self, model_name_or_path) -> None:
        base_model = SKLearnWrapper()
        base_model.load(model_name_or_path)
        self.model = base_model

    def __call__(self, inputs, *args, **kwargs):
        model_outputs = self.model.predict(inputs)
        return model_outputs
