from .modeling import SetFitModel
from sentence_transformers import models
import numpy as np
import onnx
import warnings
from typing import Optional, Union
import torch
from sklearn.linear_model import LogisticRegression


class ONNXSetFitModel(torch.nn.Module):
    def __init__(self, model_body, pooler, model_head: Optional[Union[torch.nn.Module, LogisticRegression]] = None):
        super().__init__()

        self.model_body = model_body
        self.pooler = pooler
        self.model_head = model_head

    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden = self.model_body(input_ids, attention_mask, token_type_ids)
        hidden = {"token_embeddings": hidden[0], "attention_mask": attention_mask}

        hidden = self.pooler(hidden)["sentence_embedding"]
        if self.model_head is None:
            return hidden

        out = self.model_head(hidden)
        return out


def sklearn_head_to_onnx(model: SetFitModel, opset: int):
    """
    Convert the sklearn head from a SetFitModel to ONNX format.

    :param model: The trained SetFit model with a sklearn model_head.
    :param opset: The ONNX opset to use for optimizing this model. The opset is not
        gauranteed and will default to the maximum version possible for the sklearn
        model.
    :raises ImportError: If skl2onnx is not installed an error will be raised asking
        to install this package.
    :return: The ONNX model generated from the sklearn head.
    """

    # Check if skl2onnx is installed
    try:
        from skl2onnx.common.data_types import guess_data_type
        from skl2onnx import convert_sklearn
        import onnxconverter_common
        from skl2onnx.sklapi import CastTransformer
        from sklearn.pipeline import Pipeline
    except ImportError:
        msg = "skl2onnx must be installed in order to convert a model with an sklearn head."
        raise ImportError(msg)

    # Check to see that the head has a coef_
    if not hasattr(model.model_head, "coef_"):
        raise ValueError(
            "Head must have coef_ attribute check that this is supported by your model and the model has been fit."
        )

    # Determine the initial type and the shape of the output.
    input_shape = (None, *model.model_head.coef_.shape[1:])
    dtype = guess_data_type(model.model_head.coef_, shape=input_shape)[0][1]
    dtype.shape = input_shape

    # If the datatype of the model is double we need to cast the outputs
    # from the setfit model to doubles for compatibility inside of ONNX.
    if type(dtype) == onnxconverter_common.data_types.DoubleTensorType:
        # TODO:: TALK ABOUT FLOAT CONVERSION ISSUES.
        sk_model = Pipeline([("castdouble", CastTransformer(dtype=np.double)), ("head", model.model_head)])
    else:
        sk_model = model.model_head

    # Convert sklearn head into ONNX format
    initial_type = [("model_head", dtype)]
    onx = convert_sklearn(
        sk_model, initial_types=initial_type, target_opset=opset, options={id(sk_model): {"zipmap": False}},
    )

    return onx


def export_onnx(model_name: str, opset: int, output: str, ignore_ir_version: bool = True):
    """
    Export a PyTorch backed setfit model to ONNX Intermediate Representation.

    Args:
        model: The name of the pretrained setfit model to load and use. It can be a path
            to a locally trained model.
        opset: The actual version of the ONNX operator set to use.  The final opset used
            might be lower. ONNX will use the highest version supported by both the
            sklearn head and the model body. If versions can't be rectified an error
            will be thrown.
        output: The path where will be stored the generated ONNX model. If the head is an
            sklearn.estimator the models will be saved in
        ignore_ir_version: Whether to ignore the IR version used in sklearn. The version is often missmatched
            with the transformer models. Setting this to true coerces the versions to be the same. This might
            cause errors but in practice works.  If this is set to False you need to ensure that the IR versions
            align between the transformer and the sklearn onnx representation.
    Returns:
    """

    # Load the model and get all of the parts.
    model = SetFitModel.from_pretrained(model_name)
    tokenizer = transformer.tokenizer
    max_length = transformer.max_seq_length
    transformer = model.model_body._modules["0"]
    model_body = transformer.auto_model
    model_pooler = model.model_body._modules["1"]
    model_head = model.model_head

    # Create dummy data to use during onnx export.
    tokenizer_kwargs = dict(
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )
    dummy_sample = "It's a test."
    dummy_inputs = tokenizer(
        dummy_sample,
        **tokenizer_kwargs
    )
    symbolic_names = {0: "batch_size", 1: "max_seq_len"}

    # Check to see if the model uses a sklearn head or a torch dense layer.
    if issubclass(type(model_head), models.Dense):
        setfit_model = ONNXSetFitModel(model_body, model_pooler, model_head).cpu()
        setfit_model.eval()
        with torch.no_grad():
            torch.onnx.export(
                setfit_model,
                args=tuple(dummy_inputs.values()),
                f=output,
                opset_version=opset,
                input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["prediction"],
                dynamic_axes={
                    'input_ids': symbolic_names,        # variable length axes
                    'attention_mask': symbolic_names,
                    'token_type_ids': symbolic_names,
                    'prediction': {0: 'batch_size'},
                }
            )
        
        # store meta data of the tokenizer for getting the correct tokenizer during inference
        onnx_setfit_model = onnx.load(output)
        meta = onnx_setfit_model.metadata_props.add()
        meta.key = "model_name"
        meta.value = model_name
        for key, value in tokenizer_kwargs.items():
            meta = onnx_setfit_model.metadata_props.add()  # create a new key-value pair to store
            meta.key = key
            meta.value = value
        onnx.save(onnx_setfit_model, output)
    else:
        # TODO:: Make this work for other sklearn models without coef_.
        if not hasattr(model_head, "coef_"):
            raise ValueError("Model head must have coef_ attribute for weights.")

        # Export the sklearn head first to get the minimum opset.  sklearn is behind
        # in supported opsets.
        onnx_head = sklearn_head_to_onnx(model, opset)
        max_opset = onnx_head.opset_import[0].version

        if max_opset != opset:
            warnings.warn(
                f"sklearn onnx max opset is {max_opset} requested opset {opset} using opset {max_opset} for compatibility."
            )

        model_body.eval()
        with torch.no_grad():
            torch.onnx.export(
                ONNXSetFitModel(model_body, model_pooler),
                args=tuple(dummy_inputs.values()),
                f=output,
                opset_version=max_opset,
                input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["prediction"],
                dynamic_axes={
                    "input_ids": symbolic_names,  # variable length axes
                    "attention_mask": symbolic_names,
                    "token_type_ids": symbolic_names,
                    "prediction": {0: "batch_size"},
                },
            )

        onnx_body = onnx.load(output)

        # Check that the ir_versions are aligned and if not align them.
        if ignore_ir_version:
            onnx_head.ir_version = onnx_body.ir_version
        elif onnx_head.ir_version != onnx_body.ir_version:
            msg = f"""
            IR Version mismatch between head={onnx_head.ir_version} and body={onnx_body.ir_version}
            Make sure that the ONNX IR versions are aligned and supported between the chosen Sklearn model
            and the transformer.  You can set ignore_ir_version=True to coerce them but this might cause errors.
            """
            raise ValueError(msg)

        # Combine the onnx body and head by mapping the pooled output to the input of the sklearn model.
        combined_model = onnx.compose.merge_models(onnx_body, onnx_head, io_map=[("prediction", "model_head")],)

        onnx.save(combined_model, output)
