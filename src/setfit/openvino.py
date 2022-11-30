import warnings
from typing import Callable, Optional, Union
import copy

import numpy as np
import onnx
import openvino.runtime as ov
import torch
from sentence_transformers import SentenceTransformer, models
from sklearn.linear_model import LogisticRegression
from hummingbird.ml import convert
from torch import nn
from transformers.modeling_utils import PreTrainedModel


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Perform attention-aware mean pooling.

    This method takes in embeddings of shape (batch, sequence, embedding_size) and performs average
    pooling across the sequence dimension to yield embeddings of size (batch, embedding_size).

    From:
    https://github.com/UKPLab/sentence-transformers/blob/0b5ef4be93d2b21de3a918a084b48aab6ba48595/sentence_transformers/model_card_templates.py#L134  # noqa: E501

    Args:
        token_embeddings (`torch.Tensor`): The embeddings we wish to pool over of shape
            (batch, sequence, embedding_size).  This will pool over the sequence to yield
            (batch, embedding_size).
        attention_mask (`torch.Tensor`): The binary attention mask across the embedings of shape

    Returns:
        (`torch.Tensor`) The mean pooled embeddings of size (batch, embedding_size).
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class OnnxSetFitModel(torch.nn.Module):
    """A wrapper around SetFit model body, pooler, and model head which makes ONNX exporting easier.

    This wrapper creates a `nn.Module` with different levels of connectivity. We can set
    `model_body` and `pooler` and have a Module which maps inputs to embeddings or we can set all three
    and have a model which maps inputs to final predictions. This is useful because `torch.onnx.export`
    will work with a `nn.Module`.

    Attributes:
        model_body (`PreTrainedModel`): The pretrained model body of a setfit model.
        pooler (`Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]`, *optional*, defaults to `None`): The
            callable function that can map  tensors of shape (batch, sequence, embedding_dim) to shape
            (batch, embedding_dim).
        model_head: (`Union[nn.Module, LogisticRegression]`, *optional*, defaults to `None`): The model head from
            the pretrained SetFit model. If `None`, then the resulting `OnnxSetFitModel.forward` forward  method will
            return embeddings instead of predictions.
    """

    def __init__(
        self,
        model_body: PreTrainedModel,
        pooler: Optional[Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]] = None,
        model_head: Optional[Union[torch.nn.Module, LogisticRegression]] = None,
    ):
        super().__init__()

        self.model_body = model_body
        if pooler is None:
            print("No pooler was set so defaulting to mean pooling.")
            self.pooler = mean_pooling
        else:
            self.pooler = pooler
        self.model_head = model_head

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor):
        hidden_states = self.model_body(input_ids, attention_mask, token_type_ids)
        hidden_states = {"token_embeddings": hidden_states[0], "attention_mask": attention_mask}

        embeddings = self.pooler(hidden_states)

        # If the model_head is none we are using a sklearn head and only output
        # the embeddings from the setfit model
        if self.model_head is None:
            return embeddings

        # If head is set then we have a fully torch based model and make the final predictions
        # with the head.
        out = self.model_head(embeddings)
        return out

def hummingbird_export(model, data_sample):
    onnx_model = convert(model, "onnx", data_sample)
    return onnx_model._model

def export_onnx_setfit_model(setfit_model: OnnxSetFitModel, inputs, output_path, opset: int = 12):
    """Export the `OnnxSetFitModel`.

    This exports the model created by the `OnnxSetFitModel` wrapper using `torch.onnx.export`.

    Args:
        setfit_model (`OnnxSetFitModel`): The `OnnxSetFitModel` we want to export to .onnx format.
        inputs (`Dict[str, torch.Tensor]`): The inputs we would hypothetically pass to the model. These are
            generated using a tokenizer.
        output_path (`str`): The local path to save the onnx model to.
        opset (`int`): The ONNX opset to use for the export.
    """
    input_names = list(inputs.keys())
    output_names = ["logits"]

    # Setup the dynamic axes for onnx conversion.
    dynamic_axes_input = {}
    for input_name in input_names:
        dynamic_axes_input[input_name] = {0: "batch_size", 1: "sequence"}

    dynamic_axes_output = {}
    for output_name in output_names:
        dynamic_axes_output[output_name] = {0: "batch_size"}

    setfit_model.eval()
    with torch.no_grad():
        torch.onnx.export(
            setfit_model,
            args=tuple(inputs.values()),
            f=output_path,
            opset_version=opset,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=output_names,
            dynamic_axes={**dynamic_axes_input, **dynamic_axes_output},
        )

def export_openvino(
    model_body: SentenceTransformer,
    model_head: Union[torch.nn.Module, LogisticRegression],
    opset: int,
    output_path: str = "model.onnx",
    ignore_ir_version: bool = True,
) -> None:
    """Export a PyTorch backed SetFit model to ONNX Intermediate Representation.

    Args:
        model_body (`SentenceTransformer`): The model_body from a SetFit model body. This should be a
            SentenceTransformer.
        model_head (`torch.nn.Module` or `LogisticRegression`): The SetFit model head. This can be either a
            dense layer SetFitHead or a Sklearn estimator.
        opset (`int`): The actual version of the ONNX operator set to use.  The final opset used might be lower.
            ONNX will use the highest version supported by both the sklearn head and the model body. If versions
            can't be rectified an error will be thrown.
        output_path (`str`): The path where will be stored the generated ONNX model. At a minimum it needs to contain
            the name of the final file.
        ignore_ir_version (`bool`): Whether to ignore the IR version used in sklearn. The version is often missmatched
            with the transformer models. Setting this to true coerces the versions to be the same. This might
            cause errors but in practice works.  If this is set to False you need to ensure that the IR versions
            align between the transformer and the sklearn onnx representation.
    """

    # Load the model and get all of the parts.
    model_body_module = model_body._modules["0"]
    model_pooler = model_body._modules["1"]
    tokenizer = model_body_module.tokenizer
    max_length = model_body_module.max_seq_length
    transformer = model_body_module.auto_model
    transformer.eval()

    # Create dummy data to use during onnx export.
    tokenizer_kwargs = dict(
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )
    dummy_sample = "It's a test."
    dummy_inputs = tokenizer(dummy_sample, **tokenizer_kwargs)

    # Check to see if the model uses a sklearn head or a torch dense layer.
    if issubclass(type(model_head), models.Dense):
        setfit_model = OnnxSetFitModel(transformer, lambda x: model_pooler(x)["sentence_embedding"], model_head).cpu()
        export_onnx_setfit_model(setfit_model, dummy_inputs, output_path, opset)

        # store meta data of the tokenizer for getting the correct tokenizer during inference
        onnx_setfit_model = onnx.load(output_path)
        meta = onnx_setfit_model.metadata_props.add()
        meta.key = "model_name"
        meta.value = transformer.name_or_path
        for key, value in tokenizer_kwargs.items():
            meta = onnx_setfit_model.metadata_props.add()  # create a new key-value pair to store
            meta.key = key
            meta.value = value

    else:
        # Export the sklearn head first to get the minimum opset.  sklearn is behind
        # in supported opsets.
        test_input = copy.deepcopy(dummy_inputs)
        head_input = model_body(test_input)["sentence_embedding"]

        onnx_head = hummingbird_export(model_head, head_input.detach().numpy())
        max_opset = onnx_head.opset_import[0].version

        if max_opset != opset:
            warnings.warn(
                f"sklearn onnx max opset is {max_opset} requested opset {opset} using opset {max_opset} for compatibility."
            )
        export_onnx_setfit_model(
            OnnxSetFitModel(transformer, lambda x: model_pooler(x)["sentence_embedding"]),
            dummy_inputs,
            output_path,
            max_opset,
        )

        onnx_body = onnx.load(output_path)

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
        head_input_name = next(iter(onnx_head.graph.input)).name
        onnx_setfit_model = onnx.compose.merge_models(
            onnx_body,
            onnx_head,
            io_map=[("logits", head_input_name)],
        )

    # Save the final model.
    onnx_path = output_path.replace(".xml", ".onnx")
    onnx.save(onnx_setfit_model, onnx_path)
    ov_model = ov.Core().read_model(onnx_path)
    ov.serialize(ov_model, output_path)
