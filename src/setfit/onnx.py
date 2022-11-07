from pathlib import Path
from .modeling import SetFitModel
from sentence_transformers import models
from transformers.convert_graph_to_onnx import convert_pytorch
from transformers import pipeline
import numpy as np
import onnx
import warnings


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
    except ImportError as e:
        msg = "skl2onnx must be installed in order to convert a model with an sklearn head."
        raise ImportError(msg)

    # Determine the initial type and the shape of the output.
    input_shape = (None, *model.model_head.coef_.shape[1:])
    dtype = guess_data_type(model.model_head.coef_, shape=input_shape)[0][1]
    dtype.shape = input_shape

    # If the datatype of the model is double we need to cast the outputs
    # from the setfit model to doubles for compatibility inside of ONNX.
    if type(dtype) == onnxconverter_common.data_types.DoubleTensorType:
        # TODO:: TALK ABOUT FLOAT CONVERSION ISSUES.
        sk_model = Pipeline(
            [
                ("castdouble", CastTransformer(dtype=np.double)),
                ("head", model.model_head),
            ]
        )
    else:
        sk_model = model.model_head

    # Convert sklearn head into ONNX format
    initial_type = [("model_head", dtype)]
    onx = convert_sklearn(
        sk_model,
        initial_types=initial_type,
        target_opset=opset,
        options={id(sk_model): {"zipmap": False}},
    )

    return onx


def export_onnx(
    model_name: str, opset: int, output: Path, use_external_format: bool = False
):
    """
    Export a PyTorch backed setfit model to ONNX Intermediate Representation.

    Args:
        model: The name of the pretrained setfit model to load and use. It can be a path
            to a locally trained model.
        opset: The actual version of the ONNX operator set to use.  The final opset used
            might be lower. ONNX will use the highest version supported by both the
            sklearn head and the model body. If versions can't be rectified an error
            will be thrown.
        output: Path where will be stored the generated ONNX model. If the head is an
            sklearn.estimator the models will be saved in
        use_external_format: Split the model definition from its parameters to allow
            model bigger than 2GB
    Returns:
    """

    model = SetFitModel.from_pretrained(model_name)

    # Check to see if the model uses a sklearn head or a torch dense layer.
    if issubclass(type(model.model_head), models.Dense):
        # TODO:: Combine the dense layer on top of the normal model and export as onnx.
        raise (
            NotImplementedError(
                "Full pytorch model conversion is not implemented please use sklearn head."
            )
        )
    else:

        # TODO:: Make this work for other sklearn models without coef_.
        if not hasattr(model.model_head, "coef_"):
            raise ValueError("Model head must have coef_ attribute for weights.")

        # Export the sklearn head first to get the minimum opset.  sklearn is behind
        # in supported opsets.
        onnx_head = sklearn_head_to_onnx(model, opset)
        min_opset = onnx_head.opset_import[0].version

        if min_opset < 13:
            raise ValueError(
                "MeanPooling layer uses ONNX operations requireing opset >=13."
            )

        if min_opset != opset:
            warnings.warn(
                f"sklearn onnx max opset is {min_opset} requested opset {opset} using opset {min_opset} for compatibility."
            )

        nlp = pipeline(
            "feature-extraction",
            model=model_name,
            tokenizer=model_name,
            framework="pt",
        )

        # FIXME:: The `transformers.convert_graph_to_onnx` package is deprecated and will be
        # removed in version 5 of transformers we need to update this.
        convert_pytorch(nlp, min_opset, output, use_external_format)
        onnx_body = onnx.load(output)

        # Assign attention_mask as an output so we can use it as an input to the mean pooling.
        intermediate_layer_value_info = onnx.helper.ValueInfoProto()
        intermediate_layer_value_info.name = "attention_mask"
        onnx_body.graph.output.extend([intermediate_layer_value_info])

        if onnx_body.ir_version != onnx_head.ir_version:
            raise ValueError(
                f"""IR version mismatch
                model_body ir_verion: {onnx_body.ir_version} != model_head ir_verion: {onnx_head.ir_version}
                This can sometimes be fixed by reducing the opset version. Try opset = {min_opset - 1}
                """
            )

        # Create the mean pooling layer.
        onnx_mean_pool_graph = _create_mean_pooling_onnx_graph(
            embedding_size=model.model_head.coef_.shape[1]
        )

        # Pass the attention_mask as an output.
        intermediate_layer_value_info = onnx.helper.ValueInfoProto()
        intermediate_layer_value_info.name = "attention_mask"
        onnx_body.graph.output.extend([intermediate_layer_value_info])
        onnx_mean_pool_graph.ir_version = onnx_body.ir_version
        onnx_mean_pool_graph.opset_import[0].version = min_opset

        # Attach the mean pooling layer to the model body.
        onnx_body_with_mean_pooling = onnx.compose.merge_models(
            onnx_body,
            onnx_mean_pool_graph,
            io_map=[
                ("output_0", "embeddings"),
                ("attention_mask", "attention_mask_pool"),
            ],
        )

        combined_model = onnx.compose.merge_models(
            onnx_body_with_mean_pooling,
            onnx_head,
            io_map=[("pooled_embeddings", "model_head")],
        )

        onnx.save(combined_model, output)


def _create_mean_pooling_onnx_graph(embedding_size: int) -> onnx.onnx_ml_pb2.ModelProto:
    """ONNX graph definition of mean pooling used by SetFit.

    Implements the following mean pooling method in onnx primitives.

    def mean_pooling(model_output: np.array, attention_mask: np.array):
        token_embeddings = model_output[0]
        input_mask_expanded = np.broadcast_to(
            np.expand_dims(attention_mask, axis=2), token_embeddings.shape
        )
        sum_embeddings = np.sum(input_mask_expanded * token_embeddings, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(1), 1e-9, sys.maxsize)
        return sum_embeddings / sum_mask

    :param embedding_size: The size of the embedding layer. This is needed to set
        the shape of the broadcasts in the ONNX operations.
    :return: An onnx graph
    """

    # Get the shape from the model output_0 and map it to embeddings
    embedding_shape_node = onnx.helper.make_node(
        "Shape",
        inputs=["embeddings"],
        outputs=["embedding_shape"],
    )

    # Reshape attention_mask to (batch_size, sequence_len, 1)
    # reshaped = np.expand_dims(attention_mask, axis=2)
    cast_attention_node = onnx.helper.make_node(
        "Cast",
        inputs=["attention_mask_pool"],
        outputs=["attention_mask_pool_float"],
        to=getattr(onnx.TensorProto, "FLOAT"),
    )
    unsqueeze_constant_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["unsqueeze_constant"],
        value=onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.INT64,
            dims=[
                1,
            ],
            vals=np.array([2]),
        ),
    )
    attention_unsqueeze_node = onnx.helper.make_node(
        "Unsqueeze",
        inputs=["attention_mask_pool_float", "unsqueeze_constant"],
        outputs=["attention_unsqueezed"],
    )

    # Expand the attention_mask
    # input_mask_expanded = np.broadcast_to(reshaped, token_embeddings.shape)
    expanded_attention_mask_node = onnx.helper.make_node(
        "Expand",
        inputs=["attention_unsqueezed", "embedding_shape"],
        outputs=["expanded_attention_mask"],
    )

    # multiply the expanded_attention_mask by the embeddings (embeddings)
    # multiplied = input_mask_expanded * token_embeddings
    masked_embeddings_node = onnx.helper.make_node(
        "Mul",
        inputs=["expanded_attention_mask", "embeddings"],
        outputs=["masked_embeddings"],
    )

    # Sum across the sequence elements
    # sum_embeddings = np.sum(input_mask_expanded * token_embeddings, axis=1)
    sum_constant_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["sum_constant"],
        value=onnx.helper.make_tensor(
            name="sum_const_tensor",
            data_type=onnx.TensorProto.INT64,
            dims=[
                1,
            ],
            vals=np.array([1]),
        ),
    )
    sum_node = onnx.helper.make_node(
        "ReduceSum",
        inputs=["masked_embeddings", "sum_constant"],
        outputs=["sum_embeddings"],
        keepdims=0,
    )

    # Sum the number of attended tokens
    # attention_sum = input_mask_expanded.sum(axis=1)
    sum_attention_node = onnx.helper.make_node(
        "ReduceSum",
        inputs=["expanded_attention_mask", "sum_constant"],
        outputs=["sum_attention"],
        keepdims=0,
    )

    # Clip the sum to the 0
    # sum_mask = np.clip(attention_sum, 1e-9, sys.maxsize)
    min_clip_constant_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["min_clip_constant"],
        value=onnx.helper.make_tensor(
            name="min_clip_const_tensor",
            data_type=onnx.TensorProto.FLOAT,
            dims=[
                1,
            ],
            vals=np.array([1e-6]),
        ),
    )
    max_clip_constant_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["max_clip_constant"],
        value=onnx.helper.make_tensor(
            name="max_clip_const_tensor",
            data_type=onnx.TensorProto.FLOAT,
            dims=[
                1,
            ],
            vals=np.array([1e6]),
        ),
    )
    clipped_attention_node = onnx.helper.make_node(
        "Clip",
        inputs=["sum_attention", "min_clip_constant", "max_clip_constant"],
        outputs=["clipped_attention_sum"],
    )

    # Divide the embeddings by the mask
    # sum_embeddings / sum_mask
    pooled_embeddings_node = onnx.helper.make_node(
        "Div",
        inputs=["sum_embeddings", "clipped_attention_sum"],
        outputs=["pooled_embeddings"],
    )

    graph = onnx.helper.make_graph(
        nodes=[
            embedding_shape_node,
            cast_attention_node,
            unsqueeze_constant_node,
            attention_unsqueeze_node,
            expanded_attention_mask_node,
            masked_embeddings_node,
            sum_constant_node,
            sum_node,
            sum_attention_node,
            min_clip_constant_node,
            max_clip_constant_node,
            clipped_attention_node,
            pooled_embeddings_node,
        ],
        name="mean_pooling",
        inputs=[
            onnx.helper.make_tensor_value_info(
                "embeddings", onnx.TensorProto.FLOAT, [None, None, embedding_size]
            ),
            onnx.helper.make_tensor_value_info(
                "attention_mask_pool", onnx.TensorProto.FLOAT, [None, None]
            ),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info(
                "pooled_embeddings", onnx.TensorProto.FLOAT, [None, embedding_size]
            )
        ],
    )

    onnx_model = onnx.helper.make_model_gen_version(graph)
    return onnx_model
