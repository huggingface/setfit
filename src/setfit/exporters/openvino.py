import os

import openvino.runtime as ov

from setfit import SetFitModel
from setfit.exporters.onnx import export_onnx


def export_to_openvino(
    model: SetFitModel,
    output_path: str = "model.xml",
) -> None:
    """Export a PyTorch backed SetFit model to OpenVINO Intermediate Representation.

    Args:
        model_body (`SentenceTransformer`): The model_body from a SetFit model body. This should be a
            SentenceTransformer.
        model_head (`torch.nn.Module` or `LogisticRegression`): The SetFit model head. This can be either a
            dense layer SetFitHead or a Sklearn estimator.
        output_path (`str`): The path where will be stored the generated OpenVINO model. At a minimum it needs to contain
            the name of the final file.
        ignore_ir_version (`bool`): Whether to ignore the IR version used in sklearn. The version is often missmatched
            with the transformer models. Setting this to true coerces the versions to be the same. This might
            cause errors but in practice works.  If this is set to False you need to ensure that the IR versions
            align between the transformer and the sklearn onnx representation.
    """

    # Load the model and get all of the parts.
    OPENVINO_SUPPORTED_OPSET = 13

    model.model_body.cpu()
    onnx_path = output_path.replace(".xml", ".onnx")

    export_onnx(
        model.model_body,
        model.model_head,
        opset=OPENVINO_SUPPORTED_OPSET,
        output_path=onnx_path,
        ignore_ir_version=True,
        use_hummingbird=True,
    )

    # Save the final model.
    ov_model = ov.Core().read_model(onnx_path)
    ov.serialize(ov_model, output_path)

    os.remove(onnx_path)
