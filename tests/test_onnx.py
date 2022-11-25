import os

import numpy as np
import onnxruntime
import torch
from transformers import AutoTokenizer

from setfit import SetFitModel
from setfit.onnx import export_onnx


def test_export_onnx_sklearn_head():
    """Test that the exported `ONNX` model returns the same predictions as the original model."""
    model_path = "lewtun/my-awesome-setfit-model"
    model = SetFitModel.from_pretrained(model_path)

    # Export the sklearn based model
    output_path = "model.onnx"
    export_onnx(model.model_body, model.model_head, opset=12, output_path=output_path)

    # Check that the model was saved.
    assert output_path in os.listdir(), "Model not saved to output_path"

    # Run inference using the original model.
    input_text = ["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"]
    pytorch_preds = model(input_text)
    pytorch_preds

    # Run inference using the exported onnx model.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(
        input_text,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="np",
    )

    session = onnxruntime.InferenceSession(output_path)

    onnx_preds = session.run(None, dict(inputs))[0]

    # Compare the results and ensure that we get the same predictions.
    assert np.array_equal(onnx_preds, pytorch_preds)

    # Cleanup the model.
    os.remove(output_path)
