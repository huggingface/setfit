import os

import numpy as np
from transformers import AutoTokenizer

import openvino.runtime as ov

from setfit import SetFitModel
from setfit.openvino import export_openvino


def test_export_to_openvino():
    """Test that the exported `OpenVINO` model returns the same predictions as the original model."""
    model_path = "lewtun/my-awesome-setfit-model"
    model = SetFitModel.from_pretrained(model_path)

    # Export the sklearn based model
    output_path = "model.xml"
    export_openvino(model, output_path=output_path)

    # Check that the model was saved.
    assert output_path in os.listdir(), "Model not saved to output_path"

    # Run inference using the original model.
    input_text = ["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"]
    pytorch_preds = model(input_text)

    # Run inference using the exported OpenVINO model.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(
        input_text,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="np",
    )

    inputs_dict = dict(inputs)

    core = ov.Core()
    ov_model = core.read_model(output_path)
    compiled_model = core.compile_model(ov_model, "CPU")

    ov_preds = compiled_model(inputs_dict)[compiled_model.outputs[0]]

    # Compare the results and ensure that we get the same predictions.
    assert np.array_equal(ov_preds, pytorch_preds)
