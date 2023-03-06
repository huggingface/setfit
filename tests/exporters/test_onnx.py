import os

import numpy as np
import onnxruntime
import pytest
from transformers import AutoTokenizer

from setfit import SetFitModel
from setfit.data import get_templated_dataset
from setfit.exporters.onnx import export_onnx
from setfit.trainer import Trainer
from setfit.training_args import TrainingArguments


def test_export_onnx_sklearn_head():
    """Test that the exported `ONNX` model returns the same predictions as the original model."""
    model_path = "lewtun/my-awesome-setfit-model"
    model = SetFitModel.from_pretrained(model_path)

    # Export the sklearn based model
    output_path = "model.onnx"
    try:
        export_onnx(model.model_body, model.model_head, opset=12, output_path=output_path)

        # Check that the model was saved.
        assert output_path in os.listdir(), "Model not saved to output_path"

        # Run inference using the original model.
        input_text = ["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"]
        pytorch_preds = model(input_text)

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
        # Map inputs to int64 from int32
        inputs = {key: value.astype("int64") for key, value in inputs.items()}

        session = onnxruntime.InferenceSession(output_path)

        onnx_preds = session.run(None, dict(inputs))[0]

        # Compare the results and ensure that we get the same predictions.
        assert np.array_equal(onnx_preds, pytorch_preds)

    finally:
        # Cleanup the model.
        os.remove(output_path)


@pytest.mark.skip("ONNX exporting of SetFit model with Torch head not yet supported.")
@pytest.mark.parametrize("out_features", [1, 2, 3])
def test_export_onnx_torch_head(out_features):
    """Test that the exported `ONNX` model returns the same predictions as the original model."""
    dataset = get_templated_dataset(reference_dataset="SetFit/SentEval-CR")
    model_path = "sentence-transformers/paraphrase-albert-small-v2"
    model = SetFitModel.from_pretrained(
        model_path, use_differentiable_head=True, head_params={"out_features": out_features}
    )

    args = TrainingArguments(
        num_iterations=15,
        num_epochs=(1, 15),
        batch_size=16,
        body_learning_rate=(2e-5, 1e-5),
        head_learning_rate=1e-2,
        l2_weight=0.0,
        end_to_end=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset,
        column_mapping={"text": "text", "label": "label"},
    )
    trainer.train()

    # Export the sklearn based model
    output_path = "model.onnx"
    try:
        export_onnx(model.model_body, model.model_head, opset=12, output_path=output_path)

        # Check that the model was saved.
        assert output_path in os.listdir(), "Model not saved to output_path"

        # Run inference using the original model.
        input_text = ["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"]
        pytorch_preds = model(input_text)

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
        # Map inputs to int64 from int32
        inputs = {key: value.astype("int64") for key, value in inputs.items()}

        session = onnxruntime.InferenceSession(output_path)

        onnx_preds = session.run(None, dict(inputs))[0]

        # Compare the results and ensure that we get the same predictions.
        assert np.array_equal(onnx_preds, pytorch_preds)

    finally:
        # Cleanup the model.
        os.remove(output_path)
