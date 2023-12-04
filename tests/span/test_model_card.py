from pathlib import Path

from datasets import load_dataset

from setfit import AbsaModel, AbsaTrainer, SetFitModelCardData, TrainingArguments

from .aspect_model_card_pattern import ASPECT_MODEL_CARD_PATTERN
from .polarity_model_card_pattern import POLARITY_MODEL_CARD_PATTERN


def test_model_card(tmp_path: Path) -> None:
    dataset = load_dataset("tomaarsen/setfit-absa-semeval-laptops", split="train")
    train_dataset = dataset.select(range(10))
    eval_dataset = dataset.select(range(10, 20))
    model = AbsaModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2",
        model_card_data=SetFitModelCardData(
            model_id="tomaarsen/setfit-absa-paraphrase-albert-small-v2-laptops",
            language=["en"],
            license="apache-2.0",
        ),
    )

    args = TrainingArguments(
        str(tmp_path),
        report_to="codecarbon",
        batch_size=1,
        eval_steps=1,
        logging_steps=1,
        max_steps=2,
        evaluation_strategy="steps",
    )
    trainer = AbsaTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.evaluate()
    # for model_type, model in (("aspect", model.aspect_model), ("polarity", model.polarity_model)):
    #     model.create_model_card(tmp_path / model_type, tmp_path / model_type)
    #     with open(tmp_path / model_type / "README.md", "r", encoding="utf8") as f:
    #         model_card = f.read()
    #     with open(f"{model_type}_model_card.md", "w", encoding="utf8") as f:
    #         f.write(model_card)
    #     assert MODEL_CARD_PATTERN.fullmatch(model_card)

    path = tmp_path / "aspect"
    model.aspect_model.create_model_card(path, path)
    with open(path / "README.md", "r", encoding="utf8") as f:
        model_card = f.read()
    assert ASPECT_MODEL_CARD_PATTERN.fullmatch(model_card)

    path = tmp_path / "polarity"
    model.polarity_model.create_model_card(path, path)
    with open(path / "README.md", "r", encoding="utf8") as f:
        model_card = f.read()
    assert POLARITY_MODEL_CARD_PATTERN.fullmatch(model_card)
