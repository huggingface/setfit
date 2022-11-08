import pandas as pd

GITHUB_PATH_PREFIX = "huggingface/setfit/blob/main/notebooks/"

CHAPTER_TO_NB = {
    "Text Classification (Multiclass)": "text-classification",
    "Text Classification (Multilabel)": "text-classification_multilabel",
    "Zero-Shot Text Classification": "zero-shot-classification",
    "Hyperparameter search": "text-classification_hyperparameter-search",
}


def _find_text_in_file(filename, start_prompt, end_prompt):
    """
    Find the text in `filename` between a line beginning with `start_prompt` and before `end_prompt`, removing empty
    lines.

    Copied from: https://github.com/huggingface/transformers/blob/16f0b7d72c6d4e122957392c342b074aa2c5c519/utils/check_table.py#L30
    """
    with open(filename, "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()
    # Find the start prompt.
    start_index = 0
    while not lines[start_index].startswith(start_prompt):
        start_index += 1
    start_index += 1

    end_index = start_index
    while not lines[end_index].startswith(end_prompt):
        end_index += 1
    end_index -= 1

    while len(lines[start_index]) <= 1:
        start_index += 1
    while len(lines[end_index]) <= 1:
        end_index -= 1
    end_index += 1
    return "".join(lines[start_index:end_index]), start_index, end_index, lines


def create_table():
    data = {"Notebook": [], "Colab": [], "Kaggle": [], "Gradient": [], "Studio Lab": []}
    for title, nb in CHAPTER_TO_NB.items():
        nb_path = f"{GITHUB_PATH_PREFIX}{nb}.ipynb"
        data["Notebook"].append(title)
        data["Colab"].append(
            f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{nb_path})"
        )
        data["Kaggle"].append(
            f"[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/{nb_path})"
        )
        data["Gradient"].append(
            f"[![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/{nb_path})"
        )
        data["Studio Lab"].append(
            f"[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/{nb_path})"
        )
    return pd.DataFrame(data).to_markdown(index=False) + "\n"


def main():
    table = create_table()
    _, start_index, end_index, lines = _find_text_in_file(
        filename="notebooks/README.md",
        start_prompt="<!--This table is automatically generated, do not fill manually!-->",
        end_prompt="<!--End of table-->",
    )

    with open("notebooks/README.md", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines[:start_index] + [table] + lines[end_index:])


if __name__ == "__main__":
    main()
