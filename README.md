<img src="assets/setfit.png">

<p align="center">
    🤗 <a href="https://huggingface.co/setfit" target="_blank">Models & Datasets</a> | 📖 <a href="https://huggingface.co/blog/setfit" target="_blank">Blog</a> | 📃 Paper (coming soon!)</a>
</p>

# SetFit - Efficient Few-shot Learning with Sentence Transformers

We introduce SetFit, an efficient and prompt-free framework for few-shot fine-tuning of [Sentence Transformers](https://sbert.net/). Compared to other few-shot learning methods, SetFit has several unique features:

* 📈 **High accuracy with little labeled data:** SetFit achieves comparable (or better) results than current state-of-the-art methods for text classification. For example, with only 8 labelled examples per class on the CR sentiment dataset, SetFit is competitive with fine-tuning RoBERTa-large on the full training set of 3k examples.
* 🗣 **No prompts or verbalisers:** Current techniques for few-shot fine-tuning require handcrafted prompts or verbalisers to convert examples into a format that's suitable for the underlying language model. SetFit dispenses with prompts altogether by generating rich embeddings directly from text examples.
* 🏎 **Fast to train:** SetFit doesn't require large-scale models like T0 or GPT-3 to achieve high accuracy. As a result, it is typically an order of magnitude (or more) faster to train and run inference with.

## Getting started

### Installation

Download and install `setfit` by running:

```bash
python -m pip install setfit
```

### Training a SetFit model

`setfit` is integrated with the [Hugging Face Hub](https://huggingface.co/) and provides two main classes:

* `SetFitModel`: a wrapper that combines a pretrained body from `sentence_transformers` and a classification head from `scikit-learn`
* `SetFitTrainer`: a helper class that wraps the fine-tuning process of SetFit.

Here is an end-to-end example:


```python
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, SetFitTrainer


# Load a dataset from the Hugging Face Hub
dataset = load_dataset("emotion")

# Simulate the few-shot regime by sampling 8 examples per class
num_classes = 6
train_ds = dataset["train"].shuffle(seed=42).select(range(8 * num_classes))
test_ds = dataset["test"]

# Load SetFit model from Hub
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    loss_class=CosineSimilarityLoss,
    batch_size=16,
    num_iterations=20, # The number of text pairs to generate
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate()

# Push model to the Hub
trainer.push_to_hub("my-awesome-setfit-model")
```

For more examples, check out the `notebooks/` folder.

## Reproducing the results from the paper

We provide scripts to reproduce the results for SetFit and various baselines presented in Table 2 of our paper. Checkout the setup and training instructions in the `scripts/` directory.

## Developer installation

To run the code in this project, first create a Python virtual environment using e.g. Conda:

```bash
conda create -n setfit python=3.9 && conda activate setfit
```

Then install the base requirements with:

```bash
python -m pip install -e '.[dev]'
```

This will install `datasets` and packages like `black` and `isort` that we use to ensure consistent code formatting. Next, go to one of the dedicated baseline directories and install the extra dependencies, e.g.

```bash
cd scripts/setfit
python -m pip install -r requirements.txt
```

### Formatting your code

We use `black` and `isort` to ensure consistent code formatting. After following the installation steps, you can check your code locally by running:

```
make style && make quality
```



## Project structure

```
├── LICENSE
├── Makefile        <- Makefile with commands like `make style` or `make tests`
├── README.md       <- The top-level README for developers using this project.
├── notebooks       <- Jupyter notebooks.
├── final_results   <- Model predictions from the paper
├── scripts         <- Scripts for training and inference
├── setup.cfg       <- Configuration file to define package metadata
├── setup.py        <- Make this project pip installable with `pip install -e`
├── src             <- Source code for SetFit
└── tests           <- Unit tests
```


## Citation

[Coming soon!]

