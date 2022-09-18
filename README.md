# SetFit

Official repository for SetFit.

## Getting started

```python
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit.modeling import SetFitModel
from setfit.trainer import SetFitTrainer


# Load a dataset
dataset = load_dataset("SetFit/SentEval-CR")

# Select N examples per class (8 in this case)
train_ds = dataset["train"].shuffle(seed=42).select(range(8 * 2))
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
    num_epochs=20,
)

# Train and evaluate!
trainer.train()
metrics = trainer.evaluate()

# Push model to the Hub
trainer.push_to_hub("my-awesome-setfit-model")
```

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
├── Makefile    <- Makefile with commands like `make style` or `make tests`
├── README.md   <- The top-level README for developers using this project.
├── notebooks   <- Jupyter notebooks.
├── final_results     <- Model predictions from the paper
├── scripts     <- Scripts for training and inference
├── setup.cfg   <- Configuration file to define package metadata
├── setup.py    <- Make this project pip installable with `pip install -e`
├── src         <- Source code for SetFit
└── tests       <- Unit tests
```


## Citation

