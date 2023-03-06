<img src="https://raw.githubusercontent.com/huggingface/setfit/main/assets/setfit.png">

<p align="center">
    🤗 <a href="https://huggingface.co/setfit" target="_blank">Models & Datasets</a> | 📖 <a href="https://huggingface.co/blog/setfit" target="_blank">Blog</a> | 📃 <a href="https://arxiv.org/abs/2209.11055" target="_blank">Paper</a>
</p>

# SetFit - Efficient Few-shot Learning with Sentence Transformers

SetFit is an efficient and prompt-free framework for few-shot fine-tuning of [Sentence Transformers](https://sbert.net/). It achieves high accuracy with little labeled data - for instance, with only 8 labeled examples per class on the Customer Reviews sentiment dataset, SetFit is competitive with fine-tuning RoBERTa Large on the full training set of 3k examples 🤯!


Compared to other few-shot learning methods, SetFit has several unique features:

* 🗣 **No prompts or verbalisers:** Current techniques for few-shot fine-tuning require handcrafted prompts or verbalisers to convert examples into a format that's suitable for the underlying language model. SetFit dispenses with prompts altogether by generating rich embeddings directly from text examples.
* 🏎 **Fast to train:** SetFit doesn't require large-scale models like T0 or GPT-3 to achieve high accuracy. As a result, it is typically an order of magnitude (or more) faster to train and run inference with.
* 🌎 **Multilingual support**: SetFit can be used with any [Sentence Transformer](https://huggingface.co/models?library=sentence-transformers&sort=downloads) on the Hub, which means you can classify text in multiple languages by simply fine-tuning a multilingual checkpoint.

## Installation

Download and install `setfit` by running:

```bash
python -m pip install setfit
```

If you want the bleeding-edge version, install from source by running:

```bash
python -m pip install git+https://github.com/huggingface/setfit.git
```

## Usage

The examples below provide a quick overview on the various features supported in `setfit`. For more examples, check out the [`notebooks`](https://github.com/huggingface/setfit/tree/main/notebooks) folder.


### Training a SetFit model

`setfit` is integrated with the [Hugging Face Hub](https://huggingface.co/) and provides two main classes:

* `SetFitModel`: a wrapper that combines a pretrained body from `sentence_transformers` and a classification head from either [`scikit-learn`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) or [`SetFitHead`](https://github.com/huggingface/setfit/blob/main/src/setfit/modeling.py) (a differentiable head built upon `PyTorch` with similar APIs to `sentence_transformers`).
* `SetFitTrainer`: a helper class that wraps the fine-tuning process of SetFit.

Here is an end-to-end example using a classification head from `scikit-learn`:


```python
from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset


# Load a dataset from the Hugging Face Hub
dataset = load_dataset("sst2")

# Simulate the few-shot regime by sampling 8 examples per class
train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=8)
eval_dataset = dataset["validation"]

# Load a SetFit model from Hub
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

args = TrainingArguments(
    batch_size=16,
    num_iterations=20,  # The number of text pairs to generate for contrastive learning
    num_epochs=1  # The number of epochs to use for contrastive learning
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    metric="accuracy",
    column_mapping={"sentence": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate()

# Push model to the Hub
trainer.push_to_hub("my-awesome-setfit-model")

# Download from Hub
model = SetFitModel.from_pretrained("lewtun/my-awesome-setfit-model")
# Run inference
preds = model(["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"])
```

Here is an end-to-end example using `SetFitHead`:


```python
from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset


# Load a dataset from the Hugging Face Hub
dataset = load_dataset("sst2")

# Simulate the few-shot regime by sampling 8 examples per class
train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=8)
eval_dataset = dataset["validation"]
num_classes = 2

# Load a SetFit model from Hub
model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-mpnet-base-v2",
    use_differentiable_head=True,
    head_params={"out_features": num_classes},
)

args = TrainingArguments(
    body_learning_rate=2e-5,
    head_learning_rate=1e-2,
    batch_size=16,
    num_iterations=20,  # The number of text pairs to generate for contrastive learning
    num_epochs=(1, 25),  # For finetuning the embeddings and training the classifier, respectively
    l2_weight=0.0,
    end_to_end=False,  # Don't train the classifier end-to-end, i.e. only train the head
)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    metric="accuracy",
    column_mapping={"sentence": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate()

# Push model to the Hub
trainer.push_to_hub("my-awesome-setfit-model")

# Download from Hub and run inference
model = SetFitModel.from_pretrained("lewtun/my-awesome-setfit-model")
# Run inference
preds = model(["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"])
```

Based on our experiments, `SetFitHead` can achieve similar performance as using a `scikit-learn` head. We use `AdamW` as the optimizer and scale down learning rates by 0.5 every 5 epochs. For more details about the experiments, please check out [here](https://github.com/huggingface/setfit/pull/112#issuecomment-1295773537). We recommend using a large learning rate (e.g. `1e-2`) for `SetFitHead` and a small learning rate (e.g. `1e-5`) for the body in your first attempt.

### Training on multilabel datasets

To train SetFit models on multilabel datasets, specify the `multi_target_strategy` argument when loading the pretrained model:

#### Example using a classification head from `scikit-learn`:

```python
from setfit import SetFitModel

model = SetFitModel.from_pretrained(
    model_id,
    multi_target_strategy="one-vs-rest",
)
```

This will initialise a multilabel classification head from `sklearn` - the following options are available for `multi_target_strategy`:

* `one-vs-rest`: uses a `OneVsRestClassifier` head.
* `multi-output`: uses a `MultiOutputClassifier` head.
* `classifier-chain`: uses a `ClassifierChain` head.

From here, you can instantiate a `Trainer` using the same example above, and train it as usual.

#### Example using the differentiable `SetFitHead`:

```python
from setfit import SetFitModel

model = SetFitModel.from_pretrained(
    model_id,
    multi_target_strategy="one-vs-rest"
    use_differentiable_head=True,
    head_params={"out_features": num_classes},
)
```
**Note:** If you use the differentiable `SetFitHead` classifier head, it will automatically use `BCEWithLogitsLoss` for training. The prediction involves a `sigmoid` after which probabilities are rounded to 1 or 0. Furthermore, the `"one-vs-rest"` and `"multi-output"` multi-target strategies are equivalent for the differentiable `SetFitHead`.

### Zero-shot text classification

SetFit can also be applied to scenarios where no labels are available. To do so, create a synthetic dataset of training examples:

```python
from setfit import get_templated_dataset

candidate_labels = ["negative", "positive"]
train_dataset = get_templated_dataset(candidate_labels=candidate_labels, sample_size=8)
```

This will create examples of the form `"This sentence is {}"`, where the `{}` is filled in with one of the candidate labels. From here you can train a SetFit model as usual:

```python
from setfit import SetFitModel, Trainer

model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
trainer = Trainer(
    model=model,
    train_dataset=train_dataset
)
trainer.train()
```

We find this approach typically outperforms the [zero-shot pipeline](https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline) in 🤗 Transformers (based on MNLI with BART), while being 5x faster to generate predictions with.


### Running hyperparameter search

`Trainer` provides a `hyperparameter_search()` method that you can use to find good hyperparameters for your data. To use this feature, first install the `optuna` backend:

```bash
python -m pip install setfit[optuna]
```

To use this method, you need to define two functions:

* `model_init()`: A function that instantiates the model to be used. If provided, each call to `train()` will start from a new instance of the model as given by this function.
* `hp_space()`: A function that defines the hyperparameter search space.

Here is an example of a `model_init()` function that we'll use to scan over the hyperparameters associated with the classification head in `SetFitModel`:

```python
from setfit import SetFitModel

def model_init(params):
    params = params or {}
    max_iter = params.get("max_iter", 100)
    solver = params.get("solver", "liblinear")
    params = {
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        }
    }
    return SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2", **params)
```

Similarly, to scan over hyperparameters associated with the SetFit training process, we can define a `hp_space()` function as follows:

```python
def hp_space(trial):  # Training parameters
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 1, 5),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64]),
        "seed": trial.suggest_int("seed", 1, 40),
        "num_iterations": trial.suggest_categorical("num_iterations", [5, 10, 20]),
        "max_iter": trial.suggest_int("max_iter", 50, 300),
        "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
    }
```

**Note:** In practice, we found `num_iterations` to be the most important hyperparameter for the contrastive learning process.

The next step is to instantiate a `Trainer` and call `hyperparameter_search()`:

```python
from datasets import Dataset
from setfit import Trainer

dataset = Dataset.from_dict(
    {"text_new": ["a", "b", "c"], "label_new": [0, 1, 2], "extra_column": ["d", "e", "f"]}
)

trainer = Trainer(
    train_dataset=dataset,
    eval_dataset=dataset,
    model_init=model_init,
    column_mapping={"text_new": "text", "label_new": "label"},
)
best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=5)
```

Finally, you can apply the hyperparameters you found to the trainer, and lock in the optimal model, before training for
a final time.

```python
trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
trainer.train()
```

## Compressing a SetFit model with knowledge distillation

If you have access to unlabeled data, you can use knowledge distillation to compress a trained SetFit model into a smaller version. The result is a model that can run inference much faster, with little to no drop in accuracy. Here's an end-to-end example (see our paper for more details):

```python
from datasets import load_dataset
from setfit import SetFitModel, Trainer, DistillationTrainer, sample_dataset
from setfit.training_args import TrainingArguments

# Load a dataset from the Hugging Face Hub
dataset = load_dataset("ag_news")

# Create a sample few-shot dataset to train the teacher model
train_dataset_teacher = sample_dataset(dataset["train"], label_column="label", num_samples=16)
# Create a dataset of unlabeled examples to train the student
train_dataset_student = dataset["train"].shuffle(seed=0).select(range(500))
# Dataset for evaluation
eval_dataset = dataset["test"]

# Load teacher model
teacher_model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-mpnet-base-v2"
)

# Create trainer for teacher model
teacher_trainer = Trainer(
    model=teacher_model,
    train_dataset=train_dataset_teacher,
    eval_dataset=eval_dataset,
)

# Train teacher model
teacher_trainer.train()
teacher_metrics = teacher_trainer.evaluate()

# Load small student model
student_model = SetFitModel.from_pretrained("paraphrase-MiniLM-L3-v2")

args = TrainingArguments(
    batch_size=16,
    num_iterations=20,
    num_epochs=1
)

# Create trainer for knowledge distillation
student_trainer = DistillationTrainer(
    teacher_model=teacher_model,
    student_model=student_model,
    args=args,
    train_dataset=train_dataset_student,
    eval_dataset=eval_dataset,
)

# Train student with knowledge distillation
student_trainer.train()
student_metrics = student_trainer.evaluate()
```


## Reproducing the results from the paper

We provide scripts to reproduce the results for SetFit and various baselines presented in Table 2 of our paper. Check out the setup and training instructions in the `scripts/` directory.

## Developer installation

To run the code in this project, first create a Python virtual environment using e.g. Conda:

```bash
conda create -n setfit python=3.9 && conda activate setfit
```

Then install the base requirements with:

```bash
python -m pip install -e '.[dev]'
```

This will install `datasets` and packages like `black` and `isort` that we use to ensure consistent code formatting.

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

## Related work

* [jxpress/setfit-pytorch-lightning](https://github.com/jxpress/setfit-pytorch-lightning) - A PyTorch Lightning implementation of SetFit.

## Citation

```
@misc{https://doi.org/10.48550/arxiv.2209.11055,
  doi = {10.48550/ARXIV.2209.11055},
  url = {https://arxiv.org/abs/2209.11055},
  author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Efficient Few-Shot Learning Without Prompts},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
