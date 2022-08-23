# Transformers Baselines

This folder contains the scripts used to train the 🤗 Transformers baselines quoted in the SetFit paper [ADD LINK].

## Setup

To run the scripts, first create a Python virtual environment, e.g. with `conda`:

```
conda create -n baselines-transformers python=3.9 && conda activate baselines-transformers
```

Next, install the required dependencies

```
python -m pip install setfit
python -m pip install -r requirements.txt
```

## Usage

### Fewshot finetuning

To finetune a pretrained model on a single dataset under the SetFit organization, run:

```
python run_fewshot.py train-single-dataset \
--model-id=distilbert-base-uncased \
--dataset-id=sst2 \
--metric=accuracy \
--learning-rate=2e-5 \
--batch-size=4
```

To finetune a pretrained model on all the test datasets used in SetFit, run:

```
python run_fewshot.py train-all-datasets --model-ckpt=distilbert-base-uncased --batch-size=4
```

### Full finetuning

To finetune a pretrained model on a single dataset under the SetFit organization, run:

```
python run_full.py train-single-dataset \
--model-id=distilbert-base-uncased \
--dataset-id=sst2 \
--metric=accuracy \
--learning-rate=2e-5 \
--batch-size=24
```

To finetune a pretrained model on all the test datasets used in SetFit, run:

```
python run_full.py train-all-datasets --model-id=distilbert-base-uncased --batch-size=24
```

### Multilingual finetuning

We provide three different ways to run SetFit in multilingual settings:

* `each`: train on data in target language
* `en`: train on English data only
* `all`: train on data in all languages

To finetune a baseline in one of these setting, run:

```
python run_fewshot_multilingual.py train-single-dataset \
--model-id=xlm-roberta-base \
--dataset-id=amazon_reviews_multi_en \
--metric=mae \
--learning-rate=2e-5 \
--batch-size=4 \
--multilinguality=each
```

To finetune a baseline on all the multilingual test sets in the paper, run:

```
python run_fewshot_multilingual.py train-all-datasets \
    --model=xlm-roberta-base \
    --learning-rate=2e-5 \
    --batch-size=4 \
    --multilinguality=each
```

### Inference benchmark

To run the inference benchmark, run:

```
python run_inference.py --model-id=distilbert-base-uncased__sst2__train-16-4 --dataset-id=sst2 --num-samples=100
```