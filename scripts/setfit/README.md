# Running SetFit

## Setup
See `README.md` in the root folder of this repository.

## Run your first experiment

To train and evaluate `SetFit` on 8 examlpes (per class) from the `sst2`:

```
python -m src.setfit.run_fewshot --sample_sizes=8 --dataset=sst2
```

This will use the default settings used in the paper, including `paraphrase-mpnet-base-v2` as the backbone model. Results will be saved under `results/stefit/`.

## Exaustive example

Following is an example with all argument options and their default values.
Note that you can run on a series of datasets and sample sizes:

```
python -m src.setfit.run_fewshot \
    --model paraphrase-mpnet-base-v2 \
    --datasets sst2 ag_news bbc-news \
    --sample_sizes 8 16 32 \
    --num_epochs 20 \
    --batch_size 16 \
    --max_seq_length 256 \
    --classifier logistic_regression \
    --loss CosineSimilarityLoss \
    --exp_name "" \
    --add_normalization_layer \
```