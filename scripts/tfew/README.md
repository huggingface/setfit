# Running T-Few

These scripts run the baselines based on the `T-Few` paper: [_Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning_](https://arxiv.org/abs/2205.05638).

## Setup

To run the scripts, first create a Python virtual environment, e.g. with `conda`:

```
conda create -n baselines-tfew python=3.7 && conda activate baselines-tfew
```

Next, clone our `T-Few` fork, and install the required dependencies:

```
cd scripts/tfew
git clone https://github.com/SetFit/t-few.git
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
```

The steps above only need to be done once. In addition, every time you start a new session, you will need to run:
```
cd scripts/tfew
. t-few/bin/start.sh
```
This sets up some required environment variables, including `PYTHONPATH`, `OUTPUT_PATH` (where results will be saved) and `CONFIG_PATH` (where the config `.json` files are stored).
It also sets `CUDA_VISIBLE_DEVICES=0`. To use a different GPU, edit the file `t-few/bin/start.sh`.

## Usage example

To train and evaluate `T-Few` (3B) on 8 examples (per class) on the `sst2` dataset, run:

```
python -m t-few.src.pl_train \
        -c t03b.json+ia3.json+emotion.json \
        -k load_weight="t-few/pretrained_checkpoints/t03b_ia3_finish.pt" \
        exp_name=tfew_03b_pretrained/emotion/train-8 \
        num_shot=8 \
        batch_size=1 \
        eval_batch_size=2 \
        grad_accum_factor=8 \
```

This will fine-tune the 3 billion parameter pretrained model using the (IA)^3 method from the `T-Few` paper, and then run the evaluation. For all our baselines, we use the default settings from the `T-Few` paper. `T-Few` comes in 2 versions: one with a 3 billion (3B) base model, and one with an 11 billion (11B) base model.

You can run `T-Few` (3B) over all the supported test datasets in the `SetFit` paper by running:

```
./run_tfew_test.sh
```

Similarly, to run `T-Few` (11B) over all test datasets, run:

```
./run_tfew_11b.sh
```

Results will be saved to the `scripts/tfew/results` directory. 
The results are comprised of 10 directories, one for each training split.
Each of these directories contains 5 results, one for each randomly selected training prompt.
To retrieve the median score across all prompts (for each split), run the following on each dataset:

```
python scripts/create_summary_table.py --path scripts/tfew/results/{experiment_name}
```

The summary table will be saved in `results/{experiment_name}`.
