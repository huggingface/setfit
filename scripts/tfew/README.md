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
git clone https://github.com/SetFit/t-few.git && mv t-few/.git t-few/git
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
```
The steps above only need to be done once. In addition, every time you start a new session, you will need to run:
```
. t-few/bin/start.sh
```
This sets up some required environment variables, including `PYTHONPATH`, `OUTPUT_PATH` (where results will be saved) and `CONFIG_PATH` (where the config `.json` files are stored)

## Usage example

To train and evaluate `T-Few` on 8 examples (per class) on the `sst2` dataset, run:

```
export CUDA_VISIBLE_DEVICES=0
python -m t-few.src.pl_train \
        -c t03b.json+ia3.json+sst2.json \
        -k load_weight="t-few/pretrained_checkpoints/t03b_ia3_finish.pt" \
        exp_name=tfew_03b_pretrained/sst2/train-8 \
        num_shot=8
```

This will fine-tune the 3 billion parameter pretrained model using the (IA)^3 method from the `T-Few` paper, and then run the evaluation. For all our baselines, we use the default settings from the `T-Few` paper.
Results will be saved to the `results` directory. 

To run `T-Few` across all the development datasets used in the `SetFit` paper:

```
./run_tfew_dev.sh
```

Similarly, you can run `T-Few` over all the test datasets in the `SetFit` paper by running:

```
./run_tfew_test.sh
```

To create the summary table of results with average metrics per dataset:
```
python scripts/create_summary_table.py scripts/tfew/results/experiment_name
```

The summary table will be saved in `results/experiment_name`.
