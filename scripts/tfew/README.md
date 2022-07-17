# Running T-Few

These scripts run the method from the `T-Few` paper: "[Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638)".

## Setup

To run the scripts, first create a Python virtual environment, e.g. with `conda`:

```
conda create -n baselines-tfew python=3.7 && conda activate baselines-tfew
```

Next, clone our `T-Few` fork, and install the required dependencies:

```
cd scripts/tfew
git clone https://github.com/SetFit/t-few.git
mv t-few/.git t-few/git
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
The steps above only need to be done once. In addition, every time you start a new session, you will need to run:
```
. t-few/bin/start.sh
```

## Usage

To train and evaluate `T-Few` on selected datasets and seeds:
```
export DATASETS=(sst2 rte ag-news)
export SEEDS=(0 1 2 3 4)
python run_tfew.sh -e experiment_name
```

Results will be saved to `tfew/results_orig/experiment_name`. 

To create the summary table of results:
```
python create_tfew_sumary_table.py results_orig/experiment_name
```

The summary table will be saved in `tfew/results/experiment_name`.
