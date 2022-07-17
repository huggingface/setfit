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

To train and evaluate `T-Few` on 8 and 16 examples (per class) on the `sst2` and `ag-news` datasets, run:

```
python run_tfew.py --sample_sizes=8 16 --datasets=sst2 ag-news
```

This will fine-tune 3 billion pre-trained (IA)^3 on all specified datasets. Results will be saved in the `results` directory. To run `T-Few` across all the development datasets used in the paper, run:

```
python run_tfew.py --sample_sizes=8 16 32 --is_dev_set=true
```

Similarly, you can run `SetFit` over all the test datasets in the paper by running:

```
python run_tfew.py --sample_sizes=8 16 32 --is_test_set=true
```

To create the summary table of results:
```
python create_summary_table.py results/experiment_name
```

The summary table will be saved in `results/experiment_name`.
