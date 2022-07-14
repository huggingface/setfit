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
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
The steps above only needs to be done once. In addition, every time you start a new session, you will need to run:
```
. t-few/bin/start.sh
```

## Usage

To train and evaluate `T-Few` on the `sst2` dataset (using random seed=0), run:

```
cd t-few
python -m src.pl_train -c t03b.json+ia3.json+$sst2.json -k load_weight="pretrained_checkpoints/t03b_ia3_finish.pt" exp_name=t03b_$sst2_seed0_ia3_pretrained100k few_shot_random_seed=0 seed=0
```

Results will be saved to `scripts/tfew/results_orig`. 
You can then run the following script to convert the results to the format used in our library:
```
python results_to_setfit_format.py results_orig/t03b_$sst2_seed0_ia3_pretrained100k
```