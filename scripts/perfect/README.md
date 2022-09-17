# Running PERFECT

Follow the steps below to run the baselines based on the `PERFECT` paper: [_PERFECT: Prompt-free and Efficient Few-shot Learning with Language Models_](https://arxiv.org/abs/2204.01172).

## Setup

To get started, first create a Python virtual environment, e.g. with `conda`:

```
conda create -n baselines-perfect python=3.7 && conda activate baselines-perfect
```

Next, clone [our fork](https://github.com/SetFit/perfect) of the [`PERFECT` codebase](https://github.com/facebookresearch/perfect), and install the required dependencies:

```
git clone git+https://github.com/SetFit/perfect.git
cd perfect
python setup.py develop 
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
python -m pip install -r requirements.txt
```

Next, download and process the datasets:

```
wget https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar
tar -xvf datasets.tar
mv original/ datasets
python fewshot/process_datasets.py
```

## Usage example

To train and evaluate `PERFECT` on 8 and 64 examples (per class) across all the SetFit test datasets, run:

```
cd fewshot/
bash scripts/run_setfit_baselines.sh
```