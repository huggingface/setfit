# ADAPET for SetFit #
This is a fork of the original ADAPET, which can be found [here](https://github.com/rrmenon10/ADAPET).

Our results were created in Python 3.6.8 with a 40 GB NVIDIA A100 Tensor Core GPU

To setup, please clone the repo and follow the instructions below.
````
python3.6 -m venv venv
source mvenv/bin/activate
pip install -r requirements.txt
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
````



To run ADAPET on SetFit datasets, specify the dataset and PLM as argument to python. Additionally, if you wish to
run on a multilingual dataset, then you can choose to prompt/verbalize in the English or the language in question.
You can also remove the prompt.

For example, if you wish to run ADAPET on `sst2` with a prompt and with `albert-xxlarge-v2` as the PLM, simply run the following
```
python setfit_adapet.py --pretrained_weight="albert-xxlarge-v2"\
                --english=True\
                --prompt=True\
                --task_name='SetFit/sst2'\
```

If you wish to run ADAPET on amazon_reviews_multi_ja, prompt in Japanese, with mdeberta-base
```
python setfit_adapet.py --pretrained_weight="microsoft/mdeberta-v3-base"\
                --english=False\
                --prompt=True\
                --task_name='SetFit/amazon_reviews_multi_ja'\
```

This will run ADAPET and evaluate it on the test set for the 8 and 64 splits.

In the multilingual case, ADAPET runs in the "each" scenario as described in the paper by default. You can change this by adding a multilingual argument, such as

```
python setfit_adapet.py --pretrained_weight="microsoft/mdeberta-v3-base"\
                --english=False\
                --prompt=True\
                --task_name='SetFit/amazon_reviews_multi_ja'\
                --multilingual='all'\
```


Note that with default hyperparameters this will take a very long time. If you change to distilbert-base-uncase or another smaller model, ADAPET will be much faster.

Once ADAPET is done, the results will be written as follows
```
seed_output / model_name / dataset_name / train-{num_samples}-{split_dx} / results.json
```
 
For non-English datasets, the plm may have different endings. The ending describes the prompting situation:
````
{plm}__eng_prompt == prompt and verbalize in english
{plm}__lang_prompt == prompt and verbalize in the language in question.
{plm}__lang_no-prompt == take the prompt away and verbalize in the language in question
{plm}__eng_no-prompt == take the prompt away and verbalize english
````
