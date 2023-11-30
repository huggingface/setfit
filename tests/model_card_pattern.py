import re


MODEL_CARD_PATTERN = re.compile(
    """\
---
language:
- en
license: apache-2\.0
library_name: setfit
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
datasets:
- sst2
metrics:
- accuracy
widget:
- text: .*
pipeline_tag: text-classification
inference: true
co2_eq_emissions:
  emissions: [\d\.\-e]+
  source: codecarbon
  training_type: fine-tuning
  on_cloud: (false|true)
  cpu_model: .+
  ram_total_size: [\d\.]+
  hours_used: [\d\.]+
(  hardware_used: .+
)?base_model: sentence-transformers/paraphrase-albert-small-v2
model-index:
- name: SetFit with sentence-transformers\/paraphrase-albert-small-v2 on SST2
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: SST2
      type: sst2
      split: test
    metrics:
    - type: accuracy
      value: [\d\.]+
      name: Accuracy
---

\# SetFit with sentence\-transformers/paraphrase\-albert\-small\-v2 on SST2

This is a \[SetFit\]\(https://github\.com/huggingface/setfit\) model trained on the \[SST2\]\(https://huggingface\.co/datasets/sst2\) dataset that can be used for Text Classification\. This SetFit model uses \[sentence\-transformers/paraphrase\-albert\-small\-v2\]\(https://huggingface\.co/sentence\-transformers/paraphrase\-albert\-small\-v2\) as the Sentence Transformer embedding model\. A \[LogisticRegression\]\(https://scikit\-learn\.org/stable/modules/generated/sklearn\.linear_model\.LogisticRegression\.html\) instance is used for classification\.

The model has been trained using an efficient few\-shot learning technique that involves:

1\. Fine\-tuning a \[Sentence Transformer\]\(https://www\.sbert\.net\) with contrastive learning\.
2\. Training a classification head with features from the fine\-tuned Sentence Transformer\.

## Model Details

### Model Description
- \*\*Model Type:\*\* SetFit
- \*\*Sentence Transformer body:\*\* \[sentence-transformers/paraphrase-albert-small-v2\]\(https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2\)
- \*\*Classification head:\*\* a \[LogisticRegression\]\(https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html\) instance.
- \*\*Maximum Sequence Length:\*\* 100 tokens
- \*\*Number of Classes:\*\* 2 classes
- \*\*Training Dataset:\*\* \[SST2\]\(https://huggingface.co/datasets/sst2\)
- \*\*Language:\*\* en
- \*\*License:\*\* apache-2.0

### Model Sources

- \*\*Repository:\*\* \[SetFit on GitHub\]\(https://github.com/huggingface/setfit\)
- \*\*Paper:\*\* \[Efficient Few-Shot Learning Without Prompts\]\(https://arxiv.org/abs/2209.11055\)
- \*\*Blogpost:\*\* \[SetFit: Efficient Few-Shot Learning Without Prompts\]\(https://huggingface.co/blog/setfit\)

### Model Labels
\| Label\s+\| Examples\s+\|
\|:-+\|:-+\|
\| negative\s+\| [^\|]+ \|
\| positive\s+\| [^\|]+ \|

## Evaluation

### Metrics
\| Label   \| Accuracy \|
\|:--------\|:---------\|
\| \*\*all\*\* \| [\d\.]+\s+\|

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from [^H]+ Hub
model = SetFitModel.from_pretrained\("tomaarsen/setfit-paraphrase-albert-small-v2-sst2"\)
# Run inference
preds = model\(".+"\)
```
<!--
### Downstream Use

\*List how someone could finetune this model on their own dataset\.\*
-->

<!--
### Out-of-Scope Use

\*List how the model may foreseeably be misused and address what users ought not to do with the model\.\*
-->

<!--
## Bias, Risks and Limitations

\*What are the known or foreseeable issues stemming from this model\? You could also flag here known failure cases or weaknesses of the model\.\*
-->

<!--
### Recommendations

\*What are recommendations with respect to the foreseeable issues\? For example, filtering explicit content\.\*
-->

## Training Details

### Training Set Metrics
\| Training set \| Min \| Median  \| Max \|
\|:-------------\|:----\|:--------\|:----\|
\| Word count   \| 2   \| 11.4375 \| 33  \|

\| Label    \| Training Sample Count \|
\|:---------\|:----------------------\|
\| negative \| 8                     \|
\| positive \| 8                     \|

### Training Hyperparameters
- batch_size: \(1, 1\)
- num_epochs: \(1, 16\)
- max_steps: 2
- sampling_strategy: oversampling
- body_learning_rate: \(2e-05, 1e-05\)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- seed: 42
- load_best_model_at_end: False

### Training Results
\| Epoch  \| Step \| Training Loss \| Validation Loss \|
\|:------:\|:----:\|:-------------:\|:---------------:\|
(\| [\d\.]+ +\| [\d\.]+ +\| [\d\.]+ +\| [\d\.]+ +\|\n)+
### Environmental Impact
Carbon emissions were measured using \[CodeCarbon\]\(https://github.com/mlco2/codecarbon\)\.
- \*\*Carbon Emitted\*\*: [\d\.]+ kg of CO2
- \*\*Hours Used\*\*: [\d\.]+ hours

### Training Hardware
- \*\*On Cloud\*\*: (Yes|No)
- \*\*GPU Model\*\*: [^\n]+
- \*\*CPU Model\*\*: [^\n]+
- \*\*RAM Size\*\*: [\d\.]+ GB

### Framework Versions
- Python: [^\n]+
- SetFit: [^\n]+
- Sentence Transformers: [^\n]+
- Transformers: [^\n]+
- PyTorch: [^\n]+
- Datasets: [^\n]+
- Tokenizers: [^\n]+

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language \(cs.CL\), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = \{2022\},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

\*Clearly define terms in order to be accessible across audiences\.\*
-->

<!--
## Model Card Authors

\*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction\.\*
-->

<!--
## Model Card Contact

\*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors\.\*
-->""",
    flags=re.DOTALL,
)
