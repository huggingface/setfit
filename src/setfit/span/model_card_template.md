---
license: apache-2.0
tags:
- setfit
- sentence-transformers
- absa
- token-classification
pipeline_tag: token-classification
---

# {{ model_name | default("SetFit ABSA Model", true) }}

This is a [SetFit ABSA model](https://github.com/huggingface/setfit) that can be used for Aspect Based Sentiment Analysis (ABSA). \
In particular, this model is in charge of {{ "filtering aspect span candidates" if is_aspect else "classifying aspect polarities"}}.
It has been trained using SetFit, an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

This model was trained within the context of a larger system for ABSA, which looks like so:

1. Use a spaCy model to select possible aspect span candidates.
2. {{ "**" if is_aspect else "" }}Use {{ "this" if is_aspect else "a" }} SetFit model to filter these possible aspect span candidates.{{ "**" if is_aspect else "" }}
3. {{ "**" if not is_aspect else "" }}Use {{ "this" if not is_aspect else "a" }} SetFit model to classify the filtered aspect span candidates.{{ "**" if not is_aspect else "" }}

## Usage

To use this model for inference, first install the SetFit library:

```bash
pip install setfit
```

You can then run inference as follows:

```python
from setfit import AbsaModel

# Download from Hub and run inference
model = AbsaModel.from_pretrained(
    "{{ aspect_model }}",
    "{{ polarity_model }}",
)
# Run inference
preds = model([
    "The best pizza outside of Italy and really tasty.",
    "The food here is great but the service is terrible",
])
```

## BibTeX entry and citation info

```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```