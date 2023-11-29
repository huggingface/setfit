---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# {{ model_name | default("SetFit for Text Classification", true) }}

This is a [SetFit](https://github.com/huggingface/setfit) model{% if dataset_id %} trained on the [{{ dataset_name if dataset_name else dataset_id }}](https://huggingface.co/datasets/{{ dataset_id }}) dataset{% endif %} that can be used for {{ task_name | default("Text Classification", true) }}.{% if st_id %} This SetFit model uses [{{ st_id }}](https://huggingface.co/{{ st_id }}) as the Sentence Transformer embedding model.{% endif %} A {{ head_class }} instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
{% if st_id -%}
    - **Sentence Transformer body:** [{{ st_id }}](https://huggingface.co/{{ st_id }})
{%- else -%}
    <!-- - **Sentence Transformer:** [Unknown](https://huggingface.co/unknown) -->
{%- endif %}
{% if head_class -%}
    - **Classification head:** a {{ head_class }} instance.
{%- else -%}
    <!-- - **Clasification head:** Unknown -->
{%- endif %}
- **Maximum Sequence Length:** {{ model_max_length }} tokens
{% if num_classes -%}
    - **Number of Classes:** {{ num_classes }} classes
{%- else -%}
    <!-- - **Number of Classes:** Unknown -->
{%- endif %}
{% if dataset_id -%}
    - **Training Dataset:** [{{ dataset_name if dataset_name else dataset_id }}](https://huggingface.co/datasets/{{ dataset_id }})
{%- else -%}
    <!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
{%- endif %}
{% if language -%}
    - **Language{{"s" if language is not string and language | length > 1 else ""}}:**
    {%- if language is string %} {{ language }}
    {%- else %} {% for lang in language -%}
            {{ lang }}{{ ", " if not loop.last else "" }}
        {%- endfor %}
    {%- endif %}
{%- else -%}
    <!-- - **Language:** Unknown -->
{%- endif %}
{% if license -%}
    - **License:** {{ license }}
{%- else -%}
    <!-- - **License:** Unknown -->
{%- endif %}

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)
{% if label_examples %}
### Model Labels
{{ label_examples }}{% endif -%}
{% if metrics_table %}
## Evaluation

### Metrics
{{ metrics_table }}{% endif %}
## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from {{ hf_emoji }} Hub
model = SetFitModel.from_pretrained("{{ model_id | default('setfit_model_id', true) }}")
# Run inference
preds = model("{{ predict_example | default("I loved the spiderman movie!", true) | replace('"', '\\"') }}")
```
<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details
{% if train_set_metrics %}
### Training Set Metrics
{{ train_set_metrics }}{% if train_set_sentences_per_label_list %}
{{ train_set_sentences_per_label_list }}{% endif %}{% endif %}{% if hyperparameters %}
### Training Hyperparameters
{% for name, value in hyperparameters.items() %}- {{ name }}: {{ value }}
{% endfor %}{% endif %}{% if eval_lines %}
### Training Results
{{ eval_lines }}{% if explain_bold_in_eval %}
* The bold row denotes the saved checkpoint.{% endif %}{% endif %}{% if co2_eq_emissions %}
### Environmental Impact
Carbon emissions were measured using [CodeCarbon](https://github.com/mlco2/codecarbon).
- **Carbon Emitted**: {{ "%.3f"|format(co2_eq_emissions["emissions"] / 1000) }} kg of CO2
- **Hours Used**: {{ co2_eq_emissions["hours_used"] }} hours

### Training Hardware
- **On Cloud**: {{ "Yes" if co2_eq_emissions["on_cloud"] else "No" }}
- **GPU Model**: {{ co2_eq_emissions["hardware_used"] or "No GPU used" }}
- **CPU Model**: {{ co2_eq_emissions["cpu_model"] }}
- **RAM Size**: {{ "%.2f"|format(co2_eq_emissions["ram_total_size"]) }} GB
{% endif %}
### Framework Versions
- Python: {{ version["python"] }}
- SetFit: {{ version["setfit"] }}
- Sentence Transformers: {{ version["sentence_transformers"] }}
- Transformers: {{ version["transformers"] }}
- PyTorch: {{ version["torch"] }}
- Datasets: {{ version["datasets"] }}
- Tokenizers: {{ version["tokenizers"] }}

## Citation

### BibTeX
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

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->