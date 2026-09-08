# A Comparative Framework for Native–Foreign Token Annotation in Congolese Personal Names

[![audit](https://github.com/bernard-ng/drc-names-tagging/actions/workflows/audit.yml/badge.svg)](https://github.com/bernard-ng/drc-names-tagging/actions/workflows/audit.yml)
[![quality](https://github.com/bernard-ng/drc-names-tagging/actions/workflows/quality.yml/badge.svg)](https://github.com/bernard-ng/drc-names-tagging/actions/workflows/quality.yml)

---

## Abstract

The internal structure of personal names is an important source of information
for computational onomastics and downstream name classification. However,
available name corpora rarely provide token-level annotations that distinguish
native components from components of foreign origin. This project introduces a
reproducible framework for annotating tokens in Congolese personal names with
two mutually exclusive labels: `native` and `foreign`. The framework operates
on the published CongoNames dataset and supports comparative experiments with
the associated name-classification research.

To support comparative evaluation, the framework defines a common tagging
interface and a consistent token-level output format. It currently implements
two complementary annotators: a character-transition Markov model and a local
large language model annotator based on Mistral 7B running through Ollama. The
latter uses constrained structured output to preserve token order and enforce
the annotation schema. Taggers are applied to identical records, enabling
measurement of pairwise agreement, processing throughput, and annotation
coverage. The resulting software provides an extensible experimental basis for
studying native–foreign token annotation before supervised name classification.

## How to cite this work

```bib
@software{drc_names_tagging,
  author = {Bernard-Ng, Bernard},
  title = {A Comparative Framework for Native--Foreign Token Annotation in Congolese Personal Names},
  year = {2026},
  url = {https://github.com/bernard-ng/drc-names-tagging}
}
```

## Workflows

Clone the repository and install its dependencies:

```bash
git clone https://github.com/bernard-ng/drc-names-tagging.git
cd drc-names-tagging
uv sync
```

The project uses the published [CongoNames dataset](https://github.com/bernard-ng/drc-names-corpus),
which is also used by `drc-names-classifier`. Make the published dataset
available locally before running an experiment. The repository's default local
copy is excluded from version control.

List the available experiments:

```bash
uv run drc-names-tagging experiments list
```

Run the Mistral 7B annotation experiment:

```bash
ollama pull mistral:7b
ollama serve
uv run drc-names-tagging tag --name ollama_mistral_7b
```

Compare the Markov and Mistral annotators on the same records:

```bash
uv run drc-names-tagging compare \
  --name markov \
  --name ollama_mistral_7b
```

## Published dataset

The published dataset is maintained by the
[CongoNames corpus project](https://github.com/bernard-ng/drc-names-corpus).
This repository does not recreate or redistribute the dataset; it applies
token-level annotation to the published records. The same records are used by
the classifier project so that tagging and classification experiments remain
comparable.

## Experiments

Experiment definitions are kept in `config/experiment_templates.yaml`. Each
experiment specifies an annotator, its model settings, and the size of the
sample to process. This keeps runs reproducible and avoids repeating model
parameters in the command line.

The available baseline experiments are:

| Experiment | Annotator | Description |
| --- | --- | --- |
| `markov` | Character transitions | Probabilistic native/foreign baseline. |
| `ollama_mistral_7b` | Ollama | Structured-output Mistral 7B annotation. |

## Annotation outputs

Results are written to `data/outputs/tagging/`:

| Output | Description |
| --- | --- |
| Experiment result | Token-level labels and confidence/score values. |
| Comparison table | Pairwise agreement between annotators. |
| Comparison summary | Agreement and processing-throughput summary. |

The two labels are mutually exclusive:

- `native`: the token is judged native to the relevant Congolese naming context.
- `foreign`: the token is judged not native to that context; this is not a surname label.

## Markov baseline methodology

The versioned Markov matrices are derived from the CongoNames character-transition
reports for [probable native](https://github.com/bernard-ng/drc-names-corpus/blob/main/reports/name_analysis/probable_native_transition_matrix.csv)
and [probable surname](https://github.com/bernard-ng/drc-names-corpus/blob/main/reports/name_analysis/probable_surname_transition_matrix.csv)
name groups. The latter distribution is used here as the foreign baseline.

To regenerate the matrices from an annotated source, run:

```bash
uv run drc-names-tagging transitions --dataset /path/to/annotated-dataset.csv
```

## Adding an annotator

Implement the common annotator interface, register the implementation, and add
an experiment entry. The existing progress display, output format, and
comparison workflow will then apply automatically.

## Responsible use

Native/foreign annotations are model-generated research labels and should not
be treated as definitive statements about identity, ethnicity, nationality, or
the cultural origin of an individual. The dataset also reflects the coverage
and historical biases of its public sources. Do not use these annotations for
eligibility, employment, credit, health, surveillance, or other consequential
decisions.

## Quality checks

```bash
uv run pyright
uv run ruff check .
```
