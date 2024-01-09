# <h2 align="center"> The Curious Case of Hallucinatory (Un)answerability: Finding Truths in the Hidden States of Over-Confident Large Language Models </h2>

Repository for our EMNLP 2023 paper "[The Curious Case of Hallucinatory (Un)answerability: Finding Truths in the Hidden States of Over-Confident Large Language Models](https://aclanthology.org/2023.emnlp-main.220/)"

# Getting Started
* Adjust `prefix` in `unanswerability_env.yml` to your Anaconda environment path.
* Run these commands:
```
conda env create -f unanswerability_env.yml
python -m spacy download en_core_web_sm
conda activate unanswerability_env
```
# Download Dataset
1. To download the dataset, run:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1E3wZLRUi4JZ2ebD0rSKHTq8ISnecOj6_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1E3wZLRUi4JZ2ebD0rSKHTq8ISnecOj6_" -O data.zip && rm -rf /tmp/cookies.txt

```
or directly download the file from [Google Drive](https://drive.google.com/file/d/1E3wZLRUi4JZ2ebD0rSKHTq8ISnecOj6_/view?usp=sharing)

2. uzip:
```
unzip data.zip
```

# Prompt Manipulations and Beam Relaxation Experiments

## Zero-shot Prompting
To perform the zero-shot prompt-manipulation experiment, run:
```
python zero_shot_prompting.py --models <MODELS> --datasets <DATASETS> --return-only-generated-text --outdir /path/to/outdir
```
* `<MODELS>` - any one of `Flan-UL2`, `Flan-T5-xxl`, or `OPT-IML`(can pass more than one).
* `<DATASETS>` - any one of `squad`, `NQ`, or `musique` (can pass more than one).
* For prompt variants, add `--prompt-variant <VARIANT_LIST>`:
  - `<VARIANT_LIST>` - any one of `variant1`, `variant2`, `variant3` (can pass more than one).
    - Default - `variant1`.
* For development set experiments, add `--devset`.
* **Output**: Saves two `.pt` files in the specified outdir, one for answerable and one for un-answerable prompts.
  - Also saves the actual generated outputs in the sub-directory **regular_decoding**.


## Few-shot Prompting
To perform the few-shot prompt-manipulation experiment, run:
```
python few_shot_prompting.py --models <MODELS> --datasets <DATASETS> --return-only-generated-text --outdir /path/to/outdir
```
* `<MODELS>` and `<DATASETS>` are similar to those in [Zero-shot Prompting](#zero-shot-prompting).
* Prompt variant can be changed like in [Zero-shot Prompting](#zero-shot-prompting).
* For in-context-learning examples variants - add `--icl-examples-variant <ICL_VARIANT_LIST>`:
  * `<ICL_VARIANT_LIST>` - any one of `1`, `2`, `3` (can pass more than one). 

## Beam Relaxation
For beam relaxation experiments, just add `--k-beams <BEAM_SIZE>` to the [Zero-shot Prompting](#zero-shot-prompting) command.

* **Output**: In addition to the sub-directory **regular_decoding**, an additional **beam-relaxation** sub-directory will be generated, with the beam-relaxed responses.


## Evaluation
To evaluate the generated texts, run:
```
python -m evaluation.evaluate --indirs <INDIRS> --outdir /path/to/outdir 
```
* `<INDIRS>`: output directories from the prompting experiments.
* **output**: save under `outdir`:
  -  <ins>QA-task-results.csv</ins> - results on the QA task for each prompt type (e.g., `Regular-Prompt` or `Hint-Prompt`).
  -  <ins>unanswerability_classification_results.xlsx</ins> - unanswerability classification results for each prompt type.
* For results on development set, add `--devset`.

# Probing Experiments
## Preliminaries - Get Embeddings
1. **Generate Test Set Embeddings**: Run the [Zero-shot Prompting](#zero-shot-prompting) experiments without the `--return-only-generated-text` parameter.
    * This will also save the generations' embeddings (last hidden layer of first generated token) of <ins>the test set</ins>. 
2. **Generate Train Set Embeddings**: In addition to step 1, also add `--trainset`.
* to run steps 1 and 2 on the <ins>first</ins> hidden layer of the first generated token, add `--return-first-layer`.
* Prompt variant can be changed like in [Zero-shot Prompting](#zero-shot-prompting).

### Train Answerability Linear Classifiers

Run:

```
python train_linear_classifiers.py --indir <INDIR> --outdir /path/to/outdir --dataset <DATASET> --prompt-type <PROMPT_TYPE> --epochs 100 --batch-size 16 --num-instances 1000
```
* `<INDIR>` - path to the directory with the saved embeddings (pt files) of <ins>the train set</ins>.
* `<DATASET>` - any one of `squad`, `NQ`, `musique`.
* `<PROMPT_TYPE>` - `Regular-Prompt` or `Hint-Prompt`.
* To train a classifier on the <ins>first</ins> hidden layer of the first generated token, add `--embedding-type first_hidden_embedding`.
* **output** - save under `outdir/<DATASET>/<EMBEDDING_TYPE>/<PROMPT_TYPE>/only_first_tkn/<MODEL_NAME>_1000N"` the trained classifier.
  * `<EMBEDDING_TYPE>` - `first_hidden_embedding` or `last_hidden_embedding`.
  * `<MODEL_NAME>` - name of the model whose embeddings were used to train the classifier.

### Evaluate Answerability Linear Classifiers
Run:
```
python evaluation/eval_linear_classifiers.py --indir <DATA_INDIR> --classifier-dir <CLASSIFIER_INDIR> --dataset <DATASET> --prompt-type <PROMPT_TYPE> --embedding-type <EMBEDDING_TYPE>
```

* `<DATA_INDIR>` - path to directory with the <ins>the test set</ins> saved embeddings (pt files).
* `<CLASSIFIER_INDIR>` - path to the trained linear classifier.
* `<DATASET>` - any one of `squad`, `NQ`, `musique` (should represent the dataset of the test set).
* `<PROMPT_TYPE>` - `Regular-Prompt` or `Hint-Prompt`.
*  `<EMBEDDING_TYPE>` - `first_hidden_embedding` or `last_hidden_embedding`.

### Visualize Embedding Space
Run:

```
python figures_generation/PCA_plots_generation.py -i /path/to/folder/with/pt_files -o /path/to/outdir --prompt-type <PROMPT_TYPE> 
```

* `<PROMPT_TYPE>` - `Regular-Prompt` or `Hint-Prompt`.
* **output** - The generated 3-D PCA plots of the embedding space will be saved under `/path/to/outdir/last_hidden_embedding/only_first_tkn/<PROMPT_TYPE>`.

# Answerability Subspace Erasure
## Preliminaries
1. **Set Up Environment** - Create a separate Conda environment for this experiment:
   * Adjust `prefix` in `subspace_erasure.yml` to your Anaconda environment path.
   * Run these commands:
```
conda env create -f subspace_erasure.yml
conda activate subspace_erasure
```
2. Make sure you have the embeddings of the **train set** from [Preliminaries - Get Embeddings](#preliminaries---get-embeddings).

## Train Concept Eraser
Run:

```
python train_concept_eraser.py --indir <INDIR> --outdir /path/to/outdir --dataset <DATASET> --prompt-type <PROMPT_TYPE> --epochs 500 --batch-size 16 --num-instances 1000
```

* `<INDIR>` - path to the directory with the embeddings (pt files) of <ins>the train set</ins>.
* `<DATASET>` - any one of `squad`, `NQ`, `musique`
* `<PROMPT_TYPE>` - `Regular-Prompt` or `Hint-Prompt`.
* **output** - trained eraser will be under `/path/to/outdir/<DATASET>/<PROMPT_TYPE>`.

## Prompting with Concept Erasure
Run:
```
python zero_shot_erasure_prompting.py --models <MODELS> --datasets <DATASETS> --outdir /path/to/outdir --eraser-dir /path/to/trained_eraser --only-first-decoding
```
* `<MODELS>` and `<DATASETS>` are similar to those in [Zero-shot Prompting](#zero-shot-prompting).
* **Output**: Saves two `.pt` files in the specified outdir, one for answerable and one for un-answerable prompts.
  - Also saves the actual generated outputs in the sub-directory **regular_decoding**.
* To evaluate the responses, follow the instructions under [Evaluation](#evaluation). 
* To visualize the embeddings, follow the instructions under [Visualize Embedding Space](#visualize-embedding-space).

# Citation

If you use this in your work, please cite:

```
@inproceedings{slobodkin-etal-2023-curious,
    title = "The Curious Case of Hallucinatory (Un)answerability: Finding Truths in the Hidden States of Over-Confident Large Language Models",
    author = "Slobodkin, Aviv  and
      Goldman, Omer  and
      Caciularu, Avi  and
      Dagan, Ido  and
      Ravfogel, Shauli",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.220",
    doi = "10.18653/v1/2023.emnlp-main.220",
    pages = "3607--3625",
    abstract = "Large language models (LLMs) have been shown to possess impressive capabilities, while also raising crucial concerns about the faithfulness of their responses. A primary issue arising in this context is the management of (un)answerable queries by LLMs, which often results in hallucinatory behavior due to overconfidence. In this paper, we explore the behavior of LLMs when presented with (un)answerable queries. We ask: do models \textit{represent} the fact that the question is (un)answerable when generating a hallucinatory answer? Our results show strong indications that such models encode the answerability of an input query, with the representation of the first decoded token often being a strong indicator. These findings shed new light on the spatial organization within the latent representations of LLMs, unveiling previously unexplored facets of these models. Moreover, they pave the way for the development of improved decoding techniques with better adherence to factual generation, particularly in scenarios where query (un)answerability is a concern.",
}
```
