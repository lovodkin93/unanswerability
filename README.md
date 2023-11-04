# Unanswerability
Begin by setting the `MAX_GPU_MEM` and `MAX_CPU_MEM` parameters in `constants.py` to the maximum GPU and CPU (respectively) memory capacity of the machine you work on.

Additionally, create the conda env of the project by setting the `prefix` variable in `unanswerability_env.yml` to your `path/to/anaconda3/envs/unanswerability_env` location, and then run:

```
conda env create -f unanswerability_env.yml
python -m spacy download en_core_web_sm
conda activate unanswerability_env
```
## Download Dataset
To download the dataset, run:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Ind9cwSS94_5qo8_iakpD87PBbbwNb79' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Ind9cwSS94_5qo8_iakpD87PBbbwNb79" -O data.zip && rm -rf /tmp/cookies.txt

```
or directly download the zip file of the data from [link to data zip](https://drive.google.com/file/d/1Ind9cwSS94_5qo8_iakpD87PBbbwNb79/view?usp=sharing)

and then uzip it:
```
unzip data.zip
```

## Prompt Manipulations and Beam Relaxation Experiments

### zero-shot Prompting
To run the zero-shot prompt-manipulation experiment, run the following code:
```
python zero_shot_embeddings.py --models <MODELS> --datasets <DATASETS> --return-only-generated-text --outdir /path/to/outdir
```
where `<MODELS>` should be replaced by either one of `Flan-UL2`, `Flan-T5-xxl`, `OPT-IML` (or their concatenation - for running on several models), and `<DATASETS>` should be replaced by either one of `squad`, `NQ`, `musique` (or their concatenation - for running on several datasets).

This should save in the outdir folder two pt files - one starting with `un-answerable` and one starting with `answerable`. The former would be the model's responses for the un-answerable prompts, whereas the latter would be the model's responses for the answerable prompts.
It would also save the actual generated outputs in a sub-directory named "regular_decoding".

Additionally, to run this script on the develpment set, also pass the `--devset` flag.

Also, to run on different prompt variants (affects only cases where there is a hint of the un-answerability) - pass `--prompt-variant <VARIANT_LIST>` where `<VARIANT_LIST>` could be any concatenation of `variant1`, `variant2`, `variant3` (default is only `variant1`).

### Few-shot Prompting
To run the few-shot prompt-manipulation experiment, run the following code:
```
python few_shot_with_instructions_embeddings.py --models <MODELS> --datasets <DATASETS> --return-only-generated-text --outdir /path/to/outdir
```
`<MODELS>` and `<DATASETS>` are similar to those in the Zero-shot prompting experiments.

As for the zero-shot case, you can also change the prompt variant by passing `--prompt-variant <VARIANT_LIST>` (`<VARIANT_LIST>` is the same as before).

Lastly, you can choose one of the in-context-learning examples variants by passing `--icl-examples-variant <ICL_VARIANT_LIST>` where `<ICL_VARIANT_LIST>` could be any concatenation of `1`, `2`, `3` (default is only `1`).

### Beam Relaxation
To run the beam relaxation experiments, simply run the zero-shot experiment with the additional `--k-beams <BEAM_SIZE>` parameter.

In addition to the actual generated outputs saved in the "regular_decoding" sub-directory, the beam-relaxation version would be saved under the sub-directory "beam-relaxation".


### Evaluation
To evaluate the generated texts, run:
```
python -m evaluation_scripts.evaluate --indirs <INDIRS> --outdir /path/to/outdir 
```

Where `<INDIRS>` should be all the `outdirs` passed to either one of `zero_shot_embeddings.py` or `few_shot_with_instructions_embeddings.py` (separated by a single space). This will save under `/path/to/outdir` a csv file `QA-task-results.csv` with the results on the QA task for each of the prompt types (e.g., `Regular-Prompt` or `Hint-Prompt`), and an excel file `unanswerability_classification_results.xlsx`, with the unanswerability classification results for each of the prompt types. 

Additionally, to get the results on the development set, add the parameter `--devset`.

## Probing Experiments
To run the probing experiments, you first need to run the aforementioned zero-shot experiments **without** the `--return-only-generated-text` parameter, which will also save the embeddings of the generations. In other words, run:

```
python zero_shot_embeddings.py --models <MODELS> --datasets <DATASETS> --outdir /path/to/outdir
```

This will save the embeddings of the <ins>last</ins> hidden layer of the first generated token for each instance of <ins>the test set</ins>.

Additionally, to train the linear classifiers, we also need to extract the embeddings of the **train set**. For that, we need to also pass `--trainset`:
```
python zero_shot_embeddings.py --models <MODELS> --datasets <DATASETS> --outdir /path/to/outdir --trainset
```
This will also save the embeddings of the <ins>last</ins> hidden layer of the first generated token for each instance, but for <ins>the train set</ins>.

To also save the <ins>first</ins> hidden layer of the first generated token, pass also `--return-first-layer`.

As before, we can also pass `--prompt-variant <VARIANT_LIST>` to control which (hint) prompt variant to use.


### Linear Classifiers

#### Train

To train the answerability linear classifiers, run:

```
python train_linear_classifiers.py --indir <INDIR> --outdir /path/to/outdir --dataset <DATASET> --prompt-type <PROMPT_TYPE> --epochs 100 --batch-size 16 --num-instances 1000
```

where `<INDIR>` is the path to the directory with the pt files of <ins>the train set</ins>, `<DATASET>` should be replaced by either one of `squad`, `NQ`, `musique` and `<PROMPT_TYPE>` should be either `Regular-Prompt` or `Hint-Prompt`.

Additionally, to train a classifier on the <ins>first</ins> hidden layer of the first generated token, also pass `--embedding-type first_hidden_embedding`.

This will save under `outdir/<DATASET>/<EMBEDDING_TYPE>/<PROMPT_TYPE>/only_first_tkn/<MODEL_NAME>_1000N"` the trained classifier (where `<EMBEDDING_TYPE>` is either `first_hidden_embedding` or `last_hidden_embedding` and `<MODEL_NAME>` is the name of the model whose embeddings were used to train the classifier).

#### Evaluate
To evaluate the answerability linear classifiers, run:
```
python evaluation_scripts/eval_linear_classifiers.py --indir <DATA_INDIR> --classifier-dir <CLASSIFIER_INDIR> --dataset <DATASET> --prompt-type <PROMPT_TYPE> --embedding-type <EMBEDDING_TYPE>
```

where `<DATA_INDIR>` is the path to the directory with the pt files of <ins>the test set</ins>, `<CLASSIFIER_INDIR>` is the path to the trained linear classifier, `<DATASET>` should be replaced by either one of `squad`, `NQ`, `musique` and should represent the dataset of the test set, `<PROMPT_TYPE>` should be either `Regular-Prompt` or `Hint-Prompt` and `<EMBEDDING_TYPE>` should be either `first_hidden_embedding` or `last_hidden_embedding`.

#### Visualize Embedding Space
To visualize the embedding space, run:

```
python figures_generation/PCA_plots_generation.py -i /path/to/folder/with/pt_files -o /path/to/outdir --prompt-type <PROMPT_TYPE> 
```

where `<PROMPT_TYPE>` should be either `Regular-Prompt` or `Hint-Prompt`. The generated 3-D PCA plots of the embedding space will be saved under `/path/to/outdir/last_hidden_embedding/only_first_tkn/<PROMPT_TYPE>`.

### Answerability Subspace Erasure
To perform this experiment, we first need to create a separate conda env. For that, set the `prefix` variable in `subspace_erasure.yml` to your `path/to/anaconda3/envs/subspace_erasure` location, and then run:

```
conda env create -f subspace_erasure.yml
conda activate subspace_erasure
```

Before starting these experiments, please make sure you have the embeddings of the **train set** instances mentioned at the beginning of the **Probing Experiments** section.
Once you have the embeddings of the **train set** instances, we will start by training the concept eraser, by running:

```
python train_concept_eraser.py --indir <INDIR> --outdir /path/to/outdir --dataset <DATASET> --prompt-type <PROMPT_TYPE> --epochs 500 --batch-size 16 --num-instances 1000
```

where `<INDIR>` is the path to the directory with the pt files of <ins>the train set</ins>, `<DATASET>` should be replaced by either one of `squad`, `NQ`, `musique` and `<PROMPT_TYPE>` should be either `Regular-Prompt` or `Hint-Prompt`.

Once the training is finished, you will find the trained eraser under `/path/to/outdir/<DATASET>/<PROMPT_TYPE>`.

Next, to perform the actual prompting of the LLMs with the erasure component, run:

```
python zero_shot_embeddings_erasure.py --models <MODELS> --datasets <DATASETS> --outdir /path/to/outdir --eraser-dir /path/to/trained_eraser --only-first-decoding
```

where `<MODELS>` should be replaced by either one of `Flan-UL2`, `Flan-T5-xxl`, `OPT-IML` (or their concatenation - for running on several models), and `<DATASETS>` should be replaced by either one of `squad`, `NQ`, `musique` (or their concatenation - for running on several datasets).

This should save in the outdir folder two pt files - one starting with `un-answerable` and one starting with `answerable`. The former would be the model's responses for the un-answerable prompts, whereas the latter would be the model's responses for the answerable prompts.

Now, to evaluate the responses, follow the the instructions under the **Evaluation** sub-section of the **Prompt Manipulations and Beam Relaxation Experiments** section. 

Additionally, to visualize the embeddings, follow the **Visualize Embedding Space** instructions under the **Linear Classifiers** sub-section.