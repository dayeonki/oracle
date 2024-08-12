# Mitigating Semantic Leakage in Cross-lingual Embeddings via Orthogonality Constraint

This repository contains the code and dataset for our ACL 2024 RepL4NLP workshop paper **Mitigating Semantic Leakage in Cross-lingual Embeddings via Orthogonality Constraint**.

<div align="center">
[🤖 <b><a href=https://github.com/dayeonki/oracle/code>Code</a></b> / 🤗 <b><a href=https://huggingface.co/datasets/zoeyki/oracle_dataset>Dataset</a></b> / 📄 <b><a href=https://aclanthology.org/2024.repl4nlp-1.19>Paper</a></b>]
</div>


## Abstract
Accurately aligning contextual representations in cross-lingual sentence embeddings is key for effective parallel data mining. A common strategy for achieving this alignment involves disentangling semantics and language in sentence embeddings derived from multilingual pre-trained models. However, we discover that current disentangled representation learning methods suffer from **_semantic leakage_** — a term we introduce to describe when a substantial amount of language-specific information is unintentionally leaked into semantic representations. This hinders the effective disentanglement of semantic and language representations, making it difficult to retrieve embeddings that distinctively represent the meaning of the sentence.

To address this challenge, we propose a novel training objective, ORthogonAlity Constraint LEarning (**ORACLE**), tailored to enforce orthogonality between semantic and language embeddings. ORACLE builds upon two components: intra-class clustering and inter-class separation. Through experiments on cross-lingual retrieval and semantic textual similarity tasks, we demonstrate that training with the ORACLE objective effectively reduces semantic leakage and enhances semantic alignment within the embedding space.
<p align="center">
  <img src="https://github.com/user-attachments/assets/0852047a-00d1-49e2-b556-e02db7c9c4f6" width="500">
</p>

## Quick Links
- [Overview](#overview)
- [Train with ORACLE](#train-with-oracle)
- [Retrieval Inference](#retrieval-inference)
- [Visualization](#visualization)


## Overview
ORACLE consists of two key components: (1) **intra-class clustering** and (2) **inter-class separation**. Intra-class clustering aligns related components more closely, while inter-class separation enforces orthogonality between unrelated components. Our method is designed to be simple and effective, capable of being implemented atop any disentanglement methods.

We explore a range of pre-trained multilingual encoders (LASER, InfoXLM, LaBSE) to generate initial sentence embeddings. Subsequently, we train each semantic and language multi-layer perceptrons (MLPs) with ORACLE to disentangle the sentence embeddings into semantics and language-specific information. Experimental results on both cross-lingual sentence retrieval tasks and the Semantic Textual Similarity (STS) task demonstrate higher performance on semantic embeddings and lower performance on language embeddings with ORACLE. The following figure is an illustration of our work.
<p align="center">
  <img src="https://github.com/user-attachments/assets/d2a0978d-7820-4003-84be-8025e804d728" width="600">
</p>

## Train with ORACLE
Install all requirements in `requirements.txt`.
```bash
pip install -r requirements.txt
```
### [Step 1] Data preparation
Place the parallel sentences in `data/` and transform each file into text file with each sentence in each line. Below is an example for English-French language pair.

**en-fr.en**
```
I enjoy reading books in my free time.
The weather today is perfect for a picnic.
She is learning how to cook traditional French cuisine.
...
```

**en-fr.fr**
```
J'aime lire des livres pendant mon temps libre.
Le temps aujourd'hui est parfait pour un pique-nique.
Elle apprend à cuisiner des plats traditionnels français.
...
```

### [Step 2] Create embeddings
We use 3 different pre-trained multilingual encoders: <a href=https://github.com/facebookresearch/LASER>LASER</a>, <a href=https://huggingface.co/microsoft/infoxlm-base>InfoXLM</a> and <a href=https://huggingface.co/sentence-transformers/LaBSE>LaBSE</a>. To create embeddings for the bitext dataset in `data/`, run `script/embed.py` as below:

```
python -u embed.py \
   --model_name_or_path $MODEL_NAME \
   --src_data_path $PATH_TO_SOURCE_DATA \
   --tgt_data_path $PATH_TO_TARGET_DATA \
   --src_embed_path $PATH_TO_SOURCE_EMBEDDINGS \
   --tgt_embed_path $PATH_TO_TARGET_EMBEDDINGS \
   --train_embed_path $PATH_TO_TRAIN_EMBEDDINGS \
   --valid_embed_path $PATH_TO_VALIDATION_EMBEDDINGS \
   --src_lang $SOURCE_LANG \
   --tgt_lang $TARGET_LANG \
   --batch_size $BATCH_SIZE \
   --seed $SEED_NUM
```

Arguments for the create embeddings script are as follows,
- `--model_name_or_path`: Path or name of the pre-trained multilingual encoder
- `--src_data_path`: Path to source dataset (ex. `data/en-fr.en`)
- `--tgt_data_path`: Path to target dataset (ex. `data/en-fr.fr`)
- `--src_embed_path`: Path to save the source embeddings created
- `--tgt_embed_path`: Path to save the target embeddings created
- `--train_embed_path`: Path to save the train split of embeddings
- `--valid_embed_path`: Path to save the validation split of embeddings
- `--batch_size`: Batch size of the model (default: 512)
- `--seed_num`: Seed number (default: 42)

### [Step 3] Train
Using the embedding created from previous step, we train the decomposer with ORACLE objective. To train, you have to choose each variation from below options:
- Decomposer type : {DREAM, MEAT}
- Encoder type : {LASER, InfoXLM, LaBSE}
- Train type : {Vanilla, ORACLE}
    - Vanilla is the training method introduced in DREAM, MEAT papers.
    - ORACLE is our approach, composed with both intra-class clustering and inter-class separation.
- If you wish to train a DREAM Decomposer, run `script/train_dream.py`, if train a MEAT Decomposer, run `script/train_meat.py`.

For example,
```bash
python -u train_dream.py -c config/labse/dream_labse.yaml
python -u train_meat.py -c config/infoxlm/meat_infoxlm_oracle.yaml
```

There are configuration parameters for training in each yaml file:

```yaml
train_path: $PATH_TO_TRAIN_EMBEDDINGS
valid_path: $PATH_TO_VALIDATION_EMBEDDINGS
save_pooler_path: $PATH_TO_SAVE_MODEL
logging_path: $LOG_FILE_PATH
train_type: $TRAIN_TYPE
learning_rate: 1e-5
n_languages: 13
batch_size: 512
seed: 42
weights: [[1,1]]
model_name_or_path: $MODEL_NAME
```

- `train_path`: Path to the saved train split of embeddings
- `valid_path`: Path to the saved validation split of embeddings
- `model_name_or_path`: Path or name of the pre-trained multilingual encoder
- `save_pooler_path`: Path to save the pooler after training
- `logging_path`: Path to save the log file (Log file saves the loss values for each training epoch)
- `train_type`: {vanilla, oracle}
- `n_languages`: Number of languages (default: 13)
- `weights`: Weight values for each losses


## Retrieval Inference
We provide inference codes for following retrieval tasks in `code/inference/`:
- **BUCC** : Crosslingual retrieval task, run `bucc.py` for InfoXLM and LaBSE and run `bucc_laser.py` for LASER
- **Tatoeba**: Crosslingual retrieval task
    - To retrieve semantic embeddings, run `tatoeba_sem.py` for InfoXLM and LaBSE and run `tatoeba_sem_laser.py` for LASER
    - To retrieve language embeddings, run `tatoeba_lang.py` for InfoXLM and LaBSE and run `tatoeba_lang_laser.py` for LASER
- **STS** : Semantic textual similarity task, run `sts.py` for InfoXLM and LaBSE and run `sts_laser.py` for LASER


## Visualization
To visualize the embedding space of the trained decomposers using `datavis` library, run `code/visualize.py`. This code will save a html bokeh file in the `output_figure` directory.
    
```
python -u visualize.py \
   --model_name_or_path $MODEL_NAME \
   --pooler_path $TRAINED_POOLER_PATH$ \
   --src_data_path $SOURCE_RETRIEVAL_DATA_PATH \
   --tgt_data_path $TARGET_RETRIEVAL_DATA_PATH \
   --src_lang $SOURCE_LANG \
   --tgt_lang $TARGET_LANG \
   --batch_size $BATCH_SIZE \
   --output_figure $PATH_TO_OUTPUT_FIGURE
```

## Citation
```
```
