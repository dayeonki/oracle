#!/bin/bash

PATH_TO_MSIMCSE_CKPT=""


# LaBSE
python -u retrieve_bucc.py --model_name_or_path sentence-transformers/LaBSE \
    --model_nickname labse \
    --pooler_path pooler/labse/dream-labse.pt \
    --decomposer_type dream;
python -u retrieve_bucc.py --model_name_or_path sentence-transformers/LaBSE \
    --model_nickname labse \
    --pooler_path pooler/labse/dream-labse-opl-w1.pt \
    --decomposer_type dream;
python -u retrieve_bucc.py --model_name_or_path sentence-transformers/LaBSE \
    --model_nickname labse \
    --pooler_path pooler/labse/meat-labse.pt \
    --decomposer_type meat;
python -u retrieve_bucc.py --model_name_or_path sentence-transformers/LaBSE \
    --model_nickname labse \
    --pooler_path pooler/labse/meat-labse-opl-w1.pt \
    --decomposer_type meat;

# mSimCSE
python -u retrieve_bucc.py --model_name_or_path $PATH_TO_MSIMCSE_CKPT \
    --model_nickname msimcse \
    --pooler_path pooler/msimcse/dream-msimcse.pt \
    --decomposer_type dream;
python -u retrieve_bucc.py --model_name_or_path $PATH_TO_MSIMCSE_CKPT \
    --model_nickname msimcse \
    --pooler_path pooler/msimcse/dream-msimcse-opl-w1.pt \
    --decomposer_type dream;
python -u retrieve_bucc.py --model_name_or_path $PATH_TO_MSIMCSE_CKPT \
    --model_nickname msimcse \
    --pooler_path pooler/msimcse/meat-msimcse.pt \
    --decomposer_type meat;
python -u retrieve_bucc.py --model_name_or_path $PATH_TO_MSIMCSE_CKPT \
    --model_nickname msimcse \
    --pooler_path pooler/msimcse/meat-msimcse-opl-w1.pt \
    --decomposer_type meat;

# LASER
python -u retrieve_bucc_laser.py --model_nickname laser \
    --pooler_path pooler/laser/dream-laser.pt \
    --decomposer_type dream;
python -u retrieve_bucc.py --model_nickname laser \
    --pooler_path pooler/laser/dream-laser-opl-w1.pt \
    --decomposer_type dream;
python -u retrieve_bucc.py --model_nickname laser \
    --pooler_path pooler/laser/meat-laser.pt \
    --decomposer_type meat;
python -u retrieve_bucc.py --model_nickname laser \
    --pooler_path pooler/laser/meat-laser-opl-w1.pt \
    --decomposer_type meat;