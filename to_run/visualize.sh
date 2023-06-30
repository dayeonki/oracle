#!/bin/bash

python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse.pt \
    --src_data_path data/en-ar/en-ar.en \
    --tgt_data_path data/en-ar/en-ar.ar \
    --decomposer_type dream \
    --output_figure dream-labse-enar \
    --src_lang en \
    --tgt_lang ar;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse-opl-w1.pt \
    --src_data_path data/en-ar/en-ar.en \
    --tgt_data_path data/en-ar/en-ar.ar \
    --decomposer_type dream \
    --output_figure dream-labse-opl-enar \
    --src_lang en \
    --tgt_lang ar;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse.pt \
    --src_data_path data/en-ar/en-ar.en \
    --tgt_data_path data/en-ar/en-ar.ar \
    --decomposer_type meat \
    --output_figure meat-labse-enar \
    --src_lang en \
    --tgt_lang ar;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse-opl-w1.pt \
    --src_data_path data/en-ar/en-ar.en \
    --tgt_data_path data/en-ar/en-ar.ar \
    --decomposer_type meat \
    --output_figure meat-labse-opl-enar \
    --src_lang en \
    --tgt_lang ar;

python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse.pt \
    --src_data_path data/en-ay/en-ay.en \
    --tgt_data_path data/en-ay/n-ay.ay \
    --decomposer_type dream \
    --output_figure dream-labse-enay \
    --src_lang en \
    --tgt_lang ay;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse-opl-w1.pt \
    --src_data_path data/en-ay/en-ay.en \
    --tgt_data_path data/en-ay/en-ay.ay \
    --decomposer_type dream \
    --output_figure dream-labse-opl-enay \
    --src_lang en \
    --tgt_lang ay;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse.pt \
    --src_data_path data/en-ay/en-ay.en \
    --tgt_data_path data/en-ay/en-ay.ay \
    --decomposer_type meat \
    --output_figure meat-labse-enay \
    --src_lang en \
    --tgt_lang ay;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse-opl-w1.pt \
    --src_data_path data/en-ay/en-ay.en \
    --tgt_data_path data/en-ay/en-ay.ay \
    --decomposer_type meat \
    --output_figure meat-labse-opl-enay \
    --src_lang en \
    --tgt_lang ay;

python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse.pt \
    --src_data_path data/en-de/en-de.en \
    --tgt_data_path data/en-de/en-de.de \
    --decomposer_type dream \
    --output_figure dream-labse-ende \
    --src_lang en \
    --tgt_lang de;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse-opl-w1.pt \
    --src_data_path data/en-de/en-de.en \
    --tgt_data_path data/en-de/en-de.de \
    --decomposer_type dream \
    --output_figure dream-labse-opl-ende \
    --src_lang en \
    --tgt_lang de;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse.pt \
    --src_data_path data/en-de/en-de.en \
    --tgt_data_path data/en-de/en-de.de \
    --decomposer_type meat \
    --output_figure meat-labse-ende \
    --src_lang en \
    --tgt_lang de;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse-opl-w1.pt \
    --src_data_path data/en-de/en-de.en \
    --tgt_data_path data/en-de/en-de.de \
    --decomposer_type meat \
    --output_figure meat-labse-opl-ende \
    --src_lang en \
    --tgt_lang de;

python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse.pt \
    --src_data_path data/en-es/en-es.en \
    --tgt_data_path data/en-es/en-es.es \
    --decomposer_type dream \
    --output_figure dream-labse-enes \
    --src_lang en \
    --tgt_lang es;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse-opl-w1.pt \
    --src_data_path data/en-es/en-es.en \
    --tgt_data_path data/en-es/en-es.es \
    --decomposer_type dream \
    --output_figure dream-labse-opl-enes \
    --src_lang en \
    --tgt_lang es;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse.pt \
    --src_data_path data/en-es/en-es.en \
    --tgt_data_path data/en-es/en-es.es \
    --decomposer_type meat \
    --output_figure meat-labse-enes \
    --src_lang en \
    --tgt_lang es;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse-opl-w1.pt \
    --src_data_path data/en-es/en-es.en \
    --tgt_data_path data/en-es/en-es.es \
    --decomposer_type meat \
    --output_figure meat-labse-opl-enes \
    --src_lang en \
    --tgt_lang es;

python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse.pt \
    --src_data_path data/en-fr/en-fr.en \
    --tgt_data_path data/en-fr/en-fr.fr \
    --decomposer_type dream \
    --output_figure dream-labse-enfr \
    --src_lang en \
    --tgt_lang fr;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse-opl-w1.pt \
    --src_data_path data/en-fr/en-fr.en \
    --tgt_data_path data/en-fr/en-fr.fr \
    --decomposer_type dream \
    --output_figure dream-labse-opl-enfr \
    --src_lang en \
    --tgt_lang fr;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse.pt \
    --src_data_path data/en-fr/en-fr.en \
    --tgt_data_path data/en-fr/en-fr.fr \
    --decomposer_type meat \
    --output_figure meat-labse-enfr \
    --src_lang en \
    --tgt_lang fr;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse-opl-w1.pt \
    --src_data_path data/en-fr/en-fr.en \
    --tgt_data_path data/en-fr/en-fr.fr \
    --decomposer_type meat \
    --output_figure meat-labse-opl-enfr \
    --src_lang en \
    --tgt_lang fr;

python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse.pt \
    --src_data_path data/en-gn/en-gn.en \
    --tgt_data_path data/en-gn/en-gn.gn \
    --decomposer_type dream \
    --output_figure dream-labse-engn \
    --src_lang en \
    --tgt_lang gn;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse-opl-w1.pt \
    --src_data_path data/en-gn/en-gn.en \
    --tgt_data_path data/en-gn/en-gn.gn \
    --decomposer_type dream \
    --output_figure dream-labse-opl-engn \
    --src_lang en \
    --tgt_lang gn;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse.pt \
    --src_data_path data/en-gn/en-gn.en \
    --tgt_data_path data/en-gn/en-gn.gn \
    --decomposer_type meat \
    --output_figure meat-labse-engn \
    --src_lang en \
    --tgt_lang gn;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse-opl-w1.pt \
    --src_data_path data/en-gn/en-gn.en \
    --tgt_data_path data/en-gn/en-gn.gn \
    --decomposer_type meat \
    --output_figure meat-labse-opl-engn \
    --src_lang en \
    --tgt_lang gn;

python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse.pt \
    --src_data_path data/en-it/en-it.en \
    --tgt_data_path data/en-it/en-it.it \
    --decomposer_type dream \
    --output_figure dream-labse-enit \
    --src_lang en \
    --tgt_lang it;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse-opl-w1.pt \
    --src_data_path data/en-it/en-it.en \
    --tgt_data_path data/en-it/en-it.it \
    --decomposer_type dream \
    --output_figure dream-labse-opl-enit \
    --src_lang en \
    --tgt_lang it;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse.pt \
    --src_data_path data/en-it/en-it.en \
    --tgt_data_path data/en-it/en-it.it \
    --decomposer_type meat \
    --output_figure meat-labse-enit \
    --src_lang en \
    --tgt_lang it;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse-opl-w1.pt \
    --src_data_path data/en-it/en-it.en \
    --tgt_data_path data/en-it/en-it.it \
    --decomposer_type meat \
    --output_figure meat-labse-opl-enit \
    --src_lang en \
    --tgt_lang it;

python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse.pt \
    --src_data_path data/en-ja/en-ja.en \
    --tgt_data_path data/en-ja/en-ja.ja \
    --decomposer_type dream \
    --output_figure dream-labse-enja \
    --src_lang en \
    --tgt_lang ja;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse-opl-w1.pt \
    --src_data_path data/en-ja/en-ja.en \
    --tgt_data_path data/en-ja/en-ja.ja \
    --decomposer_type dream \
    --output_figure dream-labse-opl-enja \
    --src_lang en \
    --tgt_lang ja;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse.pt \
    --src_data_path data/en-ja/en-ja.en \
    --tgt_data_path data/en-ja/en-ja.ja \
    --decomposer_type meat \
    --output_figure meat-labse-enja \
    --src_lang en \
    --tgt_lang ja;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse-opl-w1.pt \
    --src_data_path data/en-ja/en-ja.en \
    --tgt_data_path data/en-ja/en-ja.ja \
    --decomposer_type meat \
    --output_figure meat-labse-opl-enja \
    --src_lang en \
    --tgt_lang ja;

python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse.pt \
    --src_data_path data/en-nl/en-nl.en \
    --tgt_data_path data/en-nl/en-nl.nl \
    --decomposer_type dream \
    --output_figure dream-labse-ennl \
    --src_lang en \
    --tgt_lang nl;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse-opl-w1.pt \
    --src_data_path data/en-nl/en-nl.en \
    --tgt_data_path data/en-nl/en-nl.nl \
    --decomposer_type dream \
    --output_figure dream-labse-opl-ennl \
    --src_lang en \
    --tgt_lang nl;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse.pt \
    --src_data_path data/en-nl/en-nl.en \
    --tgt_data_path data/en-nl/en-nl.nl \
    --decomposer_type meat \
    --output_figure meat-labse-ennl \
    --src_lang en \
    --tgt_lang nl;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse-opl-w1.pt \
    --src_data_path data/en-nl/en-nl.en \
    --tgt_data_path data/en-nl/en-nl.nl \
    --decomposer_type meat \
    --output_figure meat-labse-opl-ennl \
    --src_lang en \
    --tgt_lang nl;

python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse.pt \
    --src_data_path data/en-pt/en-pt.en \
    --tgt_data_path data/en-pt/en-pt.pt \
    --decomposer_type dream \
    --output_figure dream-labse-enpt \
    --src_lang en \
    --tgt_lang pt;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse-opl-w1.pt \
    --src_data_path data/en-pt/en-pt.en \
    --tgt_data_path data/en-pt/en-pt.pt \
    --decomposer_type dream \
    --output_figure dream-labse-opl-enpt \
    --src_lang en \
    --tgt_lang pt;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse.pt \
    --src_data_path data/en-pt/en-pt.en \
    --tgt_data_path data/en-pt/en-pt.pt \
    --decomposer_type meat \
    --output_figure meat-labse-enpt \
    --src_lang en \
    --tgt_lang pt;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse-opl-w1.pt \
    --src_data_path data/en-pt/en-pt.en \
    --tgt_data_path data/en-pt/en-pt.pt \
    --decomposer_type meat \
    --output_figure meat-labse-opl-enpt \
    --src_lang en \
    --tgt_lang pt;

python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse.pt \
    --src_data_path data/en-ro/en-ro.en \
    --tgt_data_path data/en-ro/en-ro.ro \
    --decomposer_type dream \
    --output_figure dream-labse-enro \
    --src_lang en \
    --tgt_lang ro;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse-opl-w1.pt \
    --src_data_path data/en-ro/en-ro.en \
    --tgt_data_path data/en-ro/en-ro.ro \
    --decomposer_type dream \
    --output_figure dream-labse-opl-enro \
    --src_lang en \
    --tgt_lang ro;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse.pt \
    --src_data_path data/en-ro/en-ro.en \
    --tgt_data_path data/en-ro/en-ro.ro \
    --decomposer_type meat \
    --output_figure meat-labse-enro \
    --src_lang en \
    --tgt_lang ro;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse-opl-w1.pt \
    --src_data_path data/en-ro/en-ro.en \
    --tgt_data_path data/en-ro/en-ro.ro \
    --decomposer_type meat \
    --output_figure meat-labse-opl-enro \
    --src_lang en \
    --tgt_lang ro;

python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse.pt \
    --src_data_path data/en-zh/en-zh.en \
    --tgt_data_path data/en-zh/en-zh.zh \
    --decomposer_type dream \
    --output_figure dream-labse-enzh \
    --src_lang en \
    --tgt_lang zh;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/dream-labse-opl-w1.pt \
    --src_data_path data/en-zh/en-zh.en \
    --tgt_data_path data/en-zh/en-zh.zh \
    --decomposer_type dream \
    --output_figure dream-labse-opl-enzh \
    --src_lang en \
    --tgt_lang zh;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse.pt \
    --src_data_path data/en-zh/en-zh.en \
    --tgt_data_path data/en-zh/en-zh.zh \
    --decomposer_type meat \
    --output_figure meat-labse-enzh \
    --src_lang en \
    --tgt_lang zh;
python -u visualization.py --model_name_or_path sentence-transformers/LaBSE \
    --pooler_path pooler/labse/meat-labse-opl-w1.pt \
    --src_data_path data/en-zh/en-zh.en \
    --tgt_data_path data/en-zh/en-zh.zh \
    --decomposer_type meat \
    --output_figure meat-labse-opl-enzh \
    --src_lang en \
    --tgt_lang zh;