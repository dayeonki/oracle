## DREAM / MEAT Advanced Decomposer
아래는 DREAM와 MEAT Decomposer에 관한 재구현 및 Basic orthogonal constraint (BOC), Orthogonal projection loss (OPL)를 구현해놓은 레포지토리입니다.
관심 있는 언어쌍의 bitext 데이터를 이용해서 Embedding - Train - Retrieval의 순서로 활용할 수 있습니다. 아래는 jako (일본어-한국어) 언어쌍에 대한 예시입니다.

### (0) Install requirements.txt

- 필요한 라이브러리들을 설치합니다.

    ```bash
    pip install -r requirements.txt
    ```

### (1) Data preparation

- Bitext 데이터를 `decomposer/data/` 위치에 넣고 각 언어에 대한 파일은 txt 파일의 줄글 형태로 통일되게 입력됩니다.

### (2) Embed bitext data

- Bitext 데이터에 대해서 관심 있는 모델을 이용해 임베딩 생성을 진행하는 단계입니다.
- Run `script/embed.py`
    - `src_embed_path`와 `tgt_embed_path`는 임베딩 생성 후 해당 임베딩을 저장할 디렉토리를 의미합니다.
    - `train_embed_path`와 `valid_embed_path`는 임베딩 생성 후 train과 valid dataset으로 쪼갠 뒤의 임베딩을 저장할 디렉토리를 의미합니다.
    
    ```bash
    python3 embed.py --model_name_or_path [MODEL NAME] \
       --src_data_path [SOURCE DATASET PATH] \
       --tgt_data_path [TARGET DATASET PATH] \
       --src_embed_path [SOURCE EMBED DATASET PATH] \
       --tgt_embed_path [TARGET EMBED DATASET PATH] \
       --train_embed_path [TRAIN EMBED DATASET PATH] \
       --valid_embed_path [VALID EMBED DATASET PATH] \
       --src_lang [SOURCE LANGUAGE] \
       --tgt_lang [TARGET LANGUAGE] \
       --batch_size [BATCH SIZE, default=512] \
       --seed [SEED NUM, default=42]
    ```
    
- 임베딩 생성 후에는 지정한 path에 각각 train용 임베딩 파일과 validation용 임베딩 파일이 저장됩니다.

### (3) Train

- (2)에서 만든 임베딩을 이용해서 Decomposer 학습을 진행하는 단계입니다.
- Decomposer 변인 : {DREAM, MEAT}
- Train type 변인 : {Vanilla, BOC, OPL}
    - Vanilla는 기본적인 DREAM, MEAT 논문상의 Decomposer 학습 방식입니다.
    - BOC (Basic orthogonal constraint)는 source와 target language embedding간 orthogonality에 대한 제약을 준 학습 방식입니다.
    - OPL (Orthogonal projection loss)는 intra-class clustering (batch 내 동일한 언어의 language embedding은 유사하게끔)과 inter-class separation (batch 내 동일한 언어의 semantic과 language embedding은 orthogonal하게끔) 제약을 준 학습 방식입니다.
- DREAM Decomposer로 학습하고 싶은 경우에는 Run `script/train_dream.py`, MEAT Decomposer로 학습하고 싶은 경우에는 Run `script/train_meat.py`.
    
    ```python
    python3 train_dream.py -c ../config/dream_cnli_boc.yaml
    python3 train_meat.py -c ../config/meat_labse_opl.yaml
    ```
    
- 학습시 사용하는 configuration 세팅에 대해서 config 파일로 관리해줍니다.
    - `train_path`는 위에는 만든 train 임베딩 파일의 경로이고, `valid_path`는 valid 임베딩 파일의 경로입니다.
    - `model_name_or_path`는 임베딩 생성시 사용한 모델의 경로입니다. (LaBSE의 경우에는 huggingface 경로 사용)
    - `save_pooler_path`는 학습이 끝난 뒤에 가장 좋은 성능을 보이는 pooler를 저장하는 경로입니다.
    - `logging_path`는 1 train epoch마다 계산되는 loss의 값들을 기록하는 로그 파일 저장 경로입니다.
    - `train_type`는 학습 방식으로 vanilla, boc, opl 중 하나를 입력합니다.
    - `n_languages`는 사용하는 언어의 개수입니다. (jako Decomposer의 경우, 2개의 언어로 학습되었기 때문에 2입니다)
    - `weights`는 계산되는 loss별 가중치에 대한 하이퍼파라미터입니다.
        - vanilla 학습시 weight[0~3]만 사용되고 마지막 weight[4]는 사용되지 않습니다.
        - boc 학습시 weight[4]는 basic orthogonal loss에 대한 가중치입니다.
        - opl 학습시 weight[4]는 orthogonal projection loss에 대한 가중치입니다.
    
    ```yaml
    train_path: ../data/TRAIN_DATASET.pt
    valid_path: ../data/VALID_DATASET.pt
    model_name_or_path: {CNLI: path/to/models/XLM-R-large-cnli, LaBSE: sentence-transformers/LaBSE}
    save_pooler_path: ../models/POOLER_FILE_NAME.pt
    logging_path: ../logs/LOG_FILE_NAME.log
    train_type: {vanilla, boc, opl}
    learning_rate: 1e-5
    n_languages: 2
    batch_size: 512
    seed: 42
    weights: [[1,1,1,1,1]]
    ```
    

### (4) Retrieval inference

- (3)에서 학습된 Decomposer에 대해서 성능 평가를 하기 위해서는 Run `script/retrieve.py`.
    
    ```bash
    python3 retrieve.py --model_name_or_path [MODEL NAME] \
       --pooler_path [TRAINED POOLER PATH] \
       --src_data_path [SOURCE RETRIEVAL DATASET PATH] \
       --tgt_data_path [TARGET RETRIEVAL DATASET PATH] \
       --save_path [JSONL PATH] \
       --src_lang [SOURCE LANGUAGE] \
       --tgt_lang [TARGET LANGUAGE] \
       --batch_size [BATCH SIZE, default=512]
    ```
    
- `save_path`에 지정된 경로로 아래 형태의 jsonl 파일이 저장됩니다.
    
    ```json
    {
        "source_lang": "ja"
        "target_lang": "ko"
        "semantic accuracy": 0.943,
        "language accuracy": 0.052,
        "source_text": 今日はいい日で、ローラースケートの練習ができる。
        "target_text": 오늘은 날씨가 좋아서 롤러스케이트 연습을 할 수 있다.,
        "prediction_idx": 2,
        "prediction": 오늘은 날씨가 좋아서 롤러스케이트 연습을 할 수 있다.
    }
    ```
    

### (5) Visualize

- 추가적으로 학습한 Decomposer를 이용해서 임베딩을 만들고 이를 `datavis` 라이브러리로 시각화하고 싶은 경우, Run `script/visualize.py`.
    - `output_figure`에 지정된 경로로 html bokeh 파일이 저장됩니다.
    
    ```bash
    python3 visualize.py --model_name_or_path [MODEL NAME] \
       --pooler_path [TRAINED POOLER PATH] \
       --src_data_path [SOURCE RETRIEVAL DATASET PATH] \
       --tgt_data_path [TARGET RETRIEVAL DATASET PATH] \
       --src_lang [SOURCE LANGUAGE] \
       --tgt_lang [TARGET LANGUAGE] \
       --batch_size [BATCH SIZE, default=512] \
       --output_figure [OUTPUT FIGURE PATH]
    ```

### (6) Fine-tune

- 만약 Decomposer과 backbone model의 학습을 동시에 하는 fine-tuning을 진행하고 싶은 경우에는 `embed.py` 대신에 `tokenize.py`를 돌리고, 학습 코드는 `train_dream.py`이나 `train_meat.py` 대신에 `fine_tune_dream.py`와 `fine_tune_meat.py` 코드를 돌립니다.
- Fine-tuning의 경우, backbone model의 parameter update가 GPU 메모리 연산량을 많이 요구하기 때문에 `accelerator`를 이용해서 구현했습니다.
    
    
    |  | Decomposer (no fine-tune) | Decomposer (fine-tune) |
    | --- | --- | --- |
    | Input data | Bitext 형태의 txt 파일 (source.txt, target.txt) | Bitext 형태의 txt 파일 (source.txt, target.txt) |
    | Tokenize + Embed | Run `script/embed.py` <br> 미리 사용하려는 bitext 데이터에 대해서  tokenization과 embedding을 진행해 놓는다. <br> Embedding된 데이터를 dataloader로 불러와서 Decomposer 학습에 사용한다.  | Run `script/tokenizer.py` <br> 미리 사용하려는 bitext 데이터에 대해서  tokenization을 진행해 놓는다. <br> Tokenize된 데이터를 dataloader로 불러와서 Decomposer 및 모델 학습시 실시간으로 dynamic하게 embedding을 만들어준다. |
    | Parameter update | Run `script/train_dream.py`, `script/train_meat.py` <br> Decomposer 학습 및 parameter update | Run `script/fine_tune_dream.py`, `script/fine_tune_meat.py` <br> Decomposer와 backbone model 학습 및 parameter update |
    | Inference | Run `script/retrieve.py` | Run `script/retrieve.py` |
- Run `script/tokenize.py`
    
    ```bash
    python3 tokenize.py --model_name_or_path [MODEL NAME] \
       --src_data_path [SOURCE DATASET PATH] \
       --tgt_data_path [TARGET DATASET PATH] \
       --output_dir [TOKENIZED OUTPUT PATH] \
       --src_lang [SOURCE LANGUAGE] \
       --tgt_lang [TARGET LANGUAGE]
    ```
    
- Run `script/fine_tune_dream.py`이나 `script/fine_tune_meat.py`
    
    ```bash
    accelerate config
    
    accelerate launch fine_tune_dream.py -c ../config/fine_tune_dream_labse_boc.yaml
    accelerate launch fine_tune_meat.py -c ../config/fine_tune_meat_labse_boc.yaml
    ```
    
- 학습시 사용하는 configuration 세팅에 대해서 config 파일로 관리해줍니다. 각각의 configuration은 위 yaml 파일 형식과 동일합니다.
    - `train_path`는 위에는 만든 train 데이터의 tokenized 파일의 경로이고, `valid_path`는 valid 데이터의 tokenized 파일의 경로입니다.
    
    ```yaml
    train_path: ../data/TOKENIZED_TRAIN_DATASET_PATH
    valid_path: ../data/TOKENIZED_VALID_DATASET_PATH
    model_name_or_path: {CNLI: path/to/models/XLM-R-large-cnli, LaBSE: sentence-transformers/LaBSE}
    save_pooler_path: ../models/SAVE_FILE_PATH
    logging_path: ../logs/LOG_FILE_NAME.log
    train_type: {vanilla, boc, opl}
    learning_rate: 1e-5
    n_languages: 2
    batch_size: 512
    seed: 42
    weights: [[1,1,1,1,1]]
    ```

- 학습 후에는 `config.json`, `pytorch_model.bin` (학습된 backbone model), 그리고 `pooler_model.bin` (학습된 pooler decomposer)이 저장됩니다.
