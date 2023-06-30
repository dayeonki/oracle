## 돌려야 하는 파일
### (0) Install requirements

- 필요한 라이브러리들을 설치합니다.

    ```bash
    pip install -r requirements.txt
    ```
- <a href="https://github.com/yaushian/mSimCSE#pre-trained-model">mSimCSE 모델 체크포인트</a>를 다운로드 받습니다.
    - Our pre-trained model is available at here. For pre-trained cross-lingual model trained on English NLI, please download model here.
    - 2가지 모델 중에서 cross-lingual NLI 데이터로 학습된 cross-lingual model을 사용해야 합니다.

### (1) BUCC retrieval
- BUCC2018 데이터를 다운로드 받습니다.

    ```bash
    sh download_bucc.sh
    ```
- LaBSE, mSimCSE, LASER에 대해서 BUCC retrieval을 합니다.

    ```bash
    sh retrieve_bucc.sh
    ```
### (2) Visualization
- visualization 라이브러리를 이용해서 각 언어쌍별로 1,000개의 문장을 시각화합니다.
- 각 언어쌍별 데이터는 용량이 너무 커서 구글 드라이브 링크로 대신합니다. 여기서 바로 다운로드를 받으시면 됩니다: https://drive.google.com/file/d/1o4qoDZ_UV0PJuoH3yOUAPDpVqsdzYtbe/view?usp=sharing

    ```bash
    sh visualize.sh
    ```

### (3) Massive retrieval
- (현재 생각하기로는) en-fr와 en-zh 언어쌍에 대해서 massive retrieval을 진행합니다.
- Massive retrieval 코드는 `to_run` 코드 밖에 `massive_retrieval` 폴더에 위치해있습니다.
