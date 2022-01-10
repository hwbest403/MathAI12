# Seq2Seq Models for Math Word Problem(MWP)

## 개요

본 프로젝트는 딥러닝을 이용해 수학 문제를 풀기 위해 Transformer Seq2Seq 모델을 응용하여 구현하였습니다. 본 코드의 [원본 구현](https://github.com/sogang-isds/Seq2Seq-model)에서 하이퍼 파라미터 및 모델 후처리 등 일부 수정하였습니다.

## 설치

### Python 가상환경 및 패키지

- git 받아오기

  ```bash
  https://github.com/sogang-isds/aichallenge-5th-model.git
  ```

- Python 가상환경 설치

  ```bash
  cd aichallenge-5th-model
  virtualenv -p python3 myenv
  source myenv/bin/activate
  ```

- Python 패키지 설치

  ```bash
  pip install -r requirements.txt
  ```



### GloVe 추가

[한국어 임베딩 튜토리얼](https://github.com/ratsgo/embedding/releases) 사이트에서 **[word-embeddings.zip](https://drive.google.com/open?id=1yHGtccC2FV3_d6C6_Q4cozYSOgA7bG-e)** 파일을 받아  `glove.txt` 파일을 `embedding` 디렉토리에 추가합니다.



## 데이터셋

데이터셋은 [aichallenge-5th](https://github.com/sogang-isds/aichallenge-5th
) 프로젝트에서 확인하실 수 있습니다.



## 설정

설정파일은 `Seq2Seq/config` 밑에 있는 `train_config.json`과 `test_config.json`을 확인합니다.



## 실행

- 학습

  ```bash
  python run_train.py --model_type Transformer
  ```

- 테스트

  ```bash
  python run_test.py --model_type Transformer
  ```

- Equation 예측결과에 대한 Python 코드 실행

  예측된 수식에 대해 Python 코드 생성 오류 및 코드 실행 오류를 확인하기 위해 사용

  ```bash
  python run_math_equation.py
  ```

- prediction 코드

  데이터를 1건씩 예측하면서 결과 확인 및 성능 체크. 챕터별 데이터 성능 확인에 사용

  ```bash
  python run_predict.py
  ```



## Copyright

본 프로젝트에 대한 저작권은 서강대학교 지능형 음성대화 인터페이스 연구실에 있습니다.

본 저장소는 비공개 저장소로 외부 유출을 금합니다.



