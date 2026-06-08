# 📰 Fake News Detection System (LSTM 기반 허위 정보 탐지)

딥러닝(LSTM)과 자연어 처리(NLP) 기법을 활용하여 뉴스 기사의 진위 여부를 판별하는 프로젝트입니다.
<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/7fd5895f-9fc4-48be-be18-38d652be3f97" />
<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/634d0929-d0f1-489a-8e74-a1d84f4bb4b6" />





## 🚀 프로젝트 소개
본 프로젝트는 최근 미디어 환경에서 심각한 문제로 대두되는 허위 정보 확산을 기술적으로 해결하고자 기획되었습니다. LSTM 신경망을 통해 뉴스 기사의 문맥적 패턴을 학습합니다.

## 📊 주요 기능 및 분석
### 1. 데이터 분석(EDA)
뉴스 기사의 어휘 가중치(TF-IDF)와 길이 분포를 시각화하여 데이터의 특성을 파악했습니다.

<img width="354" height="134" alt="image" src="https://github.com/user-attachments/assets/2a39dcfc-4371-4d4e-a189-e0ba6a7aa275" />


### 2. 성능 평가
모델의 예측 성능을 혼동 행렬(Confusion Matrix)과 ROC 커브를 통해 검증했습니다.

<img width="328" height="142" alt="image" src="https://github.com/user-attachments/assets/5aeb2753-8807-42b6-80bd-561b2719f6fd" />


## 📊 모델 성능
- **Test Accuracy:** 94.4%
- **AUC Score:** (본인의 AUC 수치 입력)

## 📂 프로젝트 구조
<img width="903" height="997" alt="image" src="https://github.com/user-attachments/assets/a0ec7a7f-8f95-47bb-bc16-3299b4cf479d" />


## ⚙️ 실행 방법
본 프로젝트를 로컬 환경에서 실행하려면 아래 절차를 따라주세요.

### 1. 저장소 클론 및 환경 설정
터미널(또는 CMD)에서 프로젝트를 저장할 폴더로 이동한 뒤, 아래 명령어를 순서대로 입력합니다.

# 필수 라이브러리 설치
pip install -r requirements.txt
## 📁 대용량 파일 다운로드
아래 구글 드라이브에서 다운로드 후 프로젝트 폴더에 넣어주세요.

| 파일 | 링크 |
|------|------|
| WELFake_Dataset.csv | [다운로드](https://drive.google.com/file/d/1BPSu-kBhPNrGJvU4dmBqFwyvWnmPircn/view?usp=sharing) |
| fake_news_lstm_model.h5 | [다운로드](https://drive.google.com/file/d/1ixIPxEpIYoY5-cNUbOz2DLwywu03r1my/view?usp=sharing) |
| tokenizer.pkl | [다운로드](https://drive.google.com/file/d/1Xc-1DT3gQ7XzHyB3vaFKLh1r2zR7LBax/view?usp=sharing) |

##웹 대시보드 실행
준비가 완료되었다면, 터미널에서 다음 명령어를 입력하여 대시보드를 실행합니다.
streamlit run add.py
---
## 👨‍💻 개발자
- **Name:** Heo Jun-seong (허준성)
- **Affiliation:** Gwangju University (Computer Science Engineering)
