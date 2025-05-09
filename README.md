# ETF PREDICT PROJECT


## 개요

최근 K-뷰티 산업은 아시아 및 북미 시장을 중심으로 급속히 성장하며, 국가 주요 산업으로서의 중요성이 더욱 부각되고 있다.    
이러한 배경에서 본 프로젝트에서는 뉴스 기사, 기업 매출 및 성장률, 주식 시장 변동성과의 연관성을 분석하고자 한다.     
또한 이러한 데이터를 기반으로 최종적으로 2025년 국내 화장품 ETF 상품의 지수를 예측해보는 것을 목표로 한다.    

---

## 프로젝트 단계별 코드

```
ETF_PREDICT_PROJECT
├── Crawler
│   ├── news_crawler1.py
│   ├── news_crawler2.py
│   ├── research_report_crawler.py
│   └── stock_price_crawler.py
│
├── Data_Preprocessing
│   ├── news_cleansing.py
│   ├── news_tokenizer_sentences_words.py
│   ├── pdf_to_text.py
│   ├── research_report_cleansing.py
│   ├── research_tokenizer.py
│   └── news_report_merge.py
│
├── Sentiment_Analysis
│   ├── sentiment_analysis.py
│   └── sentiment_weights_apply.py
│
├── Modeling
│   ├── LinearRegression.py
│   ├── GradientBoosting.py
│   ├── XGBoost_LSTM.py
│   └── XGBoost.py
│
├── Streamlit
│   ├── app.py
│   └── requirements

```

---

## 흐름

### 1. 자료 수집  

**대상 데이터**: TIGER ETF에 포함된 17개 기업의 최근 약 10년치 데이터(2015-2025)

- **네이버 뉴스 크롤링** (`news_crawler1.py`, `news_crawler2.py`)  
  - 키워드: 각 기업명  
  - `BeautifulSoup`, `Selenium` 활용  
  - 중복 크롤링 방지 포함

- **기업별 주가 데이터 수집** (`stock_price_crawler.py`)
  - `BeautifulSoup`, `Selenium` 결합하여 활용

- **기업별 사업보고서 및 증권사 리서치 보고서 수집** (`research_report_crawler.py`)  
  - `Selenium`을 활용, PDF 형식의 보고서 다운로드

- **TIGER ETF 지수 변동 데이터 수집**  

<br>

### 2. 데이터 전처리

- **뉴스 데이터 클렌징** (`news_cleansing.py`)  
  - 불용어 리스트 및 정규표현식 활용

- **리서치 보고서 텍스트 변환 및 클렌징** (`pdf_to_text.py`, `research_report_cleansing.py`)  
  - PDF 형식의 리서치 보고서를 텍스트로 변환  
  - 불용어 리스트 및 정규표현식 활용하여 클렌징

- **데이터 토큰화** (`news_tokenizer_sentences_words.py`, `research_tokenizer.py`)  
  - `Mecab` 및 `konlpy` 활용

- **데이터 병합** (`news_report_merge.py`)  
  - 뉴스 데이터와 리서치 보고서 데이터 하나의 파일로 병합

- **재무지표 데이터 구축**  
  - 사업보고서를 기반으로 기업별 재무지표 데이터 구축 (수작업)

<br>

### 3. 감성 분석

- **감성 분석 수행** (`sentiment_analysis.py`)  
  - `KoBERT`를 활용하여 뉴스 기사와 리서치 보고서에 대한 감성 분석 진행

- **감성 점수 계산 및 가중치 적용** (`sentiment_weights_apply.py`)  
  - 각 데이터별 감성 점수를 계산 후, 기업의 일자별 평균 감성 점수를 산출  
  - ETF 상품 내 기업별 비중을 바탕으로 가중치 적용하여 최종 감성 지수 도출 (기업별, 일간)

<br>

### 4. 모델 학습 및 평가

모든 모델은 일자 기준 전후 2일, 총 5일간의 데이터를 활용한 예측을 진행

### 1) LinearRegression (`LinearRegression.py`)  
- 2015-2023 학습 → 2024 예측 (분기)  
- 분기별 재무제표, 소비자 물가지수, 화장품 지수 데이터 활용  
- 변수 중요도 확인하여 피쳐 선택  
- 전반적인 추세 분석 중심  

### 2) GradientBoosting (`GradientBoosting.py`)
- 2015-2024 학습 → 2025 예측 (일자)   
- ETF 가격 데이터 전반 활용('종가', '시가', '고가', '저가', '거래량', '변동 %')  
- 종가가 아닌 수익률 예측  
- 1차 모델 학습 후 변수 중요도를 파악하여 주요 피처 선택 및 재학습  
- 데이터 스케일링 적용

### 3) XGBoost+LSTM (`XGBoost_LSTM.py`)  
- 2015-2024 학습 → 2025 예측 (일자)   
- 감성점수 데이터만을 활용  
- 학습 단계에서 정규화 진행  
- 모델 성능평가 단계에서는 역정규화 진행  
- 오차 크기를 기준으로 예측 실패한 일자들에 대한 시각화

### 4) XGBoost (`XGBoost.py`)  
- 2015-2024 학습 → 2025 예측 (분기)   
- 중국 경제 지표, 거시 지표 등 모든 데이터 활용  
- 다양한 거시적 변수와 시장 환경을 반영   
- 분기 단위의 예측 → 중장기적 ETF 흐름 전망 가능

<br>

### 5. 대시보드 구현

- Streamlit을 활용, 예측 결과를 시각화한 대시보드를 구성
- `app.py`, `requirements.txt`

---

## 주요 기술 스택 및 라이브러리

- 데이터 크롤링: `BeautifulSoup`, `Selenium`, `requests`  
- 전처리 및 정규화: `pandas`, `MinMaxScaler`, `re`, `KoBERT`, `konlpy`, `Mecab`, `regex`  
- 머신러닝(딥러닝): `scikit-learn`, `XGBoost`, `PyTorch`, `GradientBoosting`  
- 시각화: `matplotlib`, `Streamlit`

---

## 프로젝트 결과 요약

| 모델              | 장점 | 적합한 용도 |
|-------------------|------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **LinearRegression**        | - 선형회귀 모델만으로 **높은 R² 값을 확보**<br>- 정규화 및 변수 선별을 통해 **높은 정확도**를 확보 | - 프로젝트	**기초 예측 모델**로 적합               |
| **Gradient Boosting** | - **여러 변수들의 비선형관계를 효과적으로 모델링** | - **복잡한 데이터셋에 대한 단기 예측**에 유리                                 |
| **XGBoost + LSTM**  | - 중단기 예측 성능이 뛰어남<br>- 두 모델의 결합으로 **시계열 학습 및 정확도 향상**에 효과적 | -	ETF, 주식 등 **시계열 데이터의 중단기 예측**에 적합 |
| **XGBoost**        | - ‘분기별 수익률’에 대한 모델로, **장기 예측에 효과적**<br>- 비선형 관계의 다양한 변수들을 효과적으로 반영 | - 대규모 **복잡한 데이터셋에 대한 장기적 정밀 예측**에 적합 |


---

## 한계점

| 모델                  | 한계 |
|-----------------------|------|
| **LinearRegression**          | - 선형회귀 특성상 변수들의 **비선형 관계나 시계열 특성 반영이 어려워** 실제 투자 판단에 활용하기에는 정밀도가 부족함 |
| **Gradient Boosting** | - 시계열 데이터를 반영하지 못해 장기적 예측은 어려움<br>- 평균적인 경향을 학습하는 모델로 급격한 **상승/하락은 예측하기 어려움** |
| **XGBoost + LSTM**    | - 두 모델을 결합하기 때문에 **데이터 전처리 및 유지 보수가 복잡함**<br>- 다양한 데이터를 반영하지 못하고 오직 **감성점수에만 의존함**<br>- Epoch 및 window size를 늘리면 성능 개선 가능성이 있음<br>- 예측 결과에 대한 원인 분석 어려움 |
| **XGBoost**           | - 다양한 변수를 활용한 것은 장기 예측에는 효과적이지만 **단기예측에 있어서는 오히려 노이즈로 작용**<br>- 하이퍼파라미터 튜닝이 복잡함<br>- 모델 자체가 블랙박스 성향이 강해 **예측 원인을 분석하는 것이 어려움** |


---
