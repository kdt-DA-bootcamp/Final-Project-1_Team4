# 모델1. 선형회귀모델
# 가중치 적용된 감성점수 파일 활용

# 라이브러리 모음
import chardet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 파일 열기
with open("분기별 재무제표.csv", "rb") as f:
    rawdata = f.read()
    result = chardet.detect(rawdata)
    print(result)

cosmetics_cpi = pd.read_csv("분기별 소비자물가지수.csv", encoding="utf-8")
etf_prices = pd.read_csv("분기별 화장품 지수.csv", encoding="utf-8")
sentiment_scores = pd.read_csv("분기별 감성점수.csv", encoding="utf-8")

financials = pd.read_csv("분기별 재무제표.csv", encoding="euc-kr")

# 분기 기준으로 데이터 병합 및 전처리
data = cosmetics_cpi.merge(financials, on='분기').merge(etf_prices, on='분기').merge(sentiment_scores, on='분기')

# 쉼표(,)가 포함된 숫자 데이터를 정수 또는 실수로 변환
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].str.replace(",", "").astype("float", errors="ignore")
# 결측치 처리 (평균값으로 대체)
data.fillna(data.mean(numeric_only=True), inplace=True)

# 학습/예측 데이터 분할
data['년도'] = data['분기'].str[:4].astype(int)
train_data = data[data['년도'] <= 2023]
test_data = data[data['년도'] == 2024]

# 상관관계 분석
data_corr = data.corr(numeric_only=True)
print("변수 간 상관관계:\n", data_corr['종가'].sort_values(ascending=False))

# 중요한 변수 선택 (상관관계 절댓값이 높은 변수들)
important_features = data_corr['종가'].abs().sort_values(ascending=False).index[1:6]  # '종가' 제외하고 상위 5개 선택
print("선택된 변수:", important_features)

# 학습 및 테스트 데이터 분리
y_train = data[data['년도'] <= 2023]['종가']
X_train = data[data['년도'] <= 2023][important_features]
y_test = data[data['년도'] == 2024]['종가']
X_test = data[data['년도'] == 2024][important_features]

# 정규화 진행
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 회귀 모델 학습
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 2024년 데이터 예측
y_pred = model.predict(X_test_scaled)

# 예측 성능 평가
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("2024년 예측 MAE:", mae)
print("2024년 예측 R^2 Score:", r2)

# 결과 출력 및 비교
test_data = data[data['년도'] == 2024][['분기']].copy()
test_data['실제 종가'] = y_test.values
test_data['예측 종가'] = y_pred
print(test_data)

# 그래프 시각화
plt.figure(figsize=(10, 5))
plt.plot(test_data['분기'], test_data['실제 종가'], marker='o', label='실제 종가', linestyle='-')
plt.plot(test_data['분기'], test_data['예측 종가'], marker='s', label='예측 종가', linestyle='--')
plt.xlabel('분기')
plt.ylabel('ETF 종가')
plt.title('2024년 실제 종가 vs 예측 종가')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()