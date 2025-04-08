# ✅ 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ✅ 1. 데이터 불러오기
df = pd.read_csv('/content/drive/MyDrive/화장품/etf_감성_통합.csv')  # ← 파일 경로 맞게 수정
df['날짜'] = pd.to_datetime(df['날짜'])
df = df[(df['날짜'] >= '2015-01-01') & (df['날짜'] <= '2025-04-03')]

# ✅ 2. X, y 설정
features = ['종가', '시가', '고가', '저가', '거래량', '변동 %']
X = df[features]
y = df['수익률']

# ✅ 3. 날짜 기준 분리
train_mask = df['날짜'] <= '2024-12-31'
test_mask = df['날짜'] >= '2025-01-01'

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
dates_test = df['날짜'][test_mask]

# ✅ 4. 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ 5. 모델 학습
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

# ✅ 6. 예측
y_pred = model.predict(X_test_scaled)

# ✅ 7. 평가
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ [GradientBoosting] RMSE: {rmse:.6f}")
print(f"✅ [GradientBoosting] R² Score: {r2:.4f}")

# ✅ 8. 시각화
plt.figure(figsize=(10, 5))
plt.plot(dates_test.values, y_test.values, label='True', marker='o')
plt.plot(dates_test.values, y_pred, label='Predicted', marker='x')
plt.title('2025 Profit Forecast (GradientBoosting)')
plt.xlabel('Date')
plt.ylabel('Profit')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ 9. CSV 저장
result_df = pd.DataFrame({
    '날짜': dates_test.values,
    '실제 수익률': y_test.values,
    '예측 수익률': y_pred
})
result_df.to_csv('/content/GradientBoosting_예측결과_2025.csv', index=False)
print("📁 저장 완료: GradientBoosting_예측결과_2025.csv")


##피쳐셀렉션
# ✅ SHAP 설치
!pip install shap

# ✅ SHAP 모듈 불러오기
import shap

# ✅ SHAP explainer 정의
explainer = shap.Explainer(model, X_train_scaled)
shap_values = explainer(X_train_scaled)

# ✅ 요약 플롯 (변수 중요도)
shap.plots.beeswarm(shap_values)

# ✅ 바 플롯 (가장 영향 큰 변수 순서)
shap.plots.bar(shap_values)


#모델 재학습
# ✅ 필수 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ✅ 1. 데이터 불러오기
df = pd.read_csv('/content/drive/MyDrive/화장품/ETF_감성_중국.csv')
df['날짜'] = pd.to_datetime(df['날짜'])
df = df[(df['날짜'] >= '2015-01-01') & (df['날짜'] <= '2025-04-03')]

# ✅ 2. SHAP 기반 상위 3개 피처 선택
features = ['시가', '종가', '변동 %']  # ← 중요도 높은 피처만 사용
X = df[features]
y = df['수익률']

# ✅ 3. 날짜 기준 분할
train_mask = df['날짜'] <= '2024-12-31'
test_mask = df['날짜'] >= '2025-01-01'

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
dates_test = df['날짜'][test_mask]

# ✅ 4. 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ 5. 모델 학습
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

# ✅ 6. 예측
y_pred = model.predict(X_test_scaled)

# ✅ 7. 평가
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ [GradientBoosting - 핵심 피처] RMSE: {rmse:.6f}")
print(f"✅ [GradientBoosting - 핵심 피처] R² Score: {r2:.4f}")

# ✅ 8. 시각화
plt.figure(figsize=(10, 5))
plt.plot(dates_test.values, y_test.values, label='True', marker='o')
plt.plot(dates_test.values, y_pred, label='Predicted', marker='x')
plt.title('2025 Forecast (GradientBoosting)')
plt.xlabel('Date')
plt.ylabel('Profit')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ 9. 저장
result_df = pd.DataFrame({
    '날짜': dates_test.values,
    '실제 수익률': y_test.values,
    '예측 수익률': y_pred
})
result_df.to_csv('/content/GradientBoosting_핵심피처_예측결과.csv', index=False)
print("📁 저장 완료: GradientBoosting_핵심피처_예측결과.csv")



# 시각화
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load result
df_result = pd.read_csv("/content/drive/MyDrive/화장품/GradientBoosting_핵심피처_예측결과.csv")
df_result['Date'] = pd.to_datetime(df_result['날짜'])
df_result = df_result.sort_values('Date')

# Rename for consistency
df_result['Actual Return'] = df_result['실제 수익률']
df_result['Predicted Return'] = df_result['예측 수익률']
df_result['Error'] = df_result['Actual Return'] - df_result['Predicted Return']
df_result['Absolute Error'] = np.abs(df_result['Error'])

# ✅ 1. Line plot
plt.figure(figsize=(10, 5))
plt.plot(df_result['Date'], df_result['Actual Return'], label='Actual', marker='o')
plt.plot(df_result['Date'], df_result['Predicted Return'], label='Predicted', marker='x')
plt.title('Actual vs Predicted Return')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ✅ 2. Histogram (no seaborn)
plt.figure(figsize=(8, 4))
plt.hist(df_result['Error'], bins=20, edgecolor='black')
plt.title('Prediction Error Distribution')
plt.xlabel('Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ 3. Direction Accuracy
df_result['Actual Direction'] = np.sign(df_result['Actual Return'])
df_result['Predicted Direction'] = np.sign(df_result['Predicted Return'])
direction_accuracy = (df_result['Actual Direction'] == df_result['Predicted Direction']).mean()
print(f"✅ Direction Accuracy: {direction_accuracy:.2%}")

# ✅ 4. Opportunity Capture (profit zone)
threshold = 0.005  # 0.5%
profitable_actual = df_result[df_result['Actual Return'] >= threshold]
profitable_predicted = profitable_actual[profitable_actual['Predicted Return'] >= threshold]
capture_rate = len(profitable_predicted) / len(profitable_actual) if len(profitable_actual) > 0 else 0.0
print(f"✅ Opportunity Capture (≥ 0.5% actual return): {capture_rate:.2%}")
