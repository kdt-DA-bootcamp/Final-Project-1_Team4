# ✅ 정규화 + LSTM + XGBoost 앙상블 (2024년까지 학습, 2025년 예측)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import datetime
import matplotlib.dates as mdates

# 📌 1. 데이터 로딩
merged_df = pd.read_csv(r"C:\Users\bona_\OneDrive\Desktop\FINAL PROJECT 1\코드\merged_df.csv")
merged_df["날짜"] = pd.to_datetime(merged_df["날짜"])
merged_df = merged_df.dropna().sort_values("날짜").reset_index(drop=True)

# 📌 2. 정규화 (MinMax)
def parse_number(x):
    if isinstance(x, str):
        x = x.replace(",", "")
        if 'K' in x:
            return float(x.replace('K', '')) * 1e3
        elif 'M' in x:
            return float(x.replace('M', '')) * 1e6
        elif 'B' in x:
            return float(x.replace('B', '')) * 1e9
        else:
            try:
                return float(x)
            except:
                return np.nan
    return x

merged_df["거래량"] = merged_df["거래량"].apply(parse_number)

# 📌 3. 학습/예측 데이터 분할
train_df = merged_df[merged_df["날짜"] < "2025-01-01"]
future_df = merged_df[merged_df["날짜"] >= "2025-01-01"]

# 정규화는 학습 데이터 기준으로 fit
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df[["종가", "거래량", "가중 감성 점수"]])
train_scaled_df = pd.DataFrame(train_scaled, columns=["종가", "거래량", "감성"])
train_scaled_df = train_scaled_df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

# 📌 4. Dataset 클래스 정의
class ETFLSTMDataset(Dataset):
    def __init__(self, df, window_size=5):
        self.X, self.y = [], []
        data = df.values
        for i in range(window_size, len(data)):
            self.X.append(data[i-window_size:i])
            self.y.append(data[i][0])
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# 📌 5. Train Dataset 준비
window_size = 5
dataset = ETFLSTMDataset(train_scaled_df, window_size)
train_loader = DataLoader(dataset, batch_size=32)

# 📌 6. LSTM 모델 정의
class ETF_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1]).squeeze()

# 📌 7. 모델 학습
model = ETF_LSTM(input_size=3).to("cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# 📌 8. LSTM + XGBoost 예측용 피처 구성
train_lstm_preds, train_true_vals = [], []
model.eval()
with torch.no_grad():
    for x_batch, y_batch in train_loader:
        output = model(x_batch)
        train_lstm_preds.extend(output.view(-1).numpy())
        train_true_vals.extend(y_batch.view(-1).numpy())
train_residuals = np.array(train_true_vals) - np.array(train_lstm_preds)

# XGBoost 학습용 피처 만들기
def make_xgb_features(df, window_size=5):
    features = []
    for i in range(window_size, len(df)):
        window = df.iloc[i-window_size:i]
        feat = []
        for _, row in window.iterrows():
            feat.extend([row["종가"], row["거래량"], row["감성"]])
        features.append(feat)
    return np.array(features)

xgb_train_X = make_xgb_features(train_scaled_df, window_size)
xgb_model = XGBRegressor(n_estimators=100)
xgb_model.fit(xgb_train_X, train_residuals)

# ✅ 학습셋 시각화 (정규화 해제)
train_true_vals_rescaled = scaler.inverse_transform(
    np.concatenate([np.array(train_true_vals).reshape(-1, 1), np.zeros((len(train_true_vals), 2))], axis=1)
)[:, 0]
train_preds_rescaled = scaler.inverse_transform(
    np.concatenate([(np.array(train_lstm_preds) + xgb_model.predict(xgb_train_X)).reshape(-1, 1), np.zeros((len(train_true_vals), 2))], axis=1)
)[:, 0]
train_dates = train_df["날짜"].iloc[window_size:]

plt.figure(figsize=(12, 5))
plt.plot(train_dates, train_true_vals_rescaled, label="실제 ETF (Train)")
plt.plot(train_dates, train_preds_rescaled, label="예측 ETF (Train)", linestyle="--")
plt.title("📈 학습셋 예측 결과")
plt.xlabel("날짜")
plt.ylabel("ETF 종가")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# 📌 9. 2025년 예측 준비
future_scaled = scaler.transform(future_df[["종가", "거래량", "가중 감성 점수"]])
future_scaled_df = pd.DataFrame(future_scaled, columns=["종가", "거래량", "감성"])

# LSTM 예측용 시퀀스 만들기
full_df = pd.concat([train_scaled_df, future_scaled_df], ignore_index=True)
X_future = []
for i in range(len(train_scaled_df), len(full_df)):
    if i - window_size < 0:
        continue
    X_future.append(full_df.iloc[i-window_size:i].values)
X_future_tensor = torch.tensor(X_future, dtype=torch.float32)

# LSTM 예측
model.eval()
with torch.no_grad():
    lstm_future_preds = model(X_future_tensor).numpy()

# XGBoost 피처 만들기
xgb_future_X = make_xgb_features(full_df, window_size)[len(train_scaled_df) - window_size:]
residual_future = xgb_model.predict(xgb_future_X)
final_preds = lstm_future_preds + residual_future

# 📌 10. 역정규화 및 시각화
future_dates = future_df["날짜"].iloc[:len(final_preds)]
final_preds_rescaled = scaler.inverse_transform(
    np.concatenate([final_preds.reshape(-1, 1), np.zeros((len(final_preds), 2))], axis=1)
)[:, 0]

# 실제 2025 ETF와 비교
true_future_etf = future_df["종가"].iloc[:len(final_preds)].values

# 평가 지표
rmse = np.sqrt(mean_squared_error(true_future_etf, final_preds_rescaled))
mae = mean_absolute_error(true_future_etf, final_preds_rescaled)
r2 = r2_score(true_future_etf, final_preds_rescaled)
print("\n📊 [2025년 예측 평가 결과]")
print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ MAE:  {mae:.2f}")
print(f"✅ R²:   {r2:.4f}")

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(future_dates, true_future_etf, label="실제 ETF (2025)", linewidth=2)
plt.plot(future_dates, final_preds_rescaled, label="예측 ETF (2025)", linestyle="--")
plt.gca().xaxis.set_major_locator(mdates.DateLocator(interval=10))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.title("📈 ETF 지수 예측 (2025년)")
plt.xlabel("날짜")
plt.ylabel("ETF 종가")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()







pred_df = pd.DataFrame({
    "날짜": future_dates,
    "실제 ETF": true_future_etf,
    "예측 ETF": final_preds_rescaled
})
pred_df.to_csv("etf_predictions_2025.csv", index=False, encoding="utf-8-sig")

# 📌 12. 예측 실패일 추출 및 저장
pred_df["오차"] = np.abs(pred_df["실제 ETF"] - pred_df["예측 ETF"])
failure_df = pred_df.sort_values(by="오차", ascending=False).head(20)
failure_df.to_csv("etf_prediction_failures.csv", index=False, encoding="utf-8-sig")


import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 📌 13. 예측 실패일 시각화
plt.figure(figsize=(12, 6))
plt.plot(future_dates, true_future_etf, label="실제 ETF (2025)", linewidth=2)
plt.plot(future_dates, final_preds_rescaled, label="예측 ETF (2025)", linestyle="--")
plt.scatter(failure_df["날짜"], failure_df["실제 ETF"], color="red", label="예측 실패일", zorder=5)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.title("2025년 ETF 예측, 실제, 실패일")
plt.xlabel("날짜")
plt.ylabel("ETF 종가")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 📌 14. 실패일 원인 탐색용 출력
print("\n📌 예측 오차 큰 상위 5개 일자:")
print(failure_df.head(5))
print("\n오차 평균:", failure_df["오차"].mean())

print("\n✅ 예측 결과 및 상위 오류일 시각화/출력 완료!")
