# 모델4. xgboost

# 지표 설명
# 재무지표: 매출액증가율, 영업이익증가율, 당기순이익증가율, ROE변화율, 부채비율, 영업이익률, 순이익률, 재고자산증가율, 영업활동현금흐름 증가율, EPS증가율
# 감성지표: 뉴스/리포트 기반 분기 평균 감성 점수
# 활동비중: 광고선전비 비중, R&D비중 (총매출 또는 자산 대비 비율)
# 국내 거시지표: 소비자물가지수(CPI), 기준금리, 화장품 물가지수
# 중국 거시지표: 중국 GDP, 중국 CPI, 중국 소매판매 증가율, 위안화 환율(CNY/KRW)
# 대외 거시지표: USD/KRW 환율, 국제유가(WTI)

# 라이브러리 모음
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# 파일 경로 설정 및 로드
base_path = "C:/Users/likel/Desktop/project_team4"
train_path = os.path.join(base_path, "0406/train_data_with_all_y.xlsx")
test_path = os.path.join(base_path, "0406/2025Q1_test_data_with_all_labeled.xlsx")
china_path = os.path.join(base_path, "거시지표/중국지표_정제_통합_v3.csv")
korea_path = os.path.join(base_path, "거시지표/국내_거시지표_통합.csv")
external_macro_path = os.path.join(base_path, r"C:\Users\likel\Desktop\project_team4\거시지표\대외거시지표(환율_유가).csv")
sentiment_path = os.path.join(base_path, "감성지수/감성지수_분기별통합_cleaned.csv")

df_train = pd.read_excel(train_path)
df_test = pd.read_excel(test_path)
df_china = pd.read_csv(china_path)
df_korea = pd.read_csv(korea_path)
df_external_macro = pd.read_csv(external_macro_path)
df_sentiment = pd.read_csv(sentiment_path)

# 모든 데이터 병합
df_train = df_train.merge(df_china, on="분기", how="left")
df_train = df_train.merge(df_korea, on="분기", how="left")
df_train = df_train.merge(df_external_macro, on="분기", how="left")
df_train = df_train.merge(df_sentiment, on=["분기", "기업명"], how="left")

df_test = df_test.merge(df_china, on="분기", how="left")
df_test = df_test.merge(df_korea, on="분기", how="left")
df_test = df_test.merge(df_external_macro, on="분기", how="left")
df_test = df_test.merge(df_sentiment, on=["분기", "기업명"], how="left")

# 결측치가 있는 행 제거
df_train = df_train[df_train["라벨_y_q"].notna()].copy()
df_test = df_test[df_test["라벨_y_q"].notna()].copy()

# 피쳐셀렉션 및 학습/예측 데이터 분할
y_train_cls = df_train["라벨_y_q"]
y_test_cls = df_test["라벨_y_q"]

exclude_cols = ["기업명", "분기", "기준일", "기준가_q", "익분기종가_q", 
                "수익률_y_q", "라벨_y_q", "예측_라벨_y_q"]
feature_cols = [col for col in df_train.columns if col not in exclude_cols]

X_train = df_train[feature_cols].copy()
X_test = df_test[feature_cols].copy()

# 데이터 전처리
# 결측치 제거
X_train.replace("-", np.nan, inplace=True)
X_test.replace("-", np.nan, inplace=True)
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
# 문자형 → 숫자형 변환 및 중앙값으로 결측치 대체
for col in feature_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    median = X_train[col].median()
    X_train[col].fillna(median, inplace=True)
    X_test[col].fillna(median, inplace=True)


# 모델 학습 및 예측
clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss")
clf.fit(X_train, y_train_cls)
y_pred_cls = clf.predict(X_test)

# 예측 결과 저장
df_test["예측_라벨_y_q"] = y_pred_cls
save_path = os.path.join(base_path, "0406/test_with_all_macro_sentiment_predicted_q.xlsx")
df_test.to_excel(save_path, index=False)

# 모델 평가
accuracy = accuracy_score(y_test_cls, y_pred_cls)
conf_matrix = confusion_matrix(y_test_cls, y_pred_cls)
report = classification_report(y_test_cls, y_pred_cls)

print(f"\n예측 결과 저장 완료: {save_path}")
print(f"\n예측 정확도 (분기 수익률 기준): {accuracy * 100:.2f}% (총 비교 개수: {len(y_test_cls)})")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)
