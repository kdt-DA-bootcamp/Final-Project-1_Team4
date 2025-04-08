# 일별 감성점수에 ETF 비중에 따른 기업별 가중치 부여

# 라이브러리 모음
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob

# 파일 불러오기
sentiment_folder = r"C:\Users\bona_\OneDrive\Desktop\FINAL PROJECT 1\감성점수" 
etf_path = r"C:\Users\bona_\OneDrive\Desktop\FINAL PROJECT 1\TIGER 2015-2025 일간.csv"

# 기업별 가중치 설정
weights = {
    "파마리서치": 0.131,
    "에이피알": 0.12,
    "코스맥스": 0.109,
    "한국콜마": 0.107,
    "LG생활건강": 0.101,
    "아모레퍼시픽": 0.086,
    "브이티": 0.079,
    "실리콘투": 0.075,
    "코스메카코리아": 0.039,
    "펌텍코리아": 0.029,
    "콜마비앤에이치": 0.024,
    "아이패밀리에스씨": 0.023,
    "마녀공장": 0.021,
    "씨앤씨인터내셔널": 0.018,
    "클리오": 0.016,
    "토니모리": 0.014,
    "잇츠한불": 0.007
}

# 기업별 가중치 적용 및 기업명 컬럼 추가
sentiment_dfs = []
for corp, weight in weights.items():
    path = f"{sentiment_folder}/sentiment_{corp}.csv"
    df = pd.read_csv(path)
    df["날짜"] = pd.to_datetime(df["문서발표일"])
    df["기업명"] = corp
    df["가중 감성 점수"] = df["감성 점수"] * weight
    sentiment_dfs.append(df[["날짜", "기업명", "가중 감성 점수"]])

sentiment_dfs[:10]
sentiment_all = sentiment_dfs.sort_values("날짜").reset_index(drop=True)

# 기간 설정
start_date = "2015-01-01"
end_date = "2025-03-23"

# 날짜 기준 필터링
filtered_sentiment = sentiment_all[
    (sentiment_all["날짜"] >= start_date) &
    (sentiment_all["날짜"] <= end_date)
].reset_index(drop=True)

filtered_sentiment[:100]

# 감성 점수 통합 (날짜+기업 기준 평균 점수)
filtered_sentiment = filtered_sentiment.groupby(["날짜", "기업명"])["가중 감성 점수"].mean().reset_index()
filtered_sentiment[:100]

# 날짜 기준 가중 평균으로 ETF 단일 시계열 변환
merged_weighted = filtered_sentiment.copy()
for col in filtered_sentiment.columns:
    if col not in ["날짜", "기업명"]:
        merged_weighted[col] = merged[col]

etf_grouped = merged_weighted.groupby("날짜").sum().reset_index()

# ETF 일간 종가 데이터 불러오기
etf_price = pd.read_csv(etf_path)
etf_price["날짜"] = pd.to_datetime(etf_price["날짜"].astype(str).str.strip())
etf_price["종가"] = etf_price["종가"].astype(str).str.replace(",", "").astype(float)

# 최종 병합 후 저장장
final_df = pd.merge(filtered_sentiment, etf_price[["날짜", "종가", "거래량"]], on="날짜", how="inner")
final_df = final_df.sort_values("날짜").reset_index(drop=True)
final_df.to_csv("merged_df.csv", index=False, encoding="utf-8-sig")
