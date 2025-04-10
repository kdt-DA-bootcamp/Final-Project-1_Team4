# 기업별 뉴스기사 및 증권사 분석 토큰화 파일 병합
# 이후 감성분석 후 가중치 적용 위해 기업별 구분

import pandas as pd

# CSV 파일 불러오기(각각 뉴스기사 토큰화 파일과 증권리포트 토큰화 파일)
df1 = pd.read_csv("문장토큰화_생건.csv")
df2 = pd.read_csv("tokenized_LG생활건강.csv")
df2 = df2.drop(columns=['Company'])


# df1의 칼럼 순서와 이름을 기준으로 df2의 칼럼 이름 통일
df2.columns = df1.columns

# 파일 병합 후 확인
merged_df = pd.concat([df1, df2], ignore_index=True)
print(merged_df.head())

# 날짜 형식 통일
merged_df["날짜"] = pd.to_datetime(merged_df["날짜"], infer_datetime_format=True, errors='coerce')
merged_df["날짜"] = merged_df["날짜"].dt.strftime("%Y-%m-%d")

# 최종 병합 파일 저장
merged_df.to_csv("merged_LG생활건강.csv", index=False, encoding='utf-8-sig')