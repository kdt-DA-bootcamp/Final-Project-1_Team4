# 뉴스 기사 및 증권사 리포트 텍스트 감성분석(colab 사용)
# 주가 변동에 따른 라벨링 코드도 함께 작성했으나 사용하지 않게 되어 주석처리함 

# 라이브러리 모음
!pip install kobert-transformers
!pip install transformers
!pip install sentencepiece
!pip install pandas
!pip install numpy

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


# 데이터 로드드 및 날짜 형식 통일
news_df = pd.read_csv(r"C:\Users\bona_\OneDrive\Desktop\FINAL PROJECT 1\문장토큰화 병합\코스맥스.csv")
news_df["날짜"] = pd.to_datetime(news_df["날짜"])
# 주가 데이터 주석처리
# stock_df = pd.read_csv(r"C:\Users\bona_\Downloads\LG생활건강_주가데이터.csv")
# stock_df["날짜"] = pd.to_datetime(stock_df["날짜"])


# kobert 모델 활용
# 이미 문장 토큰화 이루어진 파일이지만 다시 한 번 적용
MODEL_NAME = "monologg/kobert"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)
model.eval()


# 감성분석 함수 
def get_sentiment_score(sentence):

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = F.softmax(logits, dim=1)
    probs = probs[0].cpu().numpy()

    sentiment_score = probs[2] - probs[0]
    return sentiment_score



# 데이터 내 문장별 감성 점수 계산 & 일자별 평균 계산 / 진행률 표시 추가
tqdm.pandas()
start_time = time.time()

news_df["감성점수"] = news_df["문장리스트"].progress_apply(get_sentiment_score)

end_time = time.time()

elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)
print(f"\n 감성 점수 계산 완료. 소요 시간: {int(minutes)}분 {int(seconds)}초")


# 날짜별 평균 감성점수 계산
sentiment = news_df.groupby(["날짜"])["감성점수"].mean().reset_index()
sentiment.rename(columns={"감성점수":"평균감성점수"}, inplace=True)

print("\n 문서 발표일별 평균 감성 점수:\n", sentiment)


## 문서 발표일 기준 -2~+2일 동안의 주가 변동률 계산
## 이후 과정에서 사용하지 않게 되어 우선 주석처리

# def get_return_in_range(row, stock_df, day_range=2):

#     base_date = row["날짜"]

#     start_date = base_date - pd.Timedelta(days=day_range)
#     end_date = base_date + pd.Timedelta(days=day_range)

#     mask = (
#         (stock_df["날짜"] >= start_date) &
#         (stock_df["날짜"] <= end_date)
#     )
#     df_range = stock_df.loc[mask].sort_values("날짜")

#     if df_range.empty:
#         return np.nan

#     first_close = df_range.iloc[0]["종가"]
#     last_close = df_range.iloc[-1]["종가"]
#     rate_of_change = (last_close - first_close) / first_close * 100.0

#     return rate_of_change

# news_df["-2~+2 기간 주가 변동률"] = news_df.apply(
#     lambda row: get_return_in_range(row, stock_df, day_range=2), axis=1
# )



## 라벨링
## 이후 과정에서 사용하지 않게 되어 우선 주석처리

# def assign_label(change):
#     if pd.isna(change):
#         return np.nan
#     if change > 0:
#         return 1
#     elif change == 0:
#         return 0
#     else:
#         return -1

# news_df["라벨"] = news_df["-2~+2 기간 주가 변동률"].apply(assign_label)



# 최종 결과 정리
final_df = sentiment
# 필요한 컬럼만 선택
final_df.rename(columns={
    "날짜": "문서발표일",
    "평균감성점수": "감성 점수",
    # "-2~+2 기간 주가 변동률": "-2~+2 기간 주가 변동률",
    # "라벨": "라벨"
}, inplace=True)

final_df = final_df[["문서발표일", "감성 점수"]]
print(final_df.head())
# 파일 구성 형식 확인 후 최종 파일 저장
final_df.to_csv("sentiment_코스맥스.csv", index=False, encoding='utf-8-sig')


# # 감성점수 분포 시각화
# news_df["감성점수"].hist(bins=50)
# plt.title("전체 감성점수 분포")
# plt.xlabel("감성점수")
# plt.ylabel("문서 수")
# plt.grid(True)
# plt.show()