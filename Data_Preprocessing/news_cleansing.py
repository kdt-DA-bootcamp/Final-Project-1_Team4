# 뉴스 기사 1차 클렌징 코드

# 라이브러리 모음
import pandas as pd
import os
import re
import csv

# 컬럼 형식 및 이름 통일
column_names = ['날짜', '제목', '본문']
df = pd.read_csv(r'C:\Users\bona_\OneDrive\Desktop\FINAL PROJECT 1\코드\raw\토니모리_뉴스_통합본.csv', names=column_names, header=None)
df.to_csv(r'C:\Users\bona_\OneDrive\Desktop\FINAL PROJECT 1\코드\raw\토니모리_뉴스_통합본.csv', index=False, encoding='utf-8-sig')

# 본문 텍스트 클렌징
source_root = r"C:\Users\bona_\OneDrive\Desktop\FINAL PROJECT 1\코드\"
output_root = r"C:\Users\bona_\OneDrive\Desktop\FINAL PROJECT 1\코드\cleaned_news"

# 불용어 리스트(특수 문자 및 ~기자, ~특파원, 이메일 주소, 출처 및 저작권 관련 문장)
REMOVE_PATTERNS = [
    r"\[.*?\]", r"\(.*?\)", r"\{.*?\}", r"<.*?>",
    r"(\w+\s?기자|\w+\s?특파원)",
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
    r"[☞ⓒ※●★△□■◆▲◇]", r"·",
    r"세상을 보는 눈,?", r"ⓒ.*?(무단전재|배포금지).*?",
    r"\w+ 기자", r"저작권자 ⓒ.*?무단 전재-재배포 금지",
    r"성공을 꿈꾸는 사람들의 경제 뉴스", r"돈이 보이는 리얼타임 뉴스",
    r"창조기획팀", r"세종\s+서울신문", r"\s{2,}"
]

# 언론사 리스트
NEWS_AGENCIES = [
    "한국경제", "헤럴드경제", "머니S", "머니투데이", "서울신문", "아시아경제", "데일리안",
    "연합뉴스", "조선일보", "중앙일보", "동아일보", "경향신문", "한겨레", "매일경제", "서울경제",
    "파이낸셜뉴스", "이데일리", "뉴스1", "노컷뉴스", "YTN", "SBS", "MBC", "KBS", "JTBC", "MBN"
]

# 기사 본문 문장단위(./? 혹은 공백 기준)로 분리
def sentence_splitter(text):
    sentences = re.split(r'(?<=[\.\?])\s+', text)
    return "\n".join(sentences)

# 언론사명 제거
def remove_news_agencies(text):
    for agency in NEWS_AGENCIES:
        text = re.sub(rf"\b{agency}(신문|방송|뉴스)?\b", " ", text)
    return text.strip()

# 불용어 제거
def clean_text(text):
    for pattern in REMOVE_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = remove_news_agencies(text)
    text = sentence_splitter(text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n ", "\n")
    return text.strip()

# 본문 말미 저작권 문구 한 번 더 제거
def remove_trailing_copyrights(content):
    if not isinstance(content, str):
        return ""
    lines = content.strip().split("\n")
    for _ in range(3):
        if lines and re.search(r"(무단전재|배포금지|저작권)", lines[-1]):
            lines.pop()
        else:
            break
    return "\n".join(lines)


# 클렌징 된 텍스트 csv 파일로 저장
def process_csv_file(file_path, output_folder):
    file_name = os.path.basename(file_path)
    df = pd.read_csv(file_path, dtype=str)

    if "제목" not in df.columns or "본문" not in df.columns:
        print(f":경고: 필수 컬럼 누락: {file_name}")
        return

  
    title_col = "제목"
    content_col = "본문"

    if title_col not in df.columns or content_col not in df.columns:
        df.dropna(subset=column_names, inplace=True)
        return


    df = pd.read_csv(file_path, dtype=str, encoding="utf-8", quotechar='"') 
    df[title_col] = df[title_col].fillna("").apply(clean_text)
    df[content_col] = df[content_col].fillna("").apply(clean_text)
    df[content_col] = df[content_col].astype(str).apply(remove_trailing_copyrights)


    output_csv = os.path.join(output_folder, f"cleaned_{file_name}")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_NONNUMERIC)
    print(f"{file_name} 클렌징 완료. 저장 경로: {output_csv}")

csv_files = [os.path.join(source_root, f) for f in os.listdir(source_root) if f.endswith(".csv")]

os.makedirs(output_root, exist_ok=True)

for file_path in csv_files:
    process_csv_file(file_path, output_root)
