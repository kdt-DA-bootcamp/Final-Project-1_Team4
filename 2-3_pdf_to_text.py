# 라이브러리 모음
import os
import pandas as pd
import fitz  # PyMuPDF 사용 위함
import re
from tqdm import tqdm

# 폴더 경로 및 저장될 파일명 설정
pdf_folder = "아모레퍼시픽"
csv_file = "report_아모레퍼시픽.csv"

# PDF 파일 목록
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

# 간단한 전처리(특수문자 및 null, 공백 제거)
def clean_text(text):
    text = re.sub(r'\(cid:\d+\)', '', text)
    text = text.replace("\x00", "")
    return text.strip()

# 파일명에서 날짜 및 증권사 추출
def extract_date_company(filename):
    #날짜_기업명_증권사 형식일 경우우
    match = re.search(r'(\d{2}\.\d{2}\.\d{2})_.*?_(.+?)(?:증권)?\.pdf$', filename)

    # 날짜_증권사 형식일 경우
    # match = re.search(r'(\d{2}\.\d{2}\.\d{2})_(.+?)(증권)?\.pdf$', filename)

    if match:
        date = match.group(1)
        company = match.group(2).strip()
        return date, company
    return "Unknown", "Unknown"

# pdf to text 변환 함수 선언
def process_pdf(pdf_file):
    pdf_path = os.path.join(pdf_folder, pdf_file)
    # 기존 pdf 파일에 txt 추가로 저장
    txt_path = pdf_path.replace(".pdf", ".txt")

    try:
        # PyMuPDF 사용, pdf 한 장씩 불러와서 한 줄로 결합
        with fitz.open(pdf_path) as doc:
            text = "\n".join([page.get_text("text") for page in doc])

        text = clean_text(text)  # 정리된 텍스트 적용

        if not text:
            print(f"경고: {pdf_file}에서 텍스트를 추출 불가. 건너뜀.")
            return None

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)  # TXT 파일 저장

        # 파일명에서 날짜와 증권사 추출
        date, company = extract_date_company(pdf_file)

        return [date, company, text]

    except Exception as e:
        print(f"오류 발생: {pdf_file} 변환 중 문제 발생 ({e}). 건너뜁니다.")
        return None

# PDF 변환 실행
data = []
for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
    result = process_pdf(pdf_file)
    if result is not None:
        data.append(result)

# CSV 저장
if data:
    df = pd.DataFrame(data, columns=["Date", "Company", "Text"])
    df.to_csv(csv_file, index=False, encoding="utf-8-sig", errors="replace")
    print(f"CSV 파일 저장 완료: {csv_file}")
else:
    print("처리된 데이터 없음. CSV 파일을 생성하지 않음.")