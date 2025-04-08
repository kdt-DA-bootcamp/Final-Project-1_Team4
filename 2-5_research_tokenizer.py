# 증권사 리포트 문장 토큰화 코드

# 라이브러리 모음
import pandas as pd
import re

# 기업명 입력
company_name = '펌텍코리아'
df = pd.read_csv(f'report_{company_name}.csv')

# 문장 토큰화 함수
def tokenize_sentences(text):
    if isinstance(text, str):
        # 모든 종류의 괄호와 내용 제거
        text = re.sub(r'[\[\{].*?[\]\}]', '', text)  
        text = re.sub(r'^[으로가에의를은는수]+(\s*)(\S+)', r'\2', text)
        # 숫자 사이의 마침표 임시 변경
        text = re.sub(r'([0-9０-９])(\.)([0-9０-９])', r'\1_\3', text)
        # ▶, ■, ◼를 마침표로 변환
        text = re.sub(r'[▶■◼]', '.', text)  
        # '그림', '1)', '2)', '3)', '-'을 마침표로 변환
        text = re.sub(r'(그림|\d\)|-|①|②|③)', '.', text) 
        # 숫자와 마침표를 제외한 구분자 
        sentences = re.split(r'(?<!\d)\.(?!\d)', text)  
        # 임시 변경된 마침표 복원
        sentences = [re.sub(r'_', '.', sentence.strip()) for sentence in sentences if sentence.strip()]
        return sentences
    return []

# 문장 토큰화 적용
df['Sentences'] = df['Text'].apply(tokenize_sentences)

# 빈 Text 칼럼 제거
df = df[df['Text'].str.strip() != '']

# 결과 저장
df.to_csv(f'tokenized_{company_name}.csv', index=False)