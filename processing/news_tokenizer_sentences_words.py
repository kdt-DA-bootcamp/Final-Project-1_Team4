# 뉴스 기사 토큰화 코드(colab 사용)
# 데이터 크기를 줄이기 위해 토큰화된 파일에는 원문 텍스트를 포함하지 않음음

# Mecab 설치 및 설정
# Mecab 설치 GitHub 활용
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
cd Mecab-ko-for-Google-Colab
!bash install_mecab-ko_on_colab_light_220429.sh
# 수동 설치할 경우
# !apt-get update
# !apt-get install -y mecab libmecab-dev mecab-ipadic-utf8 git make

# mecab-ko-dic 설치 (기존 폴더가 있으면 삭제 후 재설치)
!rm -rf mecab-ko-dic
!git clone https://bitbucket.org/eunjeon/mecab-ko-dic.git
!cd mecab-ko-dic && ./autogen.sh && ./configure --with-charset=utf8 && make && sudo make install
# 사전의 위치 확인
!find / -type d -name mecab-ko-dic 2>/dev/null

# python과 konlpy 연동
!pip install mecab-python3
!pip install konlpy

# MECABRC 환경 변수 설정
import os
!sudo mkdir -p /usr/local/etc
!sudo bash -c 'echo "dicdir = /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ko-dic" > /usr/local/etc/mecabrc'
os.environ["MECABRC"] = "/usr/local/etc/mecabrc"

# MeCab 테스트
from konlpy.tag import Mecab
mecab = Mecab(dicpath='/usr/local/lib/mecab/dic/mecab-ko-dic')
print(mecab.morphs("Colab에서 MeCab 테스트 중"))



# 문장 토큰화 코드
# 라이브러리 모음
!pip install kss
import pandas as pd
import kss

# 뉴스기사 CSV 파일 불러오기
df = pd.read_csv("/content/cleaned_클리오_뉴스_통합본.csv")

# 문장 토큰화 함수
def split_sentences(text):
    if pd.isna(text):
        return []
    return kss.split_sentences(text)

# 문장리스트 생성
df['문장리스트'] = df['본문'].apply(split_sentences)
# 원문 식별용 ID 부여(이후 최종 파일에는 포함하지 않음)
df['원문_ID'] = df.index  

# '날짜', '문장리스트' 컬럼만 선택
df_filtered = df[['날짜', '문장리스트']]

# 새 파일로 저장 및 로컬에 다운로드
df_filtered.to_csv("문장토큰화_클리오.csv", index=False, encoding='utf-8-sig')
from google.colab import files
files.download("문장토큰화_클리오.csv")



# 단어 토큰화 코드
# 라이브러리 모음
import pandas as pd
from konlpy.tag import Okt

# 뉴스기사 CSV 파일 불러오기
df = pd.read_csv("/content/cleaned_아모레_뉴스_통합본.csv")

# 단어 토큰화 함수 선언(명사, 동사, 형용사, 부사만 추출)
okt = Okt()
def extract_tokens(text):
    if pd.isna(text):
        return []
    pos_tags = okt.pos(text, stem=True)
    filtered = [word for word, tag in pos_tags if tag in ['Noun', 'Verb', 'Adjective', 'Adverb']]
    return filtered

# 단어 토큰 리스트 컬럼 생성
df['단어리스트'] = df['본문'].apply(extract_tokens)
# 원문 식별용 ID 부여(이후 최종 파일에는 포함하지 않음)
df['원문_ID'] = df.index

# 필요한 컬럼만 선택('날짜', '단어리스트')
df_filtered = df[['날짜', '단어리스트']]

# 새 파일로 저장 및 다운로드
df_filtered.to_csv("단어토큰화_아모레.csv", index=False, encoding='utf-8-sig')
from google.colab import files
files.download("단어토큰화_아모레.csv")