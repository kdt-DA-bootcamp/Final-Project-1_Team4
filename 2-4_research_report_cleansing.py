# 증권사 보고서 전처리 코드(colab 사용)
 
# 라이브러리 모음
!pip install kss pandas
!pip install git+https://github.com/haven-jeon/PyKoSpacing.git
import pandas as pd
import re
import kss
import os
import json
import glob

# 띄어쓰기 교정 모델 설치 및 테스트
# (pdf 파일 텍스트 변환 과정에서 발생한 띄어쓰기 오류 수정)
from pykospacing import Spacing
spacing = Spacing()
print(spacing("이문장은띄어쓰기가필요해보입니다"))

# 구글 드라이브 연동 및 폴더 경로 설정
from google.colab import drive
drive.mount('/content/drive')
folder_path = "/content/drive/MyDrive/화장품/증권리포트/1_txt"
files = glob.glob(os.path.join(folder_path, "*.csv"))

# 불용문 리스트
stop_sentences = [
    "본 조사분석자료에 게재된 내용들이 본인의 의견을 정확하게 반영하고 있으며",
    "외부의 부당한 압력이나 간섭없이 신의성실하게 작성되었음을 확인합니다",
    "담당종목을 보유하고 있지 않습니다",
    "기관투자가 또는 제 3 자에게 사전 제공된 사실이 없습니다",
    "특별한 이해 관계가 없습니다",
    "종목별 투자의견은 다음과 같습니다",
    "당사는 추천일 현재 해당회사와 관련하여 특별한 이해관계가 없습니다",
    "투자의견은 향후 개월간 당사 목표 대비 초과 상승률 기준임",
    "최근 년간 누적 기준",
    "투자자 자신의 판단과 책임하에 최종결정을 하시기 바랍니다",
    "추천기준일 종가대비 이상",
    "추천기준일 종가대비 이상 미만",
    "추천기준일 종가대비 이상 미만",
    "당사는 상기 명시한 사항 외 고지해야 하는 특별한 이해관계가 없습니다"
]


# 리포트 문장 전처리 함수 정의
def preprocess_text_kss_based(text):
    if pd.isna(text):
        return []

    # 전체 줄바꿈 제거 후 문장 단위 분리
    text = text.replace('\n', ' ')
    raw_sentences = kss.split_sentences(text)

    # 중요 키워드 설정(문장이 짧아도 해당 키워드 포함시 생략하지 않음)
    important_keywords = ['매출', '성장', '개선', '감소', '호조', '부진', '유지', '하향', '상향', '실적', '이익','동 자료', '이해관계', '최종결정', '책임소재', '당사', '배우자', '투자의견', '조사자료', '본자료', '자료:', '비고', '주:',
                    '본 보고서', '본 자료', '도표', '|', '투자정보 확충', '보고서 발간 소식', '투자자문업', '신의성실', '이 자료', '증빙자료',
                    '분석자료', '투자행위', '예상되는 경우에', '예상되는 종목에', '기관투자자', '부당한 압력', '정확성이나 완전성',
                    '실제 펀드 편입 여부', '보유하고 있지 않습니다', '해당 정보가', '법으로 금지되어', '당 자료는',
                    '연계되어 있지 않습니다', '보유하고 있지 않습니다', '제시하고 있습니다', '의견을 제시합니다', '부당한 압박']

    # 불용어 리스트(해당 키워드 포함 문장은 전부 제거)
    remove_keywords = [
        "목표주가", "주가", "시가총액", "조", "억원", "원", "화장품의류", "조경진", "주가동향",
        "최고가", "최저가", "최고최저가", "등락률", "수익률", "절대", "상대", "발행주식수", "천주",
        "일평균 거래량", "외국인 지분율", "배당수익률", "주요 주주", "외인", "투자지표", "십억원",
        "연결", "매출액", "영업이익", "세전이익", "순이익", "지배주주지분순이익", "증감률",
        "배", "영업이익률", "순차입금비율", "이베스트투자증권", "리서치센터"
    ]

    # 정규표현식 기반 불용문 패턴
    stop_patterns = [
        r"본 조사분석자료.*신의성실하게 작성되었음을 확인합니다",
        r"외부의 부당한 압력.*확인합니다",
        r"담당종목을 보유하고 있지 않습니다",
        r"기관투자가.*사전 제공된 사실이 없습니다",
        r"특별한 이해 관계가 없습니다",
        r"종목별 투자의견은.*",
        r"당사는 추천일.*이해관계가 없습니다",
        r"투자의견은 향후.*기준임",
        r"최근\s*\d*\s*년간 누적 기준",
        r"투자자 자신의 판단과 책임하에.*",
        r"추천기준일 종가대비.*",
        r"당사는 상기 명시한 사항.*이해관계가 없습니다"
    ]

    # 문장 전처리
    clean_sentences = []
    for sent in raw_sentences:
        
        # 한글이 아닌 문자 사이 공백만 제거(한글 사이 공백은 처리하지 않음)
        def remove_spaces(sent):
            if isinstance(sent, str):
                def replace_match(match):
                    before, space, after = match.groups()

                    if re.match(r'^[가-힣]+$', before) and re.match(r'^[가-힣]+$', after):
                        return match.group(0)
                    return before + after
                return re.sub(r'([^가-힣])(\s+)([^가-힣])', replace_match, sent)
            return sent
        
        # 정규 표현식 불용문 제거
        if any(re.search(pattern, sent) for pattern in stop_patterns):
            continue

        # 불용문 제거
        if any(stop in sent for stop in stop_sentences):
            continue

        # 불용어 포함 문장 제거
        if any(kw in sent for kw in remove_keywords):
            continue

        
        # 짧은 문장 및 내용이 없는 문장 제거(그림, 표 등)
        if len(sent) < 10:
            continue
        if '자료' in sent or '그림' in sent or '페이지' in sent:
            continue
        eng_count = len(re.findall(r'[a-zA-Z]', sent))
        if eng_count / len(sent) >= 0.2:
            continue
        if re.match(r'^[가-힣\s/]+$', sent):
            continue

        # 정규표현식을 활용한 추가 클렌징
        sent = re.sub(r'\d+(\.\d+)?%?', '', sent)
        sent = re.sub(r'\d{4}[년\./-]\d{1,2}([월\./-]\d{1,2}[일]?)?', '', sent)
        sent = re.sub(r'\d{2}[\.]\d{1,2}', '', sent)
        sent = re.sub(r'[1-4]Q\d{2}', '', sent)
        sent = re.sub(r'[1-4]분기', '', sent)
        sent = re.sub(r'\d+년', '', sent)
        sent = re.sub(r'\d+월', '', sent)
        sent = re.sub(r'\d+일', '', sent)
        sent = re.sub(r'\d+원', '', sent)
        sent = re.sub(r'[가-힣]+ *\d+.*?원', '', sent)
        sent = re.sub(r'[가-힣]+ *\d+.*?억원', '', sent)
        sent = re.sub(r'\S+@\S+', '', sent)
        sent = re.sub(r'-{2,}', '', sent)
        sent = re.sub(r'[▶■▬]', '', sent)
        sent = re.sub(r'[a-zA-Z]', '', sent)
        sent = re.sub(r'[^\w\s가-힣]', '', sent)
        sent = re.sub(r'\s+', ' ', sent).strip()

        # 짧은 문장 제거(주요 키워드 포함 시 유지)
        if len(sent) <= 6 and not any(kw in sent for kw in important_keywords):
            continue

        # 띄어쓰기 교정
        sent = spacing(sent)

        if sent:
            clean_sentences.append(sent)

    return clean_sentences


# 폴더 내 모든 파일 처리 반복 설정
for file in files:
    df = pd.read_csv(file)
    print(f"파일 {file} 로딩 완료")

    # 비어 있는 행 제거
    df = df[df['Text'].notna() & (df['Text'].str.strip() != '')]

    # 전처리 함수 적용
    df['Cleaned_Sentences'] = df['Text'].apply(preprocess_text_kss_based)

    # 리스트 형태로 저장
    df['Cleaned_Sentences_List'] = df['Cleaned_Sentences'].apply(lambda x: str(x))

    # 저장 설정 (각각의 파일에 맞는 이름으로 저장)
    output_dir = "/content/drive/MyDrive/화장품/증권리포트/2_results"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"cleaned_{os.path.basename(file)}")
    df[['Date', 'Company', 'Cleaned_Sentences_List']].to_csv(csv_path, index=False)
    print(f"저장 완료: {csv_path}")



# 낱개로 처리한 파일 통합
# 문장 토큰화된 파일 경로
files = glob.glob("/content/drive/MyDrive/화장품/증권리포트/3_전체파일/*.csv")
merged_all = []

for file in files:
    df = pd.read_csv(file)

    # 컬럼명 통일
    if 'Sentences' in df.columns:
        df['Sentences'] = df['Sentences']
    elif 'Cleaned_Sentences_List' in df.columns:
        df['Sentences'] = df['Cleaned_Sentences_List']
    else:
        continue

    # 날짜 형식 통일
    df['Date'] = pd.to_datetime(df['Date'], format='%y.%m.%d', errors='coerce')
    df = df.dropna(subset=['Date'])

    # 기업명 추출
    company = os.path.basename(file).replace("tokenized_", "").replace(".csv", "")
    df['Company'] = company

    # 필요한 컬럼 선택(기업명, 날짜, 전처리된 문장)
    df = df[['Company', 'Date', 'Sentences']]
    merged_all.append(df)

# 전체 병합 및 저장
merged_df = pd.concat(merged_all, ignore_index=True)
merged_df = merged_df.sort_values(['Company', 'Date'])
merged_df.to_csv("통합_문장토큰화_리포트.csv", index=False)

print(merged_df.shape)
print(merged_df.head())
