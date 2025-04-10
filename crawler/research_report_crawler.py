# 증권사 종목분석보고서 및 기업별 사업보고서 크롤러

#라이브러리 모음
import os  
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from collections import defaultdict

# 기업명 및 기업코드, 상장일(시작일) 리스트 생성
stock_info = {
    "콜마비앤에이치": ("200130", "2014-07-23"),
    "파마리서치": ("214450", "2015-07-24"),
    "펌텍코리아": ("251970", "2019-07-04"),
    "에이피알": ("281820", "2024-02-27"),
    "씨앤씨인터내셔널": ("352480", "2021-12-17"),
    "아이패밀리에스씨": ("114840", "2021-08-11"),
    "토니모리": ("214420", "2015-07-10"),
    "마녀공장": ("439090", "2023-06-08"),
    "코스메카코리아": ("241710", "2015-12-29"),
    "잇츠한불": ("226320", "2016-03-10"),
    "한국콜마": ("161890", "2011-12-21"),
    "코스맥스": ("192820", "2011-10-17")
}
company_names = list(stock_info.keys())

# 저장 폴더
base_download_folder = r"C:\Users\likel\Desktop\project_team4\리서치 보고서"
os.makedirs(base_download_folder, exist_ok=True)

# Selenium 드라이버 설정
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("user-agent=Mozilla/5.0")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# 페이지 범위 설정
start_page = 1
end_page = 1697
base_url = "https://finance.naver.com/research/company_list.naver"
download_counts = defaultdict(int)
print(f"총 {len(company_names)}개 종목 대상으로 리포트 수집 시작")

# 페이지 반복 설정
for current_page in range(start_page, end_page + 1):
    print(f"\n{current_page} 페이지 확인 중")
    search_url = f"{base_url}?&page={current_page}"
    driver.get(search_url)
    time.sleep(1.5)

    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.type_1 tbody tr")))
    except:
        print("페이지 로딩 실패, 종료")
        break

    soup = BeautifulSoup(driver.page_source, "html.parser")
    rows = soup.select("table.type_1 tbody tr")

    if not rows:
        print("더 이상 데이터 없음. 종료.")
        break

    for row in rows:
        columns = row.find_all("td")
        if len(columns) < 5:
            continue

        company_tag = columns[0].find("a")
        if not company_tag:
            continue
        
        # 기업 리스트에 포함된 기업일 경우에만 pdf 파일 다운로드
        company_name = company_tag.get("title", "").strip() or company_tag.text.strip()
        if company_name not in company_names:
            continue

        brokerage = columns[2].text.strip() or "Unknown"
        date_tag = row.find("td", class_="date")
        date_str = date_tag.text.strip() if date_tag else "unknown_date"

        pdf_td = row.find("td", class_="file")
        if not pdf_td:
            print(f"[{company_name}] PDF 없음 → 스킵")
            continue

        pdf_tag = pdf_td.find("a", href=True)
        if not pdf_tag:
            print(f"[{company_name}] PDF 링크 없음 → 스킵")
            continue

        pdf_link = pdf_tag["href"]

        # 절대 URL인지 확인 후 조합
        if not pdf_link.startswith("http"):
            pdf_link = f"https://finance.naver.com{pdf_link}"

        company_folder = os.path.join(base_download_folder, company_name)
        os.makedirs(company_folder, exist_ok=True)

        #'날짜_증권사 이름'으로 파일명 설정
        file_name = f"{date_str}_{brokerage}.pdf"
        file_path = os.path.join(company_folder, file_name)

        if os.path.exists(file_path):
            print(f"[{company_name}] 이미 존재: {file_name}")
            continue

        print(f"[{company_name}] 다운로드 시작: {file_name}")

        # pdf 다운로드
        try:
            response = requests.get(pdf_link, headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://finance.naver.com"
            }, stream=True)

            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    # chunk 설정으로 대용량 데이터 처리
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                download_counts[company_name] += 1
                print(f"[{company_name}] 저장 완료: {file_name}")
            else:
                print(f"[{company_name}] 상태코드 {response.status_code} → 다운로드 실패: {file_name}")
        except Exception as e:
            print(f"[{company_name}] 예외 발생: {e}")
            continue

driver.quit()

# 결과 요약
print("\n다운로드 완료 요약")
for company, count in download_counts.items():
    print(f"{company}: {count}개 리포트 저장됨")