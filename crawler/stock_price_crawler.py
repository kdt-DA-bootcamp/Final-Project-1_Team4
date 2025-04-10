# 기업별 주가 데이터 크롤러(종가 및 거래량)

# 라이브러리 모음
import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from datetime import datetime

# 종목명-종목코드-상장일 정보 리스트 생성
# 상장일은 아래 시작일보다 상장일이 이후일 경우 필터링 위함
stock_info = {
    "파마리서치": ("214450", "2015-07-24"),
    "실리콘투": ("257720", "2021-10-28"),
    "펌텍코리아": ("251970", "2019-07-04"),
    "에이피알": ("281820", "2024-02-27"),
    "씨앤씨인터내셔널": ("352480", "2021-12-17"),
    "아이패밀리에스씨": ("114840", "2021-08-11"),
    "토니모리": ("214420", "2015-07-10"),
    "코스메카코리아": ("241710", "2015-12-29"),
    "잇츠한불": ("226320", "2016-03-10")
}

# 시작일 설정
BASE_DATE = datetime.strptime("2015-01-01", "%Y-%m-%d")

# 저장 폴더 경로
base_folder = r"C:\Users\likel\Desktop\project_team4\주가정보"
os.makedirs(base_folder, exist_ok=True)

# Selenium 드라이브 설정
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("user-agent=Mozilla/5.0")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# 최대 페이지 수 설정
MAX_PAGES = 150

# 종목별 크롤링(페이지 반복 설정)
for name, (code, ipo_str) in stock_info.items():
    print(f"\n{name} ({code}) 시작")

    ipo_date = datetime.strptime(ipo_str, "%Y-%m-%d")
    start_date = max(BASE_DATE, ipo_date)

    data = []
    page = 1

    while page <= MAX_PAGES:
        url = f"https://finance.naver.com/item/frgn.naver?code={code}&page={page}"
        print(f"{name} - {page}페이지 수집 중...")
        driver.get(url)

        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.type2 tbody tr"))
            )
        except:
            print("로딩 실패")
            break
        
        time.sleep(2.5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        rows = soup.select("table.type2 tbody tr")

        if all(len(row.find_all("td")) < 5 for row in rows):
            print("유효한 데이터 없는 페이지 → 종료")
            break

        page_data = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 5:
                continue

            try:
                date = cols[0].text.strip()
                close_price = cols[1].text.strip().replace(",", "")
                volume = cols[4].text.strip().replace(",", "")
                date_obj = datetime.strptime(date, "%Y.%m.%d")

                # 날짜 필터링 적용(상장일 이전 데이터는 저장하지 않음)
                if date_obj < ipo_date:
                    print(f"{name}: 상장일 이전 {date}, 저장 안 함")
                    continue

                print(f"{date} | 종가: {close_price} | 거래량: {volume}")
                page_data.append([name, date, close_price, volume])
            except:
                continue

        if not page_data:
            break

        data.extend(page_data)
        page += 1

    # 기업명, 날짜, 종가, 거래량에 대해 기업명으로 파일명 설정 후 csv 파일로 저장
    if data:
        df = pd.DataFrame(data, columns=["종목명", "날짜", "종가", "거래량"])
        file_path = os.path.join(base_folder, f"{name}_주가데이터.csv")
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"저장 완료: {file_path} ({len(df)}건)")
    else:
        print(f"유효 데이터 없음: {name}")

# 브라우저 종료
driver.quit()
print("\n 전체 종목 크롤링 완료")