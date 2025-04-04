# BeautifulSoup & Scrpay 활용 뉴스기사 크롤러(url 저장 파일 따로 생성)
# 라이브러리 모음
from selenium import webdriver
from bs4 import BeautifulSoup
import time
import csv
from datetime import datetime, timedelta
import os
import scrapy
import pandas as pd
import re



## 뉴스 url 수집
# 검색어 및 검색 기간 설정
search_query = "토니모리"

start_date = "2015.07.01"
duration = 7
min_duration = 1

# 오늘 날짜 계산 및 날짜 변환
today = datetime.now().strftime("%Y.%m.%d")

def add_days_to_date(date_str, days):
    date_obj = datetime.strptime(date_str, "%Y.%m.%d")
    new_date_obj = date_obj + timedelta(days=days)
    return new_date_obj.strftime("%Y.%m.%d")

# 저장 설정
save_folder = r"C:\Users\likel\Desktop\project_team4\네이버뉴스"
os.makedirs(save_folder, exist_ok=True) 
output_file = os.path.join(save_folder, f"news_tonimori.csv")
file_exists = os.path.exists(output_file)

# CSV 파일 열기
with open(output_file, "a", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["url", "date"])
    if not file_exists:
        writer.writeheader()

    ds = start_date 

    while True:
        try:
            # ds가 오늘 날짜를 초과하면 중단
            if datetime.strptime(ds, "%Y.%m.%d") > datetime.strptime(today, "%Y.%m.%d"):
                print(f"크롤링 완료 '{output_file}'에 저장됨.")
                break

            # 종료 날짜 계산
            de = add_days_to_date(ds, duration)

            # Selenium 설정
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

            driver = webdriver.Chrome(options=options)

            # 한국경제 기사만 검색 허용
            search_url = (
                f"https://search.naver.com/search.naver?where=news&query={search_query}&sm=tab_opt&sort=2&"
                f"photo=0&field=0&pd=3&ds={ds}&de={de}&docid=&related=0&mynews=1&office_type=1&"
                f"office_section_code=3&news_office_checked=015&nso=so%3Ar%2Cp%3Afrom{ds.replace('.', '')}to{de.replace('.', '')}"
            )

            # URL 출력하여 크롬에서 직접 테스트
            print("테스트 URL:", search_url)
            driver.get(search_url)
            time.sleep(3)

            # 무한 스크롤 처리
            SCROLL_PAUSE_TIME = 2
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(SCROLL_PAUSE_TIME)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            # BeautifulSoup으로 HTML 파싱
            soup = BeautifulSoup(driver.page_source, "html.parser")
            articles = soup.select(".news_wrap")

            # 기사 URL과 날짜 수집, 기사만 저장
            results = []
            for article in articles:
                try:
                    url_tag = article.select_one("a.info[href^='https://n.news.naver.com']")
                    if url_tag:
                        url = url_tag["href"]
                        date_tag = url_tag.find_previous_sibling("span", class_="info")
                        date = date_tag.text.strip() if date_tag else "날짜 없음"
                        results.append({"url": url, "date": date})
                except Exception as e:
                    print(f"Error parsing article: {e}")

            # CSV 파일 저장
            writer.writerows(results)
            print(f"{ds} ~ {de} 기간 크롤링 완료. 한국경제 기사 수집: {len(results)}개")
            driver.quit()

            # 시작일자 업데이트
            ds = add_days_to_date(de, 1)

        except Exception as e:
            print(f"오류: {e}")

            # 오류 발생시 기간 줄여 재시도
            duration -= 1
            if duration < min_duration:
                print("최소 기간에 도달하여 크롤링을 중단.")
                break

        finally:
            try:
                driver.quit()
            except:
                pass


## 뉴스기사 본문 수집
class NewsSpider(scrapy.Spider):
    name = "news"

    custom_settings = {
        'JOBDIR': 'crawls/news_spider_state',
        'DUPEFILTER_CLASS': 'scrapy.dupefilters.BaseDupeFilter',
    }

    # 기본 저장 경로 (기사별 월별 폴더 생성)
    base_save_path = r"C:/Users/likel/Desktop/project_team4/navernews/news_texts/브이티"

    # 기사 목록 저장된 csv 파일 불러오기기
    csv_file_path = r"C:\Users\likel\Desktop\project_team4\네이버뉴스\news_vtgmp_final.csv"

    def start_requests(self):
        """ CSV 파일에서 URL 목록을 불러와 크롤링 시작 """
        df = pd.read_csv(self.csv_file_path)
        urls = df['url'].dropna().tolist()

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        """ 기사 정보 크롤링 및 저장 """

        # 제목 크롤링
        title = response.css('h2#title_area span::text').get(default='N/A')

        # 날짜 크롤링
        date_raw = response.css('span.media_end_head_info_datestamp_time._ARTICLE_DATE_TIME::attr(data-date-time)').get(default="N/A")
        self.log(f"DEBUG: Extracted date_raw: {date_raw}")


        # 본문 크롤링
        content = ' '.join(response.css('div#contents.newsct_body::text').getall()).strip()
        if not content:
            content = ' '.join(response.css('article#dic_area::text').getall()).strip()

        # 날짜 형식 변환(YYYYMM)
        if date_raw and date_raw != 'N/A':
            clean_date = date_raw.split(' ')[0].replace("-", "")
            month_folder = date_raw[:7].replace("-", "")  # YYYYMM 형식 (월별 폴더 생성용)
        else:
            clean_date = 'unknown_date'
            month_folder = 'unknown_month'

        # 기사 ID 추출
        article_id_match = re.search(r'/(\d{10})\?', response.url)
        article_id = article_id_match.group(1) if article_id_match else "no_id"

        # 월별 폴더 생성
        save_path = os.path.join(self.base_save_path, month_folder)
        os.makedirs(save_path, exist_ok=True)

        # 파일명 만들기 (특수문자 제거 & 길이 제한)
        file_title = re.sub(r'[\\/*?:"<>|]', "_", title)[:30]  # 특수문자 제거 및 길이 제한
        file_name = f'news_{clean_date}_{file_title}_{article_id}.txt'
        file_path = os.path.join(save_path, file_name)

        # 파일 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f'제목: {title}\n')
            f.write(f'날짜: {clean_date}\n')
            f.write(f'본문:\n{content}\n')

        self.log(f'저장 완료: {file_path}')
