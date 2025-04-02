import random
import requests
from bs4 import BeautifulSoup
import time
import csv
from datetime import datetime, timedelta
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://search.naver.com/",
    "Accept-Language": "ko-KR,ko;q=0.9"
}

def get_random_headers():
    headers = {"User-Agent": "Mozilla/5.0"}
    options = [
        {"Referer": "https://search.naver.com/", "Accept-Language": "ko-KR,ko;q=0.9"},
        {"Referer": "https://search.naver.com/"},
        {"Accept-Language": "ko-KR,ko;q=0.9"},
        {}
    ]
    headers.update(random.choice(options))
    return headers

def robust_get(url, max_retries=5, timeout=10):
    global headers
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else None
            if status == 403 or (status is None and "403" in str(e)):
                headers = get_random_headers()
                wait_time = random.uniform(10, 30)
                print(f"HTTP 403 에러 발생. 헤더 변경 후 {wait_time:.0f}초 대기.")
                time.sleep(wait_time)
            else:
                wait_time = 2 ** retries
                print(f"Error fetching {url}: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            retries += 1
    raise Exception(f"Failed to fetch {url} after {max_retries} retries.")

def get_article_content(url):
    res = robust_get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    
    title_tag = soup.select_one("h2#title_area.media_end_head_headline")
    title = title_tag.get_text(strip=True) if title_tag else "제목 없음"
    
    content_tag = soup.select_one("article#dic_area.go_trans._article_content")
    content = content_tag.get_text(strip=True) if content_tag else "본문 없음"
    
    return title, content

def load_existing_articles(csv_filename):
    existing_links = set()
    if os.path.exists(csv_filename):
        with open(csv_filename, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    existing_links.add(row[1])
    return existing_links

def crawl_date(date_str, keyword, existing_links):
    results = []
    page_no = 1
    date_nso = date_str.replace(".", "")
    
    while True:
        search_url = (
            f"https://search.naver.com/search.naver?where=news&query={keyword}&sm=tab_opt&sort=1&photo=0&field=0&pd=3"
            f"&ds={date_str}&de={date_str}&nso=so%3Add%2Cp%3Afrom{date_nso}to{date_nso}&start={page_no}"
        )
        
        print(f"[{date_str}] 요청 중: start={page_no}")
        try:
            res = robust_get(search_url)
        except Exception as e:
            print(f"[{date_str}] 검색 URL 요청 실패: {e}")
            break
        
        soup = BeautifulSoup(res.text, "html.parser")
        news_items = soup.select("div.news_area")
        if not news_items:
            print(f"[{date_str}] 더 이상 뉴스가 없거나, 페이지 구조 변경으로 종료.")
            break
        
        valid_links = []
        for item in news_items:
            info_links = item.select("div.info_group a.info")
            for a_tag in info_links:
                href = a_tag.get("href", "")
                if "https://n.news.naver.com/mnews/article/" in href and href not in existing_links:
                    existing_links.add(href)
                    valid_links.append(href)
        
        if valid_links:
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_link = {executor.submit(get_article_content, link): link for link in valid_links}
                for future in as_completed(future_to_link):
                    link = future_to_link[future]
                    try:
                        title, content = future.result()
                        results.append((date_str, link, title, content))
                    except Exception as e:
                        print(f"오류 발생 ({link}): {e}")
        
        if page_no >= 1500:
            print(f"[{date_str}] 최대 페이지 수(1500) 도달. 종료.")
            break
        
        page_no += 10
        time.sleep(random.uniform(0.5, 1.5))
    
    return results

def main():
    start_date_str = input("시작 날짜 (YYYY-MM-DD): ").strip()
    end_date_str = input("종료 날짜 (YYYY-MM-DD): ").strip()
    keyword = input("검색할 키워드를 입력하세요: ").strip()
    
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    csv_filename = f"news_{keyword}.csv"
    
    existing_links = load_existing_articles(csv_filename)
    current_date = end_date
    while current_date >= start_date:
        date_str = current_date.strftime("%Y.%m.%d")
        print(f"\n====== {date_str} 크롤링 시작 ({keyword}) ======")
        results = crawl_date(date_str, keyword, existing_links)
        
        with open(csv_filename, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            for row in results:
                writer.writerow(row)
        
        current_date -= timedelta(days=1)
    
    print(f"\n=== CSV 파일 저장 완료: {csv_filename} ===")

if __name__ == "__main__":
    main()