from selenium import webdriver
from bs4 import BeautifulSoup
import time
import csv
from datetime import datetime, timedelta
import os

search_query = "토니모리"

start_date = "2015.07.01"
duration = 7
min_duration = 1

today = datetime.now().strftime("%Y.%m.%d")

def add_days_to_date(date_str, days):
    date_obj = datetime.strptime(date_str, "%Y.%m.%d")
    new_date_obj = date_obj + timedelta(days=days)
    return new_date_obj.strftime("%Y.%m.%d")

save_folder = r"C:\Users\likel\Desktop\project_team4\네이버뉴스"
os.makedirs(save_folder, exist_ok=True) 
output_file = os.path.join(save_folder, f"news_tonimori.csv")
file_exists = os.path.exists(output_file)

with open(output_file, "a", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["url", "date"])
    if not file_exists:
        writer.writeheader()

    ds = start_date 

    while True:
        try:
            if datetime.strptime(ds, "%Y.%m.%d") > datetime.strptime(today, "%Y.%m.%d"):
                print(f"크롤링 완료 '{output_file}'에 저장됨.")
                break

            de = add_days_to_date(ds, duration)

            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

            driver = webdriver.Chrome(options=options)

            search_url = (
                f"https://search.naver.com/search.naver?where=news&query={search_query}&sm=tab_opt&sort=2&"
                f"photo=0&field=0&pd=3&ds={ds}&de={de}&docid=&related=0&mynews=1&office_type=1&"
                f"office_section_code=3&news_office_checked=015&nso=so%3Ar%2Cp%3Afrom{ds.replace('.', '')}to{de.replace('.', '')}"
            )

            print("테스트 URL:", search_url)
            driver.get(search_url)
            time.sleep(3)

            SCROLL_PAUSE_TIME = 2
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(SCROLL_PAUSE_TIME)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            soup = BeautifulSoup(driver.page_source, "html.parser")
            articles = soup.select(".news_wrap")

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

            writer.writerows(results)
            print(f"{ds} ~ {de} 기간 크롤링 완료. 한국경제 기사 수집: {len(results)}개")
            driver.quit()

            ds = add_days_to_date(de, 1)

        except Exception as e:
            print(f"오류: {e}")

            duration -= 1
            if duration < min_duration:
                print("최소 기간에 도달하여 크롤링을 중단.")
                break

        finally:
            try:
                driver.quit()
            except:
                pass

