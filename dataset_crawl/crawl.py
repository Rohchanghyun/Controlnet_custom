from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup
import os
import time
import requests
from PIL import Image
from io import BytesIO
import re
from selenium.webdriver.firefox.service import Service

class CrawlingStats:
    def __init__(self):
        self.total_attempts = 0
        self.successful_downloads = 0
        self.failed_downloads = 0
        self.corrupted_images = 0

def download_with_retry(url, max_retries=3, delay_between_retries=5):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Failed to download after {max_retries} attempts: {url}")
                print(f"Error: {e}")
                return None
            print(f"Attempt {attempt + 1} failed, retrying in {delay_between_retries} seconds...")
            time.sleep(delay_between_retries)

# Firefox 옵션 개선
firefox_options = Options()
firefox_options.add_argument('--headless=new')  # 최신 헤드리스 모드
firefox_options.add_argument('--disable-gpu')
firefox_options.add_argument('--no-sandbox')
firefox_options.add_argument('--disable-dev-shm-usage')
firefox_options.set_preference('dom.ipc.processCount', 1)

# 수정된 geckodriver 경로 (도커 환경에 맞게 변경)
service = Service(
    executable_path='/usr/bin/geckodriver',  # 도커에서 일반적인 설치 경로
    log_path='geckodriver.log'
)

# 드라이버 초기화 방식 변경
driver = webdriver.Firefox(
    service=service,
    options=firefox_options
)

# 암시적 대기 시간 조정
driver.implicitly_wait(15)  # 10초 → 15초로 변경

# 저장할 디렉토리 설정
save_dir = '/workspace/data/changhyun/dataset/emoji_crawl'
os.makedirs(save_dir, exist_ok=True)

# 페계 객체 생성
stats = CrawlingStats()

# 페이지 수 설정
num_pages = 100

# 각 페이지를 순회하며 이미지 크롤링
for page in range(23, num_pages + 1):
    try:
        url = f"https://www.freepik.com/search?format=search&last_filter=page&last_value={page}&page={page}&query=Character+Sticker&selection=1"
        driver.get(url)
        time.sleep(3)
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        images = soup.find_all('img', class_='$block $object-cover $object-center $w-full $h-auto $rounded')

        for img in images:
            img_url = img.get('src')
            if not img_url:
                continue
            
            stats.total_attempts += 1
            print(f"Downloading {img_url}")
            
            # 이미지 다운로드 시도
            img_response = download_with_retry(img_url)
            if img_response is None:
                stats.failed_downloads += 1
                continue
            
            # 파일 이름 처리
            img_name = os.path.basename(img_url)
            img_name = re.sub(r'\.jpg.*$', '.png', img_name)
            
            # 이미지 변환 및 저장
            try:
                image = Image.open(BytesIO(img_response.content))
                if image.mode != 'RGBA':
                    image = image.convert('RGBA')
                image.save(os.path.join(save_dir, img_name), 'PNG')
                stats.successful_downloads += 1
            except Exception as e:
                print(f"Corrupted image, skipping: {img_url}")
                stats.corrupted_images += 1
                continue

        print(f"Page {page} done.")
        time.sleep(1)

    except Exception as e:
        print(f"Error on page {page}: {e}")
        print("Waiting before retry...")
        time.sleep(30)
        continue

driver.quit()
print("\nCrawling Statistics:")
print(f"Total attempts: {stats.total_attempts}")
print(f"Successful downloads: {stats.successful_downloads}")
print(f"Failed downloads: {stats.failed_downloads}")
print(f"Corrupted images: {stats.corrupted_images}")
print(f"Success rate: {(stats.successful_downloads/stats.total_attempts)*100:.2f}%")
