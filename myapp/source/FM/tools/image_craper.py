# player_image_viewer.py
import requests
from PIL import Image
from io import BytesIO
from typing import Optional, List
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import chromedriver_autoinstaller
import time
import random
import logging
import threading
from queue import Queue
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from django.core.cache import cache

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WebDriver 풀 관리
class WebDriverPool:
    def __init__(self, max_size=3):
        self.max_size = max_size
        self.pool = Queue(maxsize=max_size)
        self.lock = threading.Lock()
    
    def get_driver(self):
        try:
            return self.pool.get_nowait()
        except:
            return self._create_driver()
    
    def return_driver(self, driver):
        if not self.pool.full():
            try:
                self.pool.put_nowait(driver)
            except:
                driver.quit()
        else:
            driver.quit()
    
    def _create_driver(self):
        chromedriver_autoinstaller.install()
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        options.add_argument(f"user-agent={random.choice(user_agents)}")
        
        driver = webdriver.Chrome(options=options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    
    def cleanup(self):
        while not self.pool.empty():
            try:
                driver = self.pool.get_nowait()
                driver.quit()
            except:
                break

# 전역 WebDriver 풀 인스턴스
driver_pool = WebDriverPool()

def get_player_image_from_bing(name: str) -> Optional[str]:
    """
    Bing에서 축구 선수 이미지를 검색하여 URL을 반환합니다.
    캐싱을 통해 성능을 개선했습니다.
    
    Args:
        name (str): 검색할 선수 이름
        
    Returns:
        Optional[str]: 이미지 URL 또는 None (실패시)
    """
    # 캐시 확인
    cache_key = f"player_image_{name.lower().replace(' ', '_')}"
    cached_url = cache.get(cache_key)
    if cached_url:
        logger.info(f"🎯 {name}: 캐시에서 이미지 URL 반환")
        return cached_url
    
    driver = driver_pool.get_driver()
    
    try:
            
        # 검색 쿼리 다양화 (한국어 선수명 + 영어 키워드)
        search_terms = [
            f"football player {name}",
            f"soccer player {name}",
            f"{name} football",
            f"{name} soccer player profile",
            f"축구선수 {name}"
        ]
        
        query = random.choice(search_terms).replace(" ", "+")
        
        # 검색 페이지 랜덤화
        first_param = random.choice([1, 21, 41])  # 다른 페이지에서 시작
        url = f"https://www.bing.com/images/search?q={query}&form=HDRSC2&first={first_param}"
        
        logger.info(f"🔍 {name} 검색 URL: {url}")
        driver.get(url)
        
        # 최소한의 대기시간으로 최적화
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "img"))
        )
            
        # 다양한 이미지 선택자 시도
        image_urls = _extract_multiple_image_urls(driver)
        
        if not image_urls:
            logger.warning(f"❌ {name}: 이미지 URL을 찾을 수 없음")
            return None
        
        # 유효한 URL 찾기 (빠른 검증)
        for url in image_urls:
            if _is_valid_image_url_format(url):
                # 캐시에 저장 (24시간)
                cache.set(cache_key, url, 60 * 60 * 24)
                logger.info(f"✅ {name}: 선택된 이미지 URL - {url}")
                return url
        
        logger.warning(f"❌ {name}: 유효한 이미지 URL을 찾을 수 없음")
        return None

    except Exception as e:
        logger.error(f"❌ {name} 이미지 검색 중 오류: {e}")
        return None
    finally:
        driver_pool.return_driver(driver)

def _extract_multiple_image_urls(driver) -> List[str]:
    """
    여러 선택자를 사용하여 이미지 URL들을 추출합니다.
    
    Returns:
        List[str]: 찾은 이미지 URL 리스트
    """
    image_urls = []
    
    # 다양한 CSS 선택자 시도
    selectors = [
        "img.mimg",
        "img[class*='img']",
        ".imgpt img", 
        ".richImageCard img",
        ".iusc img",
        "img[src*='bing.com']",
        "img[data-src*='bing.com']"
    ]
    
    for selector in selectors:
        try:
            img_elements = driver.find_elements(By.CSS_SELECTOR, selector)
            
            for img_element in img_elements[:10]:  # 상위 10개만 확인
                # src 또는 data-src 속성에서 URL 추출
                img_url = img_element.get_attribute("src") or img_element.get_attribute("data-src")
                
                if img_url and img_url.startswith("http") and _is_valid_image_url_format(img_url):
                    image_urls.append(img_url)
                    
        except Exception as e:
            logger.debug(f"⚠️ {selector} 선택자 처리 중 오류: {e}")
            continue
    
    # 중복 제거하고 섞기
    unique_urls = list(set(image_urls))
    random.shuffle(unique_urls)
    
    logger.info(f"📸 총 {len(unique_urls)}개의 고유 이미지 URL 발견")
    return unique_urls[:5]  # 최대 5개 반환

def _is_valid_image_url_format(url: str) -> bool:
    """
    이미지 URL 형식이 유효한지 빠르게 검사합니다.
    
    Args:
        url (str): 검사할 URL
        
    Returns:
        bool: 유효한 이미지 URL 형식인지 여부
    """
    if not url or not url.startswith("http"):
        return False
    
    # Bing 이미지 URL 패턴 확인
    valid_patterns = [
        "th.bing.com/th/id/",
        "tse1.mm.bing.net",
        "tse2.mm.bing.net", 
        "tse3.mm.bing.net",
        "tse4.mm.bing.net"
    ]
    
    return any(pattern in url for pattern in valid_patterns)

def _validate_image_url(url: str) -> bool:
    """
    이미지 URL의 실제 유효성을 검증합니다.
    
    Args:
        url (str): 검증할 URL
        
    Returns:
        bool: 유효한 URL인지 여부
    """
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        
        # 상태 코드 확인
        if response.status_code != 200:
            return False
            
        # Content-Type 확인
        content_type = response.headers.get('content-type', '').lower()
        valid_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif']
        
        return any(img_type in content_type for img_type in valid_types)
        
    except Exception as e:
        logger.debug(f"⚠️ URL 검증 실패: {e}")
        return False

def get_multiple_player_images(player_names: List[str]) -> List[dict]:
    """
    여러 선수의 이미지를 동시에 크롤링합니다.
    
    Args:
        player_names (List[str]): 검색할 선수 이름 리스트
        
    Returns:
        List[dict]: {'name': str, 'image_url': str} 형태의 리스트
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_name = {
            executor.submit(get_player_image_from_bing, name): name 
            for name in player_names
        }
        
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                image_url = future.result()
                results.append({
                    'name': name,
                    'image_url': image_url
                })
            except Exception as e:
                logger.error(f"❌ {name} 처리 중 오류: {e}")
                results.append({
                    'name': name,
                    'image_url': None
                })
    
    return results

def get_player_image_as_pil(name: str) -> Optional[Image.Image]:
    """
    Bing에서 축구 선수 이미지를 검색하여 PIL Image 객체로 반환합니다.
    
    Args:
        name (str): 검색할 선수 이름
        
    Returns:
        Optional[Image.Image]: PIL Image 객체 또는 None (실패시)
    """
    img_url = get_player_image_from_bing(name)
    
    if not img_url:
        return None
        
    try:
        response = requests.get(img_url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            logger.info(f"✅ {name} 이미지 다운로드 성공")
            return img
        else:
            logger.error(f"❌ 이미지 다운로드 실패: HTTP {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"❌ 이미지 처리 실패: {e}")
        return None

