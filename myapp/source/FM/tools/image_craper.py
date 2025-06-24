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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_player_image_from_bing(name: str) -> Optional[str]:
    """
    Bing에서 축구 선수 이미지를 검색하여 URL을 반환합니다.
    
    Args:
        name (str): 검색할 선수 이름
        
    Returns:
        Optional[str]: 이미지 URL 또는 None (실패시)
    """
    try:
        chromedriver_autoinstaller.install()

        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # 랜덤 User-Agent 사용
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        options.add_argument(f"user-agent={random.choice(user_agents)}")

        driver = webdriver.Chrome(options=options)
        
        try:
            # 웹드라이버 숨김
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
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
            
            # 페이지 로드 대기 (랜덤 시간)
            time.sleep(random.uniform(2, 4))
            
            # 다양한 이미지 선택자 시도
            image_urls = _extract_multiple_image_urls(driver)
            
            if not image_urls:
                logger.warning(f"❌ {name}: 이미지 URL을 찾을 수 없음")
                return None
            
            # 랜덤하게 이미지 선택
            selected_url = random.choice(image_urls)
            
            # URL 유효성 검증
            if _validate_image_url(selected_url):
                logger.info(f"✅ {name}: 선택된 이미지 URL - {selected_url}")
                return selected_url
            else:
                logger.warning(f"❌ {name}: 유효하지 않은 이미지 URL")
                return None

        except Exception as e:
            logger.error(f"❌ {name} 이미지 검색 중 오류: {e}")
            return None
        finally:
            driver.quit()
            
    except Exception as e:
        logger.error(f"💥 {name} 크롤링 시스템 오류: {e}")
        return None

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

