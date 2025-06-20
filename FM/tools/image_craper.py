# player_image_viewer.py
import requests
from PIL import Image
from io import BytesIO
from typing import Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import chromedriver_autoinstaller

def get_player_image_from_bing(name: str) -> Optional[Image.Image]:
    chromedriver_autoinstaller.install()

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("user-agent=Mozilla/5.0")

    driver = webdriver.Chrome(options=options)

    try:
        query = name.replace(" ", "+")
        url = f"https://www.bing.com/images/search?q=soccer+player+{query}&form=HDRSC2"
        driver.get(url)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "img.mimg"))
        )
        img_element = driver.find_element(By.CSS_SELECTOR, "img.mimg")
        
        img_url = img_element.get_attribute("src")

        if not img_url or not img_url.startswith("http"):
            return None

        response = requests.get(img_url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img

    except Exception as e:
        print(f"이미지 불러오기 실패: {e}")
        return None
    finally:
        driver.quit()

