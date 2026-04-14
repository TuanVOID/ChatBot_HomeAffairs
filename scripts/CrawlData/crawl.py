import os
import json
import time
import logging
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# ── Config ──────────────────────────────────────────────────────────────────
START_URL = "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-89-2026-ND-CP-dieu-kien-kinh-doanh-dich-vu-kiem-dinh-xe-co-gioi-688213.aspx"
OUTPUT_DIR = "output"
VISITED_FILE = "visited.json"   # cache tên văn bản đã crawl (chống duplicate)
DATA_FILE = "data.jsonl"        # mỗi dòng là 1 văn bản (JSON Lines)
DELAY = 2                       # giây chờ giữa các request
MAX_DOCS = 500                  # giới hạn số văn bản (0 = không giới hạn)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_visited() -> set:
    if os.path.exists(VISITED_FILE):
        with open(VISITED_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_visited(visited: set):
    with open(VISITED_FILE, "w", encoding="utf-8") as f:
        json.dump(list(visited), f, ensure_ascii=False, indent=2)


def append_data(record: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, DATA_FILE)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1280,900")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=opts)


# ── Core scraping ─────────────────────────────────────────────────────────────
def scrape_page(driver: webdriver.Chrome, url: str) -> dict | None:
    """Scrape nội dung văn bản pháp luật từ 1 URL."""
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.ID, "tab1")))
    except TimeoutException:
        log.warning("Timeout khi tải: %s", url)
        return None

    record: dict = {"url": url}

    # Tiêu đề văn bản
    try:
        record["title"] = driver.find_element(By.CSS_SELECTOR, "h1.title-vb, h1").text.strip()
    except NoSuchElementException:
        record["title"] = ""

    # Nội dung chính trong #tab1 .contentDoc
    try:
        content_div = driver.find_element(By.CSS_SELECTOR, "#tab1.contentDoc, #tab1 .contentDoc, #tab1")
        record["content"] = content_div.text.strip()
    except NoSuchElementException:
        record["content"] = ""

    # Metadata: số hiệu, ngày ban hành, hiệu lực, tình trạng
    meta: dict = {}
    for row in driver.find_elements(By.CSS_SELECTOR, ".right-col p"):
        text = row.text.strip()
        if "Ban hành:" in text:
            meta["ban_hanh"] = text.replace("Ban hành:", "").strip()
        elif "Hiệu lực:" in text:
            meta["hieu_luc"] = text.replace("Hiệu lực:", "").strip()
        elif "Tình trạng:" in text:
            meta["tinh_trang"] = text.replace("Tình trạng:", "").strip()
        elif "Cập nhật:" in text:
            meta["cap_nhat"] = text.replace("Cập nhật:", "").strip()
    record["meta"] = meta

    return record


def collect_related_urls(driver: webdriver.Chrome) -> list[str]:
    """Thu thập các URL văn bản liên quan từ .GridBaseVBCT."""
    urls = []
    try:
        cards = driver.find_elements(By.CSS_SELECTOR, ".GridBaseVBCT .nqTitle a")
        for a in cards:
            href = a.get_attribute("href")
            if href and "thuvienphapluat.vn/van-ban/" in href:
                # Chỉ lấy tab mặc định (bỏ ?tab=...)
                clean = href.split("?")[0]
                urls.append(clean)
    except Exception as e:
        log.debug("Không lấy được related urls: %s", e)
    return urls


# ── Main crawler ──────────────────────────────────────────────────────────────
def crawl():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    visited = load_visited()
    queue: list[str] = [START_URL]
    driver = make_driver()
    count = 0

    try:
        while queue:
            if MAX_DOCS and count >= MAX_DOCS:
                log.info("Đã đạt giới hạn %d văn bản.", MAX_DOCS)
                break

            url = queue.pop(0)
            if url in visited:
                log.info("Bỏ qua (đã crawl): %s", url)
                continue

            log.info("[%d] Crawling: %s", count + 1, url)
            record = scrape_page(driver, url)

            if record:
                append_data(record)
                visited.add(url)
                save_visited(visited)
                count += 1

                # Thêm các văn bản liên quan vào queue
                related = collect_related_urls(driver)
                new_urls = [u for u in related if u not in visited and u not in queue]
                log.info("  → Tìm thấy %d văn bản liên quan (%d mới)", len(related), len(new_urls))
                queue.extend(new_urls)
            else:
                # Đánh dấu là đã thử để không lặp lại
                visited.add(url)
                save_visited(visited)

            time.sleep(DELAY)

    except KeyboardInterrupt:
        log.info("Dừng bởi người dùng.")
    finally:
        driver.quit()
        log.info("Hoàn thành. Đã crawl %d văn bản. Dữ liệu lưu tại: %s/%s", count, OUTPUT_DIR, DATA_FILE)


if __name__ == "__main__":
    crawl()
