"""
Resume crawl TVPL bằng Chrome THẬT của bạn (Remote Debugging).

Script kết nối vào Chrome đang mở (đã pass Cloudflare) và crawl 251 URLs còn thiếu.
Keywords từ crawl_by_topic.py đã được dùng trước đó để search URLs.

CÁCH DÙNG:

Bước 1: Đóng Chrome hiện tại (nếu đang mở)

Bước 2: Mở Chrome debug (CMD hoặc PowerShell):
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%TEMP%\\chrome_debug"

Bước 3: Trong Chrome đó → vào https://thuvienphapluat.vn → qua Cloudflare

Bước 4: Giữ Chrome MỞ, mở terminal KHÁC:
  python resume_crawl.py                  # Xem thống kê
  python resume_crawl.py --go --max 50    # Crawl 50 VB
  python resume_crawl.py --go --delay 6   # Chậm hơn
  python resume_crawl.py --go --topic 1   # Đầu mục 1

LƯU Ý:
  - Script KHÔNG tắt Chrome khi xong → chạy lại nhiều lần
  - Nếu Cloudflare chặn giữa chừng → giải trên Chrome → script tự tiếp
  - Delay mặc định 4s, tăng lên 6-8s nếu bị chặn
"""

import os
import sys
import io
import re
import json
import time
import random
import logging
import argparse

# Fix encoding trên Windows
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from crawl_by_topic import append_record, TOPICS, load_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_URL = "https://thuvienphapluat.vn"
DEBUG_PORT = 9222


def connect_to_chrome() -> webdriver.Chrome:
    """Ket noi vao Chrome dang chay (da pass Cloudflare)."""
    opts = Options()
    opts.add_experimental_option("debuggerAddress", f"127.0.0.1:{DEBUG_PORT}")

    try:
        driver = webdriver.Chrome(options=opts)
        log.info("Da ket noi Chrome (port %d)", DEBUG_PORT)
        log.info("   Tieu de: %s", driver.title[:60] if driver.title else "(trong)")
        return driver
    except Exception as e:
        log.error("Khong ket noi duoc Chrome!")
        log.error("")
        log.error('   Mo Chrome debug truoc:')
        log.error('   "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" '
                   '--remote-debugging-port=9222 '
                   '--user-data-dir="%%TEMP%%\\chrome_debug"')
        log.error("")
        log.error("   Sau do vao https://thuvienphapluat.vn va pass Cloudflare.")
        log.error("   Roi chay lai script nay.")
        log.error("   Loi: %s", e)
        return None


def scrape_doc(driver, url: str) -> dict | None:
    """Scrape noi dung 1 VB phap luat."""
    try:
        driver.get(url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "#tab1, .content1, .main-content, .the-document-body")
            )
        )
        time.sleep(1.5)
    except TimeoutException:
        # Check Cloudflare
        src = driver.page_source[:500].lower()
        if "cloudflare" in src or "just a moment" in src or "verify" in src:
            log.warning("  Cloudflare challenge — cho 20s...")
            time.sleep(20)
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "#tab1, .content1, .main-content")
                    )
                )
            except TimeoutException:
                return None
        else:
            return None

    record = {"url": url}

    # Title
    for sel in ["div.title-vb h1", "h1.title-vb", ".content1 h1", "h1"]:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            text = el.text.strip()
            if text and len(text) > 10:
                record["title"] = text
                break
        except NoSuchElementException:
            continue
    if "title" not in record:
        record["title"] = ""

    # So hieu
    record["document_number"] = ""
    if record["title"]:
        m = re.search(r'[Ss]o[:\s]*(\d+/\d{4}/[\w-]+)', record["title"])
        if m:
            record["document_number"] = m.group(1)
    if not record["document_number"]:
        m = re.search(r'/([A-Za-z-]+-\d+-\d{4}-[A-Za-z-]+)', url)
        if m:
            record["document_number"] = m.group(1)

    # Noi dung chinh
    content = ""
    for sel in ["#tab1 .content1", "#tab1.contentDoc", "#tab1 .contentDoc", "#tab1",
                ".noidung-vanban", ".the-document-body"]:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            content = el.text.strip()
            if len(content) > 200:
                break
        except NoSuchElementException:
            continue

    # Strip noise
    for marker in [
        "Luu tru\nGhi chu\nY kien",
        "Bai lien quan:",
        "Hoi dap phap luat",
        "Ban an lien quan",
        "Facebook\nEmail\nIn",
    ]:
        idx = content.find(marker)
        if idx > 0:
            content = content[:idx].rstrip()

    record["content"] = content

    # Metadata
    meta = {}
    for sel in [".boxTTVB .item", ".right-col p", ".ttvb .item"]:
        try:
            rows = driver.find_elements(By.CSS_SELECTOR, sel)
            for row in rows:
                text = row.text.strip()
                if not text:
                    continue
                for label, key in [
                    ("Loai van ban", "loai_van_ban"),
                    ("So hieu", "so_hieu"),
                    ("Co quan ban hanh", "co_quan"),
                    ("Nguoi ky", "nguoi_ky"),
                    ("Ngay ban hanh", "ngay_ban_hanh"),
                    ("Ngay hieu luc", "ngay_hieu_luc"),
                    ("Tinh trang", "tinh_trang"),
                ]:
                    if label in text:
                        val = text.replace(label, "").strip(": \t\n")
                        if val:
                            meta[key] = val
            if meta:
                break
        except Exception:
            continue

    if not record["document_number"] and meta.get("so_hieu"):
        record["document_number"] = meta["so_hieu"]

    record["meta"] = meta
    return record


def get_crawled_urls(topic_id: int) -> set:
    topic = TOPICS[topic_id]
    jsonl_file = os.path.join("output", f"topic_{topic_id:02d}_{topic['short']}.jsonl")
    urls = set()
    if not os.path.exists(jsonl_file):
        return urls
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("url"):
                    urls.add(r["url"])
            except json.JSONDecodeError:
                continue
    return urls


def main():
    parser = argparse.ArgumentParser(
        description="Resume crawl TVPL — dung Chrome that (Remote Debugging)",
    )
    parser.add_argument("--go", action="store_true", help="Bat dau crawl")
    parser.add_argument("--max", type=int, default=9999, help="So VB toi da")
    parser.add_argument("--delay", type=int, default=4, help="Delay giua requests (giay)")
    parser.add_argument("--topic", type=int, default=1, help="Dau muc (1-9)")
    args = parser.parse_args()

    topic_id = args.topic
    if topic_id not in TOPICS:
        log.error("Dau muc %d khong ton tai", topic_id)
        return

    # ── Thong ke ──
    state = load_state()
    topic_str = str(topic_id)
    # Lấy các link nháp chưa quét của Topic bằng "pending_urls"
    pending_urls = state.get("pending_urls", {}).get(topic_str, [])
    crawled_urls = get_crawled_urls(topic_id)
    
    # Loại bỏ những URL mà đã thực sự crawl vào file jsonl
    missing_urls = [u for u in pending_urls if u not in crawled_urls]

    print(f"\n{'='*60}")
    print(f"  Dau muc {topic_id}: {TOPICS[topic_id]['name']}")
    print(f"{'='*60}")
    print(f"  URLs trong pending: {len(pending_urls)}")
    print(f"  Da crawl (JSONL) : {len(crawled_urls)}")
    print(f"  Can crawl them   : {len(missing_urls)}")
    print(f"{'='*60}")

    if not missing_urls:
        print("  Da crawl het!")
        return

    if not args.go:
        print()
        print("  HUONG DAN:")
        print('  1. Mo Chrome debug:')
        print('     "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%TEMP%\\chrome_debug"')
        print('  2. Vao https://thuvienphapluat.vn va pass Cloudflare')
        print('  3. Giu Chrome MO, chay: python resume_crawl.py --go --max 50')
        return

    # ── Ket noi Chrome ──
    driver = connect_to_chrome()
    if not driver:
        return

    # Check Cloudflare
    current = driver.current_url
    if "thuvienphapluat" not in current:
        log.info("Chuyen den TVPL...")
        driver.get(BASE_URL)
        time.sleep(5)

    src = driver.page_source[:500].lower()
    if "cloudflare" in src or "just a moment" in src:
        log.error("Chrome van dang o Cloudflare!")
        log.error("   Giai Cloudflare tren Chrome truoc.")
        return

    log.info("Da xac nhan pass Cloudflare — bat dau crawl!")

    # ── Crawl ──
    max_docs = min(args.max, len(missing_urls))
    count = 0
    skipped = 0
    consecutive_fails = 0

    try:
        for i, url in enumerate(missing_urls):
            if count >= max_docs:
                log.info("Dat gioi han %d VB.", max_docs)
                break

            if consecutive_fails >= 10:
                log.error("10 fail lien tiep!")
                log.error("   Kiem tra Chrome — neu bi challenge, giai xong roi chay lai.")
                break

            log.info("[%d/%d] %s", count + 1, max_docs, url[-70:])
            record = scrape_doc(driver, url)

            if record and record.get("content") and len(record["content"]) > 100:
                record["topic_id"] = topic_id
                record["topic_name"] = TOPICS[topic_id]["name"]
                append_record(record, topic_id)
                count += 1
                consecutive_fails = 0

                title = record.get("title", "")[:55] or "(?)"
                log.info("   OK: %s [%d chars]", title, len(record["content"]))
            else:
                skipped += 1
                consecutive_fails += 1
                content_len = len(record.get("content", "")) if record else 0

                if content_len > 0:
                    log.warning("   Content ngan (%d chars)", content_len)
                else:
                    try:
                        ps = driver.page_source[:300].lower()
                        if "cloudflare" in ps or "just a moment" in ps:
                            log.warning("   Cloudflare! Cho 20s...")
                            time.sleep(20)
                            consecutive_fails = 0
                        else:
                            log.warning("   Rong (VB can dang nhap?)")
                    except Exception:
                        log.warning("   Khong doc duoc")

            actual_delay = random.uniform(max(1, args.delay - 2), args.delay + 2)
            time.sleep(actual_delay)

    except KeyboardInterrupt:
        log.info("\nDung. Da crawl %d VB.", count)

    # KHONG quit Chrome
    log.info("=" * 60)
    log.info("Crawl them: %d VB | Bo qua: %d", count, skipped)
    log.info("   Tong JSONL: ~%d VB", len(crawled_urls) + count)
    remaining = len(missing_urls) - count - skipped
    if remaining > 0:
        log.info("   Con %d URLs — chay lai script", remaining)
    log.info("   Chrome van mo — co the chay lai ngay")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
