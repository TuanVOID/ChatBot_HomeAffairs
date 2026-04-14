"""
05_crawl_by_org.py — Parallel Crawler theo Cơ quan ban hành
============================================================
Kết nối vào Chrome Debug (port tuỳ chọn), duyệt trang tìm kiếm
theo bộ lọc Cơ quan ban hành, thu thập URL rồi cào nội dung.

Hỗ trợ chia trang (--start-page / --end-page) để nhiều worker
cùng cào song song 1 cơ quan lớn mà không đụng nhau.

Cách dùng:
    # Worker cào toàn bộ 1 cơ quan
    python 05_crawl_by_org.py --port 9222 --orgs 12 --delay 8

    # Worker cào Cơ quan TW, chỉ trang 1-50
    python 05_crawl_by_org.py --port 9222 --orgs 1 --delay 8 --start-page 1 --end-page 50

    # Resume sau khi tạm dừng
    python 05_crawl_by_org.py --port 9222 --orgs 1 --delay 8 --start-page 1 --end-page 50 --resume
"""

import os
import json
import time
import random
import logging
import argparse
import re
import io
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
)

# OCR cho captcha
try:
    import pytesseract
    from PIL import Image, ImageOps, ImageFilter
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 13 Cơ quan cần tải ──
ORG_MAP = {
    1:   {"slug": "co-quan-tw",           "name": "Cơ quan TW",                        "target": 4000},
    3:   {"slug": "bhxh-viet-nam",        "name": "BHXH Việt Nam",                     "target": 80},
    6:   {"slug": "bo-giao-duc-dao-tao",  "name": "Bộ Giáo dục & Đào tạo",            "target": 200},
    12:  {"slug": "bo-noi-vu",            "name": "Bộ Nội vụ",                         "target": 120},
    19:  {"slug": "bo-van-hoa-tt-dl",     "name": "Bộ Văn hoá, TT & DL",              "target": 140},
    22:  {"slug": "chinh-phu",            "name": "Chính phủ",                         "target": 460},
    23:  {"slug": "chu-tich-nuoc",        "name": "Chủ tịch nước",                     "target": 80},
    26:  {"slug": "quoc-hoi",             "name": "Quốc hội",                          "target": 300},
    33:  {"slug": "thu-tuong-chinh-phu",  "name": "Thủ tướng Chính phủ",               "target": 500},
    95:  {"slug": "tong-ldld-viet-nam",   "name": "Tổng LĐLĐ Việt Nam",               "target": 10},
    97:  {"slug": "uy-ban-tvqh",          "name": "Ủy ban TVQH",                       "target": 120},
    98:  {"slug": "van-phong-chinh-phu",  "name": "Văn phòng Chính phủ",               "target": 300},
    104: {"slug": "bo-dan-toc-ton-giao",  "name": "Bộ Dân tộc & Tôn giáo",            "target": 40},
}

OUTPUT_DIR = "output"
STATE_DIR  = "state"
BASE_URL   = "https://thuvienphapluat.vn"
SEARCH_TPL = BASE_URL + "/page/tim-van-ban.aspx?keyword=&area=0&type=0&status=0&lan=1&org={org}&signer=0&match=True&sort=1&bdate=12/04/1946&edate=13/04/2026&page={page}"

DELAY = 8
OCR_MIN_LEN = 5
OCR_MAX_LEN = 8
OCR_TRY_LIMIT = 4
CAPTCHA_NAV_RETRY = 3


# ══════════════════════════════════════════════════════════
#  DRIVER
# ══════════════════════════════════════════════════════════
def make_driver(port: int) -> webdriver.Chrome:
    opts = Options()
    opts.add_experimental_option("debuggerAddress", f"127.0.0.1:{port}")
    try:
        driver = webdriver.Chrome(options=opts)
        log.info("✅ Kết nối Chrome thành công (port %d)", port)
        return driver
    except Exception as e:
        log.error("❌ Không kết nối được Chrome port %d: %s", port, e)
        raise
# ══════════════════════════════════════════════════════════
#  AUTO-SOLVE TVPL CAPTCHA (Screenshot + OCR)
# ══════════════════════════════════════════════════════════
def _build_ocr_variants(raw_img: Image.Image) -> list[Image.Image]:
    """Tạo nhiều biến thể ảnh để tăng tỉ lệ đọc captcha."""
    variants: list[Image.Image] = []
    gray = ImageOps.autocontrast(raw_img.convert("L"))

    for scale in (2, 3):
        up = gray.resize((gray.width * scale, gray.height * scale), Image.Resampling.LANCZOS)
        denoise = up.filter(ImageFilter.MedianFilter(size=3))

        for threshold in (90, 110, 130, 150, 170):
            bw = denoise.point(lambda x, t=threshold: 255 if x > t else 0)
            variants.append(bw)
            variants.append(ImageOps.invert(bw))

    return variants


def _ocr_captcha_code(raw_img: Image.Image) -> tuple[str | None, list[str]]:
    """Trả về mã captcha tốt nhất + danh sách ứng viên để debug."""
    configs = (
        "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789",
        "--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789",
        "--oem 1 --psm 13 -c tessedit_char_whitelist=0123456789",
    )

    score_map: dict[str, int] = {}
    seen_raw: list[str] = []
    variants = _build_ocr_variants(raw_img)

    for img in variants:
        for cfg in configs:
            text = pytesseract.image_to_string(img, config=cfg).strip()
            digits = re.sub(r"\D", "", text)
            if not digits:
                continue

            seen_raw.append(digits)

            if not (OCR_MIN_LEN <= len(digits) <= OCR_MAX_LEN):
                continue

            # Ưu tiên chuỗi có chiều dài gần 6 (thường gặp) + xuất hiện lặp lại.
            score = 2 if len(digits) == 6 else 1
            score_map[digits] = score_map.get(digits, 0) + score

    if not score_map:
        return None, seen_raw

    best = max(score_map.items(), key=lambda kv: kv[1])[0]
    return best, seen_raw


def auto_solve_captcha(driver: webdriver.Chrome) -> bool:
    """Tự động giải captcha TVPL bằng screenshot + OCR.
    Captcha là ảnh chứa dãy số đơn giản.
    Trả về True nếu giải thành công hoặc đã xử lý."""
    if "/check.aspx" not in driver.current_url:
        return False

    log.warning("  🔒 TVPL captcha detected! Đang tự động giải...")
    time.sleep(1.5)

    try:
        if not HAS_OCR:
            log.warning("  ⚠️ Tesseract chưa cài. Cần nhập tay.")
            log.warning("     Hãy nhập mã thủ công trong Chrome rồi nhấn Enter...")
            input("  >> Nhấn Enter sau khi đã nhập mã & bấm OK... ")
            return True

        for attempt in range(1, OCR_TRY_LIMIT + 1):
            captcha_code = None
            raw_candidates: list[str] = []

            try:
                captcha_img = driver.find_element(By.CSS_SELECTOR, "img[src*='RegistImage']")
            except NoSuchElementException:
                log.warning("  ⚠️ Không tìm thấy ảnh RegistImage.aspx")
                break

            raw_img = Image.open(io.BytesIO(captcha_img.screenshot_as_png))
            captcha_code, raw_candidates = _ocr_captcha_code(raw_img)

            if not captcha_code:
                log.warning("  ❌ OCR lượt %d/%d chưa đọc được mã (ứng viên: %s)",
                            attempt, OCR_TRY_LIMIT, raw_candidates[:5])
                time.sleep(1)
                continue

            log.info("  🔑 OCR lượt %d/%d: %s", attempt, OCR_TRY_LIMIT, captcha_code)

            input_box = driver.find_element(By.ID, "ctl00_Content_txtSecCode")
            input_box.clear()
            input_box.send_keys(captcha_code)
            time.sleep(0.4)

            btn = driver.find_element(By.ID, "ctl00_Content_CheckButton")
            btn.click()
            time.sleep(2.5)

            if "/check.aspx" not in driver.current_url:
                log.info("  ✅ Captcha đã được giải!")
                return True

            log.warning("  ⚠️ OCR sai lượt %d/%d (%s), thử lại...", attempt, OCR_TRY_LIMIT, captcha_code)
            time.sleep(0.8)

        log.warning("  ❌ OCR thất bại sau %d lượt.", OCR_TRY_LIMIT)
        log.warning("     Hãy nhập mã thủ công trong Chrome rồi nhấn Enter...")
        input("  >> Nhấn Enter sau khi đã nhập mã thủ công & bấm OK... ")
        return True

    except Exception as e:
        log.error("  ❌ Lỗi giải captcha: %s. Nhập tay nhé...", e)
        input("  >> Nhấn Enter sau khi đã nhập mã & bấm OK... ")
        return True


def _normalize_url(url: str) -> str:
    return url.split("#")[0].split("?")[0].rstrip("/")


def _click_target_link_if_present(driver: webdriver.Chrome, target_url: str) -> bool:
    """Ưu tiên click link thật trên trang hiện tại để giống thao tác người dùng."""
    target_norm = _normalize_url(target_url)
    links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/van-ban/']")
    for a in links:
        try:
            href = a.get_attribute("href")
            if not href:
                continue
            if _normalize_url(href) != target_norm:
                continue
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", a)
            time.sleep(random.uniform(0.2, 0.6))
            a.click()
            time.sleep(random.uniform(2.0, 3.5))
            log.info("  🔗 Click link trực tiếp: %s", href)
            return True
        except Exception:
            continue
    return False


def resolve_captcha_and_return(driver: webdriver.Chrome, target_url: str) -> bool:
    """Giải captcha rồi quay lại URL đích với backoff để tránh loop check.aspx."""
    target_norm = _normalize_url(target_url)

    for attempt in range(1, CAPTCHA_NAV_RETRY + 1):
        auto_solve_captcha(driver)

        # Nếu captcha đã pass và đã ở đúng URL đích thì thôi.
        if "/check.aspx" not in driver.current_url:
            current_norm = _normalize_url(driver.current_url)
            log.info("  ↩️ Sau captcha đang ở: %s", driver.current_url)
            if current_norm == target_norm:
                return True

            # Ưu tiên click link trên trang hiện tại (nếu có) thay vì get trực tiếp.
            if _click_target_link_if_present(driver, target_url):
                if "/check.aspx" not in driver.current_url:
                    return True

            # Chỉ fallback get sau một nhịp nghỉ.
            time.sleep(random.uniform(DELAY + 2, DELAY + 5))
            log.info("  🌐 Fallback GET URL đích: %s", target_url)
            driver.get(target_url)
            time.sleep(random.uniform(DELAY + 1, DELAY + 4))

        if "/check.aspx" not in driver.current_url:
            return True

        backoff = min(45, DELAY * (attempt + 1))
        log.warning("  ⚠️ Vẫn bị chặn captcha sau lượt %d/%d, nghỉ %.1fs rồi thử lại...",
                    attempt, CAPTCHA_NAV_RETRY, backoff)
        time.sleep(backoff)

    log.error("  ❌ Không thoát được vòng captcha cho URL: %s", target_url)
    return False


# ══════════════════════════════════════════════════════════
#  THU THẬP URLs TỪ TRANG TÌM KIẾM
# ══════════════════════════════════════════════════════════
def collect_urls(driver: webdriver.Chrome, org_id: int,
                 start_page: int = 1, end_page: int = 999) -> list[str]:
    """Duyệt các trang tìm kiếm từ start_page đến end_page, trả về list URLs."""
    all_urls = []
    page = start_page

    while page <= end_page:
        url = SEARCH_TPL.format(org=org_id, page=page)
        log.info("  📄 Trang %d/%d ...", page, end_page)

        try:
            driver.get(url)
            time.sleep(random.uniform(3, 5))

            # Phát hiện TVPL captcha (tìm kiếm quá nhanh)
            if "/check.aspx" in driver.current_url:
                if not resolve_captcha_and_return(driver, url):
                    log.warning("  ⚠️ Bỏ qua trang %d do captcha lặp lại.", page)
                    page += 1
                    continue

            WebDriverWait(driver, 12).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "p.vblist, .nqTitle, .doc-title, .content-area")
                )
            )
        except TimeoutException:
            current = driver.current_url
            if "/check.aspx" in current:
                if not resolve_captcha_and_return(driver, url):
                    log.warning("  ⚠️ Bỏ qua trang %d do captcha lặp lại.", page)
                    page += 1
                    continue
                continue
            src = driver.page_source.lower()
            if "verify you are human" in src or "cloudflare" in src or "sorry" in src:
                log.error("  🛑 Cloudflare! Hãy pass thủ công rồi nhấn Enter...")
                input("  >> Nhấn Enter sau khi đã pass Cloudflare... ")
                continue
            log.warning("  ⚠️ Timeout trang %d — hết kết quả.", page)
            break

        # Bóc link VB
        found_on_page = 0
        for sel in ["p.vblist > a", "p.vblist a", ".nqTitle a", "a[href*='/van-ban/']"]:
            try:
                links = driver.find_elements(By.CSS_SELECTOR, sel)
                for a in links:
                    try:
                        href = a.get_attribute("href")
                        if href and "/van-ban/" in href and ".aspx" in href:
                            clean = href.split("?")[0].split("#")[0]
                            if clean not in all_urls:
                                all_urls.append(clean)
                                found_on_page += 1
                    except StaleElementReferenceException:
                        continue
            except Exception:
                continue
            if found_on_page > 0:
                break

        if found_on_page == 0:
            log.info("  Trang %d: 0 kết quả mới → Kết thúc.", page)
            break

        log.info("  → +%d (tổng: %d)", found_on_page, len(all_urls))
        page += 1
        time.sleep(random.uniform(max(1, DELAY - 2), DELAY + 2))

    return all_urls


# ══════════════════════════════════════════════════════════
#  CÀO NỘI DUNG 1 VĂN BẢN
# ══════════════════════════════════════════════════════════
def scrape_document(driver: webdriver.Chrome, url: str) -> dict | None:
    try:
        # Ưu tiên click link ngay trên trang hiện tại; fallback mới dùng get(url).
        opened_by_click = _click_target_link_if_present(driver, url)
        if not opened_by_click:
            log.info("  🌐 Open bằng GET: %s", url)
            driver.get(url)
        time.sleep(random.uniform(3, 5))
    except Exception as e:
        log.error("  Lỗi tải trang: %s", e)
        return None

    # Phát hiện TVPL captcha
    if "/check.aspx" in driver.current_url:
        if not resolve_captcha_and_return(driver, url):
            return None

    src = driver.page_source.lower()
    if "verify you are human" in src or "sorry" in src:
        log.warning("  🛑 Cloudflare! Pass thủ công rồi nhấn Enter...")
        input("  >> Nhấn Enter sau khi đã pass Cloudflare... ")
        try:
            driver.get(url)
            time.sleep(4)
        except Exception:
            return None

    record = {"url": url, "content": "", "meta": {}}

    # Tiêu đề
    for sel in ["h1", ".doc-title", ".title"]:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            record["title"] = el.text.strip()
            break
        except NoSuchElementException:
            continue

    # Nội dung
    content = ""
    for sel in [
        "div.content1 div.content1", "div.content1",
        ".box-content-vb", ".toan-van-container",
        "#toanvancontent", "article",
    ]:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            content = el.text.strip()
            if len(content) > 200:
                break
        except NoSuchElementException:
            continue

    for marker in [
        "Lưu trữ\nGhi chú\nÝ kiến", "Bài liên quan:",
        "Hỏi đáp pháp luật", "Bản án liên quan", "Facebook\nEmail\nIn",
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
                    ("Loại văn bản", "loai_van_ban"),
                    ("Số hiệu", "so_hieu"),
                    ("Cơ quan ban hành", "co_quan"),
                    ("Người ký", "nguoi_ky"),
                    ("Ngày ban hành", "ngay_ban_hanh"),
                    ("Ngày hiệu lực", "ngay_hieu_luc"),
                    ("Tình trạng", "tinh_trang"),
                    ("Lĩnh vực", "linh_vuc"),
                ]:
                    if label in text:
                        val = text.replace(label, "").strip().lstrip(":").strip()
                        meta[key] = val
            if meta:
                break
        except Exception:
            continue

    record["meta"] = meta
    record["document_number"] = meta.get("so_hieu", "")
    return record


# ══════════════════════════════════════════════════════════
#  STATE & OUTPUT (Có hỗ trợ page-range để tránh conflict)
# ══════════════════════════════════════════════════════════
def get_file_suffix(start_page: int | None, end_page: int | None) -> str:
    """Tạo hậu tố file dựa trên page range để nhiều worker
    cùng cào 1 org mà không đè file nhau."""
    if start_page is not None and start_page > 1:
        return f"_p{start_page:03d}"
    if end_page is not None and end_page < 999:
        return f"_p001"
    return ""

def get_state_file(org_id: int, suffix: str) -> str:
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, f"org_{org_id}{suffix}.json")

def get_output_file(org_id: int, suffix: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slug = ORG_MAP[org_id]["slug"]
    return os.path.join(OUTPUT_DIR, f"org_{org_id:03d}_{slug}{suffix}.jsonl")

def load_state(state_file: str) -> dict:
    if os.path.exists(state_file):
        with open(state_file, "r", encoding="utf-8") as fp:
            return json.load(fp)
    return {"pending_urls": [], "visited_urls": []}

def save_state(state_file: str, state: dict):
    with open(state_file, "w", encoding="utf-8") as fp:
        json.dump(state, fp, ensure_ascii=False)

def get_crawled_urls(output_file: str) -> set:
    urls = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        urls.add(json.loads(line).get("url", ""))
                    except json.JSONDecodeError:
                        continue
    return urls


# ══════════════════════════════════════════════════════════
#  MAIN CRAWLER
# ══════════════════════════════════════════════════════════
def crawl_org(driver: webdriver.Chrome, org_id: int,
              start_page: int = 1, end_page: int = 999,
              resume: bool = False, max_docs: int = 9999):
    
    info = ORG_MAP[org_id]
    suffix = get_file_suffix(start_page, end_page)
    state_file = get_state_file(org_id, suffix)
    output_file = get_output_file(org_id, suffix)

    log.info("=" * 60)
    log.info("🏛️  %s (org=%d) | Trang %d→%d", info["name"], org_id, start_page, end_page)
    log.info("   Output: %s", output_file)
    log.info("=" * 60)

    state = load_state(state_file)
    crawled_urls = get_crawled_urls(output_file)

    if resume and state.get("pending_urls"):
        all_urls = state["pending_urls"]
        log.info("📋 Resume: %d URLs đã lưu", len(all_urls))
    else:
        # Pha 1: Thu thập URLs
        log.info("🔍 Pha 1: Thu thập URLs (trang %d → %d)...", start_page, end_page)
        all_urls = collect_urls(driver, org_id, start_page, end_page)
        state["pending_urls"] = all_urls
        save_state(state_file, state)
        log.info("💾 Đã lưu %d pending URLs", len(all_urls))

    # Loại bỏ đã crawl
    visited_set = set(state.get("visited_urls", []))
    pending = [u for u in all_urls if u not in crawled_urls and u not in visited_set]

    target = min(info["target"], max_docs)
    log.info("📊 Tổng: %d | Đã xong: %d | Còn: %d | Mục tiêu: %d",
             len(all_urls), len(crawled_urls), len(pending), target)

    if not pending:
        log.info("✅ Không còn gì để cào!")
        return

    # Pha 2: Cào nội dung
    log.info("🚀 Pha 2: Cào nội dung...")
    out_f = open(output_file, "a", encoding="utf-8")

    count = len(crawled_urls)
    fail_streak = 0
    try:
        for i, url in enumerate(pending):
            if count >= target:
                log.info("⏹️  Đạt mục tiêu %d VB!", target)
                break

            log.info("[%d/%d] ...%s", i + 1, len(pending), url[-55:])
            doc = scrape_document(driver, url)

            if doc and doc.get("content"):
                out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                count += 1
                fail_streak = 0
                log.info("  ✅ OK (%d bytes) | Tổng: %d/%d",
                         len(doc["content"]), count, target)
            else:
                fail_streak += 1
                log.warning("  ❌ Thất bại (streak: %d)", fail_streak)
                if fail_streak >= 5:
                    log.error("  🛑 5 lần liên tục! Tạm dừng 60s...")
                    time.sleep(60)
                    fail_streak = 0

            # Cập nhật state
            state.setdefault("visited_urls", []).append(url)
            if (i + 1) % 5 == 0:
                save_state(state_file, state)
                out_f.flush()

            time.sleep(random.uniform(max(1, DELAY - 2), DELAY + 2))

    except KeyboardInterrupt:
        log.info("⏸️  Dừng bởi Ctrl+C")
    finally:
        save_state(state_file, state)
        out_f.close()
        log.info("💾 Đã lưu! Tổng: %d VB cho %s", count, info["name"])


def main():
    global DELAY

    parser = argparse.ArgumentParser(
        description="Crawl VB pháp luật song song theo Cơ quan ban hành")
    parser.add_argument("--port", type=int, required=True,
                        help="Chrome debug port (9222-9227)")
    parser.add_argument("--orgs", type=str, required=True,
                        help="Org IDs phân cách bằng dấu phẩy (VD: 12,22)")
    parser.add_argument("--delay", type=float, default=DELAY,
                        help=f"Delay giữa requests (mặc định {DELAY}s)")
    parser.add_argument("--max", type=int, default=9999,
                        help="Số VB tối đa mỗi cơ quan")
    parser.add_argument("--start-page", type=int, default=1,
                        help="Trang bắt đầu (dùng để chia worker)")
    parser.add_argument("--end-page", type=int, default=999,
                        help="Trang kết thúc")
    parser.add_argument("--resume", action="store_true",
                        help="Dùng URLs đã thu thập, bỏ qua pha search")
    parser.add_argument("--list", action="store_true",
                        help="Liệt kê danh sách cơ quan")
    args = parser.parse_args()

    if args.list:
        print("\n  Danh sách 13 Cơ quan ban hành:")
        print("  " + "=" * 55)
        for oid in sorted(ORG_MAP.keys()):
            info = ORG_MAP[oid]
            print(f"  org={oid:<4} │ {info['name']:<30} │ ~{info['target']} VB")
        print("  " + "=" * 55)
        total = sum(v["target"] for v in ORG_MAP.values())
        print(f"  TỔNG: ~{total} văn bản")
        return

    DELAY = args.delay
    org_ids = [int(x.strip()) for x in args.orgs.split(",")]

    for oid in org_ids:
        if oid not in ORG_MAP:
            log.error("❌ Org ID %d không tồn tại! Dùng --list để xem.", oid)
            return

    driver = make_driver(args.port)

    for oid in org_ids:
        crawl_org(driver, oid,
                  start_page=args.start_page,
                  end_page=args.end_page,
                  resume=args.resume,
                  max_docs=args.max)
        if len(org_ids) > 1:
            log.info("⏳ Chờ 10s trước cơ quan tiếp...\n")
            time.sleep(10)

    log.info("🎉 Worker (port %d) hoàn thành!", args.port)


if __name__ == "__main__":
    main()
