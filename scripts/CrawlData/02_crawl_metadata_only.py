import os
import json
import time
import random
import logging
import argparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s")

INPUT_FILE = "output/01_hf_filtered.jsonl"
OUTPUT_FILE = "output/02_hf_with_full_meta.jsonl"
STATE_FILE = "crawl_metadata_state.json"

def make_debug_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    try:
        driver = webdriver.Chrome(options=opts)
        logging.info("✅ Đã kết nối Chrome debug (port 9222)")
        return driver
    except Exception as e:
        logging.error("❌ Không kết nối được Chrome debug: %s", e)
        logging.error("   (Vui lòng mở Chrome qua cmd: chrome.exe --remote-debugging-port=9222)")
        raise

def scrape_missing_meta(driver, url):
    try:
        driver.get(url)
    except Exception as e:
        logging.error("Lỗi khi tải trang: %s", e)
        return None, None
        
    time.sleep(random.uniform(2, 4))
    
    if "verify you are human" in driver.page_source.lower() or "cloudflare" in driver.page_source.lower():
        logging.warning("⚠️ Clouldflare block! Vui lòng tự pass trong Chrome. Đang chờ 15s...")
        time.sleep(15)
        
    tinh_trang = None
    ngay_hieu_luc = None
    
    # Tìm panel metadata
    for sel in [".boxTTVB .item", ".right-col p", ".ttvb .item"]:
        try:
            rows = driver.find_elements(By.CSS_SELECTOR, sel)
            for row in rows:
                text = row.text.strip()
                if not text: continue
                
                # Check Ngày hiệu lực
                if "Ngày hiệu lực" in text:
                    ngay_hieu_luc = text.replace("Ngày hiệu lực:", "").replace("Ngày hiệu lực", "").strip()
                if "Ban hành:" in text and not ngay_hieu_luc:  # Fallback nếu panel khác format
                     pass # Tránh bắt nhầm ngày ban hành

                # Check Tình trạng
                if "Tình trạng" in text:
                    tinh_trang = text.replace("Tình trạng:", "").replace("Tình trạng", "").strip()
                    
            if tinh_trang or ngay_hieu_luc:
                break
        except Exception:
            continue
            
    return tinh_trang, ngay_hieu_luc

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"visited": []}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000, help="Số lượng VB crawl trong 1 lần (để chia nhỏ)")
    args = parser.parse_args()

    state = load_state()
    visited = set(state.get("visited", []))
    
    # Đọc hết data
    all_docs = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_docs.append(json.loads(line))
                
    # Lọc những docs cần xử lý
    pending_docs = [d for d in all_docs if d["url"] not in visited]
    
    logging.info(f"Tổng số văn bản: {len(all_docs)} | Đã crawl meta: {len(visited)} | Còn lại: {len(pending_docs)}")
    if not pending_docs:
        logging.info("Đã hoàn thành 100%!")
        return

    driver = make_debug_driver()
    
    # Mở file output ở chế độ append
    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")
    
    count = 0
    try:
        for doc in pending_docs:
            if count >= args.limit: break
            
            url = doc["url"]
            logging.info(f"[{count+1}/{args.limit}] Đang cào Meta cho: {url[-60:]}")
            t_trang, n_hieuluc = scrape_missing_meta(driver, url)
            
            # Cập nhật doc
            if not doc.get("meta"):
                doc["meta"] = {}
            doc["meta"]["tinh_trang"] = t_trang or "Chưa rõ"
            doc["meta"]["ngay_hieu_luc"] = n_hieuluc or "Chưa rõ"
            
            out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            
            visited.add(url)
            state["visited"] = list(visited)
            save_state(state)
            count += 1
            
            # Lưu raw state mỗi 10 items để an toàn
            if count % 10 == 0:
                out_f.flush()
                
    except KeyboardInterrupt:
        logging.info("Dừng thủ công...")
    finally:
        out_f.close()
        logging.info("Đã lưu tiến độ!")

if __name__ == "__main__":
    main()
