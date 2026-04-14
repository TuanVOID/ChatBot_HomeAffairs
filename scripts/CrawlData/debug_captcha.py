"""Chạy trên 1 Chrome đang mở trang check.aspx để xem cấu trúc HTML captcha."""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import sys

port = int(sys.argv[1]) if len(sys.argv) > 1 else 9222

opts = Options()
opts.add_experimental_option("debuggerAddress", f"127.0.0.1:{port}")
driver = webdriver.Chrome(options=opts)

url = driver.current_url
print(f"URL hiện tại: {url}")

if "/check.aspx" not in url:
    print("⚠️ Trang hiện tại KHÔNG phải check.aspx")
    print("Hãy mở trang captcha trên Chrome trước rồi chạy lại!")
    # Thử navigate tới captcha
    driver.get("https://thuvienphapluat.vn/page/check.aspx")
    import time; time.sleep(3)

src = driver.page_source

# Tìm vùng captcha
import re

# In ra tất cả input elements
from selenium.webdriver.common.by import By
inputs = driver.find_elements(By.TAG_NAME, "input")
print(f"\n=== TẤT CẢ INPUT ({len(inputs)}) ===")
for inp in inputs:
    try:
        print(f"  id={inp.get_attribute('id')!r}  name={inp.get_attribute('name')!r}  type={inp.get_attribute('type')!r}  visible={inp.is_displayed()}")
    except: pass

# Tìm tất cả elements chứa số 4-8 chữ số
print("\n=== ELEMENTS CHỨA SỐ (4-8 digits) ===")
all_els = driver.find_elements(By.XPATH, "//*[string-length(normalize-space(text()))>=4 and string-length(normalize-space(text()))<=8]")
for el in all_els:
    txt = el.text.strip()
    if txt and txt.isdigit():
        tag = el.tag_name
        eid = el.get_attribute("id") or ""
        cls = el.get_attribute("class") or ""
        style = el.get_attribute("style") or ""
        print(f"  TAG={tag}  id={eid!r}  class={cls!r}  style={style[:80]!r}  TEXT={txt!r}")

# Tìm buttons
print("\n=== BUTTONS & SUBMIT ===")
btns = driver.find_elements(By.CSS_SELECTOR, "input[type='submit'], input[type='button'], button")
for b in btns:
    try:
        print(f"  id={b.get_attribute('id')!r}  value={b.get_attribute('value')!r}  text={b.text!r}  visible={b.is_displayed()}")
    except: pass

# Trích HTML vùng captcha
print("\n=== HTML SNIPPET (tìm 'captcha' hoặc 'bảo vệ') ===")
matches = re.findall(r'(.{0,200}(?:captcha|bảo vệ|robot|mã bảo|xác nhận).{0,200})', src, re.IGNORECASE)
for m in matches[:5]:
    print(f"  ...{m.strip()[:300]}...")
