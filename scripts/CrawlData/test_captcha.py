"""Test auto-solve captcha - v3"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import pytesseract
import time, io, re
from PIL import Image, ImageOps, ImageFilter

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

port = int(sys.argv[1]) if len(sys.argv) > 1 else 9222
opts = Options()
opts.add_experimental_option("debuggerAddress", f"127.0.0.1:{port}")
driver = webdriver.Chrome(options=opts)

print(f"URL: {driver.current_url}")
if "/check.aspx" not in driver.current_url:
    print("Khong phai trang captcha!")
    exit()

# Tim DUNG anh captcha = RegistImage.aspx
captcha_img = driver.find_element(By.CSS_SELECTOR, "img[src*='RegistImage']")
print(f"Tim thay captcha: {captcha_img.size} src={captcha_img.get_attribute('src')}")

# Screenshot
png = captcha_img.screenshot_as_png
pil_img = Image.open(io.BytesIO(png))
pil_img.save("captcha_raw.png")
print(f"Luu captcha_raw.png ({pil_img.size})")

configs = [
    ("--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789", "psm7"),
    ("--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789", "psm8"),
    ("--oem 1 --psm 13 -c tessedit_char_whitelist=0123456789", "psm13"),
]


def build_variants(raw: Image.Image):
    variants = []
    gray = ImageOps.autocontrast(raw.convert("L"))
    for scale in (2, 3):
        up = gray.resize((gray.width * scale, gray.height * scale), Image.Resampling.LANCZOS)
        denoise = up.filter(ImageFilter.MedianFilter(size=3))
        for thresh in (90, 110, 130, 150, 170):
            bw = denoise.point(lambda x, t=thresh: 255 if x > t else 0)
            variants.append((f"s{scale}_t{thresh}", bw))
            variants.append((f"s{scale}_t{thresh}_inv", ImageOps.invert(bw)))
    return variants


score = {}
all_hits = []
best = None
for name, var_img in build_variants(pil_img):
    for cfg, cfg_name in configs:
        text = pytesseract.image_to_string(var_img, config=cfg).strip()
        digits = re.sub(r'\D', '', text)
        if not digits:
            continue
        all_hits.append((name, cfg_name, text, digits))
        if 5 <= len(digits) <= 8:
            bonus = 2 if len(digits) == 6 else 1
            score[digits] = score.get(digits, 0) + bonus

print("\nTop OCR hits:")
for name, cfg_name, text, digits in all_hits[:20]:
    print(f"  {name:12s} {cfg_name}: raw='{text}' -> digits='{digits}'")

if score:
    best = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    print("\nScoreboard:")
    for code, sc in sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"  {code}: {sc}")

if best:
    print(f"\n>>> MA CAPTCHA: {best} <<<")
    inp = driver.find_element(By.ID, "ctl00_Content_txtSecCode")
    inp.clear()
    inp.send_keys(best)
    time.sleep(0.5)
    btn = driver.find_element(By.ID, "ctl00_Content_CheckButton")
    btn.click()
    time.sleep(3)
    if "/check.aspx" not in driver.current_url:
        print("THANH CONG!")
    else:
        print("Ma sai :(")
else:
    print("\nKhong doc duoc. Kiem tra captcha_raw.png")
