"""Phân tích URLs còn thiếu — check xem có bị cắt hay lỗi gì."""
import json
import os
from collections import Counter

STATE_FILE = "crawl_state.json"
JSONL_FILE = os.path.join("output", "topic_01_to-chuc-bo-may.jsonl")

# Load state URLs
with open(STATE_FILE, "r", encoding="utf-8") as f:
    state = json.load(f)
all_urls = state.get("visited_urls", [])

# Load crawled URLs
crawled = set()
if os.path.exists(JSONL_FILE):
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    if r.get("url"):
                        crawled.add(r["url"])
                except:
                    pass

missing = [u for u in all_urls if u not in crawled]

print(f"Tổng URLs trong state: {len(all_urls)}")
print(f"Đã crawl (JSONL):      {len(crawled)}")
print(f"Còn thiếu:             {len(missing)}")
print()

# Phân tích missing URLs
print("=" * 70)
print("PHÂN TÍCH URLs CÒN THIẾU")
print("=" * 70)

# Check: URLs có kết thúc bằng .aspx không?
ends_aspx = [u for u in missing if u.endswith(".aspx")]
not_aspx = [u for u in missing if not u.endswith(".aspx")]

print(f"\nKết thúc bằng .aspx: {len(ends_aspx)}")
print(f"KHÔNG kết thúc bằng .aspx: {len(not_aspx)}")
if not_aspx:
    print("  Ví dụ URLs bất thường:")
    for u in not_aspx[:10]:
        print(f"    {u}")

# Check: URL bị cắt (không có mã số cuối)?
short_urls = [u for u in missing if len(u) < 80]
print(f"\nURLs ngắn bất thường (<80 ký tự): {len(short_urls)}")
for u in short_urls[:10]:
    print(f"  {u}")

# Check: URLs chứa ký tự lạ
import re
weird = [u for u in missing if re.search(r'[^\x20-\x7E]', u)]
print(f"\nURLs chứa ký tự non-ASCII: {len(weird)}")
for u in weird[:10]:
    print(f"  {u}")

# Phân loại theo lĩnh vực (từ URL path)
categories = Counter()
for u in missing:
    parts = u.replace("https://thuvienphapluat.vn/van-ban/", "").split("/")
    if parts:
        categories[parts[0]] += 1

print(f"\nPhân loại lĩnh vực ({len(missing)} URLs):")
for cat, cnt in categories.most_common(20):
    print(f"  {cat}: {cnt}")

# In 20 URLs thiếu đầu tiên
print(f"\n{'='*70}")
print(f"20 URLs THIẾU ĐẦU TIÊN:")
print(f"{'='*70}")
for i, u in enumerate(missing[:20]):
    print(f"  [{i+1}] {u}")

# So sánh: 10 URLs đã crawl OK
print(f"\n{'='*70}")
print(f"10 URLs ĐÃ CRAWL OK (để so sánh):")
print(f"{'='*70}")
crawled_list = list(crawled)
for i, u in enumerate(crawled_list[:10]):
    print(f"  [{i+1}] {u}")
