import os
import json
import logging
import pandas as pd
import glob
import math

# Try to import TOPICS from the crawling script
try:
    import sys
    sys.path.append(r"f:\SpeechToText-indti\ChatBot2_Opus\scripts\CrawlData")
    from crawl_by_topic import TOPICS
except Exception as e:
    print("Warning: Could not import TOPICS. Using a fallback list.")
    TOPICS = {}

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s")

META_PARQUET = r"f:\SpeechToText-indti\ChatBot2_Opus\data\vietnamese-legal-documents\metadata\data-00000-of-00001.parquet"
CONTENT_DIR = r"f:\SpeechToText-indti\ChatBot2_Opus\data\vietnamese-legal-documents\content"
OUTPUT_FILE = r"f:\SpeechToText-indti\ChatBot2_Opus\scripts\CrawlData\output\01_hf_filtered.jsonl"

def clean_val(val):
    if pd.isna(val) or (isinstance(val, float) and math.isnan(val)):
        return None
    return str(val).strip()

def main():
    logging.info("Tải danh sách từ khóa...")
    all_keywords = []
    
    # ── TỪ KHÓA CHÍNH (Đặc quyền Nội Vụ) ──
    core_sectors = [
        "bộ máy hành chính", "cán bộ", "công chức", "viên chức", 
        "thi đua", "khen thưởng", "tổ chức sự nghiệp", "chính quyền địa phương",
        "tôn giáo", "tín ngưỡng", "văn thư", "lưu trữ", "biên chế", "dân vận"
    ]
    
    for _, t in TOPICS.items():
        all_keywords.extend([k.lower() for k in t["keywords"]])
    
    all_keywords = set(all_keywords)
    if not all_keywords:
        # Fallback
        all_keywords = set(core_sectors)
    
    logging.info(f"Tổng số từ khóa nội vụ: {len(all_keywords)}")

    # 1. Đọc Metadata
    logging.info("Đọc file Metadata Parquet (518K docs)...")
    df_meta = pd.read_parquet(META_PARQUET, engine='pyarrow')
    
    matched_ids = set()
    matched_records = {}

    for _, row in df_meta.iterrows():
        rid = row['id']
        title = str(row.get('title', '')).lower()
        sectors = str(row.get('legal_sectors', '')).lower()
        
        # 1.1 Match by sector first (Rất nhanh và chuẩn xác)
        is_matched = False
        for cs in core_sectors:
            if cs in sectors:
                is_matched = True
                break
        
        # 1.2 Match by title keywords
        if not is_matched:
            for kw in all_keywords:
                if kw in title:
                    is_matched = True
                    break
                    
        if is_matched:
            matched_ids.add(rid)
            matched_records[rid] = row.to_dict()

    logging.info(f"✅ Đã lọc ra được {len(matched_ids)} văn bản phù hợp nghiệp vụ Bộ Nội vụ!")

    # 2. Extract Content & Ghi JSONL
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    out_f = open(OUTPUT_FILE, "w", encoding="utf-8")
    
    content_files = glob.glob(os.path.join(CONTENT_DIR, "*.parquet"))
    extracted_count = 0
    
    logging.info("Bắt đầu đối chiếu và trích xuất nội dung từ 11 file Content...")
    for c_file in content_files:
        logging.info(f"Đang xử lý {os.path.basename(c_file)}...")
        df_c = pd.read_parquet(c_file, engine='pyarrow')
        
        for _, row in df_c.iterrows():
            rid = row['id']
            if rid in matched_ids:
                rec_meta = matched_records[rid]
                
                # Format to our system format
                formatted = {
                    "document_number": clean_val(rec_meta.get("document_number")),
                    "title": clean_val(rec_meta.get("title")),
                    "url": clean_val(rec_meta.get("url")),
                    "content": clean_val(row.get("content")),
                    "meta": {
                        "loai_van_ban": clean_val(rec_meta.get("legal_type")),
                        "so_hieu": clean_val(rec_meta.get("document_number")),
                        "co_quan": clean_val(rec_meta.get("issuing_authority")),
                        "nguoi_ky": clean_val(rec_meta.get("signers")),
                        "ngay_ban_hanh": clean_val(rec_meta.get("issuance_date")),
                        "linh_vuc": clean_val(rec_meta.get("legal_sectors")),
                        "tinh_trang": None,    # Sẽ crawl bù ở Script 2
                        "ngay_hieu_luc": None  # Sẽ crawl bù ở Script 2
                    }
                }
                out_f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                extracted_count += 1
                
                if extracted_count % 5000 == 0:
                    logging.info(f"   Đã trích xuất {extracted_count} văn bản...")

    out_f.close()
    logging.info(f"🎉 Hoàn tất! File xuất ra tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
