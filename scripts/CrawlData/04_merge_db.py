import json
import glob
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s")

OUTPUT_MERGED = "output/FINAL_DB.jsonl"

def main():
    logging.info("=== BƯỚC 4: MERGE VÀ KHỬ TRÙNG LẶP DỮ LIỆU ===")
    
    # Tập hợp các file cần merge
    # 1. Đầu ra của bước 2 (Dataset HF đã bù đủ Metadata)
    file_hf = "output/02_hf_with_full_meta.jsonl"
    
    # 2. Đầu ra của crawler mới nhất (Các topic_01_*, topic_02_*...)
    files_new = glob.glob(os.path.join("output", "topic_*.jsonl"))
    
    all_files = []
    if os.path.exists(file_hf):
        all_files.append(file_hf)
    all_files.extend(files_new)
    
    if not all_files:
        logging.error("Không tìm thấy file jsonl nào trong thư mục output để merge!")
        return

    unique_docs = {}
    conflict_count = 0
    
    for f_path in all_files:
        logging.info(f"Đang đọc: {f_path}")
        with open(f_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    doc = json.loads(line)
                    # Xác định ID: dùng Số hiệu (document_number) hoặc URL làm ID duy nhất
                    doc_id = doc.get("document_number") or doc.get("url")
                    if not doc_id: 
                        continue
                    
                    doc_id = doc_id.strip()
                    
                    # Nếu đã tồn tại, kiểm tra xem bản nào "ngon" hơn để giữ lại
                    if doc_id in unique_docs:
                        conflict_count += 1
                        old_doc = unique_docs[doc_id]
                        # Ưu tiên bản có metadata tinh_trang rõ ràng (không phải 'Chưa rõ')
                        old_tinh_trang = old_doc.get("meta", {}).get("tinh_trang", "Chưa rõ")
                        new_tinh_trang = doc.get("meta", {}).get("tinh_trang", "Chưa rõ")
                        
                        if new_tinh_trang != "Chưa rõ" and old_tinh_trang == "Chưa rõ":
                            unique_docs[doc_id] = doc # Ghi đè bản mới tốt hơn
                    else:
                        unique_docs[doc_id] = doc
                except json.JSONDecodeError:
                    continue

    logging.info(f"Quét xong! Phát hiện {conflict_count} trường hợp trùng lặp đã được gộp.")
    logging.info(f"Tổng số văn bản ĐỘC NHẤT thu được: {len(unique_docs)}")
    
    # Ghi ra file cuối
    with open(OUTPUT_MERGED, "w", encoding="utf-8") as out:
        for doc in unique_docs.values():
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            
    logging.info(f"🎉 Hoàn thành! File Database hợp nhất tại: {OUTPUT_MERGED}")

if __name__ == "__main__":
    main()
