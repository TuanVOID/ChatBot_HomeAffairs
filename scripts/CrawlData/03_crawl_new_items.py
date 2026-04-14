import os
import argparse
import logging
from crawl_by_topic import crawl_topic, print_menu

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s")

def main():
    print("="*60)
    print("🚀 SCRIPT 3: CÀO BỔ SUNG VĂN BẢN MỚI TỪ 1/1/2025")
    print("💡 Mẹo: Script này tái sử dụng crawl_by_topic.py, nhưng bạn nên kết hợp")
    print("   từ khóa với tiền tố hoặc bóp tìm kiếm bằng cách thêm từ khóa 'năm 2025'")
    print("   vào crawl_by_topic.py. (Phiên bản cải tiến tương lai sẽ tự động hóa URL params)")
    print("="*60)
    
    print("Vui lòng chạy script crawl_by_topic.py gốc bằng lệnh:")
    print("python crawl_by_topic.py --topic <số> --debug")
    print("Hệ thống sẽ tổng hợp tự động tất cả các file JSONL trong \output\ ở Script 04.")

if __name__ == "__main__":
    main()
