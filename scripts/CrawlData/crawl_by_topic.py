"""
Crawl văn bản pháp luật từ thuvienphapluat.vn theo 9 đầu mục Bộ Nội vụ.

Cách dùng:
  python crawl_by_topic.py                  # Hiển thị menu chọn đầu mục
  python crawl_by_topic.py --topic 1        # Crawl đầu mục 1
  python crawl_by_topic.py --topic 2        # Crawl đầu mục 2
  python crawl_by_topic.py --topic all      # Crawl tất cả (lần lượt 1→9)
  python crawl_by_topic.py --list           # Liệt kê 9 đầu mục
  python crawl_by_topic.py --topic 1 --max 50   # Giới hạn 50 VB cho đầu mục 1
"""

import os
import re
import json
import time
import random
import logging
import argparse
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, StaleElementReferenceException
)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ──
OUTPUT_DIR = "output"
STATE_FILE = "crawl_state.json"
DELAY = 6                         # giây chờ giữa requests (cao hơn để tránh bị chặn)
DEFAULT_MAX = 100
BASE_URL = "https://thuvienphapluat.vn"
SEARCH_URL = BASE_URL + "/page/tim-van-ban.aspx"

# ── 9 Đầu mục Bộ Nội vụ ──
TOPICS = {
    1: {
        "name": "Tổ chức bộ máy, chính quyền địa phương",
        "short": "to-chuc-bo-may",
        "keywords": [
            "tổ chức bộ máy hành chính",
            "chính quyền địa phương",
            "đơn vị hành chính",
            "biên chế công chức",
            "tinh giản biên chế",
            "vị trí việc làm cơ quan nhà nước",
            "đơn vị sự nghiệp công lập",
            "sắp xếp đơn vị hành chính",
            "phân cấp phân quyền",
            "cơ cấu tổ chức bộ ngành",
            # ── Bổ sung ──
            "tổ chức bộ máy",
            "cơ cấu tổ chức",
            "vị trí việc làm",
            "biên chế viên chức",
            "sắp xếp tổ chức bộ máy",
            "hệ thống chính trị",
            "tổ chức hành chính nhà nước",
            "đơn vị hành chính cấp xã",
            "đơn vị hành chính cấp huyện",
            "đơn vị hành chính cấp tỉnh",
            "thành lập đơn vị hành chính",
            "nhập đơn vị hành chính",
            "chia tách đơn vị hành chính",
            "điều chỉnh địa giới hành chính",
            "mô hình chính quyền đô thị",
            "mô hình chính quyền nông thôn",
            "chính quyền địa phương 2 cấp",
            "quy chế làm việc Ủy ban nhân dân",
            "tổ chức Hội đồng nhân dân",
            "phân cấp quản lý nhà nước",
            "phân quyền trong quản lý hành chính",
            "Ủy ban nhân dân cấp xã",
            "Ủy ban nhân dân cấp huyện",
            "Ủy ban nhân dân cấp tỉnh",
            "cải cách hành chính",
            "tự chủ đơn vị sự nghiệp công",
            "cơ quan chuyên môn thuộc UBND",
            "tổ chức bên trong cơ quan hành chính",
            "quy hoạch mạng lưới đơn vị sự nghiệp",
            "kiện toàn tổ chức bộ máy",
            "kiểm soát tổ chức biên chế",
            "chức năng nhiệm vụ cơ quan hành chính",
            "quy định cơ cấu tổ chức bộ, ngành",
            "quy định cơ cấu Sở Nội vụ",
            "tổ chức phòng nội vụ",
            "ban quản lý khu kinh tế",
            "sắp xếp đơn vị hành chính cấp huyện, xã",
            "tiêu chuẩn đơn vị hành chính",
            "phân loại đơn vị hành chính",
            "khung số lượng lãnh đạo",
            "số lượng Phó Chủ tịch UBND",
            "số lượng Phó Giám đốc Sở",
            "quy hoạch đơn vị hành chính đô thị",
            "chính quyền địa phương đặc thù",
            "mô hình chính quyền đô thị thí điểm",
            "Ban Chỉ đạo sắp xếp đơn vị hành chính",
        ],
    },
    2: {
        "name": "Cán bộ, công chức, viên chức",
        "short": "can-bo-cong-chuc",
        "keywords": [
            "Luật cán bộ công chức",
            "Luật viên chức",
            "tuyển dụng công chức",
            "tuyển dụng viên chức",
            "đánh giá xếp loại công chức",
            "kỷ luật công chức viên chức",
            "nâng ngạch công chức",
            "bổ nhiệm miễn nhiệm",
            "quy hoạch cán bộ",
            "đào tạo bồi dưỡng công chức",
            "tiêu chuẩn chức danh",
            "luân chuyển biệt phái",
            # ── Bổ sung ──
            "cán bộ",
            "công chức",
            "viên chức",
            "chế độ công vụ",
            "thi tuyển công chức",
            "xét tuyển viên chức",
            "thi nâng ngạch công chức",
            "xét thăng hạng chức danh nghề nghiệp",
            "tập sự công chức",
            "tập sự viên chức",
            "đánh giá công chức",
            "đánh giá viên chức",
            "tiêu chuẩn ngạch công chức",
            "tiêu chuẩn chức danh nghề nghiệp",
            "bổ nhiệm cán bộ",
            "bổ nhiệm lại cán bộ",
            "miễn nhiệm cán bộ",
            "luân chuyển cán bộ",
            "điều động công chức",
            "biệt phái công chức",
            "kỷ luật công chức",
            "kỷ luật viên chức",
            "thôi việc công chức",
            "thôi việc viên chức",
            "nghỉ hưu trước tuổi",
            "chuyển loại công chức",
            "chuyển đổi vị trí công tác",
            "vị trí việc làm trong cơ quan hành chính",
            "cơ cấu ngạch công chức",
            "cơ cấu chức danh nghề nghiệp viên chức",
            "hồ sơ cán bộ, công chức",
            "hồ sơ viên chức",
            "tiền lương công chức",
            "phụ cấp công vụ",
            "phụ cấp thâm niên",
            "phụ cấp chức vụ lãnh đạo",
            "phụ cấp ưu đãi nghề",
            "chế độ bồi dưỡng bằng hiện vật",
            "đào tạo, bồi dưỡng cán bộ",
            "bồi dưỡng theo tiêu chuẩn chức danh",
            "chương trình bồi dưỡng CBCCVC",
            "chứng chỉ bồi dưỡng",
            "văn hóa công vụ",
            "đạo đức công vụ",
            "trách nhiệm người đứng đầu",
            "phòng chống tham nhũng trong công vụ",
            "kiểm tra công vụ",
        ],
    },
    3: {
        "name": "Lao động, việc làm, tiền lương, BHXH",
        "short": "lao-dong-tien-luong",
        "keywords": [
            "Bộ luật lao động",
            "hợp đồng lao động",
            "tiền lương cán bộ công chức",
            "mức lương cơ sở",
            "bảng lương công chức viên chức",
            "bảo hiểm xã hội",
            "bảo hiểm thất nghiệp",
            "an toàn vệ sinh lao động",
            "cải cách tiền lương",
            "phụ cấp công vụ",
            "chính sách việc làm",
            "xuất khẩu lao động",
            "quan hệ lao động",
            "tranh chấp lao động",
            # ── Bổ sung ──
            "kỷ luật lao động",
            "trách nhiệm vật chất",
            "đối thoại tại nơi làm việc",
            "thương lượng tập thể",
            "thỏa ước lao động tập thể",
            "đình công",
            "thị trường lao động",
            "trung tâm dịch vụ việc làm",
            "bảo hiểm xã hội bắt buộc",
            "bảo hiểm xã hội tự nguyện",
            "bảo hiểm tai nạn lao động",
            "bảo hiểm bệnh nghề nghiệp",
            "mức lương tối thiểu",
            "pháp luật tiền lương",
            "bảng lương mới",
            "phụ cấp lương",
            "chế độ tiền lương khu vực công",
            "tiền lương theo vị trí việc làm",
            "thang lương, bảng lương",
            "tiền làm thêm giờ",
            "thời giờ làm việc",
            "thời giờ nghỉ ngơi",
            "nghỉ phép năm",
            "an toàn lao động",
            "vệ sinh lao động",
            "tai nạn lao động",
            "bệnh nghề nghiệp",
            "danh mục nghề nặng nhọc, độc hại",
            "trang bị phương tiện bảo vệ cá nhân",
            "kiểm định kỹ thuật an toàn lao động",
            "huấn luyện an toàn vệ sinh lao động",
            "hồ sơ quốc gia về an toàn lao động",
            "chương trình quốc gia về an toàn lao động",
            "lao động nữ",
            "lao động chưa thành niên",
            "lao động cao tuổi",
            "lao động là người khuyết tật",
            "lao động giúp việc gia đình",
            "quản lý lao động ngoài nước",
            "tổ chức đại diện người lao động",
            "tổ chức đại diện người sử dụng lao động",
            "tiền lương trong doanh nghiệp nhà nước",
            "quản lý lao động, tiền lương, tiền thưởng doanh nghiệp nhà nước",
        ],
    },
    4: {
        "name": "Người có công với cách mạng",
        "short": "nguoi-co-cong",
        "keywords": [
            "người có công với cách mạng",
            "chế độ thương binh liệt sĩ",
            "Bà mẹ Việt Nam anh hùng",
            "ưu đãi người có công",
            "trợ cấp người có công",
            "Pháp lệnh ưu đãi người có công",
            "công nhận liệt sĩ",
            "chất độc hóa học",
            "điều dưỡng người có công",
            # ── Bổ sung ──
            "thương binh",
            "bệnh binh",
            "luân chuyển biệt phái",
            "cán bộ",
            "'pháp luật tiền lương",
            "bảng lương mới 2025",
            "ưu đãi người có công 2025",
            "thân nhân liệt sĩ 2025",
            "chế độ trợ cấp ưu đãi 2026",
            "trợ cấp hàng tháng 2025",
            "trợ cấp hàng tháng 2026",
            "quản lý hồ sơ người có công 2026",
            "truy tặng danh hiệu vinh dự Nhà nước 2025",
            "truy tặng danh hiệu vinh dự Nhà nước 2026"
            "bổ nhiệm lại cán bộ 2026",
            "luân chuyển cán bộ 2025",
            "kỷ luật công chức viên chức 2026",
            "liệt sĩ",
            "thân nhân liệt sĩ",
            "anh hùng lực lượng vũ trang",
            "anh hùng lao động thời kỳ kháng chiến",
            "người hoạt động cách mạng trước 1/1/1945",
            "người hoạt động cách mạng từ 1/1/1945 đến trước Tổng khởi nghĩa",
            "người hoạt động kháng chiến bị địch bắt tù đày",
            "người hoạt động kháng chiến bị nhiễm chất độc hóa học",
            "người có công giúp đỡ cách mạng",
            "chế độ trợ cấp ưu đãi",
            "trợ cấp hàng tháng",
            "trợ cấp một lần",
            "chế độ điều dưỡng",
            "chế độ bảo hiểm y tế người có công",
            "chế độ mai táng phí",
            "chế độ hỗ trợ nhà ở người có công",
            "quản lý hồ sơ người có công",
            "xác nhận người có công",
            "điều chỉnh chế độ người có công",
            "quỹ đền ơn đáp nghĩa",
            "chăm sóc thân nhân liệt sĩ",
            "chính sách ưu tiên giáo dục cho con em người có công",
            "vay vốn ưu đãi",
            "dạy nghề cho con em người có công",
            "phong tặng danh hiệu vinh dự Nhà nước",
            "truy tặng danh hiệu vinh dự Nhà nước",
            "cơ sở nuôi dưỡng người có công",
            "trung tâm điều dưỡng người có công",
            "công tác mộ liệt sĩ",
            "quy tập hài cốt liệt sĩ",
            "xác định danh tính hài cốt liệt sĩ",
            "quản lý nghĩa trang liệt sĩ",
            "lễ truy điệu, an táng liệt sĩ",
            "phong trào đền ơn đáp nghĩa",
            "chính sách ưu đãi thuế cho người có công",
            "ưu tiên trong tuyển dụng",
            "ưu tiên trong vay vốn sản xuất",
            "hỗ trợ cải thiện nhà ở",
            "chế độ BHXH cho người có công",
            "điều chỉnh mức trợ cấp ưu đãi",
            "thanh tra thực hiện chế độ người có công",
            "xử lý sai phạm trong xác nhận người có công",
            "ứng dụng CNTT quản lý người có công",
            "cơ sở dữ liệu quốc gia về người có công",
            "chương trình mục tiêu người có công",
            "chính sách người có công",
        ],
    },
    5: {
        "name": "Thanh niên, bình đẳng giới",
        "short": "thanh-nien-binh-dang",
        "keywords": [
            "Luật Thanh niên",
            "phát triển thanh niên",
            "chiến lược phát triển thanh niên",
            "bình đẳng giới",
            "Luật Bình đẳng giới",
            "lồng ghép bình đẳng giới",
            "chương trình quốc gia bình đẳng giới",
            "phòng chống bạo lực gia đình",
            # ── Bổ sung ──
            "chính sách thanh niên",
            "chương trình phát triển thanh niên",
            "quyền và nghĩa vụ của thanh niên",
            "tham gia của thanh niên vào quản lý nhà nước",
            "tư vấn, hỗ trợ thanh niên khởi nghiệp",
            "giáo dục lý tưởng cách mạng cho thanh niên",
            "phát triển nguồn nhân lực trẻ",
            "lao động thanh niên",
            "việc làm cho thanh niên",
            "đào tạo nghề cho thanh niên",
            "thanh niên tình nguyện",
            "thanh niên xung phong",
            "thanh niên nông thôn",
            "thanh niên đô thị",
            "thanh niên dân tộc thiểu số",
            "lồng ghép giới trong chính sách",
            "phòng chống bạo lực trên cơ sở giới",
            "bình đẳng giới trong lao động việc làm",
            "bình đẳng giới trong giáo dục",
            "bình đẳng giới trong y tế",
            "bình đẳng giới trong chính trị",
            "tỷ lệ nữ trong cấp ủy",
            "tỷ lệ nữ đại biểu Quốc hội, HĐND",
            "chiến lược quốc gia về bình đẳng giới",
            "thống kê giới",
            "chỉ số bình đẳng giới",
            "cơ chế phối hợp về bình đẳng giới",
            "lồng ghép giới trong lập pháp",
            "lồng ghép giới trong dự toán ngân sách",
            "nâng cao năng lực cán bộ về bình đẳng giới",
            "truyền thông về bình đẳng giới",
            "phòng chống bạo lực gia đình khía cạnh giới",
            "hỗ trợ phụ nữ khởi nghiệp",
            "thúc đẩy nữ lãnh đạo",
            "thanh tra, kiểm tra về bình đẳng giới",
            "xử lý vi phạm quy định bình đẳng giới",
            "cơ chế giám sát việc thực hiện bình đẳng giới",
            "tham vấn tổ chức xã hội về bình đẳng giới",
            "mạng lưới cộng tác viên bình đẳng giới",
            "báo cáo quốc gia về bình đẳng giới",
            "cơ sở dữ liệu giới tính",
            "chỉ số phát triển con người theo giới",
            "hỗ trợ nạn nhân bạo lực giới",
            "phòng chống định kiến giới",
            "thúc đẩy nam giới tham gia bình đẳng giới",
        ],
    },
    6: {
        "name": "Hội, quỹ, tổ chức phi chính phủ",
        "short": "hoi-quy-ngo",
        "keywords": [
            "thành lập hội",
            "quỹ xã hội quỹ từ thiện",
            "tổ chức phi chính phủ",
            "điều lệ hội",
            "quản lý hội quỹ",
            "tổ chức phi chính phủ nước ngoài",
            "đăng ký hoạt động hội",
            # ── Bổ sung ──
            "hội",
            "bình đẳng giới trong y tế",
            "tỷ lệ nữ trong cấp ủy",
            "bình đẳng giới trong chính trị",
            "tỷ lệ nữ trong cấp ủy",
            "tỷ lệ nữ đại biểu Quốc hội",
            "chiến lược quốc gia về bình đẳng giới",
            "thống kê giới",
            "chỉ số bình đẳng giới",
            "cơ chế phối hợp về bình đẳng giới",
            "cộng tác viên bình đẳng giới",
            "hội nghề nghiệp",
            "hội xã hội nghề nghiệp",
            "hội đặc thù",
            "quỹ xã hội",
            "quỹ từ thiện",
            "NGO nước ngoài tại Việt Nam",
            "tổ chức phi lợi nhuận",
            "công nhận điều lệ hội",
            "chia, tách, sáp nhập hội",
            "giải thể hội",
            "đăng ký quỹ xã hội, quỹ từ thiện",
            "cấp phép tổ chức phi chính phủ nước ngoài",
            "chấm dứt hoạt động tổ chức phi chính phủ nước ngoài",
            "quản lý quỹ xã hội, quỹ từ thiện",
            "công khai tài chính quỹ từ thiện",
            "báo cáo hoạt động hội",
            "báo cáo hoạt động quỹ",
            "thanh tra hội",
            "thanh tra quỹ từ thiện",
            "xử lý vi phạm về hội",
            "xử lý vi phạm về quỹ từ thiện",
            "quyền lập hội",
            "quyền tham gia hội",
            "tổ chức hội ở trung ương",
            "tổ chức hội ở địa phương",
            "hiệp hội doanh nghiệp",
            "liên hiệp hội",
            "mạng lưới các tổ chức phi chính phủ",
            "viện trợ phi chính phủ",
            "thỏa thuận viện trợ",
            "giám sát sử dụng viện trợ",
            "quy chế phối hợp quản lý NGO",
            "trách nhiệm giải trình của hội",
            "tự chủ tài chính hội",
            "vận động tài trợ",
            "gây quỹ từ thiện",
            "bảo trợ xã hội qua quỹ từ thiện",
            "trách nhiệm của tổ chức phi chính phủ",
            "chấm dứt tư cách thành viên hội",
            "điều chỉnh điều lệ hội",
            "đại hội hội viên",
            "Ban Chấp hành hội",
            "chi hội, tổ hội",
            "hội đặc thù do ngân sách hỗ trợ",
            "cơ quan đầu mối quản lý hội, quỹ, NGO",
        ],
    },
    7: {
        "name": "Văn thư, lưu trữ nhà nước",
        "short": "van-thu-luu-tru",
        "keywords": [
            "Luật Lưu trữ",
            "văn thư lưu trữ",
            "soạn thảo văn bản hành chính",
            "thể thức văn bản",
            "quản lý con dấu",
            "lưu trữ lịch sử",
            "số hóa tài liệu lưu trữ",
            "thời hạn bảo quản tài liệu",
            "quản lý tài liệu điện tử",
            # ── Bổ sung ──
            "công tác văn thư",
            "văn bản hành chính",
            "kỹ thuật trình bày văn bản",
            "ban hành văn bản",
            "quản lý văn bản đến",
            "quản lý văn bản đi",
            "sổ đăng ký văn bản",
            "hồ sơ công việc",
            "lập hồ sơ",
            "nộp lưu hồ sơ",
            "con dấu",
            "quản lý và sử dụng con dấu",
            "sao y văn bản hành chính",
            "chứng thực bản sao",
            "tài liệu điện tử",
            "hệ thống quản lý tài liệu điện tử",
            "chữ ký số",
            "văn bản điện tử",
            "lưu trữ cơ quan",
            "thu thập tài liệu lưu trữ",
            "phân loại tài liệu lưu trữ",
            "chỉnh lý tài liệu lưu trữ",
            "bảo quản tài liệu lưu trữ",
            "tiêu hủy tài liệu hết giá trị",
            "thống kê tài liệu lưu trữ",
            "khai thác sử dụng tài liệu lưu trữ",
            "đọc, sao tài liệu lưu trữ",
            "bảo mật tài liệu lưu trữ",
            "danh mục thành phần tài liệu nộp lưu",
            "tài liệu lưu trữ điện tử",
            "kho lưu trữ",
            "cán bộ văn thư",
            "cán bộ lưu trữ",
            "tiêu chuẩn chức danh văn thư",
            "tiêu chuẩn chức danh lưu trữ",
            "đào tạo văn thư – lưu trữ",
            "thanh tra công tác văn thư",
            "thanh tra công tác lưu trữ",
            "hướng dẫn nghiệp vụ văn thư",
            "hướng dẫn nghiệp vụ lưu trữ",
            "quy chế công tác văn thư",
            "quy chế công tác lưu trữ",
            "kiểm tra văn thư, lưu trữ",
            "ứng dụng CNTT trong văn thư",
            "ứng dụng CNTT trong lưu trữ",
            "cơ sở dữ liệu tài liệu lưu trữ",
        ],
    },
    8: {
        "name": "Thi đua, khen thưởng",
        "short": "thi-dua-khen-thuong",
        "keywords": [
            "Luật Thi đua khen thưởng",
            "danh hiệu thi đua",
            "hình thức khen thưởng",
            "Huân chương Lao động",
            "Bằng khen Thủ tướng",
            "phong trào thi đua",
            "hội đồng thi đua khen thưởng",
            "khen thưởng đột xuất",
            # ── Bổ sung ──
            "Luật Thi đua, khen thưởng",
            "chiến sĩ thi đua",
            "chiến sĩ thi đua cơ sở",
            "chiến sĩ thi đua cấp bộ, ngành, tỉnh",
            "chiến sĩ thi đua toàn quốc",
            "lao động tiên tiến",
            "tập thể lao động tiên tiến",
            "cờ thi đua của Chính phủ",
            "cờ thi đua của bộ, ngành, tỉnh",
            "bằng khen bộ, ngành, tỉnh",
            "huân chương",
            "huy chương",
            "danh hiệu vinh dự Nhà nước",
            "đối tượng khen thưởng",
            "tiêu chuẩn khen thưởng",
            "hồ sơ khen thưởng",
            "thủ tục khen thưởng",
            "thẩm quyền khen thưởng",
            "quỹ thi đua, khen thưởng",
            "điểm thưởng thành tích",
            "xét tặng danh hiệu thi đua",
            "truy tặng danh hiệu thi đua",
            "khen thưởng thành tích đột xuất",
            "khen thưởng thành tích theo chuyên đề",
            "khen thưởng quá trình cống hiến",
            "khen thưởng tập thể nhỏ",
            "khen thưởng cá nhân",
            "khen thưởng doanh nghiệp",
            "khen thưởng tổ chức đảng, đoàn thể",
            "phong trào thi đua yêu nước",
            "đăng ký thi đua",
            "giao ước thi đua",
            "tổng kết phong trào thi đua",
            "nhân rộng điển hình tiên tiến",
            "kiểm tra công tác thi đua, khen thưởng",
            "xử lý vi phạm trong thi đua, khen thưởng",
            "công khai minh bạch khen thưởng",
            "phân cấp khen thưởng",
            "phân quyền trong thi đua, khen thưởng",
            "báo cáo thống kê thi đua, khen thưởng",
            "ứng dụng CNTT trong thi đua, khen thưởng",
            "hồ sơ điện tử thi đua, khen thưởng",
            "cơ sở dữ liệu thi đua, khen thưởng",
            "thi đua cấp cơ sở",
            "thi đua cấp bộ, ngành, tỉnh",
            "thi đua theo cụm, khối",
        ],
    },
    9: {
        "name": "Dịch vụ công lĩnh vực nội vụ",
        "short": "dich-vu-cong",
        "keywords": [
            "dịch vụ sự nghiệp công",
            "đào tạo bồi dưỡng cán bộ",
            "dịch vụ lưu trữ công",
            "cơ chế tự chủ đơn vị sự nghiệp",
            "xã hội hóa dịch vụ công",
            "đấu thầu dịch vụ công",
            "dịch vụ việc làm công",
            "thủ tục hành chính Bộ Nội vụ",
            "cải cách hành chính",
            "một cửa liên thông",
            # ── Bổ sung ──
            "đơn vị sự nghiệp công lập",
            "tự chủ tài chính đơn vị sự nghiệp",
            "đặt hàng cung cấp dịch vụ công",
            "đấu thầu dịch vụ sự nghiệp công",
            "giá dịch vụ sự nghiệp công",
            "khung giá dịch vụ sự nghiệp công",
            "dịch vụ đào tạo, bồi dưỡng CBCCVC",
            "dịch vụ văn thư",
            "dịch vụ lưu trữ",
            "dịch vụ tư vấn tổ chức bộ máy",
            "dịch vụ tư vấn tiền lương",
            "dịch vụ tư vấn nhân sự công",
            "trung tâm dịch vụ việc làm công lập",
            "tổ chức cung ứng nguồn lao động",
            "dịch vụ giới thiệu việc làm",
            "dịch vụ tư vấn an toàn lao động",
            "dịch vụ kiểm định kỹ thuật an toàn",
            "xã hội hóa dịch vụ sự nghiệp công",
            "thực hiện cơ chế một cửa",
            "bộ phận một cửa",
            "dịch vụ công trực tuyến",
            "cung cấp dịch vụ công trực tuyến mức độ 4",
            "thủ tục hành chính lĩnh vực nội vụ",
            "đơn giản hóa thủ tục hành chính",
            "chuẩn hóa quy trình giải quyết TTHC",
            "đánh giá mức độ hài lòng người dân",
            "tiêu chí chất lượng dịch vụ công",
            "chuẩn đầu ra dịch vụ đào tạo bồi dưỡng",
            "tiêu chuẩn dịch vụ lưu trữ tài liệu",
            "hợp đồng dịch vụ sự nghiệp công",
            "kiểm tra chất lượng dịch vụ công",
            "giám sát cung cấp dịch vụ công",
            "báo cáo hoạt động dịch vụ công",
            "công khai tài chính đơn vị sự nghiệp",
            "khoán chi dịch vụ công",
            "thí điểm tự chủ dịch vụ công",
            "chuyển đổi đơn vị sự nghiệp thành công ty cổ phần",
            "mạng lưới đơn vị sự nghiệp ngành nội vụ",
            "ứng dụng CNTT trong cung cấp dịch vụ công",
            "số hóa quy trình dịch vụ nội vụ",
            "tích hợp dịch vụ công trên cổng dịch vụ công quốc gia",
            "đánh giá xếp hạng cải cách hành chính",
            "chỉ số PAR INDEX",
            "cung ứng dịch vụ công",
            "cơ chế đấu thầu dịch vụ công",
            "sự nghiệp công ngành nội vụ",
        ],
    },
}


# ── State management ──
def load_state() -> dict:
    """Load trạng thái crawl từ file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"visited_urls": [], "topic_counts": {}, "total": 0}


def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def append_record(record: dict, topic_id: int):
    """Lưu 1 văn bản vào file JSONL riêng theo topic."""
    topic = TOPICS[topic_id]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"topic_{topic_id:02d}_{topic['short']}.jsonl"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return filename


# ── Browser (anti-bot) ──
def make_driver(use_debug: bool = False) -> webdriver.Chrome:
    opts = Options()
    if use_debug:
        opts.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
        try:
            driver = webdriver.Chrome(options=opts)
            log.info("✅ Đã kết nối Chrome debug (port 9222)")
            return driver
        except Exception as e:
            log.error("❌ Không kết nối được Chrome debug: %s", e)
            log.error("   (Vui lòng mở Chrome qua cmd: chrome.exe --remote-debugging-port=9222 ...)")
            raise

    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1400,900")

    # ── Anti-bot: ẩn dấu vết Selenium ──
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    )

    # Tắt hình ảnh để tăng tốc
    prefs = {"profile.managed_default_content_settings.images": 2}
    opts.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=opts)

    # Xóa dấu vết webdriver
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.navigator.chrome = {runtime: {}};
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['vi-VN', 'vi', 'en-US', 'en']});
        """
    })

    return driver


def warmup_session(driver: webdriver.Chrome):
    """
    Vào trang chủ TVPL trước để lấy cookies + bypass Cloudflare.
    Bắt buộc phải làm để các request sau không bị chặn.
    """
    log.info("🌐 Khởi tạo session (vào trang chủ TVPL)...")
    driver.get(BASE_URL)
    time.sleep(3)

    # Kiểm tra Cloudflare challenge
    for attempt in range(3):
        page_source = driver.page_source.lower()
        if "verify you are human" in page_source or "cloudflare" in page_source:
            log.warning("⚠️ Cloudflare challenge detected, chờ %ds...", 5 * (attempt + 1))
            time.sleep(5 * (attempt + 1))
        else:
            break

    # Kiểm tra đã vào được trang chủ chưa
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#txtKeyWord, input[name='keyword'], .search-input"))
        )
        log.info("✅ Session OK — đã vào trang chủ TVPL")
    except TimeoutException:
        log.warning("⚠️ Không tìm thấy ô tìm kiếm, thử tiếp tục...")


# ── Search ──
def search_by_keyword(driver: webdriver.Chrome, keyword: str, max_pages: int = 3) -> list[str]:
    """
    Tìm kiếm VB trên TVPL bằng cách gõ keyword vào ô search.
    Trả về list URLs văn bản.
    """
    urls = []

    # ── Cách 1: Dùng URL search (đã có session từ warmup) ──
    encoded_kw = quote(keyword)
    search_url = f"{SEARCH_URL}?keyword={encoded_kw}&match=True&area=0&type=0&page=1"

    try:
        driver.get(search_url)
        time.sleep(3)

        # Chờ kết quả load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "p.vblist, .nqTitle, .doc-title, .content-area"))
        )
    except TimeoutException:
        # ── Cách 2 (fallback): Gõ trực tiếp vào ô tìm kiếm ──
        log.debug("  Fallback: gõ keyword vào ô tìm kiếm")
        try:
            driver.get(BASE_URL)
            time.sleep(2)
            search_input = WebDriverWait(driver, 8).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "#txtKeyWord, input[name='keyword']"))
            )
            search_input.clear()
            search_input.send_keys(keyword)
            time.sleep(0.5)
            search_input.send_keys(Keys.RETURN)
            time.sleep(3)
        except Exception as e:
            log.warning("  ❌ Không thể search '%s': %s", keyword[:30], e)
            return urls

    # ── Thu thập URLs từ tất cả các trang ──
    for page in range(1, max_pages + 1):
        if page > 1:
            # Navigate sang trang tiếp
            next_url = f"{SEARCH_URL}?keyword={encoded_kw}&match=True&area=0&type=0&page={page}"
            try:
                driver.get(next_url)
                time.sleep(3)
                WebDriverWait(driver, 8).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "p.vblist, .nqTitle, .content-area"))
                )
            except TimeoutException:
                break

        # ── Lấy links VB — thử nhiều CSS selectors ──
        found_on_page = 0
        selectors_to_try = [
            "p.vblist > a",           # Selector chính trên TVPL (confirmed từ screenshot)
            "p.vblist a",             # Backup
            ".nqTitle a",             # Format cũ
            ".doc-title a",           # Format khác
            "a[href*='/van-ban/']",   # Fallback: tất cả link VB
        ]

        for sel in selectors_to_try:
            try:
                links = driver.find_elements(By.CSS_SELECTOR, sel)
                for a in links:
                    try:
                        href = a.get_attribute("href")
                        if href and "/van-ban/" in href and ".aspx" in href:
                            clean = href.split("?")[0].split("#")[0]
                            if clean not in urls:
                                urls.append(clean)
                                found_on_page += 1
                    except StaleElementReferenceException:
                        continue
            except Exception:
                continue

            if found_on_page > 0:
                break  # Đã tìm thấy bằng selector này

        if found_on_page == 0:
            log.debug("  Trang %d: không tìm thấy kết quả", page)
            break

        time.sleep(random.uniform(max(1, DELAY - 2), DELAY + 2))

    return urls


# ── Scraping ──
def scrape_document(driver: webdriver.Chrome, url: str) -> dict | None:
    """Scrape nội dung 1 văn bản pháp luật."""
    try:
        driver.get(url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#tab1, .content1, .main-content"))
        )
        time.sleep(1.5)
    except TimeoutException:
        log.warning("⏳ Timeout: %s", url[:80])
        return None

    record = {"url": url}

    # Title — thử nhiều selector
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

    # Số hiệu — trích từ title hoặc URL
    record["document_number"] = ""
    if record["title"]:
        m = re.search(r'[Ss]ố[:\s]*(\d+/\d{4}/[\w-]+)', record["title"])
        if m:
            record["document_number"] = m.group(1)
    # Fallback: trích từ URL
    if not record["document_number"]:
        m = re.search(r'/([A-Za-z-]+-\d+-\d{4}-[A-Za-z-]+)', url)
        if m:
            record["document_number"] = m.group(1)

    # Nội dung chính
    content = ""
    for sel in ["#tab1 .content1", "#tab1.contentDoc", "#tab1 .contentDoc", "#tab1", ".noidung-vanban"]:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            content = el.text.strip()
            if len(content) > 200:
                break
        except NoSuchElementException:
            continue

    # Strip noise: bỏ phần "Bài liên quan", "Hỏi đáp", footer
    noise_markers = [
        "Lưu trữ\nGhi chú\nÝ kiến",
        "Bài liên quan:",
        "Hỏi đáp pháp luật",
        "Bản án liên quan",
        "Facebook\nEmail\nIn",
    ]
    for marker in noise_markers:
        idx = content.find(marker)
        if idx > 0:
            content = content[:idx].rstrip()

    record["content"] = content

    # Metadata panel
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
                    ("Ban hành:", "ngay_ban_hanh"),
                    ("Hiệu lực:", "ngay_hieu_luc"),
                    ("Tình trạng:", "tinh_trang"),
                    ("Cập nhật:", "cap_nhat"),
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


# ── Main crawler ──
def crawl_topic(topic_id: int, max_docs: int = DEFAULT_MAX, resume: bool = False, use_debug: bool = False, from_2025: bool = False):
    """Crawl 1 đầu mục cụ thể.
    
    Args:
        topic_id: Số đầu mục (1-9)
        max_docs: Số VB tối đa
        resume: Nếu True, bỏ qua search và dùng pending_urls đã lưu
        use_debug: Dùng Chrome Debug port 9222 để bypass bot protection
        from_2025: Bóp từ khóa để chỉ tìm kiếm văn bản từ năm 2025
    """
    if topic_id not in TOPICS:
        log.error("❌ Đầu mục %d không tồn tại (1-9)", topic_id)
        return

    topic = TOPICS[topic_id]
    state = load_state()
    visited = set(state.get("visited_urls", []))
    topic_key = str(topic_id)

    log.info("=" * 60)
    log.info("📂 ĐẦU MỤC %d: %s", topic_id, topic["name"])
    log.info("   Keywords: %d", len(topic["keywords"]))
    log.info("   Max: %d VB | Đã crawl: %d VB", max_docs, state.get("topic_counts", {}).get(topic_key, 0))
    if resume:
        log.info("   ⏩ CHẾ ĐỘ RESUME — bỏ qua search, dùng URLs đã lưu")
    log.info("=" * 60)

    driver = make_driver(use_debug)
    count = 0

    try:
        # ── Bước 0: Warmup session (bypass Cloudflare) ──
        warmup_session(driver)

        # ── Phase 1: Thu thập URLs ──
        if resume:
            # Load pending URLs từ state
            pending = state.get("pending_urls", {})
            all_urls = pending.get(topic_key, [])
            # Loại bỏ URLs đã visited
            all_urls = [u for u in all_urls if u not in visited]
            if all_urls:
                log.info("📋 Resume: %d URLs còn lại (đã loại %d visited)",
                         len(all_urls), len(pending.get(topic_key, [])) - len(all_urls))
            else:
                log.warning("⚠️ Không có pending URLs cho đầu mục %d!", topic_id)
                log.warning("   Chạy lại KHÔNG có --resume để search trước.")
                return
        else:
            # Search bình thường
            all_urls = []
            for i, keyword in enumerate(topic["keywords"]):
                if from_2025:
                    kw_list = [keyword + " 2025 2026"]
                else:
                    kw_list = [keyword]

                for kw in kw_list:
                    log.info("🔍 [%d/%d] Tìm: '%s'", i + 1, len(topic["keywords"]), kw[:50])
                    urls = search_by_keyword(driver, kw, max_pages=3)
                    new_urls = [u for u in urls if u not in visited and u not in all_urls]
                    all_urls.extend(new_urls)
                    log.info("   → %d kết quả (%d mới, tổng queue: %d)", len(urls), len(new_urls), len(all_urls))
                    time.sleep(random.uniform(max(1, DELAY - 2), DELAY + 2))

            # ── Lưu pending URLs vào state ──
            if "pending_urls" not in state:
                state["pending_urls"] = {}
            state["pending_urls"][topic_key] = all_urls
            save_state(state)
            log.info("💾 Đã lưu %d pending URLs (dùng --resume để tiếp tục sau)", len(all_urls))

        log.info("=" * 40)
        log.info("📋 Tổng URLs cần crawl: %d", len(all_urls))
        log.info("=" * 40)

        if not all_urls:
            log.warning("⚠️ Không tìm thấy URL nào! Có thể bị Cloudflare chặn.")
            log.warning("   Thử chạy lại hoặc dùng '--resume' nếu đã search trước đó.")
            return

        # ── Phase 2: Scrape từng URL ──
        for i, url in enumerate(all_urls):
            if count >= max_docs:
                log.info("✅ Đạt giới hạn %d VB.", max_docs)
                break

            if url in visited:
                continue

            log.info("[%d/%d] 📄 %s", count + 1, min(len(all_urls), max_docs), url[:80])
            record = scrape_document(driver, url)

            if record and record.get("content") and len(record["content"]) > 100:
                record["topic_id"] = topic_id
                record["topic_name"] = topic["name"]

                filename = append_record(record, topic_id)
                count += 1

                title_preview = record.get("title", "")[:60] or record.get("document_number", "") or "(?)"
                log.info("   ✅ %s [%d ký tự]", title_preview, len(record["content"]))
            else:
                log.warning("   ⚠️ Bỏ qua (nội dung rỗng/ngắn)")

            # Luôn đánh dấu đã visit
            visited.add(url)
            state["visited_urls"] = list(visited)
            state["topic_counts"] = state.get("topic_counts", {})
            state["topic_counts"][topic_key] = state["topic_counts"].get(topic_key, 0) + (1 if record and record.get("content") and len(record["content"]) > 100 else 0)
            state["total"] = sum(state["topic_counts"].values())

            # Cập nhật pending: xóa URL đã xử lý
            if "pending_urls" in state and topic_key in state["pending_urls"]:
                try:
                    state["pending_urls"][topic_key].remove(url)
                except ValueError:
                    pass

            save_state(state)
            time.sleep(random.uniform(max(1, DELAY - 2), DELAY + 2))

    except KeyboardInterrupt:
        log.info("\n⏸️ Dừng bởi người dùng. Đã lưu trạng thái.")
        log.info("   👉 Dùng '--resume' để tiếp tục: python crawl_by_topic.py --topic %d --resume", topic_id)
    finally:
        driver.quit()

    log.info("=" * 60)
    log.info("✅ Hoàn thành đầu mục %d: Crawl %d VB mới", topic_id, count)
    log.info("   File: output/topic_%02d_%s.jsonl", topic_id, topic["short"])
    log.info("   Tổng VB đã crawl: %d", state.get("total", 0))
    remaining = len(state.get("pending_urls", {}).get(topic_key, []))
    if remaining > 0:
        log.info("   📌 Còn %d URLs chưa crawl (dùng --resume để tiếp)", remaining)
    log.info("=" * 60)


def print_menu():
    """Hiển thị menu 9 đầu mục."""
    state = load_state()
    counts = state.get("topic_counts", {})

    print("\n" + "=" * 60)
    print("  🏛️  CRAWL VĂN BẢN PHÁP LUẬT — BỘ NỘI VỤ")
    print("=" * 60)
    for tid, topic in TOPICS.items():
        done = counts.get(str(tid), 0)
        status = f"({done} VB)" if done > 0 else ""
        print(f"  [{tid}] {topic['name']}  {status}")
    print(f"\n  Tổng đã crawl: {state.get('total', 0)} VB")
    print("=" * 60)
    # Hiển thị pending URLs
    pending = state.get("pending_urls", {})
    has_pending = False
    for tid_str, urls in pending.items():
        if urls:
            print(f"  ⏳ Đầu mục {tid_str}: {len(urls)} URLs chờ crawl")
            has_pending = True

    print("\n  Cách dùng:")
    print("    python crawl_by_topic.py --topic 1             # Search + Crawl")
    print("    python crawl_by_topic.py --topic 1 --resume    # Bỏ qua search, crawl tiếp")
    print("    python crawl_by_topic.py --topic 1 --debug     # Bypass CF /w Chrome 9222")
    print("    python crawl_by_topic.py --topic all           # Crawl tất cả")
    print("    python crawl_by_topic.py --topic 2 --max 50")
    if has_pending:
        print("\n  💡 Có URLs đang chờ! Dùng --resume để tiếp tục crawl.")
    print("=" * 60)


# ── CLI ──
def main():
    parser = argparse.ArgumentParser(description="Crawl VB pháp luật theo đầu mục Bộ Nội vụ")
    parser.add_argument("--topic", type=str, help="Số đầu mục (1-9) hoặc 'all'")
    parser.add_argument("--max", type=int, default=DEFAULT_MAX, help=f"Số VB tối đa mỗi đầu mục (mặc định {DEFAULT_MAX})")
    parser.add_argument("--list", action="store_true", help="Liệt kê 9 đầu mục")
    parser.add_argument("--resume", action="store_true", help="Bỏ qua search, dùng URLs đã lưu từ lần chạy trước")
    parser.add_argument("--debug", action="store_true", help="Dùng Chrome thật qua port 9222 để bypass Cloudflare")
    parser.add_argument("--from-2025", action="store_true", help="Chỉ search các văn bản mới ban hành từ 2025")
    parser.add_argument("--delay", type=float, help="Thời gian chờ giữa các request (giây)")
    args = parser.parse_args()

    global DELAY
    if args.delay is not None:
        DELAY = args.delay

    if args.list or args.topic is None:
        print_menu()
        return

    if args.topic.lower() == "all":
        for tid in range(1, 10):
            crawl_topic(tid, args.max, resume=args.resume, use_debug=args.debug, from_2025=args.from_2025)
            log.info("⏳ Chờ 5s trước đầu mục tiếp...\n")
            time.sleep(10)
    else:
        try:
            tid = int(args.topic)
            crawl_topic(tid, args.max, resume=args.resume, use_debug=args.debug, from_2025=args.from_2025)
        except ValueError:
            log.error("❌ --topic phải là số 1-9 hoặc 'all'")


if __name__ == "__main__":
    main()
