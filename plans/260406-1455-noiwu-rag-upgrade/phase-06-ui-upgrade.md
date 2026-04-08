# Phase 06: UI chuyên ngành Nội vụ
Status: ⬜ Pending
Dependencies: Phase 05

## Objective
Nâng cấp giao diện: navigation theo 10 lĩnh vực, câu hỏi gợi ý theo domain,
hiển thị metadata VB rõ ràng hơn, branding Nội vụ.

## Requirements
### Functional
- [ ] Sidebar: danh sách 10 lĩnh vực Nội vụ (có icon)
- [ ] Click lĩnh vực → filter search trong domain đó
- [ ] Câu hỏi gợi ý (suggestions) thay đổi theo lĩnh vực đang chọn
- [ ] Citation hiển thị: loại VB, số hiệu, cơ quan ban hành, ngày
- [ ] Header branding: "Trợ lý Pháp luật Nội vụ" thay vì "Legal RAG Chatbot"

## Implementation Steps
1. [ ] Sửa sidebar: thêm navigation 10 lĩnh vực
2. [ ] Sửa chat input: gửi kèm `domain` filter
3. [ ] Sửa suggestions: mỗi domain có bộ câu hỏi riêng
4. [ ] Sửa citation cards: hiển thị metadata đầy đủ
5. [ ] Cập nhật branding (header, favicon, title)

## Domain Icons & Suggestions

| Domain | Icon | Câu hỏi gợi ý |
|--------|------|----------------|
| Tổ chức bộ máy | 🏛️ | "Cơ cấu tổ chức bộ cơ quan ngang bộ?" |
| CQĐP | 🗺️ | "Mô hình chính quyền 2 cấp hoạt động thế nào?" |
| Cán bộ, CC, VC | 👤 | "Quy trình tuyển dụng công chức mới?" |
| Tiền lương | 💰 | "Bảng lương công chức mới nhất?" |
| CCHC | 📋 | "Mục tiêu CCHC giai đoạn 2021-2030?" |
| Thi đua KT | 🏆 | "Thẩm quyền tặng Bằng khen?" |
| Tôn giáo | ⛪ | "Điều kiện thành lập tổ chức tôn giáo?" |
| Văn thư LT | 📁 | "Thời hạn bảo quản hồ sơ lưu trữ?" |
| Thanh niên | 🧑 | "Chính sách hỗ trợ thanh niên khởi nghiệp?" |
| LĐ-TB&XH | 🛡️ | "Điều kiện hưởng BHXH một lần?" |

## Files to Modify
- `web/index.html` — UI chính

## Test Criteria
- [ ] 10 lĩnh vực hiển thị đúng trong sidebar
- [ ] Click lĩnh vực → suggested questions thay đổi
- [ ] Chat với domain filter → kết quả đúng lĩnh vực
- [ ] Responsive trên mobile (sidebar collapse)

---
Next Phase: → Phase 07 (Test & Deploy)
