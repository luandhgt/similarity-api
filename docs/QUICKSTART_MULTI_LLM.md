# Quick Start - Multi-LLM Provider

Hướng dẫn nhanh để chuyển đổi giữa Claude, ChatGPT, Gemini trong 3 bước đơn giản.

---

## 🚀 Cách sử dụng (3 bước)

### Bước 1: Cài đặt dependencies

```bash
pip install openai>=2.0.0
```

### Bước 2: Cấu hình API Keys trong `.env`

```bash
# Chọn provider: "claude", "chatgpt", hoặc "gemini"
LLM_PROVIDER=claude

# API Keys (chỉ cần provider bạn dùng)
CLAUDE_API_KEY=sk-ant-api03-xxxxx
OPENAI_API_KEY=sk-proj-xxxxx
GEMINI_API_KEY=your-gemini-key
```

### Bước 3: Khởi động lại server

```bash
uvicorn main:app --reload
```

**Xong! 🎉** Hệ thống sẽ tự động sử dụng provider bạn đã chọn.

---

## 🔄 Chuyển đổi Provider

### Trong file `.env`:

```bash
# Sử dụng Claude (chính xác, context lớn)
LLM_PROVIDER=claude

# Sử dụng ChatGPT (nhanh, rẻ hơn)
LLM_PROVIDER=chatgpt

# Sử dụng Gemini (coming soon)
LLM_PROVIDER=gemini
```

### Hoặc khi chạy:

```bash
# Development với ChatGPT (rẻ hơn)
LLM_PROVIDER=chatgpt uvicorn main:app --reload

# Production với Claude (chất lượng cao)
LLM_PROVIDER=claude uvicorn main:app --workers 4
```

---

## ⚙️ Configuration mẫu

### Development (ChatGPT - tiết kiệm chi phí)

```bash
# .env.development
LLM_PROVIDER=chatgpt
OPENAI_API_KEY=sk-proj-xxxxx
OPENAI_MODEL=gpt-4o
OPENAI_MAX_TOKENS=16384
```

### Production (Claude - chất lượng cao)

```bash
# .env.production
LLM_PROVIDER=claude
CLAUDE_API_KEY=sk-ant-xxxxx
CLAUDE_MODEL=claude-sonnet-4-5-20250929
CLAUDE_MAX_TOKENS=8000
```

---

## 📝 Code Example

**Không cần thay đổi code!** Provider được tự động inject:

```python
from core.container import get_container, ServiceNames

# Get current provider (tự động theo config)
container = get_container()
llm = container.get(ServiceNames.CLAUDE)

# Sử dụng như bình thường
response = await llm.generate_text("Hello AI!")

# Phân tích ảnh
result = await llm.analyze_image(
    image_path="screenshot.jpg",
    prompt="Extract text from this image"
)

# Tổng hợp nhiều ảnh + text
summary = await llm.synthesize_with_images_and_texts(
    image_paths=["img1.jpg", "img2.jpg"],
    texts=["OCR 1", "OCR 2"],
    user_prompt="Create summary"
)
```

---

## ✅ Kiểm tra Provider đang dùng

```bash
# Check trong logs
tail -f logs/image-similarity-api.log

# Hoặc check programmatically
python -c "from config import Config; print(Config.LLM_PROVIDER)"
```

---

## 🎯 Khi nào dùng Provider nào?

| Use Case | Recommended Provider | Lý do |
|----------|---------------------|-------|
| **Development/Testing** | ChatGPT | Rẻ hơn, nhanh hơn |
| **Production (High Quality)** | Claude | Chính xác hơn, context lớn |
| **High Throughput** | ChatGPT | Rate limits thoáng hơn |
| **Tiếng Việt** | Claude | Hỗ trợ tốt hơn |
| **Budget Limited** | ChatGPT | Chi phí thấp hơn |

---

## 🔧 Troubleshooting

### Lỗi: API key không hợp lệ

```bash
# Kiểm tra API key
echo $CLAUDE_API_KEY
echo $OPENAI_API_KEY

# Test API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

### Lỗi: Model không support vision

**Solution:** Dùng models có vision:
- Claude: Tất cả Claude 3+ models
- OpenAI: `gpt-4o`, `gpt-4-vision-preview`, `gpt-4-turbo`

---

## 📚 Đọc thêm

- [Full Documentation](./MULTI_LLM_PROVIDER.md) - Chi tiết đầy đủ
- [API Reference](./MULTI_LLM_PROVIDER.md#api-reference) - Tất cả methods
- [Architecture](./MULTI_LLM_PROVIDER.md#architecture) - Thiết kế hệ thống

---

## 💡 Tips

1. **Development**: Dùng ChatGPT để tiết kiệm chi phí
2. **Production**: Dùng Claude cho chất lượng cao
3. **Monitor logs**: Để track provider nào đang được dùng
4. **Test cả 2**: So sánh kết quả để chọn provider phù hợp

---

Đã xong! Bạn có thể dễ dàng chuyển đổi giữa các AI provider chỉ bằng 1 dòng config. 🎉
