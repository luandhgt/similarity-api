# Multi-LLM Provider Support

Hướng dẫn sử dụng hệ thống Multi-LLM Provider cho phép dễ dàng chuyển đổi giữa các chatbot khác nhau (Claude, ChatGPT, Gemini, v.v.)

## Tổng quan

Hệ thống Multi-LLM Provider sử dụng **Strategy Pattern** để cho phép chuyển đổi linh hoạt giữa các nhà cung cấp AI khác nhau mà không cần thay đổi code logic nghiệp vụ.

### Các Provider hiện tại được hỗ trợ:

| Provider | Model mặc định | Vision Support | API Key Required |
|----------|----------------|----------------|------------------|
| **Claude** (Anthropic) | claude-sonnet-4-5-20250929 | ✅ Yes | `CLAUDE_API_KEY` |
| **ChatGPT** (OpenAI) | gpt-4o | ✅ Yes | `OPENAI_API_KEY` |
| **Gemini** (Google) | gemini-pro | 🚧 Coming Soon | `GEMINI_API_KEY` |

---

## Cấu hình

### 1. Cài đặt dependencies

Thêm vào `requirements.txt`:

```txt
# Existing
anthropic>=0.18.0
aiohttp>=3.9.0

# New for OpenAI
openai>=2.0.0
```

Cài đặt:

```bash
pip install -r requirements.txt
```

### 2. Cấu hình trong file `.env`

Mở file `.env.development` hoặc `.env.production` và cấu hình:

```bash
# =============================================================================
# LLM PROVIDER SELECTION
# =============================================================================
# Choose one: "claude", "chatgpt", or "gemini"
LLM_PROVIDER=claude

# =============================================================================
# API KEYS
# =============================================================================
# Claude (Anthropic)
CLAUDE_API_KEY=sk-ant-api03-xxxxx

# OpenAI (ChatGPT)
OPENAI_API_KEY=sk-proj-xxxxx

# Google (Gemini)
GEMINI_API_KEY=your-gemini-key-here

# =============================================================================
# PROVIDER-SPECIFIC CONFIGURATION
# =============================================================================
# Claude Settings
CLAUDE_MODEL=claude-sonnet-4-5-20250929
CLAUDE_MAX_TOKENS=8000
CLAUDE_TEMPERATURE=0.7
CLAUDE_TIMEOUT=300

# OpenAI Settings
OPENAI_MODEL=gpt-4o
OPENAI_MAX_TOKENS=16384
OPENAI_TEMPERATURE=0.7
OPENAI_TIMEOUT=300

# Gemini Settings
GEMINI_MODEL=gemini-pro
GEMINI_MAX_TOKENS=4096
GEMINI_TEMPERATURE=0.7
GEMINI_TIMEOUT=300
```

---

## Cách sử dụng

### Chuyển đổi giữa các Provider

#### Cách 1: Thay đổi trong file `.env`

```bash
# Sử dụng Claude
LLM_PROVIDER=claude

# Sử dụng ChatGPT
LLM_PROVIDER=chatgpt

# Sử dụng Gemini (khi available)
LLM_PROVIDER=gemini
```

Sau đó restart server:

```bash
uvicorn main:app --reload
```

#### Cách 2: Set environment variable khi chạy

```bash
# Sử dụng Claude
LLM_PROVIDER=claude uvicorn main:app --reload

# Sử dụng ChatGPT
LLM_PROVIDER=chatgpt uvicorn main:app --reload
```

### Code Example - Sử dụng trong Service

Provider được tự động inject thông qua ServiceContainer. Bạn không cần thay đổi gì trong code:

```python
from core.container import get_container, ServiceNames

# Get LLM provider (automatically selected based on config)
container = get_container()
llm_provider = container.get(ServiceNames.CLAUDE)

# Use provider (interface is same for all providers)
response = await llm_provider.generate_text(
    prompt="Analyze this event",
    system_prompt="You are an event analyst"
)

# Analyze image
result = await llm_provider.analyze_image(
    image_path="/path/to/image.jpg",
    prompt="Extract text from this image"
)

# Synthesize multiple images + texts
synthesized = await llm_provider.synthesize_with_images_and_texts(
    image_paths=["img1.jpg", "img2.jpg"],
    texts=["OCR text 1", "OCR text 2"],
    system_prompt="You are a content synthesizer",
    user_prompt="Create a comprehensive summary"
)
```

### Code Example - Direct Usage (Advanced)

Nếu bạn muốn sử dụng trực tiếp provider mà không qua container:

```python
from services.llm_provider_factory import LLMProviderFactory
from config import Config

# Method 1: Using factory with config
provider_config = Config.get_llm_provider_config()
llm_provider = LLMProviderFactory.create_provider_from_config(provider_config)

# Method 2: Using factory directly
llm_provider = LLMProviderFactory.create_provider(
    provider_type="chatgpt",
    api_key="sk-proj-xxxxx",
    model="gpt-4o",
    max_tokens=4096
)

# Use provider
response = await llm_provider.generate_text("Hello, AI!")
```

---

## Architecture

### Class Diagram

```
┌─────────────────────────┐
│   BaseLLMProvider       │
│   (Abstract Interface)  │
├─────────────────────────┤
│ + generate_text()       │
│ + analyze_image()       │
│ + analyze_multiple()    │
│ + synthesize_content()  │
│ + synthesize_with_imgs()│
│ + get_provider_info()   │
└───────────┬─────────────┘
            │
            │ implements
    ┌───────┴────────┐
    │                │
┌───▼────────┐  ┌───▼──────────┐  ┌─────────────┐
│  Claude    │  │  ChatGPT     │  │   Gemini    │
│  Provider  │  │  Provider    │  │  Provider   │
└────────────┘  └──────────────┘  └─────────────┘
                                   (Coming Soon)
```

### Provider Factory Flow

```
1. Config.get_llm_provider_config()
   ↓
2. LLMProviderFactory.create_provider_from_config()
   ↓
3. Select provider class based on LLM_PROVIDER
   ↓
4. Create instance with API key and settings
   ↓
5. Return BaseLLMProvider instance
   ↓
6. Register in ServiceContainer
```

---

## API Reference

### BaseLLMProvider Interface

Tất cả providers đều implement các method sau:

#### `generate_text(prompt, system_prompt, max_tokens, temperature)`

Generate text từ text prompt đơn giản.

**Parameters:**
- `prompt` (str): User prompt
- `system_prompt` (str, optional): System/context prompt
- `max_tokens` (int, optional): Max tokens to generate
- `temperature` (float, optional): Sampling temperature

**Returns:** `str` - Generated text

#### `analyze_image(image_path, prompt, system_prompt)`

Phân tích một ảnh với text prompt (OCR, extraction).

**Parameters:**
- `image_path` (str): Path to image file
- `prompt` (str): User prompt for analysis
- `system_prompt` (str, optional): System/context prompt

**Returns:** `str` - Analysis result text

#### `analyze_multiple_images(image_paths, prompts, system_prompt, parallel)`

Phân tích nhiều ảnh (batch processing).

**Parameters:**
- `image_paths` (List[str]): List of image paths
- `prompts` (List[str]): List of prompts (one per image or one for all)
- `system_prompt` (str, optional): System prompt
- `parallel` (bool): Process in parallel or sequential

**Returns:** `List[Dict]` - List of results with success status

#### `synthesize_content(texts, system_prompt, user_prompt)`

Tổng hợp nhiều text thành nội dung coherent.

**Parameters:**
- `texts` (List[str]): List of texts to synthesize
- `system_prompt` (str, optional): System prompt
- `user_prompt` (str, optional): User instructions

**Returns:** `str` - Synthesized content

#### `synthesize_with_images_and_texts(image_paths, texts, system_prompt, user_prompt)`

Tổng hợp cả ảnh và text (multimodal analysis).

**Parameters:**
- `image_paths` (List[str]): List of image paths
- `texts` (List[str]): List of texts (e.g., OCR results)
- `system_prompt` (str, optional): System prompt
- `user_prompt` (str, optional): User instructions

**Returns:** `str` - Synthesized multimodal content

#### `get_provider_info()`

Lấy thông tin về provider.

**Returns:** `Dict` với keys:
- `provider`: Provider name
- `provider_type`: Enum value
- `model`: Model name
- `supports_vision`: Boolean
- `status`: "ready" or error status

---

## So sánh Providers

### Claude (Anthropic)

**Ưu điểm:**
- ✅ Context window lớn (200K tokens)
- ✅ Vision capabilities mạnh
- ✅ Accuracy cao trong phân tích phức tạp
- ✅ Hỗ trợ tốt cho tiếng Việt

**Nhược điểm:**
- ⚠️ Chi phí cao hơn GPT-4o
- ⚠️ Rate limits nghiêm ngặt hơn

**Use cases tốt nhất:**
- Phân tích event phức tạp
- OCR tiếng Việt
- Content synthesis dài

### ChatGPT (OpenAI)

**Ưu điểm:**
- ✅ GPT-4o có vision tốt
- ✅ Chi phí thấp hơn Claude
- ✅ Rate limits thoáng hơn
- ✅ Response nhanh

**Nhược điểm:**
- ⚠️ Context window nhỏ hơn (128K)
- ⚠️ Accuracy thấp hơn Claude một chút cho tasks phức tạp

**Use cases tốt nhất:**
- High-throughput applications
- Cost-sensitive projects
- Quick prototyping

### Gemini (Google) - Coming Soon

**Ưu điểm:**
- ✅ Miễn phí tier generous
- ✅ Tích hợp tốt với Google Cloud

**Nhược điểm:**
- ⚠️ Vision capabilities còn hạn chế
- ⚠️ Chưa stable như Claude/GPT

---

## Testing

### Test với nhiều providers

```bash
# Test với Claude
LLM_PROVIDER=claude pytest tests/

# Test với ChatGPT
LLM_PROVIDER=chatgpt pytest tests/

# Test tất cả providers
pytest tests/test_llm_providers.py
```

### Mock providers trong tests

```python
from unittest.mock import Mock
from services.llm_provider_base import BaseLLMProvider

# Create mock provider
mock_provider = Mock(spec=BaseLLMProvider)
mock_provider.generate_text.return_value = "Mocked response"

# Inject into container
container.register(ServiceNames.CLAUDE, mock_provider)
```

---

## Troubleshooting

### Provider initialization failed

**Error:** `⚠️ LLM provider: Provider initialization failed`

**Solution:**
1. Check API key trong `.env`:
   ```bash
   # For Claude
   echo $CLAUDE_API_KEY

   # For ChatGPT
   echo $OPENAI_API_KEY
   ```

2. Verify API key is valid:
   ```bash
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
        https://api.openai.com/v1/models
   ```

### Model does not support vision

**Error:** `NotImplementedError: Model gpt-3.5-turbo does not support vision`

**Solution:** Use vision-capable models:
- Claude: All Claude 3+ models support vision
- OpenAI: Use `gpt-4o`, `gpt-4-vision-preview`, or `gpt-4-turbo`

### Rate limit errors

**Error:** `429 Too Many Requests`

**Solution:**
1. Reduce concurrent requests in `.env`:
   ```bash
   MAX_CONCURRENT_REQUESTS=5
   ```

2. Add retry logic or switch to provider with higher limits

---

## Roadmap

### Current (v1.0)
- ✅ Claude Provider
- ✅ ChatGPT Provider
- ✅ Factory Pattern
- ✅ Configuration management

### Upcoming (v1.1)
- 🚧 Gemini Provider
- 🚧 Provider fallback mechanism
- 🚧 Cost tracking per provider
- 🚧 Response caching

### Future (v2.0)
- 📋 Mistral AI Provider
- 📋 Cohere Provider
- 📋 Automatic provider selection based on task
- 📋 A/B testing between providers

---

## Best Practices

### 1. Chọn provider phù hợp với use case

```python
# For high-accuracy analysis
LLM_PROVIDER=claude

# For high-throughput, cost-effective
LLM_PROVIDER=chatgpt
```

### 2. Sử dụng environment variables

```bash
# Development
LLM_PROVIDER=chatgpt  # Cheaper for testing

# Production
LLM_PROVIDER=claude   # Higher quality
```

### 3. Monitor costs

```python
provider_info = llm_provider.get_provider_info()
logger.info(f"Using provider: {provider_info['provider']}")
logger.info(f"Model: {provider_info['model']}")
```

### 4. Handle errors gracefully

```python
try:
    response = await llm_provider.generate_text(prompt)
except Exception as e:
    logger.error(f"LLM request failed: {e}")
    # Fallback to default response or retry with different provider
```

---

## Contributing

### Adding a new provider

1. Create provider class implementing `BaseLLMProvider`:
   ```python
   # services/gemini_provider.py
   from services.llm_provider_base import BaseLLMProvider

   class GeminiProvider(BaseLLMProvider):
       # Implement all abstract methods
       pass
   ```

2. Register in factory:
   ```python
   # services/llm_provider_factory.py
   _PROVIDER_CLASSES = {
       LLMProviderType.GEMINI: GeminiProvider,
   }
   ```

3. Add config in `config.py`:
   ```python
   GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
   GEMINI_MODEL: str = os.getenv('GEMINI_MODEL', 'gemini-pro')
   ```

4. Update tests and documentation

---

## Support

Nếu gặp vấn đề:
1. Check logs: `tail -f logs/image-similarity-api.log`
2. Verify config: `python -c "from config import Config; print(Config.get_llm_provider_config())"`
3. Test provider directly: `pytest tests/unit/test_llm_providers.py -v`

For issues, please create an issue on GitHub.
