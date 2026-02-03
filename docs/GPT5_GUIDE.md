# GPT-5 Usage Guide

Hướng dẫn sử dụng GPT-5 models với hệ thống Multi-LLM Provider.

---

## 🚀 GPT-5 Models Overview

GPT-5 ra mắt năm 2025 với nhiều variants khác nhau cho các use cases khác nhau.

### 📊 GPT-5 Model Comparison

| Model | Context Window | Max Output | Vision | Best For | Cost |
|-------|----------------|------------|--------|----------|------|
| **gpt-5-pro** | 400K (272K in + 128K out) | **128K** | ✅ Yes | **Maximum quality** | **$$$$$** |
| **gpt-5** | 400K (272K in + 128K out) | **128K** | ✅ Yes | Long-form generation | $$$$ |
| **gpt-5-chat** | 144K (128K + 16K) | **16K** | ✅ Yes | Conversational AI | $$$ |
| **gpt-5-mini** | 400K (272K + 128K) | **128K** | ✅ Yes | Efficient, fast | $$ |
| **gpt-5-nano** | 400K (272K + 128K) | **128K** | ✅ Yes | Ultra-fast, cheap | $ |
| **gpt-4o** | 128K | 16K | ✅ Yes | Previous gen (still good) | $$ |

### 💰 Pricing Comparison (Per 1M Tokens)

| Model | Input Cost | Output Cost | vs gpt-5 |
|-------|-----------|-------------|----------|
| **gpt-5-pro** | **$15.00** | **$120.00** | **12x more expensive!** 💰💰💰 |
| **gpt-5** | $1.25 | $10.00 | Baseline |
| **gpt-5-chat** | ~$1.00 | ~$8.00 | ~20% cheaper |
| **gpt-5-mini** | ~$0.50 | ~$4.00 | ~60% cheaper |
| **gpt-4o** | ~$2.50 | ~$10.00 | Similar to gpt-5 |

### 🎯 Token Limits Detail

#### GPT-5 Pro (Premium Model) 💎
```bash
Input:  272,000 tokens
Output: 128,000 tokens
Total:  400,000 tokens

Pricing: $15/1M input | $120/1M output

# Best for:
- Maximum quality requirements
- Critical business applications
- Complex reasoning tasks
- When accuracy is paramount (12x cost of gpt-5!)

# Warning:
⚠️ Very expensive! Use only when quality justifies cost
⚠️ 10K input + 16K output = ~$2.00
⚠️ 10K input + 128K output = ~$15.00
```

#### GPT-5 (Full Model)
```bash
Input:  272,000 tokens
Output: 128,000 tokens
Total:  400,000 tokens

Pricing: $1.25/1M input | $10/1M output

# Best for:
- Long document generation
- Complex analysis requiring large output
- Multi-document synthesis
- Good balance of quality and cost
```

#### GPT-5 Chat
```bash
Input:  128,000 tokens
Output:  16,384 tokens
Total:  144,384 tokens

# Best for:
- Conversational applications
- Chat-based interactions
- Event analysis (your use case) ✅
```

#### GPT-5 Mini/Nano
```bash
Input:  272,000 tokens
Output: 128,000 tokens
Total:  400,000 tokens

# Same capacity as GPT-5 but:
- Faster inference
- Lower cost
- Slightly lower quality
```

---

## ⚙️ Configuration

### Quick Setup for GPT-5

#### Option 1: GPT-5 Chat (Recommended for your use case)

```bash
# .env
LLM_PROVIDER=chatgpt
OPENAI_API_KEY=sk-proj-xxxxx
OPENAI_MODEL=gpt-5-chat
OPENAI_MAX_TOKENS=16384      # Max for gpt-5-chat
OPENAI_TEMPERATURE=0.7
OPENAI_TIMEOUT=300
```

**Why gpt-5-chat?**
- ✅ 16K output is enough for event analysis
- ✅ Optimized for conversational/structured tasks
- ✅ Lower cost than full gpt-5
- ✅ Faster inference

#### Option 2: GPT-5 Full (For very long outputs)

```bash
# .env
LLM_PROVIDER=chatgpt
OPENAI_API_KEY=sk-proj-xxxxx
OPENAI_MODEL=gpt-5
OPENAI_MAX_TOKENS=128000     # Max for gpt-5 (huge!)
OPENAI_TEMPERATURE=0.7
OPENAI_TIMEOUT=600           # Increase timeout for large outputs
```

**When to use full gpt-5?**
- 📝 Generating very long reports (> 16K tokens)
- 📄 Multi-document synthesis
- 📊 Complex data analysis requiring detailed output

#### Option 3: GPT-5 Pro (Maximum Quality) 💎

```bash
# .env
LLM_PROVIDER=chatgpt
OPENAI_API_KEY=sk-proj-xxxxx
OPENAI_MODEL=gpt-5-pro
OPENAI_MAX_TOKENS=128000     # Max for gpt-5-pro
OPENAI_TEMPERATURE=0.7
OPENAI_TIMEOUT=600           # Increase timeout for large outputs
```

**When to use gpt-5-pro?**
- 🎯 Maximum accuracy critical
- 💼 Critical business decisions
- 🧠 Complex reasoning tasks
- 📊 High-stakes analysis

**⚠️ Warning:**
- 💰💰💰 12x more expensive than gpt-5!
- 📈 Cost example: 100K tokens output = $12.00
- 🔍 Only use when quality justifies the cost

#### Option 4: GPT-5 Mini (Cost-effective)

```bash
# .env
LLM_PROVIDER=chatgpt
OPENAI_API_KEY=sk-proj-xxxxx
OPENAI_MODEL=gpt-5-mini
OPENAI_MAX_TOKENS=64000      # Practical limit (half of max)
OPENAI_TEMPERATURE=0.7
OPENAI_TIMEOUT=300
```

**Why gpt-5-mini?**
- 💰 Much cheaper than gpt-5
- ⚡ Faster inference
- ✅ Still very capable

---

## 💡 Usage Examples

### Example 1: Event Analysis with GPT-5 Chat

```bash
# .env.development
LLM_PROVIDER=chatgpt
OPENAI_MODEL=gpt-5-chat
OPENAI_MAX_TOKENS=16384
```

```python
# No code changes needed!
from core.container import get_container, ServiceNames

container = get_container()
llm = container.get(ServiceNames.CLAUDE)  # Returns GPT-5 Chat

# Analyze event
result = await llm.analyze_image(
    image_path="event_screenshot.jpg",
    prompt="Extract event information",
    system_prompt="You are an event analyst"
)

# Synthesize multiple images
summary = await llm.synthesize_with_images_and_texts(
    image_paths=["img1.jpg", "img2.jpg", "img3.jpg"],
    texts=["OCR text 1", "OCR text 2", "OCR text 3"],
    user_prompt="Create comprehensive event description"
)
```

### Example 2: Long-form Generation with GPT-5

```bash
# .env.production
LLM_PROVIDER=chatgpt
OPENAI_MODEL=gpt-5
OPENAI_MAX_TOKENS=64000  # Use 64K for practical purposes
```

```python
# Generate detailed analysis
detailed_report = await llm.generate_text(
    prompt="""
    Analyze these 50 events and create a comprehensive report with:
    - Trend analysis
    - Category breakdown
    - Recommendations
    - Detailed insights for each event
    """,
    system_prompt="You are a senior data analyst"
)
```

### Example 3: Cost Optimization with GPT-5 Mini

```bash
# .env.development (cost-effective)
LLM_PROVIDER=chatgpt
OPENAI_MODEL=gpt-5-mini
OPENAI_MAX_TOKENS=16384
```

---

## 📈 Performance Comparison

### Response Time (Approximate)

| Model | Simple Query | Complex Analysis | Multi-Image |
|-------|-------------|------------------|-------------|
| gpt-5 | 2-3s | 8-15s | 15-25s |
| gpt-5-chat | 1-2s | 5-10s | 10-18s ✅ |
| gpt-5-mini | 0.5-1s | 3-6s | 6-12s ⚡ |
| gpt-4o | 1-2s | 6-12s | 12-20s |

### Cost Comparison (Relative)

```
gpt-5:      $$$$  (100% baseline)
gpt-5-chat: $$$   (~70% of gpt-5)
gpt-5-mini: $$    (~40% of gpt-5)
gpt-4o:     $$    (~50% of gpt-5)
```

### Quality Comparison

```
gpt-5:      ⭐⭐⭐⭐⭐ (Best)
gpt-5-chat: ⭐⭐⭐⭐⭐ (Best for chat)
gpt-5-mini: ⭐⭐⭐⭐  (Very good)
gpt-4o:     ⭐⭐⭐⭐  (Previous gen)
```

---

## 🎯 Recommendations by Use Case

### Your Project: Event Analysis

**Recommended:** `gpt-5-chat`

```bash
OPENAI_MODEL=gpt-5-chat
OPENAI_MAX_TOKENS=16384
```

**Why?**
- ✅ 16K output is perfect for event descriptions
- ✅ Optimized for structured extraction
- ✅ Great quality-to-cost ratio
- ✅ Vision support for screenshot analysis

### Use Cases by Model

#### 1. **OCR + Simple Extraction** → `gpt-5-mini`
```bash
OPENAI_MODEL=gpt-5-mini
OPENAI_MAX_TOKENS=4096
```
- Fast and cheap
- Good enough for OCR

#### 2. **Event Description + Synthesis** → `gpt-5-chat` ✅
```bash
OPENAI_MODEL=gpt-5-chat
OPENAI_MAX_TOKENS=16384
```
- Best balanced option
- Recommended for your project

#### 3. **Long Reports + Analysis** → `gpt-5`
```bash
OPENAI_MODEL=gpt-5
OPENAI_MAX_TOKENS=64000
```
- When you need very long outputs
- Complex multi-document analysis

---

## ⚠️ Important Considerations

### 1. Token Usage & Cost

**Be mindful of output length:**

```python
# Good: Reasonable limit
OPENAI_MAX_TOKENS=16384  # ~12K words

# Caution: Very expensive
OPENAI_MAX_TOKENS=128000  # ~96K words! $$$
```

**Cost calculation:**
```
Input tokens:  Cheaper
Output tokens: More expensive (2-3x input cost)

Example (gpt-5):
- 10K input + 16K output ≈ $X
- 10K input + 128K output ≈ $8X (8x more!)
```

### 2. API Limits

**Real-world limits differ from advertised:**

Some users report hitting limits at:
- **272,000 input tokens** (actual limit, matches docs)
- **Timeout issues** with very long outputs

**Recommendations:**
```bash
# Practical limits (avoid timeout)
gpt-5:      MAX_TOKENS=64000  (not 128K)
gpt-5-chat: MAX_TOKENS=16384  (safe)
gpt-5-mini: MAX_TOKENS=32000  (not 128K)
```

### 3. ChatGPT Web vs API

**Important distinction:**

| Interface | Context | Output | Access |
|-----------|---------|--------|--------|
| **API** | 272K-400K | 16K-128K | ✅ Your app |
| **ChatGPT Free** | 8K | Limited | 🌐 Web only |
| **ChatGPT Plus** | 32K | Limited | 🌐 Web only |
| **ChatGPT Pro** | 128K | Limited | 🌐 Web only |

Your application uses **API** → Full capabilities! ✅

---

## 🔧 Troubleshooting

### Issue: "Token limit exceeded"

**Error:**
```
Input tokens exceed the configured limit of 272,000 tokens
```

**Solution:**
1. Reduce input size (fewer images, shorter texts)
2. Use `gpt-5-chat` instead (128K input max)
3. Split requests into batches

### Issue: Request timeout

**Error:**
```
Request timeout after 300 seconds
```

**Solution:**
```bash
# Increase timeout for large outputs
OPENAI_TIMEOUT=600  # 10 minutes

# Or reduce max_tokens
OPENAI_MAX_TOKENS=32000  # Instead of 128K
```

### Issue: High costs

**Solution:**
1. Use `gpt-5-mini` or `gpt-5-chat` instead of `gpt-5`
2. Reduce `max_tokens` to actual needs
3. Cache responses when possible
4. Use `gpt-4o` for less critical tasks

---

## 📊 Token Estimation

### How many tokens is X words/characters?

```
1 token ≈ 0.75 words (English)
1 token ≈ 4 characters (English)

Examples:
- 1K tokens  ≈ 750 words   ≈ 4K characters
- 16K tokens ≈ 12K words   ≈ 64K characters
- 128K tokens ≈ 96K words  ≈ 512K characters
```

### Your Event Analysis (Typical):

```
Input:
- 3 images (screenshots)
- 3 OCR texts (1K tokens each)
- System + user prompt (500 tokens)
Total input: ~3.5K tokens

Output:
- Event description (bilingual)
- Structured data
- Taxonomy classifications
Total output: ~2-4K tokens

Model recommendation: gpt-5-chat (16K max) ✅
```

---

## 🚀 Migration Path

### From GPT-4o to GPT-5

**Step 1: Test with gpt-5-chat first**
```bash
# .env.development
OPENAI_MODEL=gpt-5-chat
OPENAI_MAX_TOKENS=16384
```

**Step 2: Compare quality**
```bash
# Run same test with both
OPENAI_MODEL=gpt-4o python test.py > results_gpt4o.txt
OPENAI_MODEL=gpt-5-chat python test.py > results_gpt5.txt

# Compare
diff results_gpt4o.txt results_gpt5.txt
```

**Step 3: Monitor costs**
- Check OpenAI dashboard
- Compare costs for 100 requests

**Step 4: Roll out to production**
```bash
# .env.production
OPENAI_MODEL=gpt-5-chat
```

---

## 🔗 References

### Official Documentation
- [ChatGPT Token Limits 2025](https://www.datastudios.org/post/chatgpt-token-limits-and-context-windows-updated-for-all-models-in-2025)
- [GPT-5 Context Window Guide](https://allthings.how/gpt-5-context-window-limits-and-usage-in-chatgpt-and-api/)
- [OpenAI API Token Limits 2025](https://www.scriptbyai.com/token-limit-openai-chatgpt/)
- [GPT-5 Features & Benchmarks](https://www.leanware.co/insights/gpt-5-features-guide)

### Community Resources
- [GPT-5 Documentation Gap Discussion](https://community.openai.com/t/huge-gpt-5-documentation-gap-flaw-causing-bugs-input-tokens-exceed-the-configured-limit-of-272-000-tokens/1344734)
- [GPT-5 on Azure Token Limits](https://www.arsturn.com/blog/gpt-5-on-azure-its-token-limits-are-they-a-bug-or-a-feature)

---

## 📝 Summary

### Quick Decision Guide

```
Need fast & cheap?         → gpt-5-mini
Need balanced quality?     → gpt-5-chat ✅ (recommended)
Need very long output?     → gpt-5
Need previous gen?         → gpt-4o
```

### Configuration Cheat Sheet

```bash
# Development (testing)
OPENAI_MODEL=gpt-5-mini
OPENAI_MAX_TOKENS=16384

# Production (your event analysis)
OPENAI_MODEL=gpt-5-chat
OPENAI_MAX_TOKENS=16384

# Special cases (long reports)
OPENAI_MODEL=gpt-5
OPENAI_MAX_TOKENS=64000
```

---

**Version:** 2.1.0
**Last Updated:** 2025-11-24
**Status:** ✅ Production Ready with GPT-5 Support
