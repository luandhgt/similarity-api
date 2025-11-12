# Image Similarity API

**Version:** 2.0.0 (Fully Refactored)
**Status:** ✅ Production Ready

Multi-modal event similarity search API using Places365, FAISS, Claude AI, and Voyage embeddings.

---

## 🚀 Quick Start

```bash
# 1. Setup environment
./setup_env.sh ubuntu  # or ./setup_env.sh windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test workflow (optional)
python test_workflow.py

# 4. Start API
uvicorn main:app --reload
```

**Visit:** http://localhost:8000/api/docs

---

## 📚 Documentation

All documentation has been moved to the [`docs/`](docs/) folder:

### Quick Links
- 📖 [**Documentation Index**](docs/README.md) - Complete documentation navigation
- 🚀 [**Quick Start Guide**](docs/QUICKSTART.md) - Get started in 5 minutes
- 🔧 [**Setup Guide**](docs/SETUP.md) - Detailed setup instructions
- 🔄 [**Migration Guide**](docs/MIGRATION.md) - Migrate from old code
- ✅ [**Refactoring Complete**](docs/REFACTORING_COMPLETE.md) - What's new in v2.0

### For Developers
- 🔍 [**Code Analysis**](docs/CODE_ANALYSIS.md) - Code structure & issues
- 📝 [**Changelog**](docs/CHANGELOG_REFACTOR.md) - All changes
- 🏗️ [**Architecture**](docs/README_REFACTOR.md) - System architecture

---

## ✨ What's New in v2.0

### Major Improvements
- ✅ **ServiceContainer** - Unified dependency injection
- ✅ **Custom Exceptions** - Better error handling
- ✅ **DTO Objects** - Type-safe data structures
- ✅ **Repository Pattern** - Clean database access
- ✅ **Request Validators** - Centralized validation
- ✅ **Unit Tests** - 50+ tests with pytest
- ✅ **Multi-Environment** - Dev/Prod/Ubuntu/Windows configs

### Code Quality
- ❌ **~270 lines** of duplicate code eliminated
- ✅ **Type safety** throughout with DTOs
- ✅ **SOLID principles** implemented
- ✅ **50+ unit tests** added
- ✅ **Comprehensive** documentation

---

## 🏗️ Architecture

```
image-similarity-api/
├── core/                   # Core framework (exceptions, container, factory)
├── models/                 # DTOs and data models
├── repositories/           # Data access layer
├── services/               # Business logic
├── utils/                  # Utilities (validators, image utils)
├── routers/                # API endpoints
├── tests/                  # Unit & integration tests
├── docs/                   # All documentation
└── config.py              # Configuration management
```

---

## 🔧 Configuration

### Environment Files
- `.env.example` - Template with all options
- `.env.development` - Development settings
- `.env.production` - Production settings
- `.env.ubuntu` - Ubuntu-specific
- `.env.windows` - Windows-specific

### Switch Environment
```bash
# Development
./setup_env.sh dev

# Production
./setup_env.sh prod

# Platform-specific
./setup_env.sh ubuntu    # or windows
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test
pytest tests/unit/test_validators.py
```

**Test Coverage:** 50+ unit tests

---

## 📦 Features

### Core Features
- 🖼️ **Image Similarity** - Places365 CNN embeddings
- 📝 **Text Similarity** - Voyage AI embeddings
- 🔍 **FAISS Search** - Fast vector similarity search
- 🤖 **AI Analysis** - Claude AI for semantic understanding
- 💾 **Database** - PostgreSQL for event storage

### API Endpoints
- `/api/extract-about` - Extract event info from images
- `/api/event-similarity/find` - Find similar events
- `/api/embed_image` - Create image embeddings
- `/api/embed_text` - Create text embeddings
- `/api/search_similar` - Similarity search
- `/health` - Health check

---

## 🛠️ Tech Stack

### Core
- **FastAPI** - Web framework
- **Python 3.8+** - Programming language
- **PyTorch** - Deep learning
- **PostgreSQL** - Database

### ML/AI
- **Places365** - Scene recognition
- **FAISS** - Vector similarity
- **Voyage AI** - Text embeddings
- **Claude AI** - Language understanding

### DevOps
- **Docker** (optional) - Containerization
- **Pytest** - Testing
- **Pydantic** - Data validation

---

## 📊 Performance

- ⚡ **Fast** - Vector search in milliseconds
- 🔄 **Async** - Non-blocking I/O
- 📈 **Scalable** - Horizontal scaling ready
- 💪 **Robust** - Comprehensive error handling

---

## 🤝 Contributing

1. Read [Documentation](docs/README.md)
2. Check [Code Analysis](docs/CODE_ANALYSIS.md)
3. Follow existing patterns
4. Add unit tests
5. Update documentation

---

## 📝 License

[Your License]

---

## 📞 Support

- 📖 [Full Documentation](docs/README.md)
- 🐛 [Issue Tracker](your-issues-url)
- 💬 [Discussions](your-discussions-url)

---

## 🙏 Acknowledgments

- FastAPI team
- PyTorch team
- FAISS developers
- Anthropic (Claude AI)
- Voyage AI

---

**Built with ❤️ using Python, FastAPI, and AI**
