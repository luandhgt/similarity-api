# Image Similarity API - Refactored Version

## 🎯 Overview

This is the refactored version of the Image Similarity API with improved modularity, configuration management, and cross-platform support.

## ✨ What's New

### 1. Multi-Environment Support
- ✅ Development environment (`.env.development`)
- ✅ Production environment (`.env.production`)
- ✅ Ubuntu-specific configuration (`.env.ubuntu`)
- ✅ Windows-specific configuration (`.env.windows`)

### 2. Centralized Configuration
- ✅ Single `config.py` module for all settings
- ✅ Type-safe configuration access
- ✅ Automatic validation
- ✅ Environment detection

### 3. Modular Architecture
```
tests/
├── test_config.py          # Test configuration
├── service_initializer.py  # Service initialization
├── test_runner.py          # Test execution
└── test_reporter.py        # Results display
```

### 4. Easy Environment Switching
```bash
# Ubuntu/Linux
./setup_env.sh dev
./setup_env.sh ubuntu
./setup_env.sh prod

# Windows
setup_env.bat dev
setup_env.bat windows
setup_env.bat prod
```

## 🚀 Quick Start

### 1. Setup Environment

**On Ubuntu:**
```bash
./setup_env.sh ubuntu
```

**On Windows:**
```cmd
setup_env.bat windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Tests
```bash
python main_new.py
```

## 📁 Project Structure

```
image-similarity-api/
├── config.py                    # ⭐ New: Centralized configuration
├── main_new.py                  # ⭐ New: Refactored test runner
├── main.py                      # Old: Legacy runner (kept for reference)
│
├── .env                         # Active environment (auto-generated)
├── .env.example                 # ⭐ New: Template with all options
├── .env.development             # ⭐ New: Dev settings
├── .env.production              # ⭐ New: Prod settings
├── .env.ubuntu                  # ⭐ New: Ubuntu settings
├── .env.windows                 # ⭐ New: Windows settings
│
├── setup_env.sh                 # ⭐ New: Setup script (Linux)
├── setup_env.bat                # ⭐ New: Setup script (Windows)
│
├── tests/                       # ⭐ New: Test modules
│   ├── __init__.py
│   ├── test_config.py
│   ├── service_initializer.py
│   ├── test_runner.py
│   └── test_reporter.py
│
├── services/
│   ├── claude_service.py        # ✅ Updated to use config
│   ├── database_service.py      # ✅ Updated to use config
│   └── event_similarity_service.py
│
├── utils/
│   ├── faiss_manager.py
│   ├── image_processor.py
│   ├── prompt_manager.py
│   └── text_processor.py
│
├── models/
│   └── places365.py
│
├── config/
│   ├── prompts.yaml
│   ├── output_formats.yaml
│   ├── similarity_prompts.yaml
│   └── similarity_output_formats.yaml
│
├── SETUP.md                     # ⭐ New: Detailed setup guide
├── MIGRATION.md                 # ⭐ New: Migration guide
└── README_REFACTOR.md           # ⭐ This file
```

## 📖 Documentation

### For New Users
👉 Start with [SETUP.md](SETUP.md)

### For Existing Users
👉 Read [MIGRATION.md](MIGRATION.md)

## 🔧 Configuration

### Access Configuration in Code

**Old way:**
```python
import os
db_host = os.getenv('DB_HOST', 'localhost')
```

**New way:**
```python
from config import config
db_host = config.DB_HOST
```

### Configuration Files

| File | Purpose |
|------|---------|
| `.env.example` | Template with all available options |
| `.env.development` | Development settings (local, debug enabled) |
| `.env.production` | Production settings (optimized, secure) |
| `.env.ubuntu` | Ubuntu-specific paths and settings |
| `.env.windows` | Windows-specific paths and settings |

## 🎮 Usage Examples

### Switch to Development
```bash
./setup_env.sh dev
python main_new.py
```

### Switch to Production
```bash
./setup_env.sh prod
python main_new.py
```

### Run API Server
```bash
uvicorn main:app --reload
```

## ✅ Benefits

### 1. Cross-Platform Support
- Works on both Ubuntu and Windows
- Platform-specific configurations
- No more hardcoded paths

### 2. Environment Management
- Easy to switch between dev/prod
- Clear separation of configs
- Secure (all .env files are gitignored)

### 3. Maintainability
- Modular code structure
- Single source of truth for config
- Easy to test and debug

### 4. Type Safety
- Type hints throughout
- Better IDE support
- Catch errors early

### 5. Scalability
- Easy to add new environments
- Configuration validation
- Consistent across services

## 🔍 Key Features

### Centralized Configuration
```python
from config import config

# All settings in one place
api_key = config.CLAUDE_API_KEY
db_config = config.get_db_config()
log_level = config.LOG_LEVEL

# Environment detection
if config.IS_DEVELOPMENT:
    print("Running in development mode")
```

### Modular Test System
```python
from tests import (
    TestConfig,
    initialize_services,
    run_event_similarity_test,
    display_results
)

# Clean, reusable components
config = TestConfig()
services = await initialize_services()
result = await run_event_similarity_test(services, request)
display_results(result)
```

### Automatic Validation
```python
from config import config

# Config validates on import
errors = config.validate()
if errors:
    print("Configuration errors:", errors)
```

## 🛠️ Common Tasks

### Add New Environment Variable

1. Add to `.env.example`:
```env
NEW_VARIABLE=default_value
```

2. Add to `config.py`:
```python
NEW_VARIABLE: str = os.getenv('NEW_VARIABLE', 'default_value')
```

3. Use in code:
```python
from config import config
value = config.NEW_VARIABLE
```

### Update Test Configuration

Edit `tests/test_config.py`:
```python
class TestConfig:
    EVENT_NAME = "Your Event"
    GAME_CODE = "Your Game"
    # ...
```

### Switch Environments

```bash
# Development
./setup_env.sh dev

# Production
./setup_env.sh prod

# Platform-specific
./setup_env.sh ubuntu   # or windows
```

## 📊 Comparison

| Feature | Old | New |
|---------|-----|-----|
| **Config** | Scattered `os.getenv()` | Centralized `config` module |
| **Environments** | Single `.env` | Multiple env files |
| **Structure** | Monolithic `main.py` (483 lines) | Modular (< 150 lines each) |
| **Platform** | Manual path changes | Auto-detect platform |
| **Type Safety** | No types | Full type hints |
| **Documentation** | Minimal | Comprehensive |

## 🚨 Important Notes

### Git Ignored Files
These files are **NOT** committed to git (contains secrets):
- `.env`
- `.env.development`
- `.env.production`
- `.env.ubuntu`
- `.env.windows`

### Safe to Commit
- `.env.example` (template, no secrets)
- `config.py` (code, no secrets)
- `SETUP.md`, `MIGRATION.md` (documentation)

## 🐛 Troubleshooting

### "Config validation failed"
```bash
# Check your .env file
cat .env

# Re-run setup
./setup_env.sh dev
```

### "Module not found: config"
```python
# Add to top of your script
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

### Database connection failed
```bash
# Verify PostgreSQL is running
sudo systemctl status postgresql

# Check credentials
grep DB_ .env
```

## 📝 Next Steps

1. ✅ Read [SETUP.md](SETUP.md) for detailed setup
2. ✅ If migrating, read [MIGRATION.md](MIGRATION.md)
3. ✅ Setup your environment
4. ✅ Run tests
5. ✅ Deploy to production

## 🤝 Contributing

When adding new features:
1. Update `.env.example` with new variables
2. Add to `config.py`
3. Update documentation
4. Test on both Ubuntu and Windows

## 📜 License

Same as original project.

---

**Questions?** Check the documentation:
- [SETUP.md](SETUP.md) - Setup instructions
- [MIGRATION.md](MIGRATION.md) - Migration guide
- [README.md](README.md) - Original README
