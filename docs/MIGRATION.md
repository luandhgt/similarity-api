# Migration Guide - From Old to Refactored Code

## Overview

This guide helps you migrate from the old monolithic `main.py` to the new modular architecture.

---

## What Changed?

### 1. Environment Configuration

#### Before:
```python
# Hardcoded in code
SHARED_UPLOADS_PATH = "/media/luanpc/Video/shared/uploads/"
EVENT_NAME = "Trojan Treasures "

# Loaded from .env
load_dotenv()
```

#### After:
```python
# Centralized config module
from config import config

# Access via config object
shared_uploads_path = config.SHARED_UPLOADS_PATH
db_config = config.get_db_config()

# Multiple environment files
.env.development
.env.production
.env.ubuntu
.env.windows
```

### 2. File Structure

#### Before:
```
main.py (483 lines)
- Everything in one file
- Mixed concerns
- Hard to test
```

#### After:
```
config.py                       # Configuration
main_new.py                     # Main runner (120 lines)
tests/
  ├── test_config.py           # Test configuration
  ├── service_initializer.py   # Service setup
  ├── test_runner.py           # Test execution
  └── test_reporter.py         # Results display
```

### 3. Service Initialization

#### Before:
```python
# All in one async function
async def initialize_services():
    # 150+ lines of initialization code
    # Mixed logging, error handling
    # Hard to reuse
```

#### After:
```python
# Modular initialization
from tests import initialize_services

services = await initialize_services(verbose=True)
```

### 4. Test Configuration

#### Before:
```python
# Global constants at top of file
EVENT_NAME = "Trojan Treasures "
GAME_CODE = "Rise of Kingdoms"
EXPECTED_IMAGE_COUNT = 4
```

#### After:
```python
# Dedicated config class
from tests import TestConfig

config = TestConfig()
config.EVENT_NAME
config.GAME_CODE
```

---

## Step-by-Step Migration

### Step 1: Backup Current Setup

```bash
# Backup your current .env
cp .env .env.backup

# Backup current main.py (already exists as main.py)
# The old file is preserved
```

### Step 2: Setup New Environment

```bash
# Choose your environment
./setup_env.sh ubuntu    # For Ubuntu
# or
setup_env.bat windows    # For Windows

# Verify configuration
cat .env
```

### Step 3: Update Your Test Configuration

Edit `tests/test_config.py`:

```python
class TestConfig:
    EVENT_NAME = "Your Event Name"
    GAME_CODE = "Your Game Code"
    FOLDER_NAME = "your_folder_name"
    SHARED_UPLOADS_PATH = "/your/path/to/uploads/"
    EXPECTED_IMAGE_COUNT = 4

    VERBOSE_OUTPUT = True
    SAVE_RESULTS = True
    OUTPUT_FILE = "test_results.json"
```

### Step 4: Update Service Imports

If you have custom services:

#### Before:
```python
import os

db_host = os.getenv('DB_HOST', 'localhost')
db_port = int(os.getenv('DB_PORT', 5432))
```

#### After:
```python
from config import config

db_config = config.get_db_config()
# Or
db_host = config.DB_HOST
db_port = config.DB_PORT
```

### Step 5: Update Database Service

Edit `services/database_service.py`:

#### Before:
```python
def __init__(self):
    self.pool = None
    self.db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASS'),
        'database': os.getenv('DB_NAME')
    }
```

#### After:
```python
from config import config

def __init__(self):
    self.pool = None
    self.db_config = config.get_db_config()
```

### Step 6: Update Claude Service

Edit `services/claude_service.py`:

#### Before:
```python
import os

self.api_key = os.getenv('CLAUDE_API_KEY')
self.model = "claude-3-5-sonnet-20241022"
```

#### After:
```python
from config import config

self.api_key = config.CLAUDE_API_KEY
self.model = config.CLAUDE_MODEL
```

### Step 7: Run Tests

```bash
# Run the new test runner
python main_new.py

# Compare with old runner (if needed)
python main.py
```

---

## API Changes

### Configuration Access

| Old | New |
|-----|-----|
| `os.getenv('CLAUDE_API_KEY')` | `config.CLAUDE_API_KEY` |
| `os.getenv('DB_HOST', 'localhost')` | `config.DB_HOST` |
| `int(os.getenv('DB_PORT', 5432))` | `config.DB_PORT` |
| Hardcoded paths | `config.SHARED_UPLOADS_PATH` |

### Service Initialization

| Old | New |
|-----|-----|
| `initialize_services()` in main.py | `from tests import initialize_services` |
| 150+ lines inline | Modular, reusable function |
| Mixed concerns | Separated concerns |

### Test Configuration

| Old | New |
|-----|-----|
| Global constants | `TestConfig` class |
| In main.py | In `tests/test_config.py` |
| Not reusable | Reusable across tests |

---

## Benefits of New Architecture

### 1. **Separation of Concerns**
- Configuration separate from logic
- Services separate from tests
- Easy to maintain

### 2. **Environment Management**
- Easy to switch environments
- Platform-specific configurations
- No more hardcoded paths

### 3. **Reusability**
- Services can be imported anywhere
- Test utilities are modular
- Configuration is centralized

### 4. **Type Safety**
- Type hints throughout
- Better IDE support
- Fewer runtime errors

### 5. **Maintainability**
- Smaller files
- Clear responsibilities
- Easy to understand

### 6. **Testing**
- Easier to unit test
- Mock-friendly
- Isolated components

---

## Common Migration Issues

### Issue 1: Import Errors

**Problem:**
```python
ModuleNotFoundError: No module named 'config'
```

**Solution:**
```python
# Add project root to path (in your script)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Then import
from config import config
```

### Issue 2: Environment Variables Not Loaded

**Problem:**
```
Configuration warnings: ['VOYAGE_API_KEY is required']
```

**Solution:**
```bash
# Make sure .env file exists
ls -la .env

# Run setup script
./setup_env.sh dev

# Verify variables
grep API_KEY .env
```

### Issue 3: Path Not Found

**Problem:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'logs/'
```

**Solution:**
```bash
# Config module auto-creates dirs, but you can manually:
mkdir -p logs models index shared/uploads
```

### Issue 4: Database Connection Failed

**Problem:**
```
Database service: connection refused
```

**Solution:**
```bash
# Check database is running
sudo systemctl status postgresql

# Verify credentials in .env
cat .env | grep DB_

# Test connection
psql -h localhost -U event_user -d event_dev
```

---

## Rollback Plan

If you need to rollback to the old version:

```bash
# Restore old .env
cp .env.backup .env

# Use old main.py
python main.py

# The old file is still available and unchanged
```

---

## Gradual Migration

You can migrate gradually:

### Phase 1: Use New Config Only
```python
# In existing code
from config import config

# Replace os.getenv() calls
db_host = config.DB_HOST  # Instead of os.getenv('DB_HOST')
```

### Phase 2: Use New Test Runner
```bash
# Start using new runner
python main_new.py

# But keep old code for comparison
python main.py
```

### Phase 3: Update Services
```python
# Update one service at a time
# Start with database_service.py
# Then claude_service.py
# etc.
```

### Phase 4: Complete Migration
```bash
# When confident, remove old files
# Or keep them as .bak for reference
mv main.py main.py.old
mv main_new.py main.py
```

---

## Verification Checklist

After migration, verify:

- [ ] Environment file loaded correctly
- [ ] All services initialize
- [ ] Database connection works
- [ ] API keys are valid
- [ ] Paths are correct
- [ ] Tests run successfully
- [ ] Results are saved correctly
- [ ] Logs are written
- [ ] No deprecation warnings

---

## Getting Help

If you encounter issues:

1. Check `SETUP.md` for configuration help
2. Review error messages carefully
3. Verify environment variables
4. Check logs in `logs/` directory
5. Compare with `.env.example`

---

## Next Steps

1. Complete migration following this guide
2. Test thoroughly in development
3. Update any custom scripts
4. Deploy to production when ready
5. Remove old code after verification

---

## Notes

- Old `main.py` is preserved for reference
- New code is in `main_new.py`
- All environment files are gitignored
- Configuration is validated on startup
- Logs provide detailed debugging info
