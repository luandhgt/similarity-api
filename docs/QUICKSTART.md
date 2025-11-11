# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### On Ubuntu/Linux

```bash
# 1. Setup environment
./setup_env.sh ubuntu

# 2. Verify configuration
cat .env

# 3. Install dependencies (if not already installed)
pip install -r requirements.txt

# 4. Run test
python main_new.py
```

### On Windows

```cmd
REM 1. Setup environment
setup_env.bat windows

REM 2. Verify configuration
type .env

REM 3. Install dependencies (if not already installed)
pip install -r requirements.txt

REM 4. Run test
python main_new.py
```

---

## 📋 What Just Happened?

1. **Setup Script** copied the appropriate `.env` file for your platform
2. **Configuration** is now loaded from the `.env` file
3. **Test Runner** will initialize all services and run similarity analysis

---

## 🎯 Common Commands

### Switch Environments

```bash
# Development (with debug logging)
./setup_env.sh dev

# Production (optimized)
./setup_env.sh prod

# Platform-specific
./setup_env.sh ubuntu     # For Ubuntu/Linux
./setup_env.sh windows    # For Windows (on Linux)
```

### Run Tests

```bash
# Run refactored test
python main_new.py

# Run original test (for comparison)
python main.py
```

### Start API Server

```bash
# Development mode (with auto-reload)
uvicorn main:app --reload

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## ⚙️ Configuration Quick Reference

### Required Environment Variables

Edit your `.env` file:

```env
# API Keys
VOYAGE_API_KEY=your-voyage-api-key
CLAUDE_API_KEY=your-claude-api-key

# Database
DB_HOST=localhost
DB_PORT=5432
DB_USER=event_user
DB_PASS=your-password
DB_NAME=event_dev

# Paths (Ubuntu example)
SHARED_UPLOADS_PATH=/media/luanpc/Video/shared/uploads

# Paths (Windows example)
# SHARED_UPLOADS_PATH=C:\\Users\\YourName\\shared\\uploads
```

---

## 🧪 Test Your Setup

### 1. Check Configuration

```python
python -c "from config import config; config.print_config()"
```

### 2. Validate Configuration

```python
python -c "from config import config; print(config.validate())"
```

### 3. Test Database Connection

```python
python -c "
import asyncio
from services.database_service import DatabaseService

async def test():
    db = DatabaseService()
    await db.initialize()
    health = await db.health_check()
    print(health)

asyncio.run(test())
"
```

---

## 📁 Project Structure Quick View

```
image-similarity-api/
├── config.py                    # ← Configuration module
├── main_new.py                  # ← Refactored test runner
│
├── .env                         # ← Active environment
├── setup_env.sh / .bat          # ← Setup scripts
│
├── tests/                       # ← Test modules
│   ├── test_config.py
│   ├── service_initializer.py
│   ├── test_runner.py
│   └── test_reporter.py
│
├── services/                    # ← Service layer
│   ├── claude_service.py
│   ├── database_service.py
│   └── event_similarity_service.py
│
└── SETUP.md                     # ← Detailed setup guide
```

---

## 🔍 Troubleshooting

### Problem: "Missing environment variables"

**Solution:**
```bash
# Check if .env exists
ls -la .env

# Run setup again
./setup_env.sh ubuntu

# Verify API keys
grep API_KEY .env
```

### Problem: "Database connection failed"

**Solution:**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Verify credentials
grep DB_ .env

# Test connection
psql -h localhost -U event_user -d event_dev
```

### Problem: "Module not found: config"

**Solution:**
```bash
# Make sure you're in the project directory
cd /media/luanpc/Video/image-similarity-api

# Run from project root
python main_new.py
```

### Problem: "Permission denied: setup_env.sh"

**Solution:**
```bash
# Make script executable
chmod +x setup_env.sh

# Run again
./setup_env.sh ubuntu
```

---

## 📚 Next Steps

### For New Users
👉 Read [SETUP.md](SETUP.md) for comprehensive setup guide

### For Existing Users
👉 Read [MIGRATION.md](MIGRATION.md) for migration guide

### For Developers
👉 Read [README_REFACTOR.md](README_REFACTOR.md) for architecture overview

---

## 💡 Tips

### Development Mode
- Use `./setup_env.sh dev` for development
- Enables debug logging
- Uses local paths
- Auto-reload enabled

### Production Mode
- Use `./setup_env.sh prod` for production
- Info-level logging
- Absolute paths
- Optimized settings

### Platform-Specific
- Use `./setup_env.sh ubuntu` for Ubuntu
- Use `./setup_env.sh windows` for Windows (when developing for Windows)
- Handles path differences automatically

---

## ✅ Checklist

Before running:
- [ ] Setup script executed
- [ ] `.env` file exists
- [ ] API keys configured
- [ ] Database running
- [ ] Dependencies installed

To verify:
```bash
# All in one
ls -la .env && grep API_KEY .env && sudo systemctl status postgresql
```

---

## 🎉 Success!

If everything works, you should see:
```
🧪 EVENT SIMILARITY ANALYSIS TEST
============================================================
✅ All services initialized successfully
🚀 Starting Event Similarity Analysis Test...
✅ Analysis completed in X.XX seconds
📊 TEST RESULTS
...
🎉 Test completed successfully!
```

---

## 📞 Need Help?

1. Check error messages carefully
2. Review [SETUP.md](SETUP.md)
3. Check logs in `logs/` directory
4. Verify `.env` file settings

---

**Ready to go!** 🚀
