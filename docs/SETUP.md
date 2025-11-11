# Image Similarity API - Setup Guide

## Quick Start

### 1. Choose Your Environment

The project now supports multiple environment configurations:

- **Development** (`dev`) - For local development
- **Production** (`prod`) - For production deployment
- **Ubuntu** (`ubuntu`) - Ubuntu-specific paths and settings
- **Windows** (`windows`) - Windows-specific paths and settings

### 2. Setup Environment Configuration

#### On Ubuntu/Linux:

```bash
# Setup for development
./setup_env.sh dev

# Or setup for Ubuntu-specific configuration
./setup_env.sh ubuntu

# Or setup for production
./setup_env.sh prod
```

#### On Windows:

```cmd
REM Setup for development
setup_env.bat dev

REM Or setup for Windows-specific configuration
setup_env.bat windows

REM Or setup for production
setup_env.bat prod
```

### 3. Verify Configuration

After running the setup script, check your `.env` file:

```bash
cat .env  # On Ubuntu/Linux
type .env  # On Windows
```

### 4. Install Dependencies

```bash
# Activate your conda environment
conda activate image-similarity-api

# Install requirements
pip install -r requirements.txt
```

### 5. Setup Model

```bash
python setup_model.py
```

### 6. Run the Application

#### Test Mode:
```bash
python main_new.py
```

#### API Mode:
```bash
uvicorn main:app --reload
```

---

## Environment Files Explained

### `.env.example`
- Template file with all configuration options
- Copy this to create your custom environment files
- Never commit sensitive data to this file

### `.env.development`
- Development environment settings
- Debug logging enabled
- Uses local paths
- Contains actual API keys (gitignored)

### `.env.production`
- Production environment settings
- Info-level logging
- Uses production paths
- Should use production API keys and database

### `.env.ubuntu`
- Ubuntu-specific settings
- Linux path format (forward slashes)
- Optimized for Ubuntu systems

### `.env.windows`
- Windows-specific settings
- Windows path format (double backslashes)
- Optimized for Windows systems

---

## Configuration Structure

The project uses a centralized configuration system:

```
config.py          # Main configuration module
├── Environment detection
├── Path management
├── API configuration
├── Database settings
└── Logging setup
```

### Key Configuration Variables

#### API Keys
```env
VOYAGE_API_KEY=your-voyage-api-key
CLAUDE_API_KEY=your-claude-api-key
```

#### Database
```env
DB_HOST=localhost
DB_PORT=5432
DB_USER=event_user
DB_PASS=your-password
DB_NAME=event_dev
```

#### Paths
```env
# Ubuntu/Linux format
SHARED_UPLOADS_PATH=/media/luanpc/Video/shared/uploads
MODELS_PATH=models
INDEX_PATH=index
LOGS_DIR=logs

# Windows format
SHARED_UPLOADS_PATH=C:\\Users\\YourName\\shared\\uploads
MODELS_PATH=models
INDEX_PATH=index
LOGS_DIR=logs
```

---

## Using the Config Module

### In Your Code

```python
from config import config

# Access configuration
api_key = config.CLAUDE_API_KEY
db_host = config.DB_HOST

# Check environment
if config.IS_DEVELOPMENT:
    print("Running in development mode")

# Get database config
db_config = config.get_db_config()

# Setup logging
config.setup_logging()
```

### Validation

The config module automatically validates required settings:

```python
from config import config

# Validate configuration
errors = config.validate()
if errors:
    print("Configuration errors:", errors)
```

---

## Switching Environments

### During Development

```bash
# Switch to development
./setup_env.sh dev

# Switch to Ubuntu configuration
./setup_env.sh ubuntu
```

### For Deployment

```bash
# Switch to production
./setup_env.sh prod

# Verify settings
cat .env
```

---

## Directory Structure

```
image-similarity-api/
├── config.py                    # Configuration module
├── main_new.py                  # Refactored test runner
├── main.py                      # Legacy test runner (kept for reference)
├── .env                         # Active environment (gitignored)
├── .env.example                 # Template
├── .env.development             # Dev settings (gitignored)
├── .env.production              # Prod settings (gitignored)
├── .env.ubuntu                  # Ubuntu settings (gitignored)
├── .env.windows                 # Windows settings (gitignored)
├── setup_env.sh                 # Setup script (Ubuntu/Linux)
├── setup_env.bat                # Setup script (Windows)
├── tests/
│   ├── __init__.py
│   ├── test_config.py           # Test configuration
│   ├── service_initializer.py  # Service initialization
│   ├── test_runner.py           # Test execution
│   └── test_reporter.py         # Results display
├── services/
│   ├── claude_service.py
│   ├── database_service.py
│   └── event_similarity_service.py
├── utils/
│   ├── faiss_manager.py
│   ├── image_processor.py
│   ├── prompt_manager.py
│   └── text_processor.py
└── models/
    └── places365.py
```

---

## Troubleshooting

### Missing Environment Variables

If you see errors about missing environment variables:

1. Check your `.env` file exists
2. Verify all required variables are set
3. Run the setup script again

```bash
./setup_env.sh dev
```

### Path Issues on Windows

If you encounter path errors on Windows:

1. Use the Windows-specific environment:
   ```cmd
   setup_env.bat windows
   ```

2. Ensure paths use double backslashes:
   ```env
   LOGS_DIR=C:\\Users\\YourName\\logs
   ```

3. Or use forward slashes (Python accepts both):
   ```env
   LOGS_DIR=C:/Users/YourName/logs
   ```

### Database Connection Issues

1. Verify PostgreSQL is running
2. Check database credentials in `.env`
3. Ensure database exists:
   ```bash
   psql -U postgres
   CREATE DATABASE event_dev;
   ```

### API Key Issues

1. Verify API keys are set in `.env`:
   ```bash
   grep API_KEY .env
   ```

2. Ensure no extra spaces or quotes
3. Keys should be plain text without quotes

---

## Best Practices

### Development
- Use `.env.development` or `.env.ubuntu`/`.env.windows`
- Enable debug logging
- Use relative paths when possible

### Production
- Use `.env.production`
- Use absolute paths
- Set `API_RELOAD=false`
- Use strong database passwords
- Regularly rotate API keys

### Version Control
- Never commit `.env` files (except `.env.example`)
- Keep `.env.example` updated
- Document all new environment variables

---

## Migration from Old Code

If you're migrating from the old `main.py`:

1. Your test configuration is now in `tests/test_config.py`
2. Use `main_new.py` instead of `main.py`
3. Update imports to use `config` module:
   ```python
   # Old
   db_host = os.getenv('DB_HOST')

   # New
   from config import config
   db_host = config.DB_HOST
   ```

---

## Support

For issues or questions:
1. Check this documentation
2. Review `.env.example` for configuration options
3. Check logs in `logs/` directory
4. Review error messages for missing dependencies

---

## Next Steps

After setup:
1. ✅ Run tests: `python main_new.py`
2. ✅ Start API: `uvicorn main:app --reload`
3. ✅ Check logs: `tail -f logs/image-similarity-api.log`
4. ✅ Monitor performance
5. ✅ Deploy to production when ready
