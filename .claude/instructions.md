# Claude Code Instructions - Image Similarity API

## Project Overview
**Image Similarity API v2.0.0** - Multi-modal event similarity search API using Places365, FAISS, Claude AI, and Voyage embeddings.

---

## ⚠️ CRITICAL: Keeping Documentation in Sync

**IMPORTANT:** Khi bạn thực hiện bất kỳ thay đổi nào về cấu trúc dự án, bạn PHẢI cập nhật ngay các file sau để đảm bảo đồng bộ:

### Files cần cập nhật khi có thay đổi cấu trúc:

1. **`.claude/instructions.md`** (file này) - Phần "Directory Structure"
2. **`.claude/project-guidelines.md`** - Các phần liên quan đến cấu trúc
3. **`README.md`** - Architecture section
4. **`docs/README_REFACTOR.md`** - Architecture documentation

### Các thay đổi cần theo dõi:

- ✅ Thêm/xóa/đổi tên thư mục (core/, models/, services/, etc.)
- ✅ Thêm/xóa/đổi tên file quan trọng (services, repositories, routers)
- ✅ Thay đổi design patterns hoặc architecture
- ✅ Thêm/thay đổi dependencies hoặc tech stack
- ✅ Cập nhật naming conventions
- ✅ Thay đổi configuration management

### Quy trình khi thay đổi cấu trúc:

```bash
# 1. Thực hiện thay đổi code
# (thêm file mới, di chuyển module, etc.)

# 2. NGAY LẬP TỨC cập nhật documentation
# - Cập nhật .claude/instructions.md (Directory Structure section)
# - Cập nhật .claude/project-guidelines.md (nếu cần)
# - Cập nhật README.md (Architecture section)
# - Cập nhật docs/README_REFACTOR.md (nếu cần)

# 3. Commit cả code và documentation cùng nhau
git add .
git commit -m "feat: Add new service + update documentation"
```

**Lưu ý:** Documentation không đồng bộ sẽ gây nhầm lẫn cho AI assistant và developers khác. Luôn cập nhật documentation NGAY khi thay đổi cấu trúc!

---

## Architecture Principles (SOLID)
This codebase follows SOLID principles and clean architecture:

### Core Patterns
1. **Dependency Injection** - `ServiceContainer` manages all services
2. **Repository Pattern** - Data access through `EventRepository`
3. **DTO Pattern** - Type-safe data transfer with Pydantic models
4. **Factory Pattern** - `ServiceFactory` creates service instances
5. **Custom Exceptions** - Domain-specific error handling

### Directory Structure
```
image-similarity-api/
├── .claude/               # Claude Code configuration
│   ├── instructions.md    # Coding standards & patterns
│   └── project-guidelines.md  # Development guidelines
│
├── core/                  # Framework (exceptions, container, factory)
│   ├── __init__.py
│   ├── exceptions.py      # Custom exception classes
│   ├── container.py       # ServiceContainer (singleton DI)
│   └── service_factory.py # Service creation logic
│
├── models/                # DTOs and data models
│   ├── __init__.py
│   ├── dtos.py           # Data Transfer Objects (Pydantic)
│   ├── responses.py      # API response models
│   ├── places365.py      # Places365 CNN model
│   └── resnet50_places365.pth.tar  # Pre-trained model weights
│
├── repositories/          # Data access layer
│   ├── __init__.py
│   └── event_repository.py # PostgreSQL operations
│
├── services/              # Business logic
│   ├── about_extraction_service.py  # Extract event info from images
│   ├── event_similarity_service.py  # Find similar events
│   ├── claude_service.py            # Claude AI integration
│   └── database_service.py          # Database operations
│
├── utils/                 # Utilities
│   ├── validators.py      # Request validation
│   ├── image_utils.py     # Image utilities
│   ├── image_processor.py # Image preprocessing
│   ├── text_processor.py  # Text embeddings (Voyage)
│   ├── faiss_manager.py   # FAISS index management
│   ├── prompt_manager.py  # Prompt templates manager
│   └── output_formatter.py # Output formatting
│
├── routers/               # API endpoints (FastAPI)
│   ├── about_extraction.py
│   ├── event_similarity.py
│   └── similarity.py
│
├── tests/                 # Unit & integration tests
│   ├── __init__.py
│   ├── conftest.py        # Pytest fixtures
│   ├── test_config.py     # Test configuration
│   ├── service_initializer.py  # Service setup for tests
│   ├── test_runner.py     # Test execution
│   ├── test_reporter.py   # Results reporting
│   └── unit/              # Unit tests
│       ├── __init__.py
│       ├── test_validators.py
│       ├── test_container.py
│       └── test_dtos.py
│
├── config/                # Configuration files (YAML only!)
│   ├── event_about_prompts.yaml      # Event about generation prompts
│   ├── event_about_template.yaml     # Event about output templates
│   ├── similarity_prompts.yaml       # Similarity analysis prompts
│   └── similarity_output_formats.yaml # Similarity output formats
│
├── docs/                  # Documentation (all .md files go here!)
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── SETUP.md
│   ├── MIGRATION.md
│   ├── README_REFACTOR.md
│   ├── REFACTORING_COMPLETE.md
│   ├── CODE_ANALYSIS.md
│   ├── CHANGELOG_REFACTOR.md
│   ├── CHANGELOG_EVENT_ABOUT.md      # Event about system changelog
│   └── UPGRADE_SUMMARY.md            # Event about upgrade summary
│
├── index/                 # FAISS indices (generated)
│   ├── about/
│   ├── images/
│   └── name/
│
├── logs/                  # Log files (generated)
│
├── test_workflow.py       # Development workflow test runner
├── main.py                # Production FastAPI application
├── config.py              # Configuration management
├── setup_model.py         # Download Places365 model
│
├── .env.example           # Environment template
├── .env.development       # Dev environment
├── .env.production        # Prod environment
├── .env.ubuntu            # Ubuntu-specific
├── .env.windows           # Windows-specific
├── setup_env.sh           # Environment setup (Linux)
├── setup_env.bat          # Environment setup (Windows)
├── requirements.txt       # Python dependencies
├── pytest.ini             # Pytest configuration
└── .gitignore             # Git ignore rules
```

**Important Files:**
- `main.py` - Production API server (run with: `uvicorn main:app`)
- `test_workflow.py` - Development test runner (run with: `python test_workflow.py`)
- `setup_model.py` - Download Places365 model (run once before first use)

## Coding Standards

### 0. Service Initialization Pattern (CRITICAL!)

**ALWAYS use ServiceFactory for service initialization:**

```python
# ✅ CORRECT - Use ServiceFactory (creates services in correct dependency order)
from core.service_factory import ServiceFactory
from core.container import get_container

container = get_container()
factory = ServiceFactory(container)
await factory.create_all_services()  # Async initialization with dependency management

# Services are now available in container
claude_service = container.get(ServiceNames.CLAUDE)
```

**Key Points:**
- ServiceFactory handles **dependency order** automatically
- ServiceFactory performs **async initialization** correctly
- All services registered via **factory methods** with error handling
- Container manages **singleton lifecycle**

**Example from `main.py`:**
```python
async def lifespan(app: FastAPI):
    # Initialize services using factory
    container = get_container()
    factory = ServiceFactory(container)
    await factory.create_all_services()

    # Store in app state for access in routes
    app.state.container = container

    yield

    # Cleanup
    db_service = container.get(ServiceNames.DATABASE)
    if db_service:
        await db_service.close()
```

### 1. Always Use Dependency Injection
```python
# ✅ GOOD - Use ServiceContainer
from core.container import get_container, ServiceNames

container = get_container()
claude_service = container.get(ServiceNames.CLAUDE)

# ❌ BAD - Direct instantiation
claude_service = ClaudeService(api_key=...)
```

### 2. Always Use DTOs for Data Transfer
```python
# ✅ GOOD - Type-safe DTOs
from models.dtos import EventDTO, AboutGameDTO

event = EventDTO(
    event_id="evt_123",
    about_game=AboutGameDTO(title="...", description="...")
)

# ❌ BAD - Plain dictionaries
event = {"event_id": "evt_123", "about_game": {...}}
```

### 3. Always Use Custom Exceptions (with Rich Context!)

```python
# ✅ GOOD - Domain exceptions with context details
from core.exceptions import ImageProcessingError, DatabaseError

raise ImageProcessingError(
    "Invalid image format",
    details={
        "file_path": image_path,
        "expected_formats": [".jpg", ".png"],
        "actual_format": ".gif"
    }
)

# Exception has .to_dict() method for API responses
try:
    process_image(path)
except ImageProcessingError as e:
    return JSONResponse(
        status_code=400,
        content=e.to_dict()  # Returns structured error with details
    )

# ❌ BAD - Generic exceptions without context
raise ValueError("Invalid image format")
```

**Available Exception Types:**
- `ImageProcessingError` - Image operations
- `DatabaseError` - Database operations
- `ConfigurationError` - Config issues
- `ServiceInitializationError` - Service startup
- `ValidationError` - Input validation
- `EventNotFoundError` - Event not in DB
- `ExternalAPIError` - External API calls (Claude, Voyage)
- `FAISIndexError` - FAISS operations
- `ModelLoadError` - ML model loading
- `AuthenticationError` - Auth issues

### 4. Always Use Validators
```python
# ✅ GOOD - Centralized validation
from utils.validators import RequestValidator

RequestValidator.validate_image_file(file)

# ❌ BAD - Inline validation
if file.size > MAX_SIZE:
    raise ValueError("File too large")
```

### 5. Repository Pattern for Database
```python
# ✅ GOOD - Use repository
from repositories.event_repository import EventRepository

repo = EventRepository(db_service)
event = repo.find_by_id(event_id)

# ❌ BAD - Direct SQL in service
cursor.execute("SELECT * FROM events WHERE id = %s", (event_id,))
```

### 6. API Response Pattern (Standardized!)

**ALWAYS wrap API responses with APIResponse:**

```python
# ✅ CORRECT - Use generic APIResponse wrapper
from models.responses import success_response, error_response, APIResponse

@router.get("/events/{event_id}")
async def get_event(event_id: str):
    try:
        event = await repo.find_by_id(event_id)

        # Use helper function for success
        return success_response(
            data=event,
            message="Event retrieved successfully"
        )
    except EventNotFoundError as e:
        # Use helper function for errors
        return error_response(
            error=e,
            status_code=404
        )

# Response structure:
# {
#   "success": true,
#   "data": {...},
#   "message": "...",
#   "metadata": {
#     "timestamp": "2024-01-15T10:30:00Z",
#     "processing_time_ms": 123.45
#   }
# }

# ❌ BAD - Plain dict responses
return {"success": True, "data": event}
```

**Response Helper Functions:**
- `success_response(data, message)` - For successful operations
- `error_response(error, status_code)` - For error responses
- `paginated_response(items, total, page, page_size)` - For paginated data

### 7. FastAPI Route Patterns

**Access services via app.state.container:**

```python
# ✅ CORRECT - Get services from app state
from fastapi import Request

@router.post("/similarity/search")
async def search_similar(request: Request, payload: SearchRequest):
    # Get container from app state
    container = request.app.state.container

    # Get services from container
    similarity_service = container.get(ServiceNames.EVENT_SIMILARITY)

    # Use service
    results = await similarity_service.find_similar(payload)

    return success_response(data=results)

# ❌ BAD - Direct service instantiation in route
similarity_service = EventSimilarityService()  # Missing dependencies!
```

### 8. Logging with Emojis (Convention!)

**Use emoji prefixes for log readability:**

```python
# ✅ CORRECT - Emoji logging convention
logger.info("✅ Service initialized successfully")
logger.warning("⚠️  Low similarity score detected")
logger.error("❌ Failed to process image")
logger.debug("🔍 Searching for similar events")

# Common emoji conventions:
# ✅ - Success/Complete
# ❌ - Error/Failed
# ⚠️  - Warning
# 🔍 - Search/Query
# 📥 - Download/Input
# 📤 - Upload/Output
# 🚀 - Start/Launch
# 🧹 - Cleanup
# 🔄 - Retry/Reload
# 💾 - Database operation
# 🖼️  - Image operation
# 📝 - Text operation
# 🤖 - AI/ML operation
# ⏱️  - Timing/Performance
# 🎯 - Target/Goal

# ❌ BAD - Plain text logs
logger.info("Service initialized")  # Less scannable
```

### 9. Async/Await Patterns

**Database and external API calls MUST be async:**

```python
# ✅ CORRECT - Async for I/O operations
async def get_event_similarity(event_id: str):
    # Database calls - async
    event = await repo.find_by_id(event_id)

    # External API calls - async
    analysis = await claude_service.analyze_image(image_path)
    embedding = await voyage_client.embed(text)

    # CPU-bound operations - can be sync
    similarity_score = calculate_similarity(vec1, vec2)

    return results

# ❌ BAD - Blocking I/O in async function
def get_event_similarity(event_id: str):  # Should be async
    event = repo.find_by_id(event_id)  # Blocking call!
```

## Configuration Management

### Environment Files
- `.env.development` - Development settings (default)
- `.env.production` - Production settings
- `.env.ubuntu` - Ubuntu-specific paths
- `.env.windows` - Windows-specific paths

### Switch Environment
```bash
ENV=production uvicorn main:app --reload
# or
./setup_env.sh prod
```

### Access Configuration
```python
from config import Config

# ✅ Type-safe configuration access
api_key = Config.CLAUDE_API_KEY
model_path = Config.PLACES365_MODEL_PATH
db_config = Config.get_db_config()
```

## Testing Standards

### Test Structure
```python
# tests/unit/test_something.py
import pytest
from unittest.mock import Mock, patch

class TestSomething:
    @pytest.fixture
    def mock_service(self):
        """Setup mock service"""
        return Mock()

    def test_something(self, mock_service):
        """Test description"""
        # Arrange
        # Act
        # Assert
```

### Run Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test
pytest tests/unit/test_validators.py -v
```

## Common Tasks

### Adding a New Service
1. Create service class in `services/`
2. Register in `core/service_factory.py`
3. Add service name to `ServiceNames` in `core/container.py`
4. Create corresponding DTO in `models/dtos.py`
5. Add unit tests in `tests/unit/`

### Adding a New Endpoint
1. Create router in `routers/`
2. Define request/response models in `models/responses.py`
3. Use dependency injection to get services
4. Add validation using `RequestValidator`
5. Handle exceptions properly
6. Register router in `main.py`

### Adding a New Exception Type
1. Add exception class in `core/exceptions.py`
2. Inherit from appropriate base exception
3. Add exception handler in `main.py` if needed
4. Use in services/routers

## AI/ML Components

### Places365 Model
- **Purpose:** Extract visual features from images
- **Location:** `models/places365.py`
- **Dimension:** 2048 features
- **Download:** `python setup_model.py`

### FAISS Index
- **Purpose:** Fast similarity search
- **Manager:** `utils/faiss_manager.py`
- **Index Type:** IndexFlatL2 (configurable)
- **Location:** `index/` directory

### Claude AI
- **Purpose:** Semantic understanding and extraction
- **Service:** `services/claude_service.py`
- **Model:** claude-3-5-sonnet-20241022 (configurable)
- **Prompts:** Managed by `utils/prompt_manager.py`

### Voyage Embeddings
- **Purpose:** Text embedding generation
- **Processor:** `utils/text_processor.py`
- **Model:** voyage-2 (configurable)
- **Dimension:** 1024 features

## Important Notes

### DO
- ✅ Use type hints everywhere
- ✅ Use DTOs for all data structures
- ✅ Use ServiceContainer for all dependencies
- ✅ Use custom exceptions for error handling
- ✅ Write unit tests for new features
- ✅ Update documentation
- ✅ Follow existing naming conventions
- ✅ Use async/await for I/O operations
- ✅ Log important operations
- ✅ Validate all inputs

### DON'T
- ❌ Create direct service instances
- ❌ Use plain dictionaries for data
- ❌ Use generic exceptions
- ❌ Hardcode configuration values
- ❌ Skip validation
- ❌ Skip tests
- ❌ Break SOLID principles
- ❌ Use global variables
- ❌ Commit sensitive data (API keys, passwords)
- ❌ Modify existing tests without understanding

## File Naming Conventions
- Services: `*_service.py`
- Repositories: `*_repository.py`
- DTOs: `dtos.py`, `responses.py`
- Utilities: `*_utils.py`, `*_processor.py`, `*_manager.py`
- Tests: `test_*.py`

## Git Workflow
```bash
# Check status
git status

# Create feature branch
git checkout -b feature/your-feature

# Commit with meaningful messages
git commit -m "feat: Add image similarity endpoint"

# Use conventional commits
# feat: | fix: | docs: | test: | refactor: | perf: | chore:
```

## Documentation References
- Main docs: `docs/README.md`
- Quick Start: `docs/QUICKSTART.md`
- Migration: `docs/MIGRATION.md`
- Architecture: `docs/README_REFACTOR.md`
- Code Analysis: `docs/CODE_ANALYSIS.md`

## Performance Considerations
- Use async/await for I/O operations
- Cache FAISS indices in memory
- Batch database operations when possible
- Use connection pooling for database
- Monitor API response times
- Consider rate limiting for external APIs (Claude, Voyage)

## Security Best Practices
- Never commit `.env` files
- Validate all file uploads (size, type, content)
- Sanitize database inputs (use parameterized queries)
- Set proper CORS configuration
- Use HTTPS in production
- Implement rate limiting
- Monitor API usage

## Troubleshooting
1. **Model not found:** Run `python setup_model.py`
2. **Database connection failed:** Check `.env` credentials
3. **API key errors:** Verify VOYAGE_API_KEY and CLAUDE_API_KEY
4. **Import errors:** Ensure all dependencies installed: `pip install -r requirements.txt`
5. **Tests failing:** Reset service container between tests
