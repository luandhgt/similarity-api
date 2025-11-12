# Project Guidelines - Image Similarity API

## Project Mission
Build a production-ready, multi-modal event similarity search API that combines visual (Places365) and textual (Voyage) embeddings to find similar events using FAISS vector search and Claude AI for semantic understanding.

> **📝 Note:** Xem `.claude/instructions.md` để biết policy về việc cập nhật documentation khi thay đổi cấu trúc dự án.

## Tech Stack

### Core Technologies
- **Python 3.8+** - Programming language
- **FastAPI** - Modern web framework for APIs
- **PostgreSQL** - Relational database
- **Pydantic** - Data validation and settings

### Machine Learning / AI
- **PyTorch** - Deep learning framework
- **Places365 (ResNet50)** - Scene recognition CNN (2048-dim features)
- **FAISS** - Facebook AI Similarity Search (vector indexing)
- **Voyage AI** - Text embedding generation (1024-dim)
- **Claude AI (Anthropic)** - Natural language understanding

### Development Tools
- **pytest** - Testing framework
- **python-dotenv** - Environment variable management
- **uvicorn** - ASGI server

## Code Quality Standards

### Type Safety
```python
# ✅ ALWAYS use type hints
def process_image(image_path: str, resize: bool = True) -> np.ndarray:
    pass

# ✅ Use Pydantic for complex types
from pydantic import BaseModel

class EventRequest(BaseModel):
    event_id: str
    title: str
    description: Optional[str] = None
```

### Error Handling
```python
# ✅ Use custom exceptions
from core.exceptions import ImageProcessingError

try:
    result = process_image(path)
except FileNotFoundError:
    raise ImageProcessingError(f"Image not found: {path}")

# ✅ Log errors properly
import logging
logger = logging.getLogger(__name__)

logger.error(f"Failed to process image: {e}", exc_info=True)
```

### Documentation
```python
# ✅ Use docstrings for all public methods
def find_similar_events(
    query_event_id: str,
    top_k: int = 10,
    threshold: float = 0.7
) -> List[EventDTO]:
    """
    Find events similar to the given query event.

    Args:
        query_event_id: ID of the event to find similarities for
        top_k: Number of similar events to return
        threshold: Minimum similarity score (0-1)

    Returns:
        List of similar events sorted by similarity score

    Raises:
        EventNotFoundError: If query_event_id doesn't exist
        DatabaseError: If database operation fails
    """
    pass
```

### Logging
```python
# ✅ Use structured logging
logger.info(f"Processing event {event_id}")
logger.debug(f"Image dimensions: {width}x{height}")
logger.warning(f"Low similarity score: {score:.3f}")
logger.error(f"Failed to process: {e}", exc_info=True)

# ❌ Don't use print statements
print("Processing event...")  # BAD
```

## API Design Guidelines

### Endpoint Naming
- Use REST conventions: `/api/resource` or `/api/resource/{id}`
- Use kebab-case: `/api/event-similarity/find`
- Version APIs if needed: `/api/v2/similarity`

### Request/Response Format
```python
# ✅ Use Pydantic models
from pydantic import BaseModel, Field

class SimilarityRequest(BaseModel):
    event_id: str = Field(..., description="Event ID to find similarities")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")

class SimilarityResponse(BaseModel):
    query_event_id: str
    similar_events: List[EventDTO]
    processing_time_ms: float
```

### Error Responses
```python
# ✅ Consistent error format
{
    "detail": {
        "error_type": "ImageProcessingError",
        "message": "Invalid image format",
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

## Database Guidelines

### Use Repository Pattern
```python
# ✅ Repository handles all DB operations
class EventRepository:
    def find_by_id(self, event_id: str) -> Optional[EventDTO]:
        """Find event by ID"""
        pass

    def save(self, event: EventDTO) -> bool:
        """Save event to database"""
        pass
```

### Use Parameterized Queries
```python
# ✅ ALWAYS use parameterized queries
cursor.execute(
    "SELECT * FROM events WHERE event_id = %s AND status = %s",
    (event_id, status)
)

# ❌ NEVER use string formatting
cursor.execute(f"SELECT * FROM events WHERE event_id = '{event_id}'")  # SQL injection!
```

### Transaction Management
```python
# ✅ Use context managers for transactions
with db_service.get_connection() as conn:
    with conn.cursor() as cursor:
        cursor.execute(query, params)
        conn.commit()
```

## Testing Guidelines

### Test Organization
```
tests/
├── unit/              # Unit tests (isolated, mocked dependencies)
│   ├── test_validators.py
│   ├── test_dtos.py
│   └── test_container.py
├── integration/       # Integration tests (real dependencies)
│   └── test_api.py
└── conftest.py       # Shared fixtures
```

### Test Naming
```python
class TestEventSimilarityService:
    def test_find_similar_events_returns_correct_count(self):
        """Test that find_similar_events returns exactly top_k results"""
        pass

    def test_find_similar_events_raises_error_when_event_not_found(self):
        """Test error handling for non-existent event"""
        pass
```

### Use Fixtures
```python
# conftest.py
@pytest.fixture
def mock_claude_service():
    """Mock Claude service for testing"""
    service = Mock(spec=ClaudeService)
    service.analyze_image.return_value = AboutGameDTO(...)
    return service

# test file
def test_extract_about_game(mock_claude_service):
    """Test about game extraction"""
    result = extract_about_game(image_path, mock_claude_service)
    assert result.title is not None
```

### Test Coverage Goals
- Critical paths: 100%
- Services: >90%
- Utilities: >80%
- Overall: >75%

## Performance Guidelines

### Async Operations
```python
# ✅ Use async for I/O operations
async def get_event(event_id: str) -> EventDTO:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

### Caching Strategy
```python
# ✅ Cache expensive operations
from functools import lru_cache

@lru_cache(maxsize=1000)
def load_places365_model(model_path: str):
    """Load and cache model (loaded once)"""
    return torch.load(model_path)
```

### Batch Operations
```python
# ✅ Process in batches when possible
def embed_images_batch(image_paths: List[str]) -> np.ndarray:
    """Process multiple images in one forward pass"""
    batch_tensor = torch.stack([preprocess(img) for img in images])
    with torch.no_grad():
        embeddings = model(batch_tensor)
    return embeddings.numpy()
```

## Security Guidelines

### Input Validation
```python
# ✅ Validate ALL inputs
from utils.validators import RequestValidator

RequestValidator.validate_image_file(file)
RequestValidator.validate_event_id(event_id)
RequestValidator.validate_similarity_threshold(threshold)
```

### File Upload Security
```python
# ✅ Check file type, size, and content
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

def validate_uploaded_file(file: UploadFile):
    # Check extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(f"Invalid file type: {ext}")

    # Check size
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_FILE_SIZE:
        raise ValidationError(f"File too large: {size} bytes")

    # Verify it's actually an image
    try:
        Image.open(file.file)
    except Exception:
        raise ValidationError("Invalid image file")
```

### API Key Management
```python
# ✅ Load from environment
from config import Config
api_key = Config.CLAUDE_API_KEY

# ❌ NEVER hardcode
api_key = "sk-ant-..."  # BAD!

# ❌ NEVER commit .env files
# Add to .gitignore
```

## Dependency Management

### Service Container Pattern
```python
# ✅ Register services once at startup
from core.service_factory import ServiceFactory

def setup_services():
    """Initialize all services"""
    container = get_container()
    factory = ServiceFactory(container)
    factory.create_all_services()

# ✅ Get services via container
container = get_container()
claude_service = container.get(ServiceNames.CLAUDE)
```

### Avoid Circular Dependencies
```python
# ✅ Use dependency injection
class EventSimilarityService:
    def __init__(self, db_service: DatabaseService, claude_service: ClaudeService):
        self.db = db_service
        self.claude = claude_service

# ❌ Don't import at module level if circular
from services.event_similarity_service import EventSimilarityService  # BAD
```

## Configuration Management

### Environment-Specific Configs
```bash
# Development (default)
ENV=development uvicorn main:app --reload

# Production
ENV=production uvicorn main:app --workers 4

# Platform-specific
ENV=ubuntu uvicorn main:app
ENV=windows uvicorn main:app
```

### Configuration Access
```python
# ✅ Use Config class
from config import Config

if Config.IS_DEVELOPMENT:
    logger.setLevel(logging.DEBUG)

# ✅ Validate on startup
errors = Config.validate()
if errors:
    raise ValueError(f"Config errors: {errors}")
```

## Version Control Guidelines

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Features
git commit -m "feat: Add event similarity endpoint"
git commit -m "feat(api): Add image upload validation"

# Bug fixes
git commit -m "fix: Handle missing image files gracefully"
git commit -m "fix(db): Correct event_id query parameter"

# Documentation
git commit -m "docs: Update API endpoint documentation"

# Tests
git commit -m "test: Add unit tests for validators"

# Refactoring
git commit -m "refactor: Extract image processing to utility"

# Performance
git commit -m "perf: Optimize FAISS index loading"

# Chores
git commit -m "chore: Update dependencies"
```

### Branch Naming
```bash
# Features
feature/event-similarity-endpoint
feature/image-caching

# Bug fixes
fix/database-connection-leak
fix/invalid-image-handling

# Hotfixes
hotfix/critical-security-patch
```

### Files to Ignore
```gitignore
# Environment
.env
.env.*
!.env.example

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/

# Models & Data
models/*.pth.tar
models/*.pt
index/*.faiss
shared/uploads/*

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

## Deployment Checklist

### Pre-deployment
- [ ] All tests passing
- [ ] Environment variables configured
- [ ] Database migrations applied
- [ ] API keys verified
- [ ] Models downloaded
- [ ] FAISS indices built
- [ ] Logs directory created
- [ ] Security scan passed

### Production Settings
```bash
# .env.production
ENV=production
API_RELOAD=false
LOG_LEVEL=WARNING
WORKER_TIMEOUT=300
MAX_CONCURRENT_REQUESTS=100
CACHE_ENABLED=true
```

## Monitoring & Observability

### Logging Levels
- **DEBUG:** Development debugging
- **INFO:** Normal operations (API requests, service initialization)
- **WARNING:** Recoverable issues (low similarity scores, retries)
- **ERROR:** Errors requiring attention (API failures, DB errors)
- **CRITICAL:** System failures (service unavailable, OOM)

### Metrics to Track
- API response times (p50, p95, p99)
- Error rates by endpoint
- Database connection pool usage
- FAISS search latency
- Claude API usage and costs
- Voyage API usage and costs
- Model inference time

## Development Workflow

### 1. Setup Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Setup environment
./setup_env.sh dev

# Download models
python setup_model.py
```

### 2. Before Starting Work
```bash
# Pull latest changes
git pull origin main

# Create feature branch
git checkout -b feature/your-feature

# Verify environment
python -c "from config import Config; Config.print_config()"
```

### 3. During Development
```bash
# Run API with auto-reload
uvicorn main:app --reload

# Run tests in watch mode
pytest --watch

# Check code quality
flake8 .
black .
mypy .
```

### 4. Before Committing
```bash
# Run all tests
pytest

# Check coverage
pytest --cov=. --cov-report=html

# Format code
black .

# Check types
mypy .

# Stage changes
git add .

# Commit with conventional message
git commit -m "feat: Add your feature"
```

## Resources & References

### Documentation
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Pydantic Docs](https://docs.pydantic.dev/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [PyTorch Docs](https://pytorch.org/docs/)

### Internal Documentation
- Main: [docs/README.md](../docs/README.md)
- Architecture: [docs/README_REFACTOR.md](../docs/README_REFACTOR.md)
- Migration: [docs/MIGRATION.md](../docs/MIGRATION.md)
- Quick Start: [docs/QUICKSTART.md](../docs/QUICKSTART.md)

### AI APIs
- [Claude AI Documentation](https://docs.anthropic.com/)
- [Voyage AI Documentation](https://docs.voyageai.com/)

## Support & Contact

### Getting Help
1. Check documentation in `docs/`
2. Review existing code examples
3. Check issue tracker (if available)
4. Ask team members

### Reporting Issues
1. Check if issue already exists
2. Provide minimal reproducible example
3. Include environment details (Python version, OS, dependencies)
4. Include relevant logs
