# Code Structure Analysis Report

## Executive Summary

Phân tích toàn diện cấu trúc code của image-similarity-api, tìm ra các vấn đề về:
- Code trùng lặp
- Cấu trúc logic không hợp lý
- Chồng chéo responsibilities
- Code thừa/không dùng

---

## 🔴 Critical Issues (Ưu tiên cao)

### 1. **Singleton Pattern Implementation Issues**

#### Vấn đề:
Có **2 patterns khác nhau** cho service initialization:

**Pattern 1: Global Singleton** (trong `about_extraction_service.py`)
```python
# services/about_extraction_service.py
about_extraction_service = AboutExtractionService()

# Được import trực tiếp
from services.about_extraction_service import about_extraction_service
```

**Pattern 2: Lazy Singleton với Function** (trong `claude_service.py`)
```python
# services/claude_service.py
_claude_service_instance = None

def get_claude_service():
    global _claude_service_instance
    if _claude_service_instance is None:
        _claude_service_instance = ClaudeService()
    return _claude_service_instance
```

**Pattern 3: Dependency Injection** (trong `event_similarity_service.py`)
```python
# services/event_similarity_service.py
class EventSimilarityService:
    def __init__(self, claude_service: ClaudeService,
                 voyage_client: VoyageClient, ...):
        self.claude_service = claude_service
        # ...
```

#### Tác động:
- ❌ **Inconsistent** - khó maintain
- ❌ **Testing khó** - không mock được dễ dàng
- ❌ **Circular dependencies risk** - services phụ thuộc lẫn nhau

#### Đề xuất:
**Unified Service Container Pattern**
```python
# services/service_container.py
class ServiceContainer:
    _instance = None

    def __init__(self):
        self._services = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ServiceContainer()
        return cls._instance

    def register(self, name: str, service):
        self._services[name] = service

    def get(self, name: str):
        return self._services.get(name)

# Usage
container = ServiceContainer.get_instance()
container.register('claude', ClaudeService())
```

**Priority:** 🔴 HIGH
**Effort:** 2-3 giờ

---

### 2. **Duplicate Service Initialization Logic**

#### Vấn đề:
Service initialization bị **duplicate** ở nhiều nơi:

**Nơi 1: `main.py` (FastAPI app)**
```python
# main.py lines 32-140
async def lifespan(app: FastAPI):
    # Initialize Places365
    from models.places365 import get_places365_model
    places_model = get_places365_model()

    # Initialize Voyage
    from utils.text_processor import get_voyage_client
    voyage_client = get_voyage_client()

    # Initialize Claude
    from services.claude_service import ClaudeService
    claude_service = ClaudeService()

    # Initialize Database
    from services.database_service import DatabaseService
    db_service = DatabaseService()
    await db_service.initialize()

    # Initialize Event Similarity
    event_similarity_service = EventSimilarityService(...)
    # ...
```

**Nơi 2: `tests/service_initializer.py`**
```python
# tests/service_initializer.py lines 14-174
async def initialize_services(verbose: bool = True):
    # Initialize Places365
    from models.places365 import get_places365_model
    places_model = get_places365_model()

    # Initialize Voyage
    from utils.text_processor import get_voyage_client
    voyage_client = get_voyage_client()

    # Initialize Claude
    from services.claude_service import ClaudeService
    claude_service = ClaudeService()

    # Initialize Database
    from services.database_service import DatabaseService
    db_service = DatabaseService()
    await db_service.initialize()

    # Initialize Event Similarity
    event_similarity_service = EventSimilarityService(...)
    # ...
```

**Nơi 3: `setup_model.py`**
```python
# setup_model.py - riêng biệt cho model download
def download_places365_model():
    # Download logic
    pass

def verify_model():
    import torch
    checkpoint = torch.load(model_path)
    # ...
```

#### Tác động:
- ❌ **100+ dòng code trùng lặp**
- ❌ **Khó maintain** - sửa 1 nơi phải sửa nhiều nơi
- ❌ **Inconsistent** - có thể khác nhau về error handling

#### Đề xuất:
**Tạo Shared Service Factory**
```python
# services/service_factory.py
class ServiceFactory:
    @staticmethod
    async def initialize_all_services(verbose: bool = False) -> Dict[str, Any]:
        """Single source of truth for service initialization"""
        services = {}

        # Places365
        services['places365'] = await ServiceFactory._init_places365(verbose)

        # Voyage
        services['voyage_client'] = await ServiceFactory._init_voyage(verbose)

        # Claude
        services['claude'] = await ServiceFactory._init_claude(verbose)

        # Database
        services['database'] = await ServiceFactory._init_database(verbose)

        # Event Similarity
        services['event_similarity'] = await ServiceFactory._init_event_similarity(
            services['claude'],
            services['voyage_client'],
            services['database'],
            verbose
        )

        return services

    @staticmethod
    async def _init_places365(verbose: bool):
        if verbose:
            print("Loading Places365 model...")
        from models.places365 import get_places365_model
        return get_places365_model()

    # ... other _init methods
```

**Priority:** 🔴 HIGH
**Effort:** 3-4 giờ

---

### 3. **setup_model.py Should Use Config Module**

#### Vấn đề:
`setup_model.py` hardcodes paths thay vì dùng config:

```python
# setup_model.py lines 15-18
models_dir = Path("models")  # ❌ Hardcoded
model_path = models_dir / "resnet50_places365.pth.tar"  # ❌ Hardcoded
```

**So với:**
```python
# config.py có sẵn
MODELS_PATH: str = os.getenv('MODELS_PATH', str(PROJECT_ROOT / 'models'))
PLACES365_MODEL_FILE: str = os.getenv('PLACES365_MODEL_FILE', 'resnet50_places365.pth.tar')
```

#### Tác động:
- ❌ Không consistent với refactored code
- ❌ Không work khi user thay đổi MODELS_PATH
- ❌ Không flexible cho Windows/Ubuntu

#### Đề xuất:
```python
# setup_model.py - Refactored
from config import config

def download_places365_model():
    """Download Places365 ResNet50 pretrained weights"""

    # Use config module
    models_dir = Path(config.MODELS_PATH)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / config.PLACES365_MODEL_FILE

    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        # ...
```

**Priority:** 🔴 HIGH
**Effort:** 30 phút

---

## 🟡 Medium Priority Issues

### 4. **Overlapping Router Logic**

#### Vấn đề:
Các routers có **duplicate validation và error handling**:

**Router 1: `routers/about_extraction.py`**
```python
# Lines 43-80
async def extract_about(request: ExtractAboutRequest):
    try:
        # Validate folder exists
        folder_path = Path(request.shared_uploads_path) / request.folder_name
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")

        # Validate output format
        available_formats = about_extraction_service.get_supported_formats()
        if request.output_format not in available_formats:
            logger.warning(f"⚠️ Unknown output format '{request.output_format}', using 'default'")
            request.output_format = "default"

        # Log request details
        logger.info(f"🎯 Processing request:")
        logger.info(f"   - Folder: {folder_path}")
        # ...
```

**Router 2: `routers/event_similarity.py`**
```python
# Lines 162-210
async def find_similar_events(request: FindSimilarEventsRequest):
    try:
        # Validate folder exists
        folder_path = Path(request.shared_uploads_path) / request.folder_name
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")

        # Log request details
        logger.info(f"🔍 Processing similarity search request:")
        logger.info(f"   Event: {request.event_name}")
        # ...
```

#### Tác động:
- ❌ ~50 dòng validation logic trùng lặp
- ❌ Inconsistent error messages
- ❌ Khó maintain

#### Đề xuất:
**Shared Request Validator**
```python
# utils/request_validator.py
class RequestValidator:
    @staticmethod
    def validate_folder_path(shared_uploads_path: str, folder_name: str) -> Path:
        """Validate folder exists and return Path"""
        folder_path = Path(shared_uploads_path) / folder_name
        if not folder_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Folder not found: {folder_path}"
            )
        return folder_path

    @staticmethod
    def validate_output_format(format_name: str, available_formats: List[str], default: str = "default") -> str:
        """Validate and normalize output format"""
        if format_name not in available_formats:
            logger.warning(f"⚠️ Unknown format '{format_name}', using '{default}'")
            return default
        return format_name

# Usage in routers
folder_path = RequestValidator.validate_folder_path(
    request.shared_uploads_path,
    request.folder_name
)
```

**Priority:** 🟡 MEDIUM
**Effort:** 2 giờ

---

### 5. **Image Processing Duplication**

#### Vấn đề:
Image loading/processing logic bị duplicate:

**Nơi 1: `services/event_similarity_service.py`**
```python
async def _load_images_from_folder(self, shared_uploads_path, folder_name, expected_count):
    folder_path = os.path.join(shared_uploads_path, folder_name)

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))

    if len(image_files) != expected_count:
        logger.warning(f"Expected {expected_count} images, found {len(image_files)}")
    # ...
```

**Nơi 2: `services/about_extraction_service.py`**
```python
def _find_images_in_folder(self, folder_path: str) -> List[str]:
    folder = Path(folder_path)
    image_paths = []

    for ext in self.supported_extensions:
        image_paths.extend(folder.glob(f"*{ext}"))
        image_paths.extend(folder.glob(f"*{ext.upper()}"))

    return [str(p) for p in sorted(image_paths)]
```

#### Tác động:
- ❌ Duplicate logic (~30 dòng)
- ❌ Khác nhau về extensions
- ❌ Inconsistent error handling

#### Đề xuất:
**Shared Image Utilities**
```python
# utils/image_utils.py
class ImageUtils:
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    @staticmethod
    def find_images_in_folder(folder_path: str, expected_count: int = None) -> List[Path]:
        """Find all supported images in folder"""
        folder = Path(folder_path)

        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")

        image_files = []
        for ext in ImageUtils.SUPPORTED_EXTENSIONS:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))

        image_files = sorted(image_files)

        if expected_count is not None and len(image_files) != expected_count:
            logger.warning(f"Expected {expected_count} images, found {len(image_files)}")

        return image_files
```

**Priority:** 🟡 MEDIUM
**Effort:** 1-2 giờ

---

### 6. **Logging Duplication**

#### Vấn đề:
Logging patterns lặp lại khắp nơi:

```python
# Pattern 1: Request logging
logger.info(f"📥 Received request for: {something}")
logger.info(f"🎯 Processing request:")
logger.info(f"   - Field1: {value1}")
logger.info(f"   - Field2: {value2}")
```

```python
# Pattern 2: Success/Error
logger.info(f"✅ Successfully completed: {something}")
logger.error(f"❌ Failed: {something}")
```

#### Đề xuất:
**Structured Logging Helper**
```python
# utils/logging_helper.py
class LogHelper:
    @staticmethod
    def log_request(endpoint: str, **kwargs):
        logger.info(f"📥 Received request: {endpoint}")
        for key, value in kwargs.items():
            logger.info(f"   - {key}: {value}")

    @staticmethod
    def log_success(operation: str, duration: float = None):
        msg = f"✅ {operation} completed"
        if duration:
            msg += f" in {duration:.2f}s"
        logger.info(msg)

    @staticmethod
    def log_error(operation: str, error: Exception):
        logger.error(f"❌ {operation} failed: {error}")
```

**Priority:** 🟡 MEDIUM
**Effort:** 1 giờ

---

## 🟢 Low Priority Issues

### 7. **Dead Code / Unused Imports**

#### Tìm thấy:
```python
# services/claude_service.py
import os  # ✅ Used via config now, can remove if fully migrated
from dotenv import load_dotenv  # ❌ Not needed, config handles this
```

```python
# services/about_extraction_service.py
import os  # Used, OK
import time  # Used, OK
```

#### Đề xuất:
Run `autoflake` hoặc `pylint` để tìm unused imports:
```bash
autoflake --remove-all-unused-imports --in-place services/*.py
```

**Priority:** 🟢 LOW
**Effort:** 30 phút

---

### 8. **Magic Numbers**

#### Vấn đề:
```python
# services/event_similarity_service.py
self.similarity_config = {
    "top_k": 10,  # Magic number
    "individual_search_k": 20  # Magic number
}

# services/claude_service.py
command_timeout=60,  # Magic number
max_size=10,  # Magic number
```

#### Đề xuất:
Move to config:
```python
# config.py
SIMILARITY_TOP_K: int = int(os.getenv('SIMILARITY_TOP_K', '10'))
SIMILARITY_SEARCH_K: int = int(os.getenv('SIMILARITY_SEARCH_K', '20'))
DB_POOL_MAX_SIZE: int = int(os.getenv('DB_POOL_MAX_SIZE', '10'))
DB_COMMAND_TIMEOUT: int = int(os.getenv('DB_COMMAND_TIMEOUT', '60'))
```

**Priority:** 🟢 LOW
**Effort:** 1 giờ

---

## 📊 Code Statistics

### Duplication Summary
| Category | Duplicated Lines | Files Affected |
|----------|-----------------|----------------|
| Service Init | ~150 | 2 (main.py, service_initializer.py) |
| Request Validation | ~50 | 2 routers |
| Image Finding | ~30 | 2 services |
| Logging | ~40 | All files |
| **Total** | **~270** | **10+ files** |

### File Size Analysis
```
services/event_similarity_service.py  : 1,471 lines  ⚠️ Too large
services/claude_service.py            :   339 lines  ✅ OK
services/database_service.py          :   370 lines  ✅ OK
routers/event_similarity.py           :   314 lines  ✅ OK
routers/about_extraction.py           :   211 lines  ✅ OK
utils/faiss_manager.py                :   382 lines  ⚠️ Could split
```

---

## 🎯 Refactoring Roadmap

### Phase 1: Critical Fixes (Week 1)
1. ✅ **Unified Service Container** (Priority: HIGH)
   - Create `services/service_container.py`
   - Migrate all services to use it
   - Update main.py and tests

2. ✅ **Consolidate Service Initialization** (Priority: HIGH)
   - Create `services/service_factory.py`
   - Remove duplication from main.py and tests
   - Add comprehensive error handling

3. ✅ **Fix setup_model.py** (Priority: HIGH)
   - Use config module
   - Add environment support
   - Improve error messages

### Phase 2: Medium Priority (Week 2)
4. ✅ **Shared Request Validator** (Priority: MEDIUM)
   - Create `utils/request_validator.py`
   - Refactor routers to use it
   - Add unit tests

5. ✅ **Image Utils Consolidation** (Priority: MEDIUM)
   - Create `utils/image_utils.py`
   - Extract common image operations
   - Update services

6. ✅ **Logging Helper** (Priority: MEDIUM)
   - Create `utils/logging_helper.py`
   - Standardize logging patterns
   - Update all files

### Phase 3: Low Priority (Week 3)
7. ✅ **Clean Up Unused Code** (Priority: LOW)
   - Run autoflake
   - Remove dead code
   - Update imports

8. ✅ **Extract Magic Numbers** (Priority: LOW)
   - Add to config.py
   - Update all references
   - Document defaults

---

## 📝 Recommendations

### Immediate Actions
1. ⭐ **Setup ServiceContainer** - sẽ giải quyết nhiều vấn đề dependency
2. ⭐ **Consolidate service_init** - giảm 150+ dòng duplicate
3. ⭐ **Fix setup_model.py** - consistency với refactored code

### Long-term Improvements
1. 📚 **Add Unit Tests** - hiện tại không có tests
2. 📚 **Add Integration Tests** - test end-to-end flows
3. 📚 **Add Type Checking** - use mypy
4. 📚 **Add Linting** - use pylint/flake8
5. 📚 **Add CI/CD** - automate testing

### Architecture Improvements
1. 🏗️ **Split event_similarity_service.py** - quá lớn (1471 lines)
2. 🏗️ **Add Repository Pattern** - tách database logic
3. 🏗️ **Add DTO Objects** - thay vì Dict[str, Any]
4. 🏗️ **Add Service Interfaces** - để mock dễ hơn

---

## ⚠️ Breaking Changes Risk

### Low Risk
- ServiceContainer pattern (backward compatible)
- Image utils consolidation
- Logging helper

### Medium Risk
- Service factory refactor (changes initialization)
- Request validator (changes error messages)

### High Risk
- Splitting event_similarity_service.py
- Major architecture changes

---

## 🎓 Learning Points

### Good Things Found ✅
1. ✅ Đã có separation giữa routers/services/utils
2. ✅ Có async/await properly
3. ✅ Có error handling cơ bản
4. ✅ Có logging khá detailed

### Areas for Improvement ❌
1. ❌ Inconsistent singleton patterns
2. ❌ Significant code duplication
3. ❌ Lack of abstraction (too many Dict[str, Any])
4. ❌ Magic numbers scattered
5. ❌ No unit tests
6. ❌ Large service files

---

**Generated:** 2025-11-11
**Version:** 1.0
**Next Review:** After Phase 1 completion
