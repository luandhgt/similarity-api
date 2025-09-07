# AI Service - Project Summary & File Documentation (UPDATED - SIMPLIFIED IMAGE SCORING)

## Project Structure
```
ai-service/
├── main.py                          # FastAPI application entry point
├── setup_model.py                   # Places365 model setup script (not provided)
├── requirements.txt                 # Python dependencies (not provided)
├── .env                             # Environment variables (not provided)
├── 
├── routers/                         # FastAPI routers
│   ├── similarity.py               # Similarity search endpoints
│   ├── about_extraction.py         # About extraction endpoints
│   └── event_similarity.py         # Event alternative analysis endpoints (NOT YET IMPLEMENTED)
├── 
├── services/                       # Business logic services
│   ├── about_extraction_service.py # Main about extraction service
│   ├── claude_service.py          # Claude API integration
│   ├── database_service.py         # PostgreSQL database operations
│   └── event_similarity_service.py # Event alternative analysis service (SIMPLIFIED)
├── 
├── models/                         # AI model utilities
│   ├── places365.py               # Places365 model loading
│   └── resnet50_places365.pth.tar # Downloaded model weights (auto-downloaded)
├── 
├── utils/                          # Utility modules
│   ├── image_processor.py         # Places365 image processing
│   ├── text_processor.py          # Voyage text processing
│   ├── faiss_manager.py           # FAISS index management
│   ├── prompt_manager.py          # YAML prompt management
│   └── output_formatter.py        # Output formatting
├── 
├── config/                         # Configuration files
│   ├── prompts.yaml               # OCR/synthesis prompts
│   ├── output_formats.yaml        # Output format templates
│   ├── similarity_prompts.yaml    # Event similarity analysis prompts
│   └── similarity_output_formats.yaml # Event similarity output formats
├── 
└── index/                          # FAISS indexes (auto-created)
    ├── about/                     # Text embeddings for about content
    ├── images/                    # Image embeddings
    └── name/                      # Text embeddings for names
```

## Overview
AI Service is a FastAPI-based application providing three main functionalities:
1. **Similarity Search**: Multi-modal embedding and similarity search for images and text
2. **About Content Extraction**: OCR and AI synthesis from image collections to generate event descriptions
3. **Event Alternative Analysis**: Simplified multi-modal alternative event detection with separate text and image scoring

## Architecture
The system uses a modular architecture with separate routers, services, and utilities. It supports multiple games with separate FAISS indexes for different content types (images, about text, names). The alternative analysis system integrates FAISS similarity search across all modalities with PostgreSQL data retrieval and Claude-based text analysis.

---

## Event Alternative Analysis System (SIMPLIFIED SCORING)

### Input Structure
```python
{
    "folder_name": "event_123_images",
    "game_code": "candy_crush", 
    "event_name": "Summer Festival",
    "about": "A summer themed event with...",
    "image_count": 5,  # For validation
    "shared_uploads_path": "/shared/uploads/"
}
```

### Response Structure (UNCHANGED)
```python
{
  "query_event": {
    "name": "Original Summer Festival",
    "about": "The original event description...",
    "tags": {
      "family": "Competitions", 
      "dynamics": "Collaborative",
      "rewards": "Currencies & items"
    },
    "tag_explanation": "Query event tagged as Competition because... Collaborative because..."
  },
  "similar_events": [
    {
      "name": "Summer Festival Clone",
      "about": "A summer themed event with rewards...",
      "score_text": 0.85,      # From Claude analysis (0-1)
      "score_image": 0.0,      # 0.0 for text-found events, actual score for image-found events
      "reason": "This event is a strong alternative because...",
      "tags": {
        "family": "Competitions", 
        "dynamics": "Collaborative", 
        "rewards": "Currencies & items"
      },
      "tag_explanation": "Tagged as Competition because... Tagged as Collaborative because...",
      "image_faiss_indices": []  # Empty for text-found events, populated for image-found events
    }
  ]
}
```

### Core Workflow (SIMPLIFIED)
1. **Image Loading & Validation**: Load images from `shared_uploads_path/folder_name/`, validate count matches `image_count`, convert to Places365 embeddings
2. **Multi-Modal FAISS Search**: 
   - Name search using Voyage embeddings
   - About search using Voyage embeddings  
   - Image search using Two-Phase approach (UNCHANGED for now)
3. **Union All Results**: Combine ALL FAISS indices from 3 searches (no filtering, no limits) → max 30 unique events
4. **Database Integration**: Map ALL FAISS indices to PostgreSQL records via `faiss_index` field
5. **Simple Image Score Assignment**: 
   - **Events from image search**: Keep actual FAISS similarity scores
   - **Events from text search**: Set `score_image = 0.0` (NO reverse lookup)
6. **Claude Text-Only Analysis**: Analyze ALL 30 candidates for taxonomy tagging and text similarity
7. **Complete Response**: Return ALL analyzed events with separate text/image scores

### Simplified Image Scoring Logic (NEW)
#### For Events Found by Text Search (Name/About):
- **No reverse calculation**: `score_image = 0.0` always
- **No database queries**: Skip image embedding lookups entirely
- **No computation**: Zero overhead for text-found events

#### For Events Found by Image Search:
- **Use FAISS scores directly**: Keep similarity scores from Two-Phase search
- **Preserve image indices**: Maintain `matching_image_indices` for transparency
- **Set text score later**: `score_text` comes from Claude analysis

### Key Implementation Changes (SIMPLIFIED)
- **Removed Complex Methods**: Eliminated reverse image score calculation for text-found events
- **Clean Separation**: Text-found events have only text scores, image-found events start with only image scores
- **Single Claude Pass**: All 30 events analyzed together for text similarity and taxonomy
- **No Score Bias**: Text-found and image-found events no longer artificially compete on image similarity
- **Simplified Workflow**: Reduced complexity while maintaining functionality

---

## File-by-File Documentation

### Core Application

#### `main.py` - FastAPI Application Entry Point
**Purpose**: Main FastAPI application with startup/shutdown lifecycle management
**Key Functions**:
- `lifespan()`: Async context manager for startup/shutdown
- `root()`: Root endpoint with service status
- `health_check()`: Health monitoring endpoint
- `global_exception_handler()`: Global error handling

**Dependencies**: Initializes Places365 model, Voyage client, Claude service, and Database service
**Routers**: Includes similarity, about_extraction, and event_similarity routers
**When to modify**: App configuration, CORS settings, global middleware, health checks

---

### API Routers

#### `routers/similarity.py` - Similarity Search API
**Purpose**: REST endpoints for embedding and similarity search functionality
**Key Classes**:
- `EmbedImageRequest/EmbedTextRequest`: Input validation models
- `SearchRequest`: Search query parameters
- `EmbedResponse/SearchResponse`: Response models

**Key Endpoints**:
- `POST /embed_image`: Extract image features using Places365
- `POST /embed_text`: Extract text features using Voyage
- `POST /search`: Similarity search in FAISS indexes
- `GET /stats`: FAISS index statistics
- `GET /games`: List available games
- `GET /model_info`: Model information

**When to modify**: Adding new embedding endpoints, changing search parameters, API validation

#### `routers/about_extraction.py` - About Content Extraction API
**Purpose**: REST endpoints for OCR and AI synthesis of event descriptions
**Key Classes**:
- `ExtractAboutRequest`: Input parameters for extraction
- `ExtractAboutResponse`: Extraction results
- `ServiceStatusResponse`: Service health status

**Key Endpoints**:
- `POST /api/extract-about`: Main extraction endpoint
- `GET /api/extract-about/status`: Service status
- `GET /api/extract-about/formats`: Available output formats
- `POST /api/extract-about/reload-config`: Reload configuration

**When to modify**: Adding new extraction parameters, output formats, API endpoints

#### `routers/event_similarity.py` - Event Alternative Analysis API
**Purpose**: REST endpoints for simplified event alternative detection
**Status**: NOT YET IMPLEMENTED
**Expected Endpoints** (planned):
- `POST /api/find-similar-events`: Main alternative analysis endpoint
- `GET /api/find-similar-events/status`: Service health status
- `GET /api/find-similar-events/taxonomy`: Available taxonomy values

**When to modify**: Adding new analysis parameters, response formats, taxonomy updates

---

### Business Logic Services

#### `services/about_extraction_service.py` - Main Extraction Service
**Purpose**: Core business logic for extracting and synthesizing about content
**Key Class**: `AboutExtractionService`
**Key Methods**:
- `extract_about_from_folder()`: Main extraction workflow
- `_find_images_in_folder()`: Image file discovery
- `_extract_texts_from_images()`: OCR processing
- `_synthesize_about_content()`: AI content synthesis
- `get_supported_formats()`: Available output formats
- `get_service_status()`: Service health check

**Dependencies**: Claude service, prompt manager, output formatter
**When to modify**: Extraction workflow, supported image formats, processing logic

#### `services/claude_service.py` - Claude API Integration
**Purpose**: Handle all Claude API interactions for OCR and text synthesis
**Key Class**: `ClaudeService`
**Key Methods**:
- `_make_request()`: Core Claude API requests
- `extract_text_from_image()`: Single image OCR
- `synthesize_content()`: Text synthesis from OCR results
- `process_multiple_images()`: Batch image processing
- `get_usage_stats()`: API usage statistics

**Configuration**: Model, timeout, max tokens settings
**When to modify**: Claude API parameters, error handling, request formats

#### `services/database_service.py` - PostgreSQL Database Operations
**Purpose**: Handle PostgreSQL database connections and queries for event similarity search
**Key Class**: `DatabaseService`
**Key Methods**:
- `initialize()`: PostgreSQL connection pool initialization
- `get_events_by_faiss_indices()`: Query events by FAISS index values
- `get_events_by_codes()`: Query events by event code UUIDs
- `get_text_embeddings_by_event_codes()`: Query text embeddings by event codes
- `get_combined_event_data()`: Join events and text_embeddings tables
- `health_check()`: Database connection health monitoring
- `execute_query()`: **REQUIRED** - Execute raw SQL queries

**Database Configuration**: Uses env variables (DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME)
**Database Schema**: 
- `events` table: id, code, game_code, name, search_key, author, publish_date, created_at
- `text_embeddings` table: id, code, event_code, content_type, text_content, faiss_index, vector_dimension
- `image_embeddings` table: **REQUIRED** - id, event_code, game_code, faiss_index, image_path, created_at
**When to modify**: Database schema changes, query optimization, connection parameters

#### `services/event_similarity_service.py` - Event Alternative Analysis Service (SIMPLIFIED)
**Purpose**: Core business logic for finding alternative events using simplified scoring approach
**Key Class**: `EventSimilarityService`

**Key Methods (UPDATED)**:
- `find_similar_events()`: Main workflow with simplified image score assignment
- `_load_images_from_folder()`: Load images and convert to Places365 embeddings
- `search_by_name()`: FAISS search in name index using Voyage embeddings
- `search_by_about()`: FAISS search in about index using Voyage embeddings
- `search_by_images()`: Two-phase image search (UNCHANGED - to be simplified later)
- `_group_image_matches_by_event()`: Group individual image search results by event
- `_select_top_image_events()`: Select top candidates from Phase 1 based on partial scores
- `_combine_all_results()`: Union ALL search results (name + about + images)
- `_assign_simple_image_scores()`: **NEW** - Simple image score assignment without reverse calculation
- `analyze_text_similarity_with_claude()`: Comprehensive taxonomy tagging and text similarity analysis
- `query_postgres_for_texts()`: Database query to retrieve event details by FAISS indices
- `get_service_status()`: Health check including database and FAISS status

**Removed Methods (SIMPLIFIED)**:
- `_map_image_scores_to_events()` - **REMOVED**: No more complex image score mapping
- `_calculate_image_score_for_event()` - **REMOVED**: No reverse calculation for text-found events
- `_calculate_full_image_score_for_event()` - **REMOVED**: No fair recalculation needed
- `_get_candidate_event_images()` - **REMOVED**: No image embedding retrieval for text events
- `_get_image_indices_for_event()` - **REMOVED**: No database image queries for text events

**New Simple Image Score Assignment**:
```python
def _assign_simple_image_scores(candidate_texts, image_results):
    # For image-found events: use actual FAISS scores
    # For text-found events: score_image = 0.0 always
    # No complex calculations, no database queries
```

**Simplified Workflow**:
```
1. Multi-modal search → 30 unique events maximum
2. Simple score assignment → image-found events keep scores, text-found events get 0.0
3. Claude text analysis → all 30 events get text scores and taxonomy
4. Response building → merge text scores with pre-assigned image scores
```

**Configuration** (UNCHANGED):
```python
self.similarity_config = {
    "top_k": 10  # Only config needed
}
```

**When to modify**: Alternative detection logic, taxonomy rules, analysis workflow, multi-modal search parameters (image search still needs simplification)

---

### AI Models & Processing

#### `models/places365.py` - Places365 Model Management
**Purpose**: Load and manage Places365 ResNet50 pretrained model
**Key Functions**:
- `load_places365_model()`: Load model from checkpoint
- `get_places365_model()`: Singleton model instance

**Model Path**: `models/resnet50_places365.pth.tar`
**When to modify**: Model loading logic, checkpoint handling, architecture changes

#### `utils/image_processor.py` - Image Processing Pipeline
**Purpose**: Image preprocessing and feature extraction for Places365
**Key Functions**:
- `preprocess_image()`: Grayscale, resize, pad to 224x224
- `create_image_transform()`: PyTorch transforms
- `extract_image_features()`: 2048-dim feature extraction
- `validate_image_file()`: Input validation

**Processing**: Grayscale conversion, aspect ratio preservation, center padding
**When to modify**: Image preprocessing, feature extraction, supported formats

#### `utils/text_processor.py` - Text Processing Pipeline
**Purpose**: Text preprocessing and embedding with Voyage-3-Large
**Key Class**: `VoyageClient`
**Key Functions**:
- `embed_text()`: Text to 1024-dim embeddings
- `preprocess_text()`: Text cleaning
- `extract_text_features()`: Main feature extraction
- `validate_text_input()`: Input validation

**API**: Voyage AI embeddings service
**When to modify**: Text preprocessing, embedding models, API integration

---

### Data Management

#### `utils/faiss_manager.py` - FAISS Index Management
**Purpose**: Multi-game, multi-content-type FAISS index management
**Index Structure**: `index/{content_type}/{game_code}_index.bin`
**Content Types**: `["about", "images", "name"]`
**Vector Dimensions**: `{"about": 1024, "images": 2048, "name": 1024}`

**Key Functions**:
- `add_vector_to_faiss()`: Add embeddings to indexes
- `search_similar_vectors()`: Similarity search
- `load_faiss_index()`: Load/create indexes
- `get_faiss_stats()`: Index statistics
- `normalize_game_code()`: Game name normalization
- `get_vector_by_index()`: **STILL REQUIRED** - For remaining image search functionality

**When to modify**: Index structure, content types, vector dimensions, search algorithms

---

### Configuration Management

#### `utils/prompt_manager.py` - YAML Prompt Management
**Purpose**: Load and manage AI prompts from YAML configuration
**Config Path**: `config/prompts.yaml` (expects config directory)
**Key Class**: `PromptManager`
**Key Methods**:
- `get_ocr_prompts()`: OCR system/user prompts
- `get_synthesis_prompts()`: Synthesis prompts with variable substitution
- `get_similarity_prompts()`: Event similarity analysis prompts
- `reload_prompts()`: Hot reload configuration

**Prompt Categories**: `["ocr", "synthesis", "cleanup", "translation", "similarity"]`
**When to modify**: Prompt templates, variable substitution, prompt categories, taxonomy rules

#### `utils/output_formatter.py` - Output Format Management
**Purpose**: Handle multiple output formats based on YAML configuration
**Config Path**: `config/output_formats.yaml` (expects config directory)
**Key Class**: `OutputFormatter`
**Key Methods**:
- `format_output()`: Main formatting function
- `_format_json()/_format_markdown()/_format_html()`: Format-specific handlers
- `get_available_formats()`: List supported formats

**Format Types**: `["raw_text", "json", "markdown", "html", "similarity_analysis"]`
**Available Formats**: `["default", "json_detailed", "json_simple", "markdown", "html", "custom_summary"]`
**When to modify**: Output formats, template variables, format types

#### `config/similarity_prompts.yaml` - Event Alternative Analysis Prompts
**Purpose**: Comprehensive prompt configuration for taxonomy tagging and alternative event detection
**Key Sections**:
- Taxonomy Reference: Complete definitions for 31 taxonomy values (17 Family + 6 Dynamics + 8 Rewards)
- Alternative Classification System: Detailed rules for 5 alternative types
- Analysis Methodology: Step-by-step validation process with evidence requirements
- Edge Cases: Critical taxonomy rules from mobile game guidelines

**Taxonomy Categories**:
- Family (17 values): Accelerators, Banks, Challenges, Clubs, Collections, Competitions, Custom Design, Expansions, Hazards, Interactions, Levels, Mini-Games, Missions, Notices, Other, Purchases, Quests, Rewards
- Dynamics (6 values): Collaborative, Individualistic, Collaborative & competitive, Individualistic & competitive, Collaborative & individualistic, Indiv. collab. & comp
- Rewards (8 values): Currencies, Currencies & items, Currencies & real prize, Currencies items & real prize, Items, Items & real prize, None, Real prize

**When to modify**: Taxonomy definitions updates, alternative rules changes, analysis methodology improvements

#### `config/similarity_output_formats.yaml` - Event Similarity Output Formats
**Purpose**: Template configuration for event similarity analysis response formatting
**Key Formats**: default, detailed_json, summary, simple
**When to modify**: Response format requirements, client integration needs, output customization

---

## Key Dependencies & Integrations

### External APIs
- **Claude API**: Text analysis, taxonomy tagging, and similarity assessment (requires CLAUDE_API_KEY)
- **Voyage AI**: Text embeddings for name and about similarity search (requires VOYAGE_API_KEY)

### AI Models
- **Places365 ResNet50**: Scene recognition for image embeddings
- **Voyage-3-Large**: Text embeddings for similarity search

### Data Storage
- **FAISS**: Vector similarity search indexes
- **PostgreSQL**: Event and embedding data storage (simplified queries)
- **Local Files**: Model weights, configuration files, shared image folders

---

## Configuration Requirements

### Environment Variables
- `CLAUDE_API_KEY`: Claude API authentication
- `VOYAGE_API_KEY`: Voyage AI API authentication
- `DB_HOST`: PostgreSQL host address
- `DB_PORT`: PostgreSQL port number
- `DB_USER`: PostgreSQL username
- `DB_PASS`: PostgreSQL password
- `DB_NAME`: PostgreSQL database name

### Missing Files (Referenced but not provided)
- `config/prompts.yaml` - AI prompts configuration
- `config/output_formats.yaml` - Output format templates
- `setup_model.py` - Places365 model download script
- `requirements.txt` - Python dependencies
- `.env` - Environment variables with API keys

### Database Schema Requirements
**PostgreSQL Tables**:
- `events`: Main event records with id, code (UUID), game_code, name, search_key, author, publish_date, created_at
- `text_embeddings`: Text embedding records with faiss_index field for FAISS mapping
- `image_embeddings`: **STILL REQUIRED** - id, event_code, game_code, faiss_index, image_path, created_at (for image search functionality)

---

## Common Modification Scenarios

### To modify API endpoints:
**Files needed**: Relevant router file (`similarity.py`, `about_extraction.py`, or `event_similarity.py`)

### To modify AI processing logic:
**Files needed**: Service file (`about_extraction_service.py`, `claude_service.py`, or `event_similarity_service.py`)

### To modify embedding/similarity search:
**Files needed**: `faiss_manager.py`, `image_processor.py`, `text_processor.py`, or `event_similarity_service.py`

### To modify prompts or output formats:
**Files needed**: Configuration YAML files and corresponding manager utilities

### To modify model loading or preprocessing:
**Files needed**: `places365.py`, `image_processor.py`, or `text_processor.py`

### To add new content types or games:
**Files needed**: `faiss_manager.py` (update CONTENT_TYPES, VECTOR_DIMENSIONS)

### To modify event alternative analysis:
**Files needed**: `event_similarity_service.py` (SIMPLIFIED), `database_service.py`, `similarity_prompts.yaml`

---

## Implementation Status

### Completed ✅
- **Core Services**: `event_similarity_service.py` (SIMPLIFIED - removed complex image scoring)
- **Configuration**: `similarity_prompts.yaml`, `similarity_output_formats.yaml`
- **Analysis Logic**: Simplified multi-modal search, comprehensive Claude text analysis, clean score separation
- **Database Integration**: Simplified PostgreSQL queries (no complex image lookups for text events)
- **Simple Image Score Assignment**: Clean separation between text-found (score=0.0) and image-found events

### Still Required 🚧
- **API Router**: `routers/event_similarity.py` (NOT YET IMPLEMENTED)
- **Database Method**: Add `execute_query()` method to existing `database_service.py`
- **FAISS Utility**: Add `get_vector_by_index()` method to existing `faiss_manager.py` (still needed for image search)
- **Database Table**: Create `image_embeddings` table in PostgreSQL (still needed for image search)
- **Main Application**: Update `main.py` for service initialization and router inclusion
- **Image Search Simplification**: Simplify Two-Phase image search to single-phase
- **Testing**: Unit tests, integration tests

---

## Recent Updates Summary

### Simplified Image Scoring Implementation Complete:

#### Removed Complex Logic:
1. **Method `_map_image_scores_to_events()`**: Completely removed - no more reverse image score calculation
2. **Method `_calculate_image_score_for_event()`**: Removed - no image scoring for text-found events
3. **Method `_calculate_full_image_score_for_event()`**: Removed - no fair recalculation needed
4. **Method `_get_candidate_event_images()`**: Removed - no image retrieval for text events
5. **Method `_get_image_indices_for_event()`**: Removed - no database image queries for text events

#### Added Simple Logic:
1. **Method `_assign_simple_image_scores()`**: New simple assignment without complex calculations
2. **Updated `find_similar_events()`**: Removed image loading and complex score mapping calls
3. **Simplified `_build_final_response()`**: Clean merge of pre-assigned scores without complex lookup logic

### Key Benefits of Simplified Implementation:
- **No Reverse Calculations**: Text-found events never trigger image database queries
- **Clean Separation**: Text scores from Claude, image scores from FAISS, no cross-contamination
- **Reduced Complexity**: Eliminated complex scoring algorithms and database joins
- **Better Performance**: Fewer database queries and computations
- **Clearer Logic**: Each event type has clear score assignment rules
- **Maintainable Code**: Simplified methods are easier to debug and modify

### Next Phase:
- **Image Search Simplification**: Convert Two-Phase image search to single-phase approach
- **API Implementation**: Create the missing router and integrate with main application
- **Testing & Validation**: Ensure simplified approach maintains accuracy while improving performance