# Event About Generation System - Changelog

## [2.0.0] - 2025-11-12

### 🚀 Major Upgrade: From OCR Extraction to Intelligent Event Analysis

This version represents a complete overhaul of the Event About Generation system, transforming it from a simple OCR text extraction tool into an intelligent event analysis and classification system.

---

### ✨ New Features

#### 1. **Comprehensive Event Analysis Framework**
- **Multi-image analysis**: Process 1-10+ event screenshots simultaneously
- **Automatic classification** across 5 major dimensions:
  - Release Type (4 types)
  - Mechanic Families (25+ types)
  - Player Dynamics (6 types)
  - Reward Systems (10 types)
  - Reward Types (7 combinations)
- **Additional tag detection**: Battle Pass, IP Collaboration, Charity, etc.
- **Re-release tracking**: Identify Reskins, Updates, Redesigns, Renames, Reprices, Restructures

#### 2. **Bilingual Output (English/Vietnamese)**
- **Structured bilingual about sections** with:
  - Event name in both languages
  - Complete classification tags
  - Comprehensive 3-5 paragraph descriptions
  - Clear separation with `================` delimiter
- **Natural translations** (not literal word-for-word)
- **Professional game industry terminology** in both languages

#### 3. **Enhanced Prompt System**
- **`event_analysis` prompt**: Deep analysis framework with detailed classification guidelines
- **`synthesis_bilingual` prompt**: Intelligent synthesis of multi-image analysis into coherent bilingual narrative
- **Taxonomy-aware**: Incorporates knowledge from classification guides and re-release tracking requirements

#### 4. **Flexible Output Formats**
- **`bilingual_text`** (default): Plain text for web textarea
- **`json_bilingual`**: Structured JSON with metadata
- **`compact_text`**: Compact format without field labels
- **`markdown_bilingual`**: Rich markdown formatting
- **Field definitions reference**: Complete taxonomy for all classification dimensions

---

### 🔄 Changed

#### Configuration Files
- **Renamed**: `prompts.yaml` → `event_about_prompts.yaml`
  - More descriptive name reflecting purpose
  - Complete restructure with new prompt categories

- **Renamed**: `output_formats.yaml` → `event_about_template.yaml`
  - Better reflects template-based approach
  - New structure with `formats` section
  - Added `field_definitions` and `additional_tags` sections

#### System Behavior
- **From**: Simple OCR text extraction + synthesis
- **To**: Intelligent multi-image event analysis + classification + bilingual generation

#### Output Structure
- **From**: Plain OCR text
- **To**: Structured bilingual about with classification tags

---

### 📝 Updated Files

#### Core Configuration
1. **`config/event_about_prompts.yaml`** (previously `prompts.yaml`)
   - Added `event_analysis` prompt with full taxonomy framework
   - Added `synthesis_bilingual` prompt for bilingual output
   - Retained `ocr_simple` for backward compatibility

2. **`config/event_about_template.yaml`** (previously `output_formats.yaml`)
   - Added `bilingual_text` format (new default)
   - Added `json_bilingual`, `compact_text`, `markdown_bilingual` formats
   - Added `field_definitions` with valid values for all classification fields
   - Added `additional_tags` reference sections
   - Set `default_format: "bilingual_text"`

#### Python Code Updates
3. **`utils/prompt_manager.py`**
   - Updated file references: `prompts.yaml` → `event_about_prompts.yaml`
   - Updated file references: `output_formats.yaml` → `event_about_template.yaml`
   - Updated validation to check for new file names

4. **`utils/output_formatter.py`**
   - Updated default config path: `config/event_about_template.yaml`
   - Updated `_load_formats()` to extract from `formats` section of new structure

#### Documentation
5. **`config/README.md`** (new)
   - Comprehensive guide to configuration system
   - Event classification framework reference
   - Usage examples in code
   - Re-release & reskin tracking guide
   - Customization instructions
   - Troubleshooting section

6. **`CHANGELOG_EVENT_ABOUT.md`** (this file)
   - Complete version history
   - Migration guide from v1.x

---

### 🗑️ Deprecated

- **Old format**: Simple OCR → synthesis flow still works but is not the recommended approach
- **Single-language output**: Still supported via custom templates but not default
- **Minimal event info**: Old approach didn't capture classification tags

---

### ⚠️ Breaking Changes

#### 1. Configuration File Names
```python
# Old
prompt_manager = PromptManager()  # loads config/prompts.yaml
formatter = OutputFormatter("config/output_formats.yaml")

# New (auto-handled by updated code)
prompt_manager = PromptManager()  # loads config/event_about_prompts.yaml
formatter = OutputFormatter("config/event_about_template.yaml")
```

#### 2. Prompt References
```python
# Old
ocr_prompts = pm.prompts.get("ocr", {})
synthesis_prompts = pm.prompts.get("synthesis", {})

# New
event_analysis_prompts = pm.prompts.get("event_analysis", {})
bilingual_synthesis_prompts = pm.prompts.get("synthesis_bilingual", {})
```

#### 3. Output Format Structure
```yaml
# Old (output_formats.yaml)
formats:
  default:
    type: "raw_text"
    template: "{content}"

# New (event_about_template.yaml)
formats:
  bilingual_text:
    type: "plain_text"
    template: |
      [ENGLISH]
      Event Name: {event_name_en}
      ...
      ================
      [TIẾNG VIỆT]
      Tên sự kiện: {event_name_vi}
      ...
```

#### 4. Default Output
- **Old**: Simple extracted text
- **New**: Structured bilingual about with classification tags

---

### 📦 Migration Guide

#### For Existing Projects

**Step 1: Rename configuration files**
```bash
cd config/
mv prompts.yaml event_about_prompts.yaml
mv output_formats.yaml event_about_template.yaml
```

**Step 2: Update configuration content**
- Copy content from new `event_about_prompts.yaml` template
- Copy content from new `event_about_template.yaml` template
- Merge any custom prompts you had in old files

**Step 3: Update code (already done in codebase)**
- `prompt_manager.py` - file references updated
- `output_formatter.py` - file references updated

**Step 4: Test with sample events**
```python
from services.about_extraction_service import extract_about_from_folder

result = await extract_about_from_folder(
    folder_path="/path/to/event/images",
    event_name="Test Event",
    event_type="Event",
    game_code="test_game",
    output_format="bilingual_text"  # new default format
)

print(result["about_content"])
```

**Step 5: Verify output**
- Check for `[ENGLISH]` and `[TIẾNG VIỆT]` sections
- Verify classification tags are present
- Ensure separator `================` is included

---

### 🎯 Design Decisions

#### Why Bilingual?
- **User base**: Mix of English and Vietnamese speakers
- **Efficiency**: Generate both at once instead of separate translation step
- **Consistency**: AI ensures equivalent information in both languages

#### Why Comprehensive Classification?
- **Re-release tracking**: Need to identify reskins and updates
- **Database organization**: Proper tagging enables filtering and search
- **Analytics**: Classification enables trend analysis across events

#### Why Multiple Images?
- **Completeness**: Single "about" screenshot often incomplete
- **Accuracy**: Cross-reference multiple sources reduces OCR errors
- **Context**: Different images show mechanics, rewards, rules, etc.

#### Why Plain Text Default Format?
- **Web integration**: Easy to insert into textarea
- **Human-readable**: Simple to review and edit
- **Parser-friendly**: Structured enough to extract tags if needed

---

### 📊 Taxonomy Reference

#### Release Types
- **Feature** (permanent)
- **Season** (recurring limited-time)
- **Event** (one-time limited)
- **Content** (core game content)

#### Top Mechanic Families
- Missions, Collections, Competitions, Challenges
- Quests, Mini-Games, Rewards, Purchases
- Clubs, Leaderboards, Bonuses, Banks
- (and 13+ more - see config/README.md)

#### Player Dynamics
- Individualistic, Collaborative, Competitive
- (and 3 combination types)

#### Reward Systems
- Singles, Milestones, Prize only, Linear/Exponential
- (and 6 combination types)

#### Reward Types
- Currencies, Items, Real prizes
- (and 4 combination types)

---

### 🔮 Future Enhancements

#### Planned for v2.1
- [ ] Auto-suggest rerelease tags by comparing with previous runs
- [ ] Confidence scores for classification tags
- [ ] Support for more languages (Korean, Japanese, Chinese)
- [ ] Visual similarity detection for theme analysis

#### Under Consideration
- [ ] Integration with event database for automatic deduplication
- [ ] Machine learning model for improved classification accuracy
- [ ] Batch processing for multiple events
- [ ] Historical comparison reports

---

### 🐛 Bug Fixes

None (initial release of v2.0.0)

---

### 🙏 Acknowledgments

This system is built on the classification framework documented in:
- **Re-release & Re-skin GUIDE.pdf**: Guidelines for tracking event re-releases
- **Tagging of Release.pdf**: Complete taxonomy for event classification

Special thanks to the game operations team for providing comprehensive tagging requirements.

---

## [1.x] - Legacy System

### Features
- Basic OCR text extraction from images
- Simple synthesis of OCR results
- Single-language output
- Minimal event metadata

### Limitations
- No classification framework
- No bilingual support
- No re-release tracking
- Limited to about image text extraction
- Manual tagging required

---

**Version History**
- **v2.0.0** (2025-11-12): Complete system overhaul with bilingual classification
- **v1.x** (legacy): Basic OCR extraction system
