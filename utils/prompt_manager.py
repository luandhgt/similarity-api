"""
Prompt Manager for AI Service
Manages YAML-based prompt configurations for OCR, synthesis, and similarity analysis
"""

import logging
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages AI prompts from YAML configuration files"""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize PromptManager
        
        Args:
            config_dir: Directory containing YAML configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Prompt storage
        self.prompts = {}
        self.output_formats = {}
        self.similarity_prompts = {}
        self.similarity_output_formats = {}
        
        # Load all configurations on initialization
        self._load_all_prompts()
    
    def _load_all_prompts(self):
        """Load all prompt configurations from YAML files"""
        try:
            self._load_prompts()
            self._load_output_formats()
            self._load_similarity_prompts()
            self._load_similarity_output_formats()
            logger.info("All prompt configurations loaded successfully")
        except Exception as e:
            logger.error(f"Error loading prompt configurations: {e}")
    
    def _load_prompts(self):
        """Load main prompts from event_about_prompts.yaml"""
        try:
            prompts_file = self.config_dir / "event_about_prompts.yaml"
            if prompts_file.exists():
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    self.prompts = yaml.safe_load(f) or {}
                logger.info("Main prompts loaded successfully")
            else:
                logger.warning(f"Prompts file not found: {prompts_file}")
                self._create_default_prompts()
        except Exception as e:
            logger.error(f"Error loading main prompts: {e}")
            self._create_default_prompts()
    
    def _load_output_formats(self):
        """Load output formats from event_about_template.yaml"""
        try:
            formats_file = self.config_dir / "event_about_template.yaml"
            if formats_file.exists():
                with open(formats_file, 'r', encoding='utf-8') as f:
                    self.output_formats = yaml.safe_load(f) or {}
                logger.info("Output formats loaded successfully")
            else:
                logger.warning(f"Output formats file not found: {formats_file}")
                self._create_default_output_formats()
        except Exception as e:
            logger.error(f"Error loading output formats: {e}")
            self._create_default_output_formats()
    
    def _load_similarity_prompts(self):
        """Load similarity prompts from similarity_prompts.yaml"""
        try:
            similarity_file = self.config_dir / "similarity_prompts.yaml"
            if similarity_file.exists():
                with open(similarity_file, 'r', encoding='utf-8') as f:
                    self.similarity_prompts = yaml.safe_load(f) or {}
                logger.info("Similarity prompts loaded successfully")
            else:
                logger.warning(f"Similarity prompts file not found: {similarity_file}")
                self._create_default_similarity_prompts()
        except Exception as e:
            logger.error(f"Error loading similarity prompts: {e}")
            self._create_default_similarity_prompts()
    
    def _load_similarity_output_formats(self):
        """Load similarity output formats from similarity_output_formats.yaml"""
        try:
            formats_file = self.config_dir / "similarity_output_formats.yaml"
            if formats_file.exists():
                with open(formats_file, 'r', encoding='utf-8') as f:
                    self.similarity_output_formats = yaml.safe_load(f) or {}
                logger.info("Similarity output formats loaded successfully")
            else:
                logger.warning(f"Similarity output formats file not found: {formats_file}")
                self._create_default_similarity_output_formats()
        except Exception as e:
            logger.error(f"Error loading similarity output formats: {e}")
            self._create_default_similarity_output_formats()
    
    # OCR Prompts
    def get_ocr_prompts(self) -> Dict[str, str]:
        """Get OCR prompts for image text extraction"""
        try:
            ocr_prompts = self.prompts.get("ocr", {})
            if not ocr_prompts:
                logger.warning("OCR prompts not found, using defaults")
                return self._get_default_ocr_prompts()
            return ocr_prompts
        except Exception as e:
            logger.error(f"Error getting OCR prompts: {e}")
            return self._get_default_ocr_prompts()
    
    # Synthesis Prompts
    def get_synthesis_prompts(self, **kwargs) -> Dict[str, str]:
        """
        Get synthesis prompts with variable substitution
        
        Args:
            **kwargs: Variables for prompt template substitution
            
        Returns:
            Dict containing system and user prompts
        """
        try:
            synthesis_prompts = self.prompts.get("synthesis", {})
            if not synthesis_prompts:
                logger.warning("Synthesis prompts not found, using defaults")
                synthesis_prompts = self._get_default_synthesis_prompts()
            
            # Perform variable substitution
            formatted_prompts = {}
            for key, template in synthesis_prompts.items():
                try:
                    formatted_prompts[key] = template.format(**kwargs)
                except KeyError as e:
                    logger.warning(f"Missing variable {e} in synthesis prompt template")
                    formatted_prompts[key] = template
                except Exception as e:
                    logger.warning(f"Error formatting synthesis prompt {key}: {e}")
                    formatted_prompts[key] = template
            
            return formatted_prompts
            
        except Exception as e:
            logger.error(f"Error getting synthesis prompts: {e}")
            return self._get_default_synthesis_prompts()
    
    # Similarity Prompts
    def get_similarity_prompts(self) -> Dict[str, str]:
        """Get similarity analysis prompts"""
        try:
            if not self.similarity_prompts:
                self._load_similarity_prompts()
            
            # Parse YAML structure correctly
            similarity_section = self.similarity_prompts.get("similarity", {})
            if not similarity_section:
                logger.warning("Similarity prompts not found, using defaults")
                return self._get_default_similarity_prompts()
            
            # Extract system and user prompts from YAML structure
            return {
                "system": similarity_section.get("system", ""),
                "user": similarity_section.get("user", "")
            }
        except Exception as e:
            logger.error(f"Error getting similarity prompts: {e}")
            return self._get_default_similarity_prompts()
    
    # Cleanup and Translation Prompts
    def get_cleanup_prompts(self) -> Dict[str, str]:
        """Get text cleanup prompts"""
        try:
            return self.prompts.get("cleanup", self._get_default_cleanup_prompts())
        except Exception as e:
            logger.error(f"Error getting cleanup prompts: {e}")
            return self._get_default_cleanup_prompts()
    
    def get_translation_prompts(self) -> Dict[str, str]:
        """Get translation prompts"""
        try:
            return self.prompts.get("translation", self._get_default_translation_prompts())
        except Exception as e:
            logger.error(f"Error getting translation prompts: {e}")
            return self._get_default_translation_prompts()
    
    # Configuration Management
    def reload_prompts(self):
        """Hot reload all prompt configurations"""
        try:
            self._load_all_prompts()
            logger.info("All prompts reloaded successfully")
        except Exception as e:
            logger.error(f"Error reloading prompts: {e}")
            raise
    
    def get_available_prompt_categories(self) -> List[str]:
        """Get list of available prompt categories"""
        categories = list(self.prompts.keys())
        if self.similarity_prompts:
            categories.extend(self.similarity_prompts.keys())
        return sorted(set(categories))
    
    def validate_prompt_configs(self) -> Dict[str, Any]:
        """Validate all prompt configurations"""
        validation = {
            "valid": True,
            "issues": [],
            "files_found": [],
            "files_missing": []
        }
        
        # Check main config files
        config_files = [
            "event_about_prompts.yaml",
            "event_about_template.yaml",
            "similarity_prompts.yaml",
            "similarity_output_formats.yaml"
        ]
        
        for filename in config_files:
            file_path = self.config_dir / filename
            if file_path.exists():
                validation["files_found"].append(filename)
            else:
                validation["files_missing"].append(filename)
                validation["issues"].append(f"Missing config file: {filename}")
        
        # Check prompt structure
        required_categories = ["ocr", "synthesis", "cleanup", "translation"]
        for category in required_categories:
            if category not in self.prompts:
                validation["issues"].append(f"Missing prompt category: {category}")
        
        # Check similarity prompts
        if not self.similarity_prompts.get("similarity"):
            validation["issues"].append("Missing similarity analysis prompts")
        
        if validation["issues"]:
            validation["valid"] = False
        
        return validation
    
    # DEFAULT PROMPTS - Fallback when YAML files are missing
    
    def _create_default_prompts(self):
        """Create default prompts when YAML file is missing"""
        self.prompts = {
            "ocr": self._get_default_ocr_prompts(),
            "synthesis": self._get_default_synthesis_prompts(),
            "cleanup": self._get_default_cleanup_prompts(),
            "translation": self._get_default_translation_prompts()
        }
    
    def _get_default_ocr_prompts(self) -> Dict[str, str]:
        """Default OCR prompts"""
        return {
            "system": "You are an expert OCR assistant. Extract text accurately from images.",
            "user": "Please extract all visible text from this image. Include UI elements, buttons, descriptions, and any readable content. Format the output clearly."
        }
    
    def _get_default_synthesis_prompts(self) -> Dict[str, str]:
        """Default synthesis prompts with variable support"""
        return {
            "system": "You are an expert content synthesizer for game events.",
            "user": "Based on the extracted text content, create a comprehensive event description. Content: {extracted_texts}"
        }
    
    def _get_default_cleanup_prompts(self) -> Dict[str, str]:
        """Default cleanup prompts"""
        return {
            "system": "You are a text cleanup specialist.",
            "user": "Clean and format this text for better readability: {raw_text}"
        }
    
    def _get_default_translation_prompts(self) -> Dict[str, str]:
        """Default translation prompts"""
        return {
            "system": "You are a professional translator.",
            "user": "Translate this text to {target_language}: {source_text}"
        }
    
    def _create_default_similarity_prompts(self):
        """Create default similarity prompts when YAML file is missing"""
        self.similarity_prompts = {
            "similarity": self._get_default_similarity_prompts()
        }
    
    def _get_default_similarity_prompts(self) -> Dict[str, str]:
        """Default similarity analysis prompts - simplified fallback"""
        return {
            "system": "You are an expert game event analyzer. Analyze taxonomy and alternative relationships between events.",
            "user": "Analyze the query event and candidates for alternative relationships. Query: {query_name} - {query_about}. Candidates: {candidates}"
        }
    
    def _create_default_output_formats(self):
        """Create default output formats when YAML file is missing"""
        self.output_formats = {
            "formats": {
                "default": {
                    "template": "{content}",
                    "description": "Raw content output"
                },
                "json_detailed": {
                    "template": '{"event_name": "{event_name}", "description": "{content}", "extracted_at": "{timestamp}"}',
                    "description": "Detailed JSON format"
                },
                "json_simple": {
                    "template": '{"description": "{content}"}',
                    "description": "Simple JSON format"
                },
                "markdown": {
                    "template": "# {event_name}\n\n{content}\n\n*Extracted at: {timestamp}*",
                    "description": "Markdown format"
                },
                "html": {
                    "template": "<div class='event-description'><h2>{event_name}</h2><p>{content}</p><small>Extracted: {timestamp}</small></div>",
                    "description": "HTML format"
                }
            }
        }
    
    def _create_default_similarity_output_formats(self):
        """Create default similarity output formats when YAML file is missing"""
        self.similarity_output_formats = {
            "formats": {
                "default": {
                    "template": """{
  "query_event": {
    "name": "{query_name}",
    "about": "{query_about}",
    "tags": {tags},
    "tag_explanation": "{tag_explanation}"
  },
  "similar_events": {similar_events}
}""",
                    "description": "Standard similarity analysis format"
                },
                "detailed_json": {
                    "template": """{
  "analysis_metadata": {
    "query_name": "{query_name}",
    "game_code": "{game_code}", 
    "total_candidates": {candidate_count},
    "analysis_timestamp": "{timestamp}"
  },
  "results": {
    "query_event": {query_event},
    "similar_events": {similar_events}
  }
}""",
                    "description": "Detailed JSON with metadata"
                },
                "summary": {
                    "template": "Found {event_count} similar events for '{query_name}'. Top alternatives: {top_alternatives}",
                    "description": "Brief summary format"
                }
            }
        }
    
    # OUTPUT FORMAT METHODS
    
    def get_available_formats(self) -> List[str]:
        """Get list of available output format names"""
        try:
            formats = self.output_formats.get("formats", {})
            return list(formats.keys())
        except Exception as e:
            logger.error(f"Error getting available formats: {e}")
            return ["default"]
    
    def get_format_template(self, format_name: str) -> str:
        """
        Get output format template
        
        Args:
            format_name: Name of the format
            
        Returns:
            Template string
        """
        try:
            formats = self.output_formats.get("formats", {})
            format_config = formats.get(format_name, {})
            
            if not format_config:
                logger.warning(f"Format '{format_name}' not found, using default")
                return "{content}"
            
            return format_config.get("template", "{content}")
            
        except Exception as e:
            logger.error(f"Error getting format template for '{format_name}': {e}")
            return "{content}"
    
    def get_format_description(self, format_name: str) -> str:
        """Get description for a specific format"""
        try:
            formats = self.output_formats.get("formats", {})
            format_config = formats.get(format_name, {})
            return format_config.get("description", "No description available")
        except Exception as e:
            logger.error(f"Error getting format description: {e}")
            return "Error retrieving description"
    
    # SIMILARITY OUTPUT FORMATS
    
    def get_similarity_format_template(self, format_name: str = "default") -> str:
        """Get similarity analysis output format template"""
        try:
            formats = self.similarity_output_formats.get("formats", {})
            format_config = formats.get(format_name, {})
            
            if not format_config:
                logger.warning(f"Similarity format '{format_name}' not found, using default")
                # Return basic JSON structure
                return """{
  "query_event": {query_event},
  "similar_events": {similar_events}
}"""
            
            return format_config.get("template", "{content}")
            
        except Exception as e:
            logger.error(f"Error getting similarity format template: {e}")
            return "{content}"
    
    def get_available_similarity_formats(self) -> List[str]:
        """Get list of available similarity output formats"""
        try:
            formats = self.similarity_output_formats.get("formats", {})
            return list(formats.keys())
        except Exception as e:
            logger.error(f"Error getting similarity formats: {e}")
            return ["default"]
    
    # UTILITY METHODS
    
    def update_prompt(self, category: str, prompt_type: str, content: str):
        """Update a specific prompt programmatically"""
        try:
            if category not in self.prompts:
                self.prompts[category] = {}
            
            self.prompts[category][prompt_type] = content
            logger.info(f"Updated prompt: {category}.{prompt_type}")
            
        except Exception as e:
            logger.error(f"Error updating prompt {category}.{prompt_type}: {e}")
    
    def export_config_template(self, file_type: str = "prompts") -> str:
        """Export template YAML for missing configuration files"""
        try:
            if file_type == "prompts":
                return yaml.dump(self.prompts, default_flow_style=False, allow_unicode=True)
            elif file_type == "output_formats":
                return yaml.dump(self.output_formats, default_flow_style=False, allow_unicode=True)
            elif file_type == "similarity_prompts":
                return yaml.dump(self.similarity_prompts, default_flow_style=False, allow_unicode=True)
            elif file_type == "similarity_output_formats":
                return yaml.dump(self.similarity_output_formats, default_flow_style=False, allow_unicode=True)
            else:
                return "# Unknown file type"
        except Exception as e:
            logger.error(f"Error exporting config template: {e}")
            return f"# Error: {e}"
    
    def get_config_status(self) -> Dict[str, Any]:
        """Get status of all configuration files and prompts"""
        return {
            "config_directory": str(self.config_dir),
            "validation": self.validate_prompt_configs(),
            "loaded_categories": self.get_available_prompt_categories(),
            "available_formats": self.get_available_formats(),
            "similarity_formats": self.get_available_similarity_formats(),
            "prompt_counts": {
                "main_prompts": len(self.prompts),
                "similarity_prompts": len(self.similarity_prompts),
                "output_formats": len(self.output_formats.get("formats", {})),
                "similarity_output_formats": len(self.similarity_output_formats.get("formats", {}))
            }
        }