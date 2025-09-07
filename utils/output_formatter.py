"""
Output Formatter - Handle different output formats based on configuration
"""
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class OutputFormatter:
    def __init__(self, config_path: str = "config/output_formats.yaml"):
        self.config_path = config_path
        self._formats_cache = None
        self._load_formats()
    
    def _load_formats(self) -> None:
        """Load output formats from YAML file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Output formats config file not found: {self.config_path}")
            
            with open(config_file, 'r', encoding='utf-8') as file:
                self._formats_cache = yaml.safe_load(file)
                logger.info(f"✅ Loaded output formats from {self.config_path}")
                
        except Exception as e:
            logger.error(f"❌ Error loading output formats config: {e}")
            # Fallback to basic formats
            self._formats_cache = self._get_fallback_formats()
    
    def _get_fallback_formats(self) -> Dict[str, Any]:
        """Fallback formats if config file is not available"""
        return {
            "default": {"type": "raw_text"},
            "json_simple": {
                "type": "json",
                "schema": {
                    "about_content": "string",
                    "status": "string"
                }
            }
        }
    
    def format_output(self, 
                     content: str,
                     format_name: str = "default",
                     metadata: Optional[Dict[str, Any]] = None) -> Union[str, Dict[str, Any]]:
        """
        Format output based on specified format
        
        Args:
            content: The main about content
            format_name: Name of format to use (from config)
            metadata: Additional metadata to include
            
        Returns:
            Formatted output (string or dict)
        """
        if not metadata:
            metadata = {}
        
        # Default metadata
        default_meta = {
            "timestamp": datetime.now().isoformat(),
            "word_count": len(content.split()) if content else 0,
            "processing_time": metadata.get("processing_time", 0),
            "image_count": metadata.get("image_count", 0),
            "event_name": metadata.get("event_name", ""),
            "event_type": metadata.get("event_type", ""),
            "game_code": metadata.get("game_code", "")
        }
        
        # Merge metadata
        metadata.update(default_meta)
        
        try:
            format_config = self._formats_cache.get(format_name, self._formats_cache["default"])
            format_type = format_config.get("type", "raw_text")
            
            if format_type == "raw_text":
                return content
            
            elif format_type == "json":
                return self._format_json(content, format_config, metadata)
            
            elif format_type == "markdown":
                return self._format_markdown(content, format_config, metadata)
            
            elif format_type == "html":
                return self._format_html(content, format_config, metadata)
            
            else:
                logger.warning(f"⚠️ Unknown format type: {format_type}, falling back to raw text")
                return content
                
        except Exception as e:
            logger.error(f"❌ Error formatting output: {e}")
            return content  # Fallback to raw content
    
    def _format_json(self, content: str, format_config: Dict, metadata: Dict) -> Dict[str, Any]:
        """Format as JSON"""
        schema = format_config.get("schema", {})
        
        if "json_detailed" in str(format_config):
            return {
                "about_content": content,
                "metadata": {
                    "word_count": metadata["word_count"],
                    "image_count": metadata["image_count"],
                    "processing_time": metadata["processing_time"],
                    "ocr_confidence": metadata.get("ocr_confidence", "unknown")
                },
                "sections": metadata.get("sections", []),
                "status": "success",
                "generated_at": metadata["timestamp"]
            }
        
        elif "json_simple" in str(format_config):
            return {
                "about_content": content,
                "word_count": metadata["word_count"],
                "status": "success"
            }
        
        elif "custom_summary" in str(format_config):
            return {
                "title": metadata.get("event_name", "Event"),
                "summary": content[:200] + "..." if len(content) > 200 else content,
                "full_content": content,
                "key_points": metadata.get("key_points", []),
                "generated_at": metadata["timestamp"]
            }
        
        else:
            # Generic JSON format
            return {
                "about_content": content,
                "word_count": metadata["word_count"],
                "status": "success"
            }
    
    def _format_markdown(self, content: str, format_config: Dict, metadata: Dict) -> str:
        """Format as Markdown"""
        template = format_config.get("template", "# About\n\n{about_content}")
        
        try:
            return template.format(
                about_content=content,
                event_name=metadata.get("event_name", "Event"),
                event_type=metadata.get("event_type", ""),
                game_code=metadata.get("game_code", ""),
                image_count=metadata["image_count"],
                timestamp=metadata["timestamp"]
            )
        except KeyError as e:
            logger.warning(f"⚠️ Missing variable in markdown template: {e}")
            return f"# About\n\n{content}"
    
    def _format_html(self, content: str, format_config: Dict, metadata: Dict) -> str:
        """Format as HTML"""
        template = format_config.get("template", "<div>{about_content}</div>")
        
        # Convert line breaks to HTML
        html_content = content.replace("\n", "<br>\n")
        
        try:
            return template.format(
                about_content=html_content,
                event_name=metadata.get("event_name", "Event"),
                event_type=metadata.get("event_type", ""),
                game_code=metadata.get("game_code", ""),
                image_count=metadata["image_count"],
                timestamp=metadata["timestamp"]
            )
        except KeyError as e:
            logger.warning(f"⚠️ Missing variable in HTML template: {e}")
            return f"<div>{html_content}</div>"
    
    def get_available_formats(self) -> list:
        """Get list of available output formats"""
        return list(self._formats_cache.keys()) if self._formats_cache else []
    
    def get_format_info(self, format_name: str) -> Dict[str, Any]:
        """Get information about a specific format"""
        try:
            return self._formats_cache[format_name]
        except KeyError:
            logger.warning(f"⚠️ Format not found: {format_name}")
            return {}
    
    def reload_formats(self) -> bool:
        """Reload formats from config file"""
        try:
            self._load_formats()
            return True
        except Exception as e:
            logger.error(f"❌ Failed to reload formats: {e}")
            return False

# Global instance
output_formatter = OutputFormatter()

# Convenience functions
def format_output(content: str, 
                 format_name: str = "default", 
                 metadata: Optional[Dict[str, Any]] = None) -> Union[str, Dict[str, Any]]:
    """Convenience function to format output"""
    return output_formatter.format_output(content, format_name, metadata)