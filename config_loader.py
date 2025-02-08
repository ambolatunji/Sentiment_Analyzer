import yaml
from pathlib import Path
from typing import Dict, Any
import logging
from dataclasses import dataclass
import os

@dataclass
class ModelConfig:
    name: str
    pretrained: bool
    input_size: tuple
    normalize: Dict[str, list]

@dataclass
class FaceDetectionConfig:
    scale_factor: float
    min_neighbors: int
    min_size: tuple
    confidence_threshold: float
    detection_method: str
    draw_color: tuple
    rectangle_thickness: int

@dataclass
class EmotionAnalysisConfig:
    enforce_detection: bool
    actions: list
    confidence_threshold: float
    enabled_emotions: list

@dataclass
class CacheConfig:
    enabled: bool
    directory: str
    max_size_mb: int
    expiration_days: int
    imagenet_cache_file: str

@dataclass
class APIConfig:
    imagenet_url: str
    timeout_seconds: int
    max_retries: int
    retry_delay_seconds: int

@dataclass
class LoggingConfig:
    level: str
    format: str
    file: str
    max_file_size_mb: int
    backup_count: int

@dataclass
class UIConfig:
    page_title: str
    page_icon: str
    layout: str
    max_upload_size_mb: int
    supported_formats: list
    display: Dict[str, bool]

class ConfigLoader:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            # Open file with UTF-8 encoding and error handling
            with open(self.config_path, 'r', encoding='utf-8', errors='replace') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing YAML configuration: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading configuration: {str(e)}")
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        model_config = self.config['model']
        return ModelConfig(
            name=model_config['name'],
            pretrained=model_config['pretrained'],
            input_size=tuple(model_config['input_size']),
            normalize=model_config['normalize']
        )
    
    def get_face_detection_config(self) -> FaceDetectionConfig:
        """Get face detection configuration."""
        face_config = self.config['face_detection']
        return FaceDetectionConfig(
            scale_factor=face_config['scale_factor'],
            min_neighbors=face_config['min_neighbors'],
            min_size=tuple(face_config['min_size']),
            confidence_threshold=face_config['confidence_threshold'],
            detection_method=face_config['detection_method'],
            draw_color=tuple(face_config['draw_color']),
            rectangle_thickness=face_config['rectangle_thickness']
        )
    
    def get_emotion_analysis_config(self) -> EmotionAnalysisConfig:
        """Get emotion analysis configuration."""
        emotion_config = self.config['emotion_analysis']
        return EmotionAnalysisConfig(
            enforce_detection=emotion_config['enforce_detection'],
            actions=emotion_config['actions'],
            confidence_threshold=emotion_config['confidence_threshold'],
            enabled_emotions=emotion_config['enabled_emotions']
        )
    
    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration."""
        cache_config = self.config['cache']
        return CacheConfig(
            enabled=cache_config['enabled'],
            directory=cache_config['directory'],
            max_size_mb=cache_config['max_size_mb'],
            expiration_days=cache_config['expiration_days'],
            imagenet_cache_file=cache_config['imagenet_cache_file']
        )
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration."""
        api_config = self.config['api']
        return APIConfig(
            imagenet_url=api_config['imagenet_url'],
            timeout_seconds=api_config['timeout_seconds'],
            max_retries=api_config['max_retries'],
            retry_delay_seconds=api_config['retry_delay_seconds']
        )
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        log_config = self.config['logging']
        return LoggingConfig(
            level=log_config['level'],
            format=log_config['format'],
            file=log_config['file'],
            max_file_size_mb=log_config['max_file_size_mb'],
            backup_count=log_config['backup_count']
        )
    
    def get_ui_config(self) -> UIConfig:
        """Get UI configuration."""
        ui_config = self.config['ui']
        return UIConfig(
            page_title=ui_config['page_title'],
            page_icon=ui_config['page_icon'],
            layout=ui_config['layout'],
            max_upload_size_mb=ui_config['max_upload_size_mb'],
            supported_formats=ui_config['supported_formats'],
            display=ui_config['display']
        )

def setup_logging(config: LoggingConfig):
    """Setup logging based on configuration."""
    from logging.handlers import RotatingFileHandler
    
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config.level))
    
    # Create handlers
    file_handler = RotatingFileHandler(
        config.file,
        maxBytes=config.max_file_size_mb * 1024 * 1024,
        backupCount=config.backup_count,
        encoding='utf-8'  # Added explicit UTF-8 encoding
    )
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    formatter = logging.Formatter(config.format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)