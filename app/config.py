import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class AppSettings(BaseSettings):
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Model Settings
    # Use pretrained checkpoint by default, or path to local model
    MODEL_CHECKPOINT: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    MODEL_PATH: str = "./models/best_model"
    
    # Inference Settings
    DEFAULT_THRESHOLD: float = 0.5
    MIN_REGION_AREA: int = 50
    SIMPLIFY_TOLERANCE: float = 0.1
    
    # Hugging Face
    HF_TOKEN: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

settings = AppSettings()
