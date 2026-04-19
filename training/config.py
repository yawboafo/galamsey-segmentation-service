import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class TrainingSettings(BaseSettings):
    # Dataset paths
    DATA_DIR: str = "./data"
    TRAIN_DIR: str = "./data/processed/train"
    VAL_DIR: str = "./data/processed/val"
    TEST_DIR: str = "./data/processed/test"
    
    # Model configuration
    MODEL_CHECKPOINT: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    OUTPUT_DIR: str = "./models/checkpoints"
    BEST_MODEL_DIR: str = "./models/best_model"
    
    # Hyperparameters
    TRAIN_BATCH_SIZE: int = 8
    VAL_BATCH_SIZE: int = 8
    LEARNING_RATE: float = 6e-5
    NUM_EPOCHS: int = 20
    IMAGE_SIZE: int = 512
    
    # HF Hub
    HF_TOKEN: Optional[str] = None
    HF_MODEL_REPO: Optional[str] = None
    PUSH_TO_HUB: bool = False
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

train_settings = TrainingSettings()
