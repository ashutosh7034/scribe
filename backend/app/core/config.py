"""
Application Configuration

Centralized configuration management using Pydantic settings.
Supports environment variables and default values.
"""

from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "postgresql://scribe:scribe@localhost:5432/scribe"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    # AWS Configuration
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    
    # Speech Processing
    TRANSCRIBE_LANGUAGE_CODE: str = "en-US"
    
    # Avatar Rendering
    AVATAR_FRAME_RATE: int = 30
    MAX_AVATAR_RESOLUTION: str = "1920x1080"
    
    # Performance
    MAX_CONCURRENT_SESSIONS: int = 1000
    LATENCY_TARGET_MS: int = 300
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()