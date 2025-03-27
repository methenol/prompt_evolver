"""
Configuration module for the prompt evolution system.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file. To force override, set override=True. Resolves value cache issues.
load_dotenv()

def get_env_float(key: str, default: float) -> float:
    """Helper function to get float values from environment variables."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default

def get_env_int(key: str, default: int) -> int:
    """Helper function to get integer values from environment variables."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default

@dataclass
class Config:
    # API configuration
    api_key: str = os.getenv('API_KEY', '')
    base_url: Optional[str] = os.getenv('BASE_URL') or None
    model_name: str = os.getenv('MODEL_NAME', 'gpt-4o-mini')
    default_temperature: float = get_env_float('DEFAULT_TEMPERATURE', 0.7)
    
    # System paths
    output_dir: str = os.getenv('OUTPUT_DIR', 'output')
    
    # Evolution parameters
    population_size: int = get_env_int('POPULATION_SIZE', 20)
    num_generations: int = get_env_int('NUM_GENERATIONS', 10)
    mutation_rate: float = get_env_float('MUTATION_RATE', 0.3)
    crossover_rate: float = get_env_float('CROSSOVER_RATE', 0.7)
    tournament_size: int = get_env_int('TOURNAMENT_SIZE', 3)
    elite_size: int = get_env_int('ELITE_SIZE', 2)
    generations: int = get_env_int('GENERATIONS', 20)
    
    # Output and saving
    save_frequency: int = get_env_int('SAVE_FREQUENCY', 2)

    # Evaluation configuration
    evaluation_type: str = os.getenv('EVALUATION_TYPE', 'combined').lower()


# Create default configuration instance
default_config = Config()