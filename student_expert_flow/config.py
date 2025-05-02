import yaml
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Literal


class ExpertConfig(BaseModel):
    name: str = "Expert"
    instructions: str
    model: str = "gpt-4.1-mini"
    max_tokens: int = 150
    tools: Optional[List[str]] = None  # Placeholder for now


class StudentConfig(BaseModel):
    name: str = "Student"
    instructions: str
    goal: str
    model: str = "gpt-4.1-mini"
    max_iterations: int = 10
    critique_style: Literal['constructive',
                            'concise', 'detailed'] = 'constructive'


def load_config(config_path: str, config_type: Literal['expert', 'student']) -> BaseModel:
    """Loads and validates agent configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {config_path}: {e}")

    if not raw_config:
        raise ValueError(f"Configuration file is empty: {config_path}")

    try:
        if config_type == 'expert':
            return ExpertConfig(**raw_config)
        elif config_type == 'student':
            return StudentConfig(**raw_config)
        else:
            # This case should not happen with Literal typing but added for safety
            raise ValueError("Invalid config_type specified.")
    except ValidationError as e:
        raise ValueError(
            f"Configuration validation error in {config_path}:\n{e}")
