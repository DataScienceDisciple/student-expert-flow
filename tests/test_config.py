import pytest
from student_expert_flow.config import load_config, ExpertConfig, StudentConfig

# Define paths to sample config files
EXPERT_CONFIG_PATH = "configs/expert_config.yaml"
STUDENT_CONFIG_PATH = "configs/student_config.yaml"


def test_load_configs():
    """Tests if agent configurations load correctly from YAML files."""
    print("\nTesting config loading...")  # Keep print for visibility during run
    try:
        # Test loading expert config
        expert_config = load_config(EXPERT_CONFIG_PATH, 'expert')
        print("Expert config loaded successfully:")
        print(expert_config.model_dump_json(indent=2))
        assert isinstance(
            expert_config, ExpertConfig), "Loaded expert config should be ExpertConfig instance"
        assert expert_config.name == "DomainExpert", "Expert name mismatch"

        # Test loading student config
        student_config = load_config(STUDENT_CONFIG_PATH, 'student')
        print("\nStudent config loaded successfully:")
        print(student_config.model_dump_json(indent=2))
        assert isinstance(
            student_config, StudentConfig), "Loaded student config should be StudentConfig instance"
        assert student_config.name == "CuriousLearner", "Student name mismatch"

        print("\nConfig loading test passed.")
        # No explicit return needed; pytest passes if no assertion fails

    except FileNotFoundError as e:
        pytest.fail(f"Configuration file not found: {e}")
    except Exception as e:
        pytest.fail(
            f"Config loading test failed with an unexpected error: {e}")

# Removed the if __name__ == "__main__" block as pytest handles execution
