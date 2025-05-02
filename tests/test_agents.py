import pytest
from student_expert_flow.agents import ExpertAgent
from student_expert_flow.config import load_config, ExpertConfig
from agents import Agent  # Import the base Agent from SDK for type checking

# Define the path to the sample config file
EXPERT_CONFIG_PATH = "configs/expert_config.yaml"


def test_expert_agent_initialization():
    """Tests if the ExpertAgent initializes correctly using a sample config file."""
    try:
        # Load the configuration
        expert_config = load_config(EXPERT_CONFIG_PATH, 'expert')
        assert isinstance(
            expert_config, ExpertConfig), "Loaded config should be ExpertConfig instance"

        # Initialize the ExpertAgent
        expert_agent_instance = ExpertAgent(config=expert_config)

        # Assert that the instance was created
        assert expert_agent_instance is not None, "ExpertAgent instance should be created"

        # Assert that the internal SDK agent was created and is of the correct type
        assert hasattr(expert_agent_instance,
                       'agent'), "ExpertAgent should have an 'agent' attribute"
        assert isinstance(expert_agent_instance.agent,
                          Agent), "Internal agent should be an instance of agents.Agent"

        # Assert config values were stored/used (optional check)
        assert expert_agent_instance.config == expert_config, "Stored config should match input config"
        assert expert_agent_instance.agent.name == expert_config.name, "Agent name should match config"

    except FileNotFoundError:
        pytest.fail(
            f"Test configuration file not found at {EXPERT_CONFIG_PATH}")
    except Exception as e:
        pytest.fail(
            f"ExpertAgent initialization failed with an unexpected error: {e}")


def test_expert_agent_respond_placeholder():
    """Tests the placeholder respond method."""
    expert_config = load_config(EXPERT_CONFIG_PATH, 'expert')
    expert_agent_instance = ExpertAgent(config=expert_config)

    test_message = "Hello Agent"
    response = expert_agent_instance.respond(test_message)

    assert isinstance(response, str), "Response should be a string"
    assert expert_config.name in response, "Response should mention the agent's name"
    assert "Placeholder response" in response, "Response should indicate it's a placeholder"

# You can add more tests here for different configurations or edge cases if needed.
