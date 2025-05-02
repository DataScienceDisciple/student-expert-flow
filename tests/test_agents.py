import pytest
from student_expert_flow.agents import ExpertAgent, StudentAgent
from student_expert_flow.config import load_config, ExpertConfig, StudentConfig
from agents import Agent  # Import the base Agent from SDK for type checking

# Define paths to sample config files
EXPERT_CONFIG_PATH = "configs/expert_config.yaml"
STUDENT_CONFIG_PATH = "configs/student_config.yaml"  # Add student config path


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

# --- Student Agent Tests ---


def test_student_agent_initialization():
    """Tests if the StudentAgent initializes correctly using a sample config file."""
    try:
        # Load the configuration
        student_config = load_config(STUDENT_CONFIG_PATH, 'student')
        assert isinstance(
            student_config, StudentConfig), "Loaded config should be StudentConfig instance"
        assert student_config.goal, "Student config must have a goal"

        # Initialize the StudentAgent
        student_agent_instance = StudentAgent(config=student_config)

        # Assert that the instance was created
        assert student_agent_instance is not None, "StudentAgent instance should be created"

        # Assert that the internal SDK agent was created
        assert hasattr(student_agent_instance,
                       'agent'), "StudentAgent should have an 'agent' attribute"
        assert isinstance(student_agent_instance.agent,
                          Agent), "Internal agent should be an instance of agents.Agent"

        # Assert config values were stored/used
        assert student_agent_instance.config == student_config, "Stored config should match input config"
        assert student_agent_instance.agent.name == student_config.name, "Agent name should match config"
        # Check if goal is incorporated into instructions (basic check)
        assert student_config.goal in student_agent_instance.agent.instructions, "Goal should be in agent instructions"

    except FileNotFoundError:
        pytest.fail(
            f"Test configuration file not found at {STUDENT_CONFIG_PATH}")
    except Exception as e:
        pytest.fail(
            f"StudentAgent initialization failed with an unexpected error: {e}")


def test_student_agent_ask_placeholder():
    """Tests the placeholder ask method (initial question)."""
    student_config = load_config(STUDENT_CONFIG_PATH, 'student')
    student_agent_instance = StudentAgent(config=student_config)

    initial_question = student_agent_instance.ask()  # Call without previous dialogue

    assert isinstance(initial_question, str)
    assert student_config.goal in initial_question, "Initial question should mention the goal"
    assert "explain the basics" in initial_question.lower(
    ), "Initial question should ask for basics"


def test_student_agent_ask_followup_placeholder():
    """Tests the placeholder ask method (follow-up question)."""
    student_config = load_config(STUDENT_CONFIG_PATH, 'student')
    student_agent_instance = StudentAgent(config=student_config)

    followup_question = student_agent_instance.ask(
        previous_dialogue="Some explanation from expert.")

    assert isinstance(followup_question, str)
    assert student_config.goal in followup_question, "Follow-up question should mention the goal"
    assert "tell me more" in followup_question.lower(
    ), "Follow-up question should ask for more info"


def test_student_agent_is_goal_achieved_placeholder():
    """Tests the placeholder is_goal_achieved method."""
    student_config = load_config(STUDENT_CONFIG_PATH, 'student')
    student_agent_instance = StudentAgent(config=student_config)

    response_met = "This is the final explanation. Goal Achieved."
    response_not_met = "Here is some more info."

    assert student_agent_instance.is_goal_achieved(
        response_met) is True, "Should detect goal achieved phrase"
    assert student_agent_instance.is_goal_achieved(
        response_not_met) is False, "Should not detect goal achieved phrase"

# You can add more tests here for different configurations or edge cases if needed.
