import pytest
import asyncio
from unittest.mock import patch  # Using unittest.mock temporarily if needed

from student_expert_flow.agents import StudentAgent, ExpertAgent
from student_expert_flow.config import load_config
from student_expert_flow.runner import run_dialogue

# Config paths
EXPERT_CONFIG_PATH = "configs/expert_config.yaml"
STUDENT_CONFIG_PATH = "configs/student_config.yaml"


@pytest.mark.asyncio  # Mark test as async
async def test_run_dialogue_simulated_flow():
    """Tests the simulated dialogue flow and history generation."""

    # Load configs
    expert_config = load_config(EXPERT_CONFIG_PATH, 'expert')
    student_config = load_config(STUDENT_CONFIG_PATH, 'student')
    # Set a specific max_turns for predictability in this test
    test_max_turns = 3
    student_config.max_iterations = test_max_turns  # Or pass directly to run_dialogue

    # Initialize agents
    expert = ExpertAgent(expert_config)
    student = StudentAgent(student_config)

    # Run the simulated dialogue
    # Since the actual agent calls are placeholders, we don't need heavy mocking yet
    history = await run_dialogue(student, expert, max_turns=test_max_turns)

    # --- Assertions ---
    assert isinstance(history, list), "History should be a list"
    # Expecting student ask -> expert respond per turn, for max_turns
    assert len(history) == test_max_turns * \
        2, f"History should have {test_max_turns * 2} entries for {test_max_turns} turns"

    # Check roles and agent names in history
    expected_roles = ['user', 'assistant'] * test_max_turns
    expected_agents = [student.config.name,
                       expert.config.name] * test_max_turns

    for i, entry in enumerate(history):
        assert entry['role'] == expected_roles[i], f"Entry {i} role mismatch"
        assert entry['agent'] == expected_agents[i], f"Entry {i} agent mismatch"
        assert isinstance(entry['content'],
                          str), f"Entry {i} content should be string"

    # Check content of first and last messages (based on placeholder logic)
    assert student.config.goal in history[0]['content'], "First student message should contain goal"
    assert expert.config.name in history[-1]['content'], "Last message should be from expert"
    assert "Placeholder response" in history[-1]['content'], "Last expert message should be placeholder"


@pytest.mark.asyncio
async def test_run_dialogue_simulated_goal_achieved():
    """Tests the simulated dialogue flow ending with goal achievement."""
    expert_config = load_config(EXPERT_CONFIG_PATH, 'expert')
    student_config = load_config(STUDENT_CONFIG_PATH, 'student')
    # Ensure goal is achievable by placeholder logic - use the specific test string
    student_config.goal = "achieve goal now test string"

    expert = ExpertAgent(expert_config)
    student = StudentAgent(student_config)

    # Use a high max_turns, expect early exit
    history = await run_dialogue(student, expert, max_turns=10)

    # Based on updated placeholder logic in StudentAgent.ask:
    # Turn 1: Student asks about goal, Expert gives placeholder response
    # Turn 2: Student goal matches the specific test string,
    #         placeholder logic returns "Goal Achieved."
    assert len(
        history) == 3, "History should have 3 entries (ask, respond, ask-achieved)"
    assert history[-1]['agent'] == student.config.name, "Last entry should be student achieving goal"
    assert "Goal Achieved." in history[-1]['content'], "Last student message should state goal achieved"
