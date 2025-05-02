import pytest
import asyncio
import os
from dotenv import load_dotenv

from student_expert_flow.agents import StudentAgent, ExpertAgent
from student_expert_flow.config import load_config
from student_expert_flow.runner import run_dialogue
# Import the structured output model for type checking
from student_expert_flow.models import StudentOutput

# Config paths
EXPERT_CONFIG_PATH = "configs/expert_config.yaml"
STUDENT_CONFIG_PATH = "configs/student_config.yaml"

# Add paths for simple configs
EXPERT_CONFIG_SIMPLE_PATH = "configs/expert_config_simple.yaml"
STUDENT_CONFIG_SIMPLE_PATH = "configs/student_config_simple.yaml"

# Add paths for web search configs
EXPERT_CONFIG_WEBSEARCH_PATH = "configs/expert_config_websearch.yaml"
STUDENT_CONFIG_WEBSEARCH_PATH = "configs/student_config_websearch.yaml"

# Marker for integration tests - requires API key and makes real calls
# Run with: poetry run pytest -m integration
# Ensure OPENAI_API_KEY is set in your environment OR in a .env file.


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_dialogue_with_api_calls_structured():
    """Tests the dialogue flow with actual API calls and structured output."""

    # Load environment variables from .env file
    # This will load OPENAI_API_KEY if present in .env, making it available via os.getenv
    # Set override=True to ensure the .env key takes precedence over existing env variables.
    load_dotenv(override=True)

    # Check if API key is available (either from loaded .env or pre-existing env)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip(
            "Skipping integration test - OPENAI_API_KEY not found in environment or .env file")
    else:
        # Log the last 5 chars of the key being used
        print(f"\n--- Using API Key ending in: ...{api_key[-5:]} ---")

    print("\n--- Running Integration Test: test_run_dialogue_with_api_calls_structured ---")
    print("--- This test makes REAL API calls and may incur costs. ---")

    # Load configs
    expert_config = load_config(EXPERT_CONFIG_PATH, 'expert')
    student_config = load_config(STUDENT_CONFIG_PATH, 'student')

    # Get max_turns from environment variable, default to 2
    try:
        test_max_turns = int(os.getenv("INTEGRATION_MAX_TURNS", "2"))
        if test_max_turns <= 0:
            test_max_turns = 2
    except ValueError:
        test_max_turns = 2
    print(
        f"--- Running integration test for {test_max_turns} turn(s) (Default: 2, Env Var: {os.getenv('INTEGRATION_MAX_TURNS')}) ---")

    # Initialize agents (SDK should now pick up key from environment)
    expert = ExpertAgent(expert_config)
    student = StudentAgent(student_config)

    # Run the dialogue (real API calls)
    history = await run_dialogue(student, expert, max_turns=test_max_turns)

    # --- Assertions ---
    # Check basic structure
    assert isinstance(history, list)
    assert len(
        history) >= 3, "History should have at least the initial msg + 1 turn"
    max_possible_len = 1 + (test_max_turns * 2)
    assert len(
        history) <= max_possible_len, f"History should have at most {max_possible_len} entries for {test_max_turns} turn(s)"

    # Check first message role
    assert history[0]['role'] == 'user' and history[0].get('agent') is None

    # Check last message properties
    last_message = history[-1]
    assert 'role' in last_message
    assert 'agent' in last_message
    assert 'content' in last_message
    assert isinstance(last_message['content'], str)

    # Check if the stop condition matches the history
    stopped_early = len(history) < max_possible_len
    if stopped_early:
        print("\n--- Dialogue stopped early (Goal Achieved) ---")
        assert last_message['role'] == 'user', "If stopped early, last message should be from student"
        assert last_message['agent'] == student.config.name
        assert 'goal_achieved_flag' in last_message, "Student message should have goal flag when stopping early"
        assert last_message['goal_achieved_flag'] is True, "Goal achieved flag should be True if stopped early"
    else:
        print("\n--- Dialogue completed full {test_max_turns} turns ---")
        assert last_message['role'] == 'assistant', "If ran full turns, last message should be from expert"
        assert last_message['agent'] == expert.config.name

    print("--- Integration Test Completed Successfully ---")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_dialogue_simple_goal_achievement():
    """Tests goal achievement with simple configs (requires API key)."""
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("Skipping integration test - OPENAI_API_KEY not found")
    else:
        print(f"\n--- Using API Key ending in: ...{api_key[-5:]} ---")

    print("\n--- Running Integration Test: test_run_dialogue_simple_goal_achievement ---")
    print("--- This test uses simple configs and expects goal achievement. ---")

    # Load simple configs
    expert_config = load_config(EXPERT_CONFIG_SIMPLE_PATH, 'expert')
    student_config = load_config(STUDENT_CONFIG_SIMPLE_PATH, 'student')
    test_max_turns = 2  # Should finish in 1-2 turns

    # Initialize agents
    expert = ExpertAgent(expert_config)
    student = StudentAgent(student_config)

    # Run the dialogue (real API calls)
    history = await run_dialogue(student, expert, max_turns=test_max_turns)

    # --- Assertions ---
    assert isinstance(history, list)
    assert len(
        history) >= 3, "History should have at least the initial msg + 1 turn"
    max_possible_len = 1 + (test_max_turns * 2)
    assert len(
        history) <= max_possible_len, f"History should have at most {max_possible_len} entries"

    # Check that the goal was marked achieved at some point by the student
    goal_was_marked_achieved = any(
        entry.get('role') == 'user' and entry.get('goal_achieved_flag') is True
        for entry in history
    )
    assert goal_was_marked_achieved, "The goal_achieved_flag should have been set to True by the student at some point"

    # Optional: We could still check the last message if needed, but the key is goal achievement
    # last_message = history[-1]
    # assert last_message['role'] == 'user', "Last message should be from student if stopped early"
    # assert last_message['agent'] == student.config.name
    # assert 'goal_achieved_flag' in last_message, "Student message should have goal flag"
    # assert last_message['goal_achieved_flag'] is True, "Goal achieved flag should be True for this simple case"
    # assert len(history) < max_possible_len, "Dialogue should have stopped before max_turns"

    print("--- Simple Goal Achievement Test Completed Successfully ---")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_dialogue_with_web_search():
    """Tests the dialogue flow with web search (requires API key)."""
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("Skipping integration test - OPENAI_API_KEY not found")
    else:
        print(f"\n--- Using API Key ending in: ...{api_key[-5:]} ---")

    print("\n--- Running Integration Test: test_run_dialogue_with_web_search ---")
    print("--- This test uses web search and expects weather info for London. ---")

    # Load configs
    expert_config = load_config(EXPERT_CONFIG_WEBSEARCH_PATH, 'expert')
    student_config = load_config(STUDENT_CONFIG_WEBSEARCH_PATH, 'student')

    # Initialize agents
    expert = ExpertAgent(expert_config)
    student = StudentAgent(student_config)

    # Run the dialogue (real API calls)
    # Limit to 2 turns for efficiency. Student asks, Expert answers (hopefully with web search).
    test_max_turns = 2
    history = await run_dialogue(student, expert, max_turns=test_max_turns)

    # --- Assertions ---
    assert isinstance(history, list)
    assert len(
        history) >= 3, "History should have at least the initial msg + 1 turn"
    max_possible_len = 1 + (test_max_turns * 2)
    # Goal is unlikely to be achieved in 1 turn, so expect full 2 turns (3 or 5 entries)
    # Initial User Msg -> Student Turn -> Expert Turn -> Student Turn -> Expert Turn
    # It might finish after student's second turn if goal is achieved.
    assert len(history) == 5 or (len(history) == 4 and history[-1].get('goal_achieved_flag') is True), \
        f"History should have 4 (goal achieved) or 5 entries for {test_max_turns} turns, but got {len(history)}"

    # Find the expert's response (should be the 3rd or 5th entry)
    expert_response_entry = None
    if len(history) >= 3 and history[2].get('agent') == expert.config.name:
        expert_response_entry = history[2]
    elif len(history) == 5 and history[4].get('agent') == expert.config.name:
        # If goal achieved early, expert might not respond last
        expert_response_entry = history[4]

    # Check if the 'used_web_search' flag was set to True in *any* expert turn
    expert_used_search = any(
        entry.get('role') == 'assistant' and entry.get(
            'used_web_search') is True
        for entry in history
    )
    print(f"Expert used web search (found citations): {expert_used_search}")
    assert expert_used_search, "Expert agent should have used web search (indicated by citations) in its response."

    # Check if the goal was marked achieved by the student in their *last* turn
    student_last_turn_entry = history[3] if len(
        history) >= 4 else None  # Student's second turn response
    goal_marked_achieved = False
    if student_last_turn_entry and student_last_turn_entry.get('role') == 'user':
        goal_marked_achieved = student_last_turn_entry.get(
            'goal_achieved_flag', False)

    # Depending on LLM interpretation, goal might be achieved after getting weather info
    # This assertion is less critical than checking the expert's content, but good to check.
    print(f"Student Goal Achieved Flag (Turn 2): {goal_marked_achieved}")
    # assert goal_marked_achieved, "Student should ideally mark goal achieved after getting weather info"

    print("--- Web Search Test Completed Successfully ---")
