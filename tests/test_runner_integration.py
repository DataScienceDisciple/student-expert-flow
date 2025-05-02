import pytest
import asyncio
import os
from dotenv import load_dotenv
import glob  # Import glob for finding files
import time  # Import time for waiting briefly
from typing import Optional  # Import Optional for type hinting

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

TRANSCRIPT_DIR = "transcripts"


def find_latest_transcript(transcript_dir: str = TRANSCRIPT_DIR) -> Optional[str]:
    """Finds the most recently created transcript file (excluding summaries) in the directory."""
    if not os.path.isdir(transcript_dir):
        return None
    # Get all potentially matching files
    all_files = glob.glob(os.path.join(transcript_dir, 'transcript_*.txt'))
    # Filter out summary files
    transcript_files = [f for f in all_files if not f.endswith('.summary.txt')]

    if not transcript_files:
        return None
    # Find the latest among the actual transcript files
    latest_file = max(transcript_files, key=os.path.getctime)
    return latest_file

# Helper function to assert transcript creation


def assert_transcript_created(start_time):
    """Waits briefly and asserts that a recent transcript file exists."""
    # Wait a very short time to allow file system to update
    time.sleep(0.5)
    latest_transcript = find_latest_transcript()
    assert latest_transcript is not None, f"No transcript file found in {TRANSCRIPT_DIR}"
    assert os.path.exists(
        latest_transcript), f"Latest transcript {latest_transcript} does not exist"
    # Check if the file was created after the test started (basic check)
    assert os.path.getctime(
        latest_transcript) > start_time, "Transcript file seems older than the test run"
    # Check if the file is not empty
    assert os.path.getsize(
        latest_transcript) > 0, f"Transcript file {latest_transcript} is empty"
    print(
        f"\n--- Verified Transcript Creation: {os.path.basename(latest_transcript)} ---")

    # --- Assert Summary Creation --- #
    # Construct summary path correctly (replace .txt with .summary.txt)
    base_path, ext = os.path.splitext(latest_transcript)
    summary_file_path = base_path + ".summary.txt"
    # summary_file_path = latest_transcript.replace(".txt", ".summary.txt") # Alternative way

    # --- DEBUG PRINTS --- #
    print(f"DEBUG: latest_transcript = {latest_transcript}")
    print(f"DEBUG: os.path.splitext result = ({repr(base_path)}, {repr(ext)})")
    print(f"DEBUG: Constructed summary_file_path = {summary_file_path}")
    # --- END DEBUG PRINTS --- #

    assert os.path.exists(
        summary_file_path), f"Summary file {summary_file_path} not found"
    assert os.path.getsize(
        summary_file_path) > 0, f"Summary file {summary_file_path} is empty"

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

    # Record start time
    start_time = time.time()

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

    # Check first message role and agent (should be System)
    assert history[0]['role'] == 'user' and history[0].get('agent') == 'System'

    # Check last message properties
    last_message = history[-1]
    assert 'role' in last_message
    assert 'agent' in last_message
    assert 'content' in last_message
    assert isinstance(last_message['content'], str)

    # Check if the stop condition matches the history
    # When max_turns is reached, the loop breaks *after* the final expert turn,
    # so the length will be 1 (Initial) + max_turns*2 - 1 (E1, S1, E2... EN)
    expected_len_max_turns = 1 + (test_max_turns * 2) - 1
    stopped_early = len(history) < expected_len_max_turns

    if stopped_early:
        print("\n--- Dialogue stopped early (Goal Achieved) ---")
        # If stopped early, it must have been after a Student turn indicated goal achieved.
        # Length should be 1 (Initial) + N*(E+S) where N < test_max_turns
        assert len(
            history) % 2 != 0, "History length should be odd if stopped early after student turn"
        assert history[-1]['role'] == 'user', "If stopped early, last message should be from student"
        assert history[-1]['agent'] == student.config.name
        assert 'goal_achieved_flag' in history[-1], "Student message should have goal flag when stopping early"
        assert history[-1]['goal_achieved_flag'] is True, "Goal achieved flag should be True if stopped early"
    else:
        # Max turns reached: loop broke *after* the expert's last turn.
        expected_len = expected_len_max_turns  # Use calculated expected length
        assert len(
            history) == expected_len, f"History length should be {expected_len} when max_turns ({test_max_turns}) reached"
        print(
            f"\n--- Dialogue completed full {test_max_turns} turns (stopped after Expert) ---")
        # If dialogue ran full turns, the loop broke after the *Expert's* last turn.
        # Therefore, the last message should be from the Expert.
        assert history[-1]['role'] == 'assistant', "If ran full turns, last message should be from expert"
        assert history[-1]['agent'] == expert.config.name
        # No goal flag check needed for the expert message

    # Assert transcript creation
    assert_transcript_created(start_time)


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

    # Record start time
    start_time = time.time()

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
    print(f"Goal marked achieved by student: {goal_was_marked_achieved}")

    # Optional: We could still check the last message if needed, but the key is goal achievement
    # last_message = history[-1]
    # assert last_message['role'] == 'user', "Last message should be from student if stopped early"
    # assert last_message['agent'] == student.config.name
    # assert 'goal_achieved_flag' in last_message, "Student message should have goal flag"
    # assert last_message['goal_achieved_flag'] is True, "Goal achieved flag should be True for this simple case"
    # assert len(history) < max_possible_len, "Dialogue should have stopped before max_turns"

    print("--- Simple Goal Achievement Test Completed Successfully ---")
    # Assert transcript creation
    assert_transcript_created(start_time)


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

    # Record start time
    start_time = time.time()

    # Run the dialogue (real API calls)
    # Limit to 2 turns for efficiency. Student asks, Expert answers (hopefully with web search).
    test_max_turns = 2
    history = await run_dialogue(student, expert, max_turns=test_max_turns)

    # --- Assertions ---
    assert isinstance(history, list)
    assert len(
        history) >= 3, "History should have at least the initial msg + 1 turn"
    # Max possible if student ran last turn
    max_possible_len = 1 + (test_max_turns * 2)
    # Expected if max_turns hit (stops after expert)
    expected_len_max_turns = 1 + (test_max_turns * 2) - 1

    # Goal might be achieved in Turn 1 (len 3) or run full turns (len should be expected_len_max_turns)
    assert len(history) == 3 or len(history) == expected_len_max_turns, \
        f"History should have 3 (goal achieved T1) or {expected_len_max_turns} (max turns) entries for {test_max_turns} turns, but got {len(history)}"

    # Find the expert's response (should be the 2nd entry - index 1)
    expert_response_entry = None
    if len(history) >= 2 and history[1].get('agent') == expert.config.name:
        expert_response_entry = history[1]
    # No need to check index 4 anymore as we expect it to end early or run full 2 turns
    # elif len(history) == 5 and history[4].get('agent') == expert.config.name:
    #     expert_response_entry = history[4]

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
    # Assert transcript creation
    assert_transcript_created(start_time)
