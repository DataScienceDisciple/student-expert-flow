import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from student_expert_flow.participants import StudentAgent, ExpertAgent
from student_expert_flow.config import load_config
from student_expert_flow.runner import run_dialogue
from agents import Runner
# Import the structured output model for mocking
from student_expert_flow.models import StudentOutput

# Config paths
EXPERT_CONFIG_PATH = "configs/expert_config.yaml"
STUDENT_CONFIG_PATH = "configs/student_config.yaml"

# --- Mocking Setup ---
# Helper to create a mock RunResult for TEXT output (Expert)


def create_mock_text_run_result(final_output: str, history_list: list):
    mock_result = MagicMock()
    mock_result.final_output = final_output  # Will be string
    mock_result.to_input_list.return_value = history_list
    return mock_result

# Helper to create a mock RunResult for STRUCTURED output (Student)


def create_mock_structured_run_result(output_model: StudentOutput, history_list: list):
    mock_result = MagicMock()
    # IMPORTANT: Set final_output to the Pydantic model instance
    mock_result.final_output = output_model
    mock_result.to_input_list.return_value = history_list
    return mock_result

# --- Tests ---


@pytest.mark.asyncio
async def test_run_dialogue_mocked_flow_structured(mocker):  # Renamed test
    """Tests the dialogue flow logic with mocked Runner calls and structured output."""
    expert_config = load_config(EXPERT_CONFIG_PATH, 'expert')
    student_config = load_config(STUDENT_CONFIG_PATH, 'student')
    test_max_turns = 2
    expert = ExpertAgent(expert_config)
    student = StudentAgent(student_config)

    mock_run = mocker.patch('agents.Runner.run', new_callable=AsyncMock)

    # --- Define mock responses ---
    # Updated initial prompt content
    initial_expert_input_content = f"My learning goal is: {student.config.goal}. Please provide an initial explanation or ask clarifying questions."

    # Turn 1 - Expert -> Returns Text
    mock_expert_resp_1 = "Okay, X is..."
    # History after expert turn 1
    mock_expert_hist_1 = [{"role": "user", "content": initial_expert_input_content},  # Original user goal
                          {"role": "assistant", "content": mock_expert_resp_1}]  # Expert response
    mock_expert_result_1 = create_mock_text_run_result(
        mock_expert_resp_1, mock_expert_hist_1)

    # Turn 1 - Student -> Returns JSON
    mock_student_output_1 = StudentOutput(
        is_goal_achieved=False, response_content="Tell me more about Y.")
    # History after student turn 1 (includes the student's JSON as assistant msg)
    mock_student_hist_1 = mock_expert_hist_1 + \
        [{"role": "assistant", "content": mock_student_output_1.model_dump_json()}]
    mock_student_result_1 = create_mock_structured_run_result(
        mock_student_output_1, mock_student_hist_1)

    # Turn 2 - Expert -> Returns Text
    mock_expert_resp_2 = "Y is..."
    # History after expert turn 2
    mock_expert_hist_2 = mock_student_hist_1 + \
        [{"role": "assistant", "content": mock_expert_resp_2}]
    mock_expert_result_2 = create_mock_text_run_result(
        mock_expert_resp_2, mock_expert_hist_2)

    # Turn 2 - Student -> Returns JSON
    mock_student_output_2 = StudentOutput(
        is_goal_achieved=False, response_content="Thanks, anything else?")
    # History after student turn 2
    mock_student_hist_2 = mock_expert_hist_2 + \
        [{"role": "assistant", "content": mock_student_output_2.model_dump_json()}]
    mock_student_result_2 = create_mock_structured_run_result(
        mock_student_output_2, mock_student_hist_2)

    mock_run.side_effect = [
        mock_expert_result_1,  # E1
        mock_student_result_1,  # S1
        mock_expert_result_2,  # E2
        mock_student_result_2,  # S2
    ]

    history = await run_dialogue(student, expert, max_turns=test_max_turns)

    # Expect E1, S1, E2 calls. Loop breaks before S2.
    expected_calls = (test_max_turns * 2) - 1
    assert mock_run.call_count == expected_calls
    # History: Initial + E1(text) + S1(struct) + E2(text)
    expected_len = 1 + expected_calls
    assert len(history) == expected_len

    # Check logged history structure
    assert history[0]['role'] == 'user' and history[0]['agent'] == 'System' and history[0]['content'] == initial_expert_input_content
    assert history[1]['role'] == 'assistant' and history[1]['agent'] == expert.config.name and history[1]['content'] == mock_expert_resp_1
    assert history[2]['role'] == 'user' and history[2]['agent'] == student.config.name and history[2]['content'] == mock_student_output_1.response_content
    # Last message should be from Expert turn 2
    assert history[3]['role'] == 'assistant' and history[3]['agent'] == expert.config.name and history[3]['content'] == mock_expert_resp_2

# Mocked test for early goal achievement


@pytest.mark.asyncio
# Renamed test
async def test_run_dialogue_mocked_goal_achieved_structured(mocker):
    """Tests the dialogue flow ending early with mocked structured output."""
    expert_config = load_config(EXPERT_CONFIG_PATH, 'expert')
    student_config = load_config(STUDENT_CONFIG_PATH, 'student')
    expert = ExpertAgent(expert_config)
    student = StudentAgent(student_config)

    mock_run = mocker.patch('agents.Runner.run', new_callable=AsyncMock)

    # --- Define mock responses ---
    initial_expert_input_content = f"My learning goal is: {student.config.goal}. Please provide an initial explanation or ask clarifying questions."

    # Turn 1 - Expert -> Returns Text
    mock_expert_resp_1 = "Okay, X is... (final explanation)"
    mock_expert_hist_1 = [{"role": "user", "content": initial_expert_input_content},
                          {"role": "assistant", "content": mock_expert_resp_1}]
    mock_expert_result_1 = create_mock_text_run_result(
        mock_expert_resp_1, mock_expert_hist_1)

    # Turn 1 - Student -> Returns JSON (Goal ACHIEVED)
    mock_student_output_1 = StudentOutput(
        is_goal_achieved=True, response_content="Great, I understand now.")
    mock_student_hist_1 = mock_expert_hist_1 + \
        [{"role": "assistant", "content": mock_student_output_1.model_dump_json()}]
    mock_student_result_1 = create_mock_structured_run_result(
        mock_student_output_1, mock_student_hist_1)

    mock_run.side_effect = [
        mock_expert_result_1,   # Expert Turn 1
        mock_student_result_1,  # Student Turn 1 (achieves goal)
        # Loop should break here, no more calls expected
    ]

    history = await run_dialogue(student, expert, max_turns=5)

    # Expect 2 calls: Expert T1, Student T1
    assert mock_run.call_count == 2
    # Expect 3 entries: Initial System msg, Expert T1 resp, Student T1 resp
    assert len(history) == 3

    # Check history content
    assert history[0]['role'] == 'user' and history[0]['agent'] == 'System'
    assert history[1]['role'] == 'assistant' and history[1]['agent'] == expert.config.name
    assert history[2]['role'] == 'user' and history[2]['agent'] == student.config.name
    assert history[2]['content'] == "Great, I understand now."
    assert history[2]['goal_achieved_flag'] is True
