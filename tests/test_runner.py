import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from student_expert_flow.agents import StudentAgent, ExpertAgent
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
    initial_student_input_content = f"My learning goal is: {student.config.goal}. Please explain the basics and respond in the required JSON format."

    # Turn 1 - Student -> Returns JSON
    mock_student_output_1 = StudentOutput(
        is_goal_achieved=False, response_content="Okay, tell me about X.")
    mock_student_hist_1 = [{"role": "user", "content": initial_student_input_content}, {
        "role": "assistant", "content": mock_student_output_1.model_dump_json()}]
    # History for Expert now includes the user message derived from student's response_content
    mock_expert_input_hist_1 = mock_student_hist_1 + \
        [{"role": "user", "content": mock_student_output_1.response_content}]
    mock_student_result_1 = create_mock_structured_run_result(
        mock_student_output_1, mock_expert_input_hist_1)

    # Turn 1 - Expert -> Returns Text
    mock_expert_resp_1 = "X is..."
    mock_expert_hist_1 = mock_expert_input_hist_1 + \
        [{"role": "assistant", "content": mock_expert_resp_1}]
    mock_expert_result_1 = create_mock_text_run_result(
        mock_expert_resp_1, mock_expert_hist_1)

    # Turn 2 - Student -> Returns JSON
    mock_student_output_2 = StudentOutput(
        is_goal_achieved=False, response_content="What about Y?")
    mock_student_hist_2 = mock_expert_hist_1 + \
        [{"role": "assistant", "content": mock_student_output_2.model_dump_json()}]
    mock_expert_input_hist_2 = mock_student_hist_2 + \
        [{"role": "user", "content": mock_student_output_2.response_content}]
    mock_student_result_2 = create_mock_structured_run_result(
        mock_student_output_2, mock_expert_input_hist_2)

    # Turn 2 - Expert -> Returns Text
    mock_expert_resp_2 = "Y is..."
    mock_expert_hist_2 = mock_expert_input_hist_2 + \
        [{"role": "assistant", "content": mock_expert_resp_2}]
    mock_expert_result_2 = create_mock_text_run_result(
        mock_expert_resp_2, mock_expert_hist_2)

    mock_run.side_effect = [
        mock_student_result_1,
        mock_expert_result_1,
        mock_student_result_2,
        mock_expert_result_2,
    ]

    history = await run_dialogue(student, expert, max_turns=test_max_turns)

    assert mock_run.call_count == test_max_turns * 2
    # History: Initial + S1(struct) + E1(text) + S2(struct) + E2(text)
    assert len(history) == 1 + (test_max_turns * 2)

    # Check logged history structure (uses response_content from student)
    assert history[0]['role'] == 'user' and history[0]['content'] == initial_student_input_content
    assert history[1]['role'] == 'user' and history[1]['content'] == mock_student_output_1.response_content and history[1]['goal_achieved_flag'] is False
    assert history[2]['role'] == 'assistant' and history[2]['content'] == mock_expert_resp_1
    assert history[3]['role'] == 'user' and history[3]['content'] == mock_student_output_2.response_content and history[3]['goal_achieved_flag'] is False
    assert history[4]['role'] == 'assistant' and history[4]['content'] == mock_expert_resp_2


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
    initial_student_input_content = f"My learning goal is: {student.config.goal}. Please explain the basics and respond in the required JSON format."

    # Turn 1 - Student -> Returns JSON (Goal not achieved)
    mock_student_output_1 = StudentOutput(
        is_goal_achieved=False, response_content="Okay, tell me about X.")
    mock_student_hist_1 = [{"role": "user", "content": initial_student_input_content}, {
        "role": "assistant", "content": mock_student_output_1.model_dump_json()}]
    mock_expert_input_hist_1 = mock_student_hist_1 + \
        [{"role": "user", "content": mock_student_output_1.response_content}]
    mock_student_result_1 = create_mock_structured_run_result(
        mock_student_output_1, mock_expert_input_hist_1)

    # Turn 1 - Expert -> Returns Text
    mock_expert_resp_1 = "X is... (final explanation)"
    mock_expert_hist_1 = mock_expert_input_hist_1 + \
        [{"role": "assistant", "content": mock_expert_resp_1}]
    mock_expert_result_1 = create_mock_text_run_result(
        mock_expert_resp_1, mock_expert_hist_1)

    # Turn 2 - Student -> Returns JSON (Goal ACHIEVED)
    mock_student_output_2 = StudentOutput(
        is_goal_achieved=True, response_content="Great, I understand now.")
    mock_student_hist_2 = mock_expert_hist_1 + \
        [{"role": "assistant", "content": mock_student_output_2.model_dump_json()}]
    # History for expert doesn't strictly matter here as loop should break
    mock_expert_input_hist_2 = mock_student_hist_2 + \
        [{"role": "user", "content": mock_student_output_2.response_content}]
    mock_student_result_2 = create_mock_structured_run_result(
        mock_student_output_2, mock_expert_input_hist_2)

    mock_run.side_effect = [
        mock_student_result_1,
        mock_expert_result_1,
        mock_student_result_2,
    ]

    history = await run_dialogue(student, expert, max_turns=5)

    assert mock_run.call_count == 3
    # History: Initial + S1(struct) + E1(text) + S2(struct, achieved)
    assert len(history) == 1 + 3
    assert history[-1]['content'] == mock_student_output_2.response_content
    assert history[-1]['goal_achieved_flag'] is True
    assert history[-1]['agent'] == student.config.name
