import pytest
import os
import re
from unittest.mock import AsyncMock, MagicMock  # Import mocking utilities
# Add generate_summary
from student_expert_flow.transcript import save_transcript, _sanitize_filename, format_transcript, generate_summary
from openai import OpenAIError  # Import specific exception for testing

# Sample history data for testing
MOCK_HISTORY = [
    {"role": "user", "agent": "System", "content": "Goal: Test goal"},
    {"role": "user", "agent": "StudentA", "content": "Question 1"},
    {"role": "assistant", "agent": "ExpertB",
        "content": "Answer 1", "used_web_search": False},
    {"role": "user", "agent": "StudentA", "content": "Question 2"},
    {"role": "assistant", "agent": "ExpertB",
        "content": "Answer 2 with citation", "used_web_search": True},
]

MOCK_GOAL = "Understand the basics of transcript saving & filename sanitization?*<>|"

# Sample formatted transcript for summary test
MOCK_FORMATTED_TRANSCRIPT = format_transcript(MOCK_HISTORY, MOCK_GOAL)
EXPECTED_SUMMARY = "This is the expected summary."  # Define expected summary


def test_sanitize_filename():
    """Tests the filename sanitization function."""
    assert _sanitize_filename("Simple Goal") == "Simple_Goal"
    assert _sanitize_filename(
        "Goal with spaces and ?") == "Goal_with_spaces_and"
    assert _sanitize_filename(
        "Very\Long/Goal*Name:that|needs<truncation>", max_len=20) == "VeryLongGoalNamethat"
    assert _sanitize_filename("goal/with/slashes") == "goalwithslashes"
    assert _sanitize_filename("", max_len=10) == "transcript"
    assert _sanitize_filename("????", max_len=10) == "transcript"
    # Test the specific mock goal
    assert _sanitize_filename(
        MOCK_GOAL) == "Understand_the_basics_of_transcript_saving_&_filen"


def test_save_transcript(tmp_path):
    """Tests saving a transcript to a temporary directory."""
    output_dir = tmp_path / "test_transcripts"

    # Run the function - Pass the formatted transcript
    formatted_transcript = format_transcript(MOCK_HISTORY, MOCK_GOAL)
    saved_path = save_transcript(
        MOCK_HISTORY, MOCK_GOAL, formatted_transcript=formatted_transcript, output_dir=str(output_dir))

    # 1. Check if the path is correct and file exists
    assert os.path.exists(saved_path)
    assert str(output_dir) in saved_path

    # 2. Check filename format (timestamp part is tricky, focus on prefix and sanitized goal)
    filename = os.path.basename(saved_path)
    sanitized_goal_part = _sanitize_filename(MOCK_GOAL)
    # Regex to match: transcript_YYYYMMDD_HHMMSS_sanitizedgoal.txt
    assert re.match(
        rf"transcript_\d{{8}}_\d{{6}}_{re.escape(sanitized_goal_part)}\.txt", filename)

    # 3. Read the content and check key elements
    with open(saved_path, 'r', encoding='utf-8') as f:
        content = f.read()

    assert "--- Conversation Transcript ---" in content
    assert f"Goal: {MOCK_GOAL}" in content
    assert "Timestamp:" in content
    assert "[StudentA (user)]: Question 1" in content
    assert "[ExpertB (assistant)]: Answer 1 [Used Web Search: False]" in content
    assert "[ExpertB (assistant)]: Answer 2 with citation [Used Web Search: True]" in content
    assert "--- End Transcript ---" in content


@pytest.mark.asyncio
async def test_generate_summary_success(mocker):
    """Tests successful summary generation by mocking the OpenAI client."""
    # Mock the OpenAI client and its methods
    mock_openai_client = MagicMock()
    mock_completion = MagicMock()
    mock_message = MagicMock()
    mock_message.content = EXPECTED_SUMMARY
    mock_completion.choices = [MagicMock(message=mock_message)]

    # Use AsyncMock for the async method create
    mock_create_method = AsyncMock(return_value=mock_completion)
    mock_openai_client.chat.completions.create = mock_create_method

    # Patch the module-level client instance within the transcript module
    mocker.patch(
        'student_expert_flow.transcript.async_openai_client', mock_openai_client)

    # Call the function
    summary = await generate_summary(MOCK_FORMATTED_TRANSCRIPT)

    # Assertions
    assert summary == EXPECTED_SUMMARY
    mock_openai_client.chat.completions.create.assert_awaited_once()
    call_args = mock_openai_client.chat.completions.create.call_args
    assert call_args.kwargs['model'] == "gpt-4.1-mini"  # Default model
    assert call_args.kwargs['messages'][1]['content'] == MOCK_FORMATTED_TRANSCRIPT


@pytest.mark.asyncio
async def test_generate_summary_empty_response(mocker):
    """Tests summary generation when the API returns empty content."""
    mock_openai_client = MagicMock()
    mock_completion = MagicMock()
    mock_message = MagicMock()
    mock_message.content = ""  # Empty content
    mock_completion.choices = [MagicMock(message=mock_message)]
    mock_create_method = AsyncMock(return_value=mock_completion)
    mock_openai_client.chat.completions.create = mock_create_method
    mocker.patch(
        'student_expert_flow.transcript.async_openai_client', mock_openai_client)

    summary = await generate_summary(MOCK_FORMATTED_TRANSCRIPT)

    assert "[Summary generation failed: Empty content returned]" in summary


@pytest.mark.asyncio
async def test_generate_summary_openai_error(mocker):
    """Tests summary generation when the OpenAI API raises an error."""
    mock_openai_client = MagicMock()
    mock_create_method = AsyncMock(
        side_effect=OpenAIError("API connection error"))
    mock_openai_client.chat.completions.create = mock_create_method
    mocker.patch(
        'student_expert_flow.transcript.async_openai_client', mock_openai_client)

    summary = await generate_summary(MOCK_FORMATTED_TRANSCRIPT)

    assert "[Summary generation failed due to API error" in summary
    assert "API connection error" in summary
