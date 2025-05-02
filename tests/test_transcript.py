import pytest
import os
import re
from student_expert_flow.transcript import save_transcript, _sanitize_filename

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

    # Run the function
    saved_path = save_transcript(
        MOCK_HISTORY, MOCK_GOAL, output_dir=str(output_dir))

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
