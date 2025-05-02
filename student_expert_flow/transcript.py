import os
import datetime
import re
from typing import List, Dict, Any


def _sanitize_filename(text: str, max_len: int = 50) -> str:
    """Removes invalid characters and shortens text for use in a filename."""
    # Remove invalid file system characters
    sanitized = re.sub(r'[\\/*?:"<>|]', '', text)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Truncate and remove trailing underscores
    sanitized = sanitized[:max_len].rstrip('_')
    # Handle empty string case
    if not sanitized:
        return "transcript"
    return sanitized


def format_transcript(history: List[Dict[str, Any]], goal: str) -> str:
    """Formats the conversation history into a readable string."""
    lines = [f"--- Conversation Transcript ---"]
    lines.append(f"Goal: {goal}")
    lines.append(f"Timestamp: {datetime.datetime.now().isoformat()}")
    lines.append("-" * 30 + "\n")

    for entry in history:
        role = entry.get('role', 'unknown_role')
        # Default to System if agent missing
        agent = entry.get('agent', 'System')
        content = entry.get('content', '')
        used_search = entry.get('used_web_search')

        prefix = f"[{agent} ({role})]"
        search_suffix = f" [Used Web Search: {used_search}]" if used_search is not None else ""

        lines.append(f"{prefix}: {content}{search_suffix}")

    lines.append("\n" + "-" * 30)
    lines.append("--- End Transcript ---")
    return "\n".join(lines)


def save_transcript(history: List[Dict[str, Any]], goal: str, output_dir: str = "transcripts") -> str:
    """Saves the formatted conversation history to a file.

    Args:
        history: The conversation history list.
        goal: The student's learning goal.
        output_dir: The directory to save the transcript in . Defaults to 'transcripts'.

    Returns:
        The path to the saved transcript file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Format the content
    formatted_content = format_transcript(history, goal)

    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_goal = _sanitize_filename(goal)
    filename = f"transcript_{timestamp}_{sanitized_goal}.txt"
    filepath = os.path.join(output_dir, filename)

    # Write to file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
        # Optional: Log or print confirmation
        print(f"Transcript saved to: {filepath}")
        return filepath
    except IOError as e:
        print(f"Error saving transcript to {filepath}: {e}")
        # Consider raising the exception or returning None depending on desired error handling
        raise  # Re-raise the exception for now
