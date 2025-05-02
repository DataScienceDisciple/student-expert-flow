import os
import datetime
import re
from typing import List, Dict, Any
from openai import OpenAI, OpenAIError, AsyncOpenAI
import logging

logger = logging.getLogger(__name__)

# Initialize AsyncOpenAI client at module level
# Relies on OPENAI_API_KEY environment variable
async_openai_client = AsyncOpenAI()


def _sanitize_filename(text: str, max_len: int = 50) -> str:
    """Removes invalid characters and shortens text for use in a filename."""
    # Remove invalid file system characters (including dots)
    sanitized = re.sub(r'[\\/*?:"<>|.]', '', text)
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


def save_transcript(history: List[Dict[str, Any]], goal: str, formatted_transcript: str, output_dir: str = "transcripts") -> str:
    """Saves the formatted conversation history to a file.

    Args:
        history: The conversation history list (used for metadata/unused here but kept for potential future use).
        goal: The student's learning goal (used for filename).
        formatted_transcript: The pre-formatted transcript string to save.
        output_dir: The directory to save the transcript in . Defaults to 'transcripts'.

    Returns:
        The path to the saved transcript file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_goal = _sanitize_filename(goal)
    filename = f"transcript_{timestamp}_{sanitized_goal}.txt"
    filepath = os.path.join(output_dir, filename)

    # Write to file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(formatted_transcript)
        # Optional: Log or print confirmation
        print(f"Transcript saved to: {filepath}")
        return filepath
    except IOError as e:
        print(f"Error saving transcript to {filepath}: {e}")
        # Consider raising the exception or returning None depending on desired error handling
        raise  # Re-raise the exception for now


async def generate_summary(formatted_transcript: str, model: str = "gpt-4.1-mini") -> str:
    """Generates a concise summary of the conversation using an LLM call.

    Args:
        formatted_transcript: The formatted transcript string.
        model: The OpenAI model to use for summarization.

    Returns:
        The generated summary text, or an error message if generation failed.
    """
    logger.info(f"Generating summary using model: {model}")
    try:
        # Use the module-level client instance
        client = async_openai_client

        system_prompt = ("You are an expert summarizer. Please provide a concise summary of the following conversation transcript. "
                         "Highlight the main topic or goal, key points discussed, and whether the student's learning goal was achieved."
                         "Structure it in a way that is easy to read and understand."
                         "We want to know the main points of the conversation without reading the entire transcript.")

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_transcript}
            ],
            temperature=0.7,  # Lower temperature for more focused summary
        )

        summary = response.choices[0].message.content
        if not summary:
            logger.warning("Summary generation returned empty content.")
            return "[Summary generation failed: Empty content returned]"

        logger.info("Summary generated successfully.")
        return summary.strip()

    except OpenAIError as e:
        logger.error(f"OpenAI API error during summary generation: {e}")
        return f"[Summary generation failed due to API error: {e}]"
    except Exception as e:
        logger.error(f"Unexpected error during summary generation: {e}")
        return f"[Summary generation failed due to unexpected error: {e}]"
