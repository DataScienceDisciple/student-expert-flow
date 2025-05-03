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
    """Formats the conversation history into a readable Markdown string."""
    lines = [f"# Conversation Transcript"]
    lines.append(f"\n## Goal\n> {goal}")
    lines.append(f"\n## Timestamp\n> {datetime.datetime.now().isoformat()}")
    lines.append("\n---\n")

    turn_number = 0
    i = 0
    while i < len(history):
        entry = history[i]
        role = entry.get('role', 'unknown_role')
        agent = entry.get('agent', 'System')
        # Strip leading/trailing whitespace
        content = entry.get('content', '').strip()
        used_search = entry.get('used_web_search')
        goal_achieved = entry.get('goal_achieved_flag')

        prefix = f"**[{agent} ({role})]**"
        metadata_line = ""
        if used_search is not None:
            metadata_line += f"*Used Web Search: {used_search}*  "
        if goal_achieved is not None:
            metadata_line += f"*Goal Achieved: {goal_achieved}*"
        metadata_line = metadata_line.strip()

        # Format content with blockquotes, handle multi-line content
        formatted_content = "\n".join(
            [f"> {line}" for line in content.split('\n')])

        # Handle the initial system message
        if agent == 'System' and role == 'user':
            lines.append(f"{prefix}\n{formatted_content}")
            if metadata_line:
                lines.append(f">\n> _{metadata_line}_")
            lines.append("\n---\n")  # Separator after system message
            i += 1
            continue

        # Start a new turn when we see the expert (assistant role)
        # Note: Assuming expert is always 'assistant' and student is 'user' in the log for turn structure
        if role == 'assistant':
            turn_number += 1
            lines.append(f"## Turn {turn_number}")
            lines.append(f"\n{prefix}\n{formatted_content}")
            if metadata_line:
                lines.append(f">\n> _{metadata_line}_")
            i += 1
            # Check if the next message is the student's response in this turn
            if i < len(history) and history[i].get('role') == 'user':
                student_entry = history[i]
                student_role = student_entry.get('role', 'unknown_role')
                student_agent = student_entry.get(
                    'agent', 'Student')  # Assume name might vary
                student_content = student_entry.get('content', '').strip()
                student_goal_achieved = student_entry.get('goal_achieved_flag')
                student_prefix = f"**[{student_agent} ({student_role})]**"
                student_metadata_line = ""
                if student_goal_achieved is not None:
                    student_metadata_line += f"*Goal Achieved: {student_goal_achieved}*"
                student_metadata_line = student_metadata_line.strip()

                student_formatted_content = "\n".join(
                    [f"> {line}" for line in student_content.split('\n')])

                lines.append(
                    f"\n{student_prefix}\n{student_formatted_content}")
                if student_metadata_line:
                    lines.append(f">\n> _{student_metadata_line}_")
                i += 1
            lines.append("\n---\n")  # Separator after each turn
        else:
            # Handle cases where conversation might not follow strict Expert->Student pattern
            # Or if it starts unexpectedly with the student (print it anyway).
            # Increment turn number for log clarity
            lines.append(f"## Turn {turn_number+1} (Unexpected Start)")
            lines.append(f"\n{prefix}\n{formatted_content}")
            if metadata_line:
                lines.append(f">\n> _{metadata_line}_")
            lines.append("\n---\n")
            i += 1

    lines.append("--- End Transcript ---")
    return "\n".join(lines)


def save_transcript(history: List[Dict[str, Any]], goal: str, formatted_transcript: str, output_dir: str = "transcripts") -> str:
    """Saves the formatted conversation history to a Markdown file.

    Args:
        history: The conversation history list (used for metadata/unused here but kept for potential future use).
        goal: The student's learning goal (used for filename).
        formatted_transcript: The pre-formatted transcript string (assumed Markdown) to save.
        output_dir: The directory to save the transcript in . Defaults to 'transcripts'.

    Returns:
        The path to the saved transcript file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_goal = _sanitize_filename(goal)
    # Changed extension to .md
    filename = f"transcript_{timestamp}_{sanitized_goal}.md"
    filepath = os.path.join(output_dir, filename)

    # Write to file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(formatted_transcript)
        # Optional: Log or print confirmation
        logger.info(f"Transcript saved to Markdown: {filepath}")
        return filepath
    except IOError as e:
        logger.error(f"Error saving transcript to {filepath}: {e}")
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
